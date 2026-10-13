"""kvmgr.diag.serializer -- NAME the producer of a numpy scalar in the output.

The COERCION of a numpy scalar in vLLM's output path is a correctness fix and
stays in core (kvmgr.register._install_serializer_shim): without it one
scalar kills the engine core's output thread and every request hangs. What
lives here is the INSTRUMENT that found where such scalars came from -- a
sampled proactive scan of what is being encoded, and a thorough walk the first
time each numpy type is coerced, answering through the numpy_producer hook.

Moved verbatim from kvmgr.register on 2026-10-13. OFF by default
(diag_serializer_walk): the producer it was built to find was ours, and is
fixed at the source.
"""

from __future__ import annotations

import time as _time

from vllm.logger import init_logger

from .. import hooks

logger = init_logger("vllm." + __name__)

# The proactive scan's budget. Spacing keeps it off the per-encode path;
# the limit keeps it from being a standing cost. Both are deliberately
# generous in TIME and stingy in RATE: the event is rare, not fast.
_SCAN_EVERY_S = 5.0


_SCAN_LIMIT = 600


def _numpy_paths(root, limit: int = 12, max_depth: int = 8,
                 max_nodes: int = 200000, max_items: int = 512) -> list:
    """Dotted paths to every numpy SCALAR reachable from `root`. Bounded.

    *** THE HOOK SEES THE SCALAR, NOT ITS PARENT, and that is why two runs of
    *** "value=1" and "value=39" identified nothing. *** msgspec hands enc_hook
    the offending object alone, so the type and value are all it can report --
    and a varying small count narrows the field without naming the field. The
    only way to name it is to walk the object actually being encoded.

    Bounded on purpose, because this runs inside the engine's output thread.
    *** AND THE BOUNDS DO DIFFERENT JOBS, which I first got wrong in this very
    *** comment: *** the DEPTH CAP is what stops a cycle -- break-verification
    showed that removing the visited set changes nothing, because depth
    terminates the walk either way. The visited set avoids re-walking a SHARED
    subgraph, which is a cost saving and not a safety property. The node and
    findings caps bound the total work. It runs ONCE per process, so the cost
    is paid to answer a question rather than on every encode.
    """
    import numpy as _np

    out: list = []
    seen: set = set()
    nodes = [0]

    def walk(obj, path, depth):
        if len(out) >= limit or depth > max_depth or nodes[0] > max_nodes:
            return
        nodes[0] += 1
        if isinstance(obj, _np.generic):
            out.append("%s = numpy.%s(%r)" % (path, type(obj).__name__, obj))
            return
        if isinstance(obj, (str, bytes, bytearray, int, float, bool, type(None))):
            return
        if isinstance(obj, _np.ndarray):
            return                      # arrays ARE handled by upstream's hook
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        try:
            if isinstance(obj, dict):
                for k, v in list(obj.items())[:max_items]:
                    # *** A DICT KEY CAN BE THE SCALAR, and this walk could not
                    # *** see one. *** It descended into values only, so a
                    # numpy integer used as a KEY was invisible -- and that is
                    # not a hypothetical shape here:
                    # `OffloadingWorkerMetadata.completed_jobs` is
                    # `{job_id: count}`, a dict keyed by integers, and it
                    # travels to the scheduler in the engine's output. msgpack
                    # has to serialise a key exactly as it serialises a value,
                    # so a numpy key is the same fatal TypeError with none of
                    # the visibility.
                    walk(k, "%s.KEY" % path, depth + 1)
                    walk(v, "%s[%r]" % (path, k), depth + 1)
                return
            if isinstance(obj, (list, tuple, set, frozenset)):
                for i, v in enumerate(list(obj)[:max_items]):
                    walk(v, "%s[%d]" % (path, i), depth + 1)
                return
            slots = getattr(type(obj), "__slots__", None)
            names = (list(slots) if isinstance(slots, (list, tuple))
                     else list(getattr(obj, "__dict__", {}) or {}))
            for n in names[:256]:
                if n.startswith("__"):
                    continue
                try:
                    walk(getattr(obj, n), "%s.%s" % (path, n), depth + 1)
                except Exception:
                    continue
        except Exception:
            return

    try:
        walk(root, "outputs", 0)
    except Exception:
        pass
    return out


def install() -> None:
    """Wrap the encoder's two encode calls to remember their roots, run the
    sampled scan, and answer numpy_producer with a walk from those roots.
    Idempotent: a previous wrapper is unwrapped first, never stacked."""
    try:
        from vllm.v1 import serial_utils as _su

        cls = _su.MsgpackEncoder
    except Exception:
        logger.info("kvmgr: no MsgpackEncoder to scan (upstream layout differs)")
        return
    try:
        _cur_enc = getattr(cls, "encode_into", None)
        if getattr(_cur_enc, "_kvmgr_wrapped", False):
            _prev_enc = getattr(_cur_enc, "_kvmgr_original", None)
            if _prev_enc is not None:
                cls.encode_into = _prev_enc
        root: dict = {}

        # Remember WHAT is being encoded, so the hook can name the field rather
        # than only the value. One attribute store per encode; the walk itself
        # happens once, on the first coercion.
        orig_encode = getattr(cls, "encode_into", None)
        if orig_encode is not None and not getattr(
                orig_encode, "_kvmgr_wrapped", False):
            def encode_into(self, obj, buf, *a, **kw):
                root["obj"] = obj
                root["obj_encode_into"] = obj
                # *** LOOK PROACTIVELY, because the coercion is RARE. ***
                # Two occurrences in a whole evening (value=1, then 39), and a
                # deliberate reproduction attempt -- a restore plus three
                # concurrent 25K requests -- produced none. Waiting for a rare
                # event to name itself is not a strategy when the walk can
                # simply go and look.
                #
                # *** AND THE FIRST VERSION OF THIS SCAN COULD NOT SEE THE
                # *** EVENT IT EXISTED FOR. *** Its budget was "the first 25
                # encodes of the process", which are the BOOT's encodes --
                # handshake and startup traffic, carrying no generated output
                # at all. The scalar only ever appeared under real load. So it
                # ran, cost its walks, found nothing, and read as coverage:
                # exactly the failure this repo keeps paying for.
                #
                # Sampled over the process's LIFE instead, which is the window
                # the event actually lives in. Bounded three ways: a minimum
                # spacing so a busy output thread is not walked per encode, a
                # total count so it cannot run forever, and a stop on the
                # first find, because one path is the answer. The per-walk
                # bounds are tightened too, since this now runs against live
                # output rather than once at boot.
                if (not root.get("found")
                        and root.get("scan", 0) < _SCAN_LIMIT):
                    now = _time.monotonic()
                    if now - root.get("last", 0.0) >= _SCAN_EVERY_S:
                        root["last"] = now
                        root["scan"] = root.get("scan", 0) + 1
                        # *** SILENCE IS NOT SUCCESS, AND THE FIRST TWO
                        # *** VERSIONS OF THIS SCAN OFFERED ONLY SILENCE. ***
                        # With nothing logged unless a scalar is found, "no
                        # PRODUCER line" means EITHER the output path is clean
                        # OR the scan never ran -- and the previous version
                        # genuinely never ran usefully, so that ambiguity is
                        # not hypothetical. One line when it arms and one when
                        # the budget is spent turn a zero into a MEASUREMENT.
                        if root["scan"] == 1:
                            logger.info(
                                "kvmgr: producer scan ARMED -- sampling the "
                                "encoded output object at most once every "
                                "%.0fs, up to %d samples, stopping at the "
                                "first find.", _SCAN_EVERY_S, _SCAN_LIMIT)
                        try:
                            paths = _numpy_paths(obj, limit=6, max_depth=6,
                                                 max_nodes=20000)
                            if paths:
                                root["found"] = True
                                logger.warning(
                                    "kvmgr: numpy scalar PRODUCER (found by "
                                    "scanning encode sample #%d, before any "
                                    "coercion) -- %s",
                                    root["scan"], "; ".join(paths))
                            elif root["scan"] >= _SCAN_LIMIT:
                                logger.info(
                                    "kvmgr: producer scan EXHAUSTED -- %d "
                                    "samples of the encoded output, no numpy "
                                    "scalar reachable by attribute from any "
                                    "of them. That is a measurement, not an "
                                    "absence: the scan ran. A scalar held by "
                                    "a C extension, or one that only appears "
                                    "in a window no sample landed in, would "
                                    "still be invisible here -- the coercion "
                                    "walk remains the backstop.",
                                    root["scan"])
                        except Exception:
                            root["found"] = True  # never scan in a loop
                            logger.exception(
                                "kvmgr: the producer scan raised and is now "
                                "off for this process")
                return orig_encode(self, obj, buf, *a, **kw)
            encode_into._kvmgr_wrapped = True
            encode_into._kvmgr_original = orig_encode
            cls.encode_into = encode_into

        # *** `encode_into` IS NOT THE ONLY ENCODE CALL, and that is very
        # *** probably why the coercion walk reported "none". ***
        # `v1/engine/core.py`'s output loop has TWO: the per-output
        # `encoder.encode_into(outputs, buffer)` and, for the coordinator
        # socket, `coord_socket.send_multipart(encoder.encode(outputs))`. Only
        # the first was wrapped, so when the scalar arrived through `encode()`
        # the walk searched a STALE object from some earlier encode_into and
        # truthfully found nothing -- a walk that is looking in the wrong place
        # reports exactly like a walk that has nothing to find. Measured
        # 2026-10-09: value=1 coerced, walk found none.
        #
        # Both roots are kept SEPARATELY rather than one being overwritten,
        # because the two calls are made from different threads and a single
        # slot would race -- and the path label then says which call carried
        # it, which is the part that identifies the producer.
        _orig_plain = getattr(cls, "encode", None)
        if _orig_plain is not None:
            if getattr(_orig_plain, "_kvmgr_wrapped", False):
                _orig_plain = getattr(_orig_plain, "_kvmgr_original",
                                      _orig_plain)

            def encode(self, obj, *a, **kw):
                root["obj"] = obj
                root["obj_encode"] = obj
                return _orig_plain(self, obj, *a, **kw)

            encode._kvmgr_wrapped = True
            encode._kvmgr_original = _orig_plain
            cls.encode = encode

        def producer(obj):
            # Thorough, not cheap: this runs ONCE per process per numpy type,
            # so the bounds that keep the sampled scan off the hot path are the
            # wrong bounds here.
            found = []
            for _which in ("obj_encode_into", "obj_encode"):
                _o = root.get(_which)
                if _o is None:
                    continue
                for _p in _numpy_paths(_o, limit=12, max_depth=12,
                                       max_nodes=2000000, max_items=4096):
                    found.append("via %s: %s" % (_which, _p))
            if found:
                return "; ".join(found)
            return ("neither encode root reaches it by attribute (roots "
                    "searched: %s). A C extension or a non-attribute container "
                    "holds it, or it was built inside the encode call itself."
                    % (",".join(k for k in ("obj_encode_into", "obj_encode")
                                if root.get(k) is not None) or "none"))

        hooks.register("numpy_producer", producer)
        logger.info("kvmgr: numpy producer scan installed (diag_serializer_walk)")
    except Exception:
        logger.exception("kvmgr: could not install the producer scan")
