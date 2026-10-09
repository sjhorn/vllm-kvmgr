"""kvmgr.register -- the vllm.general_plugins entry point.

DELIVERY.  A general plugin, discovered through the entry-point group
``vllm.general_plugins``, exactly as ``patches/safetensors_e8m0_plugin`` and the
dispram carveout are.  That keeps this off the whole-file mount list: a fourth
1,809-line copy of an upstream file is how this stack ended up pinned to an
August engine in October.

WHY MUTATING A DICT IS ENOUGH.  ``get_manager_for_kv_cache_spec`` reads
``spec_manager_map[type(kv_cache_spec)]`` at CALL time, and the managers are
built inside ``KVCacheCoordinator.__init__`` in the engine-core process, which
runs after plugins load.  So an import-time mutation is picked up.  Asserted
offline in test_phase0.py rather than assumed.

WHY WE ALSO WRAP THE COORDINATOR.  Two caps divide by
``SingleTypeKVCacheManager._num_managers``, a process-global class counter that
is incremented per manager built and never decremented -- so anything that
constructs a second set of managers halves every cap in the running engine,
silently.  The coordinator's ``__init__`` is the one place that holds the real
``KVCacheConfig``, so we read the group count and the per-group block sizes off
it and stamp them on our class.  The counter stays as the fallback and says so
when it is used.

The wrap is also what makes ``protect_alloc=by_need`` possible at all: it needs
every group's block size, and a single manager only knows its own.
"""

from __future__ import annotations

import time as _time

from vllm.logger import init_logger

# *** THE LOGGER NAME IS LOAD-BEARING, and `__name__` is the wrong one. ***
#
# vLLM's logging config gives a handler to the `vllm` logger ONLY, with
# propagate=false, and configures no root handler. So a logger named
# `kvmgr.flags` has NO handler and an EFFECTIVE LEVEL OF 30 (WARNING) --
# every logger.info() it makes is silently dropped. Measured in the running
# container 2026-10-07:
#
#   init_logger("kvmgr.register")       handlers [] effective 30  -> dropped
#   init_logger("vllm.kvmgr.register")  handlers [] effective 20  -> printed
#
# That cost us the whole "did it take?" story on the first deploy: the
# registration line, the cap-on-change line, the flag-change line and the env
# audit were all invisible, which is indistinguishable from the plugin never
# having loaded. Naming the logger under `vllm.` makes it inherit that handler.
logger = init_logger("vllm." + __name__)

_REGISTERED = False


def register() -> None:
    """Idempotent: plugins load once per process, but there are several
    processes and a re-entrant import is cheaper to tolerate than to debug."""
    global _REGISTERED
    if _REGISTERED:
        return

    _install_hang_dump()
    _install_control_plane()

    from vllm.v1.core import single_type_kv_cache_manager as stm
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from .manager import DSparkMLAAttentionManager

    previous = stm.spec_manager_map.get(MLAAttentionSpec)
    if previous is DSparkMLAAttentionManager:
        _REGISTERED = True
        return

    # Phase 1 owns GROUP 0 ONLY -- MLAAttentionSpec, the full-MLA group whose
    # cap is the one that binds.  The other four spec types stay on upstream's
    # classes, which is both the narrow rollout and the fallback if this is
    # wrong.
    #
    # NOTE for Phase 5: this is NOT sufficient for disk restore.  The hybrid
    # coordinator converges the hit length to the MINIMUM across all five
    # groups, so restoring only group 0 buys nothing -- every spec type has to
    # be registered before a restore can lengthen a hit.  See
    # docs/local/KVMGR-PHASE0-20261007.md finding 1.
    stm.spec_manager_map[MLAAttentionSpec] = DSparkMLAAttentionManager
    logger.info(
        "kvmgr: registered %s for MLAAttentionSpec (was %s); group 0 only, "
        "policy still gated on kvmgr_enabled",
        DSparkMLAAttentionManager.__name__,
        getattr(previous, "__name__", previous),
    )

    _install_group_geometry_hook(DSparkMLAAttentionManager)
    _install_offload_scheduler_hook()
    _install_serializer_shim()

    # Opt-in, and it logs nothing unless offload_allow_hybrid is set. Installed
    # from register() so it lands in every process that loads the plugin --
    # the scheduler side is the one that raises.
    from . import offload

    offload.install()

    # *** OUR OWN OffloadingSpec, registered by NAME through the factory's public
    # registry -- no upstream file is patched. *** It exists because upstream's
    # tiering offload cannot capture half of this cluster's KV: it consolidates
    # every rank's shard into ONE SHARED HOST region, which is single-host TP by
    # construction, while ours is one rank per node. Measured: every .bin the
    # tiering tier wrote has rank>0's half EXACTLY ZERO.
    #
    # Registering the name is INERT. Nothing selects it unless
    # KV_TRANSFER_CONFIG names it as spec_name, and that lives in .env.dspark
    # per node -- so the committed default stays "no offload at all", and a
    # restart without it reverts to known-good rather than to a half-built path.
    #
    # Import is DEFERRED into the factory's own resolution (module_path +
    # class_name) rather than imported here, so a syntax error in diskspec.py
    # cannot take down the manager registration that every boot depends on.
    try:
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        from kvmgr.diskspec import DSPARK_SPEC_NAME
        OffloadingSpecFactory.register_spec(
            DSPARK_SPEC_NAME, "kvmgr.diskspec",
            DSPARK_SPEC_NAME,
        )
        logger.info(
            "kvmgr: registered DSparkDiskOffloadingSpec (inert until "
            "KV_TRANSFER_CONFIG names it as spec_name)"
        )
    except ValueError:
        pass          # already registered: register() ran twice in one process
    except Exception:
        # A missing kv_offload is not a reason to lose the KV manager, which is
        # what this plugin is actually for.
        logger.exception("kvmgr: could not register the disk offloading spec")

    _REGISTERED = True




def _install_control_plane() -> None:
    """Install the HTTP control plane's three hooks, best-effort.

    Three, because three kinds of process read these knobs and each needs a
    different one: the Worker methods are the RPC targets, the EngineCore wrap
    is what makes the SCHEDULER apply a knob too (the executor broadcasts only
    to workers), and the router hook attaches our routes to the app the API
    server builds. Every one is a no-op in a process that lacks its target, so
    this runs unconditionally and the misses are silent by design.

    Best-effort throughout: a control plane that can break a boot is worse than
    no control plane, which is the same rule the geometry snapshot and the hang
    dump already follow.
    """
    try:
        from . import control

        control.install_worker_methods()
        control.wrap_engine_core_rpc()
        control.wrap_engine_core_init()
        control.install_router_hook()
    except Exception:
        logger.exception("kvmgr: control plane install failed (continuing; the "
                         "flags file still governs every knob)")

def _install_hang_dump() -> None:
    """Make a HANG dumpable with no ptrace, because twice now it was not.

    *** THE PROBLEM THIS SOLVES. *** The store path has wedged the worker twice
    (Phase 1c, and again 2026-10-07 at the staging-saturation point). Both times
    the only evidence available was "the worker stopped consuming shm broadcasts"
    plus thread wchans, because the container has no `SYS_PTRACE` -- `gdb -p`
    answers `ptrace: Operation not permitted` and there is no py-spy. So a
    deadlock in OUR code was diagnosable only by reading it and guessing, which
    is how the Phase 1c attempt came to be built on a wrong premise.

    `faulthandler` needs no ptrace and no attach: the signal handler runs inside
    the process and walks every thread's Python stack itself. Output goes to
    stderr, which is what `docker logs` shows, so the dump lands beside the log
    lines that led up to it.

        docker exec <c> kill -USR1 <worker pid>   # every thread's Python stack

    SIGUSR1 rather than SIGABRT or SIGQUIT: those two kill the process, and a
    wedged engine holding a 700K-token prefix is worth more alive than dead --
    the dump is for diagnosis, not for a postmortem. Registered per process from
    the plugin, so the worker, the engine core and the API server all answer it.

    *** `chain=True` KILLS THE PROCESS, AND THE TEST IS WHAT FOUND IT. ***
    faulthandler chains to whatever handler was installed when it registered,
    and for SIGUSR1 that is normally `SIG_DFL` -- whose default action is
    TERMINATE. So the obvious spelling dumps the stack and then kills the
    engine, which on a box holding a 700K-token prefix would destroy the thing
    being diagnosed in the act of diagnosing it. Measured offline: the child
    exited with `-30`, i.e. died of its own diagnostic signal.

    So chain only to a REAL handler -- one somebody actually installed -- and
    never to the default. Checked in the running image 2026-10-07: vLLM
    installs no SIGUSR1 or SIGUSR2 handler anywhere (0 hits across its tree),
    so in practice nothing is chained and nothing is stolen; the check is there
    so that a future vLLM taking SIGUSR1 keeps working rather than losing it.

    The whole thing is best-effort: an instrument that can break the boot is
    worse than no instrument.
    """
    try:
        import faulthandler
        import signal
        import sys as _sys

        if not hasattr(signal, "SIGUSR1"):
            return
        # *** THE PASSIVE NET, AND ITS ABSENCE COST US THE 2026-10-08
        # *** TRACEBACK. `register(SIGUSR1)` only answers a signal somebody
        # SENDS, so it is useless for a process that dies on its own -- and on
        # 2026-10-08 `Worker proc VllmWorker-0 died unexpectedly` with NO Python
        # traceback anywhere, which is the signature of a fatal signal. There
        # was no kernel log either (dmesg needs root here), so the cause is
        # still unknown and cannot now be recovered.
        #
        # `faulthandler.enable()` installs handlers for SIGSEGV, SIGBUS,
        # SIGFPE, SIGABRT and SIGILL, dumps every thread's Python stack to
        # stderr -- i.e. into `docker logs` -- and then RE-RAISES, so the
        # process still dies exactly as it would have. It is a witness, not a
        # mitigation: it changes nothing about behaviour and cannot keep a
        # broken process alive.
        #
        # Unlike register(), chaining is not a hazard here: enable() always
        # re-raises on the default handler for a genuinely fatal signal, which
        # is the correct outcome. The register() trap below is specifically
        # about SIGUSR1, whose default action is terminate.
        faulthandler.enable(file=_sys.stderr, all_threads=True)

        existing = signal.getsignal(signal.SIGUSR1)
        chain = callable(existing) and existing not in (
            signal.SIG_DFL, signal.SIG_IGN)
        faulthandler.register(signal.SIGUSR1, file=_sys.stderr,
                              all_threads=True, chain=chain)
        logger.info(
            "kvmgr: hang dump armed -- faulthandler.enable() is the PASSIVE "
            "net (a fatal signal now dumps every thread's Python stack to the "
            "container log and then re-raises, so a death is no longer "
            "silent), and `kill -USR1 <pid>` is the ACTIVE one for a process "
            "that is wedged but alive. No ptrace needed, which is the point: "
            "gdb cannot attach in this container and the store path has wedged "
            "the worker twice with no stack available. chain=%s (a real "
            "handler was already installed: %s).",
            chain, chain,
        )
    except Exception:
        logger.exception("kvmgr: could not arm the hang dump (continuing)")

# The proactive scan's budget. Spacing keeps it off the per-encode path;
# the limit keeps it from being a standing cost. Both are deliberately
# generous in TIME and stingy in RATE: the event is rare, not fast.
_SCAN_EVERY_S = 5.0
_SCAN_LIMIT = 600


def _numpy_paths(root, limit: int = 12, max_depth: int = 8,
                 max_nodes: int = 200000) -> list:
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
                for k, v in list(obj.items())[:512]:
                    walk(v, "%s[%r]" % (path, k), depth + 1)
                return
            if isinstance(obj, (list, tuple, set, frozenset)):
                for i, v in enumerate(list(obj)[:512]):
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


def _install_serializer_shim() -> None:
    """Make a numpy SCALAR serialisable, and say where it came from.

    *** THIS KILLED THE ENGINE'S OUTPUT THREAD TWICE AND PRESENTED AS A HANG.
    *** `vllm/v1/serial_utils.py`'s `enc_hook` handles `np.ndarray` but NOT
    numpy scalars -- `np.int64` is not an ndarray -- so a single numpy integer
    anywhere in `EngineCoreOutputs` raises:

        TypeError: Object of type <class 'numpy.int64'> is not serializable
          core.py:1561 in process_output_sockets

    And it kills that THREAD, not the engine. Results then never reach the API
    server: requests complete, the core goes idle in `queue.get()` on an empty
    queue, clients hang for ever, and the main loop logs NOTHING. I spent an
    evening reading that as a memory ceiling.

    *** WHY A SHIM AND NOT JUST A FIX AT THE SOURCE. *** I did fix our source
    (`file_plan` coerces now) and it recurred, so at least one numpy scalar
    reaches the encoder from somewhere else -- possibly not ours at all. A
    lossless `.item()` is strictly better than killing the output thread
    whoever produced it, and the alternative on offer
    (`VLLM_ALLOW_INSECURE_SERIALIZATION=1`) buys the same robustness by
    enabling pickle for everything, which is a much larger blast radius.

    It is also the only instrument that can NAME the culprit: the hook is
    handed the offending object, so one log line carries its type and value,
    which no traceback does. Logged once per distinct type so a hot path cannot
    flood.
    """
    try:
        import numpy as _np
        from vllm.v1 import serial_utils as _su
    except Exception:
        return
    try:
        cls = _su.MsgpackEncoder
    except Exception:
        logger.info("kvmgr: no MsgpackEncoder to shim (upstream layout differs)")
        return
    try:
        # *** A SHIM THAT CAN ONLY BE INSTALLED ONCE PER PROCESS IS A
        # *** DIAGNOSTIC THAT NEEDS A RESTART TO CHANGE. ***
        # The early return here was correct about idempotence and wrong about
        # UPDATES: `importlib.reload` re-executes this module, but the class
        # attribute still held the OLD closure, so every change to the shim or
        # the producer scan cost a ~4-minute boot plus a cold prefill. That
        # happened twice in one session -- once for the scan's window, once for
        # its own observability. Same family as "a knob whose OFF state needs a
        # boot is not a knob", and as "THE RELOADER IS ITSELF THE OLD VERSION".
        #
        # So UNWRAP FIRST, then wrap fresh. The original is carried on the
        # wrapper itself, which is the only place that survives a reload --
        # a module global would be reset by the very reload we are serving.
        original = cls.enc_hook
        if getattr(original, "_kvmgr_wrapped", False):
            original = getattr(original, "_kvmgr_original", None)
            if original is None:
                # Wrapped by something that is not us, or by a version that
                # did not record its original. Refuse rather than wrap a
                # wrapper: stacking them would call the old scan as well.
                logger.warning(
                    "kvmgr: enc_hook is already wrapped and carries no "
                    "original -- leaving it alone. The shim is live; a CHANGE "
                    "to it needs a restart.")
                return
        _cur_enc = getattr(cls, "encode_into", None)
        if getattr(_cur_enc, "_kvmgr_wrapped", False):
            _prev_enc = getattr(_cur_enc, "_kvmgr_original", None)
            if _prev_enc is not None:
                cls.encode_into = _prev_enc
        seen: set = set()
        root: dict = {}

        # Remember WHAT is being encoded, so the hook can name the field rather
        # than only the value. One attribute store per encode; the walk itself
        # happens once, on the first coercion.
        orig_encode = getattr(cls, "encode_into", None)
        if orig_encode is not None and not getattr(
                orig_encode, "_kvmgr_wrapped", False):
            def encode_into(self, obj, buf, *a, **kw):
                root["obj"] = obj
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

        def enc_hook(self, obj):
            if isinstance(obj, _np.generic):
                name = type(obj).__name__
                if name not in seen:
                    seen.add(name)
                    try:
                        paths = _numpy_paths(root.get("obj"))
                        logger.warning(
                            "kvmgr: numpy scalar PRODUCER -- %s",
                            "; ".join(paths) if paths else
                            "the walk found none, so it is not reachable by "
                            "attribute from the encoded root (a __slots__ type "
                            "or a C extension holds it)")
                    except Exception:
                        logger.exception("kvmgr: the producer walk raised")
                    logger.warning(
                        "kvmgr: coerced a numpy scalar in the engine's OUTPUT "
                        "path -- numpy.%s value=%r -> %r. Upstream's enc_hook "
                        "handles ndarrays but not scalars, and the raise kills "
                        "process_output_sockets, which strands every result "
                        "and presents as a hang with no error. Lossless, but "
                        "the PRODUCER should be fixed: this value is the clue.",
                        name, obj, obj.item())
                return obj.item()
            return original(self, obj)

        enc_hook._kvmgr_wrapped = True
        enc_hook._kvmgr_original = original
        cls.enc_hook = enc_hook
        logger.info(
            "kvmgr: serializer shim installed -- a numpy scalar in the output "
            "path is now coerced and logged once with its value, instead of "
            "killing the output socket thread and stranding every result")
    except Exception:
        logger.exception("kvmgr: could not install the serializer shim")


def _install_offload_scheduler_hook() -> None:
    """Publish upstream's OffloadingConnectorScheduler so /kv can read its jobs.

    *** THE ONE NUMBER THAT WAS MISSING. *** The worker reports `jobs_open: 0`
    while the scheduler can still be holding a hundred jobs, and the
    scheduler's `self._jobs` is exactly what `jobs_to_flush` names -- so the
    fence waits on jobs the worker has already finished and forgotten. Nothing
    could see that: `jobs_to_flush` appeared only in a crash's dump_input, which
    is a terrible place to learn it.

    `complete_store` fires only when `pending_count` hits 0 and it starts at
    `num_workers`, so BOTH ranks must report. A histogram of pending_count over
    the residual jobs distinguishes "one rank's completion is lost" from
    "neither arrived" in a single reading, which is why it is worth a hook.

    Wrapping __init__ rather than hunting the object: same pattern as the
    coordinator wrap below, and it is the only place guaranteed to run before
    any job exists. Best-effort throughout -- an instrument that can break a
    boot is worse than no instrument, and this one reaches into another
    project's private state.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            scheduler as _osched,
        )
    except Exception:
        logger.info("kvmgr: no OffloadingConnectorScheduler to hook "
                    "(upstream layout differs); /kv will report sched "
                    "unavailable rather than zero")
        return
    try:
        cls = _osched.OffloadingConnectorScheduler
        if getattr(cls.__init__, "_kvmgr_wrapped", False):
            return
        original = cls.__init__

        def __init__(self, *a, **kw):
            original(self, *a, **kw)
            try:
                from kvmgr import diskspec as _ds
                # Mutate in place: a reload may already have handed the new
                # module's list to somebody else.
                del _ds._LIVE_OFFLOAD_SCHED[:]
                _ds._LIVE_OFFLOAD_SCHED.append(self)
                logger.info(
                    "kvmgr: offload scheduler published -- /kv now reports its "
                    "open job set and the pending_count histogram, which is "
                    "what says whether a store completion never arrived")
            except Exception:
                logger.exception("kvmgr: could not publish the offload scheduler")

        __init__._kvmgr_wrapped = True
        cls.__init__ = __init__
    except Exception:
        logger.exception("kvmgr: could not hook the offload scheduler (continuing)")


def _install_group_geometry_hook(manager_cls) -> None:
    """Stamp the real group count / block sizes from the KVCacheConfig."""
    from vllm.v1.core import kv_cache_coordinator as kcc

    base = kcc.KVCacheCoordinator
    if getattr(base.__init__, "_kvmgr_wrapped", False):
        return

    original = base.__init__

    def __init__(self, kv_cache_config, *args, **kwargs):
        try:
            groups = list(kv_cache_config.kv_cache_groups)
            manager_cls.group_count = len(groups)
            manager_cls.group_block_sizes = tuple(
                g.kv_cache_spec.block_size for g in groups
            )
            # `sliding_window` is absent on a full-attention spec and present on
            # the windowed ones.  getattr-with-None is the discriminator, and it
            # is what makes by_need correct rather than a 21x regression.
            manager_cls.group_sliding_windows = tuple(
                getattr(g.kv_cache_spec, "sliding_window", None) for g in groups
            )
            logger.info(
                "kvmgr: KV geometry from KVCacheConfig -- %d groups, block sizes %s, "
                "windows %s, pool %d blocks",
                manager_cls.group_count, manager_cls.group_block_sizes,
                manager_cls.group_sliding_windows, kv_cache_config.num_blocks,
            )
        except Exception:
            # Never take the engine down for observability. The counter
            # fallback in _num_groups() covers us, and logs that it did.
            logger.exception(
                "kvmgr: could not read the KV geometry off KVCacheConfig; "
                "falling back to _num_managers, which is a process-global "
                "counter and may be wrong"
            )
        out = original(self, kv_cache_config, *args, **kwargs)
        # *** AND STASH EVERY MANAGER, because a POOL-WIDE operation needs
        # POOL-WIDE releases. *** Measured 2026-10-08: `gpu_cache_epoch` fired,
        # released group 0's protection and was REFUSED, with the log naming the
        # cause in one number -- "pool free blocks 22536 -> 22581", and +45 is
        # exactly group 0's share of an 11,764-token prompt (11764/256 = 46).
        # We are registered for MLAAttentionSpec only, so our instance can
        # release only its own group; the other four run upstream's
        # SlidingWindowMLAManager and held ~1,700 blocks that patch7's
        # `num_used_blocks != 1` gate then refused on, forever.
        #
        # `release_protected_prompt_blocks` is defined on
        # SingleTypeKVCacheManager (patch6:691), so it is available on every
        # manager including the ones we did not replace. This is the only place
        # that holds the whole tuple -- the coordinator builds it at :67 -- which
        # is why the stash goes here rather than being discovered later.
        try:
            manager_cls.all_managers = tuple(
                getattr(self, "single_type_managers", ()) or ())
            logger.info(
                "kvmgr: holding %d manager(s) for pool-wide releases "
                "(ours: %d)",
                len(manager_cls.all_managers),
                sum(1 for m in manager_cls.all_managers
                    if isinstance(m, manager_cls)),
            )
        except Exception:
            logger.exception(
                "kvmgr: could not stash the manager list; a GPU cache drop will "
                "release only group 0 and patch7 will refuse the reset")
        return out

    __init__._kvmgr_wrapped = True  # type: ignore[attr-defined]
    base.__init__ = __init__  # type: ignore[method-assign]
