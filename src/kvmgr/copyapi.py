"""kvmgr.copyapi -- how a LOAD's copy descriptors reach the driver (ask9).

*** WHY. *** Class B (Xid 32 / Xid 13 / wedge on a group-0 load) is a fault in
`libcuda`'s `cuMemcpyBatchAsync` on GB10 (driver 580.178.04, the 64K-page
kernel), reproduced with nothing but ctypes and libcuda (DSpark
`local-ops/kvbatchrepro.py`; vllm-proxy `docs/kv/history/2026-10-14-xid-outage.md`,
"Test 1"). The same descriptors submitted in batches of 1,024/256/64 back to
back, or one `cuMemcpyAsync` each, ran clean. Production faulted once on a
992-descriptor copy, so no cap is known to be safe; per-copy never touches the
faulting API.

`load_copy_api` (hot, read per submit):
  batch  one `ops.swap_blocks_batch` for the whole copy -- upstream, today
  chunk  the same call on slices of `load_copy_chunk` descriptors, in order,
         back to back on the same stream (no new native code)
  loop   one `cuMemcpyAsync` per descriptor (libcuda.so.1, as the repro)

LOADS ONLY. Stores have never faulted and keep upstream's call.

*** HOW, without copying upstream. *** 0.21's
`CpuGpuOffloadingHandler.transfer_async` reaches the copy through its module's
global `ops` (`from vllm import _custom_ops as ops`), inside
`with torch.cuda.stream(stream)` and between its start and end events. The
global in the module the handler's method actually runs in
(`transfer_async.__globals__`, so a patched copy of the file is covered too) is
replaced by `OpsProxy`: every attribute passes through to the real module,
and `swap_blocks_batch` dispatches on a THREAD-LOCAL mode that only
`DiskLoadHandler` sets, around its own `inner.transfer_async`. Any other
caller -- the store handler, anything upstream -- sees the real call with
nothing in between. The stream, the events, the completion, `wait_compute`
and the ledger are unchanged: the dispatch happens where the one call was.

vLLM 0.31 resolves `self._swap_blocks_batch` once at handler init, so there
the proxy would have to wrap that instance attribute instead; not done (0.21
is what runs).
"""

from __future__ import annotations

import threading
from contextlib import contextmanager

from vllm.logger import init_logger

from . import flags

logger = init_logger("vllm." + __name__)

MODES = ("batch", "chunk", "loop")
_TLS = threading.local()
_WARNED: set = set()
STATS = {"installed": 0, "install_failed": 0, "calls_batch": 0,
         "calls_chunk": 0, "calls_loop": 0, "copies": 0, "descriptors": 0}


def _warn_once(key, msg, *a) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(msg, *a)


def selected() -> tuple:
    """(mode, chunk) from the hot knobs. Unknown -> batch, said once."""
    try:
        m = str(flags.peek("load_copy_api") or "batch").strip().lower()
    except Exception:
        m = "batch"
    if m not in MODES:
        _warn_once(("mode", m), "kvmgr: load_copy_api=%r is not one of %s; "
                   "using batch (upstream's one call)", m, MODES)
        m = "batch"
    try:
        c = int(flags.peek("load_copy_chunk") or 0)
    except Exception:
        c = 0
    if c <= 0:
        c = int(flags.DEFAULTS["load_copy_chunk"])
    return m, c


def calls_for(mode: str, n: int, chunk: int) -> int:
    """Driver calls one copy of `n` descriptors makes. PURE."""
    if n <= 0:
        return 0
    if mode == "loop":
        return n
    if mode == "chunk":
        return -(-n // max(1, chunk))
    return 1


# -- the per-descriptor call --------------------------------------------------

_LIB = []


def _libcuda():
    if not _LIB:
        import ctypes
        lib = ctypes.CDLL("libcuda.so.1")
        f = lib.cuMemcpyAsync
        f.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t,
                      ctypes.c_void_p]
        f.restype = ctypes.c_int
        _LIB.append(f)
    return _LIB[0]


def _stream_handle():
    import torch
    return torch.cuda.current_stream().cuda_stream


def memcpy_async(dst: int, src: int, n: int, stream) -> None:
    """One `cuMemcpyAsync` on `stream`. A non-zero CUresult raises."""
    rc = _libcuda()(dst, src, n, stream)
    if rc != 0:
        raise RuntimeError("kvmgr: cuMemcpyAsync returned CUresult %d" % rc)


def _as_list(t) -> list:
    return t.tolist() if hasattr(t, "tolist") else list(t)


# -- the dispatch -------------------------------------------------------------

class OpsProxy:
    """Stands in for `vllm._custom_ops` in the copy handler's module."""

    def __init__(self, real):
        self._kvmgr_real = real

    def __getattr__(self, name):
        return getattr(self._kvmgr_real, name)

    def swap_blocks_batch(self, src, dst, sizes, *a, **kw):
        real = self._kvmgr_real.swap_blocks_batch
        act = getattr(_TLS, "active", None)
        if act is None:
            return real(src, dst, sizes, *a, **kw)
        mode, chunk = act["mode"], act["chunk"]
        n = len(sizes)
        STATS["copies"] += 1
        STATS["descriptors"] += n
        if mode == "chunk":
            calls = 0
            for i in range(0, n, chunk):
                real(src[i:i + chunk], dst[i:i + chunk], sizes[i:i + chunk],
                     *a, **kw)
                calls += 1
        elif mode == "loop":
            stream = _stream_handle()
            s, d, z = _as_list(src), _as_list(dst), _as_list(sizes)
            for i in range(n):
                memcpy_async(int(d[i]), int(s[i]), int(z[i]), stream)
            calls = n
        else:
            real(src, dst, sizes, *a, **kw)
            calls = 1
        act["calls"] += calls
        STATS["calls_" + mode] += calls
        return None


def install_on(inner) -> "str | None":
    """Put the proxy where `inner.transfer_async` looks up `ops`. Idempotent.
    None = installed (or already); otherwise why not."""
    try:
        g = type(inner).transfer_async.__globals__
    except Exception as exc:
        return "no transfer_async globals: %r" % (exc,)
    cur = g.get("ops")
    # by attribute, not isinstance: a hot reload redefines the class
    if getattr(cur, "_kvmgr_real", None) is not None:
        return None
    if cur is None or not hasattr(cur, "swap_blocks_batch"):
        return "the handler's module has no ops.swap_blocks_batch"
    g["ops"] = OpsProxy(cur)
    STATS["installed"] += 1
    logger.info("kvmgr: load copies go through load_copy_api (batch|chunk|loop, "
                "hot; loads only) in %s", g.get("__name__"))
    return None


@contextmanager
def active(mode: str, chunk: int):
    """Inside: this thread's `ops.swap_blocks_batch` uses `mode`. Yields the
    record, whose `calls` is the driver calls actually made."""
    rec = {"mode": mode, "chunk": int(chunk), "calls": 0}
    prev = getattr(_TLS, "active", None)
    _TLS.active = rec
    try:
        yield rec
    finally:
        _TLS.active = prev


def snapshot() -> dict:
    m, c = selected()
    return dict(STATS, load_copy_api=m, load_copy_chunk=c)
