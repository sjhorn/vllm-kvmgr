"""kvmgr.diag.copyprobe -- is a stuck copy stuck ON THE GPU, and where does it write?

ask3 A and B (2026-10-14). Every incident that day was a disk->GPU load into
group 0. Twice the SAME copy failed on both GPUs (Xid 32 once, a hang once).
During the hang both ranks' main threads sat in cudaEventSynchronize and no CPU
thread was inside a copy call, so "the copy engine never finished" could only
be inferred from SM%.

A. The watchdog thread keeps running when the worker's main loop is blocked.
   On each of its ticks this asks upstream's own end event of every in-flight
   copy `query()` -- non-blocking -- and records `copy_done_on_gpu`. A copy
   past STUCK_S (or any copy while the watchdog says STALLED) is logged once
   at WARNING and written to the ledger as `ev: "stuck"`. That separates
   "the GPU never finished it" (false) from "finished but never observed"
   (true).

B. Once per copy, on a stall or a FAILED submit, writes the copy's real
   descriptors: per tensor its base, row stride, rows and allocation; per ref
   its tensor and size; per block its id and staging row; every descriptor's
   source and destination address; whether each destination lies inside its
   tensor's view and its allocation; alignment; and whether any two
   destination ranges in this ONE copy overlap. The addresses are computed
   the way 0.21's `compute_sub_block_ptrs` does at factor 1:
   `data_ptr() + block_id * stride(0)`, size `ref.page_size_bytes`, no
   per-ref offset. File: `<ledger dir>/copydump-r<rank>-<kind>-<inner>.json`.

Read-only throughout: nothing here submits, waits on, cancels or alters a
transfer, and nothing syncs the GPU. `query()` is the only CUDA call. Off with
`diag_copy_probe` (boot-time knob).
"""

from __future__ import annotations

import json
import os
import threading
import time

from vllm.logger import init_logger

from .. import hooks

logger = init_logger("vllm." + __name__)

# A copy is milliseconds (the 33-block / 2,046-descriptor load included). One
# still open after this is stuck whatever the watchdog's counters say.
STUCK_S = 30.0
# Descriptor rows kept in a dump. 2,046 is the largest copy seen; the cap only
# stops a pathological one from writing megabytes.
MAX_DESCRIPTORS = 8192

_LOCK = threading.Lock()
# (kind, inner) -> {"h", "job", "group", "bids", "slots", "ts", "mono",
#                   "gpu_done", "stuck", "dumped"}
_OPEN: dict = {}
_STATS = {"probes": 0, "stuck": 0, "dumps": 0, "dump_errors": 0,
          "query_errors": 0, "last": None}


# ---------------------------------------------------------------------------
# B: the descriptor dump, pure over anything with data_ptr()/stride()/shape
# ---------------------------------------------------------------------------
def _align(addr: int) -> int:
    """Largest power of two dividing addr, capped at 4096. Pure."""
    if addr <= 0:
        return 0
    return min(addr & -addr, 4096)


def _alloc_bytes(t) -> "int | None":
    try:
        return int(t.untyped_storage().nbytes())
    except Exception:
        return None


def _alloc_base(t) -> "int | None":
    try:
        return int(t.untyped_storage().data_ptr())
    except Exception:
        return None


def describe_copy(kind: str, gpu_tensors, host_tensors, refs, bids, slots) -> dict:
    """Every descriptor of one copy, with checks. Pure; never raises on data.

    kind "load" writes the GPU (dst = gpu), "store" writes staging (dst = host).
    Mirrors upstream's descriptor build for one group at one block per row:
    for each ref, for each block i, src/dst = base + row * stride(0), size =
    ref.page_size_bytes. That is 0.21's `transfer_async`, and vLLM 0.31's
    `_fill_direct_ops` (the layout a handler built without `canonical_layout`
    uses -- ours are; phase14 (14z) pins both and fails if either moves)."""
    gpu_side_is_dst = kind == "load"
    tensors = {}
    for ref in refs:
        t = int(ref.tensor_idx)
        if t in tensors:
            continue
        g, hst = gpu_tensors[t], host_tensors[t]
        tensors[t] = {
            "gpu_base": int(g.data_ptr()), "gpu_stride": int(g.stride(0)),
            "gpu_rows": int(g.shape[0]), "gpu_row_bytes": int(g.shape[1]),
            "gpu_alloc_base": _alloc_base(g), "gpu_alloc_bytes": _alloc_bytes(g),
            "host_base": int(hst.data_ptr()), "host_stride": int(hst.stride(0)),
            "host_rows": int(hst.shape[0]),
        }
    descs, out_of_view, out_of_alloc, too_big = [], [], [], []
    dst_ranges = []
    min_align = None
    for r_i, ref in enumerate(refs):
        t = int(ref.tensor_idx)
        size = int(ref.page_size_bytes)
        T = tensors[t]
        if size > T["gpu_row_bytes"]:
            too_big.append(r_i)
        for b, slot in zip(bids, slots):
            gaddr = T["gpu_base"] + int(b) * T["gpu_stride"]
            haddr = T["host_base"] + int(slot) * T["host_stride"]
            src, dst = (haddr, gaddr) if gpu_side_is_dst else (gaddr, haddr)
            if len(descs) < MAX_DESCRIPTORS:
                descs.append([r_i, t, int(b), int(slot), src, dst, size])
            view_end = T["gpu_base"] + T["gpu_rows"] * T["gpu_stride"]
            if not (T["gpu_base"] <= gaddr and gaddr + size <= view_end):
                out_of_view.append([r_i, int(b)])
            ab, an = T["gpu_alloc_base"], T["gpu_alloc_bytes"]
            if ab is not None and an is not None and not (
                    ab <= gaddr and gaddr + size <= ab + an):
                out_of_alloc.append([r_i, int(b)])
            dst_ranges.append((dst, dst + size, r_i, int(b)))
            for a in (src, dst):
                al = _align(a)
                min_align = al if min_align is None else min(min_align, al)
    # overlap of destination ranges WITHIN this copy
    dst_ranges.sort()
    overlaps, n_ov = [], 0
    for (s0, e0, r0, b0), (s1, e1, r1, b1) in zip(dst_ranges, dst_ranges[1:]):
        if s1 < e0:
            n_ov += 1
            if len(overlaps) < 32:
                overlaps.append({"a": [r0, b0], "b": [r1, b1],
                                 "bytes": int(min(e0, e1) - s1)})
    tidx = [int(r.tensor_idx) for r in refs]
    return {
        "kind": kind, "n_blocks": len(bids), "n_refs": len(refs),
        "n_descriptors": len(refs) * len(bids),
        "descriptors_kept": len(descs),
        "bids": [int(b) for b in bids], "slots": [int(s) for s in slots],
        "refs": [[int(r.tensor_idx), int(r.page_size_bytes)] for r in refs],
        "ref_dup_tensors": len(tidx) - len(set(tidx)),
        "tensors": {str(k): v for k, v in sorted(tensors.items())},
        "all_in_view": not out_of_view, "out_of_view": out_of_view[:64],
        "all_in_alloc": not out_of_alloc, "out_of_alloc": out_of_alloc[:64],
        "ref_bigger_than_row": too_big,
        "min_align": min_align,
        "dst_overlaps": n_ov, "dst_overlap_examples": overlaps,
        # [ref, tensor, block, slot, src, dst, size]
        "descriptor_cols": ["ref", "tensor", "block", "slot", "src", "dst", "size"],
        "descriptors": descs,
    }


def _handler_tensors(h, kind):
    gpu = getattr(h, "_gpu_tensors", None) or []
    host = getattr(h, "_staging", None) or []
    return gpu, host


def _rank(h):
    return getattr(getattr(h, "_spec", None), "rank", None)


def dump(key, reason: str) -> "str | None":
    """B: write one copy's descriptors once. Returns the path, or None."""
    with _LOCK:
        e = _OPEN.get(key)
        if e is None or e.get("dumped"):
            return None
        e["dumped"] = True
    kind, inner = key
    h = e["h"]
    try:
        from . import ledger as _led
        gpu, host = _handler_tensors(h, kind)
        refs = h._refs[e["group"]]
        d = describe_copy(kind, gpu, host, refs, e["bids"], e["slots"])
        d.update({"inner": int(inner), "job": e["job"], "group": e["group"],
                  "reason": reason, "submitted_at": e["ts"],
                  "age_s": round(time.monotonic() - e["mono"], 3),
                  "copy_done_on_gpu": e.get("gpu_done"),
                  "rank": _rank(h), "pid": os.getpid(),
                  "boot": _led.boot_id(),
                  "bids_sha": _led.bids_sha(set(e["bids"]))})
        dirp = _led.ledger_dir(h)
        os.makedirs(dirp, exist_ok=True)
        path = os.path.join(dirp, "copydump-r%s-%s-%d.json" % (
            _rank(h) if _rank(h) is not None else "x", kind, int(inner)))
        tmp = path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(d, fh, separators=(",", ":"))
        os.replace(tmp, path)
        _STATS["dumps"] += 1
        summary = {k: d[k] for k in ("n_blocks", "n_descriptors", "all_in_view",
                                     "all_in_alloc", "dst_overlaps",
                                     "ref_dup_tensors", "min_align")}
        with _led._LOCK:
            _led._wal_write(h, _led.record("copydump", kind, inner, job=e["job"],
                                           group=e["group"], path=path,
                                           reason=reason, **summary))
        logger.warning(
            "kvmgr: COPY DUMP %s inner=%s job=%s group=%s -- %s. %s. "
            "Written to %s", kind, inner, e["job"], e["group"], reason,
            " ".join("%s=%s" % kv for kv in summary.items()), path)
        return path
    except Exception as exc:
        _STATS["dump_errors"] += 1
        logger.warning("kvmgr: copy dump for %s inner=%s failed (%r); the "
                       "transfer is unaffected", kind, inner, exc)
        return None


# ---------------------------------------------------------------------------
# A: the GPU's own answer, asked from the watchdog thread
# ---------------------------------------------------------------------------
def query_copy(h, inner) -> tuple:
    """(done_on_gpu, why). True/False from upstream's end event; None + why
    when there is no event to ask or the query raised. Never blocks."""
    try:
        ev = h._inner._transfer_events.get(inner)
    except Exception as exc:
        return None, "no event table: %r" % (exc,)
    if ev is None:
        return None, "no event: not submitted, or already reaped"
    try:
        return bool(ev.query()), None
    except Exception as exc:
        _STATS["query_errors"] += 1
        # A faulted context answers here with the CUDA error, which is itself
        # the evidence; it is recorded, not raised.
        return None, ("query raised: %r" % (exc,))[:200]


def tick(verdict: str, stalled_for: float) -> None:
    """Watchdog-thread hook: probe every open copy; flag the stuck ones."""
    now = time.monotonic()
    with _LOCK:
        items = list(_OPEN.items())
    for key, e in items:
        kind, inner = key
        done, why = query_copy(e["h"], inner)
        age = now - e["mono"]
        e["gpu_done"] = done
        _STATS["probes"] += 1
        _STATS["last"] = {"kind": kind, "inner": int(inner), "job": e["job"],
                          "age_s": round(age, 3), "copy_done_on_gpu": done,
                          "why": why}
        if e.get("stuck") or not (age >= STUCK_S or verdict == "stalled"):
            continue
        e["stuck"] = True
        _STATS["stuck"] += 1
        meaning = ("the GPU has NOT finished this copy" if done is False else
                   "the GPU FINISHED it but the worker never observed it"
                   if done is True else "the GPU could not be asked: %s" % why)
        logger.warning(
            "kvmgr: COPY PROBE %s inner=%s job=%s group=%s n=%d open %.1fs "
            "(watchdog %s for %.0fs): copy_done_on_gpu=%s -- %s.",
            kind, inner, e["job"], e["group"], len(e["bids"]), age, verdict,
            stalled_for, done, meaning)
        try:
            from . import ledger as _led
            with _led._LOCK:
                _led._wal_write(e["h"], _led.record(
                    "stuck", kind, inner, job=e["job"], group=e["group"],
                    age_s=round(age, 3), copy_done_on_gpu=done,
                    copy_done_why=why, verdict=verdict))
        except Exception:
            pass
        dump(key, "stuck: open %.1fs, copy_done_on_gpu=%s" % (age, done))


# ---------------------------------------------------------------------------
# tracking, from the transfer hooks core already calls
# ---------------------------------------------------------------------------
def on_submit(h, kind: str, inner: int, job, group: int, bids, slots) -> None:
    try:
        e = {"h": h, "job": int(job) if job is not None else None,
             "group": int(group), "bids": [int(b) for b in (bids or ())],
             "slots": [int(s) for s in (slots or ())],
             "ts": time.time(), "mono": time.monotonic(), "gpu_done": None}
        with _LOCK:
            _OPEN[(kind, int(inner))] = e
    except Exception:
        return


def on_done(h, kind: str, inner: int, ok: bool) -> None:
    key = (kind, int(inner))
    if not ok:
        dump(key, "submit failed")
    with _LOCK:
        _OPEN.pop(key, None)


def on_dump(h, reason: str) -> None:
    """The watchdog failed stalled jobs: dump every copy still open on h."""
    with _LOCK:
        keys = [k for k, e in _OPEN.items() if e["h"] is h]
    for k in keys:
        dump(k, "ledger dump: %s" % reason)


def snapshot(h=None) -> dict:
    """Plain ints and strings only -- this reaches msgpack."""
    with _LOCK:
        return {"copy_probe_open": len(_OPEN),
                "copy_probes": int(_STATS["probes"]),
                "copy_stuck": int(_STATS["stuck"]),
                "copy_dumps": int(_STATS["dumps"]),
                "copy_dump_errors": int(_STATS["dump_errors"]),
                "copy_query_errors": int(_STATS["query_errors"]),
                "copy_probe_last": _STATS["last"]}


def install() -> None:
    hooks.register("ledger_submit", on_submit)
    hooks.register("ledger_done", on_done)
    hooks.register("ledger_dump", on_dump)
    hooks.register("watchdog_tick", tick)
    hooks.register("copy_probe_snapshot", snapshot)
