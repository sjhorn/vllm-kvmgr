"""kvmgr.diag.ledger -- the index RANGES each transfer handed the copy kernel.

A GPU fault is reported asynchronously, so its traceback lands at an unrelated
call and says nothing about which transfer was in flight. This keeps a few
integers per transfer -- block-id and staging-slot min/max and a short ring,
never data -- and logs them on a fault or a refusal, so the last thing we asked
the GPU to do survives into the container log.

Moved from kvmgr.diskio on 2026-10-13, where the store and load handlers each
carried an identical copy of the note and snapshot methods; one copy here now.
ON by default (diag_transfer_ledger): it is the only evidence an Xid leaves.

*** v0.3.0 (2026-10-14): A WRITE-AHEAD FILE, AND A LIVE OVERLAP CHECK. ***
The 2026-10-14 06:21:48 Xid 32 on the worker landed in the same second as a
summary line covering 205 transfers, with 50 store jobs outstanding -- and the
summary could not say WHICH transfer faulted, while the ring that could died
with the process. So, below the summary machinery:

  - `submit()` appends one JSON line per GPU transfer to
    `<tier parent>/diag/ledger-r<rank>.jsonl` BEFORE the copy is handed to the
    kernel, and `done()` appends its completion. The open entries after a death
    are the in-flight set at the moment of the fault. `os.write` on an O_APPEND
    descriptor, NO fsync and NO Python buffer: an Xid kills the PROCESS, the
    kernel's page cache survives it, and a userspace buffer would not.
  - a lock-guarded registry of in-flight GPU block ids per rank, checked on
    every submit, which logs `LEDGER OVERLAP` when a new transfer shares GPU
    blocks with an in-flight one and either is a load. It never refuses or
    alters a transfer; it tests the leading hypothesis (a load writing a block
    another transfer is using) on live traffic, before the next Xid.
"""

from __future__ import annotations

import json
import os
import threading
import time

from vllm.logger import init_logger

from .. import flags, hooks

logger = init_logger("vllm." + __name__)


# -- the stall watchdog ---------------------------------------------------
#
# *** WHY THIS EXISTS, AND IT IS A SPECIFIC TEN MINUTES WE WASTED. ***
# On 2026-10-08 the worker went deaf with 21 store jobs outstanding. The engine
# core logged `No available shared memory broadcast block found in 60 seconds`
# TEN TIMES -- a full ten minutes during which the worker was alive and had a
# perfectly walkable Python stack -- and then `VllmWorker-0 died unexpectedly`
# with no traceback at all. The evidence we needed existed for ten minutes and
# nobody was there to send SIGUSR1.
#
# So the dump has to be automatic. `faulthandler.enable()` (register.py) covers
# a DEATH; this covers the far more useful case of a worker that is WEDGED BUT
# ALIVE, which is every store-path failure this project has had.
#
# Three deliberate choices:
#   - the verdict is a PURE FUNCTION, so the interesting logic is testable with
#     no threads, no torch and no GPU -- the same split as file_plan and
#     check_pairing;
#   - it only ever LOGS and DUMPS. A watchdog that cancels jobs or restarts
#     anything is a second failure mode on the critical path, and this repo's
#     rule is that an instrument which can break what it measures is worse than
#     no instrument;
#   - it fires on LACK OF PROGRESS, not on elapsed time. A legitimately slow
#     flush of a 700K prefix advances its counters; a wedge does not. Timing
#     alone would cry wolf on exactly the workload this is for.
_LEDGER_RING = 24


_LEDGER_EVERY = 2000


# *** AND A TIME FLOOR, BECAUSE THE COUNT ALONE LEFT THE ONE FAULT THAT
# *** MATTERED WITH NO RECORD AT ALL. *** Measured 2026-10-11: the engine died
# of Xid 11 at 12:25:47, four seconds into its first real request, and
# `grep "transfer ledger"` over the whole boot returned NOTHING -- the ring had
# nowhere near 2,000 transfers in it. The instrument built specifically so an
# asynchronous fault could be attributed produced silence on the only occasion
# it was needed, which is this project's "a lint that cannot fail reads as
# coverage" wearing the other hat: an instrument that cannot fire reads as a
# clean run.
#
# 15s is chosen against the two things it has to beat: the fault arrived 4s
# after the first transfer, and the stall watchdog already logs every 30s, so
# this is the same order of noise for a strictly more useful line.
_LEDGER_EVERY_S = 15.0


def ledger_dump(h, reason: str) -> None:
    """Log the ledger ring at WARNING. Never raises. Module-level on purpose.

    *** THE RING IS THE FORENSIC PAYLOAD AND IT ONLY EXISTED IN MEMORY. ***
    `/kv/workers` serves it, which is useless in the case it was built for: a
    GPU fault kills the worker, so by the time anybody asks, the process that
    held the ring is gone. The ring has to reach the LOG, where `docker logs`
    keeps it after the container dies -- and this repo already learned to stream
    that log to a file from before an experiment for exactly this reason.

    Module-level so the store and load handlers share one implementation; their
    `_ledger_note` is duplicated per class and the two drifting apart on a
    post-mortem path is not a risk worth taking for symmetry.
    """
    try:
        led = getattr(h, "_ledger", None)
        cls = type(h).__name__
        if led is None:
            logger.warning("kvmgr: ledger dump (%s) -- %s has transferred "
                           "NOTHING. That is an absence, not a range of 0.",
                           reason, cls)
            return
        try:
            rows, slots = h._bounds()
        except Exception:
            rows, slots = -1, -1
        logger.warning(
            "kvmgr: LEDGER DUMP (%s) -- %s, %d transfers, block ids seen in "
            "[%s, %s] of %d rows, slots in [%s, %s] of %d. An index at or past "
            "a bound is the illegal access an Xid reports.",
            reason, cls, led["n"], led["bid_min"], led["bid_max"], rows,
            led["slot_min"], led["slot_max"], slots)
        # newest LAST, with wall-clock times, so the tail can be lined up
        # against the kernel's own Xid timestamp from journalctl -k
        for e in led["recent"][-_LEDGER_RING:]:
            logger.warning("kvmgr:   ledger %s", _ledger_fmt(e))
    except Exception:
        return


def _ledger_fmt(e) -> str:
    """One ring entry as text. Tolerates the older 7-field shape."""
    try:
        kind, group, n, lo, hi, slo, shi = e[:7]
        ts = e[7] if len(e) > 7 else None
        when = "-"
        if ts:
            import time as _t
            when = _t.strftime("%H:%M:%S", _t.localtime(ts)) + ("%.3f" % (ts % 1))[1:]
        return ("%s %s group=%d n=%d block_ids=[%s..%s] slots=[%s..%s]"
                % (when, kind, group, n, lo, hi, slo, shi))
    except Exception:
        return repr(e)


def snapshot(h) -> dict:
    led = getattr(h, "_ledger", None)
    if led is None:
        out = {"transfers": 0, "why": "nothing transferred yet -- an "
                                      "ABSENCE, not an index range of 0"}
    else:
        out = {"transfers": led["n"],
               "block_id_range": [led["bid_min"], led["bid_max"]],
               "slot_range": [led["slot_min"], led["slot_max"]],
               "recent": led["recent"][-8:]}
    out.update(wal_snapshot())
    return out


def note(h, kind: str, group: int, bids, slots) -> None:
    """Record the SHAPE of a transfer, cheaply. Never raises.

    *** WHAT WAS MISSING WHEN THE ENGINE TOOK AN Xid. *** A GPU fault is
    reported asynchronously, so the traceback lands at an unrelated API
    call and `faulthandler` gives a Python stack that says nothing about
    which transfer was in flight. What an investigation actually needs is
    the RANGE of indices recently handed to the copy kernel -- and that is
    four integers per transfer, not the data.

    Deliberately a running min/max plus a small ring rather than a line per
    transfer: a 180K prefill submits thousands of rounds, so per-transfer
    logging would be its own outage. The ring is reported through
    /kv/workers and summarised every `_LEDGER_EVERY` transfers, so a fault
    can be read against the last thing we asked the GPU to do.
    """
    try:
        led = getattr(h, "_ledger", None)
        if led is None:
            led = {"n": 0, "bid_min": None, "bid_max": None,
                   "slot_min": None, "slot_max": None, "recent": []}
            h._ledger = led
        # *** int() IS LOAD-BEARING, NOT TIDINESS. *** `bids` carries the
        # engine's block-id metadata, so its elements can be numpy
        # scalars, and `min` returns whatever it was handed. These four
        # integers travel through /kv/workers' ledger report into the
        # engine core's OUTPUT socket, where msgpack refuses a
        # numpy.int64 and kills the THREAD rather than the engine --
        # results then never reach the API server and clients hang for
        # ever with nothing logged. This was the fourth producer of that
        # fault, found 2026-10-12 when the two-encode-root walk named the
        # path `...['ledger']['block_id_range'][0]`. The serializer shim
        # coerces it downstream; coercing at the SOURCE means the shim
        # never has to, and the shim is a diagnostic rather than a
        # guarantee.
        lo, hi = (int(min(bids)), int(max(bids))) if bids else (None, None)
        slo, shi = ((int(min(slots)), int(max(slots)))
                    if slots else (None, None))
        led["n"] += 1
        for k, v, f in (("bid_min", lo, min), ("bid_max", hi, max),
                        ("slot_min", slo, min), ("slot_max", shi, max)):
            if v is not None:
                led[k] = v if led[k] is None else f(led[k], v)
        # a WALL-CLOCK stamp per entry, because attribution means lining
        # the ring up against `journalctl -k`'s Xid timestamp, and a
        # monotonic clock cannot be compared to the kernel's log
        import time as _lt
        _now_w = _lt.time()
        led["recent"].append((kind, group, len(bids), lo, hi, slo, shi,
                              _now_w))
        if len(led["recent"]) > _LEDGER_RING:
            del led["recent"][:-_LEDGER_RING]
        # EITHER enough transfers OR enough time -- the count alone left
        # the 2026-10-11 Xid 11 with no ledger line at all, four seconds
        # into the boot's first request.
        _stale = (_now_w - led.get("last_log", 0.0)) >= _LEDGER_EVERY_S
        if led["n"] % _LEDGER_EVERY == 0 or _stale:
            led["last_log"] = _now_w
            logger.info(
                "kvmgr: transfer ledger -- %d transfers, block ids seen in "
                "[%s, %s] of %d rows, slots in [%s, %s] of %d. Any index "
                "at or past a row count is the illegal access that shows "
                "up as an Xid.",
                led["n"], led["bid_min"], led["bid_max"],
                h._bounds()[0], led["slot_min"], led["slot_max"],
                h._bounds()[1])
    except Exception:
        return


# -- the write-ahead file -------------------------------------------------
#
# 64 MB and two files (the live one and one `.1`): ~42k transfers a day at
# ~150-400 bytes a line is ~6-17 MB a day, so the pair holds several days --
# far more than a post-mortem needs, and a bound on an SD-card-style concern
# that does not apply here (the tier disk is NVMe) but costs nothing to keep.
_WAL_ROTATE_BYTES = 64 << 20

_LOCK = threading.Lock()
# (kind, inner) -> {"bids": frozenset, "group", "job", "ts"}. Keyed on the
# INNER id with its kind because each direction's SingleDirectionOffloadingHandler
# numbers its own transfers from 0, so a bare id collides across directions.
_INFLIGHT: dict = {}
_STATS = {"overlaps": 0, "overlap_same_group": 0, "overlap_last": None,
          "wal_lines": 0, "wal_errors": 0, "wal_rotations": 0}
_WAL = {"fd": None, "path": None, "bytes": 0, "why": "nothing submitted yet"}


def ledger_dir(h) -> str:
    """`/kvoffload/diag` on the box: the PARENT of the tier root.

    Not inside `spec.root_dir` (`/kvoffload/dspark-kv-v1`), which the boot walk
    and the sweeper enumerate as KV files. The tier mount is the right disk
    because it is writable (the flags directory is mounted read-only) and
    host-backed, so the file outlives the container a fault kills."""
    d = os.environ.get("KVMGR_LEDGER_DIR")
    if d:
        return d
    root = getattr(getattr(h, "_spec", None), "root_dir", None) or "/kvoffload/x"
    return os.path.join(os.path.dirname(root.rstrip("/")), "diag")


def ranges(ids) -> list:
    """Sorted ids as [[lo, hi], ...] runs. Pure.

    The whole SET goes in the line, not just min/max: the post-mortem question
    is "did any in-flight transfer share a block with this one", and min/max
    cannot answer it. Block ids mostly arrive in runs, so this is short."""
    out: list = []
    for b in sorted({int(x) for x in ids}):
        if out and b == out[-1][1] + 1:
            out[-1][1] = b
        else:
            out.append([b, b])
    return out


def expand(rs) -> set:
    return {b for lo, hi in rs for b in range(int(lo), int(hi) + 1)}


def record(ev: str, kind: str, inner: int, **kw) -> str:
    """One ledger line. Pure, so the format is testable without a file."""
    d = {"ts": round(kw.pop("ts", None) or time.time(), 6), "ev": ev,
         "kind": kind, "inner": int(inner), "pid": kw.pop("pid", os.getpid())}
    d.update(kw)
    return json.dumps(d, separators=(",", ":"), sort_keys=True) + "\n"


def open_entries(lines) -> dict:
    """The submits with no matching `done`: the in-flight set. Pure.

    Keyed (pid, kind, inner), because a restarted process numbers its inner
    transfers from 0 again and appends to the same file. After a fault, the
    entries left here for the DEAD pid are exactly what the GPU had been
    handed and had not been seen to finish."""
    open_: dict = {}
    for ln in lines:
        try:
            r = json.loads(ln)
        except Exception:
            continue                    # a torn last line is expected after a death
        k = (r.get("pid"), r.get("kind"), r.get("inner"))
        if r.get("ev") == "submit":
            open_[k] = r
        elif r.get("ev") == "done":
            open_.pop(k, None)
    return open_


def overlaps(kind: str, bids, inflight: dict) -> list:
    """In-flight transfers sharing GPU blocks with a new one. Pure.

    Counted only when EITHER side is a load: a load WRITES GPU blocks, so
    load<->store (a write against a read) and load<->load (two writes) are the
    hazards. Two stores both READ, which is benign and not reported.

    Returns [(other_kind, other_inner, other_job, other_group, sorted common)].
    """
    new = set(int(b) for b in bids)
    out = []
    for (k2, inner2), e in inflight.items():
        if kind != "load" and k2 != "load":
            continue
        common = new & e["bids"]
        if common:
            out.append((k2, inner2, e.get("job"), e.get("group"),
                        sorted(common)))
    return out


def _wal_write(h, line: str) -> None:
    """Append under _LOCK. Never raises; a failure is counted and said once."""
    try:
        if not flags.peek("diag_ledger_file"):
            if _WAL["fd"] is not None:
                os.close(_WAL["fd"])
                _WAL["fd"] = None
            _WAL["why"] = "diag_ledger_file is off"
            return
        if _WAL["fd"] is None:
            d = ledger_dir(h)
            os.makedirs(d, exist_ok=True)
            rank = getattr(getattr(h, "_spec", None), "rank", None)
            path = os.path.join(d, "ledger-r%s.jsonl" % (
                rank if rank is not None else "x"))
            _WAL["fd"] = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                                 0o644)
            _WAL["path"] = path
            _WAL["bytes"] = os.fstat(_WAL["fd"]).st_size
            _WAL["why"] = None
        if _WAL["bytes"] >= _WAL_ROTATE_BYTES:
            os.close(_WAL["fd"])
            os.replace(_WAL["path"], _WAL["path"] + ".1")
            _WAL["fd"] = os.open(_WAL["path"],
                                 os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            _WAL["bytes"] = 0
            _STATS["wal_rotations"] += 1
        b = line.encode()
        os.write(_WAL["fd"], b)
        _WAL["bytes"] += len(b)
        _STATS["wal_lines"] += 1
    except Exception as e:
        _STATS["wal_errors"] += 1
        if _STATS["wal_errors"] == 1:
            logger.warning("kvmgr: ledger file write failed (%r) -- the "
                           "transfer proceeds; the write-ahead record does "
                           "not. Said once; counted in /kv/workers.", e)
        _WAL["why"] = "write failed: %r" % (e,)


def submit(h, kind: str, inner: int, job, group: int, bids, slots) -> None:
    """Called BEFORE the copy is handed to the kernel. Never raises, never
    changes the transfer.

    *** THE SPAN IS submit -> OBSERVED completion, A SUPERSET OF THE GPU's. ***
    Completion is seen when `get_finished` is polled, once per scheduler step,
    so an entry can outlive its copy by up to a step. An OVERLAP here therefore
    means "the scheduler handed a block to a load while our bookkeeping still
    held it in another transfer" -- necessary for the race, not proof of it.
    The ledger file's timestamps are what narrow it."""
    try:
        bset = frozenset(int(b) for b in (bids or ()))
        # int() for the reason note() gives: these reach json AND msgpack, and
        # a numpy scalar would make json.dumps raise -- which the except below
        # would swallow, losing exactly the line this exists to write.
        job = int(job) if job is not None else None
        group = int(group)
        now = time.time()
        with _LOCK:
            hits = overlaps(kind, bset, _INFLIGHT)
            _INFLIGHT[(kind, int(inner))] = {"bids": bset, "group": group,
                                            "job": job, "ts": now}
            _wal_write(h, record(
                "submit", kind, inner, ts=now, job=job, group=group,
                n=len(bset), bid_min=min(bset) if bset else None,
                bid_max=max(bset) if bset else None, bids=ranges(bset),
                slots=ranges(slots or ()),
                overlap=[[k2, i2] for k2, i2, *_ in hits] or None))
            for k2, i2, j2, g2, common in hits:
                _STATS["overlaps"] += 1
                if g2 == group:
                    _STATS["overlap_same_group"] += 1
                _STATS["overlap_last"] = {
                    "ts": now, "new": [kind, int(inner), group],
                    "old": [k2, int(i2), g2],
                    "blocks": len(common)}
        for k2, i2, j2, g2, common in hits:
            logger.warning(
                "kvmgr: LEDGER OVERLAP %s<->%s blocks=%s -- new %s inner=%s "
                "job=%s group=%s shares %d GPU block(s) with in-flight %s "
                "inner=%s job=%s group=%s. Observed, not prevented: the "
                "transfer proceeds.", kind, k2, common[:16], kind, inner, job,
                group, len(common), k2, i2, j2, g2)
    except Exception:
        return


def done(h, kind: str, inner: int, ok: bool) -> None:
    """Called when a transfer's completion is OBSERVED, or its submit failed."""
    try:
        with _LOCK:
            _INFLIGHT.pop((kind, int(inner)), None)
            _wal_write(h, record("done", kind, inner, ok=bool(ok)))
    except Exception:
        return


def wal_snapshot() -> dict:
    """Plain ints and strings only -- this reaches msgpack (see note())."""
    with _LOCK:
        return {"ledger_file": _WAL["path"] if _WAL["why"] is None else None,
                "ledger_file_why": _WAL["why"],
                "inflight_gpu": len(_INFLIGHT),
                "overlaps": int(_STATS["overlaps"]),
                "overlap_same_group": int(_STATS["overlap_same_group"]),
                "overlap_last": _STATS["overlap_last"],
                "wal_lines": int(_STATS["wal_lines"]),
                "wal_errors": int(_STATS["wal_errors"]),
                "wal_rotations": int(_STATS["wal_rotations"])}


def install() -> None:
    hooks.register("ledger_note", note)
    hooks.register("ledger_snapshot", snapshot)
    hooks.register("ledger_dump", ledger_dump)
    hooks.register("ledger_submit", submit)
    hooks.register("ledger_done", done)
