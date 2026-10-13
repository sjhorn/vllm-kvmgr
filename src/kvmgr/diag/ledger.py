"""kvmgr.diag.ledger -- the index RANGES each transfer handed the copy kernel.

A GPU fault is reported asynchronously, so its traceback lands at an unrelated
call and says nothing about which transfer was in flight. This keeps a few
integers per transfer -- block-id and staging-slot min/max and a short ring,
never data -- and logs them on a fault or a refusal, so the last thing we asked
the GPU to do survives into the container log.

Moved from kvmgr.diskio on 2026-10-13, where the store and load handlers each
carried an identical copy of the note and snapshot methods; one copy here now.
ON by default (diag_transfer_ledger): it is the only evidence an Xid leaves.
"""

from __future__ import annotations

from vllm.logger import init_logger

from .. import hooks

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
        return {"transfers": 0, "why": "nothing transferred yet -- an "
                                       "ABSENCE, not an index range of 0"}
    return {"transfers": led["n"],
            "block_id_range": [led["bid_min"], led["bid_max"]],
            "slot_range": [led["slot_min"], led["slot_max"]],
            "recent": led["recent"][-8:]}


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


def install() -> None:
    hooks.register("ledger_note", note)
    hooks.register("ledger_snapshot", snapshot)
    hooks.register("ledger_dump", ledger_dump)
