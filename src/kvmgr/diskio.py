"""diskio.py -- Phase 1b: the worker-side GPU -> staging -> file handler.

Split from diskspec.py on purpose: everything here that could silently corrupt
KV is a PURE FUNCTION, testable on a laptop with no torch. The torch-dependent
wiring is a thin shell around it. The parts most likely to be wrong -- which
bytes belong in which file, and which staging slot pairs with which key -- are
exactly the parts that need no GPU to check.

*** WHAT THE FILE FORMAT IS, and why it is ~10x smaller than the tiering tier's.
*** A file holds ONE group's bytes for one block for one rank:

    file(hash, group) = concat over refs in group_data_refs[group] of
                        staging[ref.tensor_idx][slot][:ref.page_size_bytes]

The tiering tier instead wrote `primary_kv_view.strides[0]` bytes -- the WHOLE
page, across every group AND every rank -- into each group's file. Hence five
groups x 2,103,552 bytes per block index against a real per-rank page of
1,051,776, with rank>0's half zero. Ours sums to the per-rank page across the
five files, with no redundancy and no zero tail.

*** THE PAIRING IS THE DANGEROUS PART, and it is CHECKED rather than trusted. ***
The scheduler hands the handler a GPULoadStoreSpec (flat block ids, ordered by
group, with group_sizes) and our DiskLoadStoreSpec (keys). Nothing in the
framework maps one to the other: the pairing is POSITIONAL, and it holds only
because `new_offload_keys` and `src_block_ids` are built by the same group-major
loop over the same `group_state.offload_keys` -- so as long as prepare_store
preserves INPUT ORDER, the orders agree.

That is a load-bearing invariant that no upstream comment states, and the second
loop separately drops blocks whose `block_id == 0`, which would desynchronise the
zip by one. A desync is the worst failure available here: every file exists, has
the right size and no zero tail, and contains ANOTHER BLOCK'S KV. Nothing
crashes. So `check_pairing()` verifies the structure on every transfer and we
REFUSE rather than write on any mismatch.
"""

from __future__ import annotations

import itertools
import os
import threading
from typing import TYPE_CHECKING, Any, NamedTuple

from vllm.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = init_logger("vllm." + __name__)


class Chunk(NamedTuple):
    """One contiguous copy: `staging[tensor_idx][slot][:nbytes]`."""

    tensor_idx: int
    slot: int
    nbytes: int


def file_plan(group_data_refs: "Sequence[Sequence[Any]]", group_idx: int,
              slot: int) -> list[Chunk]:
    """The byte plan for one (key, slot): which staging rows, in which order.

    `group_data_refs[g]` is one CanonicalKVCacheRef PER LAYER in group g, each
    naming a tensor index and that layer's UN-PADDED page size. The tensor's own
    `page_size_bytes` may be padded; the ref's is not, so the ref's is what we
    write -- padding is slack, and writing it would put uninitialised bytes on
    disk and inflate every file.

    Order matters and is the refs' own order: it is what the read path must
    replay to reassemble the block.
    """
    if group_idx < 0 or group_idx >= len(group_data_refs):
        raise IndexError(
            f"kvmgr: group_idx {group_idx} out of range for "
            f"{len(group_data_refs)} KV cache groups -- a key carrying a group "
            f"this engine does not have means the layout changed under us"
        )
    return [Chunk(r.tensor_idx, slot, r.page_size_bytes)
            for r in group_data_refs[group_idx]]


def plan_nbytes(plan: "Sequence[Chunk]") -> int:
    return sum(c.nbytes for c in plan)


def group_file_sizes(group_data_refs: "Sequence[Sequence[Any]]") -> list[int]:
    """Bytes per group, i.e. the size each group's file will be.

    Their SUM is the per-rank page, which is the invariant worth asserting:
    it is the number the tiering tier failed to respect.
    """
    return [plan_nbytes(file_plan(group_data_refs, g, 0))
            for g in range(len(group_data_refs))]


def check_pairing(keys: "Sequence[bytes]", block_ids: "Sequence[int]",
                  group_sizes: "Sequence[int]",
                  group_of: Any) -> str | None:
    """Verify the positional key<->block pairing. Returns a reason, or None.

    Two checks, each catching a distinct desync, both cheap:

      1. equal LENGTH -- catches the scheduler dropping a block whose id was 0
         while the key stayed in our spec, which shifts every later pair by one.
      2. equal PER-GROUP COUNTS -- catches reordering or misgrouping, where the
         lengths agree but a group-2 key lines up with a group-3 block.

    Together they are not a proof of correctness, and this docstring says so
    rather than implying otherwise: two keys within the SAME group could still be
    transposed and both checks would pass. What they do is make every
    desync mode the framework can actually produce loud instead of silent. The
    remaining case needs the read path to compare content, which is Phase 3.
    """
    if len(keys) != len(block_ids):
        return (f"key/block count mismatch: {len(keys)} keys against "
                f"{len(block_ids)} GPU blocks -- a positional zip here would "
                f"write one block's KV into another block's file")
    if sum(group_sizes) != len(block_ids):
        return (f"group_sizes sums to {sum(group_sizes)} but there are "
                f"{len(block_ids)} blocks")
    counts: dict[int, int] = {}
    for k in keys:
        g = group_of(k)
        counts[g] = counts.get(g, 0) + 1
    for g, want in enumerate(group_sizes):
        got = counts.get(g, 0)
        if got != want:
            return (f"group {g}: {got} keys against group_sizes[{g}]={want} -- "
                    f"the keys are not in the scheduler's group-major order")
    return None


class SlotPool:
    """Staging slots, handed out and returned. Pure, so the leak is testable.

    A pool rather than a counter because the whole point of staging is that it is
    SMALL: a few dozen blocks against the 510-block, 1.07 GB CPU tier we are
    replacing.

    *** THIS DOCSTRING USED TO SAY EXHAUSTION WAS SURVIVABLE BECAUSE "the
    connector keeps unsubmitted store jobs and retries". THAT IS FALSE AND IT
    KILLED THE ENGINE ON THE FIRST REQUEST. *** `_unsubmitted_store_jobs` is a
    DEFERRAL queue, not a retry queue: `prepare_store_kv` appends to it so the
    store starts after token sampling, and then `handle_preemptions` and
    `start_kv_transfers` BOTH submit the whole list with a bare `assert success`
    and clear it. There is no backpressure protocol anywhere in the connector,
    so `transfer_async` returning False is FATAL on every path. I inferred a
    retry mechanism from the queue's name and an offline test asserted it.

    So the pool must never be the reason a transfer is refused, and `take_upto`
    rather than `take` is the interface that makes that possible: the handler
    chunks a job across as many staging rounds as it needs. Staging stays small
    AND a job of any size is accepted.
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("kvmgr: staging pool must hold at least one block")
        self._free: list[int] = list(range(size))
        self.size = size
        self.high_water = 0

    def take(self, n: int) -> list[int] | None:
        """All-or-nothing. Kept for the tests that pin the boundary; the handler
        uses take_upto, because an all-or-nothing pool cannot serve a job larger
        than the pool."""
        if n > len(self._free):
            return None
        out = [self._free.pop() for _ in range(n)]
        self.high_water = max(self.high_water, self.size - len(self._free))
        return out

    def take_upto(self, n: int) -> list[int]:
        """As many as are free, up to n. May return fewer, never None.

        The handler needs a PARTIAL answer: a store job can be hundreds of
        blocks (a 48K prompt is ~190 blocks in group 0 alone) against a pool of
        32, and refusing it is fatal. Returning what is available lets the job
        proceed in rounds.
        """
        k = min(n, len(self._free))
        out = [self._free.pop() for _ in range(k)]
        self.high_water = max(self.high_water, self.size - len(self._free))
        return out

    def give(self, slots: "Sequence[int]") -> None:
        for s in slots:
            # A double free would silently alias two blocks onto one slot, which
            # is the corruption this class exists to make impossible.
            if s in self._free:
                raise AssertionError(f"kvmgr: staging slot {s} freed twice")
            self._free.append(s)

    @property
    def available(self) -> int:
        return len(self._free)

    def snapshot(self) -> dict[str, int]:
        return {"size": self.size, "available": len(self._free),
                "high_water": self.high_water}


_tmp_seq = itertools.count()


def write_chunks(path: str, plan: "Sequence[Chunk]", read_row: Any,
                 tmp_suffix: str = ".part") -> int:
    """Write one file ATOMICALLY: temp file, then rename.

    *** ATOMICITY IS NOT HYGIENE HERE, IT IS THE PRESENCE CHECK. *** A reader
    decides a block is on disk with os.path.exists(). A crash or a full disk
    midway through a plain write leaves a SHORT file at the real path, which
    exists() reports as a hit and which would then restore as truncated KV --
    plausible tensors, nothing crashing. Writing to a temp file and renaming
    means a path either does not exist or is complete.

    *** AND THE TEMP NAME MUST BE UNIQUE PER WRITER, WHICH IT WAS NOT. ***
    This used a FIXED `<path>.part`, and that is the bug that took the engine
    down three times on 2026-10-08:

        FileNotFoundError: [Errno 2] No such file or directory:
          '..._r0/ba6/63_g4/ba66347598f1a42....bin'
          File "kvmgr/diskio.py", line 230, in write_chunks
            os.replace(tmp, path)

    Four writer threads, and the same block hash can legitimately be in two
    IN-FLIGHT jobs at once -- `_write_batch`'s `os.path.exists(path)` check
    only skips files that have already COMPLETED, never one being written. So
    two writers took the same `path`, derived the same `<path>.part`, and:

      - one renamed first and the temp vanished, so the other's
        `os.replace(tmp, path)` raised ENOENT -- in a WRITER THREAD, re-raised
        by `fut.result()` inside `get_finished()`, which runs on the worker's
        critical path. That is the crash, and it explains why it looked like
        load: a fresh prompt generates block hashes never seen before, so the
        duplicate-key window is widest exactly when the prompt is big.

      - or they INTERLEAVED into one temp file and produced a file of exactly
        the right LENGTH holding bytes from two different blocks. That is
        worse than the crash: it passes `read_chunks`' size check and restores
        as silently wrong KV -- the `F8_E8M0` hazard class, plausible tensors
        and nothing crashing.

    A unique temp per writer fixes both. Two writers then produce two complete,
    byte-identical files and race only on the rename, where the last one wins
    and either outcome is correct.

    `read_row(tensor_idx, slot, nbytes) -> bytes-like` is injected so this is
    testable with no torch.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # pid + thread + counter: distinct across processes (both ranks share a
    # filesystem path layout), across the writer pool, and across retries.
    tmp = "%s%s.%d.%d.%d" % (path, tmp_suffix, os.getpid(),
                             threading.get_ident(), next(_tmp_seq))
    written = 0
    try:
        with open(tmp, "wb") as f:
            for c in plan:
                buf = read_row(c.tensor_idx, c.slot, c.nbytes)
                mv = memoryview(buf)
                if len(mv) != c.nbytes:
                    raise ValueError(
                        f"kvmgr: staging row gave {len(mv)} bytes, expected "
                        f"{c.nbytes} for tensor {c.tensor_idx} slot {c.slot}")
                f.write(mv)
                written += c.nbytes
        os.replace(tmp, path)
    except BaseException:
        # Leave no short file behind under the real name OR the temp one: a
        # stale .part is harmless to readers but accumulates.
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return written


# ---------------------------------------------------------------------------
# The torch-dependent shell. Everything above is pure and tested on a laptop;
# everything below needs a GPU and is covered only by the box gate. Kept
# deliberately thin for exactly that reason -- the logic that could corrupt KV
# lives above this line.
# ---------------------------------------------------------------------------


def geometry(kv_caches: Any, spec: Any) -> dict:
    """Everything about the live KV geometry, as plain JSON-able data.

    *** WHY THIS EXISTS: THE WORKERS HOLD THE ANSWERS AND HAVE NO CHANNEL. ***
    `kv_caches` lives in the WORKER processes while our HTTP endpoint lives in
    the engine core, so every question about the real geometry has cost a
    4-minute boot to observe. A file in the already-bind-mounted flags directory
    is a cross-process channel that needs no IPC and no new listener, and it is
    written once at construction, so it costs nothing per request.

    The specific question it was built for: our per-group sizes sum to 4,108,352
    B against ~957,537 B per block per rank (KV_CACHE_MEMORY_BYTES /
    num_gpu_blocks). That sum CANNOT be the per-block total -- it would need
    93.5 GB against the 21.8 GB allocated -- so either several groups reference
    the same tensor, or the groups do not all have num_gpu_blocks rows.
    Arithmetic cannot tell those apart; `tensors` and `shared_tensors` below can.
    """
    tensors = []
    for i, t in enumerate(kv_caches.tensors):
        shape = tuple(getattr(t.tensor, "shape", ()) or ())
        tensors.append({
            "idx": i,
            "page_size_bytes": t.page_size_bytes,
            "shape": [int(x) for x in shape],
            # rows is the number of BLOCKS this tensor holds, which is the half
            # of the question arithmetic cannot reach.
            "rows": int(shape[0]) if shape else None,
        })
    groups = []
    owners: dict[int, list[int]] = {}
    for g, refs in enumerate(kv_caches.group_data_refs):
        idxs = [r.tensor_idx for r in refs]
        for ti in idxs:
            owners.setdefault(ti, []).append(g)
        groups.append({
            "group": g,
            "nrefs": len(refs),
            "tensor_idxs": idxs,
            "distinct_tensor_idxs": sorted(set(idxs)),
            "bytes_per_block": sum(r.page_size_bytes for r in refs),
        })
    shared = {str(ti): sorted(set(gs)) for ti, gs in owners.items()
              if len(set(gs)) > 1}
    distinct_bytes = sum(t["page_size_bytes"] for t in tensors)
    return {
        "rank": getattr(spec, "rank", None),
        "world_size": getattr(spec, "world_size", None),
        "block_size_factor": getattr(spec, "block_size_factor", None),
        "whole_page_bytes": getattr(spec, "whole_page_bytes", None),
        "page_size_per_worker": getattr(spec, "page_size_per_worker", None),
        "gpu_block_size": list(getattr(spec, "gpu_block_size", ()) or ()),
        "hash_block_size": getattr(spec, "hash_block_size", None),
        "num_tensors": len(tensors),
        "sum_distinct_tensor_page_bytes": distinct_bytes,
        "sum_group_bytes_per_block": sum(g["bytes_per_block"] for g in groups),
        # *** THE ANSWER. *** Non-empty means groups share tensors, so summing
        # every group's refs double-counts and the 4x is redundancy. Empty means
        # they are disjoint and the per-group rows differ instead.
        "shared_tensors": shared,
        "groups": groups,
        "tensors": tensors,
    }


def write_geometry(kv_caches: Any, spec: Any, out_dir: str) -> str | None:
    """Best-effort: a snapshot must never be able to break a boot."""
    import json

    try:
        g = geometry(kv_caches, spec)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"geometry-r{g.get('rank')}.json")
        tmp = path + ".part"
        with open(tmp, "w") as f:
            json.dump(g, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
        logger.info(
            "kvmgr: geometry snapshot -> %s (%d tensors, groups share tensors: "
            "%s, sum of distinct tensor pages %d B, sum of per-group bytes %d B)",
            path, g["num_tensors"], bool(g["shared_tensors"]),
            g["sum_distinct_tensor_page_bytes"], g["sum_group_bytes_per_block"],
        )
        return path
    except Exception:
        logger.exception("kvmgr: geometry snapshot failed (continuing)")
        return None


# Live handler instances, for `kvmgr.control.reload_modules()` to re-point.
# A list the constructors append to, rather than a registry invented for the
# reload -- a second registry is a second thing to forget to populate, which is
# how `_LIVE_MANAGER` came to be published only after the counters it holds
# turned out to be unreachable from the box.
_LIVE_HANDLERS: list = []

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
_WATCHDOG_HEARTBEAT_S = 30.0
_WATCHDOG_STALL_S = 120.0
_watchdog_started = False


def watchdog_verdict(outstanding, fingerprint, prev_fingerprint,
                     stalled_since, now, stall_s=_WATCHDOG_STALL_S):
    """Decide whether the store path is stalled. Pure.

    Returns `(verdict, stalled_since)` with verdict one of:
      "idle"      nothing outstanding -- nothing to say
      "progress"  outstanding, and a counter moved since last look
      "waiting"   outstanding, nothing moved, but not long enough yet
      "stalled"   outstanding and NOTHING has moved for `stall_s`

    `fingerprint` is any comparable snapshot of progress counters. Comparing a
    fingerprint rather than timing a job is what keeps a slow-but-working flush
    out of the "stalled" bucket.
    """
    if not outstanding:
        return "idle", None
    if fingerprint != prev_fingerprint:
        return "progress", None
    since = now if stalled_since is None else stalled_since
    if now - since >= stall_s:
        return "stalled", since
    return "waiting", since


def _handler_progress():
    """(outstanding jobs, oldest age in seconds, progress fingerprint).

    Best-effort: a handler mid-construction or mid-reload must not make the
    watchdog raise, so anything unreadable is simply skipped.
    """
    outstanding = 0
    oldest = 0.0
    fp = []
    now = _now()
    for h in list(_LIVE_HANDLERS):
        try:
            jobs = getattr(h, "_jobs", None) or {}
            outstanding += len(jobs)
            for st in list(jobs.values()):
                t0 = st.get("t0")
                if t0 is not None:
                    oldest = max(oldest, now - t0)
            st = getattr(h, "stats", {}) or {}
            fp.append((type(h).__name__, st.get("files"), st.get("bytes"),
                       st.get("submitted"), st.get("errors"),
                       st.get("wait_rounds")))
        except Exception:
            continue
    return outstanding, oldest, tuple(fp)


def _watchdog_loop():
    import faulthandler
    import sys as _sys
    import time

    prev_fp = None
    stalled_since = None
    dumped_for = None
    last_beat = 0.0
    while True:
        try:
            time.sleep(5.0)
            outstanding, oldest, fp = _handler_progress()
            now = _now()
            verdict, stalled_since = watchdog_verdict(
                outstanding, fp, prev_fp, stalled_since, now)
            prev_fp = fp
            if verdict == "idle":
                dumped_for = None
                continue
            if verdict == "stalled" and dumped_for != stalled_since:
                dumped_for = stalled_since
                # ERROR first, so the reason is ABOVE the stack in the log and
                # whoever reads it knows what the dump is for.
                logger.error(
                    "kvmgr: STORE PATH STALLED -- %d job(s) outstanding, "
                    "oldest %.0fs, and NOT ONE progress counter has moved for "
                    "%.0fs. The engine is probably fenced behind these jobs "
                    "(jobs_to_flush), which presents as the worker going deaf "
                    "rather than as a slow store. Dumping every thread's "
                    "Python stack below -- this is the evidence that was "
                    "missing when a worker died on 2026-10-08. Nothing has "
                    "been cancelled or restarted.",
                    outstanding, oldest, now - stalled_since)
                try:
                    faulthandler.dump_traceback(file=_sys.stderr,
                                                all_threads=True)
                except Exception:
                    logger.exception("kvmgr: stack dump failed")
            elif now - last_beat >= _WATCHDOG_HEARTBEAT_S:
                last_beat = now
                logger.info(
                    "kvmgr: store path busy -- %d job(s) outstanding, oldest "
                    "%.0fs, verdict=%s", outstanding, oldest, verdict)
        except Exception:
            # Never let the watchdog be the thing that dies.
            try:
                logger.exception("kvmgr: watchdog iteration failed")
            except Exception:
                pass


def _start_watchdog() -> None:
    global _watchdog_started
    if _watchdog_started:
        return
    _watchdog_started = True
    try:
        import threading
        t = threading.Thread(target=_watchdog_loop, name="kvmgr-watchdog",
                             daemon=True)
        t.start()
        logger.info(
            "kvmgr: stall watchdog armed -- if store jobs are outstanding and "
            "no progress counter moves for %.0fs it logs the reason and dumps "
            "every thread's Python stack, automatically. A wedged-but-alive "
            "worker is the failure mode this path actually has, and it needs "
            "nobody present to send a signal.", _WATCHDOG_STALL_S)
    except Exception:
        logger.exception("kvmgr: could not arm the stall watchdog (continuing)")


def _now() -> float:
    import time
    return time.monotonic()


def _elapsed(t0) -> float | None:
    """Seconds since `t0`, or None if it was never recorded.

    None rather than 0.0, and that distinction is load-bearing: the connector
    gates its metric on `transfer_time` being TRUTHY, so a 0.0 would be
    silently dropped exactly as a None is -- but a floor of one microsecond
    keeps a genuinely fast transfer observable instead of invisible. A restore
    nobody can prove happened is the state this whole field exists to leave.
    """
    if t0 is None:
        return None
    return max(_now() - t0, 1e-6)


def copy_into(dst, src) -> int:
    """Copy `src` bytes into the memoryview `dst`, as RAW BYTES on both sides.

    *** THE ONE LINE THAT KILLED THE FIRST REAL RESTORE. *** It was

        self._staging[t][slot][:len(buf)].numpy().data[:] = buf

    and it raised `ValueError: memoryview assignment: lvalue and rvalue have
    different structures` on the first restore that ever reached a GPU tensor.
    The staging tensor is int8, so `.numpy().data` has format `'b'` (signed
    char), while a buffer read off a file is `bytes`, format `'B'`. memoryview
    assignment requires the FORMAT and the SHAPE to agree; it does not coerce.

    It was invisible until the box because the STORE side only ever READS that
    same view -- `_read_row` hands the memoryview to `write_chunks`, which
    writes it out, and reading needs no format agreement at all. So the two
    directions are not symmetric even though they look it, and the asymmetry
    lives in one subscript.

    `.cast("B")` on both sides is the fix, and it is why this is a named
    function rather than an expression: a cast requires a C-contiguous buffer,
    which is a precondition worth stating once and testing once rather than
    re-deriving at each call site. Returns the byte count so a caller can
    account without re-measuring.
    """
    out = memoryview(dst)
    if out.format != "B" or out.ndim != 1:
        out = out.cast("B")
    inp = memoryview(src)
    if inp.format != "B" or inp.ndim != 1:
        inp = inp.cast("B")
    if len(out) != len(inp):
        # A length disagreement here means the file and the geometry disagree,
        # which `read_chunks` is supposed to have already refused. Saying so
        # explicitly beats memoryview's own message, which names neither side.
        raise ValueError(
            f"copy_into: destination is {len(out)} bytes and the source is "
            f"{len(inp)} -- the file and the tensor geometry disagree, so the "
            f"KV would be wrong. Delete presence.json beside the KV and "
            f"restart.")
    out[:] = inp
    return len(inp)


def read_chunks(path: str, plan: "Sequence[Chunk]", write_row: Any) -> int:
    """Read one file back into staging, refusing anything that is not exact.

    *** THE SIZE CHECK HAPPENS BEFORE ANY BYTE IS READ, and that ordering is the
    whole safety property. *** Upstream does not support transfer failure -- its
    worker does `assert transfer_result.success` under the comment "we currently
    do not support job failures" -- so a load has exactly two honest outcomes on
    a bad file: kill the engine core, or hand back KV that is silently wrong.
    There is no third. Checking the length first means a wrong file is caught
    before anything reaches a GPU tensor, and the caller can refuse loudly.

    The atomic rename in write_chunks is what makes a length check sufficient
    for completeness: a path that exists was renamed into place whole, so it is
    never a partial write. What length cannot catch is bit rot or a file from an
    incompatible engine -- the first is accepted, the second is covered by
    FileMapper's config fingerprint plus our own format root.
    """
    want = plan_nbytes(plan)
    try:
        have = os.path.getsize(path)
    except OSError as e:
        raise FileNotFoundError(
            f"kvmgr: {path} is gone, but the presence index claimed it. "
            f"Delete the manifest (presence.json) beside the KV to recover: it "
            f"is a claim about every rank and can only be checked against one."
        ) from e
    if have != want:
        raise ValueError(
            f"kvmgr: {path} is {have} B, expected {want} B. REFUSING to load "
            f"it -- a short or foreign file would decode as plausible KV and "
            f"nothing downstream would notice."
        )
    got = 0
    with open(path, "rb") as f:
        for c in plan:
            buf = f.read(c.nbytes)
            # UNREACHABLE while the length check above precedes this loop, and
            # labelled rather than pretended-tested: break-verification could
            # not make it fire, because a file of the right TOTAL size cannot
            # end early at a chunk boundary of the same plan. Kept as defence in
            # depth for the day someone reorders the two.
            if len(buf) != c.nbytes:
                raise ValueError(
                    f"kvmgr: {path} ended early at chunk {c.tensor_idx} "
                    f"({len(buf)} of {c.nbytes} B)")
            write_row(c.tensor_idx, c.slot, buf)
            got += len(buf)
    return got


class DiskStoreHandler:
    """GPU -> pinned staging -> file, as one OffloadingHandler.

    Two async stages behind one job id. Stage 1 is upstream's
    SingleDirectionOffloadingHandler, which already owns the five-group
    arithmetic, `block_indices`, the sub-block expansion and the CUDA streams --
    reusing it is why this class is short. Stage 2 writes the staging rows out on
    a small thread pool.

    The handler reports a job finished only when BOTH stages are done, because
    the framework reads that completion as "this rank has it on disk" and gates
    the scheduler's all-ranks presence on it.
    """

    def __init__(self, spec, kv_caches, staging_blocks: int, write_threads: int):
        import torch
        from concurrent.futures import ThreadPoolExecutor

        from vllm.utils.platform_utils import is_pin_memory_available
        from vllm.v1.kv_offload.cpu.gpu_worker import (
            SingleDirectionOffloadingHandler,
        )

        self._spec = spec
        self._mapper = spec.file_mapper()
        self._refs = kv_caches.group_data_refs
        self._sizes = group_file_sizes(self._refs)

        # STAGING, NOT A CACHE: torch.empty rather than torch.zeros, and that is
        # deliberate. An unwritten slot read as zeros is precisely what made the
        # tiering tier's half-empty files look like padding for a whole session.
        # Garbage is a bug you can see; zeros are a bug that reads as data.
        pin = is_pin_memory_available()
        gpu_tensors, self._staging = [], []
        for t in kv_caches.tensors:
            gpu_tensors.append(t.tensor.view(torch.int8).view((-1, t.page_size_bytes)))
            self._staging.append(torch.empty(
                (staging_blocks, t.page_size_bytes * spec.block_size_factor),
                dtype=torch.int8, device="cpu", pin_memory=pin))

        self._gpu_tensors = gpu_tensors   # the load handler reuses these
        self._inner = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors, cpu_tensors=self._staging,
            block_size_factor=spec.block_size_factor,
            kv_cache_groups_data_refs=self._refs, gpu_to_cpu=True,
        )
        # Batching per group is only trivial at factor 1 (block_indices enters
        # the inner arithmetic as `block_idx % factor`). Asserted rather than
        # assumed, because a silently wrong offset is the F8_E8M0 class.
        assert spec.block_size_factor == 1, (
            "kvmgr: the disk handler batches per group, which assumes "
            f"block_size_factor == 1 (got {spec.block_size_factor}). Do not set "
            "'block_size' in kv_connector_extra_config.")
        # *** NOT the flags directory: it is mounted READ-ONLY into the
        # container, which the first attempt discovered the honest way --
        # "OSError: [Errno 30] Read-only file system". The offload root is
        # writable by definition, since it is where we write KV. ***
        #
        # And the failure was HARMLESS, which is the design working: the
        # snapshot logged its reason and the boot carried on to construct the
        # handler. A diagnostic that can break a boot is worse than no
        # diagnostic -- the inverse of the lesson from the round before, where
        # my own instrumentation broke a test.
        write_geometry(kv_caches, spec,
                       os.environ.get("KVMGR_SNAPSHOT_DIR", spec.root_dir))
        self._pool = SlotPool(staging_blocks)
        self._writers = ThreadPoolExecutor(max_workers=write_threads,
                                           thread_name_prefix="kvmgr_disk")
        self._jobs: dict[int, dict] = {}          # OUR job id -> rounds left
        self._inner_to_job: dict[int, tuple] = {}  # inner job id -> round
        self._writing: list[tuple[int, Any, list[int]]] = []
        self._next_inner = 1
        self._saturated = False
        self.stats = {"submitted": 0, "refused_pairing": 0,
                      "files": 0, "bytes": 0, "errors": 0,
                      # *** THE FENCE, MEASURED. *** `handle_preemptions`
                      # calls `worker.wait(jobs_to_flush)` SYNCHRONOUSLY on the
                      # worker's critical path, and our wait() blocks until the
                      # bytes are ON DISK. The scheduler's reason for asking is
                      # a fence -- "I am about to reuse GPU block B and a store
                      # for B is still pending" (scheduler.py ~615, ~670, ~847)
                      # -- and that is satisfied once the bytes are in pinned
                      # STAGING, not once they are on disk. These counters are
                      # what say whether moving the fence to staging is worth
                      # the design change; the ~2x prefill slowdown measured
                      # 2026-10-08 is what they have to account for.
                      "wait_calls": 0, "wait_ours": 0,
                      "wait_blocked_ms": 0, "wait_max_pending": 0,
                      "wait_rounds": 0, "saturations": 0}
        logger.info(
            "kvmgr: disk store handler -- staging %d blocks x %d tensors "
            "(pinned=%s), per-group file sizes %s, sum %d B (the per-rank page; "
            "the tiering tier wrote %d B per group instead)",
            staging_blocks, len(self._staging), pin, self._sizes,
            sum(self._sizes), spec.whole_page_bytes,
        )
        _LIVE_HANDLERS.append(self)
        _start_watchdog()

    def transfer_async(self, job_id: int, spec) -> bool:
        """Accept a job of ANY size, and submit it over as many rounds as needed.

        *** WHY THIS CANNOT JUST RETURN False WHEN STAGING IS FULL. *** It used
        to, and it killed the engine on the first request. The connector's
        `_unsubmitted_store_jobs` is a DEFERRAL queue, not a retry queue, and
        both `handle_preemptions` and `start_kv_transfers` submit it with a bare
        `assert success`. So False is fatal on every path -- there is no
        backpressure protocol to lean on. A single 48K-token prompt is ~190
        blocks in group 0 alone against a pool of 32, so "refuse when full"
        meant "refuse always".

        The job is therefore QUEUED and pumped from get_finished(), which the
        engine calls every step. Deferring submission is safe because the
        framework's notion of done is our completion report, and we only report
        a job when every one of its rounds has been written.
        """
        from vllm.v1.kv_offload.base import get_offload_group_idx

        gpu_spec, disk_spec = spec
        keys = list(disk_spec.keys)
        block_ids = list(gpu_spec.block_ids)
        group_sizes = list(gpu_spec.group_sizes)

        # The one refusal that REMAINS fatal, on purpose: a desync would write
        # one block's KV into another block's file, at the right size and with
        # no zero tail. A dead engine is strictly better than that.
        why = check_pairing(keys, block_ids, group_sizes, get_offload_group_idx)
        if why is not None:
            self.stats["refused_pairing"] += 1
            logger.error("kvmgr: REFUSING store job %d -- %s", job_id, why)
            return False

        # Batch PER GROUP, because the inner handler walks group_sizes and a
        # batch that spans groups would need its own block_indices arithmetic.
        # Per group that arithmetic is trivial, which is why the constructor
        # asserts block_size_factor == 1.
        batches = []
        off = 0
        for g, n in enumerate(group_sizes):
            if n:
                batches.append([g, keys[off:off + n], block_ids[off:off + n]])
            off += n
        self._jobs[job_id] = {"batches": batches, "pending": 0, "bytes": 0,
                              "files": 0, "failed": False, "t0": _now()}
        self.stats["submitted"] += 1
        self._pump()
        return True

    def _pump(self) -> None:
        """Submit as many queued rounds as staging allows. Never raises."""
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        ngroups = len(self._refs)
        for job_id, st in self._jobs.items():
            while st["batches"] and not st["failed"]:
                g, bkeys, bids = st["batches"][0]
                slots = self._pool.take_upto(len(bids))
                if not slots:
                    if not self._saturated:
                        # Said ONCE: a saturated pool is now a throughput note,
                        # not an error -- but silence here is what made the
                        # original failure diagnosable only by elimination.
                        self._saturated = True
                        self.stats["saturations"] += 1
                        logger.info(
                            "kvmgr: staging pool saturated (%d blocks); store "
                            "jobs are proceeding in rounds. Raise "
                            "staging_blocks if disk writes fall behind.",
                            self._pool.size)
                    return
                n = len(slots)
                gs = [0] * ngroups
                gs[g] = n
                inner = self._next_inner
                self._next_inner += 1
                try:
                    ok = self._inner.transfer_async(
                        inner,
                        (GPULoadStoreSpec(bids[:n], group_sizes=gs,
                                          block_indices=[0] * ngroups),
                         CPULoadStoreSpec(slots)))
                except Exception:
                    logger.exception("kvmgr: inner GPU->staging submit raised")
                    ok = False
                if not ok:
                    self._pool.give(slots)
                    st["failed"] = True
                    return
                self._inner_to_job[inner] = (job_id, bkeys[:n], slots, g)
                st["pending"] += 1
                if n == len(bids):
                    st["batches"].pop(0)
                else:
                    st["batches"][0] = [g, bkeys[n:], bids[n:]]

    def _read_row(self, tensor_idx: int, slot: int, nbytes: int):
        return self._staging[tensor_idx][slot][:nbytes].numpy().data

    def _write_batch(self, keys, slots, group_idx) -> tuple[int, int]:
        total = files = 0
        for key, slot in zip(keys, slots):
            path = self._mapper.get_file_name(key)
            if os.path.exists(path):
                # write_chunks renames, so an existing path is never a partial.
                continue
            total += write_chunks(path, file_plan(self._refs, group_idx, slot),
                                  self._read_row)
            files += 1
        return total, files

    def get_finished(self) -> list:
        from vllm.v1.kv_offload.worker.worker import TransferResult

        # 1. GPU -> staging completions become file writes.
        for res in self._inner.get_finished():
            ent = self._inner_to_job.pop(res.job_id, None)
            if ent is None:
                continue
            job_id, bkeys, slots, g = ent
            st = self._jobs.get(job_id)
            if st is None:
                self._pool.give(slots)
                continue
            if not res.success:
                self._pool.give(slots)
                st["failed"] = True
                st["pending"] -= 1
                continue
            self._writing.append(
                (job_id, self._writers.submit(self._write_batch, bkeys, slots, g),
                 slots))

        # 2. finished writes free their slots and credit their job.
        still = []
        for job_id, fut, slots in self._writing:
            if not fut.done():
                still.append((job_id, fut, slots))
                continue
            self._pool.give(slots)
            st = self._jobs.get(job_id)
            try:
                nbytes, nfiles = fut.result()
                if st is not None:
                    st["bytes"] += nbytes
                    st["files"] += nfiles
                self.stats["bytes"] += nbytes
                self.stats["files"] += nfiles
            except Exception:
                self.stats["errors"] += 1
                logger.exception("kvmgr: disk write failed (job %d)", job_id)
                if st is not None:
                    st["failed"] = True
            if st is not None:
                st["pending"] -= 1
        self._writing = still

        # 3. room may have freed up; keep the queue moving.
        self._pump()

        # 4. a job is done only when every round of it is.
        out = []
        for job_id in [j for j, st in self._jobs.items()
                       if st["pending"] == 0 and (not st["batches"] or st["failed"])]:
            st = self._jobs.pop(job_id)
            if st["failed"]:
                out.append(TransferResult(job_id=job_id, success=False))
            else:
        # *** transfer_time IS NOT OPTIONAL, and my own Phase 0 note said so.
        # *** `kv_connector_offloading/worker.py:274` gates record_transfer on
        # all THREE of transfer_time (truthily!), transfer_size and
        # transfer_type -- so omitting the time means
        # `vllm:kv_offload_total_bytes_total` has NO SERIES AT ALL. Measured
        # 2026-10-08: the store path was demonstrably writing 1,051 files a run
        # and `transfer_type="..."` matched nothing in /metrics.
        #
        # That is not cosmetic. The ABSENCE of a restore series is exactly what
        # proved `prepare_load` had never been called, back when the tiering
        # tier was in the path -- a stronger statement than "the lookup
        # missed". A handler that omits the time makes its own path
        # unobservable, and this one did.
                out.append(TransferResult(
                    job_id=job_id, success=True, transfer_size=st["bytes"],
                    transfer_time=_elapsed(st.get("t0")),
                    transfer_type=("GPU", self._spec.disk_medium)))
        return out

    def wait(self, job_ids: set) -> None:
        """Block until OUR jobs among `job_ids` are on disk.

        *** ONLY OURS, AND THAT ONE WORD IS A 600-SECOND ENGINE HANG. ***
        Measured live 2026-10-07: the worker stopped consuming shm broadcasts
        for seven minutes with zero files written, and this is why.
        `OffloadingWorker.wait` fans the whole id set out to EVERY registered
        handler (worker.py:172), so the store handler is routinely handed ids
        belonging to the load handler -- and ids whose completion it already
        reported and popped from `self._jobs`. Neither can ever appear in a
        future `get_finished()`, so the old `remaining = set(job_ids)` could
        never empty and spun the full deadline with the engine blocked on it.

        Upstream gets this right and the one line says how
        (`cpu/gpu_worker.py:373`):

            for job_id in job_ids:
                event = self._transfer_events.get(job_id)
                if event is not None:          # <- an unknown id is SKIPPED
                    event.synchronize()

        So an id we do not hold is not a job to wait for; it is somebody
        else's, or already done. Both mean "nothing to do here".

        The deadline is 120s rather than 600s for the same reason the bug cost
        so much: this runs on the engine's critical path with the worker
        unable to consume broadcasts, so a bounded WRONG answer that says so
        beats a correct answer nobody can wait for. A real flush of a 700K
        prefix is rounds of a 32-block pool, which is seconds, not minutes.
        """
        import time

        self.stats["wait_calls"] += 1
        # Snapshot our own ids first: get_finished() pops completed jobs, so
        # testing membership inside the loop would race with our own progress.
        remaining = {j for j in job_ids if j in self._jobs}
        if not remaining:
            return
        self.stats["wait_ours"] += len(remaining)
        self.stats["wait_max_pending"] = max(
            self.stats["wait_max_pending"], len(remaining))
        _t0 = time.monotonic()
        deadline = time.monotonic() + 120.0
        while remaining and time.monotonic() < deadline:
            self.stats["wait_rounds"] += 1
            for r in self.get_finished():
                remaining.discard(r.job_id)
            if remaining:
                time.sleep(0.0005)
        self.stats["wait_blocked_ms"] += int((time.monotonic() - _t0) * 1000)
        if remaining:
            # Name them: a flush that timed out has left KV unpersisted for a
            # block the scheduler is about to reuse, and which job it was is
            # the only thing that makes that traceable afterwards.
            logger.error(
                "kvmgr: wait() timed out after 120s with %d of OUR job(s) "
                "outstanding: %s. staging=%s jobs_open=%d writes_open=%d",
                len(remaining), sorted(remaining), self._pool.snapshot(),
                len(self._jobs), len(self._writing))

    def shutdown(self) -> None:
        self._writers.shutdown(wait=True)

    def snapshot(self) -> dict:
        return dict(self.stats, staging=self._pool.snapshot(),
                    group_file_sizes=self._sizes,
                    jobs_open=len(self._jobs), writes_open=len(self._writing))


class DiskLoadHandler:
    """file -> pinned staging -> GPU. The stages are the store's, reversed.

    *** THE DESIGN IS SHAPED BY ONE UPSTREAM FACT: THERE IS NO SUCH THING AS A
    FAILED LOAD. *** The connector worker does `assert transfer_result.success`
    under the comment "we currently do not support job failures", and the
    scheduler has no success branch at all -- it counts down pending_count and
    calls complete_load. So a load that reports done is BELIEVED.

    That leaves exactly two honest answers when a file is wrong: report failure
    and kill the engine core, or report success and hand back KV that is
    silently wrong. There is no third. We choose the first, loudly, every time --
    a dead engine beats wrong tokens, and the log names the one file to delete.

    Which is why `read_chunks` checks the length BEFORE reading a byte, and why
    `lookup` carries the real safety burden: it is the only place that can keep
    us out of that choice altogether.
    """

    def __init__(self, spec, kv_caches, staging_blocks: int, read_threads: int,
                 store: "DiskStoreHandler"):
        from concurrent.futures import ThreadPoolExecutor

        from vllm.v1.kv_offload.cpu.gpu_worker import (
            SingleDirectionOffloadingHandler,
        )

        self._spec = spec
        self._mapper = spec.file_mapper()
        self._refs = store._refs
        # SHARE the store handler's staging and pool. Two pools would double the
        # pinned memory for no reason, and a load and a store never need the same
        # slot at once -- the pool is what serialises them.
        self._staging = store._staging
        self._pool = store._pool
        self._inner = SingleDirectionOffloadingHandler(
            gpu_tensors=store._inner.gpu_tensors
            if hasattr(store._inner, "gpu_tensors") else store._gpu_tensors,
            cpu_tensors=self._staging,
            block_size_factor=spec.block_size_factor,
            kv_cache_groups_data_refs=self._refs,
            gpu_to_cpu=False,
        )
        self._readers = ThreadPoolExecutor(max_workers=read_threads,
                                          thread_name_prefix="kvmgr_diskrd")
        self._jobs: dict[int, dict] = {}
        self._reading: list[tuple[int, Any, list[int], list, int]] = []
        self._inner_to_job: dict[int, tuple] = {}
        self._next_inner = 1_000_000          # disjoint from the store's ids
        # The LOAD handler shares wait()'s shape, so it shares the counters --
        # a missing key here is a KeyError in the fence, which is exactly the
        # bug the offline fixture-parity assertion caught before it shipped.
        self.stats = {"submitted": 0, "refused_pairing": 0, "files": 0,
                      "bytes": 0, "errors": 0,
                      "wait_calls": 0, "wait_ours": 0, "wait_blocked_ms": 0,
                      "wait_max_pending": 0, "wait_rounds": 0,
                      "saturations": 0}
        logger.info("kvmgr: disk LOAD handler installed (restore is live)")
        _LIVE_HANDLERS.append(self)
        _start_watchdog()

    def _write_row(self, tensor_idx: int, slot: int, buf) -> None:
        # copy_into, never a bare `[:] = buf`: the staging tensor's view is
        # format 'b' and a file buffer is 'B', and memoryview assignment does
        # not coerce. See copy_into's docstring -- this is the line that killed
        # the first restore that ever reached a GPU tensor.
        copy_into(self._staging[tensor_idx][slot][:len(buf)].numpy().data, buf)

    def _read_batch(self, keys, slots, group_idx) -> int:
        total = 0
        for key, slot in zip(keys, slots):
            path = self._mapper.get_file_name(key)
            total += read_chunks(path, file_plan(self._refs, group_idx, slot),
                                 self._write_row)
            self.stats["files"] += 1
        return total

    def transfer_async(self, job_id: int, spec) -> bool:
        from vllm.v1.kv_offload.base import get_offload_group_idx

        disk_spec, gpu_spec = spec
        keys = list(disk_spec.keys)
        block_ids = list(gpu_spec.block_ids)
        group_sizes = list(gpu_spec.group_sizes)
        why = check_pairing(keys, block_ids, group_sizes, get_offload_group_idx)
        if why is not None:
            self.stats["refused_pairing"] += 1
            logger.error("kvmgr: REFUSING load job %d -- %s", job_id, why)
            return False
        batches, off = [], 0
        for g, n in enumerate(group_sizes):
            if n:
                batches.append([g, keys[off:off + n], block_ids[off:off + n]])
            off += n
        self._jobs[job_id] = {"batches": batches, "pending": 0, "bytes": 0,
                              "failed": False, "t0": _now()}
        self.stats["submitted"] += 1
        self._pump()
        return True

    def _pump(self) -> None:
        for job_id, st in self._jobs.items():
            while st["batches"] and not st["failed"]:
                g, bkeys, bids = st["batches"][0]
                slots = self._pool.take_upto(len(bids))
                if not slots:
                    return
                n = len(slots)
                # Stage 1 for a LOAD is the FILE READ, on the thread pool.
                fut = self._readers.submit(self._read_batch, bkeys[:n], slots, g)
                self._reading.append((job_id, fut, slots, bids[:n], g))
                st["pending"] += 1
                if n == len(bids):
                    st["batches"].pop(0)
                else:
                    st["batches"][0] = [g, bkeys[n:], bids[n:]]

    def get_finished(self) -> list:
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
        from vllm.v1.kv_offload.worker.worker import TransferResult

        ngroups = len(self._refs)
        still = []
        for job_id, fut, slots, bids, g in self._reading:
            if not fut.done():
                still.append((job_id, fut, slots, bids, g))
                continue
            st = self._jobs.get(job_id)
            try:
                nbytes = fut.result()
            except Exception:
                # A dead engine beats wrong tokens. The message read_chunks
                # raises names the single file and the recovery.
                self.stats["errors"] += 1
                logger.critical(
                    "kvmgr: *** DISK RESTORE FAILED for job %d -- refusing to "
                    "hand back KV we cannot verify. The engine core will stop. "
                    "Delete presence.json beside the KV and restart. ***",
                    job_id, exc_info=True)
                self._pool.give(slots)
                if st is not None:
                    st["failed"] = True
                    st["pending"] -= 1
                continue
            if st is not None:
                st["bytes"] += nbytes
            self.stats["bytes"] += nbytes
            # Stage 2: staging -> GPU.
            gs = [0] * ngroups
            gs[g] = len(bids)
            inner = self._next_inner
            self._next_inner += 1
            ok = self._inner.transfer_async(
                inner, (CPULoadStoreSpec(slots),
                        GPULoadStoreSpec(bids, group_sizes=gs,
                                         block_indices=[0] * ngroups)))
            if not ok:
                self._pool.give(slots)
                if st is not None:
                    st["failed"] = True
                    st["pending"] -= 1
                continue
            self._inner_to_job[inner] = (job_id, slots)
        self._reading = still

        for res in self._inner.get_finished():
            ent = self._inner_to_job.pop(res.job_id, None)
            if ent is None:
                continue
            job_id, slots = ent
            self._pool.give(slots)
            st = self._jobs.get(job_id)
            if st is None:
                continue
            if not res.success:
                st["failed"] = True
            st["pending"] -= 1

        self._pump()

        out = []
        for job_id in [j for j, st in self._jobs.items()
                       if st["pending"] == 0 and (not st["batches"] or st["failed"])]:
            st = self._jobs.pop(job_id)
            # The time matters more on this side than on the store's: see
            # DiskStoreHandler.get_finished. A restore with no series is a
            # restore nobody can prove happened.
            out.append(TransferResult(
                job_id=job_id, success=not st["failed"],
                transfer_size=st["bytes"] if not st["failed"] else None,
                transfer_time=_elapsed(st.get("t0")),
                transfer_type=(self._spec.disk_medium, "GPU")))
        return out

    def wait(self, job_ids: set) -> None:
        """Block until OUR loads among `job_ids` are in GPU memory.

        Same bug, same fix, same reason as DiskStoreHandler.wait -- see the
        long note there. `OffloadingWorker.wait` hands every id to every
        handler, so filtering to ids we actually hold is the whole of it; an
        id we do not hold is the store handler's, or already reported.

        It is written out twice rather than shared because the two handlers
        own different dicts and a shared helper would have to take the dict as
        an argument, which is more indirection than four lines deserve. What
        matters is that they do not DIVERGE, and a test asserts the property
        for both.
        """
        import time

        self.stats["wait_calls"] += 1
        rem = {j for j in job_ids if j in self._jobs}
        if not rem:
            return
        self.stats["wait_ours"] += len(rem)
        self.stats["wait_max_pending"] = max(
            self.stats["wait_max_pending"], len(rem))
        _t0 = time.monotonic()
        deadline = time.monotonic() + 120.0
        while rem and time.monotonic() < deadline:
            self.stats["wait_rounds"] += 1
            for r in self.get_finished():
                rem.discard(r.job_id)
            if rem:
                time.sleep(0.0005)
        self.stats["wait_blocked_ms"] += int((time.monotonic() - _t0) * 1000)
        if rem:
            # A load that timed out is worse than a store that did: the
            # scheduler has already told the engine those tokens are computed.
            logger.error(
                "kvmgr: LOAD wait() timed out after 120s with %d of OUR "
                "job(s) outstanding: %s. jobs_open=%d reads_open=%d",
                len(rem), sorted(rem), len(self._jobs), len(self._reading))

    def shutdown(self) -> None:
        self._readers.shutdown(wait=True)

    def snapshot(self) -> dict:
        return dict(self.stats, jobs_open=len(self._jobs),
                    reads_open=len(self._reading))
