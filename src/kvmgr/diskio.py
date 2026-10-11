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
    # *** int(), AND IT IS NOT COSMETIC: IT KILLED THE ENGINE'S OUTPUT THREAD.
    # *** `page_size_bytes` comes from the engine's own tensor metadata, so it
    # can be a numpy/torch integer. That flows Chunk.nbytes -> plan_nbytes ->
    # st["bytes"] -> TransferResult.transfer_size -> KVConnectorStats -> the
    # engine core's OUTPUT SOCKET, which serialises with msgpack and refuses:
    #
    #   TypeError: Object of type <class 'numpy.int64'> is not serializable
    #     vllm/v1/engine/core.py:1561 in process_output_sockets
    #
    # And the consequence is the nastiest shape available: that thread dies, so
    # RESULTS NEVER REACH THE API SERVER. Requests complete, the core goes
    # idle, clients hang for ever, and the main loop logs no error at all --
    # which is exactly the "wedge" that looked like a memory problem for an
    # hour. Coerced at the source so the whole byte chain is builtin.
    return [Chunk(int(r.tensor_idx), int(slot), int(r.page_size_bytes))
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


def check_bounds(block_ids: "Sequence[int]", slots: "Sequence[int]",
                 gpu_rows: "int | None", staging_rows: "int | None",
                 group_sizes: "Sequence[int]", direction: str) -> str | None:
    """Refuse a transfer whose indices are out of range. PURE. Returns a reason
    or None.

    *** WHY THIS EXISTS: AN Xid 13, AND IT IS THE ONE THING OUR CODE COULD DO
    *** TO CAUSE ONE. *** 2026-10-08, one second before the engine took
    `CUDA error: unspecified launch failure`:

        NVRM: Xid (PCI:000f:01:00): 13, Graphics Exception: ... Data deaddead

    Xid 13 is a graphics exception -- in practice an illegal memory access from
    a kernel, i.e. a SOFTWARE fault. And this box logged **zero** Xid lines
    across the previous two boots (three weeks of uptime) and **six** on the
    boot where the disk offload path went live, which is the only new thing
    doing bulk GPU<->pinned-host DMA.

    `check_pairing` already guards which KEY goes with which block, but it
    explicitly does NOT bound the indices themselves -- and a block id past the
    GPU tensor's row count, or a staging slot past the pinned buffer's, is
    exactly how a copy kernel reads or writes off the end. The kernel does not
    validate; it faults, asynchronously, and the traceback lands somewhere
    unrelated. So bound them HERE, where the numbers are still Python and a
    refusal can name them.

    This is a guard, not a diagnosis: if it never fires, our indices were never
    the cause and the Xid is somewhere else -- which is itself worth knowing and
    is why the counter is reported rather than only logged.
    """
    # *** A BOUND OF None MEANS "CANNOT CHECK", NOT "EVERYTHING IS OUT OF
    # *** BOUNDS". *** This killed the engine on 2026-10-08. `_bounds()`
    # returned 0 rows for a rebound load handler, and 0 made every legitimate
    # block id "outside [0, 0)", so the guard refused a valid restore round,
    # failed the job and took the worker's exception path with it:
    #
    #   REFUSING a restore round -- staging->GPU: GPU block id 21586 is
    #   outside [0, 0)
    #
    # ABSENT IS NOT ZERO, inside the guard written to prevent a crash. The
    # length and group-sum checks need no bounds at all, so they still run;
    # only the range checks are skipped, and the caller logs that once.
    if len(block_ids) != len(slots):
        return ("%s: %d block ids against %d staging slots -- the pairing is "
                "positional, so a length mismatch would silently transfer the "
                "wrong rows" % (direction, len(block_ids), len(slots)))
    tot = sum(group_sizes)
    if tot != len(block_ids):
        return ("%s: group_sizes sum to %d but there are %d block ids; the "
                "inner handler splits one flat array by these counts"
                % (direction, tot, len(block_ids)))
    if gpu_rows is None or staging_rows is None:
        return None          # cannot check the ranges; the shape checks passed
    for b in block_ids:
        if not (0 <= b < gpu_rows):
            return ("%s: GPU block id %d is outside [0, %d) -- a copy kernel "
                    "would read or write off the end of the KV tensor, which "
                    "is an illegal access and surfaces as Xid 13"
                    % (direction, b, gpu_rows))
    for sl in slots:
        if not (0 <= sl < staging_rows):
            return ("%s: staging slot %d is outside [0, %d) -- past the end of "
                    "the pinned buffer" % (direction, sl, staging_rows))
    return None


def staging_plan(pool_capacity: int, slots_knob: int, write_floor_knob: int,
                 read_floor_knob: int) -> tuple[int, int, int]:
    """Resolve the three staging knobs. PURE, so the arithmetic is testable.

    Returns `(live_size, write_floor, read_floor)`.

    *** THE FLOORS MUST NEVER ADD UP TO THE WHOLE POOL. *** If each direction
    reserved half, both would see zero available and both would stall for ever
    -- a deadlock built out of two anti-starvation measures, presenting as a
    hang with no cause. Each floor is therefore clamped to `size - 1`, so the
    other direction can always take at least one slot.

    0 means "derive it": the whole allocation for the size, a quarter of the
    live size for each floor. So an unset knob is not a silent constant, and
    raising the size scales the floors with it.
    """
    size = max(1, int(slots_knob) if int(slots_knob) > 0 else int(pool_capacity))

    def _floor(k: int) -> int:
        f = int(k) if int(k) > 0 else size // 4
        return max(0, min(f, size - 1))

    return size, _floor(write_floor_knob), _floor(read_floor_knob)


def load_ops_for(refs, group_idx: int, nblocks: int) -> int:
    """COPY OPS one submit will contain. PURE, because it is the real limit.

    *** THE BLOCK CAP BOUNDS THE WRONG QUANTITY, and reading upstream is what
    *** showed it. *** `_custom_ops.swap_blocks_batch` says it "submit[s] all
    copies in a single driver call" -- `cuMemcpyBatchAsync` on CUDA 12.8+ -- and
    `cpu/gpu_worker.py` builds that call as

        num_copy_ops += group_size * len(group_data_refs)

    So the descriptors in one driver call are **blocks x refs-for-that-group**,
    not blocks. `load_batch_max=192` therefore means something different per
    group: a group with 30 layer refs submits 5,760 descriptors for the same
    192 blocks that a 2-ref group submits 384 for. Group 0 has the most layers
    AND the largest block size, so it is the first to reach any descriptor
    limit, and a single cap in BLOCKS cannot express that at all.

    Pure and separate so the arithmetic can be tested without a GPU, and so the
    number is derived from the engine's own refs rather than from a constant
    somebody guessed.
    """
    try:
        return int(nblocks) * len(refs[group_idx])
    except Exception:
        # Unknown geometry must not silently become "no limit". One ref per
        # block is the most optimistic reading, so it caps at nblocks and the
        # caller still bounds something.
        return int(nblocks)


def load_ops_budget(refs, group_idx: int, want_blocks: int, ops_max: int) -> int:
    """How many blocks may go in ONE submit, bounded by the op budget. PURE.

    Never returns 0 for a positive `want_blocks`: a round of one block is
    always allowed, because refusing every round would be a restore that can
    never make progress -- and upstream's connector cannot express a refusal
    anyway, so "submit nothing, for ever" is worse than "submit one".
    """
    want = int(want_blocks)
    if want <= 0:
        return 0
    if not ops_max or ops_max <= 0:
        return want
    try:
        per_block = max(1, len(refs[group_idx]))
    except Exception:
        per_block = 1
    return max(1, min(want, int(ops_max) // per_block))


# THE SAME ARITHMETIC, FOR THE OTHER DIRECTION. `load_ops_for` /
# `load_ops_budget` compute descriptors-per-submit from the engine's own refs,
# and nothing in them is specific to reading: `num_copy_ops += group_size *
# len(group_data_refs)` is how upstream builds the driver call in BOTH
# directions. The `load_` names are kept because the load's tests and source
# lints name them; the store calls these neutral ones so a reader of the store
# path is not left wondering why it asks a load function.
ops_for = load_ops_for
ops_budget = load_ops_budget


def _store_batch_max() -> int:
    """Max blocks in ONE store round. Hot.

    *** THE STORE HAD NO CAP AT ALL, and that is what stalled the stream. ***
    The load got `load_batch_max` and `load_ops_max` because an unbounded round
    FAULTED the GPU; the store was left taking `take_upto(len(bids))`, which is
    the whole 256-slot pool, and its `while` loop then kept submitting until the
    pool drained. So one scheduler step could carry ~256 blocks x 62 refs of
    group 0.

    Measured 2026-10-11 by `scripts/test-meter.sh` section 1 in the vllm-proxy
    repo: a single 1.052s stall 72% of the way through a stream, max inter-event
    gap 1.137s against a 0.102s baseline, and A/B/A against `disk_store_paused`
    (1.109 live / 0.111 paused / 1.067 live). The writes themselves are NOT the
    cause -- they are already on a ThreadPoolExecutor and `_collect` only polls
    `fut.done()`. What runs on the engine's step is the submit: building that
    many descriptors, plus the GIL the writer threads then contend for.
    """
    try:
        from kvmgr import flags as _f
        v = int(_f.peek("store_batch_max") or 0)
        if v > 0:
            return v
        return int(_f.DEFAULTS["store_batch_max"]) or 32
    except Exception:
        return 32


def _store_ops_max() -> int:
    """Max COPY-OP DESCRIPTORS in one store round. Hot.

    The block cap bounds the wrong quantity on a hybrid layout -- see
    `load_ops_for`. Group 0 has the most layer refs and the largest block, so it
    reaches any descriptor limit first, and a cap in blocks alone means a
    different submit size per group.
    """
    try:
        from kvmgr import flags as _f
        v = int(_f.peek("store_ops_max") or 0)
        if v > 0:
            return v
        return int(_f.DEFAULTS["store_ops_max"]) or 2048
    except Exception:
        return 2048


def _store_inflight_max() -> int:
    """Max store rounds whose GPU->staging copy may be outstanding at once. Hot.

    *** THE PER-ROUND CAP ALONE DOES NOT BOUND THE STEP. *** `_pump`'s `while`
    submits round after round until the pool is empty, so capping each round at
    32 blocks against a 256-slot pool just means eight rounds instead of one and
    the same work per step. This is the knob that actually bounds what a single
    scheduler step carries, exactly as `load_inflight_max` is for the read side
    -- where the per-round cap was respected throughout and the engine died
    anyway at 861 keys.

    Returning early is safe and is not a refusal: `_pump` is called again from
    `_collect` on every step, and `kv_connector_no_forward` means those steps
    arrive even with nothing scheduled, so the work resumes the moment a round
    completes.
    """
    try:
        from kvmgr import flags as _f
        v = int(_f.peek("store_inflight_max") or 0)
        if v > 0:
            return v
        return int(_f.DEFAULTS["store_inflight_max"]) or 1
    except Exception:
        return 1


def _load_ops_max() -> int:
    """Max copy-op descriptors in ONE staging->GPU submit. Hot. 0 = unbounded.

    The DIMENSION here is read from upstream's code; the THRESHOLD is still
    empirical and this comment should not pretend otherwise. What is known:
    ~256 blocks in group 0 faults and ~224 does not, and group 0 carries the
    most refs -- so the failing descriptor count is whatever 256 x refs[0]
    comes to on this model. The default is deliberately well under that so the
    bound is a margin rather than a boundary, and the first measurement with
    `refs_per_group` exposed will let it be set from arithmetic instead.
    """
    try:
        from kvmgr import flags as _f
        v = int(_f.peek("load_ops_max") or 0)
        if v > 0:
            return v
        # the DECLARED default, never a second copy: this fallback and
        # flags.DEFAULTS drifted once (4 here against 1 there, after 4 was
        # measured to kill the engine), so a zero knob reinstated the fatal
        # value through the safety path. The literal is the import-failed case.
        return int(_f.DEFAULTS["load_ops_max"]) or 2048
    except Exception:
        return 2048


def _load_inflight_max() -> int:
    """Max load ROUNDS whose GPU copy may be outstanding at once. Hot.

    *** THE BATCH CAP WAS NOT ENOUGH, which is why this exists. ***
    `load_batch_max=192` made a 327-key restore clean 3/3 where an uncapped
    256-slot pool faulted 2/2 -- and at 861 keys (the 700K scale test,
    2026-10-10) it faulted anyway, with the per-round cap respected throughout.
    So what moved between clean and dead is not the size of any one batch.
    It is how many are in flight TOGETHER: a 700K prompt is chunked prefill, so
    the scheduler issues many `prepare_load` calls, `_pump` loops over EVERY
    job, and each takes slots until the pool empties. Upstream's inner handler
    then chains them all on one stream with an event per transfer, and nothing
    anywhere bounds that queue.

    0 means unbounded, which is the behaviour that faulted. The default is
    deliberately small: a restore that is slightly less parallel is worth far
    more than one that kills the worker, and the whole point of the measurement
    this enables is that `1` makes the path fully serial and therefore a clean
    test of the hypothesis.
    """
    try:
        from kvmgr import flags as _f
        v = int(_f.peek("load_inflight_max") or 0)
        if v > 0:
            return v
        # the DECLARED default, never a second copy: this fallback and
        # flags.DEFAULTS drifted once (4 here against 1 there, after 4 was
        # measured to kill the engine), so a zero knob reinstated the fatal
        # value through the safety path. The literal is the import-failed case.
        return int(_f.DEFAULTS["load_inflight_max"]) or 1
    except Exception:
        return 1


def _load_batch_max() -> int:
    """Max blocks in ONE staging->GPU `transfer_async`. Hot.

    ~256 faults the worker with Xid 13; 224 and below are measured clean. The
    default is deliberately below the measured-clean ceiling rather than at it,
    because the threshold is only bracketed to (224, 256] and a limit sitting
    exactly on the last value that happened to work is not a margin.

    An unreadable knob must never stop a restore, so a bad value falls back to
    the default rather than raising -- but 0 or negative would mean "no blocks
    ever", so those fall back too.
    """
    try:
        from kvmgr import flags as _f
        v = int(_f.peek("load_batch_max") or 0)
        if v > 0:
            return v
        # the DECLARED default, never a second copy: this fallback and
        # flags.DEFAULTS drifted once (4 here against 1 there, after 4 was
        # measured to kill the engine), so a zero knob reinstated the fatal
        # value through the safety path. The literal is the import-failed case.
        return int(_f.DEFAULTS["load_batch_max"]) or 192
    except Exception:
        return 192


def staging_knobs(pool_capacity: int) -> tuple[int, int, int]:
    """`staging_plan` against the live flags. Imported locally, matching this
    file's style and keeping `flags` out of the module's import graph."""
    try:
        from kvmgr import flags as _f
        return staging_plan(pool_capacity, _f.peek("staging_slots"),
                            _f.peek("staging_write_floor"),
                            _f.peek("staging_read_floor"))
    except Exception:
        # An unreadable knob must not stop a transfer: fall back to the whole
        # allocation with derived floors, which is the documented default.
        return staging_plan(pool_capacity, 0, 0, 0)


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


    *** HOT SIZE, WITHIN A FIXED ALLOCATION. *** The pinned staging is 63
    tensors held by both handlers and by two SingleDirectionOffloadingHandler
    instances, so REALLOCATING it on a live store path is not a tuning knob, it
    is a corruption risk -- on a box with ~3GB available and no OOM net. So the
    allocation is a CEILING chosen at boot (`capacity`), and the hot knob moves
    only how many of those rows the pool hands out (`limit`). A slot is an index
    into rows that already exist, so raising and lowering the limit is instant
    and cannot allocate anything.

    Shrinking does not claw back slots that are in use: they are simply not
    re-offered once returned, so the pool drains to the new limit.

    *** AND A FLOOR PER DIRECTION, NOT A SPLIT. *** The store and load handlers
    SHARE this pool (`self._pool = store._pool`), and that sharing is what made
    the same restore take 574.9s with stores live and 3.3s with them paused --
    a 174x spread. `take_upto` hands out whatever is free and the pump advances
    once per SCHEDULER STEP, so fewer slots per round means fewer blocks moved
    per step: the cold pass had the whole pool for writes and ran at 177 MB/s,
    the restore pass shared it and managed 17.7 MB/s on the same disk.
    
    A hard split would waste half the pool whenever only one direction is busy.
    A `reserve` floor keeps full utilisation when one side is idle while
    guaranteeing the other can always make progress -- which matters because a
    starved WRITE is not merely slow: its blocks stay in `jobs_to_flush`,
    fencing GPU blocks, and that is the path that deferred a request until
    NCCL's watchdog killed a worker.
    """

    def __init__(self, size: int):
        # A zero-capacity pool cannot ever hand out a slot, so every transfer
        # would refuse forever -- a mechanism present and producing nothing.
        # Refused at construction rather than discovered on the first request.
        if int(size) < 1:
            raise ValueError(
                "kvmgr: staging pool capacity must be >= 1 (got %r); a "
                "zero-slot pool refuses every transfer forever" % (size,))
        size = int(size)
        self._free: list[int] = list(range(size))
        self.capacity = size      # allocated rows -- fixed for the process
        self._limit = size        # how many we hand out -- hot
        self.high_water = 0

    @property
    def size(self) -> int:
        """The LIVE size. Named `size` because callers and the snapshot already
        used it, and it is the number that matters operationally."""
        return self._limit

    def set_limit(self, k: int) -> int:
        """Move the live size. Clamped to [1, capacity]; returns what it became.

        Clamped rather than refused: a knob that silently does nothing is this
        repo's signature failure, and a knob that takes the pool to zero would
        wedge both directions. The caller logs the change.
        """
        k = max(1, min(int(k), self.capacity))
        self._limit = k
        return k

    def _eligible(self) -> list[int]:
        return [s for s in self._free if s < self._limit]

    def take(self, n: int, reserve: int = 0) -> list[int] | None:
        """All-or-nothing. Kept for the tests that pin the boundary; the handler
        uses take_upto, because an all-or-nothing pool cannot serve a job larger
        than the pool."""
        elig = self._eligible()
        if n > max(0, len(elig) - max(0, reserve)):
            return None
        return self._hand_out(elig[:n])

    def take_upto(self, n: int, reserve: int = 0) -> list[int]:
        """As many as are free, up to n, never dropping free below `reserve`.

        The handler needs a PARTIAL answer: a store job can be hundreds of
        blocks (a 48K prompt is ~190 blocks in group 0 alone) against a pool of
        32, and refusing it is fatal. Returning what is available lets the job
        proceed in rounds.

        `reserve` is the floor left for the OTHER direction. It is advisory in
        the only way that matters: it can never stop a caller taking a slot when
        the pool is otherwise empty of competitors, because the floor is counted
        against FREE slots, and the other side releases as it finishes.
        """
        elig = self._eligible()
        k = min(max(0, n), max(0, len(elig) - max(0, reserve)))
        return self._hand_out(elig[:k])

    def _hand_out(self, out: list[int]) -> list[int]:
        for sl in out:
            self._free.remove(sl)
        in_use = self._limit - len(self._eligible())
        self.high_water = max(self.high_water, in_use)
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
        # `available` counts ELIGIBLE free slots, not every free row: with the
        # limit below capacity the others exist but are not handed out, and
        # reporting them as available would read as headroom that is not there.
        return {"size": self.size, "capacity": self.capacity,
                "available": len(self._eligible()),
                "rows_free": len(self._free),
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

# -- the WORKER-side report queue ----------------------------------------
#
# *** THE OTHER HALF OF THE COMPLETION MYSTERY. *** Measured 2026-10-08: the
# scheduler holds 123 of 133 jobs open with `pending_count: 2` on 116 of them --
# i.e. NEITHER rank's completion ever arrived -- while our handlers report
# `jobs_open: 0`, having finished and forgotten all of them. So the loss is
# between our `get_finished()` returning a TransferResult and the scheduler
# processing it.
#
# Upstream's chain is: our handler -> OffloadingWorker.get_finished() ->
# OffloadingConnectorWorker.get_finished(), which calls
# `self._connector_worker_meta.mark_completed(job_id)`; then
# `build_connector_worker_meta()` ships that meta to the scheduler AND RESETS
# it. If that build call stops happening, `completed_jobs` accumulates on the
# worker forever and the scheduler never learns -- which is precisely the
# pending_count=2 signature.
#
# So expose the size of that queue. A number that climbs and never falls is the
# proof; one that stays near zero means the loss is further along and this
# hypothesis is dead. Captured off the call stack for the same reason as the
# scheduler-side handle: a hook needs a boot and this needs to work now.
_LIVE_CONN_WORKER: list = []
CONN_WORKER_CLS_NAME = "OffloadingConnectorWorker"


def capture_conn_worker_from_caller(depth: int = 6) -> bool:
    """Grab the OffloadingConnectorWorker off the call stack. One-shot."""
    if _LIVE_CONN_WORKER:
        return True
    f = None
    try:
        import inspect
        f = inspect.currentframe()
        for _ in range(depth):
            if f is None:
                break
            f = f.f_back
            if f is None:
                break
            cand = f.f_locals.get("self")
            if cand is not None and type(cand).__name__ == CONN_WORKER_CLS_NAME:
                del _LIVE_CONN_WORKER[:]
                _LIVE_CONN_WORKER.append(cand)
                return True
    except Exception:
        return False
    finally:
        del f
    return False


def conn_worker_snapshot() -> dict:
    """Size of the worker's unshipped completion queue. Absent is not zero."""
    cw = _LIVE_CONN_WORKER[-1] if _LIVE_CONN_WORKER else None
    if cw is None:
        return {"available": False,
                "why": "OffloadingConnectorWorker not captured yet -- an "
                       "ABSENCE, not an empty report queue"}
    try:
        meta = getattr(cw, "_connector_worker_meta", None)
        jobs = getattr(meta, "completed_jobs", None)
        if jobs is None:
            return {"available": False,
                    "why": "upstream has no _connector_worker_meta."
                           "completed_jobs"}
        return {
            "available": True,
            # THE number: completions marked on this worker and not yet
            # shipped to the scheduler. Climbing without falling == never
            # shipped.
            "unshipped_completions": len(jobs),
            "unsubmitted_store_jobs": len(
                getattr(cw, "_unsubmitted_store_jobs", ()) or ()),
            "load_jobs_open": len(getattr(cw, "_load_jobs", {}) or {}),
        }
    except Exception as exc:
        return {"available": False, "why": "reading upstream raised: %r" % (exc,)}

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


def stall_fail_after() -> float:
    """Seconds of ZERO progress after which a stalled job is FAILED. Hot.

    0 disables it, which is the behaviour this path had until 2026-10-10: hang
    for ever. Default 240 -- comfortably inside NCCL's 600s watchdog, because
    the whole point is to die before `ProcessGroupNCCL::checkTimeout` does.
    """
    try:
        from kvmgr import flags as _f
        v = float(_f.peek("stall_fail_after_s") or 0.0)
        return v if v >= 0 else 240.0
    except Exception:
        return 240.0


def stall_action(verdict, stalled_for, fail_after):
    """"none" or "fail". PURE, so the deadline arithmetic is testable.

    *** THIS IS NOT A RECOVERY. *** Upstream cannot express a failed transfer:
    `offloading/worker.py:273` is a bare `assert transfer_result.success` under
    the comment "we currently do not support job failures". So reporting
    failure KILLS the worker. The choice this function makes is therefore not
    "die or survive", it is:

      hang    -> the engine goes deaf, NCCL's 600s watchdog fires, and the
                 worker aborts in C++ with NO PYTHON TRACEBACK. Measured
                 2026-10-10: Xid 32 at 14:58:49, the job's t0 at 14:58:47,
                 `verdict=stalled` climbing for ten minutes, then
                 `checkTimeout at ProcessGroupNCCL.cpp:692` and a dead worker
                 at 15:09:49. Eleven minutes of silence and nothing greppable.
      fail    -> the same death, immediately, with OUR line naming the job, its
                 age and the remedy directly above the assertion.

    A named death eleven minutes earlier is the entire win, and it is worth
    having because the alternative is undiagnosable. Nothing here makes a dead
    GPU survivable.
    """
    if verdict != "stalled":
        return "none"
    if not fail_after or fail_after <= 0:
        return "none"
    return "fail" if stalled_for >= fail_after else "none"


# Set by the WATCHDOG THREAD, honoured on the step thread. The watchdog must
# never touch `_jobs` itself: those dicts are mutated by `_collect` on the
# worker's own thread, and a second writer would be a data race in the one
# place that must not have one. A float assignment is atomic enough in CPython
# for a cross-thread request flag; the thread that owns the data does the work.
_STALL_FAIL_REQUEST = None


def honour_stall_fail(handler) -> int:
    """Mark this handler's outstanding jobs failed if the watchdog asked. Pure-ish.

    Called at the TOP of `_collect`, which runs on the worker's own thread and
    is the only writer of `_jobs` -- so the watchdog raises a request and this
    acts on it, rather than two threads mutating the same dicts.

    Returns how many jobs were marked, so the caller can log the effect rather
    than the intent. Never raises: a watchdog that kills the transfer path it
    was built to observe is worse than no watchdog, and this file already
    carries that lesson from a debug monkeypatch and a boot-breaking snapshot.
    """
    try:
        req = _STALL_FAIL_REQUEST
        if req is None:
            return 0
        jobs = getattr(handler, "_jobs", None) or {}
        if not jobs:
            return 0
        n = 0
        for st in list(jobs.values()):
            if not st.get("failed"):
                st["failed"] = True
                n += 1
        return n
    except Exception:
        return 0


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
            if verdict == "stalled":
                # RAISE A REQUEST, do not act. `_collect` owns `_jobs` on the
                # worker's thread; this thread writing them would be a race in
                # the one structure that cannot afford one.
                _fa = stall_fail_after()
                if stall_action(verdict, now - stalled_since, _fa) == "fail":
                    global _STALL_FAIL_REQUEST
                    if _STALL_FAIL_REQUEST != stalled_since:
                        _STALL_FAIL_REQUEST = stalled_since
                        logger.error(
                            "kvmgr: *** FAILING THE STALLED STORE JOB(S) *** "
                            "%d outstanding, oldest %.0fs, zero progress for "
                            "%.0fs, past the %.0fs deadline. This does NOT "
                            "rescue the engine -- upstream's "
                            "`assert transfer_result.success` has no failure "
                            "path -- it converts an eleven-minute silent wedge "
                            "ending in a C++ NCCL abort with no Python "
                            "traceback into an immediate death with this line "
                            "above it. If an Xid appeared just before the job "
                            "started, the GPU is gone and the copy can never "
                            "complete; check `journalctl -k`. Set "
                            "stall_fail_after_s=0 to go back to hanging.",
                            outstanding, oldest, now - stalled_since, _fa)
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
        # *** SAY WHAT THE ALLOCATION COSTS, because it is PINNED and this box
        # *** has no OOM net. Pinned pages cannot be swapped, and swap is the
        # gate that matters here -- earlyoom is absent under spark-os, so the
        # failure mode is a thrash livelock needing a hard reset, not a kill.
        # Measured geometry: 63 tensors summing to 1,039,680 B a block, so 32
        # blocks is 33 MB and 256 is 266 MB per rank. One log line beats
        # anybody doing that arithmetic from a shape dump again.
        _per_block = sum(t.page_size_bytes * spec.block_size_factor
                         for t in kv_caches.tensors)
        logger.info(
            "kvmgr: staging allocation -- %d blocks x %d tensors = %.1f MB "
            "PINNED per rank (%d B a block). This is the CEILING: the live pool "
            "size is hot via the staging_slots knob, because reallocating a "
            "buffer held by two handlers and two inner handlers on a live store "
            "path is a corruption risk, not a tuning knob.",
            staging_blocks, len(kv_caches.tensors),
            staging_blocks * _per_block / 1e6, _per_block)
        # The bounds a copy kernel would fault past. Captured once: these are
        # the numbers check_bounds compares against, and logging them means an
        # Xid investigation starts from facts instead of a shape dump.
        _gpu_rows = min((int(t.shape[0]) for t in gpu_tensors), default=0)
        _staging_rows = min((int(t.shape[0]) for t in self._staging), default=0)
        logger.info(
            "kvmgr: transfer bounds -- GPU rows %d, staging rows %d. Every "
            "block id and slot is checked against these BEFORE submit, because "
            "an index past them is an illegal access in a copy kernel and "
            "surfaces as Xid 13 (seen once on 2026-10-08), not as an error "
            "anywhere near the cause.",
            _gpu_rows, _staging_rows)
        self._pool = SlotPool(staging_blocks)
        self._writers = ThreadPoolExecutor(max_workers=write_threads,
                                           thread_name_prefix="kvmgr_disk")
        self._jobs: dict[int, dict] = {}          # OUR job id -> rounds left
        self._inner_to_job: dict[int, tuple] = {}  # inner job id -> round
        self._writing: list[tuple[int, Any, list[int]]] = []
        self._next_inner = 1
        self._saturated = False
        # *** COMPLETIONS MUST SURVIVE wait(). *** `wait()` looped over
        # `self.get_finished()` and kept only `r.job_id`, DISCARDING the
        # TransferResult -- and `get_finished` pops the job, so a result can
        # never be produced twice. Upstream's
        # `OffloadingConnectorWorker.get_finished()` is the ONLY caller of
        # `mark_completed()`, so every completion the fence happened to observe
        # was EATEN: the scheduler never heard it, pending_count stayed at
        # num_workers forever, complete_store never fired, the key was never
        # indexed, and the job stayed in `jobs_to_flush` fencing its blocks
        # permanently. Measured 2026-10-08: 5 store jobs, 1 report delivered;
        # 7% of keys indexed on a 41K prompt against ~90% on a 3K one, because
        # the fence waits far more under load. It also explains the +13,312
        # tokens-per-pass convergence, `jobs_open: 0` on the worker against 125
        # open on the scheduler, and the 9,300 fenced blocks that deferred a
        # request until NCCL's watchdog killed the worker.
        #
        # So COLLECTION and REPORTING are separate now. `_collect()` advances
        # the machinery and parks finished results here; `get_finished()` is the
        # only thing that drains this list, because it is the only path that
        # reaches upstream. `wait()` reads it without draining.
        self._ready: list = []
        self.stats = {"submitted": 0, "refused_pairing": 0, "refused_bounds": 0,
                      "files": 0, "bytes": 0, "errors": 0,
                      # *** INITIALISED IN THE HANDLER THAT INCREMENTS IT, and
                      # my first attempt put it in the LOAD handler, which never
                      # touches it. So the STORE reported `ABSENT` -- which reads
                      # as "no such counter" when the truth was "zero skips" --
                      # while the LOAD reported a permanent measured 0 for an
                      # event it cannot have, which is worse, because a zero
                      # looks like evidence. The two handlers have near-identical
                      # stats dicts and my `count(...) == 1` guard passed
                      # because exactly one matched: the wrong one. Scope such a
                      # search to the CLASS, never the file.
                      "skipped_before_copy": 0,
                      # `delete_errors` and `deleted` were INCREMENTED and never
                      # declared, so both read ABSENT until the first eviction --
                      # found by the per-class stats audit the moment it was
                      # written, which is this project's pattern for audits.
                      "delete_errors": 0, "deleted": 0,
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
                      "wait_rounds": 0, "saturations": 0,
                      # *** THE NEW BOUND MUST BE VISIBLE WHEN IT BINDS. ***
                      # The per-round budget made `saturations` nearly
                      # unreachable: with inflight_max=1 the pump returns before
                      # it can ever drain the pool, so the one existing signal
                      # for "the store is behind" stopped being able to fire.
                      # Replacing a visible limit with an invisible one is this
                      # project's "refusal nobody can see", so `capped` counts
                      # the rounds the budget actually SHRANK. It is not
                      # incremented by the inflight gate, which fires on every
                      # step while a round is open and would say nothing; that
                      # state is already readable as inflight/inflight_max.
                      "capped": 0}
        logger.info(
            "kvmgr: disk store handler -- staging %d blocks x %d tensors "
            "(pinned=%s), per-group file sizes %s, sum %d B (the per-rank page; "
            "the tiering tier wrote %d B per group instead)",
            staging_blocks, len(self._staging), pin, self._sizes,
            sum(self._sizes), spec.whole_page_bytes,
        )
        _LIVE_HANDLERS.append(self)
        _start_watchdog()

    def _bounds(self) -> tuple[int, int]:
        """(gpu_rows, staging_rows), derived lazily. REBIND-SAFE.

        *** A RELOAD NEVER RUNS __init__ -- third time in this file. ***
        `_gcounts` crashed the engine that way and `_ready` would have, so
        these are DERIVED from `_staging` and the gpu tensors, which every
        already-constructed handler has, rather than read from attributes only
        a new __init__ would set. Cached on the instance after the first call.
        """
        got = getattr(self, "_bounds_cache", None)
        if got is not None:
            return got

        def _rows(seq):
            """Row count, or None when it cannot be determined. NEVER 0 -- a 0
            bound makes every index illegal, which is how this guard crashed
            the engine instead of preventing a crash."""
            try:
                n = min(int(t.shape[0]) for t in seq)
                return n if n > 0 else None
            except Exception:
                return None

        # A rebound handler has only what its OLD __init__ set, so try every
        # route to the tensors before giving up: our own attribute, the inner
        # handler's, and the store handler we share staging with.
        gt = (getattr(self, "_gpu_tensors", None)
              or getattr(getattr(self, "_inner", None), "gpu_tensors", None))
        if not gt:
            for h in list(_LIVE_HANDLERS):
                cand = getattr(h, "_gpu_tensors", None)
                if cand:
                    gt = cand
                    break
        got = (_rows(gt or []), _rows(getattr(self, "_staging", None) or []))
        if got[0] is None or got[1] is None:
            logger.warning(
                "kvmgr: transfer bounds UNKNOWN (gpu_rows=%s staging_rows=%s) "
                "-- the range check is SKIPPED for this handler, which is the "
                "safe direction: a bound of 0 would refuse every valid index "
                "and that is how this guard once killed the engine. Shape and "
                "pairing checks still run.", got[0], got[1])
        else:
            self._bounds_cache = got      # only cache a usable answer
        return got

    def _ledger_snapshot(self) -> dict:
        led = getattr(self, "_ledger", None)
        if led is None:
            return {"transfers": 0, "why": "nothing transferred yet -- an "
                                           "ABSENCE, not an index range of 0"}
        return {"transfers": led["n"],
                "block_id_range": [led["bid_min"], led["bid_max"]],
                "slot_range": [led["slot_min"], led["slot_max"]],
                "recent": led["recent"][-8:]}

    def _ledger_note(self, kind: str, group: int, bids, slots) -> None:
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
            led = getattr(self, "_ledger", None)
            if led is None:
                led = {"n": 0, "bid_min": None, "bid_max": None,
                       "slot_min": None, "slot_max": None, "recent": []}
                self._ledger = led
            lo, hi = (min(bids), max(bids)) if bids else (None, None)
            slo, shi = (min(slots), max(slots)) if slots else (None, None)
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
                    self._bounds()[0], led["slot_min"], led["slot_max"],
                    self._bounds()[1])
        except Exception:
            return

    def _apply_deletes(self, spec) -> None:
        """Unlink files the scheduler evicted. THIS rank's copies only.

        *** THE ONLY PLACE FILES ARE EVER REMOVED, and it must never be fatal.
        *** The budget decision is the scheduler's (it holds the index and the
        recency); this is the hands. Each rank deletes its OWN node's file, and
        the same key is deleted on both ranks because the spec reaches both --
        which is the point, since one rank dropping a key the other kept would
        make a later restore load half the KV with nothing crashing.
        
        A failed unlink is logged and ignored: a file that will not delete
        costs disk, while raising here would fail the store job and ultimately
        the request. Already-gone is success, not an error -- that is the
        common case on a re-eviction.
        """
        to_delete = getattr(spec, "to_delete", None)
        if not to_delete:
            return
        gone = errs = 0
        for key in to_delete:
            try:
                path = self._mapper.get_file_name(key)
            except Exception:
                errs += 1
                continue
            try:
                os.unlink(path)
                gone += 1
            except FileNotFoundError:
                gone += 1          # already gone is the goal, not a failure
            except Exception:
                errs += 1
        self.stats["deleted"] = self.stats.get("deleted", 0) + gone
        if errs:
            self.stats["delete_errors"] = self.stats.get(
                "delete_errors", 0) + errs
        # *** THROTTLED FOR THE SAME REASON AS THE SCHEDULER'S LINE. *** At
        # steady state every store round unlinks ~90 files, so an unthrottled
        # line is one per round for ever -- a routine event logged as though it
        # were notable. An ERROR is never throttled, because that is the thing
        # somebody needs to see the moment it happens.
        now = _now()
        first = not getattr(self, "_del_logged", False)
        due = (now - getattr(self, "_del_log_at", 0.0)) >= 60.0
        if errs:
            logger.error(
                "kvmgr: budget eviction -- %d file(s) FAILED to unlink on this "
                "rank (%d succeeded). A file that will not delete costs disk; "
                "raising here would fail the store and then the request, so it "
                "is counted and ignored.", errs, gone)
        if gone and (first or due):
            self._del_logged = True
            self._del_log_at = now
            logger.info(
                "kvmgr: budget eviction -- unlinked %d file(s) on this rank "
                "just now, %d in total. The scheduler chose them; this rank "
                "deletes only its own slice, and the same keys go on every "
                "rank because the spec reaches them all.",
                gone, self.stats.get("deleted", 0))

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
        # Deletes ride the store spec -- see DiskPresenceManager._enforce_budget.
        try:
            self._apply_deletes(spec[1] if isinstance(spec, tuple) else spec)
        except Exception:
            logger.exception("kvmgr: applying budget deletes raised (ignored)")
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
                # *** NO RESERVE ANY MORE, and that is a consequence of the
                # split, not an oversight. *** The read floor existed to stop a
                # store draining the shared pool and starving a restore (the
                # 574.9s-against-3.3s case). With separate allocations this pool
                # has exactly one user, so reserving part of it for the other
                # direction would simply shrink the store for nothing.
                # `staging_write_floor` / `staging_read_floor` are superseded;
                # flags.py says so where they are declared.
                # *** BOUND WHAT ONE SCHEDULER STEP CARRIES. *** Two units
                # on purpose, mirroring the load: the block cap is the
                # granularity, the OP budget is what upstream's single driver
                # call actually contains (blocks x refs-for-this-group), and the
                # inflight cap is the one that bounds the STEP rather than the
                # round. Without the third, `while` submits eight 32-block
                # rounds and the step costs exactly what it did before.
                if len(self._inner_to_job) >= _store_inflight_max():
                    return
                _cap = min(_store_batch_max(),
                           ops_budget(self._refs, g, len(bids),
                                      _store_ops_max()))
                if _cap < len(bids):
                    self.stats["capped"] = self.stats.get("capped", 0) + 1
                slots = self._pool.take_upto(min(len(bids), _cap))
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
                _gr, _sr = self._bounds()
                why = check_bounds(bids[:n], slots, _gr, _sr, gs,
                                   "GPU->staging")
                if why is not None:
                    # .get, not += : a REBOUND handler's stats dict was built
                    # by the OLD __init__ and has no such key -- the same
                    # reload trap as _gcounts and _ready, now in a dict.
                    self.stats["refused_bounds"] = self.stats.get(
                        "refused_bounds", 0) + 1
                    ledger_dump(self, "store REFUSED on bounds: " + str(why))
                    logger.error(
                        "kvmgr: REFUSING a store round -- %s. job=%s group=%s "
                        "n=%d. Refused rather than submitted: handing these to "
                        "a copy kernel is an illegal access, which kills the "
                        "worker with an Xid and a traceback nowhere near here.",
                        why, job_id, g, n)
                    self._pool.give(slots)
                    st["failed"] = True
                    return
                # *** IF THE FILE ALREADY EXISTS, DO NOT COPY IT TO STAGING.
                # *** AND DO THIS *AFTER* EVERY GUARD, NOT BEFORE. ***
                # `_write_batch` has always skipped an existing path, but it
                # runs in `_collect` AFTER the GPU->staging copy completed, so
                # an already-stored block cost a full 63-tensor GPU->pinned
                # copy that was then thrown away. This project's own record has
                # the shape of that waste: a boot where "the store path wrote
                # nothing, because every block it was asked to store was
                # already on disk", reported as `stored: 92` against ZERO bytes
                # written -- ninety-two pointless copies.
                #
                # It is not just waste, it is the Xid 13 hazard. The LOAD reads
                # this same shared pinned pool under
                # CU_MEMCPY_SRC_ACCESS_ORDER_ANY, whose precondition is that no
                # concurrent GPU stream writes the source -- and these copies
                # are exactly such a write. They are commonest during a
                # restore, because a restore is by definition a scenario where
                # the files exist, which is when the load is reading the pool.
                #
                # *** THE ORDER MATTERS AND MY FIRST VERSION GOT IT BACKWARDS.
                # Skipping before check_bounds meant a present key was never
                # bounds-checked -- and test_phase1b_disk caught it only
                # because a SECOND assertion counted the refusal: the first,
                # "nothing was submitted to the copy kernel", still PASSED,
                # vacuously, because nothing was submitted for the wrong
                # reason. The bounds guard exists to DETECT corruption, not
                # only to prevent a submit, so it must run on every key.
                # Slots are cheap bookkeeping; the COPY is what costs. So take
                # slots, run every guard, and only then drop the keys whose
                # bytes are already on disk, returning their slots.
                from kvmgr import flags as _fl      # _f is local to staging_knobs
                _sk = bkeys[:n]
                _sb = bids[:n]
                _ss = slots
                if _fl.peek("disk_skip_present_before_copy"):
                    _kk, _kb, _ks2, _drop = [], [], [], []
                    for _k, _b, _sl in zip(_sk, _sb, _ss):
                        try:
                            _ex = os.path.exists(self._mapper.get_file_name(_k))
                        except Exception:
                            _ex = False               # cannot tell -> copy it
                        if _ex:
                            _drop.append(_sl)
                            continue
                        _kk.append(_k); _kb.append(_b); _ks2.append(_sl)
                    if _drop:
                        # COUNTED, because `_write_batch`'s skip is a bare
                        # `continue` with no counter at all, so the waste it
                        # stood for was invisible for the life of the path.
                        self.stats["skipped_before_copy"] = self.stats.get(
                            "skipped_before_copy", 0) + len(_drop)
                        self._pool.give(_drop)
                    if not _kk:
                        # Nothing to copy. Advance the batch WITHOUT touching
                        # `pending`: a job every one of whose blocks is already
                        # on disk must still complete, and it completes on
                        # `pending == 0 and not batches`.
                        if n == len(bids):
                            st["batches"].pop(0)
                        else:
                            st["batches"][0] = [g, bkeys[n:], bids[n:]]
                        continue
                    _sk, _sb, _ss = _kk, _kb, _ks2
                    gs = [0] * ngroups
                    gs[g] = len(_sb)

                self._ledger_note("store", g, _sb, _ss)
                try:
                    ok = self._inner.transfer_async(
                        inner,
                        (GPULoadStoreSpec(_sb, group_sizes=gs,
                                          block_indices=[0] * ngroups),
                         CPULoadStoreSpec(_ss)))
                except Exception:
                    logger.exception("kvmgr: inner GPU->staging submit raised")
                    ok = False
                if not ok:
                    self._pool.give(_ss)
                    st["failed"] = True
                    return
                self._inner_to_job[inner] = (job_id, _sk, _ss, g)
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

    def reconcile_pool(self) -> None:
        """Apply the hot `staging_slots` knob. Called every scheduler step.

        *** NOT FROM THE TAKE PATH, and the difference is user-visible. ***
        It was there first, and a resize then only happened when a transfer
        round actually ran -- so setting the knob on an engine with nothing to
        store changed nothing, and the pool read its old size. Measured live:
        256 -> 32 -> 128 applied fine while a third change to 256 did not,
        because by then every key was already present and no round ran. That is
        this repo's documented "a hot knob read on an IDLE engine shows the OLD
        derived figure", reappearing in a new place.

        `_collect()` is called once per step through `get_finished()`, which is
        the same edge `gpu_cache_epoch` uses, so the knob now lands on the next
        step rather than the next transfer.
        """
        try:
            want, _wf, _rf = staging_knobs(self._pool.capacity)
            if want != self._pool.size:
                logger.info(
                    "kvmgr: staging pool live size %d -> %d of %d allocated "
                    "(hot; the allocation itself is a boot ceiling)",
                    self._pool.size, want, self._pool.capacity)
                self._pool.set_limit(want)
        except Exception:
            # A resize that cannot be computed must never stop a transfer.
            pass

    def _collect(self) -> None:
        self.reconcile_pool()
        from vllm.v1.kv_offload.worker.worker import TransferResult

        # *** THE FENCE FIX, and it is a LEGIBILITY fix rather than a recovery.
        # A job whose copy will never complete (a dead GPU after an Xid is the
        # measured case) otherwise hangs until NCCL's 600s watchdog aborts the
        # worker in C++ with no Python traceback. Failing it here reports
        # success=False, which upstream asserts on -- the same death, minutes
        # earlier, with the watchdog's reason directly above it.
        _nf = honour_stall_fail(self)
        if _nf:
            ledger_dump(self, "watchdog failed %d stalled job(s)" % _nf)
            logger.error(
                "kvmgr: marked %d STORE job(s) failed on the watchdog's "
                "request. upstream's assert will now fire on the next "
                "get_finished, which is the intended outcome: an attributed "
                "death instead of a silent wedge.", _nf)

        if not _LIVE_CONN_WORKER:
            capture_conn_worker_from_caller()

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
                # *** THE RING, AT THE ONE MOMENT IT IS WORTH HAVING. *** The
                # GPU has reported a failed copy; an Xid usually follows or has
                # just preceded it. Dumping here is what makes the fault
                # attributable to a block-id range instead of to "something in
                # the transfer path".
                ledger_dump(self, "store transfer reported FAILURE")
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
        out = self._ready_buf()
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
                    # int(): ANYTHING HANDED TO A FRAMEWORK TO SERIALISE MUST
                    # BE A BUILTIN. A numpy int here kills the engine core's
                    # output socket thread and the clients hang for ever.
                    job_id=int(job_id), success=True,
                    transfer_size=int(st["bytes"]),
                    transfer_time=_elapsed(st.get("t0")),
                    transfer_type=("GPU", self._spec.disk_medium)))
        return None

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
            # *** _collect, NEVER get_finished. *** Calling get_finished here
            # DRAINED the results and threw them away, so the scheduler never
            # learned the store had completed. The fence only needs to know a
            # job is DONE; reporting it is upstream's call to make.
            self._collect()
            for r in self._ready_buf():
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

    def _ready_buf(self) -> list:
        """The parked-completions buffer, created on demand.

        *** A RELOAD NEVER RUNS __init__. *** `rebind` re-points a live object
        with `obj.__class__ = new`, so an attribute introduced alongside a
        method is absent on exactly the objects that matter. `_gcounts` crashed
        the engine this way earlier today; `_ready` sits on the STORE path, so
        the same mistake here would be worse. Lazily initialised on the
        instance -- never a class attribute, which every instance would share,
        aliasing two handlers' completions into one list.
        """
        buf = getattr(self, "_ready", None)
        if buf is None:
            buf = []
            self._ready = buf
        return buf

    def get_finished(self) -> list:
        """Report completions upstream. *** THE ONLY DRAIN OF self._ready. ***

        Upstream's `OffloadingConnectorWorker.get_finished()` is the only caller
        of `mark_completed()`, so a TransferResult that leaves this method and
        is not handed to it is LOST FOREVER -- the job is already popped and can
        never be reported again. That is precisely what `wait()` used to do.
        Keeping the drain in one method is what makes that checkable by reading
        one function instead of auditing every caller.
        """
        self._collect()
        out = self._ready_buf()
        self._ready = []
        return out

    def snapshot(self) -> dict:
        return dict(self.stats, staging=self._pool.snapshot(),
                    group_file_sizes=self._sizes,
                    jobs_open=len(self._jobs), writes_open=len(self._writing),
                    # the bound, readable per rank -- a cap nobody can see is a
                    # cap nobody can confirm took effect
                    inflight=len(self._inner_to_job),
                    inflight_max=_store_inflight_max(),
                    batch_max=_store_batch_max(),
                    ops_max=_store_ops_max(),
                    refs_per_group=[len(r) for r in (self._refs or [])],
                    gpu_rows=self._bounds()[0],
                    staging_rows=self._bounds()[1],
                    ledger=self._ledger_snapshot(),
                    conn_worker=conn_worker_snapshot())


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

        # *** THE LOAD GETS ITS OWN PINNED STAGING, AND SHARING IT WAS THE
        # *** ROOT CAUSE OF ALL FIVE Xid 13 EVENTS. ***
        #
        # The comment that used to be here said: "SHARE the store handler's
        # staging and pool. Two pools would double the pinned memory for no
        # reason, and a load and a store never need the same slot at once --
        # the pool is what serialises them."
        #
        # Every clause of that is true except the conclusion. The pool does
        # serialise OWNERSHIP of a slot. What it cannot do is order the
        # DRIVER's reads of one slot against another stream's writes to a
        # different slot in the SAME ALLOCATION -- and upstream's load path
        # gives exactly that ordering up. `cpu/gpu_worker.py`'s
        # `transfer_async` sets
        #
        #     is_src_access_order_any = not self.gpu_to_cpu
        #
        # and says why: "CPU->GPU reads from host pinned memory, which is never
        # written by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY
        # is safe and lets the driver pipeline source reads."
        #
        # Our load is CPU->GPU (`gpu_to_cpu=False`), so it inherits ANY-order
        # access -- while our store writes that very buffer FROM A GPU STREAM.
        # The precondition upstream states is therefore false for us, and it is
        # false structurally, not as a race in our bookkeeping. Measured at the
        # 2026-10-10 07:41 fault: `store path busy -- 1 job(s) outstanding` on
        # BOTH ranks while the load read the shared pool. The store direction
        # keeps STREAM ordering (`gpu_to_cpu=True`) and has never faulted this
        # way, which is the asymmetry the evidence shows.
        #
        # So the load allocates its OWN rows. A separate allocation is what
        # restores upstream's precondition: no GPU stream writes it, ever.
        # The memory is NOT doubled -- `staging_load_blocks` defaults to a
        # quarter of the store's, and the honest way to hold total pinned
        # memory flat is to lower `staging_blocks` by the same amount, which is
        # a per-node KV_TRANSFER_CONFIG decision and a restart. The allocation
        # is a boot ceiling either way; the LIVE size stays hot, via
        # `staging_load_slots`, exactly as the store's does via `staging_slots`.
        # The SAME import path the store uses -- `vllm.utils.platform_utils`,
        # not `torch_utils`. My first version guessed the latter, which is an
        # ImportError at boot: a change that cannot even load is worse than the
        # bug it fixes, and this file already carries that lesson from a
        # diagnostic that broke a boot.
        import torch
        from vllm.utils.platform_utils import is_pin_memory_available

        pin = is_pin_memory_available()
        self._staging = []
        for t in kv_caches.tensors:
            self._staging.append(torch.empty(
                (staging_blocks, t.page_size_bytes * spec.block_size_factor),
                dtype=torch.int8, device="cpu", pin_memory=pin))
        self._pool = SlotPool(staging_blocks)
        _per_block = sum(t.page_size_bytes * spec.block_size_factor
                         for t in kv_caches.tensors)
        logger.info(
            "kvmgr: LOAD staging allocation -- %d blocks x %d tensors = %.1f MB "
            "(pinned=%s), SEPARATE from the store's. Sharing one buffer is what "
            "faulted the worker with Xid 13: the load reads it under "
            "CU_MEMCPY_SRC_ACCESS_ORDER_ANY, whose precondition is that no "
            "concurrent GPU stream writes the source, and the store's copy is "
            "exactly such a write.",
            staging_blocks, len(kv_caches.tensors),
            staging_blocks * _per_block / 1e6, pin)
        # Kept so _bounds() can find the row count without reaching through
        # the inner handler, whose attribute name is upstream's to change.
        self._gpu_tensors = getattr(store, "_gpu_tensors", None) or []
        self._inner = SingleDirectionOffloadingHandler(
            gpu_tensors=store._inner.gpu_tensors
            if hasattr(store._inner, "gpu_tensors") else store._gpu_tensors,
            # Our own rows, so no GPU stream ever writes this buffer.
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
        self._ready: list = []
        self.stats = {"submitted": 0, "refused_pairing": 0, "files": 0,
                      "refused_bounds": 0,
                      "bytes": 0, "errors": 0,
                      "wait_calls": 0, "wait_ours": 0, "wait_blocked_ms": 0,
                      "wait_max_pending": 0, "wait_rounds": 0,
                      "saturations": 0}
        # REPORT THE FACT, NOT THE ASK. This said "(restore is live)"
        # unconditionally, from __init__, with no gate on the flag -- so a boot
        # whose flags file had disk_restore=False still announced a live
        # restore, and I trusted it over /kv, which was correct. Installing the
        # handler and being permitted to load are different things: lookup()
        # refuses while the flag is off, so the handler can exist and restore
        # still be impossible.
        try:
            from kvmgr import flags as _f0
            _live = bool(_f0.peek("disk_restore"))
        except Exception:
            _live = None
        logger.info(
            "kvmgr: disk LOAD handler installed -- disk_restore=%s (the handler "
            "exists either way; lookup() refuses while it is off)",
            "UNREADABLE" if _live is None else _live)
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

    def _bounds(self) -> tuple[int, int]:
        """(gpu_rows, staging_rows), derived lazily. REBIND-SAFE.

        *** A RELOAD NEVER RUNS __init__ -- third time in this file. ***
        `_gcounts` crashed the engine that way and `_ready` would have, so
        these are DERIVED from `_staging` and the gpu tensors, which every
        already-constructed handler has, rather than read from attributes only
        a new __init__ would set. Cached on the instance after the first call.
        """
        got = getattr(self, "_bounds_cache", None)
        if got is not None:
            return got
        def _rows(seq):
            try:
                return min(int(t.shape[0]) for t in seq)
            except Exception:
                return 0
        gt = (getattr(self, "_gpu_tensors", None)
              or getattr(getattr(self, "_inner", None), "gpu_tensors", None)
              or [])
        got = (_rows(gt), _rows(getattr(self, "_staging", None) or []))
        self._bounds_cache = got
        return got

    def _ledger_snapshot(self) -> dict:
        led = getattr(self, "_ledger", None)
        if led is None:
            return {"transfers": 0, "why": "nothing transferred yet -- an "
                                           "ABSENCE, not an index range of 0"}
        return {"transfers": led["n"],
                "block_id_range": [led["bid_min"], led["bid_max"]],
                "slot_range": [led["slot_min"], led["slot_max"]],
                "recent": led["recent"][-8:]}

    def _ledger_note(self, kind: str, group: int, bids, slots) -> None:
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
            led = getattr(self, "_ledger", None)
            if led is None:
                led = {"n": 0, "bid_min": None, "bid_max": None,
                       "slot_min": None, "slot_max": None, "recent": []}
                self._ledger = led
            lo, hi = (min(bids), max(bids)) if bids else (None, None)
            slo, shi = (min(slots), max(slots)) if slots else (None, None)
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
                    self._bounds()[0], led["slot_min"], led["slot_max"],
                    self._bounds()[1])
        except Exception:
            return

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
                # *** BOUND THE ROUNDS IN FLIGHT, not just each round's size.
                # `_inner_to_job` holds every round whose staging->GPU copy is
                # outstanding; upstream chains them on one stream with an event
                # each and bounds nothing. The 700K test respected the 192
                # per-round cap and still died at 861 keys, so the count is the
                # variable the cap cannot reach. Returning here is safe and is
                # not a refusal: `_pump` is called again from `_collect` on
                # every scheduler step, and `kv_connector_no_forward` means
                # those steps keep coming even with nothing scheduled -- so the
                # work resumes as soon as a round completes.
                g, bkeys, bids = st["batches"][0]
                # No reserve: see the store's _pump. This pool is the LOAD's
                # alone now, so a write floor inside it would starve the load to
                # protect a direction that cannot reach it.
                # *** BOUND THE BATCH, because ~256 blocks in ONE
                # *** transfer_async FAULTS THE WORKER with Xid 13.
                # Measured 2026-10-10 by walking the hot `staging_load_slots`
                # knob up against a 327-key restore: 96, 128, 160, 192 and 224
                # are all clean at 1-2s, and 256 faults on the first attempt
                # (engine dead, worker or head Xid 13). Five clean arms then a
                # hard failure at exactly 256 looks like a 256-entry limit
                # somewhere in upstream's batch copy rather than gradual
                # resource exhaustion.
                #
                # The working configuration before this was `staging_load_blocks
                # = 64`, which was an arbitrary QUARTER of the store's pool and
                # happened to be under the limit. That is not a safety property:
                # a pool resize would have silently restored the fault. The cap
                # is explicit and independent of the pool now, so the pool can
                # be sized for throughput and this for correctness.
                # TWO bounds, and they are different units on purpose: the
                # block cap is what was measured to fault, the OP budget is
                # what upstream's single driver call actually contains
                # (blocks x refs-for-this-group). A block cap alone means a
                # different descriptor count per group, which is why one
                # number could never fit all five.
                _cap = min(_load_batch_max(),
                           load_ops_budget(self._refs, g, len(bids),
                                           _load_ops_max()))
                if len(self._inner_to_job) >= _load_inflight_max():
                    return
                slots = self._pool.take_upto(min(len(bids), _cap))
                if not slots:
                    # COUNT IT. `saturations` was declared here and never
                    # incremented, so it read as a measured 0 for an event that
                    # could not be recorded -- worse than absent, because a zero
                    # looks like evidence. A starved LOAD is the 574.9s-vs-3.3s
                    # case, so it is the one worth seeing.
                    self.stats["saturations"] = self.stats.get("saturations", 0) + 1
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

    def reconcile_pool(self) -> None:
        """Apply the hot `staging_load_slots` knob to the LOAD's own pool.

        The load used to share the store's pool, so the store's reconcile
        governed both. With separate allocations it governs only its own, and a
        load pool with no reconcile would be a knob that silently did nothing --
        which is this project's most repeated defect. Called once per scheduler
        step from `_collect`, the same edge the store uses.
        """
        try:
            from kvmgr import flags as _fl
            want = staging_plan(self._pool.capacity,
                                _fl.peek("staging_load_slots"), 0, 0)[0]
            if want != self._pool.size:
                logger.info(
                    "kvmgr: LOAD staging pool live size %d -> %d of %d "
                    "allocated (hot; the allocation itself is a boot ceiling)",
                    self._pool.size, want, self._pool.capacity)
                self._pool.set_limit(want)
        except Exception:
            # A resize that cannot be computed must never stop a restore.
            pass

    def _collect(self) -> None:
        from vllm.v1.kv_offload.base import GPULoadStoreSpec
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
        from vllm.v1.kv_offload.worker.worker import TransferResult

        self.reconcile_pool()

        # Same deadline on the read side. A load that never completes fences
        # nothing, but it holds staging slots for ever and the request behind
        # it never returns -- so it is a hang with a different shape and the
        # same remedy.
        _nf = honour_stall_fail(self)
        if _nf:
            ledger_dump(self, "watchdog failed %d stalled job(s)" % _nf)
            logger.error(
                "kvmgr: marked %d LOAD job(s) failed on the watchdog's "
                "request -- see the STORE note; the assert is upstream's.", _nf)

        if not _LIVE_CONN_WORKER:
            capture_conn_worker_from_caller()

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
            _gr, _sr = self._bounds()
            why = check_bounds(bids, slots, _gr, _sr, gs, "staging->GPU")
            if why is not None:
                self.stats["refused_bounds"] = self.stats.get(
                    "refused_bounds", 0) + 1
                ledger_dump(self, "load REFUSED on bounds: " + str(why))
                logger.error(
                    "kvmgr: REFUSING a restore round -- %s. job=%s group=%s. "
                    "A bad index here would write off the end of the KV tensor, "
                    "which is worse than a missed restore.", why, job_id, g)
                self._pool.give(slots)
                if st is not None:
                    st["failed"] = True
                    st["pending"] -= 1
                continue
            self._ledger_note("load", g, bids, slots)
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
                ledger_dump(self, "load transfer reported FAILURE")
                st["failed"] = True
            st["pending"] -= 1

        self._pump()

        out = self._ready_buf()
        for job_id in [j for j, st in self._jobs.items()
                       if st["pending"] == 0 and (not st["batches"] or st["failed"])]:
            st = self._jobs.pop(job_id)
            # The time matters more on this side than on the store's: see
            # DiskStoreHandler.get_finished. A restore with no series is a
            # restore nobody can prove happened.
            out.append(TransferResult(
                job_id=int(job_id), success=not st["failed"],
                # int() for the same reason as the store side.
                transfer_size=(int(st["bytes"]) if not st["failed"] else None),
                transfer_time=_elapsed(st.get("t0")),
                transfer_type=(self._spec.disk_medium, "GPU")))
        return None

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

    def _ready_buf(self) -> list:
        """The parked-completions buffer, created on demand.

        *** A RELOAD NEVER RUNS __init__. *** `rebind` re-points a live object
        with `obj.__class__ = new`, so an attribute introduced alongside a
        method is absent on exactly the objects that matter. `_gcounts` crashed
        the engine this way earlier today; `_ready` sits on the STORE path, so
        the same mistake here would be worse. Lazily initialised on the
        instance -- never a class attribute, which every instance would share,
        aliasing two handlers' completions into one list.
        """
        buf = getattr(self, "_ready", None)
        if buf is None:
            buf = []
            self._ready = buf
        return buf

    def get_finished(self) -> list:
        """Report completions upstream. *** THE ONLY DRAIN OF self._ready. ***

        Upstream's `OffloadingConnectorWorker.get_finished()` is the only caller
        of `mark_completed()`, so a TransferResult that leaves this method and
        is not handed to it is LOST FOREVER -- the job is already popped and can
        never be reported again. That is precisely what `wait()` used to do.
        Keeping the drain in one method is what makes that checkable by reading
        one function instead of auditing every caller.
        """
        self._collect()
        out = self._ready_buf()
        self._ready = []
        return out

    def snapshot(self) -> dict:
        return dict(self.stats, jobs_open=len(self._jobs),
                    reads_open=len(self._reading),
                    # *** THE LOAD'S POOL WAS UNOBSERVABLE. *** The store's
                    # snapshot has carried `staging` all along and the load's
                    # did not, so the batch-size bisection printed
                    # "size=? cap=?" on every arm and could not verify that the
                    # hot `staging_load_slots` knob had taken effect at all.
                    # Third instance of the same shape in this file: two
                    # near-identical handlers, one of them patched.
                    staging=self._pool.snapshot(),
                    # Outstanding staging->GPU rounds, and the cap they are
                    # bounded by. Both exposed because a knob whose effect
                    # cannot be read back is a knob nobody can verify -- the
                    # load's pool was unobservable for exactly one bisection
                    # and every arm printed "size=?".
                    inflight=len(self._inner_to_job),
                    inflight_max=_load_inflight_max(),
                    # *** THE GEOMETRY THAT SETS THE REAL LIMIT, and it was
                    # invisible. *** One submit contains blocks x refs[group]
                    # descriptors, so without these the op budget cannot be
                    # checked against anything and the engine log TRUNCATES the
                    # fingerprint that would otherwise carry them.
                    refs_per_group=[len(r) for r in (self._refs or [])],
                    ops_max=_load_ops_max(),
                    # *** THESE WERE MISSING AND I MISREAD THEIR ABSENCE. ***
                    # Only the store handler reported its bounds, so the load
                    # handler showed gpu_rows=None -- and I read "the key is
                    # not here" as "the bound is unknown", which are different
                    # facts that want opposite responses. The load path is
                    # exactly where the 0-bound crash happened, so it is the
                    # one that most needs to be readable.
                    gpu_rows=self._bounds()[0],
                    staging_rows=self._bounds()[1],
                    ledger=self._ledger_snapshot())
