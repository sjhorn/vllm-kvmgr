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

import os
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
    replacing. Exhaustion is therefore normal and must be survivable -- the
    connector keeps unsubmitted store jobs and retries, so `transfer_async`
    returning False is an expected answer and not an error.
    """

    def __init__(self, size: int):
        if size <= 0:
            raise ValueError("kvmgr: staging pool must hold at least one block")
        self._free: list[int] = list(range(size))
        self.size = size
        self.high_water = 0

    def take(self, n: int) -> list[int] | None:
        if n > len(self._free):
            return None
        out = [self._free.pop() for _ in range(n)]
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


def write_chunks(path: str, plan: "Sequence[Chunk]", read_row: Any,
                 tmp_suffix: str = ".part") -> int:
    """Write one file ATOMICALLY: temp file, then rename.

    *** ATOMICITY IS NOT HYGIENE HERE, IT IS THE PRESENCE CHECK. *** A reader
    decides a block is on disk with os.path.exists(). A crash or a full disk
    midway through a plain write leaves a SHORT file at the real path, which
    exists() reports as a hit and which would then restore as truncated KV --
    plausible tensors, nothing crashing. Writing to `<path>.part` and renaming
    means a path either does not exist or is complete.

    `read_row(tensor_idx, slot, nbytes) -> bytes-like` is injected so this is
    testable with no torch.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + tmp_suffix
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

        self._inner = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors, cpu_tensors=self._staging,
            block_size_factor=spec.block_size_factor,
            kv_cache_groups_data_refs=self._refs, gpu_to_cpu=True,
        )
        self._pool = SlotPool(staging_blocks)
        self._writers = ThreadPoolExecutor(max_workers=write_threads,
                                           thread_name_prefix="kvmgr_disk")
        self._inflight: dict[int, dict] = {}   # inner job id -> state
        self._writing: list[tuple[int, Any, list[int], int]] = []
        self._next_inner = 1
        self.stats = {"submitted": 0, "refused_full": 0, "refused_pairing": 0,
                      "files": 0, "bytes": 0, "errors": 0}
        logger.info(
            "kvmgr: disk store handler -- staging %d blocks x %d tensors "
            "(pinned=%s), per-group file sizes %s, sum %d B (the per-rank page; "
            "the tiering tier wrote %d B per group instead)",
            staging_blocks, len(self._staging), pin, self._sizes,
            sum(self._sizes), spec.whole_page_bytes,
        )

    def transfer_async(self, job_id: int, spec) -> bool:
        from vllm.v1.kv_offload.base import get_offload_group_idx
        from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

        gpu_spec, disk_spec = spec
        keys = disk_spec.keys
        block_ids = list(gpu_spec.block_ids)

        # REFUSE RATHER THAN WRITE. See check_pairing's docstring: a desync here
        # produces files of the right size holding another block's KV.
        why = check_pairing(keys, block_ids, gpu_spec.group_sizes,
                            get_offload_group_idx)
        if why is not None:
            self.stats["refused_pairing"] += 1
            logger.error("kvmgr: REFUSING store job %d -- %s", job_id, why)
            return False

        slots = self._pool.take(len(block_ids))
        if slots is None:
            # Expected, not an error: staging is small by design and the
            # connector retries unsubmitted store jobs.
            self.stats["refused_full"] += 1
            return False

        inner = self._next_inner
        self._next_inner += 1
        if not self._inner.transfer_async(inner, (gpu_spec,
                                                  CPULoadStoreSpec(slots))):
            self._pool.give(slots)
            return False
        self._inflight[inner] = {"job_id": job_id, "slots": slots, "keys": keys}
        self.stats["submitted"] += 1
        return True

    def _read_row(self, tensor_idx: int, slot: int, nbytes: int):
        return self._staging[tensor_idx][slot][:nbytes].numpy().data

    def _write_job(self, keys, slots) -> int:
        from vllm.v1.kv_offload.base import get_offload_group_idx

        total = 0
        for key, slot in zip(keys, slots):
            g = get_offload_group_idx(key)
            path = self._mapper.get_file_name(key)
            if os.path.exists(path):
                # Already ours and complete -- write_chunks renames, so an
                # existing path is never a partial. This is the dedup that made
                # boot 2 write zero new files.
                continue
            total += write_chunks(path, file_plan(self._refs, g, slot),
                                  self._read_row)
            self.stats["files"] += 1
        return total

    def get_finished(self) -> list:
        from vllm.v1.kv_offload.worker.worker import TransferResult

        out = []
        for res in self._inner.get_finished():
            st = self._inflight.pop(res.job_id, None)
            if st is None:
                continue
            if not res.success:
                self._pool.give(st["slots"])
                out.append(TransferResult(job_id=st["job_id"], success=False))
                continue
            # Stage 1 done => the staging rows for these slots are valid NOW.
            fut = self._writers.submit(self._write_job, st["keys"], st["slots"])
            self._writing.append((st["job_id"], fut, st["slots"],
                                  len(st["keys"])))

        still = []
        for job_id, fut, slots, n in self._writing:
            if not fut.done():
                still.append((job_id, fut, slots, n))
                continue
            self._pool.give(slots)
            try:
                nbytes = fut.result()
                self.stats["bytes"] += nbytes
                out.append(TransferResult(
                    job_id=job_id, success=True, transfer_size=nbytes,
                    transfer_type=("GPU", self._spec.disk_medium)))
            except Exception:
                self.stats["errors"] += 1
                logger.exception("kvmgr: disk write failed for job %d", job_id)
                out.append(TransferResult(job_id=job_id, success=False))
        self._writing = still
        return out

    def wait(self, job_ids: set) -> None:
        import time
        deadline = time.monotonic() + 300.0
        while time.monotonic() < deadline:
            done = {r.job_id for r in self.get_finished()}
            job_ids -= done
            if not job_ids:
                return
            time.sleep(0.001)
        logger.error("kvmgr: wait() timed out with %d jobs outstanding",
                     len(job_ids))

    def shutdown(self) -> None:
        self._writers.shutdown(wait=True)

    def snapshot(self) -> dict:
        return dict(self.stats, staging=self._pool.snapshot(),
                    group_file_sizes=self._sizes)
