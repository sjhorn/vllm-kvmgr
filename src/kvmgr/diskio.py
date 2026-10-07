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
        # Batching per group is only trivial at factor 1 (block_indices enters
        # the inner arithmetic as `block_idx % factor`). Asserted rather than
        # assumed, because a silently wrong offset is the F8_E8M0 class.
        assert spec.block_size_factor == 1, (
            "kvmgr: the disk handler batches per group, which assumes "
            f"block_size_factor == 1 (got {spec.block_size_factor}). Do not set "
            "'block_size' in kv_connector_extra_config.")
        write_geometry(kv_caches, spec,
                       os.environ.get("KVMGR_SNAPSHOT_DIR", "/etc/vllm-kv-flags"))
        self._pool = SlotPool(staging_blocks)
        self._writers = ThreadPoolExecutor(max_workers=write_threads,
                                           thread_name_prefix="kvmgr_disk")
        self._jobs: dict[int, dict] = {}          # OUR job id -> rounds left
        self._inner_to_job: dict[int, tuple] = {}  # inner job id -> round
        self._writing: list[tuple[int, Any, list[int]]] = []
        self._next_inner = 1
        self._saturated = False
        self.stats = {"submitted": 0, "refused_pairing": 0,
                      "files": 0, "bytes": 0, "errors": 0}
        logger.info(
            "kvmgr: disk store handler -- staging %d blocks x %d tensors "
            "(pinned=%s), per-group file sizes %s, sum %d B (the per-rank page; "
            "the tiering tier wrote %d B per group instead)",
            staging_blocks, len(self._staging), pin, self._sizes,
            sum(self._sizes), spec.whole_page_bytes,
        )

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
                              "files": 0, "failed": False}
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
                out.append(TransferResult(
                    job_id=job_id, success=True, transfer_size=st["bytes"],
                    transfer_type=("GPU", self._spec.disk_medium)))
        return out

    def wait(self, job_ids: set) -> None:
        import time
        deadline = time.monotonic() + 600.0
        remaining = set(job_ids)
        while remaining and time.monotonic() < deadline:
            for r in self.get_finished():
                remaining.discard(r.job_id)
            if remaining:
                time.sleep(0.0005)
        if remaining:
            logger.error("kvmgr: wait() timed out with %d job(s) outstanding",
                         len(remaining))

    def shutdown(self) -> None:
        self._writers.shutdown(wait=True)

    def snapshot(self) -> dict:
        return dict(self.stats, staging=self._pool.snapshot(),
                    group_file_sizes=self._sizes,
                    jobs_open=len(self._jobs), writes_open=len(self._writing))
