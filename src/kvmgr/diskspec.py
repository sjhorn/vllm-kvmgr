"""diskspec.py -- an OffloadingSpec that puts the DISK transfer WORKER-SIDE.

*** WHY THIS EXISTS AT ALL. *** Upstream's TieringOffloadingSpec cannot capture
half of this cluster's KV. Its own docs say it "consolidates all shards into a
single shared host memory region" -- one shared HOST region, which is single-host
TP by construction. Ours is world_size=2 with local_world_size=1: rank 1 lives on
another machine with another /dev/shm, and the secondary tiers that cascade to
disk are built scheduler-side, in the engine core, on the head only.

Measured 2026-10-07 and this is the whole motivation: every .bin written by the
tiering tier is page*world_size bytes with the SECOND HALF EXACTLY ZERO. Rank 1's
slice, never written. A restore would have loaded half-zero pages -- half the KV
silently wrong, nothing crashing. `prepare_load` never having fired is the only
reason that has not already produced garbage.

*** THE SHAPE, and why it is single-hop. *** The scheduler sees ONE medium,
DSPARK_DISK, and one hop GPU<->DISK. The two-stage reality (GPU -> pinned staging
-> file) is the HANDLER's business, worker-side, where each rank naturally has its
own GPU shard and its own local disk. The alternative -- keeping upstream's CPU
tier and adding a CPU->DISK cascade -- puts the cascade back in the scheduler,
which is exactly where the node-locality bites. Single-hop also drops the CPU
primary tier entirely: a small staging buffer instead of a 510-block, 1.07 GB
cache, which on unified memory is the zero-sum cost the original plan wanted gone.

*** PHASE 1a IS WRITE-ONLY BY CONSTRUCTION, NOT BY OMISSION. *** `lookup()`
returns False unless `disk_restore` is set, and the scheduler only calls
`prepare_load` for keys `lookup` claimed. So with the flag off there is no code
path from a file to a GPU tensor at all. That is deliberate: until Phase 3 proves
a restored block equals what was stored, the safest restore is one that cannot
run.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadingSpec,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.file_mapper import FileMapper

from . import flags

if TYPE_CHECKING:
    from collections.abc import Collection, Iterable, Iterator

    from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadKey, ReqContext
    from vllm.v1.kv_offload.worker.worker import OffloadingHandler

# A logger named __name__ is SILENTLY DROPPED: vLLM gives a handler to the
# `vllm` logger only, so `kvmgr.*` sits at effective level 30 and every info()
# vanishes -- which looks exactly like the plugin never loading. Cost us a boot.
logger = init_logger("vllm." + __name__)

# Our medium string. It MUST NOT collide with upstream's {CPU, GPU}:
# OffloadingWorker.register_handler keys on (src_medium, dst_medium) and ASSERTS
# on a duplicate pair, so a collision is a boot-time abort, not a subtle bug.
# test_phase0_disk.py pins this against the mediums scanned out of pristine.
DISK_MEDIUM = "DSPARK_DISK"

# *** THE ON-DISK LAYOUT IS VERSIONED BY A SEPARATE ROOT, AND THAT IS
# CORRECTNESS RATHER THAN TIDINESS. *** 6,141 files written by the tiering tier
# are still on disk, each one page*world_size bytes with a zero tail. They live
# under the fingerprint FileMapper computes, and that fingerprint knows nothing
# about who wrote the file or what a "page" meant to them -- so sharing a root
# would let os.path.exists() find a half-empty file and call it a hit. A distinct
# root makes the old era unreachable by construction.
#
# Phase 4 moves this INTO the fingerprint hash proper (FileMapper's `fields`),
# which is stronger because it also covers a future change of our own format.
# A separate root is the cheap half of that and is enough today.
DISK_FORMAT_ROOT = "dspark-kv-v1"


class DiskLoadStoreSpec(LoadStoreSpec):
    """Addresses blocks by their OffloadKey, NOT by a block id.

    An OffloadKey is plain `bytes` packing (block_hash, group_idx), so this whole
    object serialises across the engine-core -> worker IPC with no help. That is
    what lets ONE spec, built once scheduler-side, reach every rank -- each of
    which then appends its OWN rank to the path. The keys are deliberately
    rank-agnostic: rank lives in the path and outside the fingerprint, which is
    precisely what makes `_r0` and `_r1` siblings under one fingerprint.
    """

    def __init__(self, keys: list["OffloadKey"]):
        self.keys = list(keys)

    @staticmethod
    def medium() -> str:
        return DISK_MEDIUM

    def __repr__(self) -> str:
        return f"DiskLoadStoreSpec({len(self.keys)} keys)"


class DiskPresenceManager(OffloadingManager):
    """Scheduler-side bookkeeping: which keys are believed to be on disk.

    *** THE ALL-RANKS PROBLEM IS ALREADY SOLVED BY THE FRAMEWORK, which was the
    find that collapsed this phase. *** I had scoped a presence handshake as its
    own phase, on the grounds that the scheduler cannot see another node's
    filesystem so os.path.exists on its own node proves only that HALF the block
    is there. But OffloadingWorkerMetadata's docstring says it outright: workers
    report {job_id: 1}, `aggregate()` sums across workers, and the scheduler
    "processes a transfer completion only when count reaches num_workers". The
    scheduler sets pending_count=world_size for every job.

    So `complete_store` is called if and ONLY if every rank wrote its slice. A
    key marked present here is all-ranks correct for free, and no protocol of
    ours is needed.

    WHAT IS STILL OWED (Phase 2): this index is in MEMORY, so it dies with the
    engine. That is harmless today -- nothing is marked present after a restart,
    so everything is re-stored and the writer's own exists() check makes that
    cheap, which is exactly the "boot 2 wrote zero new files" result. It matters
    only once `disk_restore` is on, because then an unpersisted index means a
    warm cache nobody can find. Persisting it is Phase 2.
    """

    # Flush cadence. A write per completed store would be one fsync-ish rename
    # per block; batching costs at most this many keys of a crash's worth of
    # forgotten (not wrong) presence, which only ever means a needless recompute.
    MANIFEST_FLUSH_KEYS = 512

    def __init__(self, file_mapper: FileMapper, num_workers: int,
                 manifest_path: str | None = None):
        self._mapper = file_mapper
        self._num_workers = num_workers
        self._present: set[bytes] = set()
        self._events: list[OffloadingEvent] = []
        self._stats = {"lookup": 0, "hit": 0, "stored": 0, "skipped_present": 0,
                       "manifest_loaded": 0, "manifest_pruned": 0,
                       "manifest_writes": 0}
        self._manifest = manifest_path
        self._unflushed = 0
        if manifest_path:
            self._load_manifest()

    # -- the manifest -----------------------------------------------------
    def _load_manifest(self) -> None:
        """Restore presence from disk, PRUNED against the local filesystem.

        *** WHY A MANIFEST AND NOT JUST os.path.exists(). *** exists() would
        survive a restart too, and it is what the fs tier uses -- but it is
        RACY in exactly the way that matters here. Rank 0's file lands before
        the all-ranks completion, so a lookup in the gap would claim a block
        that rank 1 is still writing or has failed to write. The manifest is
        written only from complete_store, which the framework calls if and only
        if every rank reported the job done, so a key in it was written
        EVERYWHERE.

        *** AND IT IS PRUNED, because the manifest is a claim about two
        filesystems and can only be checked against one. *** An entry whose own
        `_r0` file has gone -- somebody cleared the directory, a budget evicted
        it -- is dropped at startup. What that CANNOT catch is rank 1's files
        being deleted while the head's survive: upstream asserts on a failed
        transfer, so such a load would kill the engine core rather than return
        wrong KV. That is the right way round, and the recovery is to delete
        this file; the log line below says so.
        """
        import json
        import os as _os

        if not self._manifest or not _os.path.exists(self._manifest):
            return
        try:
            with open(self._manifest) as f:
                keys = [bytes.fromhex(h) for h in json.load(f).get("keys", [])]
        except Exception:
            logger.exception(
                "kvmgr: manifest %s is unreadable -- starting with an EMPTY "
                "index, which costs recompute and never correctness",
                self._manifest)
            return
        kept = []
        for k in keys:
            try:
                if _os.path.exists(self._mapper.get_file_name(k)):
                    kept.append(k)
            except Exception:
                pass
        self._present = set(kept)
        self._stats["manifest_loaded"] = len(kept)
        self._stats["manifest_pruned"] = len(keys) - len(kept)
        logger.info(
            "kvmgr: manifest %s -- %d keys restored, %d pruned (their own file "
            "is gone). If a load ever asserts on a missing file, DELETE THIS "
            "FILE: it is a claim about both nodes and can only be checked "
            "against this one.",
            self._manifest, len(kept), len(keys) - len(kept))

    def flush_manifest(self, force: bool = False) -> None:
        import json
        import os as _os

        if not self._manifest:
            return
        if not force and self._unflushed < self.MANIFEST_FLUSH_KEYS:
            return
        try:
            _os.makedirs(_os.path.dirname(self._manifest), exist_ok=True)
            tmp = self._manifest + ".part"
            with open(tmp, "w") as f:
                json.dump({"keys": sorted(k.hex() for k in self._present)}, f)
            _os.replace(tmp, self._manifest)
            self._unflushed = 0
            self._stats["manifest_writes"] += 1
        except Exception:
            logger.exception("kvmgr: manifest flush failed (presence kept in "
                             "memory; a restart will simply forget it)")

    # -- reads ------------------------------------------------------------
    def lookup(self, key: "OffloadKey", req_context: "ReqContext") -> bool | None:
        self._stats["lookup"] += 1
        if not flags.peek("disk_restore"):
            # Write-only by construction: the scheduler only calls prepare_load
            # for keys lookup claimed, so returning False here means no file can
            # reach a GPU tensor at all. See this module's docstring.
            return False
        hit = bytes(key) in self._present
        if hit:
            self._stats["hit"] += 1
        return hit

    def prepare_load(
        self, keys: "Collection[OffloadKey]", req_context: "ReqContext"
    ) -> LoadStoreSpec:
        if not flags.peek("disk_restore"):
            # Unreachable unless lookup lied. Loud rather than best-effort: a
            # silent half-load is the one outcome worse than no cache at all.
            raise RuntimeError(
                "kvmgr: prepare_load called while disk_restore is OFF -- the "
                "scheduler should only ask for keys lookup() claimed. Refusing "
                "rather than loading something we have not proven."
            )
        return DiskLoadStoreSpec(list(keys))

    # -- writes -----------------------------------------------------------
    def prepare_store(
        self, keys: "Collection[OffloadKey]", req_context: "ReqContext"
    ) -> PrepareStoreOutput | None:
        # Skipping keys we already hold is the dedup that made boot 2 write zero
        # new files. Cheap here, and it keeps the writer's exists() check as the
        # second line of defence rather than the only one.
        todo = [k for k in keys if bytes(k) not in self._present]
        self._stats["skipped_present"] += len(keys) - len(todo)
        if not todo:
            return None
        return PrepareStoreOutput(
            keys_to_store=todo,
            store_spec=DiskLoadStoreSpec(todo),
            # No eviction yet -- a byte budget with its own LRU is Phase 5.
            # Returning [] means "nothing was displaced", which is true.
            evicted_keys=[],
        )

    def complete_store(
        self, keys: "Collection[OffloadKey]", req_context: "ReqContext"
    ) -> None:
        new = [k for k in keys if bytes(k) not in self._present]
        for k in new:
            self._present.add(bytes(k))
        self._stats["stored"] += len(new)
        if new:
            self._events.append(
                OffloadingEvent(keys=list(new), medium=DISK_MEDIUM, removed=False)
            )
            # complete_store is the ALL-RANKS edge, so this is the only place a
            # key may be claimed present -- see _load_manifest.
            self._unflushed += len(new)
            self.flush_manifest()

    def complete_load(
        self, keys: "Collection[OffloadKey]", req_context: "ReqContext"
    ) -> None:
        return None

    def touch(self, keys: "Collection[OffloadKey]", req_context: "ReqContext") -> None:
        return None

    def take_events(self) -> "Iterable[OffloadingEvent]":
        events, self._events = self._events, []
        return events

    def reset_cache(self) -> None:
        self._present.clear()
        self.flush_manifest(force=True)

    # -- ours, for the /kv endpoint ---------------------------------------
    def snapshot(self) -> dict[str, Any]:
        return dict(self._stats, present=len(self._present),
                    num_workers=self._num_workers,
                    restore_enabled=bool(flags.peek("disk_restore")))


class DSparkDiskOffloadingSpec(OffloadingSpec):
    """GPU <-> local disk, one hop, with the transfer done worker-side.

    Inherits OffloadingSpec.__init__ unchanged, and THAT IS THE FIND OF PHASE 0:
    `offload_allow_hybrid` exists only because TieringOffloadingSpec adds
    `assert len(self.gpu_block_size) == 1`. The base asserts something weaker --
    each group's block size divisible by hash_block_size -- which all five of
    (256, 64, 64, 4, 8) satisfy against a resolved 4. So this spec inherits NO
    single-block-size limit and needs no shim. block_size_factor stays 1 because
    setting it requires a uniform block size we do not have.
    """

    def __init__(self, vllm_config, kv_cache_config):
        super().__init__(vllm_config, kv_cache_config)

        parallel = vllm_config.parallel_config
        self.world_size: int = parallel.world_size
        self.rank: int = parallel.rank

        # THE PER-WORKER SLICE, derived exactly as cpu/spec.py derives it rather
        # than hardcoded -- because getting this wrong is how the tiering tier
        # came to write a page it only half filled. The fs tier takes
        # primary_kv_view.strides[0], which is the WHOLE page across all ranks;
        # the honest number is that divided by world_size.
        if kv_cache_config is not None and kv_cache_config.num_blocks > 0:
            total_gpu_kv_bytes = sum(t.size for t in kv_cache_config.kv_cache_tensors)
            kv_bytes_per_block = (
                total_gpu_kv_bytes // kv_cache_config.num_blocks
            ) * self.world_size
        else:
            kv_bytes_per_block = 0
        kv_bytes_per_offloaded_block = kv_bytes_per_block * self.block_size_factor
        self.whole_page_bytes: int = kv_bytes_per_offloaded_block
        self.page_size_per_worker: int = (
            kv_bytes_per_offloaded_block // self.world_size if self.world_size else 0
        )

        root = self.extra_config.get("root_dir") or os.environ.get(
            "KVMGR_DISK_ROOT", "/kvoffload"
        )
        self.root_dir: str = os.path.join(root, DISK_FORMAT_ROOT)

        # Staging is SMALL by design -- that is the whole point of not having a
        # CPU tier. 32 blocks of ~1 MB against the 510-block, 1.07 GB region we
        # are replacing. Exhaustion is normal and survivable: the connector
        # retries unsubmitted store jobs, so transfer_async returning False is
        # an expected answer.
        self.staging_blocks: int = int(self.extra_config.get("staging_blocks", 32))
        self.write_threads: int = int(self.extra_config.get("write_threads", 4))
        self.disk_medium: str = DISK_MEDIUM

        self._mapper: FileMapper | None = None
        self._manager: DiskPresenceManager | None = None

        logger.info(
            "kvmgr: disk spec -- world_size=%d rank=%d, whole page %d B, "
            "per-worker slice %d B (what WE write; the tiering tier wrote the "
            "whole page and left rank>0's half zero), root=%s",
            self.world_size, self.rank, self.whole_page_bytes,
            self.page_size_per_worker, self.root_dir,
        )

    def file_mapper(self) -> FileMapper:
        if self._mapper is None:
            self._mapper = FileMapper.from_offloading_spec(
                root_dir=self.root_dir,
                offloading_spec=self,
                gpu_blocks_per_file=self.block_size_factor,
            )
            # *** LOG THE WHOLE FINGERPRINT, because the two processes DISAGREED
            # about it and nothing said so. *** Measured 2026-10-07: the worker
            # wrote every file under `..._06be92914bd2_r0` while the engine
            # core's prune looked somewhere else and discarded all 552 manifest
            # keys as "file is gone" -- with the files sitting right there. The
            # scheduler and the worker build their FileMapper independently from
            # their own vllm_config/kv_cache_config, so any field that differs
            # between the processes silently forks the path. Printing the FIELDS
            # rather than just the hash is what names the culprit in one boot
            # instead of one boot per guess.
            try:
                import json as _json
                logger.info(
                    "kvmgr: file_mapper base=%s fields=%s",
                    self._mapper.base_path,
                    _json.dumps(self._mapper.fields, sort_keys=True,
                                default=str)[:1200],
                )
            except Exception:
                logger.exception("kvmgr: could not log the mapper fingerprint")
        return self._mapper

    def get_manager(self) -> OffloadingManager:
        if self._manager is None:
            self._manager = DiskPresenceManager(
                self.file_mapper(), num_workers=self.world_size,
                # Beside the KV it describes, so clearing the directory clears
                # the claim too -- and under the format root, so a format change
                # cannot inherit a stale index.
                manifest_path=os.path.join(self.root_dir, "presence.json"),
            )
        return self._manager

    def get_handlers(
        self, kv_caches: "CanonicalKVCaches"
    ) -> "Iterator[tuple[type[LoadStoreSpec], type[LoadStoreSpec], OffloadingHandler]]":
        """Register the STORE direction only.

        No (DISK -> GPU) handler exists yet, and that is the same safety property
        `lookup()` enforces one layer up, done twice on purpose: with no load
        handler registered, OffloadingWorker has nothing to route a load to, so a
        restore cannot happen even if a future edit made lookup() optimistic.
        Phase 3 adds the pair, and only once it can prove a restored block equals
        what was stored.
        """
        from vllm.v1.kv_offload.base import GPULoadStoreSpec

        from .diskio import DiskLoadHandler, DiskStoreHandler

        handler = DiskStoreHandler(
            spec=self,
            kv_caches=kv_caches,
            staging_blocks=self.staging_blocks,
            write_threads=self.write_threads,
        )
        self._store_handler = handler
        yield (GPULoadStoreSpec, DiskLoadStoreSpec, handler)

        # *** THE LOAD DIRECTION IS REGISTERED ONLY IF disk_restore IS SET, and
        # that is a SECOND lock on the same door. *** lookup() already refuses
        # while the flag is off, so the scheduler never asks for a load; this
        # means that even if a later edit made lookup() optimistic,
        # OffloadingWorker would have no handler to route the transfer to.
        #
        # It is read ONCE here rather than per transfer, unlike every other knob
        # in this plugin, and that is deliberate: handlers are registered at boot
        # and cannot appear later, so a hot read would promise something the
        # framework cannot deliver. This is the one knob that needs a restart,
        # and saying so is better than a flag that silently does nothing.
        if flags.peek("disk_restore"):
            self._load_handler = DiskLoadHandler(
                spec=self, kv_caches=kv_caches,
                staging_blocks=self.staging_blocks,
                read_threads=self.write_threads, store=handler,
            )
            yield (DiskLoadStoreSpec, GPULoadStoreSpec, self._load_handler)
        else:
            logger.info(
                "kvmgr: restore is OFF, so no (DISK -> GPU) handler is "
                "registered at all. Set disk_restore AND restart to enable it; "
                "a hot flip cannot work, because handlers are registered once.")
