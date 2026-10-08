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

# The name KV_TRANSFER_CONFIG must use as `spec_name`, and the one
# register.py registers with. A constant rather than two string
# literals, because the store path is silently inert if they disagree.
DSPARK_SPEC_NAME = "DSparkDiskOffloadingSpec"

# The live manager, for the /kv endpoint. None until the connector builds one,
# which only happens in the process that holds the scheduler -- the same one
# that serves /kv, which is why this works without IPC.
_LIVE_MANAGER: "DiskPresenceManager | None" = None

# *** THE SCHEDULER-SIDE JOB SET, WHICH NOTHING COULD SEE. ***
# The worker's own counters said `jobs_open: 0` while the scheduler was still
# holding ~127 jobs -- and the scheduler's `self._jobs` is EXACTLY what
# `jobs_to_flush` names, so the fence was waiting on jobs the worker had
# already finished and forgotten. That asymmetry was invisible: the only time
# `jobs_to_flush` ever appeared was in a crash's dump_input.
#
# `complete_store` fires only when `pending_count` reaches 0, and it starts at
# `num_workers`, so BOTH ranks must report. The histogram of pending_count over
# the residual jobs is therefore the whole diagnosis in one field:
#   all 1  -> exactly one rank's completion never arrived
#   all 2  -> neither arrived
#   mixed  -> it is per-job, not per-rank
_LIVE_OFFLOAD_SCHED: list = []
# How the handle was obtained. *** THE FIRST VERSION OF THIS LIED. *** It
# reported "hook" vs "gc-scan" by testing whether __init__ was wrapped, which
# is a different question from how the object was actually found -- a
# call-stack capture was labelled "gc-scan", and I nearly concluded from that
# label that gc.freeze() does not hide boot-time objects after all. A field
# that answers a different question than its name is worse than no field.
_SCHED_FOUND_BY: list = []


SCHED_CLS_NAME = "OffloadingConnectorScheduler"


def capture_offload_sched_from_caller(depth: int = 5) -> bool:
    """Grab the connector scheduler off the CALL STACK. One-shot, exact.

    *** A gc SCAN CANNOT FIND IT, AND THAT IS NOT A BUG IN THE SCAN. ***
    `EngineCore.__init__` calls `freeze_gc_heap()` -> `gc.freeze()`, which moves
    every object allocated during boot into the permanent generation -- and
    `gc.get_objects()` DOES NOT RETURN frozen objects. The connector scheduler
    is built at boot, so it is permanently invisible to a scan. Measured
    2026-10-08: the scan reported "no live instance" after 17,003 objects, which
    is implausibly few for an engine core and is the tell -- those 17,003 are
    only the POST-freeze allocations.
    
    *** THE SCOPE OF THAT CLAIM, STATED HONESTLY. *** `EngineCore.__init__`
    does call `freeze_gc_heap()`, and a frozen object is not returned by
    `gc.get_objects()`. But `core.py:604` UNDOES that freeze later, so whether a
    boot-time object is visible to a scan depends on when you look -- and a
    scan did eventually find this one. So the rule is not "a scan can never
    see it"; it is that **a gc scan is UNRELIABLE for a boot-time object in
    the engine core, in a way that depends on timing you do not control**.
    The call-stack capture is exact and needs no such reasoning, which is why
    it is the primary route rather than the fallback.
    
    So: the connector scheduler is the thing that CALLS us. `prepare_store` and
    `lookup` are invoked from its own methods, so its `self` is sitting in a
    caller frame. That is exact rather than heuristic, needs no restart, and
    costs one dict lookup per call once it has succeeded.
    
    Matched on the class NAME rather than with isinstance, deliberately: an
    upstream module move would make isinstance fail against a class we imported
    from the old path, and this is a diagnostic -- a name match that is right
    99% of the time beats an import that is wrong after a version bump. The
    attribute reads in the snapshot are all getattr-guarded anyway.
    """
    if _LIVE_OFFLOAD_SCHED:
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
            if cand is not None and type(cand).__name__ == SCHED_CLS_NAME:
                del _LIVE_OFFLOAD_SCHED[:]
                _LIVE_OFFLOAD_SCHED.append(cand)
                return True
    except Exception:
        return False
    finally:
        # Frames hold references to their locals; keeping one alive would pin
        # a request's state and build a cycle on the scheduler's hot path.
        del f
    return False


def _discover_offload_sched():
    """Find a live OffloadingConnectorScheduler by scanning gc. On demand only.

    Returns `(obj_or_None, why)`. *** A REFUSAL MUST NAME ITS CAUSE, and the
    first version of this did not. *** It returned a bare None, so a failed
    discovery and an engine with genuinely no connector were indistinguishable
    -- and the three real causes (the module will not import, the class is not
    where we look, nothing of that type is alive in THIS process) want
    completely different responses. That cost a round.
    """
    try:
        import gc
    except Exception as exc:
        return None, "gc unavailable: %r" % (exc,)
    mods = (
        "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler",
        # Upstream has moved this at least once; both spellings are tried and
        # the one that worked is reported, so a future move is a named miss.
        "vllm.v1.kv_offload.scheduler",
    )
    tried = []
    cls = None
    for mod in mods:
        try:
            import importlib
            m = importlib.import_module(mod)
            cls = getattr(m, "OffloadingConnectorScheduler", None)
            if cls is not None:
                tried.append("%s: found class" % mod)
                break
            tried.append("%s: imported, no OffloadingConnectorScheduler" % mod)
        except Exception as exc:
            tried.append("%s: %r" % (mod, exc))
    if cls is None:
        return None, "class not found -- " + "; ".join(tried)
    n = 0
    try:
        for obj in gc.get_objects():
            n += 1
            try:
                if isinstance(obj, cls):
                    del _LIVE_OFFLOAD_SCHED[:]
                    _LIVE_OFFLOAD_SCHED.append(obj)
                    del _SCHED_FOUND_BY[:]
                    _SCHED_FOUND_BY.append("gc-scan")
                    return obj, "gc-scan"
            except Exception:
                continue
    except Exception as exc:
        return None, "gc scan raised after %d objects: %r" % (n, exc)
    return None, ("no live instance in THIS process (%s); scanned %d objects. "
                  "The scheduler-side connector lives in the engine core, so "
                  "this answer is only meaningful there." % (tried[-1], n))


def offload_sched_snapshot() -> dict:
    """Read-only view of upstream's scheduler-side job bookkeeping.

    Best-effort and defensive to the point of paranoia: this reaches into
    another project's private attributes, so every one of them is read with
    getattr and a default. An upstream rename must show up here as an absent
    field, never as an exception on the scheduler's hot path -- and never as a
    0, which would read as "no pending jobs" and is the one wrong answer.
    """
    out: dict = {}
    try:
        cs = _LIVE_OFFLOAD_SCHED[-1] if _LIVE_OFFLOAD_SCHED else None
    except Exception:
        cs = None
    if cs is None:
        # *** DISCOVER IT ON DEMAND, so this works on an engine that was
        # *** ALREADY RUNNING. The hook wraps __init__, which runs when the
        # connector is built -- i.e. at BOOT -- so a hot reload can never arm
        # it and the diagnostic would cost a 4-minute restart plus a cold
        # prefill to use once. That is exactly the cold-path habit Stage 6
        # exists to remove.
        #
        # A gc scan is expensive, which is fine HERE and nowhere else: this is
        # an on-demand endpoint, never the store path. The result is cached in
        # the same list the hook uses, so the scan happens at most once per
        # process. `found_by` says which route supplied it, because "the hook
        # fired" and "we went looking" are different facts about the build.
        cs, _why = _discover_offload_sched()
    else:
        _why = "hook"
    if cs is None:
        return {"available": False,
                "why": "no OffloadingConnectorScheduler reachable -- this is an "
                       "ABSENCE, not zero pending jobs",
                "discovery": _why}
    try:
        jobs = getattr(cs, "_jobs", None)
        if jobs is None:
            return {"available": False, "why": "upstream has no _jobs attribute"}
        hist: dict[str, int] = {}
        stores = loads = 0
        oldest = None
        for jid, st in list(jobs.items()):
            pc = getattr(st, "pending_count", None)
            hist[str(pc)] = hist.get(str(pc), 0) + 1
            if getattr(st, "is_store", False):
                stores += 1
            else:
                loads += 1
            oldest = jid if oldest is None else min(oldest, jid)
        out = {
            "available": True,
            "found_by": (_SCHED_FOUND_BY[-1] if _SCHED_FOUND_BY
                         else "hook-or-unknown"),
            "init_wrapped": bool(getattr(
                type(cs).__init__, "_kvmgr_wrapped", False)),
            "jobs_open": len(jobs),
            "stores_open": stores,
            "loads_open": loads,
            "oldest_job_id": oldest,
            "pending_count_hist": hist,
            "job_counter": getattr(cs, "_job_counter", None),
            "stale_job_threshold": getattr(cs, "_stale_job_threshold", None),
            "blocks_with_pending_jobs": len(
                getattr(cs, "_block_id_to_pending_jobs", {}) or {}),
            "batch_jobs_to_flush": len(
                getattr(cs, "_current_batch_jobs_to_flush", ()) or ()),
            "num_workers": getattr(getattr(cs, "config", None),
                                   "num_workers", None),
        }
    except Exception as exc:
        return {"available": False, "why": "reading upstream state raised: %r" % (exc,)}
    return out



def live_manager_snapshot() -> dict | None:
    """The disk index's own counters, or None if no spec is in force.

    None rather than an empty dict, so "no disk spec configured" and "a disk
    spec reporting zeros" are different answers on the endpoint -- which is
    this project's absent-is-not-zero rule applied to its own instrument.
    """
    m = _LIVE_MANAGER
    if m is None:
        return None
    try:
        return m.snapshot()
    except Exception:
        return {"error": "snapshot raised"}


def _now() -> float:
    """Monotonic seconds. A helper so the manifest's time floor cannot be
    accidentally written against the wall clock, which a clock step would
    then make either permanently stale or never stale."""
    import time
    return time.monotonic()


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
    # And a time floor, because a session adding fewer than 512 keys would
    # otherwise never flush at all. See flush_manifest for the measurement.
    MANIFEST_FLUSH_S = 30.0

    def __init__(self, file_mapper: FileMapper, num_workers: int,
                 manifest_path: str | None = None):
        self._mapper = file_mapper
        self._num_workers = num_workers
        self._present: set[bytes] = set()
        self._events: list[OffloadingEvent] = []
        self._stats = {"lookup": 0, "hit": 0, "stored": 0, "skipped_present": 0,
                       "manifest_loaded": 0, "manifest_pruned": 0,
                       "manifest_writes": 0, "paused": 0}
        # offered / to_store / completed, keyed by group index. A plain dict of
        # dicts rather than a list, because the group count is not known here
        # and a missing group must read as ABSENT rather than as 0.
        self._gcounts: dict[str, dict[int, int]] = {}
        self._manifest = manifest_path
        self._unflushed = 0
        self._last_flush = _now()
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
        # *** A COUNT THRESHOLD ALONE LOSES EVERY SHORT SESSION. *** Measured
        # 2026-10-08: a run stored 1,052 new files (~210 keys) and
        # presence.json kept the PREVIOUS day's mtime -- under the 512-key
        # threshold, so it never flushed, so those keys existed only in memory
        # and a restart forgot them. The files were on disk and the index could
        # not name them: the restore-after-restart case, broken by construction,
        # with nothing saying so.
        #
        # So EITHER enough keys OR enough time. The time floor is also what
        # covers an UNCLEAN kill, which no shutdown hook can -- and being killed
        # mid-session is the normal way this process ends. The manifest is tens
        # of KB on an NVMe, so a bounded periodic write costs nothing worth
        # counting; this is not the Pi's SD card.
        if not force:
            if not self._unflushed:
                return
            stale = (_now() - self._last_flush) >= self.MANIFEST_FLUSH_S
            if self._unflushed < self.MANIFEST_FLUSH_KEYS and not stale:
                return
        try:
            _os.makedirs(_os.path.dirname(self._manifest), exist_ok=True)
            tmp = self._manifest + ".part"
            with open(tmp, "w") as f:
                json.dump({"keys": sorted(k.hex() for k in self._present)}, f)
            _os.replace(tmp, self._manifest)
            self._unflushed = 0
            self._last_flush = _now()
            self._stats["manifest_writes"] += 1
        except Exception:
            logger.exception("kvmgr: manifest flush failed (presence kept in "
                             "memory; a restart will simply forget it)")

    # -- reads ------------------------------------------------------------
    def lookup(self, key: "OffloadKey", req_context: "ReqContext") -> bool | None:
        self._stats["lookup"] += 1
        # One-shot, and `lookup` is the right place: it is called by the
        # connector scheduler on every request whether or not the store path is
        # enabled, so the handle is captured even with the tier paused.
        if not _LIVE_OFFLOAD_SCHED:
            capture_offload_sched_from_caller()
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
        # *** THE HOT OFF SWITCH, AND prepare_store IS THE ONLY SAFE PLACE FOR
        # IT. *** A mechanism gets a hot off switch before it gets a feature --
        # a rule this cost a user-visible outage to learn, on 2026-10-08: the
        # store path had no way to be disabled without a SIX-MINUTE restart, so
        # the only mitigation was one the box could not afford and a suspect
        # path was left running on a live engine.
        #
        # Pausing in the HANDLER would be a correctness trap, which is worth
        # stating because it is the obvious place to put it. The handler cannot
        # report False -- the connector's `assert success` makes that fatal on
        # every path (the Phase 1c lesson) -- so it would have to report
        # SUCCESS with zero bytes. That fires `complete_store`, which marks
        # those keys PRESENT in the index while nothing was written, so a later
        # `lookup` claims a hit and the restore then dies on a missing file.
        # A pause that poisons the index is worse than no pause.
        #
        # Returning None here means "nothing to store", which `prepare_store`
        # already has a path for two lines down: no job is created, nothing is
        # written, and nothing is claimed. The pause is therefore invisible to
        # correctness by construction rather than by care.
        # Capture the connector scheduler here as well as in `lookup`, because
        # `lookup` is only reached on an external prefix-cache path and may not
        # run for hours, whereas `prepare_store` is called on every request that
        # fills a block -- the `paused` counter climbing proves it is called
        # even with the tier paused, since the gate below is OURS and sits
        # inside the call rather than preventing it.
        if not _LIVE_OFFLOAD_SCHED:
            capture_offload_sched_from_caller()

        if flags.peek("disk_store_paused"):
            self._stats["paused"] += len(keys)
            return _nothing_to_store()

        # Skipping keys we already hold is the dedup that made boot 2 write zero
        # new files. Cheap here, and it keeps the writer's exists() check as the
        # second line of defence rather than the only one.
        todo = [k for k in keys if bytes(k) not in self._present]
        self._stats["skipped_present"] += len(keys) - len(todo)
        # *** THE RATIONING INSTRUMENT. *** The restorable prefix grows by a
        # fixed +13,312 tokens / +1,196 keys per pass, LINEARLY, so one 180K
        # prompt needs ~14 passes before it fully restores. Three
        # indistinguishable causes: upstream only OFFERS a fraction per pass,
        # or we offer them all and the store jobs never COMPLETE (so
        # complete_store never marks them present and the next pass re-offers
        # them -- which is also the shape of the 21 never-completing jobs that
        # killed a worker), or the already-present filter is misfiring.
        #
        # One pass tells them apart only if all three quantities are counted
        # SEPARATELY and PER GROUP, because the five groups have wildly
        # different block sizes and the SWA groups are deliberately filtered
        # upstream. `skipped_present` reading 0 on every sample so far -- with
        # 102,386 keys held and the same prompt sent seven times -- is either
        # upstream never re-offering a stored key, or this filter not working,
        # and those want opposite fixes.
        self._count_groups("offered", keys)
        self._count_groups("to_store", todo)
        if not todo:
            return _nothing_to_store()
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
        # The all-ranks edge. Counted per group so `offered - completed` is
        # readable: a gap here is a job that never finished, not a key nobody
        # wanted.
        self._count_groups("completed", keys)
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

    def shutdown(self) -> None:
        """Flush on the way out, so a CLEAN restart keeps what it stored.

        The time floor above bounds the loss on an unclean kill; this makes it
        ZERO in the ordinary case. `shutdown` is concrete on the base class, so
        overriding it is in-contract rather than a hook of our own.
        """
        self.flush_manifest(force=True)

    def reset_cache(self) -> None:
        self._present.clear()
        self.flush_manifest(force=True)

    def _count_groups(self, what: str, keys: "Collection[OffloadKey]") -> None:
        """Tally `keys` by group index under `what`. Best-effort by design.

        An instrument that can break the path it measures is worse than no
        instrument -- this repo has paid for that twice in two days (a debug
        monkeypatch that lied for several rounds, and a geometry snapshot that
        nearly broke a boot by writing to a read-only mount). So this never
        raises into prepare_store: a key whose group cannot be read is counted
        under -1 and named as such, which is a visible absence rather than a
        silent miscount.
        """
        # *** THIS CRASHED THE ENGINE, AND THE DOCSTRING ABOVE PROMISED IT
        # *** COULD NOT. *** Measured live 2026-10-08:
        #
        #   AttributeError: 'DiskPresenceManager' object has no attribute
        #   '_gcounts'   <- in prepare_store, i.e. FATAL to the scheduler
        #
        # A RELOAD NEVER RUNS A CONSTRUCTOR. `rebind` re-points a live object
        # with `obj.__class__ = new`, so the instance gets the new METHODS and
        # never the new `__init__`. `_gcounts` is created in __init__, so the
        # live manager had a method that read an attribute it could not have.
        # I had written that exact lesson one commit earlier -- for a module
        # global and for the watchdog -- and did not carry it to the instance
        # attribute my own new code depended on.
        #
        # So: lazily initialise on the instance (NOT a class-level dict, which
        # every instance would then share), and wrap the WHOLE body. The
        # original guarded the import and the group extraction and left the one
        # line that actually failed outside -- a partial guard on a path where
        # the promise was total.
        try:
            from vllm.v1.kv_offload.base import get_offload_group_idx

            gc = getattr(self, "_gcounts", None)
            if gc is None:
                gc = {}
                self._gcounts = gc
            bucket = gc.setdefault(what, {})
            for k in keys:
                try:
                    g = get_offload_group_idx(k)
                except Exception:
                    g = -1
                bucket[g] = bucket.get(g, 0) + 1
        except Exception:
            # Never into the caller. A lost count is a lost count.
            return

    # -- ours, for the /kv endpoint ---------------------------------------
    def snapshot(self) -> dict[str, Any]:
        # by_group is the rationing diagnostic. Read it as three numbers per
        # group: OFFERED is what upstream asked us to store, TO_STORE is what
        # survived the already-present filter, COMPLETED is what every rank
        # confirmed. offered >> completed means jobs are not finishing;
        # offered small means upstream is rationing; to_store == offered
        # forever means the present filter never matches.
        # getattr, for the same rebind reason as _count_groups: a reloaded
        # class on a live instance has no __init__-assigned attributes.
        by_group = {
            what: {str(g): n for g, n in sorted(d.items())}
            for what, d in sorted(getattr(self, "_gcounts", {}).items())
        }
        return dict(self._stats, present=len(self._present),
                    num_workers=self._num_workers,
                    by_group=by_group,
                    sched=offload_sched_snapshot(),
                    restore_enabled=bool(flags.peek("disk_restore")),
                    store_paused=bool(flags.peek("disk_store_paused")))


def _nothing_to_store() -> PrepareStoreOutput:
    """"Nothing to store" -- which is NOT the same answer as `None`.

    *** RETURNING None HERE MADE THE SCHEDULER RETRY FOREVER, and the streamed
    log is the only reason we know. *** `kv_connector_offloading/scheduler.py`
    distinguishes the two in six lines:

        store_output = self.manager.prepare_store(...)
        if store_output is None:
            logger.warning("Request %s: cannot store blocks", req_id)
            continue                                   # does NOT advance
        if not store_output.keys_to_store:
            req_status.advance_stored_idx(num_offloadable_tokens)
            continue                                   # DOES advance

    So `None` means *"I cannot"* and leaves the stored index where it was, so
    the very same blocks are attempted again on the next step, and the next,
    indefinitely. An empty output means *"there is nothing new here"* and lets
    the request make progress.

    We were returning `None` for BOTH of our nothing-to-store cases -- paused,
    and every key already on disk. Measured 2026-10-08: **3,309 warnings at
    ~20/second**, sustained, with the scheduler re-attempting the same blocks
    every step. The log noise is the symptom; the absent progress is the
    defect.

    *** THE TRADE, STATED: advancing while PAUSED means those blocks are not
    stored later if the cache is unpaused. *** That is deliberate. "Paused"
    means we declined, and the alternative is an unbounded retry loop, which
    is strictly worse -- and it matches how the GPU cache behaves, where you
    do not get back what you did not cache.
    """
    return PrepareStoreOutput(keys_to_store=[],
                              store_spec=DiskLoadStoreSpec([]),
                              evicted_keys=[])


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
            # *** NOT `from_offloading_spec`, AND THAT IS THE WHOLE BUG. ***
            # Measured 2026-10-07: the worker wrote every file under
            # `..._06be92914bd2_r0` while the engine core's manifest prune
            # looked under `..._500ceb39bfc9` and discarded all 552 keys as
            # "file is gone", with the files sitting right there. The two
            # fingerprints differed in EXACTLY ONE field: `hash_block_size`,
            # 256 in the worker against 4 in the engine core.
            #
            # `from_offloading_spec` reads `vllm_config.cache_config.block_size`
            # (file_mapper.py:87) -- and on a HYBRID layout the engine core
            # rewrites that attribute to the resolved GCD of the five group
            # block sizes while the worker keeps the CLI's `--block-size 256`.
            # So upstream's helper forks the path between the store side and the
            # lookup side on any model with non-uniform groups. It is correct
            # for a single-block-size model, which is why nothing upstream has
            # tripped on it.
            #
            # The honest field is the spec's OWN `self.hash_block_size`, which
            # base.py:360 resolves with `resolve_kv_cache_block_sizes` from the
            # kv_cache_config -- the same input in both processes, so the same
            # answer (4) in both. Every other field the helper derives already
            # agreed, which the logged `fields=` dicts are the evidence for.
            #
            # Constructing directly also means the fingerprint's contents are
            # ours rather than inherited: a fingerprint is the one thing here
            # that must not change shape without us noticing, and
            # `test_mapper_fields_cover_upstreams` fails if upstream adds a key.
            parallel_config = self.vllm_config.parallel_config
            self._mapper = FileMapper(
                root_dir=self.root_dir,
                model_name=self.vllm_config.model_config.model,
                hash_block_size=self.hash_block_size,
                gpu_blocks_per_file=self.block_size_factor,
                tp_size=parallel_config.tensor_parallel_size,
                pp_size=parallel_config.pipeline_parallel_size,
                pcp_size=parallel_config.prefill_context_parallel_size,
                dcp_size=parallel_config.decode_context_parallel_size,
                rank=parallel_config.rank,
                dtype=str(self.vllm_config.cache_config.cache_dtype).replace(
                    "torch.", ""
                ),
                kv_cache_groups=[
                    {
                        "block_size": group.kv_cache_spec.block_size,
                        "layer_names": list(group.layer_names),
                    }
                    for group in self.kv_cache_config.kv_cache_groups
                ],
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
            # *** AND PUBLISH IT, because its counters were unreachable. ***
            # Measured 2026-10-08: the first working restore hit 2,048 of
            # 11,764 tokens and the question "how many keys does the index
            # actually hold" could not be answered from the box at all --
            # `DiskPresenceManager.snapshot()` existed and nothing served it,
            # so /kv's disk section was `{}`. The counters that would say
            # whether the shortfall is in `stored`, in `lookup` or in the
            # scheduler's contiguity rule are the whole diagnosis.
            #
            # A module global rather than a constructor argument: the spec is
            # built by the factory from a config string, so there is no call
            # site of ours to thread a reference through. Same shape as the
            # manager list the coordinator wrap stashes.
            global _LIVE_MANAGER
            self._manager = DiskPresenceManager(
                self.file_mapper(), num_workers=self.world_size,
                # Beside the KV it describes, so clearing the directory clears
                # the claim too -- and under the format root, so a format change
                # cannot inherit a stale index.
                manifest_path=os.path.join(self.root_dir, "presence.json"),
            )
            _LIVE_MANAGER = self._manager
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
        # *** REGISTERED UNCONDITIONALLY, AND THE OLD GATE KILLED THE ENGINE
        # TWICE OVER. *** This used to register the load handler only when
        # `disk_restore` was set, on the stated reasoning that withholding it
        # was a second gate -- "done twice so a later optimistic lookup() still
        # cannot restore". That reasoning was wrong about the FAILURE MODE, and
        # the box said so on 2026-10-08:
        #
        #   worker.py:133 in transfer_async
        #       assert handler is not None
        #   AssertionError
        #   RuntimeError: Worker failed with error ''
        #
        # The SCHEDULER reads `disk_restore` hot, so flipping it on a running
        # engine makes `lookup()` claim keys and `prepare_load` issue load jobs
        # -- while the WORKER still has no DISK->GPU handler, because handlers
        # register at construction. `OffloadingWorker.transfer_async` looks the
        # handler up by medium pair, finds None, and dies on a BARE assert,
        # taking the worker and then the engine core with it.
        #
        # So the second gate does not fail safe; it fails fatal. A gate whose
        # failure mode is a bare assert in a worker is not a safety feature,
        # and `lookup()` returning False is already structural: the scheduler
        # only ever calls `prepare_load` for keys lookup claimed, so with
        # restore off no file can reach a GPU tensor whether the handler exists
        # or not.
        #
        # *** AND IT FIXES A SECOND BUG FOR FREE: THE RANKS NO LONGER DISAGREE
        # ABOUT THEIR HANDLER SETS. *** Measured the same day, rank 1 built a
        # DiskLoadHandler and rank 0 did not, because each node read its own
        # (differently stale) flags file at boot. Registering unconditionally
        # makes the handler sets identical by construction, which no amount of
        # knob-syncing could guarantee -- the boot sync lands after
        # construction.
        #
        # The cost is one object per worker that may never be used. That is the
        # right price for a knob that is now genuinely hot on both sides.
        self._load_handler = DiskLoadHandler(
            spec=self, kv_caches=kv_caches,
            staging_blocks=self.staging_blocks,
            read_threads=self.write_threads, store=handler,
        )
        yield (DiskLoadStoreSpec, GPULoadStoreSpec, self._load_handler)
        logger.info(
            "kvmgr: DISK -> GPU handler registered unconditionally; "
            "`disk_restore` is now genuinely HOT because lookup() is the only "
            "gate. Currently %s. Withholding the handler used to be a second "
            "gate and it killed the engine instead -- see the note here.",
            "ON" if flags.peek("disk_restore") else "OFF")
