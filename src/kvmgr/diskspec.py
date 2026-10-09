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

from collections import OrderedDict
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


# Counters for upstream's completion ARRIVAL point. Module-level so a reload
# preserves them (and so the wrap is idempotent against the class).
# `installed` is the field that keeps the zeros honest: without it, "nobody is
# counting" and "nothing arrives" are the same reading, and they point the
# investigation in opposite directions.
_ARRIVAL = {"installed": False, "calls": 0, "calls_with_meta": 0,
            "job_ids_seen": 0, "stale_skipped": 0}


def install_arrival_counter() -> bool:
    """Count what actually reaches `update_connector_output`. Read-only.

    *** EVERY GUARD ON THE COMPLETION PATH IS SATISFIED AND THE COMPLETIONS
    *** STILL DO NOT ARRIVE, so reading more code cannot settle it.
    Established by inspection: the worker attaches the meta
    (`gpu/kv_connector.py` sets `output.kv_connector_worker_meta`), the
    aggregator carries it across ranks (`utils.py` lines 133-140 and 168), the
    aggregator IS initialised (`core.py:159`, guarded only on the connector
    existing), and the scheduler calls `update_connector_output`
    unconditionally (`scheduler.py:2318`). Measured: the worker has 0 unshipped
    completions and the scheduler has 125 of 135 jobs open with pending_count=2
    on 118.
    
    Those two sets of facts cannot both be true, so one of the calls is not
    happening at runtime. This counts the arrival point and separates the three
    possibilities in one reading:
    
      calls == 0                  -> update_connector_output is never reached
      calls > 0, with_meta == 0   -> it is reached and the meta is always None,
                                     i.e. the loss is in transit or aggregation
      with_meta > 0               -> the meta DOES arrive and the loss is in
                                     the processing (stale threshold, or a
                                     count that never sums to num_workers)
    
    Wraps rather than patches: no upstream file is edited, and `_kvmgr_wrapped`
    makes it idempotent across reloads.
    """
    try:
        import importlib
        m = importlib.import_module(
            "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler")
        cls = m.OffloadingConnectorScheduler
    except Exception:
        return False
    try:
        original = cls.update_connector_output
        if getattr(original, "_kvmgr_wrapped", False):
            return True

        def update_connector_output(self, connector_output):
            try:
                _ARRIVAL["calls"] += 1
                meta = getattr(connector_output, "kv_connector_worker_meta", None)
                jobs = getattr(meta, "completed_jobs", None) if meta else None
                if jobs:
                    _ARRIVAL["calls_with_meta"] += 1
                    _ARRIVAL["job_ids_seen"] += len(jobs)
                    thr = getattr(self, "_stale_job_threshold", 0) or 0
                    _ARRIVAL["stale_skipped"] += sum(
                        1 for j in jobs if j < thr)
            except Exception:
                pass
            return original(self, connector_output)

        update_connector_output._kvmgr_wrapped = True
        cls.update_connector_output = update_connector_output
        logger.info(
            "kvmgr: completion-arrival counter installed on "
            "update_connector_output -- /kv reports calls / calls_with_meta / "
            "job_ids_seen, which is what says whether the store completions "
            "reach the scheduler at all")
        return True
    except Exception:
        logger.exception("kvmgr: could not install the arrival counter")
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
            # If `installed` is False the counts mean NOTHING -- say so here
            # rather than letting a reader infer it from four zeros.
            "arrival": (dict(_ARRIVAL) if _ARRIVAL.get("installed") else
                        {"installed": False,
                         "why": "the arrival counter is not installed, so "
                                "these counts would be an ABSENCE rendered "
                                "as zero"}),
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


def choose_victims(items: "Sequence[tuple]", need_bytes: int) -> list:
    """Pick keys to evict until `need_bytes` is freed. PURE.

    `items` is `(key, nbytes, pos, recency)` in NO required order; `recency` is
    any comparable where SMALLER is older. Returns the victim keys in the order
    they should be dropped.

    *** THE VICTIM RULE IS THE WHOLE POINT OF PHASE 5, AND PLAIN LRU IS THE
    *** WRONG ANSWER. *** External reuse is a CONTIGUOUS RUN FROM TOKEN 0,
    exactly as GPU reuse is. So evicting a prefix's TAIL leaves a shorter
    prefix that still hits, while evicting its HEAD writes off every block
    behind it -- those bytes stay on disk and can never be read again. That is
    the lesson `evict_order=lru_tail_first` already paid for on the GPU side,
    measured: head-lost left 228 blocks reusable as 0, while tail-first kept
    495 of 495.

    So the order is **recency first, then TAIL-FIRST within an age**:

      - recency primary, because that is what a byte budget is for -- a cold
        prefix should go before a hot one, whatever its shape;
      - `-pos` secondary, so within one age the highest block index goes first.
        Keys of one prefix are touched together on reuse, so they share a
        recency and this orders them tail-to-head, which is what stops a head
        being orphaned.

    The happy consequence is that heads are evicted only once their own tail is
    already gone, so orphaning is rare BY CONSTRUCTION rather than by a second
    pass that would need prefix identity we do not have on disk.
    """
    if need_bytes <= 0:
        return []
    # Oldest first; within an age, the furthest from token 0 first.
    ordered = sorted(items, key=lambda it: (it[3], -it[2]))
    out, freed = [], 0
    for key, nbytes, _pos, _rec in ordered:
        out.append(key)
        freed += int(nbytes)
        if freed >= need_bytes:
            break
    return out


def budget_bytes(budget_gb: float) -> int:
    """GB -> bytes, 0 meaning NO BUDGET. Pure.

    0 is "unbounded" rather than "evict everything", which is the direction a
    misread must fail in: an unset knob must not wipe 518 GB of index. The
    units are GB because that is how the disk is discussed, and the conversion
    is here rather than at three call sites.
    """
    try:
        gb = float(budget_gb)
    except Exception:
        return 0
    return int(gb * 1e9) if gb > 0 else 0


class DiskLoadStoreSpec(LoadStoreSpec):
    """Addresses blocks by their OffloadKey, NOT by a block id.

    An OffloadKey is plain `bytes` packing (block_hash, group_idx), so this whole
    object serialises across the engine-core -> worker IPC with no help. That is
    what lets ONE spec, built once scheduler-side, reach every rank -- each of
    which then appends its OWN rank to the path. The keys are deliberately
    rank-agnostic: rank lives in the path and outside the fingerprint, which is
    precisely what makes `_r0` and `_r1` siblings under one fingerprint.
    """

    def __init__(self, keys: list["OffloadKey"], to_delete: list | None = None):
        self.keys = list(keys)
        # *** THE DELETE CHANNEL. *** Each rank writes its own slice to its own
        # node's disk, so an eviction must reach both -- and upstream never
        # reads `evicted_keys`, while a collective_rpc from the scheduler step
        # is the blocking mistake that has already cost an NCCL kill. This
        # object already serialises to every rank with no help, so it carries
        # the delete. Plain bytes, deliberately: a numpy scalar in a serialised
        # structure killed the engine's output thread earlier tonight.
        self.to_delete = [bytes(k) for k in (to_delete or [])]

    @staticmethod
    def medium() -> str:
        return DISK_MEDIUM

    def __repr__(self) -> str:
        return (f"DiskLoadStoreSpec({len(self.keys)} keys"
                f"{', %d to delete' % len(self.to_delete) if self.to_delete else ''})")


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
    # How often a steady-state eviction summary may be logged.
    EVICT_LOG_S = 60.0
    # And a time floor, because a session adding fewer than 512 keys would
    # otherwise never flush at all. See flush_manifest for the measurement.
    MANIFEST_FLUSH_S = 30.0

    def __init__(self, file_mapper: FileMapper, num_workers: int,
                 manifest_path: str | None = None):
        self._mapper = file_mapper
        self._num_workers = num_workers
        # *** AN ORDERED MAP, NOT A SET, AND THE ORDER IS THE LRU. *** Phase 5
        # needs three things a set cannot carry: a byte cost (to enforce a
        # budget), a block POSITION (so eviction can be tail-first), and a
        # recency (so a cold prefix goes before a hot one). Insertion order is
        # the recency and `move_to_end` on touch is the update, which is why
        # this is an OrderedDict rather than a plain dict: `move_to_end` is the
        # whole mechanism and a plain dict has no such method.
        #
        # Membership (`k in self._present`) and `len()` read identically to the
        # set this replaces, so every existing call site is unchanged.
        self._present: "OrderedDict[bytes, tuple[int, int]]" = OrderedDict()
        self._bytes = 0            # running total, so the budget needs no stat
        self._evicted_pending: list[bytes] = []
        # Keys with a load in flight, which must never be evicted. Set here
        # for the normal path AND created lazily by _loading_map(), because a
        # hot reload rebinds a live manager without running __init__.
        self._loading: dict[bytes, int] = {}
        # key -> block position, recorded when OFFERED and consumed at
        # completion. Bounded by what is in flight, and cleared on reset.
        self._pos_hint: dict[bytes, int] = {}
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

    def _ensure_index(self) -> None:
        """Migrate a REBOUND manager's state in place. Idempotent, cheap.

        *** A RELOAD NEVER RUNS __init__ -- SIXTH TIME, and this one is worse
        *** than the other five. *** They were MISSING attributes; Phase 5 also
        CHANGED A TYPE: `_present` went from a `set` to an `OrderedDict` so it
        could carry a byte cost and a block position. A rebound live manager
        keeps the old set, so `move_to_end` does not exist and `_bytes` was
        never assigned. Measured on the box, fatal to the scheduler:

            AttributeError: 'DiskPresenceManager' object has no attribute
            '_bytes'

        A getattr at each site would be six more chances to miss one, so the
        migration is ONE function called at the top of every entry point, and a
        test drives every public method against a deliberately legacy-shaped
        instance. *** A STATE TYPE CHANGE IS NOT HOT-RELOADABLE BY DEFAULT ***
        -- this is what makes it so.
        """
        pres = getattr(self, "_present", None)
        if not isinstance(pres, OrderedDict):
            migrated = OrderedDict()
            self._present = migrated
            self._bytes = 0
            for k in (pres or ()):        # a legacy set: order is unknown
                kb = bytes(k)
                nb = self._key_bytes(kb)
                # Position -1 means UNKNOWN, which sorts before 0 under the
                # tail-first tie-break and so makes these the LAST victims of
                # their age. Conservative on purpose: any of them might be a
                # prefix head, and evicting a head orphans everything behind it.
                migrated[kb] = (nb, -1)
                self._bytes += nb
            if migrated:
                logger.info(
                    "kvmgr: migrated a legacy presence set of %d keys to the "
                    "ordered index (%.1f GB); positions unknown, so they are "
                    "the last victims of their age",
                    len(migrated), self._bytes / 1e9)
        if not hasattr(self, "_bytes"):
            self._bytes = sum(nb for nb, _p in self._present.values())
        # *** AND A VALUE-LEVEL REPAIR, NOT JUST A TYPE-LEVEL ONE. *** The
        # first accounting fix was deployed hot onto an index that was ALREADY
        # an OrderedDict -- built by the previous, broken version -- so the
        # type migration above found nothing to do and every entry kept the
        # cost 0 it had been given by code that could never compute one.
        # Measured after that reload: 523,194 keys, held 0.0 GB,
        # byte_accounting_ok False.
        #
        # A migration has to repair bad VALUES as well as a bad shape, or a fix
        # only takes effect for keys stored after it. Cheap: _key_bytes caches
        # per group, so this is five stats and a dictionary walk however many
        # keys there are.
        if self._present and self._bytes <= 0:
            total = 0
            for k, (nb, pos) in list(self._present.items()):
                if nb <= 0:
                    nb = self._key_bytes(k)
                    self._present[k] = (nb, pos)
                total += nb
            self._bytes = total
            logger.info(
                "kvmgr: re-costed %d index entries that carried no byte cost "
                "(%.1f GB) -- they were recorded by a version that read the "
                "group sizes from the worker, which this process is not",
                len(self._present), total / 1e9)
        if not hasattr(self, "_pos_hint"):
            self._pos_hint = {}
        if not hasattr(self, "_evicted_pending"):
            self._evicted_pending = []
        if not hasattr(self, "_gcounts"):
            self._gcounts = {}

    def _key_bytes(self, key: bytes) -> int:
        """On-disk cost of one key, from its GROUP -- never a stat() call.

        The five groups have measured, different per-block file sizes
        (985,664 / 822,272 / 784,896 / 860,160 / 655,360 B), so a uniform
        estimate would be wrong by up to 50% and the budget would be a
        fiction. Derived from the group index packed into the key itself, so
        this costs no I/O on the scheduler's path -- which matters, because it
        is called for every key of every store.
        """
        try:
            from vllm.v1.kv_offload.base import get_offload_group_idx
            g = int(get_offload_group_idx(key))
        except Exception:
            return 0
        cache = getattr(self, "_group_bytes", None)
        if cache is None:
            cache = {}
            self._group_bytes = cache
        if g in cache:
            return cache[g]
        # *** STAT THE FILE. The first version asked the live HANDLER for its
        # *** sizes, and the handler lives in the WORKER while this manager
        # lives in the ENGINE CORE -- so `_LIVE_HANDLERS` was always empty here
        # and every key cost 0. Measured on the box: 521,628 keys reporting
        # "0.0 GB held", a budget of 1 GB, and `over_budget: False`. The
        # accounting was a fiction and the budget could never fire.
        #
        # One stat per GROUP, ever, cached: the five groups have fixed and
        # different per-block sizes (985,664 / 822,272 / 784,896 / 860,160 /
        # 655,360 B), the files are on this node by definition, and a uniform
        # estimate would be up to 50% wrong. Five syscalls for an exact answer
        # beats importing worker state that does not exist here.
        try:
            import os as _os
            size = int(_os.path.getsize(self._mapper.get_file_name(key)))
            if size > 0:
                cache[g] = size
                return size
        except Exception:
            pass
        return 0

    def _remember(self, key: bytes, pos: int) -> None:
        """Record a key as present, with its cost and position. Idempotent."""
        kb = bytes(key)
        if kb in self._present:
            return
        nb = self._key_bytes(kb)
        self._present[kb] = (nb, int(pos))
        self._bytes += nb

    def _forget(self, key: bytes) -> int:
        """Drop a key from the index, returning the bytes reclaimed."""
        nb, _pos = self._present.pop(bytes(key), (0, -1))
        self._bytes -= nb
        if self._bytes < 0:
            self._bytes = 0
        return nb

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
                raw = json.load(f).get("keys", [])
            # *** BOTH FORMATS, AND THE OLD ONE IS 520,446 KEYS ON DISK RIGHT
            # *** NOW. *** Phase 5 needs a block POSITION per key, so entries
            # became `[hex, pos]`. A reader that only understood the new shape
            # would throw away the entire existing index -- 518 GB of files
            # that are still perfectly restorable -- and the loss would look
            # like a cache that simply went cold. So a bare string is still
            # accepted and given pos=-1, meaning "position unknown".
            #
            # pos=-1 sorts BEFORE 0 under the `-pos` tail-first tie-break,
            # which makes unknown-position keys the LAST victims within their
            # age rather than the first. That is the conservative direction: a
            # key we cannot place in a prefix might be a head, and evicting a
            # head orphans everything behind it.
            keys, poss = [], []
            for item in raw:
                if isinstance(item, str):
                    keys.append(bytes.fromhex(item)); poss.append(-1)
                else:
                    keys.append(bytes.fromhex(item[0])); poss.append(int(item[1]))
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
        self._present = OrderedDict()
        self._bytes = 0
        _pos_by_key = dict(zip(keys, poss))
        for k in kept:
            self._remember(k, _pos_by_key.get(k, -1))
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
                # [hex, pos] pairs. Sorted by key so a diff of two manifests
                # is readable; the LRU order is deliberately NOT persisted,
                # because a restart's recency is meaningless and pretending
                # otherwise would evict by the age of a previous boot.
                json.dump({"keys": sorted([k.hex(), p]
                                          for k, (nb, p) in self._present.items())}, f)
            _os.replace(tmp, self._manifest)
            self._unflushed = 0
            self._last_flush = _now()
            self._stats["manifest_writes"] += 1
        except Exception:
            logger.exception("kvmgr: manifest flush failed (presence kept in "
                             "memory; a restart will simply forget it)")

    # -- reads ------------------------------------------------------------
    def lookup(self, key: "OffloadKey", req_context: "ReqContext") -> bool | None:
        self._ensure_index()
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

    def _loading_map(self) -> dict:
        """The in-flight-load holds, created on first use.

        *** A RELOAD NEVER RUNS `__init__`. *** `rebind` re-points a live
        manager with `obj.__class__ = new`, so an instance already serving
        requests gets every new METHOD and no new attribute -- and an
        AttributeError in prepare_store is fatal to the scheduler. This repo
        has paid for that six times now, once as a type change rather than a
        missing name. So the attribute is never assumed: it is created here, on
        the INSTANCE (a class-level dict would be shared by every manager), and
        every reader goes through this one accessor.
        """
        m = getattr(self, "_loading", None)
        if m is None:
            m = {}
            self._loading = m
        return m

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
        # *** A KEY WITH A LOAD IN FLIGHT MUST NOT BE EVICTED, AND THE
        # *** FRAMEWORK CANNOT SURVIVE IT IF IT IS. ***
        # `offloading/worker.py:273` is a bare `assert transfer_result.success`
        # under the comment "we currently do not support job failures" -- so a
        # load that cannot find its file does not degrade, it raises
        # AssertionError in the worker's output path. Phase 5's eviction made
        # that reachable for the first time: lookup() says present, the load is
        # issued, a concurrent store's budget picks the same key as a victim,
        # _apply_deletes unlinks it on both ranks, and the read hits
        # FileNotFoundError. The LRU touch makes it unlikely, because a
        # just-looked-up key is the most recently used and so the last victim
        # -- but UNLIKELY IS NOT A SAFETY PROPERTY, and the failure is a dead
        # engine rather than a miss.
        loading = self._loading_map()
        for k in keys:
            kb = bytes(k)
            loading[kb] = loading.get(kb, 0) + 1
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
        self._ensure_index()
        if not _LIVE_OFFLOAD_SCHED:
            capture_offload_sched_from_caller()
        # *** INSTALLED UNCONDITIONALLY, and gating it was a real bug. ***
        # It used to sit inside the capture's one-shot, so once the handle was
        # preserved across a reload the capture branch never ran again and the
        # counter was never installed -- and it then reported `calls: 0`, which
        # reads as "update_connector_output is never reached" when the truth was
        # "nobody was counting". Absent rendered as zero, inside the instrument
        # built to avoid exactly that. It is idempotent via _kvmgr_wrapped, so
        # calling it every time costs one getattr.
        if not _ARRIVAL.get("installed"):
            _ARRIVAL["installed"] = install_arrival_counter()

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

        # *** RECORD THE POSITION NOW, because this is the only moment it is
        # *** known. *** The scheduler hands keys in block order within a
        # group -- the same input-order invariant check_pairing depends on --
        # so the i-th key of a group is block i of that prefix. By the time
        # complete_store fires, that ordering is gone. Tail-first eviction is
        # impossible without it.
        try:
            from vllm.v1.kv_offload.base import get_offload_group_idx
            _seen: dict[int, int] = {}
            for k in todo:
                g = get_offload_group_idx(k)
                i = _seen.get(g, 0)
                _seen[g] = i + 1
                self._pos_hint[bytes(k)] = i
        except Exception:
            pass

        evicted = self._enforce_budget(todo)
        return PrepareStoreOutput(
            keys_to_store=todo,
            store_spec=DiskLoadStoreSpec(todo, to_delete=evicted),
            # *** UPSTREAM IGNORES THIS FIELD -- checked, it is declared in
            # PrepareStoreOutput and never read in scheduler.py. *** It is
            # returned anyway because it is the documented contract and a
            # future version may honour it; the delete that actually happens
            # travels in the store_spec above, which reaches BOTH ranks.
            evicted_keys=list(evicted),
        )

    def _enforce_budget(self, incoming: "Sequence[OffloadKey]") -> list:
        """Evict until the budget holds. Returns the keys to delete.

        *** WHY THE DELETE RIDES THE STORE SPEC. *** Each rank writes its own
        slice to its OWN node's disk, so a delete has to reach both -- and
        `evicted_keys` is never read by upstream, while a `collective_rpc`
        from inside the scheduler step is precisely the blocking-the-critical-
        path mistake that has already cost this project an NCCL kill. Our own
        `DiskLoadStoreSpec` already serialises to every rank with no help, and
        a store job always accompanies an eviction because eviction is what
        makes room for that store. So the existing channel carries it.
        
        *** AND EVICTION IS CLUSTER-WIDE BY NECESSITY, NOT TIDINESS. *** If one
        rank dropped a key the other kept, a later restore would load half the
        KV and nothing would crash -- the F8_E8M0 hazard. Per-node budgets
        would be simpler and are not safe.
        """
        budget = budget_bytes(flags.peek("disk_budget_gb"))
        if not budget:
            return []
        incoming_bytes = sum(self._key_bytes(bytes(k)) for k in incoming)
        need = (self._bytes + incoming_bytes) - budget
        if need <= 0:
            return []
        # Never consider what we are about to store as a victim.
        # Never evict a key the scheduler is already loading -- see
        # prepare_load. Read through _loading_map(), because a hot reload
        # rebinds the class without running __init__, so a manager already
        # serving requests gets the new METHODS and not the new attribute.
        skip = {bytes(k) for k in incoming}
        skip |= set(self._loading_map())
        items = [(k, nb, pos, i)
                 for i, (k, (nb, pos)) in enumerate(self._present.items())
                 if k not in skip]
        victims = choose_victims(items, need)
        freed = 0
        for v in victims:
            freed += self._forget(v)
        if victims:
            self._stats["evicted"] = self._stats.get("evicted", 0) + len(victims)
            self._stats["evicted_bytes"] = (
                self._stats.get("evicted_bytes", 0) + freed)
            self._events.append(OffloadingEvent(
                keys=list(victims), medium=DISK_MEDIUM, removed=True))
            self._unflushed += len(victims)
            # *** THROTTLED, BECAUSE AT STEADY STATE THIS IS NOT NEWS. ***
            # Measured: once the index is at budget every store evicts ~90 keys
            # to make room for the same bytes, so an unthrottled line is one
            # per store round -- about one a second, for ever. That is a
            # ROUTINE event logged as though it were notable, which trains a
            # reader to skip exactly the channel the interesting first eviction
            # arrives on ("say a thing once").
            #
            # So: the first eviction after crossing the budget logs in full,
            # because dropping 100,665 keys / 71.8 GB genuinely is an event;
            # after that it is a periodic SUMMARY carrying the totals, which is
            # what somebody watching a steady-state cache actually wants.
            now = _now()
            first = not getattr(self, "_evict_logged", False)
            due = (now - getattr(self, "_evict_log_at", 0.0)) >= self.EVICT_LOG_S
            if first or due:
                self._evict_logged = True
                self._evict_log_at = now
                since = self._stats.get("evicted", 0) - getattr(
                    self, "_evict_log_count", 0)
                self._evict_log_count = self._stats.get("evicted", 0)
                logger.info(
                    "kvmgr: disk budget -- evicted %d key(s) / %.2f GB just now"
                    "%s; held %.2f GB against a %.2f GB budget, %d keys / "
                    "%.1f GB evicted in total. Tail-first within an age, so a "
                    "prefix shortens rather than losing its head; the delete "
                    "rides the store spec to BOTH ranks.",
                    len(victims), freed / 1e9,
                    "" if first else " (%d since the last line)" % since,
                    self._bytes / 1e9, budget / 1e9,
                    self._stats.get("evicted", 0),
                    self._stats.get("evicted_bytes", 0) / 1e9)
        return victims

    def complete_store(
        self, keys: "Collection[OffloadKey]", req_context: "ReqContext"
    ) -> None:
        self._ensure_index()
        new = [k for k in keys if bytes(k) not in self._present]
        for k in new:
            # The position was recorded when the key was OFFERED, because that
            # is the only moment it is known: the scheduler hands keys in block
            # order within a group, which is the same invariant check_pairing
            # relies on. By completion time that ordering is gone.
            self._remember(bytes(k), self._pos_hint.pop(bytes(k), -1))
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
        """Release the eviction hold `prepare_load` took.

        Counted rather than a plain set, because two requests can have a load
        in flight for the same key and the first completion must not release
        the second's hold.
        """
        loading = self._loading_map()
        for k in keys:
            kb = bytes(k)
            n = loading.get(kb, 0) - 1
            if n > 0:
                loading[kb] = n
            else:
                loading.pop(kb, None)
        return None

    def touch(self, keys: "Collection[OffloadKey]", req_context: "ReqContext") -> None:
        """Mark keys as recently used. *** THE LRU, AND IT WAS A NO-OP. ***

        The scheduler calls this when a lookup reuses a prefix, so it is the
        only reuse signal that reaches this layer -- without it a byte budget
        could only evict by insertion age, which would throw away the hottest
        long conversation on the box simply because it started first. That is
        the opposite of what this cache is for.

        `move_to_end` on an OrderedDict is the whole mechanism. Touching keys
        of one prefix together is what gives them a shared recency, which is
        what makes the tail-first tie-break in choose_victims meaningful.
        """
        self._ensure_index()
        for k in keys:
            kb = bytes(k)
            if kb in self._present:
                self._present.move_to_end(kb)
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
        self._bytes = 0
        self._pos_hint.clear()
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
        self._ensure_index()
        by_group = {
            what: {str(g): n for g, n in sorted(d.items())}
            for what, d in sorted(getattr(self, "_gcounts", {}).items())
        }
        _budget = budget_bytes(flags.peek("disk_budget_gb"))
        # *** 0 BYTES HELD WITH KEYS PRESENT IS AN ABSENCE, NOT A MEASUREMENT.
        # *** That exact reading -- 521,628 keys at "0.0 GB" -- was the broken
        # accounting above, and it made a 1 GB budget report `over: False`. A
        # budget computed from an unknown total must say so, because silently
        # never firing is the worst of the three possible behaviours.
        _sized = bool(getattr(self, "_group_bytes", None))
        _accounting_ok = (not self._present) or (self._bytes > 0 and _sized)
        return dict(self._stats, present=len(self._present),
                    bytes_held=self._bytes,
                    budget_bytes=_budget,
                    budget_gb=round(_budget / 1e9, 2) if _budget else 0,
                    over_budget=bool(_budget and self._bytes > _budget),
                    byte_accounting_ok=_accounting_ok,
                    group_bytes_known=sorted(
                        getattr(self, "_group_bytes", {}) or {}),
                    budget_enforced=bool(_budget),
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
