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
# *** PHASE 4: THE VERSION OF *OUR* ON-DISK FORMAT. ***
# Bumping this orphans every file written by an older format, by construction,
# because it lands in a different directory AND a different fingerprint hash.
# That is the whole point: KV bytes are only valid for the engine AND the writer
# that produced them, and a restore across a changed layout returns plausible
# tensors that decode to nonsense with nothing crashing -- the F8_E8M0 failure
# class. A refusal has to be structural, not remembered.
#
# *** AND THE DANGEROUS CASE IS THE ACCIDENTAL CHANGE, NOT THE DELIBERATE ONE.
# *** Anybody bumping this constant has already thought about it. The failure to
# design against is somebody editing `file_plan`, `write_chunks` or the chunk
# geometry and NOT bumping it -- after which every pre-change file restores
# silently as wrong KV. A runtime hash of the writer's own source would catch
# that, and would also orphan 400 GB the first time anybody fixed a typo in a
# comment. So the guard is OFFLINE: `test_phase1_disk.py` pins a hash of the
# layout-determining code with comments and docstrings stripped, and FAILS if it
# moves without this constant moving too. Zero runtime cost, no mass orphaning
# on a cosmetic edit, and the failure arrives on a laptop instead of on disk.
DISK_FORMAT_VERSION = 1
DISK_FORMAT_ROOT = "dspark-kv-v%d" % DISK_FORMAT_VERSION

# The functions whose source decides WHICH BYTES LAND IN WHICH FILE. The offline
# guard hashes exactly these; anything else in diskio.py may change freely.
# Keeping the list here rather than in the test means the module being guarded
# declares its own surface, so adding a layout function and forgetting the guard
# shows up as a NAMED test failure rather than as silent coverage loss.
DISK_LAYOUT_FUNCTIONS = ("file_plan", "write_chunks", "plan_nbytes")

def model_dir(model: str, revision: "str | None" = None,
              hub: "str | None" = None) -> "str | None":
    """The local directory the model is actually loaded from, or None.

    Either a plain path, or a huggingface_hub cache entry resolved the way the
    hub itself resolves it: `models--<org>--<name>/refs/<rev>` names a commit,
    and the commit names a `snapshots/<commit>` directory.
    """
    import os as _os
    if model and _os.path.isdir(model):
        return model
    if not model:
        return None
    # *** THE FALLBACK ORDER MATTERS, AND MY FIRST VERSION WAS BOX-SPECIFIC. ***
    # It ended at "/models/hub", which is THIS deployment's bind mount -- so on
    # any other machine the lookup would miss and the fingerprint would quietly
    # degrade to `unknown`, i.e. the protection would silently not be there.
    # huggingface_hub's own default is ~/.cache/huggingface/hub, so that comes
    # first and the local convention is merely the last resort. Found by
    # re-reading my own code for portability rather than by a failure.
    candidates = [hub, _os.environ.get("HF_HUB_CACHE")]
    if _os.environ.get("HF_HOME"):
        candidates.append(_os.path.join(_os.environ["HF_HOME"], "hub"))
    candidates.append(
        _os.path.join(_os.path.expanduser("~"), ".cache", "huggingface", "hub"))
    candidates.append("/models/hub")

    repo = None
    for root in candidates:
        if not root:
            continue
        cand = _os.path.join(root, "models--" + model.replace("/", "--"))
        if _os.path.isdir(cand):
            repo = cand
            break
    if repo is None:
        return None
    commit = revision
    if commit:
        # A pinned revision may still be a ref name rather than a commit.
        ref = _os.path.join(repo, "refs", commit)
        if _os.path.isfile(ref):
            try:
                commit = open(ref).read().strip()
            except OSError:
                return None
    else:
        ref = _os.path.join(repo, "refs", "main")
        if not _os.path.isfile(ref):
            return None
        try:
            commit = open(ref).read().strip()
        except OSError:
            return None
    snap = _os.path.join(repo, "snapshots", commit)
    return snap if _os.path.isdir(snap) else None


def model_content_id(d: "str | None", _hash=None) -> str:
    """A 12-hex identity for the model's KV-RELEVANT CONTENT, or `unknown`.

    *** THE HF COMMIT IS THE WRONG THING TO FINGERPRINT, and this box proves
    *** it. *** `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` has TWO snapshots in
    the hub cache, and measured 2026-10-09 they share all 49 safetensors blob
    targets and a byte-identical `config.json` -- they differ in `LICENSE`,
    `README.md`, `encoding/` and `inference/`. So the KV written under one is
    byte-valid under the other, and a commit-keyed fingerprint would have
    orphaned 400 GB of warm cache to protect against a docs-only revision bump.
    That is a false positive expensive enough to design away.

    What actually determines KV validity is the WEIGHTS and the CONFIG, so that
    is what this hashes: `config.json` verbatim, plus the (filename -> blob)
    mapping of every safetensors entry. Reading the blob NAMES costs nothing --
    they are content-addressed by the hub, so the mapping changes exactly when
    the weights change. No weight bytes are read.

    *** `unknown` IS NOT A REFUSAL, and that is deliberate. *** If the layout is
    one this cannot read, the identity degrades to the model NAME alone -- which
    is exactly today's behaviour, so the change is never a regression. It is
    logged and exposed in /kv rather than hidden, because the honest statement
    is "this cache is no better protected than before", not "it is protected".
    """
    import hashlib as _hl
    import os as _os
    if not d:
        return "unknown"
    h = _hash or _hl.sha256()
    try:
        cfg = _os.path.join(d, "config.json")
        if _os.path.isfile(cfg):
            with open(cfg, "rb") as f:
                h.update(b"config.json\0")
                h.update(f.read())
        rows = []
        for name in sorted(_os.listdir(d)):
            if ".safetensors" not in name:
                continue
            full = _os.path.join(d, name)
            try:
                target = _os.path.basename(_os.readlink(full))
            except OSError:
                # A real file rather than a hub symlink: size is the cheapest
                # content proxy that does not read the weights.
                try:
                    target = "size:%d" % _os.path.getsize(full)
                except OSError:
                    target = "?"
            rows.append("%s=%s" % (name, target))
        if not rows:
            return "unknown"
        h.update(("\0".join(rows)).encode("utf-8"))
    except OSError:
        return "unknown"
    return h.hexdigest()[:12]


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


def capture_engine_core_from_caller(depth: int = 30) -> "Any | None":
    """Grab the EngineCore off the CALL STACK, at manager construction.

    *** THIS IS WHAT MAKES BOOT-TIME CROSS-RANK VERIFICATION POSSIBLE, and the
    *** absence of it is why the first design verified lazily and broke the
    *** restore. *** `DiskPresenceManager.__init__` holds no executor
    reference, which was read as "the executor is not reachable there". It is:
    `EngineCore.__init__` assigns `self.model_executor` at core.py:121 and only
    then builds the Scheduler at :148, which is what constructs the connector
    and therefore us. So the engine core is sitting in a caller frame with a
    live executor on it, exactly as the connector scheduler is for
    `capture_offload_sched_from_caller`.

    *** MATCHED ON CAPABILITY, NOT ON A CLASS NAME. *** `EngineCoreProc`
    subclasses `EngineCore`, so a name match needs a list that an upstream
    rename or a new subclass silently invalidates. What this needs is
    `collective_rpc` and a `model_executor`, so that is what it asks for -- and
    anything with both is, by definition, something it can verify through.

    The frame is deeper than the offload scheduler's: core -> Scheduler ->
    KVConnectorFactory -> connector -> spec.get_manager -> our __init__ ->
    _load_manifest. Hence depth 30 rather than 5.
    """
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
            if cand is None:
                continue
            if callable(getattr(cand, "collective_rpc", None)) and \
                    getattr(cand, "model_executor", None) is not None:
                return cand
    except Exception:
        return None
    finally:
        # Frames pin their locals; holding one would keep the whole boot's
        # object graph alive.
        del f
    return None


# *** THE BOOT RPC'S TIMEOUT IS A CODE CONSTANT AND IT IS GENEROUS. ***
# A collective_rpc timeout is unrecoverable at any time -- it leaves the reply
# unclaimed on an ordered response queue -- and at BOOT it is unrecoverable at
# the worst possible moment, because the engine has not served a request yet
# and the desync would land on the first one. Measured: a 4,000-key chunk is
# ~1.6s even with the start script dropping the dentry cache every 2s, so this
# is ~190x headroom. It is not a knob because there is no operational reason to
# lower it: a chunk that takes five minutes means a rank is gone, which is what
# the number exists to detect.
BOOT_VERIFY_TIMEOUT_S = 300.0


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


def pending_block_ids():
    """GPU block ids with a KV transfer in flight, or None if unknown.

    *** THIS IS ON THE PROTECT HOT PATH, so it must be O(1) and allocate
    *** nothing. *** It returns the connector's own `_block_id_to_pending_jobs`
    KEYS VIEW rather than a copy -- membership is O(1) against a live dict, and
    copying a few hundred ids on every protect would be a cost paid thousands of
    times a minute for a set that changes constantly.

    Returning the live view does mean the caller sees mutations mid-iteration.
    That is fine for the one use: a membership test per candidate block, where a
    concurrent insert simply means the next protect skips that block instead of
    this one.

    *** None MEANS "I CANNOT SAY", NOT "NOTHING IS PENDING". *** The caller
    fails OPEN on None (it releases), which is the right trade -- a cap that
    cannot be enforced means memory pressure and preemptions -- but the two must
    stay distinguishable at this boundary rather than collapsing into an empty
    set that reads as a measurement.
    """
    cs = _LIVE_OFFLOAD_SCHED[0] if _LIVE_OFFLOAD_SCHED else None
    if cs is None:
        return None
    d = getattr(cs, "_block_id_to_pending_jobs", None)
    if d is None:
        return None
    return d.keys()


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


# ---------------------------------------------------------------------------
# THE PRESENCE JOURNAL -- the pure half
# ---------------------------------------------------------------------------
# WHY THIS EXISTS. The manifest is O(index) to persist O(changes): 772,561 keys
# rewritten for the ~71 a single response adds. Measured on the live box under
# a real store load, that saturates: `manifest_writes 109` against
# `manifest_skipped_busy 462`, so 81% of flush attempts were refused because a
# write was already in flight. A refused flush is not merely slow -- it is a
# window in which a key is on disk with nothing able to NAME it, which is
# precisely how 155 GiB of unreachable orphans accumulated before anybody was
# counting. So this is a durability fix that happens to also be a cost fix,
# and not the other way round.
#
# WHAT IT IS NOT. It does NOT shrink the boot walk. Every key's file is still
# `stat`ed at startup because `offloading/worker.py:273` asserts on a failed
# load, and that is independent of the index's file FORMAT. Anybody sizing the
# 256s boot against this is sizing against nothing.
#
# These four functions are pure on purpose: this repo's rule is that anything
# which could silently corrupt KV must be testable with no GPU and no torch.
# The manager's I/O is a thin shell over them.

JOURNAL_ADD = "a"
JOURNAL_REMOVE = "r"
JOURNAL_CLEAR = "c"

_JMISSING = object()


# ---------------------------------------------------------------------------
# CROSS-RANK VERIFICATION AND THE FREE-SPACE FLOOR -- the pure halves
# ---------------------------------------------------------------------------
# Both mechanisms exist for one reason: `offloading/worker.py:273` is a bare
# `assert transfer_result.success` under the comment "we currently do not
# support job failures". So a missing file on ONE node, a truncated file, or a
# full disk is not a miss -- it is a dead engine core. "Handle the error
# gracefully" is not on the menu; a hazard here is closed only by making that
# assert UNREACHABLE.
#
# The decision logic is pure and lives here, for the reason everything
# dangerous in this plugin is pure: it is the half that can be wrong in a way
# nothing crashes on, and it has to be testable with no GPU, no torch and no
# filesystem.

# *** A RANK THAT CANNOT ANSWER, and "I cannot tell" must never read as
# *** "nothing wrong". *** A missing rank, an exception or a timeout records a
# persistent fault here and presence REFUSES for the rest of the boot. That is
# the drop guard's precedent: for a lever whose failure mode is a dead engine,
# unreadable has to be treated as bad. It is a module global so a hot reload
# cannot silently clear a fault -- see control._PRESERVE.
_VERIFY_FAULT: "dict | None" = None

# *** THE OPERATOR'S RE-VERIFY REQUEST, AND IT IS A CROSS-THREAD HANDOFF WITH
# *** EXACTLY ONE WRITER ON EACH SIDE. ***
#
# The lever belongs in THIS process -- the engine core is where the index is --
# and the only HTTP surface this process owns is `kvmgr.serve`'s listener,
# which answers on its own daemon thread. So the request arrives on a thread
# that must not touch the manager: `_wanted` and `_verified` are mutated by
# `lookup` and `pump` on the SCHEDULER's thread, and a second writer in those
# is a data race in the structures that decide whether a key is claimed. That
# is the lesson the stall watchdog already paid for -- it raises a request and
# never writes `_jobs`.
#
# So the listener only ever REBINDS this to a new dict (one store, so a reader
# sees the whole request or none of it) and the pump takes it on the scheduler's
# thread. Same shape as `gpu_cache_epoch`, which is acted on at the edge in
# `new_step_starts()` rather than where it is set.
#
# *** AND THE STEP ONLY RUNS WHEN THERE ARE REQUESTS, so a request posted to an
# *** IDLE engine sits here unapplied. *** That is upstream's own documented
# behaviour (`_process_engine_step`: "Called only when there are unfinished
# local requests") and it is why the route answers 202 with a sequence number
# rather than a result: the caller confirms by reading `verify.last_request`
# back off GET /kv, which is the only honest way to report work that happens
# somewhere else later.
_VERIFY_REQUEST: "dict | None" = None

# The outcome of the last request the pump applied, so a 202 is checkable.
# Without this the route would be unfalsifiable: nothing else distinguishes
# "applied and there was nothing to do" from "never ran because the engine is
# idle".
_VERIFY_LAST: "dict | None" = None

# Monotonic, so two requests are distinguishable even inside one second.
_VERIFY_SEQ = 0


def request_verify(requeue_all: bool = False,
                   clear_fault: bool = False) -> dict:
    """Ask the pump for a re-verification. *** CALLED FROM THE LISTENER THREAD.

    *** IT TOUCHES NO MANAGER STATE, DELIBERATELY. *** Everything it does is
    rebind the two module globals above; the work happens on the scheduler's
    thread in `_apply_verify_request`. Keep it that way -- the whole safety
    argument for having a mutating route in the engine core at all is that the
    mutation happens in one place, on the thread that owns the structures.

    Returns what was accepted, for the 202. It cannot report an outcome,
    because there is not one yet.
    """
    global _VERIFY_REQUEST, _VERIFY_SEQ

    _VERIFY_SEQ += 1
    req = {"seq": _VERIFY_SEQ,
           "all": bool(requeue_all),
           "clear_fault": bool(clear_fault),
           "at": _now()}
    _VERIFY_REQUEST = req  # one store, never a partially built request
    return dict(req)

# The absolute floor, in code and not a knob. Below this the tier stops
# admitting even the single key that carries the unlinks: at ~1 MB per key per
# rank, one key cannot rescue a filesystem this close to full, and the write it
# would attempt is the ENOSPC we are avoiding.
FREE_ABSOLUTE_MIN_BYTES = 1 << 30

BAND_OK = "ok"
BAND_DEGRADED = "degraded"
BAND_FROZEN = "frozen"


def free_band(free: int, floor: int, hard: int, hyst: int,
              prev: str = BAND_OK) -> str:
    """Which free-space band are we in? Pure, with hysteresis.

    *** HYSTERESIS IS NOT POLISH HERE. *** Without it a tier sitting exactly at
    the floor flaps band on every store round, and the band EDGE is what gets
    logged -- so a flapping tier is also a log flood, which is how a reader
    learns to ignore the one transition that matters.

    `free <= 0` is treated as FROZEN rather than as "unknown": a zero from a
    real statvfs means a full disk. An UNREADABLE statvfs never reaches here --
    the caller treats that as OK and says so loudly, which is the opposite of
    the drop guard's choice and for the opposite reason: there "unknown" risks
    a dead worker, here it risks only a cold cache.
    """
    if free <= hard:
        return BAND_FROZEN
    if free < floor:
        return BAND_DEGRADED
    # Coming back up needs to clear the floor by the hysteresis, and a frozen
    # tier has to clear the DEGRADED gate as well -- it cannot jump to ok.
    if prev in (BAND_DEGRADED, BAND_FROZEN) and free < floor + hyst:
        return BAND_DEGRADED
    return BAND_OK


def verify_verdict(results: "Sequence[dict]", expect_ranks: int) -> dict:
    """Fold per-rank verification replies into one verdict. Pure.

    Returns `{"bad": [key, ...], "checked": int, "ranks": int,
              "fault": None | str}` -- `bad` being every key that ANY rank
    could not vouch for, because a key is only restorable if every rank holds
    its own slice. One rank's miss is the whole key's miss, which is the same
    all-ranks property `complete_store` rests on, read in the other direction.

    *** A SHORT REPLY LIST IS A FAULT, NOT A SMALLER CHECK. *** Fewer replies
    than ranks means a rank did not answer, and treating that as "the ranks
    that did answer all said yes" is exactly the manufactured zero this repo
    keeps paying for. It is named as a fault and the caller refuses.
    """
    out = {"bad": [], "checked": 0, "ranks": len(results), "fault": None}
    if len(results) < expect_ranks:
        out["fault"] = (f"only {len(results)} of {expect_ranks} rank(s) "
                        f"answered the presence verification")
        return out
    bad: dict[bytes, None] = {}
    for r in results:
        if not isinstance(r, dict):
            out["fault"] = f"a rank replied with {type(r).__name__}, not a dict"
            return out
        if r.get("error"):
            out["fault"] = f"rank {r.get('rank')}: {r['error']}"
            return out
        out["checked"] = max(out["checked"], int(r.get("checked") or 0))
        for hexkey in (r.get("missing") or ()):
            bad[bytes.fromhex(hexkey)] = None
        for hexkey in (r.get("wrong_size") or ()):
            bad[bytes.fromhex(hexkey)] = None
    out["bad"] = list(bad)
    return out


def journal_encode(kind: str, items) -> str:
    """One journal record, newline-terminated.

    *** NEWLINE-DELIMITED RATHER THAN ONE JSON DOCUMENT, and that is the whole
    design. *** The point of a journal is that a reader can still use the
    PREFIX of a file whose tail was lost to an unclean kill. `json.load` of a
    truncated array yields nothing at all -- it would turn a partial loss into
    a total one, which is the failure the manifest already has.
    """
    import json
    if kind == JOURNAL_ADD:
        payload = [[bytes(k).hex(), int(p)] for k, p in items]
    elif kind == JOURNAL_REMOVE:
        payload = [bytes(k).hex() for k in items]
    elif kind == JOURNAL_CLEAR:
        payload = 1
    else:
        raise ValueError("unknown journal record kind: %r" % (kind,))
    return json.dumps({kind: payload}, separators=(",", ":")) + "\n"


def journal_decode(line):
    """`(kind, payload)`, or None when the line cannot be trusted.

    None covers the torn final record -- the EXPECTED shape of a kill
    mid-append -- and anything else unparseable. The caller must STOP on None
    rather than skip it: a record that did not land completely means the writer
    was interrupted, so nothing after it is evidence about anything. Skipping
    would silently apply changes from the far side of a gap.
    """
    import json
    s = line.strip()
    if not s:
        return None
    try:
        d = json.loads(s)
    except Exception:
        return None
    if not isinstance(d, dict) or len(d) != 1:
        return None
    kind, payload = next(iter(d.items()))
    if kind == JOURNAL_ADD:
        out = []
        try:
            for item in payload:
                out.append((bytes.fromhex(item[0]), int(item[1])))
        except Exception:
            return None
        return (kind, out)
    if kind == JOURNAL_REMOVE:
        try:
            return (kind, [bytes.fromhex(h) for h in payload])
        except Exception:
            return None
    if kind == JOURNAL_CLEAR:
        return (kind, [])
    return None


def journal_replay(base, lines):
    """Replay a journal over a snapshot. Returns `(entries, stats)`.

    *** IDEMPOTENT BY CONSTRUCTION, AND THAT IS WHAT MAKES COMPACTION
    CRASH-SAFE. *** Compaction writes the new snapshot and only THEN truncates
    the journal, so a crash in between leaves records the snapshot already
    contains. An add of a present key is a no-op (mirroring `_remember`'s early
    return) and a remove of an absent key is a no-op (mirroring `_forget`'s
    `pop` default), so replaying them a second time cannot change the answer.
    Reverse that order -- truncate, then write -- and a crash there loses the
    index entirely.

    `base` is `(key, pos)` pairs from the snapshot. Order is append order,
    which is all that is wanted: `_write_manifest` already records that LRU
    recency is deliberately NOT persisted, because a previous boot's recency is
    meaningless and evicting by it would be worse than evicting by nothing.
    """
    entries = OrderedDict()
    for k, p in base:
        entries[bytes(k)] = int(p)
    stats = {"adds": 0, "removes": 0, "clears": 0, "torn": 0, "records": 0}
    for line in lines:
        rec = journal_decode(line)
        if rec is None:
            stats["torn"] = 1
            break
        kind, payload = rec
        stats["records"] += 1
        if kind == JOURNAL_ADD:
            for k, p in payload:
                if k not in entries:
                    entries[k] = p
                    stats["adds"] += 1
        elif kind == JOURNAL_REMOVE:
            for k in payload:
                if entries.pop(k, _JMISSING) is not _JMISSING:
                    stats["removes"] += 1
        elif kind == JOURNAL_CLEAR:
            entries.clear()
            stats["clears"] += 1
    return list(entries.items()), stats


def compaction_due(records: int, journal_bytes: int,
                   max_records: int, max_bytes: int) -> bool:
    """Should the journal be folded back into a snapshot? Either bound trips.

    Records bound the REPLAY cost at boot; bytes bound the disk cost of a
    pathological run of tiny records. A zero or negative bound DISABLES that
    half rather than meaning "always" -- a knob somebody zeroed must not turn
    compaction into a whole-index rewrite per record, which is the exact cost
    this mechanism exists to remove. That is this repo's rule that a knob's
    degenerate value must fail safe, learnt from `load_inflight_max`.
    """
    if max_records and max_records > 0 and records >= max_records:
        return True
    if max_bytes and max_bytes > 0 and journal_bytes >= max_bytes:
        return True
    return False


def _now() -> float:
    """Monotonic seconds. A helper so the manifest's time floor cannot be
    accidentally written against the wall clock, which a clock step would
    then make either permanently stale or never stale."""
    import time
    return time.monotonic()


def decay_hits(count: float, last_ts: float, now: float,
               halflife_s: float) -> float:
    """A hit count aged by elapsed time. PURE.

    *** A RAW COUNT NEVER FORGETS, which is the whole problem with "matched N
    *** times this week". *** A prefix hit fifty times last Tuesday and never
    since would hold reserved budget for ever, and the budget's job is to
    reflect what is hot NOW. So the count decays: every `halflife_s` of not
    being hit, it halves.

    *** AND THE DECAY IS LAZY, computed on read rather than swept. *** The
    obvious implementation is TinyLFU's periodic halving of every counter, and
    at 560,000 keys that is a 560,000-entry sweep on a timer, on the engine
    core, for a number nobody is looking at between evictions. Decaying on
    access is O(1) and needs no timer at all; the cost is that a stored count
    means nothing without its timestamp, which is why they travel together
    everywhere including the manifest.

    A zero halflife means NO DECAY -- a raw count -- rather than instant
    forgetting, because 0 reads as "feature off" everywhere else in this file
    and the alternative would silently zero every counter.
    """
    if count <= 0:
        return 0.0
    if halflife_s <= 0:
        return float(count)
    dt = now - last_ts
    if dt <= 0:
        return float(count)
    # 2 ** -(dt/halflife): one halving per halflife of silence.
    return float(count) * (0.5 ** (dt / float(halflife_s)))


def bump_hits(count: float, last_ts: float, now: float,
              halflife_s: float) -> "tuple[float, float]":
    """Register one hit, decaying what was there first. PURE.

    Returns `(count, last_ts)` to store. Decay-then-add is the order that
    matters: adding first would let a long-cold key jump straight back to its
    old height on a single hit, which is exactly the "never forgets" behaviour
    the decay exists to remove.
    """
    return (decay_hits(count, last_ts, now, halflife_s) + 1.0, float(now))


def choose_victims(items: "Sequence[tuple]", need_bytes: int,
                   hot_threshold: float = 0.0) -> list:
    """Pick keys to evict until `need_bytes` is freed. PURE.

    `items` is `(key, nbytes, pos, recency)` or `(key, nbytes, pos, recency,
    hits)` in NO required order; `recency` is any comparable where SMALLER is
    older, and `hits` is an ALREADY-DECAYED count (see `decay_hits`). A
    4-tuple means "no hit information", which bands as cold -- so every
    existing caller keeps its behaviour exactly.

    *** HOTNESS DEPRIORITISES; IT DOES NOT VETO. *** A hot key that could never
    be evicted makes the budget unenforceable the moment everything is hot,
    which is the "refusal that does nothing" shape this project keeps finding:
    the budget would report a target it cannot meet and the disk would grow
    anyway. So hot keys sort LAST and are reached only if the cold ones did not
    free enough.

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

    def band(it: tuple) -> int:
        """0 = evict freely, 1 = only if we must."""
        if hot_threshold <= 0:
            return 0
        hits = it[4] if len(it) > 4 else 0.0
        return 1 if (hits or 0.0) >= hot_threshold else 0

    # Cold before hot; then oldest first; then, within an age, the furthest
    # from token 0 -- which is what stops a head being orphaned behind its own
    # tail.
    ordered = sorted(items, key=lambda it: (band(it), it[3], -it[2]))
    out, freed = [], 0
    for it in ordered:
        key, nbytes, _pos = it[0], it[1], it[2]
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
    # The deferral queue's hard cap. Demand-fed, and demand can be a
    # million-token prompt, so it is bounded in code rather than by a knob: a
    # knob here would be a knob on how much of the scheduler's thread a bad
    # day may have. Past the cap a lookup answers miss, which is safe.
    VERIFY_WANTED_MAX = 200_000

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
                       "manifest_writes": 0, "paused": 0,
                       # the flush is off the hot path now, so the two ways
                       # that can go wrong need to be visible: a trigger
                       # skipped because one was already running, and a retry
                       # forced by a concurrent store.
                       "manifest_skipped_busy": 0}
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
        # Hygiene, and it matters for the verified set specifically: a key
        # evicted and later re-stored must not inherit the old vouching, since
        # the file it refers to is a different file.
        self._verified_set().discard(bytes(key))
        self._wanted_map().pop(bytes(key), None)
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

        if not self._manifest:
            return
        # *** THE SNAPSHOT MAY NOT EXIST AND THAT IS NOT AN EARLY RETURN. ***
        # It used to be, and the journal then could not be read at all without
        # one -- which is precisely the state the journal exists to cover: keys
        # stored and appended, no snapshot yet, process killed. Six assertions
        # caught this the first time they ran. An absent snapshot is an empty
        # base, not a reason to skip the replay.
        raw = []
        if _os.path.exists(self._manifest):
            try:
                with open(self._manifest) as f:
                    raw = json.load(f).get("keys", [])
            except Exception:
                logger.exception(
                    "kvmgr: manifest %s is unreadable -- continuing with an "
                    "EMPTY base. The journal beside it is still replayed, so "
                    "this costs recompute and never correctness",
                    self._manifest)
                raw = []
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
        try:
            for item in raw:
                if isinstance(item, str):
                    keys.append(bytes.fromhex(item)); poss.append(-1)
                else:
                    keys.append(bytes.fromhex(item[0])); poss.append(int(item[1]))
        except Exception:
            logger.exception(
                "kvmgr: manifest %s has an unreadable ENTRY -- continuing with "
                "what parsed, plus the journal", self._manifest)
        # *** THE JOURNAL IS REPLAYED HERE, BEFORE THE PRUNE. *** The prune is
        # what makes a stale claim safe -- upstream asserts on a failed load --
        # so a replayed key whose file has gone must still meet it. Replaying
        # after the prune would smuggle unverified keys into the index.
        _base = self._journal_load(list(zip(keys, poss)))
        keys = [k for k, _p in _base]
        poss = [p for _k, p in _base]

        kept, how = self._boot_verify(keys)
        # Published, because "how was the index checked" is the difference
        # between a guarantee and a hope, and a log line is not something a
        # script can gate on.
        self._boot_how = how
        self._present = OrderedDict()
        self._bytes = 0
        _pos_by_key = dict(zip(keys, poss))
        for k in kept:
            self._remember(k, _pos_by_key.get(k, -1))
        self._stats["manifest_loaded"] = len(kept)
        self._stats["manifest_pruned"] = len(keys) - len(kept)
        # *** THE LOG LINE SAYS *HOW* IT WAS CHECKED, because the two answers
        # *** have different strengths and the old text asserted the weaker one
        # *** as if it were the only possibility. *** It used to end "it is a
        # claim about both nodes and can only be checked against this one",
        # which is true of the head-only walk and FALSE once the ranks are
        # asked. A reader must be able to tell which boot they are looking at.
        logger.info(
            "kvmgr: manifest %s -- %d keys restored, %d pruned. CHECKED: %s. "
            "%s",
            self._manifest, len(kept), len(keys) - len(kept), how,
            ("Every surviving key was vouched for by EVERY rank, including its "
             "SIZE, so a load cannot hit a missing or foreign file -- which is "
             "upstream's bare assert and a dead engine core."
             if how.startswith("all-ranks") else
             "This is a claim about both nodes and was checked against ONE. If "
             "a load ever asserts on a missing file, DELETE THIS FILE."))

    # -- the boot check ----------------------------------------------------

    def _head_only_prune(self, keys) -> list:
        """Today's walk: does the file exist ON THIS NODE? Rank-local."""
        import os as _os
        kept = []
        for k in keys:
            try:
                if _os.path.exists(self._mapper.get_file_name(k)):
                    kept.append(k)
            except Exception:
                pass
        return kept

    def _boot_verify(self, keys) -> tuple:
        """Prune the index at boot. Returns `(kept, how)`. NEVER raises.

        *** THE HEAD-ONLY WALK PROTECTS AGAINST HALF OF WHAT IT IS FOR, and it
        *** costs 365s to do it. *** `get_file_name` is rank-dependent, so a
        walk in the engine core checks the engine core's node; a file lost on
        the other one survives the gate and becomes upstream's bare assert on
        the first load. The docstring above has always admitted this.

        *** ASKING EVERY RANK COSTS +28% OF BOOT, AND I PREDICTED A SAVING.
        *** Measured live 2026-10-12: **467.1s** for 922,977 keys in 231 RPCs,
        against the rank-local walk's **365s** -- **+102s**. The ~4,600 keys/s
        I sized it from was the background sweep measured **while serving, on a
        warm dentry cache**, and a boot's cache is cold *and* dropped every 2
        seconds by the start script. The parallelism also does not compose the
        way the estimate assumed: 231 SYNCHRONOUS chunks each pay
        `max(rank0, rank1)`, so the cost is `231 x max(...)` and not
        `max(sum, sum)`.

        So the trade, stated as one: +28% boot time buys coverage of BOTH
        nodes, a SIZE check that existence cannot make, and the removal of
        every runtime deferral. Worth it against a dead engine core -- but a
        trade, not a free win.
        
        *** IT ALSO REMOVES THE DEFERRAL ENTIRELY, which is the whole reason it
        *** exists. *** The first design verified on demand, from `lookup`, and
        the box refuted it: `_sliding_window_lookup` keeps scanning past a
        deferred key, so one pass walked a ~175,000-key SWA group and the 700K
        restore returned ZERO tokens. Verifying before the API opens means no
        lookup ever defers, because every key in the index is already vouched
        for.

        *** THE FALLBACK IS THE OLD WALK, AND IT IS NAMED IN THE RETURN VALUE.
        *** If the engine core cannot be reached off the stack, or a rank
        cannot answer, this degrades to the rank-local walk rather than to
        trusting the index -- a weaker check, never no check -- and `how` says
        which happened so the log cannot imply a guarantee it did not make.
        """
        if not keys:
            return ([], "nothing to check")
        try:
            if not flags.peek("disk_verify_enabled"):
                return (self._head_only_prune(keys),
                        "head-only (disk_verify_enabled is off)")
        except Exception:
            return (self._head_only_prune(keys), "head-only (knob unreadable)")

        core = capture_engine_core_from_caller()
        if core is None:
            logger.warning(
                "kvmgr: boot verification could not find the engine core on "
                "the call stack, so the index is pruned against THIS NODE "
                "ONLY -- the weaker check, which is what every boot before "
                "2026-10-12 did. A file lost on the other node will not be "
                "caught here; the background sweep will still find it once "
                "the engine is busy.")
            return (self._head_only_prune(keys), "head-only (no engine core)")

        # *** THE BOOT'S OWN CHUNK, not the runtime pump's. *** A chunk is a
        # blocking RPC and therefore a step stall, so while serving it must
        # stay small; at boot nothing is being served and a bigger one is
        # measurably faster (4,000 -> 3,794 keys/s, 40,000 -> 7,998 keys/s,
        # A/B/A on the live engine). Two knobs because one value cannot be
        # right for both.
        try:
            cap = max(1, int(flags.peek("disk_boot_verify_chunk") or 1))
        except Exception:
            cap = int(flags.DEFAULTS["disk_boot_verify_chunk"])
        kept: list = []
        bad_total = 0
        t0 = _now()
        for i in range(0, len(keys), cap):
            chunk = [bytes(k) for k in keys[i:i + cap]]
            try:
                results = core.collective_rpc(
                    "kvmgr_verify_presence", timeout=BOOT_VERIFY_TIMEOUT_S,
                    kwargs={"keys": [k.hex() for k in chunk]})
                if not isinstance(results, list):
                    results = [results]
                v = verify_verdict(results, self._num_workers)
                if v["fault"]:
                    raise RuntimeError(v["fault"])
            except Exception as e:
                # *** FALL BACK FOR THE REST, DO NOT GIVE UP ON WHAT IS DONE.
                # *** The chunks already verified are verified; redoing them
                # with the weaker check would throw away a strictly better
                # answer. Everything from here on gets the rank-local walk.
                logger.exception(
                    "kvmgr: boot verification failed on chunk %d of %d (%r) -- "
                    "the %d key(s) already checked keep their ALL-RANKS "
                    "verdict, and the remaining %d are pruned against THIS "
                    "NODE ONLY. That is the weaker check, never no check.",
                    i // cap + 1, (len(keys) + cap - 1) // cap, e,
                    len(kept), len(keys) - i)
                # *** `kept[:i]` WOULD BE WRONG HERE, and it was. *** `i`
                # indexes the INPUT keys and `kept` is the survivors, which is
                # a shorter and differently-aligned list once anything has
                # been dropped. The vouched-for set is exactly what `kept`
                # held BEFORE the fallback appended to it.
                n_allranks = len(kept)
                self._verified_set().update(kept)
                kept.extend(self._head_only_prune(keys[i:]))
                self._stats["boot_verify_fell_back"] = 1
                return (kept, "all-ranks for %d key(s), then head-only for the "
                              "remaining %d after a failure"
                              % (n_allranks, len(keys) - i))
            bad = {bytes(b) for b in v["bad"]}
            bad_total += len(bad)
            kept.extend(k for k in chunk if k not in bad)
        # *** BORN VERIFIED: that is the point. *** Every survivor has been
        # vouched for by every rank, at the right size, so `lookup` can answer
        # True immediately and nothing ever defers.
        self._verified_set().update(kept)
        ms = int((_now() - t0) * 1000)
        self._stats["boot_verify_ms"] = ms
        self._stats["boot_verify_dropped"] = bad_total
        self._stats["boot_verify_rpcs"] = (len(keys) + cap - 1) // cap
        return (kept, "all-ranks (%d key(s) in %d RPC(s), %.1fs, %d dropped "
                      "as missing or wrong-sized on some rank)"
                      % (len(keys), (len(keys) + cap - 1) // cap, ms / 1000.0,
                         bad_total))

    # -- the presence journal: the I/O shell over the pure functions -------
    #
    # Every piece of state here is created LAZILY ON THE INSTANCE and read with
    # getattr. *** THAT IS NOT STYLE, IT IS A FATAL BUG THIS REPO HAS ALREADY
    # SHIPPED. *** `rebind` re-points a live manager with `obj.__class__ = new`,
    # so the instance gets every new METHOD and never the new constructor -- an
    # attribute added in __init__ and read on a hot path raises AttributeError
    # inside the scheduler, which kills the engine core. `_gcounts` in
    # prepare_store cost exactly that. Never a class-level dict either, which
    # every instance would share.

    def _jstate(self) -> dict:
        """Journal state, created on first use."""
        st = getattr(self, "_jst", None)
        if st is None:
            st = {"fh": None, "records": 0, "bytes": 0}
            self._jst = st
        return st

    def _journal_names(self):
        """`(journal, compacting)` beside the snapshot, or `(None, None)`."""
        if not self._manifest:
            return (None, None)
        return (self._manifest + ".journal", self._manifest + ".compacting")

    def _journal_on(self) -> bool:
        """Does the journal get WRITTEN? Replay does not consult this -- see
        `_journal_load`."""
        try:
            from . import flags as _f
            return bool(_f.peek("journal_enabled"))
        except Exception:
            return False

    def _journal_bound(self, name: str) -> int:
        try:
            from . import flags as _f
            v = _f.peek(name)
            return int(v) if v is not None else int(_f.DEFAULTS[name])
        except Exception:
            try:
                from . import flags as _f
                return int(_f.DEFAULTS[name])
            except Exception:
                return 0

    def _journal_append(self, kind, items) -> None:
        """Append one record. *** NEVER RAISES. ***

        It sits on `complete_store`'s path, which is the scheduler's thread, so
        an exception here is an engine outage. A journal that cannot be written
        degrades to exactly today's behaviour -- the index is still in memory
        and the snapshot is still the backstop -- so the honest response is to
        count it, say so once, and carry on.
        """
        if not items and kind != JOURNAL_CLEAR:
            return
        if not self._journal_on():
            return
        jpath, _c = self._journal_names()
        if not jpath:
            return
        st = self._jstate()
        try:
            line = journal_encode(kind, items)
        except Exception:
            self._stats["journal_errors"] = self._stats.get("journal_errors", 0) + 1
            return
        fh = st.get("fh")
        try:
            if fh is None:
                import os as _os
                _os.makedirs(_os.path.dirname(jpath), exist_ok=True)
                fh = open(jpath, "a")
                st["fh"] = fh
                try:
                    st["bytes"] = _os.path.getsize(jpath)
                except Exception:
                    st["bytes"] = 0
            fh.write(line)
            # *** flush() AND NOT fsync, AND THE THREAT MODEL IS THE ARGUMENT.
            # *** The failure this mechanism exists to prevent is the ENGINE
            # dying with stored keys unnamed. A write plus flush puts the bytes
            # in the OS page cache, which survives the process being killed --
            # and being killed is the normal way this process ends. fsync only
            # adds cover for machine power loss, and costs a sync per store
            # round on the scheduler's own thread. It is a knob, default off.
            fh.flush()
            if self._journal_fsync():
                import os as _os
                _os.fsync(fh.fileno())
            st["records"] += 1
            st["bytes"] += len(line)
            self._stats["journal_appends"] = (
                self._stats.get("journal_appends", 0) + 1)
        except Exception:
            self._stats["journal_errors"] = self._stats.get("journal_errors", 0) + 1
            try:
                st["fh"] = None
                if fh is not None:
                    fh.close()
            except Exception:
                pass
            logger.exception(
                "kvmgr: journal append failed -- the index is still in memory "
                "and presence.json is still the backstop, so this costs "
                "durability between snapshots and never correctness")

    def _journal_fsync(self) -> bool:
        try:
            from . import flags as _f
            return bool(_f.peek("journal_fsync"))
        except Exception:
            return False

    def _journal_rotate(self) -> None:
        """`journal` -> `compacting`, and begin a fresh journal. Never raises.

        *** CALLED IMMEDIATELY AFTER THE SNAPSHOT IS TAKEN, NEVER BEFORE. ***
        A record appended between the two lands in `compacting` and is replayed
        at load, which is harmless because replay is idempotent. Rotate first
        and that record is simply lost -- the orphan this whole mechanism
        exists to prevent, reintroduced by an ordering.
        """
        jpath, cpath = self._journal_names()
        if not jpath:
            return
        st = self._jstate()
        try:
            fh = st.get("fh")
            if fh is not None:
                try:
                    fh.flush()
                    fh.close()
                except Exception:
                    pass
                st["fh"] = None
            import os as _os
            # *** NEVER OVERWRITE AN EXISTING `compacting`. ***
            # The busy check in flush_manifest is `not force and _flush_busy()`,
            # so a FORCED flush -- which `shutdown` and `reset_cache` both do --
            # bypasses it and can land here while an async write still holds its
            # own `compacting`. `os.replace` would then destroy records that the
            # on-disk snapshot does not yet contain, and if the forced write
            # went on to fail they would be gone for good: the exact loss this
            # journal exists to prevent, caused by its own compaction.
            #
            # Skipping the rotate is safe and the asymmetry is why. The caller's
            # snapshot is taken from the live index, so it is a SUPERSET of
            # whatever `compacting` holds -- writing it loses nothing. The only
            # cost is that the journal is not truncated this round, so its
            # records are replayed next boot; replay is idempotent, so that is
            # wasted work rather than a wrong answer.
            if cpath and _os.path.exists(cpath):
                self._stats["journal_rotate_deferred"] = (
                    self._stats.get("journal_rotate_deferred", 0) + 1)
                logger.warning(
                    "kvmgr: journal rotate DEFERRED -- %s already exists, so a "
                    "compaction is still in flight. The snapshot about to be "
                    "written is a superset of it, so nothing is lost; the "
                    "journal simply is not truncated this round.", cpath)
                return
            if _os.path.exists(jpath):
                _os.replace(jpath, cpath)
            st["records"] = 0
            st["bytes"] = 0
            self._stats["journal_rotations"] = (
                self._stats.get("journal_rotations", 0) + 1)
        except Exception:
            logger.exception("kvmgr: journal rotate failed")

    def _journal_drop_compacting(self) -> None:
        """Discard `compacting` -- ONLY ever after the snapshot is durable.

        That ordering is the crash-safety contract: snapshot first, discard
        second. Die in between and the next boot replays records the snapshot
        already holds, which replay is built to absorb. Discard first and a
        crash loses them.
        """
        _j, cpath = self._journal_names()
        if not cpath:
            return
        try:
            import os as _os
            if _os.path.exists(cpath):
                _os.unlink(cpath)
        except Exception:
            logger.exception("kvmgr: could not discard the compacted journal")

    def _journal_load(self, base):
        """Replay the journal over `base`. Returns the new `(key, pos)` list.

        *** REPLAY DOES NOT CONSULT `journal_enabled`. *** The knob gates
        WRITING. If the journal was on and is now off, the records on disk are
        still NEWER than the snapshot, so ignoring them would silently discard
        real presence. The recovery from a journal you distrust is to delete the
        files, exactly as `_load_manifest` says of presence.json itself.

        `compacting` is replayed first: it was frozen before the snapshot it is
        paired with, so its records are older than anything in `journal`.
        """
        jpath, cpath = self._journal_names()
        if not jpath:
            return base
        total = {"records": 0, "adds": 0, "removes": 0, "clears": 0, "torn": 0}
        for path in (cpath, jpath):
            try:
                import os as _os
                if not path or not _os.path.exists(path):
                    continue
                with open(path) as f:
                    lines = f.readlines()
            except Exception:
                logger.exception("kvmgr: journal %s is unreadable -- skipped", path)
                continue
            try:
                base, st = journal_replay(base, lines)
            except Exception:
                logger.exception("kvmgr: journal %s failed to replay -- skipped", path)
                continue
            for k in total:
                total[k] += st.get(k, 0)
            if st.get("torn"):
                logger.warning(
                    "kvmgr: journal %s has a TORN final record -- %d record(s) "
                    "applied and the remainder discarded. This is the expected "
                    "shape of an unclean kill mid-append, not corruption: the "
                    "lost records are keys whose files are on disk and which "
                    "will simply be re-stored.", path, st.get("records", 0))
        for k, v in total.items():
            self._stats["journal_" + k] = v
        if total["records"]:
            logger.info(
                "kvmgr: journal replayed -- %d record(s): %d add(s), %d "
                "remove(s), %d clear(s), torn=%d. The snapshot is pruned "
                "against the filesystem AFTER this, so a replayed key whose "
                "file has gone is still dropped.",
                total["records"], total["adds"], total["removes"],
                total["clears"], total["torn"])
        return base

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
            if self._journal_on():
                # *** WITH THE JOURNAL ON, THE SNAPSHOT IS NO LONGER THE
                # DURABILITY MECHANISM -- it is the compaction of one. *** Every
                # presence edge is already an append, so the 512-key / 30s
                # trigger above would keep paying the O(index) cost it exists
                # to remove, and keep refusing 81% of its own attempts. The
                # snapshot now fires only when the journal has grown enough
                # that replaying it at boot would cost more than rewriting the
                # index once.
                _st = self._jstate()
                if not compaction_due(
                        _st["records"], _st["bytes"],
                        self._journal_bound("journal_max_records"),
                        self._journal_bound("journal_max_bytes")):
                    return
                self._stats["journal_compactions"] = (
                    self._stats.get("journal_compactions", 0) + 1)
            else:
                stale = (_now() - self._last_flush) >= self.MANIFEST_FLUSH_S
                if self._unflushed < self.MANIFEST_FLUSH_KEYS and not stale:
                    return
        # *** THIS FLUSH WAS A ~0.8s HOLE IN THE ENGINE'S THREAD, and the
        # *** comment above justified it with a figure three orders of
        # *** magnitude stale: "the manifest is tens of KB". It is 63 MB.
        #
        # Measured on spark-45ea against the live 771,775-key index:
        #     hex + sorted   0.227s
        #     json.dump      0.552s
        #     TOTAL          0.779s      on an IDLE box
        #
        # Under a live engine, with decode contending for the GIL, that is the
        # 1.05-1.14s mid-stream stall scripts/test-meter.sh section 1 failed on
        # (max inter-event gap 1.137s against a 0.102s baseline). It fires on
        # EITHER 512 new keys OR 30s stale, so any stream that stores anything
        # pays it once.
        #
        # THE COST SCALES WITH THE WHOLE INDEX TO PERSIST A HANDFUL OF NEW
        # KEYS -- 771,775 rewritten for the 71 a single response added. The
        # proper fix is an append-only journal with occasional compaction, and
        # it is NOT what this is; this moves the existing write off the hot
        # path, which is the part that is a user-visible defect today.
        #
        # Three decisions worth stating:
        #
        # 1. THE SORT IS GONE. It existed so a diff of two manifests reads
        #    nicely -- its own comment says so -- and it is 0.227s of
        #    CONTIGUOUS GIL. json.dump to a file object writes in chunks and
        #    releases the GIL on each write, so the dump interleaves with
        #    decode while a sort cannot. A readable diff is not worth a
        #    quarter-second hole in every stream; `sort_keys` on a reader's
        #    own copy gets it back for free.
        # 2. IT RUNS ON A BACKGROUND THREAD, single-flight. A second trigger
        #    while one is in flight is SKIPPED and counted, not queued: the
        #    gate is "enough keys or enough time", so the next one will come.
        # 3. IT ITERATES `_present` WITHOUT A SNAPSHOT, and retries on the
        #    RuntimeError a concurrent mutation causes. A snapshot would be
        #    `list(_present.items())` = 0.153s on the engine's thread, still
        #    above the 0.102s baseline, so it would simply be a smaller
        #    version of the bug. The race is safe HERE and only here: a key
        #    missed by a racing flush is persisted by the next one, and a key
        #    whose file has gone is pruned by the existence walk every load
        #    already does -- so the failure mode is "a few keys wait a cycle",
        #    never a manifest that claims a file it does not have. That
        #    property is what makes this acceptable; it is not a general
        #    licence to iterate a mutating dict.
        if not force and self._flush_busy():
            self._stats["manifest_skipped_busy"] = self._stats.get(
                "manifest_skipped_busy", 0) + 1
            return
        # claim the cycle NOW, on this thread, or the time/count gate re-fires
        # on every step for as long as the write takes and every one of them
        # starts another.
        self._unflushed = 0
        self._last_flush = _now()
        # *** THE SNAPSHOT IS NOT OPTIONAL, AND LEAVING IT OUT BROKE THE
        # *** MANIFEST COMPLETELY. *** The first version iterated the live
        # `_present` on the writer thread and retried on the RuntimeError a
        # concurrent mutation raises, on the reasoning that a missed key is
        # picked up by the next flush. Measured on the box with the store live:
        #
        #   12:38:41 kvmgr: manifest flush gave up after 3 attempts against a
        #            mutating index; the next trigger will try again
        #   12:39:53 (again)
        #
        # With a live store the index mutates CONTINUOUSLY, so a full-index
        # iteration essentially never completes and the manifest is never
        # written at all. That is strictly worse than the stall it replaced:
        # the stall cost a second of stream, this costs restore-after-restart,
        # which is the entire point of the index. "A few keys wait a cycle" was
        # true of one racing mutation and false of a stream of them.
        #
        # So: one `list()` call, which is atomic under the GIL and therefore
        # consistent by construction, and the expensive part (hex, json, 63 MB
        # of I/O -- 0.55s of the 0.78s) happens on the writer thread off it.
        # The snapshot costs 0.153s measured against 771,775 keys, which is a
        # real hiccup and is accepted deliberately: it is rate-limited to once
        # per 512 keys or 30s, it is a fifth of what it replaces, and it is the
        # price of the manifest existing at all. Making it cheap needs the
        # append-only journal, which is O(changes) rather than O(index).
        snap = list(self._present.items())
        # Snapshot first, THEN rotate -- never the other way round. See
        # _journal_rotate: a record landing between the two is replayed and
        # harmless, whereas rotating first loses it.
        self._journal_rotate()
        if force:
            self._write_manifest(snap)
        else:
            self._flush_async(snap)

    # -- the manifest writer ----------------------------------------------
    def flush_wait(self, timeout: float = 30.0) -> bool:
        """Block until any in-flight manifest write has landed.

        The flush is asynchronous now, so "I triggered it" and "it is on disk"
        are different statements. Anything that needs the second one -- a test
        asserting the file exists, a caller about to read it back -- says so by
        calling this instead of sleeping and hoping. Returns True if nothing is
        in flight by the deadline.
        """
        t = getattr(self, "_flush_thread", None)
        if t is None:
            return True
        try:
            t.join(timeout)
            return not t.is_alive()
        except Exception:
            return False

    def _flush_busy(self) -> bool:
        t = getattr(self, "_flush_thread", None)
        return bool(t is not None and t.is_alive())

    def _flush_async(self, snap) -> None:
        """Start the manifest write on a background thread. Never raises."""
        import threading
        try:
            t = threading.Thread(target=self._write_manifest, args=(snap,),
                                 name="kvmgr-manifest", daemon=True)
            # daemon: an unclean kill is the normal way this process ends, and
            # a half-written .part is harmless because the rename is atomic.
            self._flush_thread = t
            t.start()
        except Exception:
            # could not start a thread -- do it inline rather than never
            logger.exception("kvmgr: could not start the manifest thread; "
                             "writing inline")
            self._write_manifest(snap)

    def _write_manifest(self, snap) -> None:
        """Serialise the index to disk, atomically. Never raises.

        *** THE IMPORTS ARE LOCAL AND THAT IS NOT STYLE. *** `flush_manifest`
        imports json and os INSIDE itself, so the first version of this method
        -- split out of that body -- referenced names it did not have and
        raised NameError on every call. The broad `except Exception` below then
        swallowed it, so the manifest silently stopped being written and
        nothing said so. The offline test caught it only because it asserts the
        EFFECT (the file exists) rather than that the call was made, which is
        the difference between a test and a mock.
        """
        import json
        import os as _os
        try:
            _os.makedirs(_os.path.dirname(self._manifest), exist_ok=True)
            tmp = self._manifest + ".part"
            with open(tmp, "w") as f:
                # [hex, pos] pairs, UNSORTED. The sort was 0.227s of CONTIGUOUS
                # GIL and existed only so a diff of two manifests reads nicely;
                # `sort_keys` on a reader's own copy gets that back for free.
                # The LRU order is deliberately not persisted either: a
                # restart's recency is meaningless and pretending otherwise
                # would evict by the age of a previous boot.
                json.dump({"keys": [[k.hex(), p] for k, (nb, p) in snap]}, f)
            _os.replace(tmp, self._manifest)
            self._stats["manifest_writes"] += 1
            # The snapshot is durable, so the records it absorbed may go. This
            # ordering is the crash-safety contract and must not be inverted.
            self._journal_drop_compacting()
        except Exception:
            # NO RETRY LOOP ANY MORE: it existed to survive a mutating index,
            # and the snapshot means there is no longer one to survive. A retry
            # here would be retrying a real I/O error, which is not something
            # to paper over.
            logger.exception("kvmgr: manifest flush failed (presence kept "
                             "in memory; a restart will simply forget it)")

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
        kb = bytes(key)
        if kb not in self._present:
            return False
        # *** THE INDEX IS A CLAIM ABOUT TWO FILESYSTEMS. *** Present is not
        # the same statement as restorable: the boot walk checked this node
        # only, and a file lost on the other one makes the load upstream's bare
        # assert. So a key the cluster has not vouched for DEFERS rather than
        # being claimed -- `None` is upstream's own "ask me again", and the
        # scheduler re-queues the request with nothing allocated and no worker
        # involved. See _verify_answer for the three outcomes.
        ans = self._verify_answer(kb)
        if ans is None:
            self._stats["verify_deferred"] = (
                self._stats.get("verify_deferred", 0) + 1)
            return None
        if ans:
            self._stats["hit"] += 1
            return True
        return False

    # -- cross-rank verification ------------------------------------------
    #
    # Every piece of state here is created LAZILY ON THE INSTANCE and read
    # through one accessor, for the reason stated at _loading_map: `rebind`
    # re-points a live manager with `obj.__class__ = new`, so an instance
    # already serving requests gets every new METHOD and no new attribute, and
    # an AttributeError on this path is fatal to the scheduler.

    def _verified_set(self) -> set:
        """Keys every rank has vouched for, created on first use.

        It holds references to the SAME bytes objects `_present` keys on, so
        the cost is the set's slot array rather than a second copy of 919,669
        keys.
        """
        v = getattr(self, "_verified", None)
        if v is None:
            v = set()
            self._verified = v
        return v

    def _wanted_map(self) -> "OrderedDict":
        """Keys a lookup is waiting on -> when it first asked (monotonic).

        An OrderedDict so the pump drains oldest-first: the request that has
        been deferring longest is the one closest to answering False.
        """
        w = getattr(self, "_wanted", None)
        if w is None:
            w = OrderedDict()
            self._wanted = w
        return w

    def _verify_answer(self, kb: bytes) -> "bool | None":
        """True = vouched for, None = defer, False = answer miss.

        Three outcomes and a strict order of precedence:

        1. verification OFF -> True. The knob means "trust the index", which is
           exactly today's behaviour, so turning it off cannot make anything
           slower or stranger than it already is.
        2. a persistent FAULT -> False. A rank that cannot answer must not read
           as a rank that said yes; refusing costs a cold cache, and claiming
           costs the engine.
        3. otherwise: verified -> True; else enqueue and defer until the
           deferral budget is spent, then False.

        *** A MISS IS ALWAYS SAFE, AND THAT IS WHY THE EXPIRY EXISTS. *** The
        worst this can do by answering False is recompute a prefix -- which is
        the thing the whole tier is measured against -- while answering True
        wrongly is upstream's assert.
        """
        if not flags.peek("disk_verify_enabled"):
            return True
        if _VERIFY_FAULT is not None:
            self._stats["verify_refused_fault"] = (
                self._stats.get("verify_refused_fault", 0) + 1)
            return False
        if kb in self._verified_set():
            return True
        w = self._wanted_map()
        now = _now()
        t0 = w.get(kb)
        if t0 is None:
            # Bounded, because the queue is fed by demand and demand can be a
            # million-token prompt. Past the cap a key is simply answered as a
            # miss rather than queued -- the cap is reached only when the pump
            # is not keeping up, and in that state a cold prefix beats an
            # unbounded dict on the scheduler's thread.
            if len(w) >= self.VERIFY_WANTED_MAX:
                self._stats["verify_queue_full"] = (
                    self._stats.get("verify_queue_full", 0) + 1)
                return False
            w[kb] = now
            return None
        budget = float(flags.peek("disk_verify_defer_s") or 0.0)
        if budget > 0 and (now - t0) < budget:
            return None
        self._stats["verify_defer_expired"] = (
            self._stats.get("verify_defer_expired", 0) + 1)
        return False

    def _verify_state(self) -> dict:
        """Pump bookkeeping, created on first use."""
        st = getattr(self, "_vst", None)
        if st is None:
            st = {"cursor": 0, "bg_credit": 0.0, "bg_at": 0.0, "rpc_at": 0.0}
            self._vst = st
        return st

    def _verify_chunk(self) -> list:
        """The next keys to verify: demand first, then slow background.

        *** DEMAND BEFORE BACKGROUND, ALWAYS. *** A deferring request is a
        request not being served, and the background sweep has all boot to
        finish. The chunk bounds the RPC MESSAGE, not the work: a long
        request's whole candidate set (2,758 keys for a 700K prefix, measured)
        fits inside one.
        """
        st = self._verify_state()
        cap = max(1, int(flags.peek("disk_verify_chunk") or 1))
        w = self._wanted_map()
        # oldest-first: the request closest to giving up is served first
        out = [k for k, _t in list(w.items())[:cap]]
        if len(out) >= cap:
            return out
        # -- the slow background re-verification --------------------------
        #
        # It exists because divergence can appear MID-BOOT: an operator `rm`, a
        # partial kvsweep.sh, a filesystem error. Rate-limited in keys per
        # second rather than per step, because step rate is a property of the
        # traffic and this must not speed up under load -- it is the one thing
        # here that has all day.
        rate = float(flags.peek("disk_verify_bg_per_s") or 0.0)
        if rate <= 0:
            return out
        now = _now()
        last = st.get("bg_at") or now
        st["bg_credit"] = min(float(cap),
                              float(st.get("bg_credit") or 0.0)
                              + max(0.0, now - last) * rate)
        st["bg_at"] = now
        take = min(int(st["bg_credit"]), cap - len(out))
        if take <= 0:
            return out
        # A CURSOR OVER THE INDEX, not a timestamp per key: 919,669 floats is
        # memory this box does not have spare, and the question "has this been
        # re-checked lately" only ever needs answering in aggregate.
        keys = list(self._present.keys())
        if not keys:
            return out
        i = int(st.get("cursor") or 0) % len(keys)
        picked, seen = [], 0
        while len(picked) < take and seen < len(keys):
            k = keys[i]
            i = (i + 1) % len(keys)
            seen += 1
            if k not in w:
                picked.append(k)
        st["cursor"] = i
        st["bg_credit"] -= len(picked)
        # *** BACKGROUND KEYS ARE NOT ENQUEUED IN `_wanted`, AND THAT IS THE
        # *** POINT. *** `_wanted` is what makes a lookup defer, and a key
        # already vouched for must stay claimable while it is re-checked -- the
        # only thing a re-check can do is remove a key that was already going
        # to kill the engine. Re-verification must never cost a cache miss.
        self._stats["verify_bg_checked"] = (
            self._stats.get("verify_bg_checked", 0) + len(picked))
        return out + picked

    def verify_apply(self, asked: "Sequence[bytes]", results: "Sequence[dict]",
                     expect_ranks: int) -> dict:
        """Apply per-rank verdicts. Returns the folded verdict, never raises.

        *** THE SELF-HEAL IS THE BEST PROPERTY OF THIS DESIGN AND IT IS THREE
        *** LINES. *** A key any rank cannot vouch for is FORGOTTEN and a
        remove is journalled. It leaves the index, so the next `prepare_store`
        offers it again; the rank that still holds a good file SKIPS the write
        (`disk_skip_present_before_copy`, and `_write_batch`'s own exists()
        behind it), the rank that lost it writes it, and `complete_store`
        re-indexes it as verified. **A file lost on one node is repaired by the
        next store instead of killing the engine.**
        """
        v = verify_verdict(results, expect_ranks)
        if v["fault"]:
            return v
        asked_set = {bytes(k) for k in asked}
        bad = {bytes(k) for k in v["bad"]}
        good = asked_set - bad
        ver = self._verified_set()
        w = self._wanted_map()
        for k in good:
            if k in self._present:
                ver.add(k)
            w.pop(k, None)
        if bad:
            freed = 0
            for k in bad:
                freed += self._forget(k)
                w.pop(k, None)
            # Journalled, so a boot cannot resurrect a key the cluster has just
            # told us it cannot serve. Same ordering argument as eviction: a
            # remove recorded for a file that survives costs one re-store, a
            # delete with no record is an orphan.
            self._journal_append(JOURNAL_REMOVE, list(bad))
            self._stats["verify_forgot"] = (
                self._stats.get("verify_forgot", 0) + len(bad))
            self._stats["verify_forgot_bytes"] = (
                self._stats.get("verify_forgot_bytes", 0) + freed)
            self._unflushed += len(bad)
            logger.warning(
                "kvmgr: cross-rank verification DROPPED %d key(s) (%.2f GB) "
                "that at least one rank could not vouch for -- a missing or "
                "wrong-sized file on EITHER node. They are out of the index, "
                "so the next store re-offers them: the rank that still has a "
                "good file skips the write, the rank that lost it writes it, "
                "and the key comes back verified. This is the self-heal, not "
                "a loss -- without it the next load of one of these keys "
                "would have been upstream's assert and a dead engine.",
                len(bad), freed / 1e9)
        self._stats["verify_checked"] = (
            self._stats.get("verify_checked", 0) + len(asked_set))
        self._stats["verify_rpcs"] = self._stats.get("verify_rpcs", 0) + 1
        return v

    def _apply_verify_request(self) -> None:
        """Take the listener's request, if any. *** SCHEDULER THREAD ONLY. ***

        The single writer of `_wanted` besides `lookup`, and it runs in the
        same place `lookup` does -- which is the entire reason the listener
        hands over a dict instead of doing this itself.

        Total: a request that cannot be applied is logged and DROPPED rather
        than retried, because a request that raises every step forever is the
        hot-path trap this codebase already records twice.
        """
        global _VERIFY_REQUEST, _VERIFY_LAST, _VERIFY_FAULT

        req = _VERIFY_REQUEST
        if req is None:
            return
        _VERIFY_REQUEST = None  # take it first, so a raise cannot re-run it
        out = {"seq": req.get("seq"), "requested_at": req.get("at"),
               "applied_at": _now(), "all": bool(req.get("all")),
               "clear_fault": bool(req.get("clear_fault"))}
        try:
            if req.get("clear_fault"):
                # *** DELIBERATELY MANUAL, and this is the only lever. *** A
                # fault means a rank could not be asked whether it holds a
                # file; healing it on a timer would mean nothing ever gets
                # looked at, which is how a refusal nobody can see becomes a
                # capability nobody checked.
                out["fault_was"] = _VERIFY_FAULT
                _VERIFY_FAULT = None
                if out["fault_was"] is not None:
                    logger.warning(
                        "kvmgr: verification fault CLEARED on request (was "
                        "%r) -- presence can be claimed again and the next "
                        "pump round re-checks what it is asked for",
                        out["fault_was"])
            if req.get("all"):
                # *** IT RE-QUEUES WITHOUT UNVERIFYING. *** `lookup` consults
                # `_verified` BEFORE `_wanted`, so a key that is both
                # already-vouched-for and queued still answers True
                # immediately while the pump re-checks it behind the request.
                # Clearing `_verified` instead -- the first version of this --
                # would make every lookup in the index defer, turning a
                # diagnostic into a cache outage, and all a re-check can
                # ever do is REMOVE a key that was going to kill the engine.
                w = self._wanted_map()
                now = _now()
                n = 0
                for k in self._present:
                    if len(w) >= self.VERIFY_WANTED_MAX:
                        break
                    if k not in w:
                        w[k] = now
                    n += 1
                out["requeued"] = n
                out["capped"] = n < len(self._present)
                logger.warning(
                    "kvmgr: %d key(s) re-queued for cross-rank verification "
                    "on request (capped=%s). *** THIS ISSUES BLOCKING RPCs "
                    "FROM THE SCHEDULER STEP, AND A CHUNK IS A STALL *** -- "
                    "%d keys per round, measured at ~1054ms for 4000, so the "
                    "stream pays it until the queue drains. The runtime rate "
                    "is 0 by default because a step-time RPC is what is "
                    "suspected in the 2026-10-12 NCCL kill, so this is an "
                    "operator action with a known cost, not a sweep.",
                    n, out["capped"],
                    int(flags.peek("disk_verify_chunk") or 0))
            out["ok"] = True
        except Exception as e:
            out["ok"] = False
            out["error"] = repr(e)
            logger.exception("kvmgr: could not apply verification request %r; "
                             "it is DROPPED rather than retried", req)
        _VERIFY_LAST = out

    def pump(self, engine_core) -> None:
        """One verification round, from the scheduler step. Called every step.

        *** THE RPC IS BLOCKING, AND THAT IS SAFE FOR A REASON THAT HAD TO BE
        *** READ OFF THE IMAGE RATHER THAN ASSUMED. ***
        `MultiprocExecutor.collective_rpc` wraps every call in a
        `FutureWrapper` on a shared `futures_queue`, and a blocking
        `result()` DRAINS that queue oldest-first
        (`while not self.done(): self.futures_queue.pop()._wait_for_response()`).
        So with an `execute_model` future outstanding -- and
        `--async-scheduling` means there normally IS one -- ordering is
        preserved by construction. The cost is latency, not corruption: our
        call forces the outstanding model step to be awaited inline. See
        patches/pristine/vllm/v1/executor/multiproc_executor.py, captured for
        exactly this question.

        *** AND IT IS WHY A TIMEOUT IS FATAL. *** A `TimeoutError` does NOT set
        the executor's `is_failed`; the future takes the exception and is
        popped, leaving the worker's eventual reply UNCLAIMED on an ordered
        response queue. Every later dequeue is then off by one, so the engine
        receives a verification dict where it expects a `ModelRunnerOutput`.
        Swallowing that is silent corruption; the honest choice is to die here
        with a line naming the cause rather than fifty milliseconds later with
        an unattributable traceback. Same argument as `stall_fail_after_s`.

        Everything else is caught: a chunk that cannot be built, a verdict that
        cannot be applied, a rank that errored. Those become a FAULT, which
        refuses presence and costs a cold cache.
        """
        # Before anything else, and before the early return below: a request
        # must be taken even on a round that has no keys and no RPC due, or an
        # idle-ish engine would never apply one.
        try:
            self._apply_verify_request()
        except Exception:
            logger.exception("kvmgr: taking the verification request failed")
        try:
            keys = []
            if flags.peek("disk_verify_enabled") and _VERIFY_FAULT is None:
                keys = self._verify_chunk()
            st = self._verify_state()
            now = _now()
            rpc_iv = float(flags.peek("disk_free_rpc_s") or 0.0)
            due = rpc_iv > 0 and (now - float(st.get("rpc_at") or 0.0)) >= rpc_iv
            # *** A KEEP-ALIVE WITH NO KEYS IS NOT A WASTED CALL. *** It is how
            # the worker's free space is refreshed at all, and it proves the
            # channel works BEFORE a sweep needs it -- a verification path
            # whose first ever use is an emergency is a path nobody has tested.
            if not keys and not due:
                return
            st["rpc_at"] = now
        except Exception:
            logger.exception("kvmgr: verification pump could not build a "
                             "chunk; skipping this round")
            return

        timeout = float(flags.peek("disk_verify_rpc_timeout_s") or 0.0)
        if timeout <= 0:
            # NEVER None, which collective_rpc turns into dequeue(timeout=None)
            # and therefore into a wait with no end. A zeroed knob falls back to
            # the declaration rather than to the dangerous value -- the
            # load_inflight_max lesson, where a reader's own fallback literal
            # reinstated the fatal number through the safety path.
            timeout = float(flags.DEFAULTS["disk_verify_rpc_timeout_s"])
        try:
            results = engine_core.collective_rpc(
                "kvmgr_verify_presence", timeout=timeout,
                kwargs={"keys": [k.hex() for k in keys]})
        except TimeoutError:
            self.verify_fault(
                "the presence-verification RPC timed out after %.0fs. THE "
                "ENGINE CANNOT CONTINUE: a collective_rpc timeout leaves the "
                "worker's reply unclaimed on an ordered response queue, so "
                "every later dequeue is off by one and the engine core would "
                "receive this verification dict where it expects a "
                "ModelRunnerOutput. Dying here with the cause named beats "
                "dying in a moment with an unattributable traceback. The "
                "likely cause is a rank whose filesystem is not answering."
                % timeout)
            raise
        except Exception as e:
            # A dead executor, a missing method, a worker that raised. None of
            # those desync the queue, so this is survivable -- as a fault.
            self.verify_fault("the presence-verification RPC failed: %r" % (e,))
            return

        try:
            if not isinstance(results, list):
                results = [results]
            # The free-space half rides the same reply, so it is applied even
            # when the key list was empty.
            self.note_worker_free([r.get("free_bytes") for r in results
                                   if isinstance(r, dict)])
            if not keys:
                return
            v = self.verify_apply(keys, results, self._num_workers)
            if v.get("fault"):
                self.verify_fault(v["fault"])
        except Exception as e:
            self.verify_fault("the verification reply could not be "
                              "applied: %r" % (e,))

    def verify_fault(self, reason: str) -> None:
        """Record a rank that could not answer. Logged ONCE, queryable always.

        *** "I CANNOT TELL" MUST NEVER READ AS "NOTHING WRONG". *** The drop
        guard set this precedent: for a lever whose failure mode is a dead
        engine, unreadable is treated as bad. So presence refuses for the rest
        of the boot, the fault goes in `/kv` where a script can fail on it, and
        it clears only on an explicit knob reset -- never by itself, because a
        fault that heals silently is a fault nobody fixes.
        """
        global _VERIFY_FAULT
        if _VERIFY_FAULT is not None:
            _VERIFY_FAULT["seen"] = int(_VERIFY_FAULT.get("seen", 0)) + 1
            return
        _VERIFY_FAULT = {"reason": str(reason), "at": _now(), "seen": 1}
        logger.error(
            "kvmgr: PRESENCE VERIFICATION FAULT -- %s. The disk index is a "
            "claim about every rank and one of them cannot be asked, so "
            "presence now REFUSES for the rest of this boot: restores go cold "
            "rather than risking a load whose file is gone on the node that "
            "could not answer, which is upstream's bare assert and a dead "
            "engine core. Visible as `verify.fault` on :9199/kv; clears only "
            "on an explicit reset.", reason)

    # -- the free-space floor ---------------------------------------------

    def _free_state(self) -> dict:
        """Free-space bookkeeping, created on first use."""
        st = getattr(self, "_fst", None)
        if st is None:
            st = {"at": 0.0, "local": None, "ranks": None, "ranks_at": 0.0,
                  "band": BAND_OK}
            self._fst = st
        return st

    def _statvfs_free(self) -> "int | None":
        """Free bytes where the tier writes, or None if it cannot be read.

        The target is the manifest's own directory, which IS `spec.root_dir` --
        no new reference is needed, and it is by construction the filesystem
        the files land on.
        """
        try:
            import os as _os
            if not self._manifest:
                return None
            vfs = _os.statvfs(_os.path.dirname(self._manifest))
            return int(vfs.f_bavail) * int(vfs.f_frsize)
        except Exception:
            self._stats["free_check_errors"] = (
                self._stats.get("free_check_errors", 0) + 1)
            return None

    def note_worker_free(self, free_by_rank: "Sequence[int]") -> None:
        """Record what the WORKERS say about their own filesystems.

        *** THE ACTUATOR OTHERWISE SEES ONLY THE HEAD, AND THE WORKER IS
        *** CURRENTLY THE FULLER OF THE TWO. *** Each rank writes its own slice
        to its own node's disk, so the binding constraint is the MINIMUM across
        nodes and a head-only reading is the wrong number on exactly the day it
        matters. It rides the verification RPC, so it costs no channel of its
        own and doubles as a keep-alive proving that channel works before a
        sweep needs it.
        """
        vals = [int(v) for v in free_by_rank if v is not None and int(v) >= 0]
        st = self._free_state()
        st["ranks"] = min(vals) if vals else None
        st["ranks_at"] = _now()

    def _free_pressure(self) -> tuple:
        """`(band, evict_step_bytes, free_bytes_or_None)`. Throttled.

        *** AN UNREADABLE statvfs READS AS OK, AND SAYS SO. *** That is the
        OPPOSITE of the drop guard's choice and for the opposite reason: there
        "I cannot tell" risks a dead worker, so it refuses; here it risks only
        a cold cache, so refusing would turn a diagnostic failure into a
        feature outage. It is counted and logged rather than assumed.
        """
        st = self._free_state()
        now = _now()
        iv = float(flags.peek("disk_free_check_s") or 0.0)
        if (now - float(st.get("at") or 0.0)) >= iv:
            st["at"] = now
            st["local"] = self._statvfs_free()
            self._stats["free_checks"] = self._stats.get("free_checks", 0) + 1
        seen = [v for v in (st.get("local"), st.get("ranks")) if v is not None]
        step = int(float(flags.peek("disk_free_evict_step_gb") or 0.0) * 1e9)
        if not seen:
            return (BAND_OK, step, None)
        free = min(seen)
        floor = int(float(flags.peek("disk_free_floor_gb") or 0.0) * 1e9)
        hard = int(float(flags.peek("disk_free_hard_gb") or 0.0) * 1e9)
        hyst = int(float(flags.peek("disk_free_hysteresis_gb") or 0.0) * 1e9)
        if floor <= 0:
            return (BAND_OK, step, free)
        band = free_band(free, floor, hard, hyst, st.get("band") or BAND_OK)
        if band != st.get("band"):
            # *** THE BAND EDGE IS THE EVENT; THE 400TH STORE ROUND IN A BAND
            # *** IS NOT. *** Logging per round is how a reader learns to skip
            # the channel the first transition arrives on.
            logger.warning(
                "kvmgr: free-space band %s -> %s (%.1f GB free, floor %.1f, "
                "hard %.1f, min across %d node(s)). DEGRADED makes every store "
                "round net-negative by one %.1f GB step, so the drain is paid "
                "by the traffic that caused it; FROZEN admits exactly one key "
                "per round so the store job still exists to carry the unlinks "
                "out, because a refusal would destroy the only delete channel "
                "there is and make the freeze permanent.",
                st.get("band"), band, free / 1e9, floor / 1e9, hard / 1e9,
                len(seen), step / 1e9)
            st["band"] = band
        return (band, step, free)

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

        # *** A FULL DISK IS A DEAD ENGINE, AND THE REFUSAL HAS TO HAPPEN
        # *** HERE. *** The worker CANNOT refuse: `success=False` is fatal on
        # every path, and reporting success without writing poisons the index
        # so a later load dies on a missing file instead. The engine core is
        # the only process that can both decline a store and choose victims.
        _band, _step, _free = self._free_pressure()
        if _band == BAND_FROZEN:
            if _free is not None and _free <= FREE_ABSOLUTE_MIN_BYTES:
                # The one place the tier truly stops. Below a gigabyte free,
                # the single key below cannot rescue anything and its write IS
                # the ENOSPC being avoided.
                self._stats["free_refusals"] = (
                    self._stats.get("free_refusals", 0) + len(todo))
                # _nothing_to_store(), NEVER None: None means "I cannot", which
                # leaves stored_idx unadvanced and re-offers the same blocks
                # every step for ever -- measured here at 3,309 warnings at
                # ~20/s. This is the same refusal `disk_store_paused` makes.
                return _nothing_to_store()
            # *** ADMIT EXACTLY ONE KEY, AND THE REASON IS NOT OBVIOUS. *** A
            # hard refusal destroys the only delete channel there is:
            # `scheduler.py:745` does `if not store_output.keys_to_store:
            # advance_stored_idx(...); continue`, which drops the store spec --
            # and the unlinks ride that spec by design. So a refusal would make
            # the freeze PERMANENT. One ~1 MB write buys a step-sized reclaim.
            self._stats["free_frozen_rounds"] = (
                self._stats.get("free_frozen_rounds", 0) + 1)
            self._stats["free_refusals"] = (
                self._stats.get("free_refusals", 0) + max(0, len(todo) - 1))
            todo = todo[:1]

        # *** RECORD THE POSITION NOW, because this is the only moment it is
        # *** known. *** The scheduler hands keys in block order within a
        # group -- the same input-order invariant check_pairing depends on --
        # so the i-th key of a group is block i of that prefix. By the time
        # complete_store fires, that ordering is gone. Tail-first eviction is
        # impossible without it.
        # *** INDEX OVER `keys`, NOT `todo`, AND THAT ONE WORD WAS THE BUG. ***
        # `todo` is the NOT-PRESENT subset, so indexing it numbers the keys
        # within THIS BATCH rather than within the prefix. For an incremental
        # store -- a conversation growing, which is the common case -- blocks
        # 50..60 of a prefix arrive as the only new ones and get pos 0..10, so a
        # block deep in the prefix is labelled a HEAD while the genuine head
        # from an earlier batch carries the same pos. Tail-first then orders on
        # a number that means what the comment below claims only when an entire
        # prefix arrives in one batch.
        #
        # Measured 2026-10-09: after evicting 150,358 keys, 41% of blocks
        # offered were already on disk and 27 external lookups produced ZERO
        # hits -- the data present and unreachable, because the "tail" being
        # evicted was sometimes the head.
        #
        # `keys` is the full offered run (that is exactly why `todo` filters
        # it), so the index within `keys` is the block's position in the prefix
        # for its group. Recorded only for the keys actually being stored.
        #
        # HONEST LIMIT: this is the position within the OFFERED RUN. It equals
        # the prefix position when the run starts at token 0, which is the shape
        # the scheduler uses for a contiguous reuse match. It is not a promise
        # about a run that starts elsewhere -- better, not perfect.
        try:
            from vllm.v1.kv_offload.base import get_offload_group_idx
            _todo_set = {bytes(k) for k in todo}
            _seen: dict[int, int] = {}
            for k in keys:
                g = get_offload_group_idx(k)
                i = _seen.get(g, 0)
                _seen[g] = i + 1
                kb = bytes(k)
                if kb in _todo_set:
                    self._pos_hint[kb] = i
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
        # *** THE FLOOR IS EVALUATED BEFORE THE `budget == 0` EARLY RETURN, AND
        # *** GETTING THAT ORDER WRONG WOULD HAVE LEFT A disk_budget_gb=0
        # *** DEPLOYMENT WITH NO FLOOR AT ALL. *** `budget_bytes` returns 0 for
        # "unbounded", and this function used to return [] on that immediately.
        # A free-space floor is not a budget: it is the thing standing between
        # ENOSPC and upstream's bare assert, and it has to hold whether or not
        # anybody set a byte budget.
        band, step, _free = self._free_pressure()
        incoming_bytes = sum(self._key_bytes(bytes(k)) for k in incoming)
        need = (self._bytes + incoming_bytes) - budget if budget else 0
        if band != BAND_OK and step > 0:
            # *** DRIVE THE EXISTING EVICTION MACHINERY VIA AN *EFFECTIVE*
            # *** BUDGET -- never by lowering disk_budget_gb. *** Writing the
            # knob would be an actuator mutating its own configuration: the
            # operator's number would be gone, and nothing would restore it
            # when the disk recovered. So the pressure is expressed as a budget
            # for THIS CALL only.
            #
            # `max(0, held - step)` makes `need` come out as
            # `incoming + min(step, held)`, so every store round is
            # net-negative by one step and the drain is paid by the traffic
            # that caused it rather than in one burst.
            eff = max(0, self._bytes - step)
            need = max(need, (self._bytes + incoming_bytes) - eff)
            self._stats["free_evict_rounds"] = (
                self._stats.get("free_evict_rounds", 0) + 1)
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
        # Journalled BEFORE the delete rides out to the workers: a remove
        # recorded for a file that then survives is a key we will simply
        # re-store, while a delete that happened with no record is a file no
        # future boot can name -- the orphan. Err toward the recoverable one.
        self._journal_append(JOURNAL_REMOVE, [bytes(v) for v in victims])
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
        _jadds = []
        for k in new:
            # The position was recorded when the key was OFFERED, because that
            # is the only moment it is known: the scheduler hands keys in block
            # order within a group, which is the same invariant check_pairing
            # relies on. By completion time that ordering is gone.
            _p = self._pos_hint.pop(bytes(k), -1)
            self._remember(bytes(k), _p)
            _jadds.append((bytes(k), _p))
            # *** KEYS STORED THIS BOOT ARE BORN VERIFIED, and that is not an
            # *** optimisation -- it is the same all-ranks property the index
            # *** already rests on. *** complete_store fires if and only if
            # EVERY rank reported the job done (`pending_count=world_size`,
            # aggregated by the framework), so a key arriving here was written
            # everywhere, by construction. The cross-rank sweep therefore only
            # ever covers keys restored from the manifest or the journal --
            # which is also why a long-running boot converges to zero
            # verification traffic.
            self._verified_set().add(bytes(k))
            self._wanted_map().pop(bytes(k), None)
        # The journal's add edge is THIS one and only this one, for the same
        # reason the manifest's is: complete_store fires if and only if every
        # rank reported the job done, so a key recorded here was written
        # everywhere. Appending anywhere earlier would claim a key rank 1 may
        # still be writing.
        self._journal_append(JOURNAL_ADD, _jadds)
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
        # *** CLEAR THE VOUCHING TOO, or a reset leaks verdicts across it. ***
        # The keys are gone, so every verdict about them is about files this
        # index no longer claims; keeping them would vouch for whatever a
        # future key with the same hash pointed at.
        self._verified_set().clear()
        self._wanted_map().clear()
        self._verify_state()["cursor"] = 0
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
                    store_paused=bool(flags.peek("disk_store_paused")),
                    verify=self._verify_snapshot(),
                    free=self._free_snapshot(),
                    fingerprint=self._fingerprint())

    def _verify_snapshot(self) -> dict:
        """What the cross-rank check knows. *** A FAULT MUST BE QUERYABLE. ***

        Logged-once is not enough: `kvbootconfirm.sh` has to be able to FAIL on
        a down worker, and a log line is not something a script should parse.
        `fault` is null when there is none and a dict naming the rank and
        reason when there is -- never absent, so a reader cannot mistake "no
        key" for "no problem".
        """
        return {
            "enabled": bool(flags.peek("disk_verify_enabled")),
            "verified": len(self._verified_set()),
            "wanted": len(self._wanted_map()),
            "unverified": max(0, len(self._present) - len(self._verified_set())),
            "cursor": int(self._verify_state().get("cursor") or 0),
            "boot_check": getattr(self, "_boot_how", None),
            "fault": _VERIFY_FAULT,
            # Never absent, for the same reason `fault` is not: a 202 from
            # POST /kv/verify is only checkable if "nothing has been applied"
            # is a value rather than a missing key.
            "last_request": _VERIFY_LAST,
            "pending_request": _VERIFY_REQUEST,
        }

    def _free_snapshot(self) -> dict:
        """The floor, as numbers. *** A FLOOR NOBODY CAN SEE IS A FLOOR NOBODY
        WILL TRUST, which is why this exists before it has ever fired. ***

        `bytes` is null when statvfs could not be read -- which is treated as
        OK, deliberately, and must therefore be VISIBLE as "not measured"
        rather than rendered as a zero that would read as a full disk.
        """
        st = self._free_state()
        band, step, free = self._free_pressure()
        return {
            "bytes": free,
            "local_bytes": st.get("local"),
            "worker_min_bytes": st.get("ranks"),
            "floor_bytes": int(float(flags.peek("disk_free_floor_gb") or 0) * 1e9),
            "hard_bytes": int(float(flags.peek("disk_free_hard_gb") or 0) * 1e9),
            "absolute_min_bytes": FREE_ABSOLUTE_MIN_BYTES,
            "evict_step_bytes": step,
            "band": band,
        }

    def _fingerprint(self) -> dict:
        """What this cache is KEYED ON, exposed so the proxy can see it.

        *** A FINGERPRINT NOBODY CAN READ IS A CLAIM NOBODY CAN CHECK. *** The
        2026-10-07 failure was the worker and the engine core computing
        DIFFERENT base paths and nothing saying so -- 552 keys pruned as "the
        file is gone" with the files right there. It cost one boot per guess
        until the FIELDS were printed. Publishing it means the proxy can show
        which checkpoint a 400 GB cache belongs to, and a future reader can tell
        "the model changed" from "the cache is cold" without reading a log.
        `model_identity` is the part worth watching: `unknown` means the cache
        is no better protected than a model NAME, which is a fact about this
        deployment rather than a bug.
        """
        out = {"format_version": DISK_FORMAT_VERSION,
               "format_root": DISK_FORMAT_ROOT}
        m = getattr(self, "_mapper", None)
        if m is None:
            return out
        try:
            out["base_path"] = m.base_path
            name = (m.fields or {}).get("model_name") or ""
            out["model_name"] = name.split("@")[0]
            out["model_identity"] = (
                name.split("@", 1)[1] if "@" in name else "unknown")
            out["writer"] = (m.fields or {}).get("inference_engine")
            out["hash_block_size"] = (m.fields or {}).get("hash_block_size")
        except Exception:
            out["error"] = "fields unreadable"
        return out


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
        # The LOAD's own pinned rows. Separate from the store's because sharing
        # one buffer is what faulted the worker with Xid 13 five times: the load
        # reads it under CU_MEMCPY_SRC_ACCESS_ORDER_ANY, whose precondition is
        # that no concurrent GPU stream writes the source, and the store's copy
        # is exactly such a write. A quarter of the store's by default, so the
        # added pinned memory is a quarter rather than a doubling -- and the way
        # to hold the total flat is to lower `staging_blocks` by the same
        # amount, which is a per-node KV_TRANSFER_CONFIG change and a restart.
        self.staging_load_blocks: int = int(self.extra_config.get(
            "staging_load_blocks", max(1, self.staging_blocks // 4)))
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

    def _migrate_legacy_base(self, parallel_config) -> None:
        """Rename the pre-Phase-4 fingerprint directory onto the new one.

        *** STRENGTHENING A FINGERPRINT ORPHANS EVERY FILE UNDER THE OLD ONE,
        *** and here that is 400 GB and 560,000 keys of warm cache. *** The
        bytes are not wrong -- the LAYOUT did not change, only the identity
        string did -- so throwing them away would be a pure loss, paid in cold
        prefills at 430-795 s each. A directory rename is O(1) on the same
        filesystem and preserves the manifest too, because `presence.json`
        lives at the format root and is keyed by block hash, not by path.

        *** THE GUARD IS THE FORMAT VERSION, and it is not optional. *** This
        may only ever adopt files whose layout matches ours. At
        DISK_FORMAT_VERSION 1 the old directory was written by this same layout,
        so adopting it is correct; at any later version it would be adopting
        bytes written by a different writer, which is the exact hazard the
        version exists to prevent. So this refuses outright past v1 -- a
        migration that outlives its own precondition is worse than no
        migration.
        """
        import os as _os
        if DISK_FORMAT_VERSION != 1:
            return
        try:
            legacy = FileMapper(
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
                    "torch.", ""),
                kv_cache_groups=[
                    {
                        "block_size": group.kv_cache_spec.block_size,
                        "layer_names": list(group.layer_names),
                    }
                    for group in self.kv_cache_config.kv_cache_groups
                ],
            )
        except Exception:
            logger.exception("kvmgr: could not compute the legacy base path")
            return

        old = "%s_r%d" % (legacy.base_path, self.rank)
        new = "%s_r%d" % (self._mapper.base_path, self.rank)
        if old == new or not _os.path.isdir(old):
            return
        if _os.path.exists(new):
            logger.info(
                "kvmgr: legacy cache at %s left in place -- the Phase 4 path "
                "%s already exists. Nothing is merged: two directories of the "
                "same layout are both valid, and picking files out of one is "
                "not something a rename can do atomically.", old, new)
            return
        try:
            _os.rename(old, new)
            logger.warning(
                "kvmgr: MIGRATED the disk tier to the Phase 4 fingerprint -- "
                "%s -> %s. Same layout (format v1), so every file and the "
                "manifest stay valid; this is a rename, not a rewrite, and it "
                "is what stops a stronger fingerprint from costing 400 GB of "
                "warm cache.", old, new)
        except OSError as exc:
            # Harmless: without the rename the tier simply starts empty under
            # the new path. A migration that can break a boot is worse than no
            # migration -- the same rule the geometry snapshot already follows.
            logger.warning(
                "kvmgr: legacy cache migration failed (%s). The tier will "
                "refill under the new fingerprint; nothing is lost but the "
                "warm bytes.", exc)

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
            mc = self.vllm_config.model_config
            # *** PHASE 4: OUR FORMAT VERSION AND THE MODEL'S CONTENT IDENTITY
            # *** GO INSIDE THE HASH. ***
            # `inference_engine` is the one free-form string FileMapper takes,
            # so it is where our format version belongs -- in the fingerprint
            # proper rather than only in the parent directory name. And the
            # model identity is appended to `model_name` because that field
            # also names the directory, so an operator reading the filesystem
            # can see WHICH checkpoint a cache belongs to instead of only a
            # sha prefix.
            self._content_id = model_content_id(
                model_dir(mc.model, getattr(mc, "revision", None)))
            if self._content_id == "unknown":
                logger.warning(
                    "kvmgr: could not identify the model's CONTENT (dir not "
                    "resolvable from model=%r) -- the cache fingerprint falls "
                    "back to the model NAME, exactly as before Phase 4. Not a "
                    "regression, but this cache is no better protected against "
                    "a re-quantised checkpoint at the same id than it was.",
                    mc.model)
            self._mapper = FileMapper(
                root_dir=self.root_dir,
                inference_engine="vllm+dspark-kv/v%d" % DISK_FORMAT_VERSION,
                model_name="%s@%s" % (mc.model, self._content_id),
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
            # *** `file_mapper` IS DELIBERATELY CALLABLE WITH A DUCK-TYPED
            # *** SELF, so the migration is an OPTIONAL side effect of it. ***
            # test_phase1d_disk.py invokes this function unbound against a
            # minimal fake carrying only the fields the path depends on -- that
            # is the test proving the worker and the engine core compute the
            # SAME base path, which is the 2026-10-07 fingerprint fork (552
            # keys pruned as "file is gone" with the files right there). Making
            # a path accessor require a full spec would have broken exactly the
            # test that guards the hardest bug this module has had.
            _mig = getattr(self, "_migrate_legacy_base", None)
            if _mig is not None:
                _mig(parallel_config)

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
            # ITS OWN row count: the load no longer shares the store's buffer.
            staging_blocks=self.staging_load_blocks,
            read_threads=self.write_threads, store=handler,
        )
        yield (DiskLoadStoreSpec, GPULoadStoreSpec, self._load_handler)
        logger.info(
            "kvmgr: DISK -> GPU handler registered unconditionally; "
            "`disk_restore` is now genuinely HOT because lookup() is the only "
            "gate. Currently %s. Withholding the handler used to be a second "
            "gate and it killed the engine instead -- see the note here.",
            "ON" if flags.peek("disk_restore") else "OFF")
