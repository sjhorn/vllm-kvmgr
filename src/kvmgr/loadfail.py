"""kvmgr.loadfail -- a failed disk restore becomes a recompute, not a dead engine.

*** WHY. *** Upstream's offloading connector does `assert transfer_result.success`
("we currently do not support job failures", `offloading/worker.py:273`), so
every way a restore can fail -- a missing file, a wrong size, a bad checksum,
a refused bound -- kills the engine core. That one assert is also why the boot
walk exists at all: every key has to be vouched for on every node before the
API opens, because a load that misses is fatal.

vLLM 0.21 already has the other half. A connector reports the GPU blocks it
could not fill through `get_block_ids_with_load_errors()`; the model runner
puts them in `KVConnectorOutput.invalid_block_ids`; the scheduler's
`_handle_invalid_blocks` rolls the affected requests back and, with
`kv_load_failure_policy="recompute"`, reschedules them ("fail" finishes them
with an error instead -- still no dead engine). The offloading connector simply
never reports anything. Borrowed from sparkcache (FujitsuPolycom), which does
the same through its own connector.

THREE PIECES, all off unless `disk_load_failure_recompute` is set:

  1. WORKER. `DiskLoadHandler` reports a failed job as completed (success=True,
     no size) and hands the job's GPU block ids, every group, to `note_failed`.
     `OffloadingConnector.get_block_ids_with_load_errors` drains them, in the
     same get_finished call that completes the job, so the scheduler sees the
     invalid blocks no later than the completion.

  2. SCHEDULER, the multi-group defect. Stock 0.21's
     `_update_requests_with_invalid_blocks` starts with
     `(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)` -- one KV
     group only ("TODO: add support for hybrid memory allocator"). DeepSeek V4
     has five, so the recovery path would itself raise ValueError. For more
     than one group we apply sparkcache's patch 031 semantics: any invalid block
     anywhere in a request's groups invalidates its whole external prefix, and
     it restarts from token 0. Conservative and correct for any layout; a
     single-group engine keeps upstream's code untouched. DSpark's mnt10
     scheduler has this function byte-identical to pristine (checked 2026-10-14).

  3. SCHEDULER, the index. The affected requests' load jobs name their keys
     (`OffloadingConnectorScheduler._jobs`). Those keys are dropped from the
     presence index (and journalled), or the retry would look up the same bad
     file and fail again for ever. The worker has already unlinked a provably
     bad file, so the next store rewrites it: the same self-heal as boot
     verification.

The aggregation across TP ranks (one rank fails, the other does not) is
upstream's: `KVOutputAggregator` unions `invalid_block_ids` over workers (read in
0.31's `kv_connector/utils.py`; 0.21's is not vendored here -- the cluster's
corruption gate is what proves it on this build).
"""

from __future__ import annotations

import threading

from vllm.logger import init_logger

from . import flags

logger = init_logger("vllm." + __name__)

# Worker process: GPU block ids of failed loads, until the connector drains them.
_ERRORS: set = set()
_LOCK = threading.Lock()
# Counters, plain ints (they reach /kv through msgpack).
STATS = {"installed_connector": False, "installed_scheduler": False,
         "noted_jobs": 0, "noted_blocks": 0, "drained_blocks": 0,
         "hma_requests": 0, "hma_tokens": 0, "forgot_keys": 0,
         "forget_errors": 0}


def enabled() -> bool:
    try:
        return bool(flags.peek("disk_load_failure_recompute"))
    except Exception:
        return False


def note_failed(block_ids) -> None:
    """A load job failed: its destination blocks hold nothing trustworthy."""
    ids = {int(b) for b in block_ids}
    with _LOCK:
        _ERRORS.update(ids)
        STATS["noted_jobs"] += 1
        STATS["noted_blocks"] += len(ids)


def drain() -> set:
    with _LOCK:
        out = set(_ERRORS)
        _ERRORS.clear()
        STATS["drained_blocks"] += len(out)
    return out


# -- 2. the multi-group invalid-block handler --------------------------------

def hma_invalid_blocks(sched, requests, invalid_block_ids, num_scheduled_tokens,
                       evict_blocks=True):
    """sparkcache 031's semantics: any invalid block in any group restarts the
    request from token 0. Same return shape as upstream's function."""
    affected, total, to_evict = set(), 0, set()
    for request in requests:
        req_id = request.request_id
        groups = sched.kv_cache_manager.get_block_ids(req_id)
        ids = {b for g in groups for b in g}
        if ids.isdisjoint(invalid_block_ids):
            continue
        computed = request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
        affected.add(req_id)
        total += computed
        request.num_computed_tokens = 0
        if evict_blocks:
            to_evict.update(ids)
    STATS["hma_requests"] += len(affected)
    STATS["hma_tokens"] += total
    return affected, total, to_evict


def forget_failed_loads(sched, req_ids) -> int:
    """Drop the keys of these requests' load jobs from the presence index."""
    if not req_ids:
        return 0
    try:
        from . import diskspec
        osched = (diskspec._LIVE_OFFLOAD_SCHED[-1]
                  if diskspec._LIVE_OFFLOAD_SCHED else None)
        if osched is None:
            return 0
        keys = set()
        for st in list(getattr(osched, "_jobs", {}).values()):
            if not getattr(st, "is_store", True) and st.req_id in req_ids:
                keys.update(bytes(k) for k in st.keys)
        mgr = getattr(osched, "manager", None)
        fn = getattr(mgr, "forget_failed_load", None)
        if not keys or fn is None:
            return 0
        n = int(fn(keys))
        STATS["forgot_keys"] += n
        return n
    except Exception:
        STATS["forget_errors"] += 1
        logger.exception("kvmgr: could not drop a failed load's keys from the "
                         "index; the retry may look them up again")
        return 0


def _install_scheduler() -> None:
    try:
        from vllm.v1.core.sched import scheduler as _sm
    except Exception:
        logger.info("kvmgr: no v1 Scheduler to patch for load failures")
        return
    cls = _sm.Scheduler
    orig = getattr(cls, "_update_requests_with_invalid_blocks", None)
    if orig is None or getattr(orig, "_kvmgr_wrapped", False):
        return

    def _update_requests_with_invalid_blocks(self, requests, invalid_block_ids,
                                             num_scheduled_tokens,
                                             evict_blocks=True):
        requests = list(requests)
        multi = False
        try:
            multi = len(self.kv_cache_config.kv_cache_groups) > 1
        except Exception:
            # no config to read: decide per request below
            multi = any(len(self.kv_cache_manager.get_block_ids(r.request_id)) > 1
                        for r in requests)
        if multi:
            out = hma_invalid_blocks(self, requests, invalid_block_ids,
                                     num_scheduled_tokens, evict_blocks)
        else:
            out = orig(self, requests, invalid_block_ids, num_scheduled_tokens,
                       evict_blocks)
        if out[0]:
            logger.warning(
                "kvmgr: %d request(s) had KV blocks that failed to load (%d "
                "tokens to recompute%s). Their disk keys leave the index.",
                len(out[0]), out[1],
                ", whole prefix: multi-group" if multi else "")
            forget_failed_loads(self, out[0])
        return out

    _update_requests_with_invalid_blocks._kvmgr_wrapped = True
    cls._update_requests_with_invalid_blocks = _update_requests_with_invalid_blocks
    STATS["installed_scheduler"] = True
    logger.info("kvmgr: Scheduler invalid-block handling made multi-group safe "
                "(sparkcache 031 semantics) -- only reached when a load fails")


# -- 1. the connector's report ----------------------------------------------

_CONNECTOR_MODULES = (
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector",
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.connector",
)


def _install_connector() -> None:
    import importlib
    cls = None
    for name in _CONNECTOR_MODULES:
        try:
            cls = getattr(importlib.import_module(name), "OffloadingConnector", None)
        except Exception:
            cls = None
        if cls is not None:
            break
    if cls is None:
        logger.info("kvmgr: no OffloadingConnector to report load errors through")
        return
    orig = getattr(cls, "get_block_ids_with_load_errors", None)
    if getattr(orig, "_kvmgr_wrapped", False):
        return

    def get_block_ids_with_load_errors(self):
        base = set(orig(self)) if orig is not None else set()
        return base | drain()

    get_block_ids_with_load_errors._kvmgr_wrapped = True
    cls.get_block_ids_with_load_errors = get_block_ids_with_load_errors
    STATS["installed_connector"] = True


def install() -> None:
    """Both patches, always: each only acts when a load has failed, and the
    worker only reports failures that way while the knob is on."""
    for fn in (_install_connector, _install_scheduler):
        try:
            fn()
        except Exception:
            logger.exception("kvmgr: %s failed (continuing)", fn.__name__)


def snapshot() -> dict:
    with _LOCK:
        return dict(STATS, enabled=enabled(), pending_blocks=len(_ERRORS))
