"""kvmgr.register -- the vllm.general_plugins entry point.

DELIVERY.  A general plugin, discovered through the entry-point group
``vllm.general_plugins``, exactly as ``patches/safetensors_e8m0_plugin`` and the
dispram carveout are.  That keeps this off the whole-file mount list: a fourth
1,809-line copy of an upstream file is how this stack ended up pinned to an
August engine in October.

WHY MUTATING A DICT IS ENOUGH.  ``get_manager_for_kv_cache_spec`` reads
``spec_manager_map[type(kv_cache_spec)]`` at CALL time, and the managers are
built inside ``KVCacheCoordinator.__init__`` in the engine-core process, which
runs after plugins load.  So an import-time mutation is picked up.  Asserted
offline in test_phase0.py rather than assumed.

WHY WE ALSO WRAP THE COORDINATOR.  Two caps divide by
``SingleTypeKVCacheManager._num_managers``, a process-global class counter that
is incremented per manager built and never decremented -- so anything that
constructs a second set of managers halves every cap in the running engine,
silently.  The coordinator's ``__init__`` is the one place that holds the real
``KVCacheConfig``, so we read the group count and the per-group block sizes off
it and stamp them on our class.  The counter stays as the fallback and says so
when it is used.

The wrap is also what makes ``protect_alloc=by_need`` possible at all: it needs
every group's block size, and a single manager only knows its own.
"""

from __future__ import annotations


from vllm.logger import init_logger

from . import hooks

# *** THE LOGGER NAME IS LOAD-BEARING, and `__name__` is the wrong one. ***
#
# vLLM's logging config gives a handler to the `vllm` logger ONLY, with
# propagate=false, and configures no root handler. So a logger named
# `kvmgr.flags` has NO handler and an EFFECTIVE LEVEL OF 30 (WARNING) --
# every logger.info() it makes is silently dropped. Measured in the running
# container 2026-10-07:
#
#   init_logger("kvmgr.register")       handlers [] effective 30  -> dropped
#   init_logger("vllm.kvmgr.register")  handlers [] effective 20  -> printed
#
# That cost us the whole "did it take?" story on the first deploy: the
# registration line, the cap-on-change line, the flag-change line and the env
# audit were all invisible, which is indistinguishable from the plugin never
# having loaded. Naming the logger under `vllm.` makes it inherit that handler.
logger = init_logger("vllm." + __name__)

_REGISTERED = False


def register() -> None:
    """Idempotent: plugins load once per process, but there are several
    processes and a re-entrant import is cheaper to tolerate than to debug."""
    global _REGISTERED
    if _REGISTERED:
        return

    _install_diagnostics()
    _install_control_plane()
    _install_fork_bridges()

    from vllm.v1.core import single_type_kv_cache_manager as stm
    from vllm.v1.kv_cache_interface import MLAAttentionSpec

    from .manager import DSparkMLAAttentionManager

    previous = stm.spec_manager_map.get(MLAAttentionSpec)
    if previous is DSparkMLAAttentionManager:
        _REGISTERED = True
        return

    # Phase 1 owns GROUP 0 ONLY -- MLAAttentionSpec, the full-MLA group whose
    # cap is the one that binds.  The other four spec types stay on upstream's
    # classes, which is both the narrow rollout and the fallback if this is
    # wrong.
    #
    # NOTE for Phase 5: this is NOT sufficient for disk restore.  The hybrid
    # coordinator converges the hit length to the MINIMUM across all five
    # groups, so restoring only group 0 buys nothing -- every spec type has to
    # be registered before a restore can lengthen a hit.  See
    # docs/history/KVMGR-PHASE0-20261007.md finding 1.
    stm.spec_manager_map[MLAAttentionSpec] = DSparkMLAAttentionManager
    logger.info(
        "kvmgr: registered %s for MLAAttentionSpec (was %s); group 0 only, "
        "policy still gated on kvmgr_enabled",
        DSparkMLAAttentionManager.__name__,
        getattr(previous, "__name__", previous),
    )

    _install_group_geometry_hook(DSparkMLAAttentionManager)
    _install_offload_scheduler_hook()
    _install_serializer_shim()

    # The tiering shims (O_DIRECT, the hybrid relax, verify) moved to
    # kvmgr.diag.tiering and install from _install_diagnostics() above, off by
    # default: they patch upstream's TieringOffloadingSpec, which ours is not.

    # *** OUR OWN OffloadingSpec, registered by NAME through the factory's public
    # registry -- no upstream file is patched. *** It exists because upstream's
    # tiering offload cannot capture half of this cluster's KV: it consolidates
    # every rank's shard into ONE SHARED HOST region, which is single-host TP by
    # construction, while ours is one rank per node. Measured: every .bin the
    # tiering tier wrote has rank>0's half EXACTLY ZERO.
    #
    # Registering the name is INERT. Nothing selects it unless
    # KV_TRANSFER_CONFIG names it as spec_name, and that lives in .env.dspark
    # per node -- so the committed default stays "no offload at all", and a
    # restart without it reverts to known-good rather than to a half-built path.
    #
    # Import is DEFERRED into the factory's own resolution (module_path +
    # class_name) rather than imported here, so a syntax error in diskspec.py
    # cannot take down the manager registration that every boot depends on.
    try:
        from vllm.v1.kv_offload.factory import OffloadingSpecFactory

        from kvmgr.diskspec import DSPARK_SPEC_NAME, SPEC_NAMES
        for _name in SPEC_NAMES:
            try:
                # Every name resolves to the ONE class.
                OffloadingSpecFactory.register_spec(
                    _name, "kvmgr.diskspec", DSPARK_SPEC_NAME)
            except ValueError:
                pass  # already registered: register() ran twice in one process
        logger.info(
            "kvmgr: registered the disk offloading spec as %s (inert until "
            "KV_TRANSFER_CONFIG names one of them as spec_name)",
            " / ".join(SPEC_NAMES))
    except Exception:
        # A missing kv_offload is not a reason to lose the KV manager, which is
        # what this plugin is actually for.
        logger.exception("kvmgr: could not register the disk offloading spec")

    _REGISTERED = True




# Diagnostic module -> its knob. THE ONLY PLACE IN kvmgr's CORE THAT IMPORTS
# kvmgr.diag, and every import is behind its knob: a lint in the tests parses
# the package and fails on any other site. Order matters only for `tiering`,
# whose shims patch upstream's tiering spec before anything builds one.
_DIAGNOSTICS = (
    ("hangdump", "diag_hang_dump"),
    ("ledger", "diag_transfer_ledger"),
    ("serializer", "diag_serializer_walk"),
    ("geometry", "diag_geometry_snapshot"),
    ("tiering", "diag_tiering_shims"),
    ("reload", "diag_hot_reload"),
)


def _install_diagnostics() -> dict:
    """Install each opt-in diagnostic whose knob is on. Best-effort, per item.

    Read at plugin load, so these are BOOT-TIME knobs: the value is hot (the
    file is re-read) and the effect is not, exactly like kvmgr_control_plane.
    Returns {module: installed?} so a reload can report what it re-armed.
    """
    import importlib

    from . import flags

    out = {}
    for mod, knob in _DIAGNOSTICS:
        try:
            if not flags.peek(knob):
                out[mod] = False
                continue
            importlib.import_module(f"kvmgr.diag.{mod}").install()
            out[mod] = True
        except Exception:
            out[mod] = False
            logger.exception("kvmgr: diagnostic %s failed to install "
                             "(continuing -- it is a diagnostic)", mod)
    off = sorted(m for m, on in out.items() if not on)
    logger.info("kvmgr: diagnostics on: %s; off: %s",
                sorted(m for m, on in out.items() if on) or "none", off or "none")
    return out


def _install_control_plane() -> None:
    """Install the HTTP control plane's three hooks, best-effort.

    Three, because three kinds of process read these knobs and each needs a
    different one: the Worker methods are the RPC targets, the EngineCore wrap
    is what makes the SCHEDULER apply a knob too (the executor broadcasts only
    to workers), and the router hook attaches our routes to the app the API
    server builds. Every one is a no-op in a process that lacks its target, so
    this runs unconditionally and the misses are silent by design.

    Best-effort throughout: a control plane that can break a boot is worse than
    no control plane, which is the same rule the geometry snapshot and the hang
    dump already follow.
    """
    try:
        from . import control

        control.install_worker_methods()
        control.wrap_engine_core_rpc()
        control.wrap_engine_core_init()
        control.install_router_hook()
    except Exception:
        logger.exception("kvmgr: control plane install failed (continuing; the "
                         "flags file still governs every knob)")






def _install_fork_bridges() -> None:
    """Bridge to a vLLM fork's own overlays, if one is loaded.

    Today that is the DSpark recipe's patch6 (one flags cache, overrides that
    reach the fork's MNT-8/9 knobs, its pending-blocks hook). On any other
    vLLM this finds nothing and does nothing. Best-effort: a bridge that can
    break a boot is worse than none.
    """
    try:
        from .compat import dspark

        dspark.install()
    except Exception:
        logger.exception("kvmgr: fork bridge install failed (continuing)")


def _install_serializer_shim() -> None:
    """Make a numpy SCALAR serialisable, and say where it came from.

    *** THIS KILLED THE ENGINE'S OUTPUT THREAD TWICE AND PRESENTED AS A HANG.
    *** `vllm/v1/serial_utils.py`'s `enc_hook` handles `np.ndarray` but NOT
    numpy scalars -- `np.int64` is not an ndarray -- so a single numpy integer
    anywhere in `EngineCoreOutputs` raises:

        TypeError: Object of type <class 'numpy.int64'> is not serializable
          core.py:1561 in process_output_sockets

    And it kills that THREAD, not the engine. Results then never reach the API
    server: requests complete, the core goes idle in `queue.get()` on an empty
    queue, clients hang for ever, and the main loop logs NOTHING. I spent an
    evening reading that as a memory ceiling.

    *** WHY A SHIM AND NOT JUST A FIX AT THE SOURCE. *** I did fix our source
    (`file_plan` coerces now) and it recurred, so at least one numpy scalar
    reaches the encoder from somewhere else -- possibly not ours at all. A
    lossless `.item()` is strictly better than killing the output thread
    whoever produced it, and the alternative on offer
    (`VLLM_ALLOW_INSECURE_SERIALIZATION=1`) buys the same robustness by
    enabling pickle for everything, which is a much larger blast radius.

    It is also the only instrument that can NAME the culprit: the hook is
    handed the offending object, so one log line carries its type and value,
    which no traceback does. Logged once per distinct type so a hot path cannot
    flood.
    """
    try:
        import numpy as _np
        from vllm.v1 import serial_utils as _su
    except Exception:
        return
    try:
        cls = _su.MsgpackEncoder
    except Exception:
        logger.info("kvmgr: no MsgpackEncoder to shim (upstream layout differs)")
        return
    try:
        # *** A SHIM THAT CAN ONLY BE INSTALLED ONCE PER PROCESS IS A
        # *** DIAGNOSTIC THAT NEEDS A RESTART TO CHANGE. ***
        # The early return here was correct about idempotence and wrong about
        # UPDATES: `importlib.reload` re-executes this module, but the class
        # attribute still held the OLD closure, so every change to the shim or
        # the producer scan cost a ~4-minute boot plus a cold prefill. That
        # happened twice in one session -- once for the scan's window, once for
        # its own observability. Same family as "a knob whose OFF state needs a
        # boot is not a knob", and as "THE RELOADER IS ITSELF THE OLD VERSION".
        #
        # So UNWRAP FIRST, then wrap fresh. The original is carried on the
        # wrapper itself, which is the only place that survives a reload --
        # a module global would be reset by the very reload we are serving.
        original = cls.enc_hook
        if getattr(original, "_kvmgr_wrapped", False):
            original = getattr(original, "_kvmgr_original", None)
            if original is None:
                # Wrapped by something that is not us, or by a version that
                # did not record its original. Refuse rather than wrap a
                # wrapper: stacking them would call the old scan as well.
                logger.warning(
                    "kvmgr: enc_hook is already wrapped and carries no "
                    "original -- leaving it alone. The shim is live; a CHANGE "
                    "to it needs a restart.")
                return
        seen: set = set()

        def enc_hook(self, obj):
            if isinstance(obj, _np.generic):
                name = type(obj).__name__
                if name not in seen:
                    seen.add(name)
                    # WHO produced it is a diagnostic (kvmgr.diag.serializer,
                    # off by default); THAT it is coerced is the fix, and core.
                    _producer = hooks.first("numpy_producer", obj)
                    if _producer:
                        logger.warning("kvmgr: numpy scalar PRODUCER -- %s",
                                       _producer)
                    logger.warning(
                        "kvmgr: coerced a numpy scalar in the engine's OUTPUT "
                        "path -- numpy.%s value=%r -> %r. Upstream's enc_hook "
                        "handles ndarrays but not scalars, and the raise kills "
                        "process_output_sockets, which strands every result "
                        "and presents as a hang with no error. Lossless, but "
                        "the PRODUCER should be fixed: this value is the clue.",
                        name, obj, obj.item())
                return obj.item()
            return original(self, obj)

        enc_hook._kvmgr_wrapped = True
        enc_hook._kvmgr_original = original
        cls.enc_hook = enc_hook
        logger.info(
            "kvmgr: serializer shim installed -- a numpy scalar in the output "
            "path is now coerced and logged once with its value, instead of "
            "killing the output socket thread and stranding every result")
    except Exception:
        logger.exception("kvmgr: could not install the serializer shim")


def _install_offload_scheduler_hook() -> None:
    """Publish upstream's OffloadingConnectorScheduler so /kv can read its jobs.

    *** THE ONE NUMBER THAT WAS MISSING. *** The worker reports `jobs_open: 0`
    while the scheduler can still be holding a hundred jobs, and the
    scheduler's `self._jobs` is exactly what `jobs_to_flush` names -- so the
    fence waits on jobs the worker has already finished and forgotten. Nothing
    could see that: `jobs_to_flush` appeared only in a crash's dump_input, which
    is a terrible place to learn it.

    `complete_store` fires only when `pending_count` hits 0 and it starts at
    `num_workers`, so BOTH ranks must report. A histogram of pending_count over
    the residual jobs distinguishes "one rank's completion is lost" from
    "neither arrived" in a single reading, which is why it is worth a hook.

    Wrapping __init__ rather than hunting the object: same pattern as the
    coordinator wrap below, and it is the only place guaranteed to run before
    any job exists. Best-effort throughout -- an instrument that can break a
    boot is worse than no instrument, and this one reaches into another
    project's private state.
    """
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            scheduler as _osched,
        )
    except Exception:
        logger.info("kvmgr: no OffloadingConnectorScheduler to hook "
                    "(upstream layout differs); /kv will report sched "
                    "unavailable rather than zero")
        return
    try:
        cls = _osched.OffloadingConnectorScheduler
        if getattr(cls.__init__, "_kvmgr_wrapped", False):
            return
        original = cls.__init__

        def __init__(self, *a, **kw):
            original(self, *a, **kw)
            try:
                from kvmgr import diskspec as _ds
                # Mutate in place: a reload may already have handed the new
                # module's list to somebody else.
                del _ds._LIVE_OFFLOAD_SCHED[:]
                _ds._LIVE_OFFLOAD_SCHED.append(self)
                logger.info(
                    "kvmgr: offload scheduler published -- /kv now reports its "
                    "open job set and the pending_count histogram, which is "
                    "what says whether a store completion never arrived")
            except Exception:
                logger.exception("kvmgr: could not publish the offload scheduler")

        __init__._kvmgr_wrapped = True
        cls.__init__ = __init__
    except Exception:
        logger.exception("kvmgr: could not hook the offload scheduler (continuing)")


def _install_group_geometry_hook(manager_cls) -> None:
    """Stamp the real group count / block sizes from the KVCacheConfig."""
    from vllm.v1.core import kv_cache_coordinator as kcc

    base = kcc.KVCacheCoordinator
    if getattr(base.__init__, "_kvmgr_wrapped", False):
        return

    original = base.__init__

    def __init__(self, kv_cache_config, *args, **kwargs):
        try:
            groups = list(kv_cache_config.kv_cache_groups)
            manager_cls.group_count = len(groups)
            manager_cls.group_block_sizes = tuple(
                g.kv_cache_spec.block_size for g in groups
            )
            # `sliding_window` is absent on a full-attention spec and present on
            # the windowed ones.  getattr-with-None is the discriminator, and it
            # is what makes by_need correct rather than a 21x regression.
            manager_cls.group_sliding_windows = tuple(
                getattr(g.kv_cache_spec, "sliding_window", None) for g in groups
            )
            logger.info(
                "kvmgr: KV geometry from KVCacheConfig -- %d groups, block sizes %s, "
                "windows %s, pool %d blocks",
                manager_cls.group_count, manager_cls.group_block_sizes,
                manager_cls.group_sliding_windows, kv_cache_config.num_blocks,
            )
        except Exception:
            # Never take the engine down for observability. The counter
            # fallback in _num_groups() covers us, and logs that it did.
            logger.exception(
                "kvmgr: could not read the KV geometry off KVCacheConfig; "
                "falling back to _num_managers, which is a process-global "
                "counter and may be wrong"
            )
        out = original(self, kv_cache_config, *args, **kwargs)
        # *** AND STASH EVERY MANAGER, because a POOL-WIDE operation needs
        # POOL-WIDE releases. *** Measured 2026-10-08: `gpu_cache_epoch` fired,
        # released group 0's protection and was REFUSED, with the log naming the
        # cause in one number -- "pool free blocks 22536 -> 22581", and +45 is
        # exactly group 0's share of an 11,764-token prompt (11764/256 = 46).
        # We are registered for MLAAttentionSpec only, so our instance can
        # release only its own group; the other four run upstream's
        # SlidingWindowMLAManager and held ~1,700 blocks that patch7's
        # `num_used_blocks != 1` gate then refused on, forever.
        #
        # `release_protected_prompt_blocks` is defined on
        # SingleTypeKVCacheManager (patch6:691), so it is available on every
        # manager including the ones we did not replace. This is the only place
        # that holds the whole tuple -- the coordinator builds it at :67 -- which
        # is why the stash goes here rather than being discovered later.
        try:
            manager_cls.all_managers = tuple(
                getattr(self, "single_type_managers", ()) or ())
            logger.info(
                "kvmgr: holding %d manager(s) for pool-wide releases "
                "(ours: %d)",
                len(manager_cls.all_managers),
                sum(1 for m in manager_cls.all_managers
                    if isinstance(m, manager_cls)),
            )
        except Exception:
            logger.exception(
                "kvmgr: could not stash the manager list; a GPU cache drop will "
                "release only group 0 and patch7 will refuse the reset")
        return out

    __init__._kvmgr_wrapped = True  # type: ignore[attr-defined]
    base.__init__ = __init__  # type: ignore[method-assign]
