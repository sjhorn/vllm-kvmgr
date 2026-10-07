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
    # docs/local/KVMGR-PHASE0-20261007.md finding 1.
    stm.spec_manager_map[MLAAttentionSpec] = DSparkMLAAttentionManager
    logger.info(
        "kvmgr: registered %s for MLAAttentionSpec (was %s); group 0 only, "
        "policy still gated on kvmgr_enabled",
        DSparkMLAAttentionManager.__name__,
        getattr(previous, "__name__", previous),
    )

    _install_group_geometry_hook(DSparkMLAAttentionManager)

    # Opt-in, and it logs nothing unless offload_allow_hybrid is set. Installed
    # from register() so it lands in every process that loads the plugin --
    # the scheduler side is the one that raises.
    from . import offload

    offload.install()

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

        OffloadingSpecFactory.register_spec(
            "DSparkDiskOffloadingSpec", "kvmgr.diskspec",
            "DSparkDiskOffloadingSpec",
        )
        logger.info(
            "kvmgr: registered DSparkDiskOffloadingSpec (inert until "
            "KV_TRANSFER_CONFIG names it as spec_name)"
        )
    except ValueError:
        pass          # already registered: register() ran twice in one process
    except Exception:
        # A missing kv_offload is not a reason to lose the KV manager, which is
        # what this plugin is actually for.
        logger.exception("kvmgr: could not register the disk offloading spec")

    _REGISTERED = True


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
        return original(self, kv_cache_config, *args, **kwargs)

    __init__._kvmgr_wrapped = True  # type: ignore[attr-defined]
    base.__init__ = __init__  # type: ignore[method-assign]
