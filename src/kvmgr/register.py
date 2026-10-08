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

    _install_hang_dump()

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

        from kvmgr.diskspec import DSPARK_SPEC_NAME
        OffloadingSpecFactory.register_spec(
            DSPARK_SPEC_NAME, "kvmgr.diskspec",
            DSPARK_SPEC_NAME,
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



def _install_hang_dump() -> None:
    """Make a HANG dumpable with no ptrace, because twice now it was not.

    *** THE PROBLEM THIS SOLVES. *** The store path has wedged the worker twice
    (Phase 1c, and again 2026-10-07 at the staging-saturation point). Both times
    the only evidence available was "the worker stopped consuming shm broadcasts"
    plus thread wchans, because the container has no `SYS_PTRACE` -- `gdb -p`
    answers `ptrace: Operation not permitted` and there is no py-spy. So a
    deadlock in OUR code was diagnosable only by reading it and guessing, which
    is how the Phase 1c attempt came to be built on a wrong premise.

    `faulthandler` needs no ptrace and no attach: the signal handler runs inside
    the process and walks every thread's Python stack itself. Output goes to
    stderr, which is what `docker logs` shows, so the dump lands beside the log
    lines that led up to it.

        docker exec <c> kill -USR1 <worker pid>   # every thread's Python stack

    SIGUSR1 rather than SIGABRT or SIGQUIT: those two kill the process, and a
    wedged engine holding a 700K-token prefix is worth more alive than dead --
    the dump is for diagnosis, not for a postmortem. Registered per process from
    the plugin, so the worker, the engine core and the API server all answer it.

    *** `chain=True` KILLS THE PROCESS, AND THE TEST IS WHAT FOUND IT. ***
    faulthandler chains to whatever handler was installed when it registered,
    and for SIGUSR1 that is normally `SIG_DFL` -- whose default action is
    TERMINATE. So the obvious spelling dumps the stack and then kills the
    engine, which on a box holding a 700K-token prefix would destroy the thing
    being diagnosed in the act of diagnosing it. Measured offline: the child
    exited with `-30`, i.e. died of its own diagnostic signal.

    So chain only to a REAL handler -- one somebody actually installed -- and
    never to the default. Checked in the running image 2026-10-07: vLLM
    installs no SIGUSR1 or SIGUSR2 handler anywhere (0 hits across its tree),
    so in practice nothing is chained and nothing is stolen; the check is there
    so that a future vLLM taking SIGUSR1 keeps working rather than losing it.

    The whole thing is best-effort: an instrument that can break the boot is
    worse than no instrument.
    """
    try:
        import faulthandler
        import signal
        import sys as _sys

        if not hasattr(signal, "SIGUSR1"):
            return
        existing = signal.getsignal(signal.SIGUSR1)
        chain = callable(existing) and existing not in (
            signal.SIG_DFL, signal.SIG_IGN)
        faulthandler.register(signal.SIGUSR1, file=_sys.stderr,
                              all_threads=True, chain=chain)
        logger.info(
            "kvmgr: hang dump armed -- `kill -USR1 <pid>` prints every "
            "thread's Python stack to the container log. No ptrace needed, "
            "which is the point: gdb cannot attach in this container and the "
            "store path has wedged the worker twice with no stack available. "
            "chain=%s (a real handler was already installed: %s).",
            chain, chain,
        )
    except Exception:
        logger.exception("kvmgr: could not arm the hang dump (continuing)")

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
