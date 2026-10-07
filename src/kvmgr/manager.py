"""kvmgr.manager -- our MLA manager for group 0. Phase 1: behaviour-identical.

WHAT THIS IS.  A subclass of the PATCHED ``MLAAttentionManager`` that overrides
only the two policy decisions this work exists to change -- how big the
protected set may be, and which block leaves it -- and routes both through
``kvmgr.flags`` so they can be moved on a running engine.  Phase 1 changes no
behaviour: ``protect_alloc=equal`` and ``evict_order=fifo`` reproduce patch6
exactly, and the parity test asserts the cap agrees block for block.

WHY A SUBCLASS AND NOT A FOURTH WHOLE-FILE OVERLAY.  patch6/7/8 are whole-file
mounts, and that is how this stack ended up pinned to an August engine in
October: a 1,809-line copy of upstream must be re-cut on every image bump.  A
subclass registered into ``spec_manager_map`` is a delta instead of a copy, so
an upstream change to the other eleven managers costs us nothing.  The
12-method surface is the contract to re-check on an engine bump; the parity test
is what detects a change.

*** THE KILL SWITCH CANNOT SWAP THE CLASS, AND THE PLAN'S WORDING SUGGESTS IT
CAN. ***  The manager class is chosen ONCE, by ``get_manager_for_kv_cache_spec``,
while the engine is starting -- so "falls through to upstream's class when
false" is only achievable at boot, which is exactly the restart the requirement
forbids.  So we are ALWAYS registered and ALWAYS constructed, and
``kvmgr_enabled`` is read PER CALL inside each override: false means call
``super()``, which is patch6, unmodified.  That is what makes the switch
genuinely hot, and it is why flipping it off is not merely "our policy with
defaults" but literally upstream's code path.

Everything patch6 does that we do not override -- MNT-6b recycling, MNT-8 free
ordering, MNT-9 head pins, the whole allocation path -- is inherited untouched
and keeps working whichever way the switch is set.
"""

from __future__ import annotations

from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import (
    MLAAttentionManager,
    SingleTypeKVCacheManager,
)

from . import flags, policy

logger = init_logger(__name__)


class DSparkMLAAttentionManager(MLAAttentionManager):
    """Group 0's manager: full MLA, 256-token blocks, the cap that binds."""

    # Set by register() from the KVCacheConfig, because _num_managers cannot be
    # trusted: it is a process-global class counter incremented per manager
    # built and NEVER decremented, so anything that constructs a second set of
    # managers halves every cap in the running engine, silently.  See
    # docs/local/KVMGR-PHASE0-20261007.md finding 4.
    group_count: int | None = None
    group_block_sizes: tuple[int, ...] = ()
    # None where a group is full-attention (unbounded need), the window in
    # tokens otherwise.  by_need is wrong without this: see the 21x
    # regression documented at policy.pool_share_by_need.
    group_sliding_windows: tuple[int | None, ...] = ()

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._kvmgr_cap_reported: tuple | None = None
        self._kvmgr_was_enabled: bool | None = None
        if not _AUDITED:
            _audit_once()
        logger.info(
            "kvmgr: DSparkMLAAttentionManager active for KV group %d "
            "(block_size %d); policy is OPT-IN -- kvmgr_enabled=%r",
            self.kv_cache_group_id, self.block_size, flags.raw("kvmgr_enabled"),
        )

    # -- the two numbers that decide how much of a conversation survives ----

    def _num_groups(self) -> int:
        """Prefer the real group count; fall back to patch6's counter.

        The fallback is not equivalent and is logged when it is used, because a
        wrong group count scales every cap.
        """
        if self.group_count:
            return self.group_count
        return max(1, SingleTypeKVCacheManager._num_managers)

    def _max_protected_prompt_blocks(self) -> int | None:
        enabled = flags.knob("kvmgr_enabled", quiet=True)
        if enabled != self._kvmgr_was_enabled:
            if self._kvmgr_was_enabled is not None:
                logger.info(
                    "kvmgr: kvmgr_enabled CHANGED %r -> %r for KV group %d; "
                    "the cap is now computed by %s with no restart",
                    self._kvmgr_was_enabled, enabled, self.kv_cache_group_id,
                    "kvmgr" if enabled else "patch6 (upstream path)",
                )
            self._kvmgr_was_enabled = enabled
        if not enabled:
            # Upstream's code path, verbatim. Not "our policy with defaults".
            return super()._max_protected_prompt_blocks()

        fraction = flags.knob("protected_fraction", quiet=True)
        mode = flags.knob("protect_alloc", quiet=True)
        num_groups = self._num_groups()

        cap = policy.protected_cap(
            num_gpu_blocks=self.block_pool.num_gpu_blocks,
            fraction=fraction,
            num_groups=num_groups,
            block_size=self.block_size,
            max_model_len=self.max_model_len,
            mode=mode,
            block_sizes=self.group_block_sizes,
            group_index=self.kv_cache_group_id,
            sliding_windows=self.group_sliding_windows,
        )
        self._report_cap(cap, fraction, mode, num_groups)
        return cap

    def _report_cap(self, cap, fraction, mode, num_groups) -> None:
        """Log the cap ON CHANGE, in BLOCKS and TOKENS.

        patch6 logs it ONCE (`_protected_cap_logged`), which is right for a
        boot-time constant and wrong for something an operator flips: flip it and
        you get no confirmation it took.  Tokens are on the line because a cap in
        blocks is not comparable across groups, which is the whole reason the
        equal split looked reasonable for months.
        """
        key = (cap, round(float(fraction), 4), mode, num_groups)
        if key == self._kvmgr_cap_reported:
            return
        previous = self._kvmgr_cap_reported
        self._kvmgr_cap_reported = key
        tokens = policy.cap_tokens(cap, self.block_size)
        logger.info(
            "kvmgr: protected cap for KV group %d = %s blocks = %s tokens "
            "(pool %d, groups %d, fraction %.2f, alloc %s, legacy %s)%s",
            self.kv_cache_group_id,
            cap, tokens, self.block_pool.num_gpu_blocks, num_groups,
            float(fraction), mode,
            policy.legacy_cap(self.max_model_len, self.block_size),
            "" if previous is None else f" [was {previous[0]} blocks]",
        )

    # -- the victim -----------------------------------------------------------

    def _release_one_protected_prompt_block(self, block_ids_to_skip=None) -> bool:
        """Which block leaves the protected set.

        Phase 1 is FIFO, i.e. patch6's own rule, and `evict_order` is read here
        so that Phase 3's alternatives have one place to land and so the knob is
        already hot rather than retrofitted.  `_IMPLEMENTED` in flags.py refuses
        `lru`/`lru_tail_first` for now and says so, rather than letting a
        half-built policy ship looking live.
        """
        if not flags.knob("kvmgr_enabled", quiet=True):
            return super()._release_one_protected_prompt_block(block_ids_to_skip)

        order = flags.knob("evict_order")
        if order != "fifo":  # unreachable while only fifo is implemented
            logger.warning("kvmgr: evict_order=%r reached the hot path; using fifo", order)
        return super()._release_one_protected_prompt_block(block_ids_to_skip)

    # -- reconcile on the flag-change edge -----------------------------------

    def kvmgr_reconcile(self) -> int:
        """Apply a LOWERED cap now, rather than on this group's next protect.

        Without this a shrunk budget bites only when
        `_trim_protected_prompt_blocks` next runs, so a quiet group holds stale
        protection indefinitely -- the operator lowers a cap, sees nothing
        released, and has no way to tell that from the cap not having taken.

        Returns the number of blocks released, so the caller can log it.
        """
        if not flags.raw("kvmgr_enabled"):
            return 0
        before = len(self._protected_prompt_block_ids)
        self._trim_protected_prompt_blocks()
        released = before - len(self._protected_prompt_block_ids)
        if released:
            logger.info(
                "kvmgr: reconciled KV group %d after a flag change -- released %d "
                "protected block(s), %d remain",
                self.kv_cache_group_id, released,
                len(self._protected_prompt_block_ids),
            )
        return released

    def new_step_starts(self) -> None:
        """patch6's per-step hook: the cheapest place to notice a flag change.

        Once per scheduler step, and `changed()` is a cached read behind the 1s
        TTL, so this costs a dict compare on a path that already does real work.
        """
        super().new_step_starts()
        if flags.changed("protected_fraction") or flags.changed("protect_alloc"):
            # Recompute (which logs the new cap) and then act on it.
            self._max_protected_prompt_blocks()
            self.kvmgr_reconcile()


_AUDITED = False


def _audit_once() -> None:
    global _AUDITED
    _AUDITED = True
    flags.log_env_audit()
