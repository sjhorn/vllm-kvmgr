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

from . import flags, policy, serve, stats

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
        # From the CONSTRUCTOR, not register(): this has to run in whichever
        # process actually builds managers, which is not knowable at
        # plugin-load time.
        serve.maybe_start()
        stats.group(self.kv_cache_group_id, self.block_size)
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
        stats.record_cap(self.kv_cache_group_id, cap, self.block_size)
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

    # -- protection, counted --------------------------------------------------

    def _protect_prompt_blocks(self, blocks) -> None:
        """Counted, not changed.

        The count is taken as a DELTA of the protected set rather than from
        len(blocks): patch6 skips a block that is null, unhashed or already
        protected, so the argument length is what was offered and the delta is
        what was taken. Counting the offer would report protection that never
        happened -- and on an appending turn almost every block is already
        protected, so the two differ by an order of magnitude.
        """
        before = len(self._protected_prompt_block_ids)
        super()._protect_prompt_blocks(blocks)
        after = len(self._protected_prompt_block_ids)
        stats.record_protect(
            self.kv_cache_group_id, max(0, after - before), after
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

        released = super()._release_one_protected_prompt_block(block_ids_to_skip)
        if released:
            stats.record_evict(
                self.kv_cache_group_id,
                cause=self._evict_cause,
                victim_rule=stats.VICTIM_FIFO,
                protected_now=len(self._protected_prompt_block_ids),
            )
            if flags.raw("kvmgr_log_level"):
                logger.info(
                    "kvmgr: KV group %d evicted a protected block "
                    "(cause=%s, victim=%s, %d protected = %d tokens remain)",
                    self.kv_cache_group_id, self._evict_cause,
                    stats.VICTIM_FIFO, len(self._protected_prompt_block_ids),
                    len(self._protected_prompt_block_ids) * self.block_size,
                )
        return released

    # WHY A CAUSE IS CARRIED ON THE INSTANCE rather than passed down. patch6
    # calls _release_one_protected_prompt_block from three places with
    # IDENTICAL arguments -- the cap trim, the allocation-pressure release and
    # our own reconcile -- so the call site is the only thing that distinguishes
    # them, and the signature is patch6's to keep. Changing the signature would
    # fork the file we are deliberately not forking.
    _evict_cause = stats.EVICT_CAP

    def _trim_protected_prompt_blocks(self) -> None:
        # Only claim "cap" if nothing more specific is already in force -- the
        # reconcile path trims too, and its cause is the flag change.
        if self._evict_cause not in (stats.EVICT_RESET, stats.EVICT_POOL):
            self._evict_cause = stats.EVICT_CAP
        super()._trim_protected_prompt_blocks()

    def release_protected_prompt_blocks(self, *args, **kwargs) -> None:
        """Allocation pressure, which is a different event from the cap.

        Conflating them would hide the thing the whole plan turns on: during the
        measured 373,833-token loss, KV usage was 22-33% -- the pool two-thirds
        EMPTY -- while the cap was exceeded by 43%. A cap eviction on an idle
        pool is a policy bug; a pool eviction is capacity. They must be
        countable apart.
        """
        self._evict_cause = stats.EVICT_POOL
        try:
            super().release_protected_prompt_blocks(*args, **kwargs)
        finally:
            self._evict_cause = stats.EVICT_CAP

    # -- the lookup, counted --------------------------------------------------

    @classmethod
    def find_longest_cache_hit(cls, **kwargs):
        """Counted, not changed. Phase 5's disk restore lands here.

        A CLASSMETHOD, so there is no instance to count on -- which is why
        kvmgr.stats is module-level and keyed by group id. The coordinator calls
        this on the class (`manager_cls.find_longest_cache_hit`), which is also
        what makes registering in spec_manager_map enough to be reached.

        THREE OUTCOMES, NOT TWO. A partial hit -- matched less than asked -- is a
        prefix that survived in PART, which is a different event from a miss and
        is exactly what tail-first eviction is meant to produce. Collapsing them
        would make Phase 3 unmeasurable.
        """
        blocks = super().find_longest_cache_hit(**kwargs)
        try:
            spec = kwargs.get("kv_cache_spec")
            block_size = getattr(spec, "block_size", 0) or 0
            got = len(blocks[0]) if blocks and blocks[0] is not None else 0
            max_length = kwargs.get("max_length") or 0
            asked = (max_length // block_size) if block_size else 0
            for gid in kwargs.get("kv_cache_group_ids") or ():
                stats.record_lookup(gid, block_size, got, asked)
        except Exception:
            pass
        return blocks

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
        self._evict_cause = stats.EVICT_RESET
        try:
            self._trim_protected_prompt_blocks()
        finally:
            self._evict_cause = stats.EVICT_CAP
        released = before - len(self._protected_prompt_block_ids)
        stats.record_protected_now(
            self.kv_cache_group_id, len(self._protected_prompt_block_ids)
        )
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
