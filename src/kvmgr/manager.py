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

from collections import OrderedDict

from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import (
    MLAAttentionManager,
    SingleTypeKVCacheManager,
)

from . import flags, policy, serve, stats

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
        # Phase 3. PREFIX -> its protected block ids IN PREFIX ORDER, the dict
        # itself in LRU order of last protect (least-recently-touched first).
        # Kept BESIDE patch6's own set and deque rather than replacing them, so
        # `fifo` remains literally patch6's code path and the two cannot drift.
        self._kv_lineages: OrderedDict[object, list[int]] = OrderedDict()
        self._kv_block_lineage: dict[int, object] = {}
        # The FIRST block id ever seen for a lineage, kept even after that
        # block is evicted. Without it a survivor list cannot answer the only
        # question that matters -- "did the HEAD survive" -- because after an
        # eviction ids[0] is just the lowest surviving index, not the head.
        self._kv_lineage_head: dict[object, int] = {}
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
        before = set(self._protected_prompt_block_ids)
        super()._protect_prompt_blocks(blocks)
        after_ids = self._protected_prompt_block_ids
        stats.record_protect(
            self.kv_cache_group_id, max(0, len(after_ids) - len(before)),
            len(after_ids),
        )
        self._kv_track(blocks, newly=after_ids - before)
        stats.record_lineages(
            self.kv_cache_group_id, self.kvmgr_lineage_report()
        )

    def _kv_track(self, blocks, newly: set[int]) -> None:
        """Record this prefix's identity, order, and that it was just used.

        `blocks` is a PREFIX -- MLAAttentionManager.cache_blocks passes
        `req_to_blocks[rid][:num_hit_blocks]` -- so its order IS prefix order
        and `blocks[0]` is the head. That is the whole reason tail-first is
        possible at all: nothing else in the manager knows which end is which.

        A SHARED HEAD IS ATTRIBUTED TO THE FIRST LINEAGE THAT PROTECTED IT, and
        that is accepted rather than fixed. patch6 skips a block that is already
        protected, so a second conversation sharing a prefix adds only its own
        new blocks. The consequence is benign: the shared part is refreshed
        whenever EITHER conversation protects, because the touch is on the
        lineage, so a shared head is harder to evict, which is the right bias.
        """
        if not blocks:
            return
        try:
            key = policy.lineage_key(blocks[0])
            if key is None:
                return
            ids = self._kv_lineages.get(key)
            if ids is None:
                ids = []
                self._kv_lineages[key] = ids
            if key not in self._kv_lineage_head:
                head = getattr(blocks[0], "block_id", None)
                if head is not None:
                    self._kv_lineage_head[key] = head
            for block in blocks:
                bid = getattr(block, "block_id", None)
                if bid is None or bid not in newly:
                    continue
                ids.append(bid)
                self._kv_block_lineage[bid] = key
            if not ids:
                # *** AN EMPTY LINEAGE MUST NOT BE RETAINED. ***
                # _kv_track is called on every protect, including ones where
                # patch6 took nothing new (null, unhashed, or already
                # protected). That inserted a key with an empty list which
                # _kv_forget never reaches -- no block maps to it -- so empty
                # lineages accumulated for the life of the process. Harmless to
                # choose_victim, which skips them, but it grows without bound
                # and it showed up live as rows of
                #   blocks=0 head_intact=False reusable=0
                # in the per-lineage report. Found by reading that report.
                #
                # Dropping the key loses nothing: with no blocks there is
                # nothing to evict and nothing to order. A later protect that
                # DOES take a block re-creates it at the most-recent position,
                # which is the same place move_to_end would have put it.
                self._kv_lineages.pop(key, None)
                self._kv_lineage_head.pop(key, None)
                return
            # Touched: this prefix is the most recently used one now.
            self._kv_lineages.move_to_end(key)
        except Exception:
            # Bookkeeping must never cost a request. Losing a lineage degrades
            # the victim choice toward patch6's; it cannot corrupt the pool.
            logger.exception("kvmgr: lineage tracking failed; victim choice degrades")

    def _kv_forget(self, block_id: int) -> None:
        key = self._kv_block_lineage.pop(block_id, None)
        if key is None:
            return
        ids = self._kv_lineages.get(key)
        if ids is None:
            return
        try:
            ids.remove(block_id)
        except ValueError:
            pass
        if not ids:
            self._kv_lineages.pop(key, None)
            self._kv_lineage_head.pop(key, None)

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

        order = flags.knob("evict_order", quiet=True)

        if order == "fifo":
            released = super()._release_one_protected_prompt_block(block_ids_to_skip)
            rule = stats.VICTIM_FIFO
            if released:
                # Keep our own view consistent with patch6's, which just took a
                # block we are not tracking the choice of.
                for bid in list(self._kv_block_lineage):
                    if bid not in self._protected_prompt_block_ids:
                        self._kv_forget(bid)
        else:
            tail_first = order == "lru_tail_first"
            rule = stats.VICTIM_LRU_TAIL if tail_first else stats.VICTIM_LRU
            released = self._kv_release_lru(block_ids_to_skip, tail_first)

        if released:
            stats.record_evict(
                self.kv_cache_group_id,
                cause=self._evict_cause,
                victim_rule=rule,
                protected_now=len(self._protected_prompt_block_ids),
            )
            if flags.raw("kvmgr_log_level"):
                logger.info(
                    "kvmgr: KV group %d evicted a protected block "
                    "(cause=%s, victim=%s, %d lineage(s) held, "
                    "%d protected = %d tokens remain)",
                    self.kv_cache_group_id, self._evict_cause, rule,
                    len(self._kv_lineages),
                    len(self._protected_prompt_block_ids),
                    len(self._protected_prompt_block_ids) * self.block_size,
                )
        return released

    def _kv_release_lru(self, block_ids_to_skip, tail_first: bool) -> bool:
        """Release one block, chosen by lineage LRU and optionally tail-first.

        The bookkeeping is patch6's, deliberately line for line: remove from
        `_protected_prompt_block_ids`, and free the block ONLY if it still
        carries a reference. patch6's stale entries in `_protected_prompt_block_queue`
        are left alone -- its own release skips an id that is no longer in the
        set, so a stale entry is already a case it handles, and pruning the
        deque here would be a second writer to a structure we do not own.
        """
        chosen = policy.choose_victim(
            self._kv_lineages, self._protected_prompt_block_ids,
            block_ids_to_skip, tail_first,
        )
        if chosen is None:
            # Nothing we track can go. Fall back to patch6 rather than
            # reporting "cannot release" -- a block protected before this
            # manager started tracking lineages is still a legitimate victim,
            # and refusing would hold the cap over its budget forever.
            return super()._release_one_protected_prompt_block(block_ids_to_skip)

        _key, block_id = chosen
        self._kv_forget(block_id)
        self._protected_prompt_block_ids.discard(block_id)
        block = self.block_pool.blocks[block_id]
        if block.ref_cnt > 0:
            self.block_pool.free_blocks([block])
        return True

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

    def kvmgr_lineage_report(self, limit: int = 20) -> list[dict]:
        """Per-prefix survivors, and whether the HEAD is still protected.

        This is the only thing that distinguishes head-first from tail-first
        eviction from OUTSIDE the engine. Counters cannot: both rules evict the
        same NUMBER of blocks. And a re-send measures nothing either, because a
        freed-but-still-cached block keeps serving a cache hit until something
        actually reuses it -- so until real pressure has drained the free queue,
        a head-first eviction and a tail-first one look identical from the API.

        `head_intact` is what the prefix is WORTH: reuse is a contiguous match
        from token 0, so a prefix that lost its head reuses NOTHING however many
        of its blocks are still resident.

        Bounded, and in the LRU order the eviction rule itself walks, so the
        first entry is the next victim.
        """
        out: list[dict] = []
        try:
            protected = self._protected_prompt_block_ids
            for key, ids in list(self._kv_lineages.items())[:limit]:
                head = self._kv_lineage_head.get(key)
                survivors = [b for b in ids if b in protected]
                out.append({
                    "blocks": len(survivors),
                    "tokens": len(survivors) * self.block_size,
                    "head_intact": head is not None and head in protected,
                    "reusable_blocks": self._kv_reusable(ids, protected, head),
                })
        except Exception:
            logger.exception("kvmgr: lineage report failed")
        return out

    def _kv_reusable(self, ids, protected, head) -> int:
        """Blocks reusable from the head: the contiguous run from the head
        onward. ZERO if the head itself is gone, however much else survived --
        that is the whole asymmetry tail-first eviction exists to exploit."""
        if head is None or head not in protected:
            return 0
        n = 0
        for b in ids:
            if b not in protected:
                break
            n += 1
        return n

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
        stats.record_lineages(
            self.kv_cache_group_id, self.kvmgr_lineage_report()
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
        if flags.changed("gpu_cache_epoch"):
            self.kvmgr_drop_gpu_cache()

    def kvmgr_drop_gpu_cache(self) -> bool:
        """Forget every cached prefix in the GPU pool, KEEPING the disk tier.

        *** THIS IS THE LEVER THAT REMOVES THE REBOOT FROM A RESTORE TEST. ***
        Proving the disk read path works needs a GPU cache that has forgotten a
        prefix the disk still holds, and until now the only way to get there was
        a full restart: ~4 minutes of boot plus a cold prefill, twice per
        experiment. This build exposes no reset at all -- POST
        /reset_prefix_cache is 404 and no such route is registered -- so there
        was no other way in.

        What makes it work is an asymmetry worth stating plainly:
        `block_pool.reset_prefix_cache()` is the POOL's method, while the thing
        that would throw away our disk index is the CONNECTOR's `reset_cache()`.
        Going at the pool directly never touches the index, so the GPU forgets
        and the disk remembers -- which is exactly the state a restore test
        needs and is otherwise only reachable by rebooting.

        *** AND IT HAS TO RELEASE OUR OWN PROTECTED BLOCKS FIRST. *** patch7
        refuses the reset unless every block is free (`num_used_blocks != 1`
        counting the null block), and MNT-6a's whole job is to hold references
        so prompt blocks survive their request. So with protection in force the
        reset could only ever return False, and the useful operation is the pair:
        release, then reset.

        Returns True only if the pool really did reset, and says so either way --
        a drop that silently did nothing would send somebody hunting for a bug in
        the read path that was never exercised.
        """
        pool = getattr(self, "block_pool", None)
        if pool is None or not hasattr(pool, "reset_prefix_cache"):
            logger.error("kvmgr: no block pool to drop (nothing was reset)")
            return False
        # Report FREE BLOCKS rather than a release count: our override returns
        # None, and free blocks are the quantity patch7 actually gates on
        # (`num_used_blocks != 1`), so this says whether the reset COULD work.
        def _free():
            try:
                return pool.get_num_free_blocks()
            except Exception:
                return None

        before = _free()

        # *** RELEASE EVERY GROUP, NOT JUST OURS, AND THAT WAS A REAL REFUSAL.
        # *** Measured 2026-10-08: the drop fired, released group 0 and was
        # REFUSED -- and the log named the cause in one number, "pool free
        # blocks 22536 -> 22581". +45 is exactly group 0's share of an
        # 11,764-token prompt (11764/256 = 46), so the other ~1,700 blocks
        # belonged to the four windowed groups, which run upstream's
        # SlidingWindowMLAManager because we register for MLAAttentionSpec
        # only. patch7 gates on `num_used_blocks != 1` across the WHOLE pool,
        # so releasing one group of five could never be enough.
        #
        # `release_protected_prompt_blocks` is on SingleTypeKVCacheManager, so
        # it exists on the managers we did not replace. The tuple comes from
        # the coordinator wrap, which is the only place that holds it.
        # `self` is in that tuple, so it is not released separately -- and the
        # fallback covers the case where the stash failed, because releasing
        # one group is still better than releasing none.
        targets = tuple(getattr(type(self), "all_managers", ()) or ())
        if not targets:
            logger.warning(
                "kvmgr: no manager list stashed, so only THIS group can be "
                "released -- patch7 gates on the whole pool, so expect the "
                "reset to be refused")
            targets = (self,)
        released = 0
        for m in targets:
            try:
                m.release_protected_prompt_blocks()
                released += 1
            except Exception:
                logger.exception(
                    "kvmgr: releasing protected blocks failed for %s",
                    type(m).__name__)
        after = _free()
        try:
            ok = bool(pool.reset_prefix_cache())
        except Exception:
            logger.exception("kvmgr: block_pool.reset_prefix_cache() raised")
            ok = False
        logger.info(
            "kvmgr: GPU prefix cache drop -- released %d/%d manager(s), "
            "pool free blocks %s -> %s, "
            "reset %s. The DISK index is untouched by design (this is the "
            "pool's reset, not the connector's), so the next identical prompt "
            "must come back off disk or not at all.%s",
            released, len(targets), before, after, "OK" if ok else "REFUSED",
            "" if ok else " Refused means blocks are still referenced -- let the"
                          " engine go idle and bump gpu_cache_epoch again.",
        )
        return ok


_AUDITED = False


def _audit_once() -> None:
    global _AUDITED
    _AUDITED = True
    flags.log_env_audit()
