"""kvmgr.flags -- hot configuration, and the rule that every knob goes through it.

WHY NOT A CONSTANT, EVER.  A restart of this engine costs a 672-795s cold
prefill (measured 2026-10-06: 725K tokens in 795s, 767K in 672s).  Re-tuning a
policy knob must never cost that, so a knob captured at import, in a
constructor, or in a cached attribute is a DEFECT here rather than a detail.

WE DO NOT INVENT A MECHANISM.  patch6 already has one, proven and in use for
MNT-6b/8/9: a small JSON file re-stat'ed at most once a second, with
``VLLM_<NAME>`` winning over the file so a boot can still pin an arm.  This
module IMPORTS that ``_flag`` rather than copying it.  Copying would mean two
caches and two TTLs over one file, and the drift would be invisible -- the same
"two tables describing one wire" failure this stack keeps paying for.

WHAT THIS ADDS over bare ``_flag()``:

  * ``knob()`` -- logs ON CHANGE.  patch6 logs its cap ONCE
    (``_protected_cap_logged``), which is right for a boot-time constant and
    wrong for something an operator flips: flip it and you get no confirmation
    it took.  A silent config change is the failure this repo keeps paying for.
  * ``changed()`` -- the edge, so a caller can RECONCILE rather than wait for
    the next natural call.  A lowered cap otherwise bites only on the next
    protect for that group, so a quiet group holds stale protection
    indefinitely.
  * ``protected_fraction()`` -- the one knob with a legacy env name to honour.

THE TRAP THIS FILE CANNOT FIX, only document: ``_flag()`` checks
``VLLM_<NAME>`` FIRST.  So any knob also present in ``.env.dspark`` cannot be
moved from the file -- you would edit it, see nothing change, and get no
indication why.  ``VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION=0.75`` is set in
``.env.dspark`` on both nodes TODAY and must be removed from it when this
lands, or ``protected_fraction`` is hot in name only.  ``audit_env()`` reports
exactly that, and the manager calls it once at startup so it is in the log
rather than in somebody's memory.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

# The single source of truth for flag reading.  Importing it binds us to
# patch6's module-level cache, which is the point.
from vllm.v1.core.single_type_kv_cache_manager import _flag as _patch6_flag

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

# ---------------------------------------------------------------------------
# The knob table.  Defaults are chosen so that an EMPTY FLAGS FILE BEHAVES
# EXACTLY AS TODAY: every new mechanism off, every existing one at its current
# value.  A deploy is therefore a no-op until somebody writes a flag.
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    # ours
    "kvmgr_enabled": False,       # our policy vs upstream's, per call (see manager)
    "protected_fraction": 0.30,   # patch6's own default; .env.dspark pins 0.75 today
    "protect_alloc": "equal",     # equal (today's /num_managers) | by_need
    "evict_order": "fifo",        # fifo (today) | lru | lru_tail_first
    "disk_enabled": False,
    "disk_budget_gb": 0.0,
    "disk_restore": False,
    # *** THE OFF SWITCH, AND ITS DEFAULT IS THE POINT. *** False means the
    # store path RUNS, because that is what every other knob's default means
    # here: the default is today's behaviour, never a safety position. A knob
    # that silently disabled a working feature on an empty flags file would be
    # the "a deploy with an empty file behaves exactly as today" rule broken.
    #
    # It exists because a mechanism must get a hot off switch before it gets a
    # feature, which cost a user-visible outage on 2026-10-08: the store path
    # could only be disabled by a SIX-MINUTE restart, so the mitigation was
    # unaffordable and a suspect path stayed on a live engine. Gated in
    # `prepare_store`, never in the handler -- see the note there for why the
    # obvious placement poisons the index.
    "disk_store_paused": False,
    # Skip a store key whose FILE ALREADY EXISTS before the GPU->staging copy,
    # not after. `_write_batch` has always skipped an existing path, but it runs
    # once the copy has completed -- so an already-stored block cost a full
    # 63-tensor GPU->pinned copy that was then discarded ("stored: 92" against
    # zero bytes written, measured). Those copies are also the concurrent GPU
    # write that breaks CU_MEMCPY_SRC_ACCESS_ORDER_ANY on the shared staging
    # pool, which is the Xid 13 root cause -- and they are commonest during a
    # restore, precisely when the load is reading that pool.
    #
    # Default ON, and that does NOT violate "an empty flags file behaves as
    # today": the skipped copies were discarded anyway, so the bytes that reach
    # disk are identical. Only the wasted copy disappears.
    "disk_skip_present_before_copy": True,
    # ----------------------------------------------------------------
    # CROSS-RANK PRESENCE VERIFICATION
    # ----------------------------------------------------------------
    # *** THE INDEX IS A CLAIM ABOUT TWO FILESYSTEMS AND THE BOOT WALK CAN
    # *** ONLY CHECK ONE. *** `_load_manifest`'s prune is
    # `os.path.exists(self._mapper.get_file_name(k))` on the ENGINE CORE's own
    # node, and `get_file_name` is rank-dependent. Its own docstring admits the
    # hole: "what that CANNOT catch is rank 1's files being deleted while the
    # head's survive". And the consequence is not a miss -- upstream's
    # `offloading/worker.py:273` is a bare `assert transfer_result.success`, so
    # a load whose file is gone on ONE node kills the engine core. The 371s we
    # pay at every boot protects against exactly half of what it is for.
    #
    # *** IT SHIPPED DEFAULTING ON AND THE BOX SAID NO. DEFAULT OFF UNTIL THE
    # *** TRIGGER MOVES TO BOOT. *** Measured 2026-10-12 on the live engine:
    # with this on, the 700K restore that costs 11s warm returned
    # **0 tokens off disk** and took a full cold prefill. The mechanism is
    # right and the TRIGGER is wrong, and the counters say exactly why --
    # `lookup 500,720` against the 2,758 a normal run costs, 79,328 deferrals,
    # 5,470 of them EXPIRING, and 2,300 RPCs.
    #
    # The cause is upstream's scan shape, not our check:
    # `_sliding_window_lookup` treats a DEFERRED key as a non-hit and keeps
    # scanning ("continue lookup ... until a hit is detected"), so ONE
    # deferring pass walks the WHOLE sliding-window key list. A 700K prompt has
    # ~2,736 keys in group 0 and ~175,000 in a 4-token SWA group, so deferring
    # turns a 2,758-lookup request into a ~287,000-key verification job -- past
    # any sane deferral budget, so keys expire to a miss, so the prefix answer
    # collapses and the restore never fires.
    #
    # *** THE FIX IS A DIFFERENT TRIGGER, NOT A BIGGER BUDGET: verify at BOOT,
    # *** ALL RANKS, IN PARALLEL. *** The head already stats all 922,992 keys in
    # 365s; having each rank stat its OWN files at the same time costs the same
    # WALL TIME and checks both nodes AND the sizes. Then no lookup ever
    # defers, every key is vouched for before the API opens, and the prize is
    # intact. The blocker is that `DiskPresenceManager.__init__` holds no
    # executor reference -- the same problem `capture_offload_sched_from_caller`
    # already solves off the call stack.
    #
    # *** AND THEN THE TRIGGER MOVED TO BOOT, SO IT IS ON AGAIN. ***
    # `_boot_verify` asks EVERY rank before the API opens, in place of the
    # rank-local walk. Verified live 2026-10-12: 922,977 keys, 231 RPCs,
    # **0 dropped**, `unverified 0`, and the 700K restore back at **13s /
    # 700,416 tokens** with **not one deferral** -- against `restored=0` on the
    # lazy version.
    #
    # ~~that is cheaper than the walk it replaces: ~200s against 365s~~
    # *** THAT CLAIM WAS MINE AND THE BOOT IT PREDICTED REFUTED IT. *** The
    # all-ranks pass took **467.1s** against the rank-local walk's **365s** --
    # **+102s, 28% more boot**, not a saving. The ~4,600 keys/s I generalised
    # from was measured by the background sweep **while serving, on a warm
    # dentry cache**; a boot's cache is cold *and* the start script drops it
    # every 2 seconds. And the parallelism does not compose the way I assumed:
    # 231 SYNCHRONOUS chunks each pay `max(rank0, rank1)`, so the cost is
    # `231 x max(...)` rather than `max(sum, sum)`.
    #
    # So the honest trade, stated as one: **+28% boot time** buys coverage of
    # BOTH nodes, a SIZE check that `os.path.exists` cannot make, and the
    # removal of every runtime deferral. That is worth it -- a 467s boot
    # against a dead engine core -- but it is a trade and not a free win, and
    # the lever left for boot time is `DROP_CACHES_ON_START`, which is now
    # worth much more than it was.
    #
    # The deferral that broke the restore is then **structurally absent**:
    # every key in the index is vouched for before the first request, so
    # `lookup` never returns None. With `disk_verify_defer_s` at 0 as well,
    # even the leftover cases (a boot that fell back, a deliberate re-queue)
    # answer MISS rather than deferring, so the 500,720-lookup storm cannot
    # recur.
    #
    # This still breaks the table's "an empty flags file behaves as today"
    # rule, knowingly, on the load_inflight_max reasoning: the state in which
    # it changes anything is one where today's behaviour is a DEAD ENGINE. And
    # the degradation is safe in both directions -- it falls back to the old
    # walk if the engine core cannot be reached, and a fault refuses presence,
    # which costs a cold cache and never correctness.
    "disk_verify_enabled": True,
    # Keys per `collective_rpc`. Bounds the MESSAGE, and must stay ~100x
    # smaller than the RPC timeout's worth of work so the bound below is never
    # approached in healthy operation. 4,000 stats is tens of milliseconds on a
    # warm dentry cache and well under a second on a cold one.
    #
    # It is a bound and not a batch target: a long request's whole candidate
    # set (measured: 2,758 keys for a 700K prefix) fits in one chunk.
    #
    # *** IT MUST STAY SMALL *WHILE SERVING*, and the measurement below is why.
    # *** A chunk is a blocking RPC on the scheduler's thread, so its duration
    # is a step stall. At 4,000 keys that is ~1,054ms warm -- already past
    # test-meter section 1's 1.0s gate if a batch ever filled. It does not
    # today only because the batch is `min(credit, cap)` and the background
    # rate keeps credit tiny (~40 keys), so the shipped pair is safe. Raise one
    # without the other at your peril.
    "disk_verify_chunk": 4000,
    # *** THE BOOT GETS ITS OWN, TEN TIMES LARGER, BECAUSE NOTHING IS BEING
    # *** SERVED YET. *** Measured on the live engine 2026-10-12, rate pinned
    # so the chunk was the only limiter, A/B/A so cache warming could not
    # flatter an arm:
    #
    #     cap  4,000 -> 3,794 keys/s   1,054 ms/RPC
    #     cap 20,000 -> 5,332 keys/s   3,751 ms/RPC
    #     cap  4,000 -> 3,799 keys/s   1,053 ms/RPC   <- the returning arm
    #     cap 40,000 -> 7,998 keys/s   5,001 ms/RPC
    #
    # Throughput DOUBLES, because a large part of each round trip is fixed
    # cost rather than per-key work -- which is the same fact as "231
    # synchronous chunks each pay max(rank0, rank1)", read from the other side.
    # A 5s stall per RPC is free at boot and intolerable while serving, which
    # is the whole reason this is a separate knob and not a bigger default for
    # the one above.
    #
    # *** WHAT THIS DOES *NOT* SAY. *** The boot's 467.1s was paid COLD, with
    # the start script dropping the dentry cache every 2s, and at 4,000 it
    # managed 1,976 keys/s against this 3,794 warm -- so cold costs ~1.9x and
    # chunk size ~2x, and they are different levers. Extrapolating this warm
    # measurement onto the boot is exactly the mistake that produced the
    # retracted "cheaper than the walk" claim, so the expectation here is
    # "materially less than 467s", not a number.
    "disk_boot_verify_chunk": 40000,
    # How long a lookup may DEFER one key before answering False.
    #
    # *** DEFERRING IS UPSTREAM'S OWN MECHANISM, not a trick. *** `lookup`
    # returns `bool | None` and None means defer:
    # `_maximal_prefix_lookup` keeps scanning (so one pass yields the whole
    # candidate set), `get_num_new_matched_tokens` returns (None, False), and
    # the scheduler re-queues the request next step with NOTHING allocated and
    # no worker involved -- tiering's own docstring calls it "transparent
    # retry". Expiry answers False, which is always safe: a miss costs
    # recompute, and recompute is what we are comparing against.
    #
    # *** AND THE DEFAULT IS NOW 0, WHICH MEANS NEVER DEFER. *** Measured on
    # the box (see disk_verify_enabled): deferring is cheap for a request whose
    # candidate set is small and catastrophic for one whose sliding-window
    # groups hold ~175,000 keys, because upstream keeps scanning past a
    # deferred key. At 0 an unverified key answers MISS immediately -- SAFE by
    # construction, since a miss costs recompute -- and costs nothing on the
    # scheduler's thread. The verified portion of the index still restores
    # normally, so with the background sweep running the cache CONVERGES
    # rather than stalling.
    #
    # So there are two sane settings: 0 (safe, converging, what ships) and a
    # value large enough for a request to wait out its own verification, which
    # is only reasonable once a boot-time all-ranks pass makes that wait empty.
    "disk_verify_defer_s": 0.0,
    # *** GENEROUS AND FATAL. NEVER ZERO, WHICH WOULD MEAN FOREVER. ***
    # Read off the image (patches/pristine/vllm/v1/executor/
    # multiproc_executor.py): a TimeoutError does NOT set the executor's
    # `is_failed`, and the worker's eventual response is left UNCLAIMED on an
    # ordered response MessageQueue -- so every later dequeue is off by one and
    # the engine receives a verification dict where it expects a
    # ModelRunnerOutput. Swallowing that is silent corruption. So a timeout is
    # not caught-and-continued anywhere; it is reported with our own line
    # naming the cause, because the alternative is the same death a moment
    # later with an unattributable traceback.
    #
    # A 0 or negative value is coerced UP to this default rather than becoming
    # `timeout=None`, which `deadline = None if timeout is None` turns into
    # `mq.dequeue(timeout=None)`: a rank whose disk is in D-state -- exactly
    # what this mechanism defends against -- would then block the engine core
    # for ever with NO collective pending, so no NCCL watchdog can end it. A
    # hung engine with no traceback is worse than the crash it replaces.
    "disk_verify_rpc_timeout_s": 60.0,
    # SLOW BACKGROUND RE-VERIFICATION, in keys per second.
    #
    # Demand-driven verification only ever covers what a request asks for, and
    # divergence can appear MID-BOOT: an operator `rm`, a partial kvsweep.sh, a
    # filesystem error. This re-checks already-verified keys in the background
    # so that is caught inside a boot rather than at the next restart. It never
    # defers a lookup -- a key stays claimable while it is re-checked, because
    # the only thing a re-check can do is remove a key that was already going
    # to kill the engine.
    #
    # 100/s walks a 919,669-key index in about 2.5 hours OF BUSY ENGINE, and
    # that qualifier is load-bearing: the pump rides the scheduler step, and
    # `EngineCoreProc._process_engine_step`'s own docstring is "Called only
    # when there are unfinished local requests". *** SO THE BACKGROUND SWEEP
    # MAKES NO PROGRESS ON A FULLY IDLE ENGINE. *** That is acceptable rather
    # than merely tolerated -- an idle engine is issuing no loads, so there is
    # nothing to be wrong about, and it resumes the moment traffic arrives.
    # The DEMAND path is unaffected by this, because a deferring lookup implies
    # a waiting request, which implies steps. 0 disables it.
    #
    # *** AND THE DEFAULT IS 0 -- OFF -- AFTER AN NCCL KILL I CANNOT RULE THIS
    # *** OUT OF. *** 2026-10-12 17:37:35, on the first boot with runtime
    # verification RPCs enabled through a 700K restore: `Xid 32` twice on the
    # head, `torch.AcceleratorError: CUDA error: unspecified launch failure` at
    # upstream's `async_copy_ready_event.synchronize()`, then
    # `ProcessGroupNCCL` terminated rank 0 and Docker restarted the head alone.
    # Our ledger dumped `1 transfers, block ids [7, 39] of 22756, slots [0, 32]
    # of 256` -- every index in range, so the bounds guards correctly refused
    # nothing and the memory simply stopped being what a valid index pointed
    # at, which is this project's recorded signature for this fault.
    #
    # *** NOT PROVEN, AND THE CONTRARY EVIDENCE IS STATED: *** the PREVIOUS
    # boot ran the same sweep at the same rate through the same 700K restore
    # with Xid unchanged at 30/22. One event is not a cause.
    #
    # What makes OFF the right default anyway is that this repo's own comment
    # in `_enforce_budget` says so, and I overrode it: *"a collective_rpc from
    # inside the scheduler step is precisely the blocking-the-critical-path
    # mistake that has already cost this project an NCCL kill."* A blocking
    # RPC forces an outstanding `execute_model` future to be awaited INLINE,
    # which moves a synchronisation point relative to the offload handler's
    # async copies -- and those copies are driven from `get_finished()` inside
    # the very step being blocked. The load path here has been exquisitely
    # concurrency-sensitive before (inflight 4 -> Xid 32; a shared staging
    # buffer -> Xid 13 five times).
    #
    # The cost of OFF is nil, which is what settles it: the BOOT check does all
    # the safety work, before any transfer exists, and it leaves the index
    # fully verified with new keys born verified. The sweep only ever covered
    # divergence appearing MID-boot -- a real but secondary case, and one
    # `POST /kv/knob` away for anyone who wants it on a quiet engine.
    "disk_verify_bg_per_s": 0.0,
    # ----------------------------------------------------------------
    # THE FREE-SPACE FLOOR
    # ----------------------------------------------------------------
    # *** A FULL DISK IS A DEAD ENGINE, AND THERE WAS NO PREFLIGHT ANYWHERE.
    # *** No statvfs existed in the plugin. ENOSPC inside `write_chunks`
    # becomes `st["failed"]`, becomes `TransferResult(success=False)`, becomes
    # upstream's bare assert. So the floor's job is to make that
    # UNREACHABLE BY CONSTRUCTION, and it does it by EVICTING rather than by
    # refusing -- a refusal destroys the only delete channel there is, since
    # deletes ride the store spec and `scheduler.py:745` drops that spec when
    # `keys_to_store` is empty.
    #
    # *** NON-ZERO DEFAULTS, AS THE DOCUMENTED SAFETY-BOUND EXCEPTION. ***
    # Same reasoning as load_inflight_max=1 again: the only state in which a
    # non-zero floor changes behaviour is one where today's behaviour is a dead
    # engine. Sized against measurement, not taste -- both nodes hold 3,755 GiB
    # total with ~2.26-2.32 TiB free and 620 G of tier, so the tier's remaining
    # growth to its 1000 GB cap CANNOT reach any sane floor by itself. The floor
    # fires only on a foreign filler: weights, logs, captures. 64 GB is ~100x
    # the largest single prepare_store a 1M-token prompt can offer.
    "disk_free_floor_gb": 64.0,
    # Below this, the tier FREEZES -- it admits exactly one key per round so
    # the store job still exists to carry the unlinks out, and stops when even
    # that is too much (a fixed code-level minimum, not a knob).
    "disk_free_hard_gb": 16.0,
    # Leaving DEGRADED needs more than entering it, or a tier sitting exactly
    # at the floor flaps between bands on every store round.
    "disk_free_hysteresis_gb": 8.0,
    # How much the effective budget is cut below what is held, per store round,
    # while DEGRADED. Every round is then net-negative by one step, so the
    # drain is paid by the traffic that caused it rather than in one burst.
    "disk_free_evict_step_gb": 2.0,
    # statvfs is cheap and this is the scheduler's thread, so it is throttled
    # anyway: a free-space reading older than this is re-taken, never more
    # often.
    "disk_free_check_s": 5.0,
    # How often the step pump refreshes the WORKERS' free space over the same
    # RPC the verification uses. *** THE ACTUATOR SEES ONLY THE HEAD'S
    # FILESYSTEM, and the worker is currently the fuller of the two. *** One
    # RPC a minute closes that, and doubles as a keep-alive proving the channel
    # works before a sweep needs it.
    #
    # *** ALSO 0 -- OFF -- FOR THE SAME REASON AS disk_verify_bg_per_s. *** It
    # is the only other thing that issues a blocking `collective_rpc` from the
    # scheduler step while requests are being served, and with the sweep off it
    # would be the sole remaining one. Together they make the runtime RPC rate
    # exactly ZERO by default, which is the conservative position after the
    # 17:37:35 NCCL kill.
    #
    # What is lost is small and already stated as a limit elsewhere: the floor's
    # actuator sees only the HEAD's filesystem. The worker's free space is still
    # visible to a human on `/kv/workers`, because the store handler computes it
    # on its OWN thread in `snapshot()` and needs no step RPC to do it.
    "disk_free_rpc_s": 0.0,
    # The LOAD pool's live size, mirroring `staging_slots` for the store. The
    # load has its OWN pinned allocation now (sharing one buffer is what
    # faulted the worker with Xid 13 five times), so it needs its own live
    # limit or the knob surface would govern only half the staging.
    # 0 = the whole load allocation.
    "staging_load_slots": 0,
    # Max blocks in ONE staging->GPU transfer_async. ~256 FAULTS the worker with
    # Xid 13 (measured: 96/128/160/192/224 clean at 1-2s, 256 dead on the first
    # attempt), so this is a CORRECTNESS limit, not a throughput tunable. It is
    # separate from the pool size on purpose: the pool used to imply it by
    # accident at 64, and a resize would silently have restored the fault.
    "load_batch_max": 192,
    # Max load ROUNDS whose staging->GPU copy may be outstanding at once.
    # The per-round cap above was NOT enough: the 700K scale test respected it
    # throughout and still faulted at 861 keys, because a chunked prefill
    # issues many prepare_load calls and nothing bounded how many rounds were
    # in flight together. 0 = unbounded, which is the behaviour that faulted.
    # 1 makes the path fully serial, which is how the hypothesis gets tested.
    #
    # *** THE HYPOTHESIS IS TESTED AND THE DEFAULT IS NOW 1. *** Measured
    # 2026-10-10 with load_ops_max PINNED at 2048 (~33 blocks, a SEVENTH of the
    # smallest per-call value ever seen to fault): inflight 1 and 2 restore a
    # 700K prefix in 12s each, and inflight 4 KILLED THE ENGINE -- Xid 32 twice
    # on the head, `CUDA error: unspecified launch failure`, EngineDeadError.
    # So concurrency is load-bearing INDEPENDENTLY of the per-call bound, and 4
    # is not a conservative default, it is the lethal one.
    #
    # AND IT COSTS NOTHING TO BE SAFE HERE: 1 and 2 are indistinguishable,
    # because the restore is bounded by device and link bandwidth rather than by
    # outstanding submissions. There is no throughput being traded away.
    #
    # The reason this matters as a DEFAULT and not just a knob: the safe value
    # was carried as a runtime override through every 700K measurement, and a
    # restart discards overrides. The engine came back up reading
    # `load_inflight_max 4 (default)` from /kv -- the value measured to be
    # fatal -- while the file still had disk_restore=False.
    #
    # SO THE HAZARD WAS LATENT, NOT LIVE, and the distinction is the point: with
    # restore off no load can fire, so nothing was in danger that minute. What
    # was true is that enabling disk_restore -- one hot knob, the obvious next
    # step, and the whole reason this tier exists -- would have armed the fatal
    # pair without anyone touching concurrency at all. A safe value that exists
    # only as an override is not a safe default; it is a safe value you lose on
    # the next boot, silently, because a default is not a change.
    "load_inflight_max": 1,
    # Max COPY-OP DESCRIPTORS in one staging->GPU submit. This is the quantity
    # upstream's single `cuMemcpyBatchAsync` actually contains --
    # `num_copy_ops += group_size * len(group_data_refs)` -- so a cap in BLOCKS
    # means a different descriptor count per group, and group 0 (most layers,
    # largest block size) reaches any limit first. 0 = unbounded.
    # The DIMENSION is read from upstream; the threshold is still empirical.
    "load_ops_max": 2048,
    # --- the STORE's per-round budget. ---------------------------------
    # *** THE STORE PATH HAD NO CAP AND IT STALLED THE STREAM BY ~1.1s. ***
    # Measured 2026-10-11 by scripts/test-meter.sh section 1 (vllm-proxy):
    # max inter-event gap 1.137s against a 0.102s baseline, A/B/A on
    # disk_store_paused (1.109 live / 0.111 paused / 1.067 live), and located as
    # ONE 1.052s stall 72% of the way through a 103-event stream. The disk
    # writes are not the cause -- they are already on a thread pool and
    # _collect only polls them. What lands on the engine's step is the submit.
    #
    # Three knobs because the load needed three, for reasons that apply
    # unchanged here: blocks are the granularity, DESCRIPTORS are what one
    # driver call contains (blocks x refs-for-the-group, so group 0 reaches any
    # limit first), and the INFLIGHT count is the only one that bounds a step --
    # a per-round cap alone just turns one big round into eight small ones.
    "store_batch_max": 32,
    "store_ops_max": 2048,
    "store_inflight_max": 1,
    # -- the presence journal ------------------------------------------
    # OFF BY DEFAULT, and that is this module's own invariant rather than
    # timidity: an empty flags file must behave exactly as today, so a new
    # mechanism arrives inert and `local-ops/flags.reference.json` turns it on.
    #
    # WHAT IT FIXES. The manifest is O(index) to persist O(changes) -- 772,561
    # keys rewritten for the ~71 a response adds -- and under a real store load
    # that saturates: manifest_writes 109 against manifest_skipped_busy 462, so
    # 81% of flush attempts were refused as already busy. A refused flush is a
    # window in which a key is on disk with nothing able to NAME it, which is
    # how 155 GiB of unreachable orphans accumulated. With the journal on, the
    # snapshot fires only on compaction and every presence edge is one append.
    #
    # IT DOES NOT SHRINK THE BOOT WALK. Every key's file is still stat'ed
    # because offloading/worker.py:273 asserts on a failed load, and that is
    # independent of the index's file format.
    "journal_enabled": False,
    # Compaction bounds. Records bound the REPLAY cost at boot; bytes bound the
    # disk cost of a pathological run of tiny records. Either trips it, and a
    # ZERO OR NEGATIVE value disables that half rather than meaning "always" --
    # a knob somebody zeroed must not turn compaction into a whole-index
    # rewrite per record, which is the exact cost this mechanism removes. That
    # fail-safe direction is the load_inflight_max lesson applied to a bound.
    #
    # 20,000 records is ~70 responses' worth of adds at the measured ~71 keys
    # each, so a snapshot every few minutes under sustained store rather than
    # the 109-plus-462-refused seen without it.
    "journal_max_records": 20000,
    "journal_max_bytes": 33554432,
    # fsync EVERY RECORD. Off, deliberately, and the threat model is the whole
    # argument: the failure this exists to prevent is the ENGINE dying with
    # stored keys unnamed, and a plain write plus flush() already survives
    # process death because the bytes are in the OS page cache. fsync only adds
    # cover for machine power loss, at a sync per store round on the engine
    # core's own thread. Turn it on to trade that latency for that cover.
    "journal_fsync": False,
    # Seconds of ZERO progress after which a stalled transfer job is FAILED.
    # 0 = hang for ever, which is what this path did until 2026-10-10. Default
    # 240, comfortably inside NCCL's 600s watchdog, because the entire point is
    # to die before ProcessGroupNCCL::checkTimeout does -- that abort is C++,
    # so it leaves no Python traceback and `grep -i traceback` cannot find it.
    # This does NOT make a dead GPU survivable; it makes the death legible.
    "stall_fail_after_s": 240.0,
    # -- the staging pool, hot ------------------------------------------------
    # *** THE ALLOCATION IS A BOOT CEILING; THESE MOVE WITHIN IT. *** The
    # pinned staging is 63 tensors shared by both handlers and two
    # SingleDirectionOffloadingHandler instances, so reallocating it live is a
    # corruption risk rather than a tuning knob -- on a box with ~3GB available
    # and no OOM net. `staging_blocks` in KV_TRANSFER_CONFIG sizes the
    # allocation once; these decide how much of it is used, instantly.
    #
    # 0 means "the whole allocation", so an unset knob is not a silent 32.
    "staging_slots": 0,
    # Slots each direction must leave free for the other. The store and load
    # handlers SHARE one pool, and that sharing made the same restore take
    # 574.9s with stores live against 3.3s with them paused -- a 174x spread,
    # because the pump advances once per scheduler step and fewer slots per
    # round means fewer blocks per step (177 MB/s cold, 17.7 MB/s shared).
    #
    # A floor, not a split: a split wastes half the pool whenever one direction
    # is idle. 0 means "a quarter of the live size", which scales when the size
    # is raised. A starved WRITE is not just slow -- its blocks stay in
    # jobs_to_flush fencing GPU blocks, which is the path that deferred a
    # request until NCCL's watchdog killed a worker.
    # SUPERSEDED by the staging split: the floors existed to stop one
    # direction draining a SHARED pool and starving the other. The load
    # now has its own allocation, so each pool has exactly one user and a
    # reserve inside it would shrink that user for nothing. Kept so an
    # existing flags file still parses; read by nothing on the take path.
    "staging_write_floor": 0,
    # SUPERSEDED by the staging split: the floors existed to stop one
    # direction draining a SHARED pool and starving the other. The load
    # now has its own allocation, so each pool has exactly one user and a
    # reserve inside it would shrink that user for nothing. Kept so an
    # existing flags file still parses; read by nothing on the take path.
    "staging_read_floor": 0,
    # *** THE ONE KNOB THAT GENUINELY CANNOT BE HOT, and it says so. *** A
    # route either exists in the FastAPI app or it does not, and the app is
    # built once at boot -- so this is read at ATTACH time and enabling it
    # needs a restart. Off by default: the plane shares the engine's port,
    # which Caddy's catch-all proxies, so an unattended engine must not have
    # it open. Mirrors VLLM_SERVER_DEV_MODE's own gate-at-attach shape.
    "kvmgr_control_plane": False,
    "kvmgr_log_level": 0,
    # 0 means BIND NOTHING, deliberately unlike a route that answers 404
    # because a boolean said so. Same convention as METER_TAP_PORT.
    "kvmgr_stats_port": 0,
    # Relax TieringOffloadingSpec's UNCONDITIONAL single-block-size assert so a
    # hybrid KV layout can reach the disk tier. See kvmgr/offload.py for why
    # that is defensible and -- more importantly -- what it does NOT prove.
    "offload_allow_hybrid": False,
    # The fs tier opens every block with O_DIRECT, which needs 512-aligned
    # lengths; our 2,103,552-byte blocks are not (mod 512 = 256), so every
    # write failed with EINVAL. False drops the flag. See kvmgr/offload.py.
    "offload_odirect": False,
    # Round-trip every disk write and byte-compare it. A TEST MODE: it doubles
    # store I/O. Exists because this engine's OUTPUT is not deterministic, so an
    # output-equivalence gate proves nothing -- see kvmgr/offload.install_verify.
    "offload_verify": False,
    # *** DROP THE GPU PREFIX CACHE WITHOUT A REBOOT, by bumping this integer.
    # *** The point is the DISK TIER SURVIVES: block_pool.reset_prefix_cache()
    # is the POOL's method, not the connector's, so it never reaches our
    # presence index. That asymmetry is what turns "test the restore path" from
    # a two-boot, twelve-minute ritual into an edit of this file -- and this
    # build exposes no reset endpoint at all (POST /reset_prefix_cache is 404
    # and no such route is registered), so there was no other lever.
    #
    # An INTEGER rather than a boolean because the useful operation is "do it
    # again", and a bool can only be flipped once. The manager acts on the EDGE.
    "gpu_cache_epoch": 0,
    # pre-existing, preserved unchanged -- listed so this table is the whole
    # surface and nothing is hot-by-accident or cold-by-omission.
    "swa_recycle": True,
    "swa_recycle_cached": False,
    "free_queue_prepend_uncached": False,
    "pin_static_head": False,
    "head_pin_fraction": 0.05,
    # *** READ BY patch6 AND DECLARED NOWHERE until 2026-10-09. ***
    # `_maybe_pin_static_head` reads it with an inline default, so it worked --
    # and it was invisible: absent from this table, so absent from the knob
    # listing, from the published policy and from anything auditing coverage.
    # An undeclared knob is one nobody can discover, which is the quiet half of
    # "a flag that governs nothing": here the flag governs something and no
    # reader can tell it exists. Found by auditing reads against declarations.
    "head_pin_partial": False,
    # *** THE GPU CACHE DROP HAS KILLED THE ENGINE TWICE, SO IT IS OPT-IN. ***
    # Xid 32 on 2026-10-09 at 11:50 (worker node) and again at 17:07 (head),
    # both within ~4 seconds of a `gpu_cache_epoch` bump, on different ranks.
    # The second had ZERO eviction pressure (budget 1000 GB, 700 GB headroom)
    # and the pending-blocks fix live in all six processes, so neither churn nor
    # that fix explains it and the MECHANISM IS UNESTABLISHED.
    #
    # Off by default because of that, not in spite of it: a lever whose failure
    # mode is a dead engine and whose cause is unknown should cost somebody a
    # deliberate keystroke. It is a TEST lever -- no production path drops the
    # prefix cache -- so gating it costs nothing a workload needs.
    "gpu_cache_drop_allowed": False,
}

# Knobs whose value must stay OUT of .env.dspark or the file cannot move them.
# (Every knob, really -- but these are the ones with a live env setting or a
# legacy name, i.e. the ones where the mistake is already half-made.)
MUST_NOT_BE_ENV_PINNED = ("protected_fraction",)

# protected_fraction predates the VLLM_<NAME> convention.  Renaming it would
# silently drop the setting that is live on both nodes, so the old name is
# honoured for boot-pinning and documented rather than migrated.
LEGACY_FRACTION_ENV = "VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION"

_VALID = {
    "protect_alloc": ("equal", "by_need"),
    "evict_order": ("fifo", "lru", "lru_tail_first"),
}

# *** KNOBS THAT CANNOT BE BUILT AT THIS LAYER, with the reason. ***
#
# Setting one of these does nothing useful, so it must say why rather than fail
# silently.  The reason is carried here, next to the knob, because the next
# person to try will reach for the flag before the write-up.
# *** disk_restore WAS IN HERE AND IT WAS LIVE ANYWAY, which is the defect this
# removal fixes rather than hides. *** The restore gates read it with `raw()`,
# which is documented as doing no validation -- so the flags file turned the
# feature ON while BLOCKED said it was off and the /kv endpoint reported
# `disk_restore: False`. A feature running while every reader denies it is the
# "refusal nobody can see" failure pointed the other way round, and it is worse,
# because the thing nobody can see is ENABLED.
#
# Two changes together: it is out of this table (it is built, and Phase 3 is
# deliberately exercising it), and the gates now use a VALIDATED read so that an
# entry here means blocked FOR REAL rather than blocked-unless-someone-used-raw.
BLOCKED: dict[str, str] = {
    # *** THESE REASONS WERE REWRITTEN 2026-10-07 AND THE OLD ONES WERE WRONG,
    # which is worth keeping because they were wrong in an instructive way. ***
    # They said a disk tier was unbuildable at the manager layer -- true, a
    # KVCacheBlock holds no bytes -- and then concluded the whole path was
    # parked on vllm#45663. It is not. vllm/v1/kv_offload RAN, wrote 27 GB, and
    # the real obstacles turned out to be two things that reason never mentioned:
    # a random per-boot Merkle root (fixed: PYTHONHASHSEED), and upstream's
    # offload consolidating every rank's shard into ONE SHARED HOST region,
    # which is single-host TP by construction while ours is one rank per node.
    "disk_enabled": (
        "superseded. The write path works and is being replaced rather than "
        "unblocked: kvmgr/diskspec.py puts the disk transfer WORKER-SIDE so "
        "each rank persists its own slice. Until that spec is the live one, "
        "this knob toggles nothing, because what writes today is upstream's "
        "tiering tier via KV_TRANSFER_CONFIG and not a flag. "
        "See kvmgr/diskspec.py and docs/history/KVMGR-PHASE45-20261007.md"
    ),
    # *** disk_budget_gb IS OUT OF HERE: Phase 5 built it. *** The old reason
    # said "nothing we control writes to disk yet", which stopped being true
    # the moment our own spec wrote 518 GB. Leaving a built knob in this table
    # is the defect `disk_restore` already demonstrated -- a capability running
    # behind readers that deny it, which is worse than a refusal nobody can
    # see, because the invisible thing is ENABLED. Enforcement is in
    # DiskPresenceManager._enforce_budget; 0 means unbounded.
}

# Values implemented SO FAR.  A knob set to a recognised-but-unbuilt value falls
# back to the safe default and SAYS SO -- shipping a half-built policy as though
# it were live is worse than refusing it.
# Phase 3 built all of them. The table stays, because the next unbuilt value
# must refuse rather than ship looking live.
_IMPLEMENTED: dict[str, tuple] = {}

_last_seen: dict[str, Any] = {}

# (name, repr(value)) pairs already announced, so a REFUSAL is reported once and
# not on every poll. Needed because the refusal is about the VALUE, not the edge:
# `peek()` is called by readers on a timer, and `knob()` by the manager.
_announced: set[tuple[str, str]] = set()


def _announce(key: tuple[str, str], emit) -> None:
    if key in _announced:
        return
    _announced.add(key)
    emit()


def _validate(name: str, value: Any) -> Any:
    """The refusals, shared by knob() and peek().

    *** THIS LIVES HERE AND NOT IN knob() BECAUSE OF A BUG I INTRODUCED. ***
    The BLOCKED warning was in knob(), and `snapshot()` was its only production
    caller. Making snapshot() use peek() -- correct, so a reader cannot consume
    the change edge -- left NOTHING calling knob() on a disk flag, so
    `disk_enabled=1` was refused SILENTLY. Verified on the live engine
    2026-10-07: the flag was set, correctly ignored, and no line said why.
    A refusal nobody can see is the defect it exists to prevent.
    """
    allowed = _VALID.get(name)
    if allowed is not None and value not in allowed:
        _announce((name, repr(value)), lambda: logger.warning(
            "kvmgr: %s=%r is not one of %s; falling back to %r",
            name, value, allowed, DEFAULTS[name]))
        return DEFAULTS[name]

    built = _IMPLEMENTED.get(name)
    if built is not None and value not in built:
        _announce((name, repr(value)), lambda: logger.warning(
            "kvmgr: %s=%r is recognised but NOT YET IMPLEMENTED (built: %s); "
            "using %r. This is a refusal, not a silent downgrade.",
            name, value, built, DEFAULTS[name]))
        return DEFAULTS[name]

    reason = BLOCKED.get(name)
    if reason is not None and value != DEFAULTS[name]:
        _announce((name, repr(value)), lambda: logger.warning(
            "kvmgr: %s=%r REFUSED -- %s. Using %r.",
            name, value, reason, DEFAULTS[name]))
        return DEFAULTS[name]
    return value


# ---------------------------------------------------------------------------
# The IN-MEMORY override, which is Stage 6's whole point.
#
# *** A HOT KNOB MUST NOT DEPEND ON DISK OR CONTAINER INJECTION. *** Everything
# above reaches a knob through `/var/tmp/vllm-kv-flags/flags.json`, which works
# and is why every policy knob is hot today -- but as the ONLY way in it has
# four defects: it is a bind mount, so the control plane depends on the
# container spec; it is PER NODE, so writing it on the head does not change the
# worker; it returns no acknowledgement, so "did it take?" is answered by
# grepping a container log; and it is not atomic, because three processes each
# re-stat it on their own 1s TTL.
#
# So an explicit set lands HERE, in memory, in whichever process was told --
# and the control plane fans it out to the others over the engine's own RPC.
# The file survives as the BOOT-TIME DEFAULT, which is a real job: a process
# restart has to start somewhere.
_OVERRIDE: dict[str, Any] = {}


def _coerce(name: str, value: Any) -> Any:
    """Coerce an HTTP-supplied value to the knob's declared type.

    HTTP and JSON hand us strings, and `bool("false")` is **True** -- which
    would make `?on=false` turn a thing ON. That one is not a hypothetical: it
    is the single most likely way this plane could do the opposite of what it
    was asked, so booleans are parsed from a word list and anything else is
    refused by name rather than coerced by truthiness.
    """
    want = type(DEFAULTS[name])
    if want is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        t = str(value).strip().lower()
        if t in ("1", "true", "yes", "on"):
            return True
        if t in ("0", "false", "no", "off"):
            return False
        raise ValueError(
            f"kvmgr: {name}={value!r} is not a boolean. Use true/false, 1/0, "
            f"yes/no or on/off -- NOT coerced by truthiness, because "
            f"bool('false') is True and that would invert the request.")
    if want is int:
        return int(value)
    if want is float:
        return float(value)
    return str(value)


def set_override(name: str, value: Any) -> dict:
    """Set a knob in THIS process, above the env and the file. Returns the act."""
    if name not in DEFAULTS:
        raise KeyError(
            f"kvmgr: unknown knob {name!r}. Known: {sorted(DEFAULTS)}")
    before = raw(name)
    _OVERRIDE[name] = _coerce(name, value)
    after = raw(name)
    logger.info(
        "kvmgr: knob %s set to %r by the control plane (was %r, now reads %r; "
        "source=override, which outranks the env pin and the file)",
        name, _OVERRIDE[name], before, after)
    return {"name": name, "set_to": _OVERRIDE[name], "was": before,
            "now": after, "source": "override"}


def clear_override(name: str | None = None) -> list[str]:
    """Drop one override, or all of them, so the file/env governs again."""
    if name is None:
        dropped = sorted(_OVERRIDE)
        _OVERRIDE.clear()
    else:
        if name not in DEFAULTS:
            # The SAME refusal as set_override's. Two spellings of one error is
            # how a caller learns the knob list from one entry point and not
            # the other, and a refusal is only useful if it is actionable.
            raise KeyError(
                f"kvmgr: unknown knob {name!r}. Known: {sorted(DEFAULTS)}")
        dropped = [name] if _OVERRIDE.pop(name, None) is not None else []
    if dropped:
        logger.info("kvmgr: override(s) cleared, file/env governs again: %s",
                    dropped)
    return dropped


def source_of(name: str) -> str:
    """Which source is winning for this knob: override | env | file | default.

    Reported beside every value, because the defect this plane exists to fix is
    a knob that was set and did nothing -- and the only honest answer to "did
    it take?" names the source that won rather than echoing the value back.
    """
    if name in _OVERRIDE:
        return "override"
    env_name = (LEGACY_FRACTION_ENV if name == "protected_fraction"
                else "VLLM_" + name.upper())
    if (os.environ.get(env_name) or "").strip():
        return "env"
    try:
        # patch6's own reader, cached on its 1s TTL -- not a second parse of the
        # same file, which is how two readers of one knob come to disagree.
        from vllm.v1.core.single_type_kv_cache_manager import _flags

        if name in (_flags() or {}):
            return "file"
    except Exception:
        pass
    return "default"


def raw(name: str) -> Any:
    """The value, with no logging and no validation. Hot-path safe."""
    if name not in DEFAULTS:
        raise KeyError(
            f"kvmgr: unknown knob {name!r}. Add it to DEFAULTS -- a knob that is "
            "not in the table is a knob no hot-config test covers."
        )
    # *** THE OVERRIDE OUTRANKS EVERYTHING, INCLUDING THE ENV PIN. *** That is
    # deliberate and it is the reversal Stage 6 is for: an env pin is a decision
    # taken at boot, and a POST is a decision taken NOW by somebody watching the
    # engine. The old precedence made `.env.dspark` able to freeze a knob with
    # no way to thaw it short of a restart, which is the trap `audit_env()`
    # exists to make visible -- visible, but not fixable. This fixes it, and
    # `source_of()` keeps it honest by naming the winner.
    if name in _OVERRIDE:
        return _OVERRIDE[name]
    if name == "protected_fraction":
        # ONE DEFINITION, patch6's. Re-implementing the precedence here is how
        # two readers of one knob end up disagreeing -- and this knob has three
        # sources (legacy env, flags file, default) plus the blank-is-unset rule
        # that lets the compose stop pinning it.
        from vllm.v1.core.single_type_kv_cache_manager import _protected_fraction

        return _protected_fraction()
    return _patch6_flag(name, DEFAULTS[name])


def knob(name: str, *, quiet: bool = False) -> Any:
    """The value, validated, logging on CHANGE.

    `quiet` is for a caller that will log a richer line itself (the cap logs
    blocks AND tokens per group, which is the figure that actually matters);
    the change is still recorded so `changed()` stays accurate.
    """
    value = _validate(name, raw(name))

    previous = _last_seen.get(name, _MISSING)
    if previous is _MISSING:
        _last_seen[name] = value
        if not quiet:
            logger.info("kvmgr: %s = %r (initial)", name, value)
    elif previous != value:
        _last_seen[name] = value
        if not quiet:
            logger.info("kvmgr: %s CHANGED %r -> %r (no restart)", name, previous, value)
    return value


class _Missing:
    def __repr__(self) -> str:
        return "<unset>"


_MISSING = _Missing()


def peek(name: str):
    """The validated value WITHOUT touching `_last_seen`.

    *** A READER MUST NOT CONSUME THE CHANGE EDGE. ***  `knob()` and `changed()`
    both record what they saw, which is what makes "log on change" and
    "reconcile on the edge" work -- and it means a READER calling `knob()` can
    swallow a flag change before the manager ever notices it.

    Observed live 2026-10-07: the stats endpoint polled `knob("protect_alloc")`,
    recorded the new value, and `new_step_starts()` then found nothing changed --
    so `protect_alloc=by_need` was reported as in force by the very endpoint that
    had prevented it being applied. Observability must not alter behaviour, and a
    dashboard polling every 30s would have made flag changes intermittent.
    """
    return _validate(name, raw(name))


def changed(name: str) -> bool:
    """True if `name` differs from the last value `knob()`/`changed()` saw.

    The edge, so a caller can reconcile promptly instead of waiting for the next
    natural call on that path.  Consumes the edge: a second call is False.
    """
    value = _validate(name, raw(name))
    previous = _last_seen.get(name, _MISSING)
    if previous is _MISSING or previous != value:
        _last_seen[name] = value
        return previous is not _MISSING
    return False


def audit_env() -> list[str]:
    """Knobs an env var is currently pinning, i.e. ones the file cannot move.

    Not a failure -- boot-pinning is a legitimate use of env -- but it has to be
    VISIBLE, because the symptom otherwise is a flag flip that does nothing and
    reports nothing.
    """
    pinned = []
    for name in DEFAULTS:
        env_name = LEGACY_FRACTION_ENV if name == "protected_fraction" else "VLLM_" + name.upper()
        value = os.environ.get(env_name)
        # *** A BLANK VALUE IS NOT A PIN. *** docker-compose emits "" for
        # ${VAR:-} when the variable is unset, and "" is not None -- so the
        # first version of this reported
        #   protected_fraction (pinned by VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION='')
        # on the very boot where the knob had just been made hot. Worse than
        # noise: it tells the operator a live knob is cold, and invites them to
        # go hunting for an env var that is already effectively unset.
        # _protected_fraction() treats blank as absent; this must agree with it.
        if value is not None and value.strip() != "":
            pinned.append(f"{name} (pinned by {env_name}={value!r})")
    return pinned


def log_env_audit() -> None:
    pinned = audit_env()
    if not pinned:
        logger.info("kvmgr: no knob is env-pinned; the flags file governs all of them")
        return
    for entry in pinned:
        severity = logger.warning if entry.split()[0] in MUST_NOT_BE_ENV_PINNED else logger.info
        severity(
            "kvmgr: %s -- the flags file CANNOT move this knob until the env var "
            "is removed", entry,
        )


def reset_for_tests() -> None:
    # Overrides are process-global, so a test that sets one and does not
    # clear it leaks into every later test in the same run -- which is the
    # quietest way for a suite to start passing for the wrong reason.
    _OVERRIDE.clear()
    _last_seen.clear()
    _announced.clear()
