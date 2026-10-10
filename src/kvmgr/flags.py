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
        "See kvmgr/diskspec.py and docs/local/KVMGR-PHASE45-20261007.md"
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
