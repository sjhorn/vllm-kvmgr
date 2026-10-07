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

logger = init_logger(__name__)

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
    "kvmgr_log_level": 0,
    # 0 means BIND NOTHING, deliberately unlike a route that answers 404
    # because a boolean said so. Same convention as METER_TAP_PORT.
    "kvmgr_stats_port": 0,
    # pre-existing, preserved unchanged -- listed so this table is the whole
    # surface and nothing is hot-by-accident or cold-by-omission.
    "swa_recycle": True,
    "swa_recycle_cached": False,
    "free_queue_prepend_uncached": False,
    "pin_static_head": False,
    "head_pin_fraction": 0.05,
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
BLOCKED: dict[str, str] = {
    "disk_enabled": (
        "a KV cache manager holds NO KV BYTES. KVCacheBlock is metadata only "
        "(block_id, ref_cnt, hash) and across the whole manager/coordinator "
        "layer there is not one reference to a tensor or a device -- the "
        "tensors live in the WORKER processes, indexed by block_id. Writing "
        "them needs worker-side code, i.e. a KVConnector, which is what "
        "vllm/v1/kv_offload already is. That path is parked on vllm#45663: the "
        "sparse indexer materialises every KV block at once and OOMs past "
        "~120K tokens against a 570K working set. See "
        "docs/local/KVMGR-PHASE45-20261007.md"
    ),
    "disk_restore": "same layer problem as disk_enabled; see its reason",
    "disk_budget_gb": "nothing writes to disk, so a byte budget bounds nothing",
}

# Values implemented SO FAR.  A knob set to a recognised-but-unbuilt value falls
# back to the safe default and SAYS SO -- shipping a half-built policy as though
# it were live is worse than refusing it.
# Phase 3 built all of them. The table stays, because the next unbuilt value
# must refuse rather than ship looking live.
_IMPLEMENTED: dict[str, tuple] = {}

_last_seen: dict[str, Any] = {}


def raw(name: str) -> Any:
    """The value, with no logging and no validation. Hot-path safe."""
    if name not in DEFAULTS:
        raise KeyError(
            f"kvmgr: unknown knob {name!r}. Add it to DEFAULTS -- a knob that is "
            "not in the table is a knob no hot-config test covers."
        )
    if name == "protected_fraction":
        legacy = os.environ.get(LEGACY_FRACTION_ENV)
        if legacy is not None:
            try:
                return float(legacy)
            except ValueError:
                return DEFAULTS[name]
    return _patch6_flag(name, DEFAULTS[name])


def knob(name: str, *, quiet: bool = False) -> Any:
    """The value, validated, logging on CHANGE.

    `quiet` is for a caller that will log a richer line itself (the cap logs
    blocks AND tokens per group, which is the figure that actually matters);
    the change is still recorded so `changed()` stays accurate.
    """
    value = raw(name)

    allowed = _VALID.get(name)
    if allowed is not None and value not in allowed:
        logger.warning(
            "kvmgr: %s=%r is not one of %s; falling back to %r",
            name, value, allowed, DEFAULTS[name],
        )
        value = DEFAULTS[name]

    built = _IMPLEMENTED.get(name)
    if built is not None and value not in built:
        logger.warning(
            "kvmgr: %s=%r is recognised but NOT YET IMPLEMENTED (built: %s); "
            "using %r. This is a refusal, not a silent downgrade.",
            name, value, built, DEFAULTS[name],
        )
        value = DEFAULTS[name]

    reason = BLOCKED.get(name)
    if reason is not None and value != DEFAULTS[name]:
        logger.warning(
            "kvmgr: %s=%r REFUSED -- %s. Using %r.",
            name, value, reason, DEFAULTS[name],
        )
        value = DEFAULTS[name]

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


def changed(name: str) -> bool:
    """True if `name` differs from the last value `knob()`/`changed()` saw.

    The edge, so a caller can reconcile promptly instead of waiting for the next
    natural call on that path.  Consumes the edge: a second call is False.
    """
    value = raw(name)
    allowed = _VALID.get(name)
    if allowed is not None and value not in allowed:
        value = DEFAULTS[name]
    built = _IMPLEMENTED.get(name)
    if built is not None and value not in built:
        value = DEFAULTS[name]
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
        if os.environ.get(env_name) is not None:
            pinned.append(f"{name} (pinned by {env_name}={os.environ[env_name]!r})")
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
    _last_seen.clear()
