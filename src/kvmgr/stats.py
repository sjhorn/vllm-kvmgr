"""kvmgr.stats -- per-group counters for what the cache actually did.

WHY THIS IS NOT A STAT LOGGER, which is what the plan asked for.

The plan says to export "via the existing vllm.stat_logger_plugins entry point
(as prefill_collector.py already does -- no fork)".  **A stat logger cannot see
these numbers.**  Stat loggers run on AsyncLLM's output-handler loop -- the API
server process -- which prefill_collector's own header states in its own words;
the KV cache managers run in the ENGINE CORE process, behind the IPC that
carries EngineCoreOutputs.  Module-level counters in this file are therefore
invisible to a stat logger, and a stat logger that read them would report zeros
forever.

That is exactly the failure this stack keeps paying for -- a number the page
knows how to draw and the wire can never fill (the prefix gauge, twice; the
Usage view's 0% cache share).  So the export is a loopback JSON endpoint served
FROM THE PROCESS THAT HOLDS THE COUNTERS, which is the same shape
prefill_collector already uses for `/prefill` and which sparkstats already knows
how to proxy and turn into Prometheus series.

WHY THE COUNTERS ARE MODULE-LEVEL rather than per-manager instance attributes.
``find_longest_cache_hit`` is a CLASSMETHOD -- the coordinator calls it on the
class, not the instance -- so the hit/miss path has no ``self`` to count on.
See docs/local/KVMGR-PHASE0-20261007.md finding 1.  Everything is keyed by
``kv_cache_group_id`` so a module-level home loses nothing.

IT MUST NOT BE ABLE TO HARM A REQUEST.  Every counter is a plain int behind the
GIL, every public entry point is total (no exceptions, no allocation in the
common path), and nothing here ever touches a request or a block.  Same
discipline as prefill_collector, for the same reason: the scheduler loop is the
least forgiving place in the engine.

IT HOLDS NO CONTENT.  Counts and group ids only.  There is no token id, no
block hash and no prompt anywhere in this file, by construction.
"""

from __future__ import annotations

import time
from typing import Any

# Why each eviction chose its victim.  The plan asks for "why" per eviction;
# a free-text reason would be unaggregatable, so the reasons are an enum and the
# counter is per (group, reason).
EVICT_CAP = "cap"           # the protected set exceeded its budget
EVICT_POOL = "pool"         # allocation pressure: release_protected_prompt_blocks
EVICT_RESET = "reset"       # a flag change shrank the budget; reconciled

VICTIM_FIFO = "fifo_oldest_protected"       # patch6's rule: first protected
VICTIM_LRU = "lru_least_recently_reused"    # Phase 3
VICTIM_LRU_TAIL = "lru_tail_first"          # Phase 3

_START = time.monotonic()


class GroupStats:
    """One KV group's counters. Plain ints; no locks on the hot path."""

    __slots__ = (
        "group_id", "block_size",
        "hit_calls", "hit_blocks", "hit_tokens",
        "miss_calls", "partial_calls",
        "protect_calls", "protect_blocks",
        "evict_cap", "evict_pool", "evict_reset", "evict_blocks",
        "victims", "cap_blocks", "cap_tokens", "protected_now",
        "restore_hits", "restore_blocks", "restore_refused",
    )

    def __init__(self, group_id: int, block_size: int):
        self.group_id = group_id
        self.block_size = block_size
        # find_longest_cache_hit outcomes
        self.hit_calls = 0
        self.hit_blocks = 0
        self.hit_tokens = 0
        self.miss_calls = 0        # returned nothing at all
        self.partial_calls = 0     # returned less than it was asked for
        # protection
        self.protect_calls = 0
        self.protect_blocks = 0
        # eviction, by cause
        self.evict_cap = 0
        self.evict_pool = 0
        self.evict_reset = 0
        self.evict_blocks = 0
        self.victims: dict[str, int] = {}
        # current state, so a reader need not compute the cap itself
        self.cap_blocks: int | None = None
        self.cap_tokens: int | None = None
        self.protected_now = 0
        # Phase 4/5, present from the start so the shape does not change later
        self.restore_hits = 0
        self.restore_blocks = 0
        self.restore_refused = 0

    def as_dict(self) -> dict[str, Any]:
        total = self.hit_calls + self.miss_calls
        return {
            "group_id": self.group_id,
            "block_size": self.block_size,
            "lookups": total,
            "hit_calls": self.hit_calls,
            "miss_calls": self.miss_calls,
            "partial_calls": self.partial_calls,
            # A RATE is derived here and not on the page, because "hits over
            # lookups" is the one figure two readers would compute differently.
            # None rather than 0 when nothing was looked up: absent is not zero.
            "hit_rate": (self.hit_calls / total) if total else None,
            "hit_blocks": self.hit_blocks,
            "hit_tokens": self.hit_tokens,
            "protect_calls": self.protect_calls,
            "protect_blocks": self.protect_blocks,
            "evict_cap": self.evict_cap,
            "evict_pool": self.evict_pool,
            "evict_reset": self.evict_reset,
            "evict_blocks": self.evict_blocks,
            "victims": dict(self.victims),
            "cap_blocks": self.cap_blocks,
            "cap_tokens": self.cap_tokens,
            "protected_now": self.protected_now,
            "protected_tokens_now": self.protected_now * self.block_size,
            "restore_hits": self.restore_hits,
            "restore_blocks": self.restore_blocks,
            "restore_refused": self.restore_refused,
        }


_groups: dict[int, GroupStats] = {}


def group(group_id: int, block_size: int = 0) -> GroupStats:
    g = _groups.get(group_id)
    if g is None:
        g = GroupStats(group_id, block_size)
        _groups[group_id] = g
    elif block_size and not g.block_size:
        g.block_size = block_size
    return g


# --- the recording entry points.  Total functions: they never raise. ---------

def record_lookup(group_id: int, block_size: int, got_blocks: int,
                  asked_blocks: int) -> None:
    """One `find_longest_cache_hit` outcome.

    THREE OUTCOMES, NOT TWO, and the third is the one worth having: a PARTIAL
    hit is a prefix that survived in part, which is a completely different
    event from a miss and is what tail-first eviction is meant to produce.
    Collapsing them would hide whether Phase 3 worked.
    """
    try:
        g = group(group_id, block_size)
        if got_blocks <= 0:
            g.miss_calls += 1
            return
        g.hit_calls += 1
        g.hit_blocks += got_blocks
        g.hit_tokens += got_blocks * (block_size or g.block_size)
        if asked_blocks > got_blocks:
            g.partial_calls += 1
    except Exception:
        pass


def record_protect(group_id: int, blocks: int, protected_now: int) -> None:
    try:
        g = group(group_id)
        g.protect_calls += 1
        g.protect_blocks += blocks
        g.protected_now = protected_now
    except Exception:
        pass


def record_evict(group_id: int, cause: str, victim_rule: str,
                 blocks: int = 1, protected_now: int | None = None) -> None:
    try:
        g = group(group_id)
        if cause == EVICT_CAP:
            g.evict_cap += 1
        elif cause == EVICT_POOL:
            g.evict_pool += 1
        elif cause == EVICT_RESET:
            g.evict_reset += 1
        g.evict_blocks += blocks
        g.victims[victim_rule] = g.victims.get(victim_rule, 0) + 1
        if protected_now is not None:
            g.protected_now = protected_now
    except Exception:
        pass


def record_cap(group_id: int, cap_blocks: int | None, block_size: int) -> None:
    try:
        g = group(group_id, block_size)
        g.cap_blocks = cap_blocks
        g.cap_tokens = None if cap_blocks is None else cap_blocks * block_size
    except Exception:
        pass


def record_protected_now(group_id: int, protected_now: int) -> None:
    try:
        group(group_id).protected_now = protected_now
    except Exception:
        pass


# --- reading ----------------------------------------------------------------

def snapshot() -> dict[str, Any]:
    from . import flags

    return {
        "available": True,
        "uptime_s": round(time.monotonic() - _START, 1),
        # The policy in force, beside the counters, so a reader never has to
        # correlate two sources to know which arm produced these numbers.
        "policy": {
            "kvmgr_enabled": bool(flags.raw("kvmgr_enabled")),
            "protected_fraction": float(flags.raw("protected_fraction")),
            "protect_alloc": flags.knob("protect_alloc", quiet=True),
            "evict_order": flags.knob("evict_order", quiet=True),
            # knob() rather than raw(), so a BLOCKED knob reports what is in
            # force (False) rather than what somebody asked for -- a reader
            # must not see disk_enabled true against a tier that cannot exist.
            "disk_enabled": bool(flags.knob("disk_enabled", quiet=True)),
            "disk_restore": bool(flags.knob("disk_restore", quiet=True)),
            "blocked": {k: True for k in flags.BLOCKED},
        },
        "env_pinned": flags.audit_env(),
        "groups": [g.as_dict() for _, g in sorted(_groups.items())],
    }


def prometheus() -> str:
    """Prometheus text format.

    Served beside the JSON rather than instead of it: sparkstats scrapes
    Prometheus and the shape is what its registry already understands, while the
    JSON is what a human reads over the tunnel.  Both are the same counters, so
    they cannot disagree.

    Names are `kvmgr_*` with a `group` label.  No metric is emitted for a value
    that is None -- an absent cap is not a cap of zero, and a hit rate over no
    lookups is not 0%.
    """
    lines: list[str] = []

    def emit(name: str, help_text: str, kind: str, samples: list[tuple[str, Any]]) -> None:
        samples = [(lbl, v) for lbl, v in samples if v is not None]
        if not samples:
            return
        lines.append(f"# HELP kvmgr_{name} {help_text}")
        lines.append(f"# TYPE kvmgr_{name} {kind}")
        for labels, value in samples:
            lines.append(f"kvmgr_{name}{{{labels}}} {value}")

    gs = [g for _, g in sorted(_groups.items())]

    def per_group(attr: str):
        return [(f'group="{g.group_id}"', getattr(g, attr)) for g in gs]

    emit("lookups_total", "find_longest_cache_hit calls", "counter",
         [(f'group="{g.group_id}"', g.hit_calls + g.miss_calls) for g in gs])
    emit("hits_total", "lookups that matched at least one block", "counter",
         per_group("hit_calls"))
    emit("misses_total", "lookups that matched nothing", "counter",
         per_group("miss_calls"))
    emit("partial_hits_total",
         "lookups that matched less than asked -- a prefix that survived in part",
         "counter", per_group("partial_calls"))
    emit("hit_blocks_total", "blocks returned by a hit", "counter",
         per_group("hit_blocks"))
    emit("hit_tokens_total", "tokens covered by a hit", "counter",
         per_group("hit_tokens"))
    emit("protect_blocks_total", "blocks added to the protected set", "counter",
         per_group("protect_blocks"))
    emit("evictions_total", "protected blocks released, by cause", "counter",
         [(f'group="{g.group_id}",cause="cap"', g.evict_cap) for g in gs]
         + [(f'group="{g.group_id}",cause="pool"', g.evict_pool) for g in gs]
         + [(f'group="{g.group_id}",cause="reset"', g.evict_reset) for g in gs])
    emit("evict_victim_total", "evictions by the rule that chose the victim",
         "counter",
         [(f'group="{g.group_id}",rule="{rule}"', n)
          for g in gs for rule, n in sorted(g.victims.items())])
    emit("protected_cap_blocks", "the protected-set budget, in blocks", "gauge",
         per_group("cap_blocks"))
    emit("protected_cap_tokens",
         "the protected-set budget as context extent -- the figure an operator "
         "can act on, since blocks are not comparable across groups", "gauge",
         per_group("cap_tokens"))
    emit("protected_blocks", "blocks currently protected", "gauge",
         per_group("protected_now"))
    emit("protected_tokens", "context extent currently protected", "gauge",
         [(f'group="{g.group_id}"', g.protected_now * g.block_size) for g in gs])
    emit("restore_hits_total", "prefixes read back from disk", "counter",
         per_group("restore_hits"))
    emit("restore_refused_total",
         "disk entries refused -- fingerprint mismatch or budget", "counter",
         per_group("restore_refused"))
    return "\n".join(lines) + "\n" if lines else ""


def reset_for_tests() -> None:
    _groups.clear()
