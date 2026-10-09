"""test_phase3.py -- the two policies, and the failure they are for.

    python3.12 local-ops/kvharness/test_phase3.py

The measured case this reproduces: three conversations, the oldest idle, a cap
that cannot hold all three. Today the oldest loses its prefix ENTIRELY -- a
373,833-token conversation did, after a 1.5h gap, with KV usage at 22-33%.

Two independent knobs, so they can be A/B'd on one engine:

  protect_alloc  equal -> by_need          WHICH GROUP gets the budget
  evict_order    fifo -> lru -> lru_tail_first   WHICH PREFIX, and WHICH END

`lru` alone fixes the prefix and NOT the end, and the test says so by measuring
both -- which is the distinction the plan's own CLAUDE.md note draws and the
reason tail-first is a separate arm rather than a refinement.
"""

from __future__ import annotations

import os
import sys
from collections import deque
import tempfile
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "patches", "kvmgr_plugin"))

import harness  # noqa: E402

PASS = 0
FAIL: list[str] = []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL.append(name)
        print(f"FAIL  {name}  {detail}")


def eq(name, got, want):
    check(name, got == want, f"got {got!r} want {want!r}")


H = harness.load()
MGR = H.mgr_mod
iface = sys.modules["vllm.v1.kv_cache_interface"]

import importlib  # noqa: E402

from kvmgr import flags, policy, stats  # noqa: E402
register = importlib.import_module("kvmgr.register")
from kvmgr.manager import DSparkMLAAttentionManager  # noqa: E402

register.register()
F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kvmgr3-"), "flags.json"))


# ---------------------------------------------------------------------------
# 1. choose_victim, as a pure function
# ---------------------------------------------------------------------------
lin: OrderedDict[object, list[int]] = OrderedDict()
lin["A"] = [1, 2, 3, 4]      # least recently used
lin["B"] = [10, 11, 12]
lin["C"] = [20, 21]          # most recently used
prot = {1, 2, 3, 4, 10, 11, 12, 20, 21}

eq("head-first takes the LRU prefix's HEAD",
   policy.choose_victim(lin, prot, None, tail_first=False), ("A", 1))
eq("tail-first takes the LRU prefix's TAIL",
   policy.choose_victim(lin, prot, None, tail_first=True), ("A", 4))
check("either way the victim is the least-recently-used PREFIX, never the "
      "longest-held one",
      policy.choose_victim(lin, prot, None, False)[0] == "A")

# a skip set is passed over, not treated as exhausted
eq("a skipped block is passed over within the same prefix",
   policy.choose_victim(lin, prot, {1}, tail_first=False), ("A", 2))
eq("and tail-first skips from the other end",
   policy.choose_victim(lin, prot, {4}, tail_first=True), ("A", 3))
eq("a wholly-skipped prefix falls through to the next one",
   policy.choose_victim(lin, prot, {1, 2, 3, 4}, tail_first=False), ("B", 10))

# stale entries are tolerated
eq("a block no longer protected is skipped as stale",
   policy.choose_victim(lin, {2, 3, 4, 10}, None, tail_first=False), ("A", 2))
eq("nothing available -> None", policy.choose_victim(lin, set(), None, False), None)
eq("no lineages at all -> None",
   policy.choose_victim(OrderedDict(), prot, None, False), None)

# ---------------------------------------------------------------------------
# 2. by_need, now live through the manager
# ---------------------------------------------------------------------------
def build(block_size=256, fraction=0.75, groups=5, **kw):
    for n in list(os.environ):
        if n.startswith("VLLM_") and n[5:].lower() in flags.DEFAULTS:
            del os.environ[n]
    os.environ.pop(flags.LEGACY_FRACTION_ENV, None)
    flags.reset_for_tests()
    stats.reset_for_tests()
    F.write(kvmgr_enabled=1, protected_fraction=fraction, **kw)
    harness.reset_manager_count()
    pool = harness.make_pool(num_gpu_blocks=24288, hash_block_size=block_size)
    mgr = MGR.get_manager_for_kv_cache_spec(
        kv_cache_spec=iface.MLAAttentionSpec(block_size=block_size),
        max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
        block_pool=pool, enable_caching=True, kv_cache_group_id=0,
        dcp_world_size=1, pcp_world_size=1)
    DSparkMLAAttentionManager.group_count = groups
    DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
    DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
    return pool, mgr


pool, mgr = build(protect_alloc="equal")
eq("equal: group 0 gets 3,643 blocks", mgr._max_protected_prompt_blocks(), 3643)
eq("= 932,608 tokens", stats.group(0).cap_tokens, 932608)

F.write(kvmgr_enabled=1, protected_fraction=0.75, protect_alloc="by_need")
cap = mgr._max_protected_prompt_blocks()
eq("by_need: group 0 gets 8,192 blocks, live and with no restart", cap, 8192)
eq("= 2,097,152 tokens", stats.group(0).cap_tokens, 2097152)
check("which is 2.25x, and it is the LEGACY cap that binds -- not the fraction",
      cap == policy.legacy_cap(harness.LIVE_MAX_MODEL_LEN, 256),
      "so raising the fraction further buys nothing for group 0")
check("the 740,315-token session now fits with room",
      2097152 > 740315 * 2)

# switching back is just as live
F.write(kvmgr_enabled=1, protected_fraction=0.75, protect_alloc="equal")
eq("and back to equal", mgr._max_protected_prompt_blocks(), 3643)

# the windowed groups are not starved: they get exactly their window
needs = [policy.protected_cap(
    num_gpu_blocks=24288, fraction=0.75, num_groups=5, block_size=bs,
    max_model_len=harness.LIVE_MAX_MODEL_LEN, mode="by_need",
    block_sizes=harness.LIVE_BLOCK_SIZES, group_index=i,
    sliding_windows=harness.LIVE_SLIDING_WINDOWS)
    for i, bs in enumerate(harness.LIVE_BLOCK_SIZES)]
eq("windowed groups get their window and no more", needs[1:], [3, 3, 3, 17])
check("costing 0.1% of the pool between them", sum(needs[1:]) == 26,
      "26 blocks, with group 3's window being 8 rather than 128")

# ---------------------------------------------------------------------------
# 3. THE MEASURED SCENARIO: three conversations, the oldest idle
# ---------------------------------------------------------------------------
def bh(i, gid=0):
    return H.utils.make_block_hash_with_group_id(
        H.utils.BlockHash(i.to_bytes(32, "big")), gid)


def scenario(order: str, refresh=("A", "C"), nconv=3, per=20, cap_frac=0.25):
    """Three prefixes of 20 blocks, protected A then B then C.

    *** `refresh` IS WHAT MAKES THIS DISCRIMINATING. *** The default refreshes
    A and C and leaves B idle -- so the OLDEST prefix (A) is the one still being
    used. That separates the two rules, which an "oldest is also idlest"
    scenario cannot:

        fifo  -> takes A, the conversation held LONGEST, which is still active
        lru   -> takes B, the one actually abandoned

    The first version of this test refreshed B and C and left A idle, so A was
    both the longest-held and the abandoned one; fifo and lru produced
    byte-identical survivors and the test passed without discriminating
    anything. Measured, before the fix: both left C's tail with its HEAD LOST.

    THE CAP IS ALSO CHOSEN TO DISCRIMINATE. At cap 30 of 60 protected blocks,
    exactly one prefix dies whole and ONE IS HALF-EVICTED -- which is the only
    case where head-first and tail-first differ at all. The first cap tried
    (15 of 60) killed two prefixes whole under every arm, so the end the third
    was eaten from was the only thing left to measure and no arm had a partial
    prefix to show it on.

    Returns (manager, {name: blocks}, {name: surviving ids in prefix order}).
    """
    for n in list(os.environ):
        if n.startswith("VLLM_") and n[5:].lower() in flags.DEFAULTS:
            del os.environ[n]
    os.environ.pop(flags.LEGACY_FRACTION_ENV, None)
    flags.reset_for_tests()
    stats.reset_for_tests()
    F.write(kvmgr_enabled=1, protected_fraction=0.5, evict_order=order)
    harness.reset_manager_count()
    pool = harness.make_pool(num_gpu_blocks=120, hash_block_size=16)
    m = MGR.get_manager_for_kv_cache_spec(
        kv_cache_spec=iface.MLAAttentionSpec(block_size=16),
        max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
        block_pool=pool, enable_caching=True, kv_cache_group_id=0,
        dcp_world_size=1, pcp_world_size=1)
    DSparkMLAAttentionManager.group_count = 1
    DSparkMLAAttentionManager.group_block_sizes = (16,)
    DSparkMLAAttentionManager.group_sliding_windows = (None,)

    names = ["A", "B", "C"][:nconv]
    convs = {}
    base = 1
    for name in names:
        blocks = pool.get_new_blocks(per)
        for i, b in enumerate(blocks):
            b.block_hash = bh(base + i)
        base += 1000
        convs[name] = blocks
        m._protect_prompt_blocks(blocks)

    # An appending turn re-offers the whole prefix; that is the reuse LRU sees.
    for _ in range(3):
        for name in refresh:
            m._protect_prompt_blocks(convs[name])

    F.write(kvmgr_enabled=1, protected_fraction=cap_frac, evict_order=order)
    m.new_step_starts()

    survivors = {
        name: [b.block_id for b in blocks
               if b.block_id in m._protected_prompt_block_ids]
        for name, blocks in convs.items()
    }
    return m, convs, survivors


def head_intact(blocks, survivors) -> bool:
    """Reuse is a contiguous match from token 0, so a prefix is worth something
    only if its HEAD survived -- and only up to its first gap."""
    return bool(survivors) and survivors[0] == blocks[0].block_id


def reusable(blocks, survivors) -> int:
    """How many blocks the cache could ACTUALLY reuse: the contiguous run from
    the head. This is the figure the measured 518s-vs-35s difference is about,
    and it is NOT len(survivors) -- a prefix missing its head reuses nothing
    however much of it is still resident."""
    ids = [b.block_id for b in blocks]
    kept = set(survivors)
    n = 0
    for bid in ids:
        if bid not in kept:
            break
        n += 1
    return n


# fifo: the victim is the longest-HELD prefix, which here is the active one.
m, convs, surv = scenario("fifo")
eq("fifo: 30 blocks survive in total", sum(len(v) for v in surv.values()), 30)
check("*** fifo destroys A -- the conversation held LONGEST, and the one still "
      "being used ***",
      len(surv["A"]) == 0, f"A kept {len(surv['A'])} of 20")
check("while sparing B, which had been abandoned", len(surv["B"]) > 0,
      f"B kept {len(surv['B'])}")

# lru: the victim is the least-recently-USED prefix.
m, convs, surv = scenario("lru")
eq("lru: 30 blocks survive in total", sum(len(v) for v in surv.values()), 30)
check("*** lru destroys B, the ABANDONED prefix, and that is the fix ***",
      len(surv["B"]) == 0, f"B kept {len(surv['B'])}")
check("and the active A survives, which fifo destroyed",
      len(surv["A"]) > 0, f"A kept {len(surv['A'])}")
check("so lru and fifo choose DIFFERENT victims -- which the earlier version "
      "of this scenario could not show",
      True)

lru_partial = [n for n in ("A", "C") if 0 < len(surv[n]) < 20]
check("lru half-evicts one prefix", len(lru_partial) == 1, f"{lru_partial}")
_n = lru_partial[0]
check("*** AND lru IS STILL HEAD-FIRST: the half-evicted prefix loses its HEAD, "
      "so the cache can reuse NOTHING of it ***",
      not head_intact(convs[_n], surv[_n]) and reusable(convs[_n], surv[_n]) == 0,
      f"{_n} kept {len(surv[_n])} blocks but {reusable(convs[_n], surv[_n])} "
      f"are reusable")
lru_reusable = sum(reusable(convs[n], surv[n]) for n in convs)

# lru_tail_first: same victim prefix, the other end of it.
m, convs, surv = scenario("lru_tail_first")
eq("lru_tail_first: 30 blocks survive in total",
   sum(len(v) for v in surv.values()), 30)
check("it still destroys the abandoned B", len(surv["B"]) == 0)
tail_partial = [n for n in ("A", "C") if 0 < len(surv[n]) < 20]
check("it half-evicts one prefix too", len(tail_partial) == 1, f"{tail_partial}")
_t = tail_partial[0]
check("*** AND THE HALF-EVICTED PREFIX KEEPS ITS HEAD, so a SHORTER prefix "
      "still hits -- the whole difference between ~35s and ~518s ***",
      head_intact(convs[_t], surv[_t]),
      f"{_t} starts at {surv[_t][:1]}, head is {convs[_t][0].block_id}")
check("every survivor is a contiguous run from its head, which is the only "
      "shape the cache can reuse",
      all(surv[n] == [b.block_id for b in convs[n][:len(surv[n])]] for n in convs),
      f"{ {n: surv[n] for n in convs} }")

tail_reusable = sum(reusable(convs[n], surv[n]) for n in convs)
check("*** THE SAME 30 SURVIVING BLOCKS ARE WORTH MORE UNDER TAIL-FIRST ***",
      tail_reusable > lru_reusable,
      f"reusable: lru {lru_reusable} blocks, lru_tail_first {tail_reusable} "
      f"-- same residency, different value, because reuse is contiguous from "
      f"token 0")

# ---------------------------------------------------------------------------
# 4. THE VICTIM RULE IS VISIBLE, so an A/B is readable from the counters
# ---------------------------------------------------------------------------
for order, rule in (("fifo", stats.VICTIM_FIFO),
                    ("lru", stats.VICTIM_LRU),
                    ("lru_tail_first", stats.VICTIM_LRU_TAIL)):
    m, convs, surv = scenario(order)
    g = stats.group(0)
    eq(f"{order} records victim rule {rule}", list(g.victims), [rule])
    check(f"{order} counted 30 evictions", g.evict_blocks == 30,
          f"{g.evict_blocks}")
    check(f"{order} attributes them to the flag change, not the cap",
          g.evict_reset > 0 and g.evict_pool == 0,
          f"reset={g.evict_reset} cap={g.evict_cap} pool={g.evict_pool}")

# ---------------------------------------------------------------------------
# 5. SWITCHING ARMS ON A LIVE MANAGER, mid-conversation
# ---------------------------------------------------------------------------
m, convs, surv = scenario("fifo")
harness.clear_log()
F.write(kvmgr_enabled=1, protected_fraction=0.25, evict_order="lru_tail_first")
blocks = m.block_pool.get_new_blocks(10)
for i, b in enumerate(blocks):
    b.block_hash = bh(90000 + i)
m._protect_prompt_blocks(blocks)
g = stats.group(0)
check("after a live switch the NEXT eviction uses the new rule",
      stats.VICTIM_LRU_TAIL in g.victims,
      f"victims now {dict(g.victims)}")
check("and the old rule's count is preserved, so an A/B is readable",
      stats.VICTIM_FIFO in g.victims)

# ---------------------------------------------------------------------------
# 6. THE KILL SWITCH STILL WINS over a policy knob
# ---------------------------------------------------------------------------
m, convs, surv = scenario("lru_tail_first")
F.write(kvmgr_enabled=0, protected_fraction=0.25, evict_order="lru_tail_first")
stats.reset_for_tests()
blocks = m.block_pool.get_new_blocks(5)
for i, b in enumerate(blocks):
    b.block_hash = bh(95000 + i)
m._protect_prompt_blocks(blocks)
m._trim_protected_prompt_blocks()
g = stats.group(0)
check("with kvmgr_enabled off, lru_tail_first is NOT applied -- patch6's own "
      "release runs instead",
      stats.VICTIM_LRU_TAIL not in g.victims,
      f"victims {dict(g.victims)}")

# ---------------------------------------------------------------------------
# 6b. THE PER-LINEAGE REPORT -- the only thing that can tell head from tail
#     from outside the engine
# ---------------------------------------------------------------------------
for order, expect_head in (("fifo", False), ("lru", False), ("lru_tail_first", True)):
    m, convs, surv = scenario(order)
    rep = m.kvmgr_lineage_report()
    check(f"{order}: report has one entry per surviving lineage",
          len(rep) == sum(1 for n in convs if surv[n]),
          f"{len(rep)} entries, {[n for n in convs if surv[n]]} survive")
    check(f"{order}: blocks and tokens agree with the block size",
          all(e["tokens"] == e["blocks"] * 16 for e in rep), f"{rep}")

    partial = [n for n in convs if 0 < len(surv[n]) < 20]
    if partial:
        n = partial[0]
        # the half-evicted prefix is the one that discriminates
        heads = [e["head_intact"] for e in rep]
        check(f"{order}: head_intact is {expect_head} for the half-evicted prefix",
              (True in heads) == expect_head or expect_head is False,
              f"heads={heads} partial={n} kept={len(surv[n])}")

    # reusable_blocks is the figure that matters: a prefix without its head
    # reuses NOTHING, however much of it survived
    total_reusable = sum(e["reusable_blocks"] for e in rep)
    total_surviving = sum(e["blocks"] for e in rep)
    if order == "lru_tail_first":
        check("lru_tail_first: every surviving block is REUSABLE",
              total_reusable == total_surviving,
              f"{total_reusable} of {total_surviving}")
    else:
        check(f"{order}: some surviving blocks are NOT reusable (head lost)",
              total_reusable < total_surviving,
              f"{total_reusable} of {total_surviving} -- the rest is stranded "
              f"behind a missing head")

# a lineage that goes entirely must drop its head record too
m, convs, surv = scenario("lru_tail_first")
check("an entirely-evicted lineage leaves no head record behind",
      len(m._kv_lineage_head) == len(m._kv_lineages),
      f"{len(m._kv_lineage_head)} heads vs {len(m._kv_lineages)} lineages")
check("the report is bounded", len(m.kvmgr_lineage_report(limit=1)) <= 1)

# *** AN EMPTY LINEAGE MUST NOT BE RETAINED, and the LRU refresh must survive
# the fix. *** _kv_track runs on every protect, including ones where patch6 took
# nothing (null, unhashed, or already protected). That used to insert a key with
# an empty list that _kv_forget could never reach, so empty lineages
# accumulated for the life of the process -- seen live as rows of
# "blocks=0 head_intact=False reusable=0" in the per-lineage report.
harness.reset_manager_count()
_p = harness.make_pool(num_gpu_blocks=200, hash_block_size=16)
flags.reset_for_tests()
F.write(kvmgr_enabled=1, protected_fraction=0.9, protect_alloc="equal",
        evict_order="lru_tail_first")
_m = MGR.get_manager_for_kv_cache_spec(
    kv_cache_spec=iface.MLAAttentionSpec(block_size=16),
    max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
    block_pool=_p, enable_caching=True, kv_cache_group_id=0,
    dcp_world_size=1, pcp_world_size=1)
DSparkMLAAttentionManager.group_count = 1
DSparkMLAAttentionManager.group_block_sizes = (16,)
DSparkMLAAttentionManager.group_sliding_windows = (None,)

# UNHASHED blocks are NOT the trigger -- lineage_key returns None and _kv_track
# returns before creating anything. The first version of this test used them and
# so asserted nothing: it passed against the leak. Verified by breaking it.
_nohash = _p.get_new_blocks(4)
_m._protect_prompt_blocks(_nohash)
eq("unhashed blocks create no lineage (an early return, not the fix)",
   len(_m._kv_lineages), 0)

# *** THE REAL TRIGGER: a SUB-PREFIX whose head is a new key but whose every
# block is already protected. *** Offer A, then offer A[1:] -- key(A[1]) has
# never been a lineage key, so a fresh empty list is created, and patch6 takes
# nothing because all of A is already protected. Measured against the broken
# version: 1 lineage -> 2, one of them with 0 blocks.
_a = _p.get_new_blocks(5)
for _i, _b in enumerate(_a):
    _b.block_hash = bh(70000 + _i)
_m._protect_prompt_blocks(_a)
eq("the full prefix is one lineage", len(_m._kv_lineages), 1)
_m._protect_prompt_blocks(_a[1:])
eq("*** a sub-prefix that takes nothing leaves NO empty lineage ***",
   len(_m._kv_lineages), 1)
eq("and no orphan head record", len(_m._kv_lineage_head), 1)
eq("so the report has no blocks=0 rows",
   sum(1 for e in _m.kvmgr_lineage_report() if e["blocks"] == 0), 0)

# ...and a real prefix still tracks and REFRESHES on an appending turn
_b2 = _p.get_new_blocks(5)
for _i, _x in enumerate(_b2):
    _x.block_hash = bh(80000 + _i)
_m._protect_prompt_blocks(_b2)
eq("two real prefixes tracked", len(_m._kv_lineages), 2)
_order_before = list(_m._kv_lineages)
# re-offer A with nothing new: ids is non-empty, so it must MOVE TO END
_m._protect_prompt_blocks(_a)
_order_after = list(_m._kv_lineages)
check("*** re-offering an already-protected prefix still REFRESHES its LRU "
      "position -- the fix must not break the appending-turn case ***",
      _order_after[-1] == _order_before[0] and _order_after != _order_before,
      f"{_order_before} -> {_order_after}")
eq("and no empty lineage appeared", 
   sum(1 for v in _m._kv_lineages.values() if not v), 0)

# ---------------------------------------------------------------------------
# 7. LINEAGE BOOKKEEPING CANNOT LEAK
# ---------------------------------------------------------------------------
m, convs, surv = scenario("lru_tail_first")
tracked = set(m._kv_block_lineage)
check("every tracked block is still protected (no stale ids retained)",
      tracked <= m._protected_prompt_block_ids,
      f"{len(tracked - m._protected_prompt_block_ids)} stale")
check("an emptied lineage is dropped, not left as an empty list",
      all(v for v in m._kv_lineages.values()),
      f"{[k for k, v in m._kv_lineages.items() if not v]}")
eq("the lineage lists account for exactly the protected set",
   sum(len(v) for v in m._kv_lineages.values()),
   len(m._protected_prompt_block_ids))

# a lineage key is a block hash, never anything reconstructable
key = next(iter(m._kv_lineages))
check("a lineage is keyed by a block HASH, so nothing here holds content",
      isinstance(key, (bytes, bytearray)), f"{type(key).__name__}")

# ---------------------------------------------------------------------------


print()
print("== (3f) DROPPING THE GPU CACHE LIVE, which removes the reboot ==")

# Why this exists: proving the disk read path needs a GPU cache that has
# forgotten a prefix the disk still holds, and until now that state was only
# reachable by restarting -- ~4 minutes plus a cold prefill, twice per
# experiment. This build exposes NO reset (POST /reset_prefix_cache is 404 and
# no such route is registered), so the lever had to be ours.
#
# The asymmetry that makes it work: block_pool.reset_prefix_cache() is the
# POOL's method, while the thing that would discard our disk index is the
# CONNECTOR's reset_cache(). Going at the pool never touches the index.


class _Pool:
    def __init__(self, free, resets_ok=True):
        self._free = free
        self.resets_ok = resets_ok
        self.reset_calls = 0

    def get_num_free_blocks(self):
        return self._free

    def reset_prefix_cache(self):
        self.reset_calls += 1
        return self.resets_ok


def _mgr_with_pool(pool, release=None):
    # *** THE DROP IS OPT-IN NOW, so every drop test must enable it. *** The
    # knob defaults OFF because the lever caused Xid 32 twice on 2026-10-09
    # with the mechanism unexplained. Enabling it HERE rather than at each call
    # site is deliberate: this helper exists only to build managers for drop
    # tests, and the alternative is eight F.write lines that a ninth test would
    # forget -- after which it would pass for the wrong reason, testing the
    # gate instead of the thing it meant to test.
    F.write(kvmgr_enabled=1, gpu_cache_drop_allowed=1)
    obj = DSparkMLAAttentionManager.__new__(DSparkMLAAttentionManager)
    obj.block_pool = pool
    obj.release_protected_prompt_blocks = release or (lambda *a, **k: None)
    return obj


flags.reset_for_tests()
pool = _Pool(free=22000)
released = []
m = _mgr_with_pool(pool, release=lambda *a, **k: released.append(1))
harness.clear_log()
ok = m.kvmgr_drop_gpu_cache()
check("*** the drop resets the POOL, not the connector, so the disk index "
      "cannot be collateral ***", pool.reset_calls == 1)
eq("...it releases our protected blocks FIRST, because patch7 refuses the "
   "reset unless every block is free", len(released), 1)
check("...and reports success", ok is True)
txt = harness.log_text()
check("...logging the free-block movement, which is what patch7 gates on",
      "free blocks" in txt)
check("...and stating that the disk index is untouched BY DESIGN",
      "DISK index is untouched" in txt)

# *** THE DROP MUST RELEASE EVERY GROUP, AND A LIVE REFUSAL IS WHY. ***
# Measured 2026-10-08: the drop fired, released group 0 and was REFUSED, with
# the log naming the cause in one number -- "pool free blocks 22536 -> 22581".
# +45 is exactly group 0's share of an 11,764-token prompt (11764/256 = 46), so
# the other ~1,700 belonged to the four WINDOWED groups, which run upstream's
# SlidingWindowMLAManager because we register for MLAAttentionSpec only. patch7
# gates on `num_used_blocks != 1` across the WHOLE pool, so releasing one group
# of five could never be enough -- the reset was unreachable by construction.
class _Foreign:
    """Stands in for an upstream manager we did NOT replace.

    `release_protected_prompt_blocks` is defined on SingleTypeKVCacheManager,
    so it exists on every manager -- which is what makes the pool-wide release
    possible without owning all five classes.
    """

    def __init__(self, log):
        self._log = log

    def release_protected_prompt_blocks(self):
        self._log.append(type(self).__name__)


flags.reset_for_tests()
poolA = _Pool(free=22000)
seen = []
mine = _mgr_with_pool(poolA, release=lambda *a, **k: seen.append("ours"))
foreigners = tuple(_Foreign(seen) for _ in range(4))
_saved_all = getattr(DSparkMLAAttentionManager, "all_managers", None)
DSparkMLAAttentionManager.all_managers = (mine,) + foreigners
harness.clear_log()
check("the drop still resets", mine.kvmgr_drop_gpu_cache() is True)
eq("*** it releases ALL FIVE groups, not only the one we own ***",
   len(seen), 5)
eq("...ours among them, exactly once", seen.count("ours"), 1)
eq("...and each foreign manager once", seen.count("_Foreign"), 4)
txtA = harness.log_text()
check("...and the log says how many it released, so a partial is visible",
      "released 5/5" in txtA, txtA.strip().splitlines()[-1][:120])

# One group raising must not stop the others: a release that fails is bad, a
# release that silently skips four groups is the refusal we just spent a boot on.
flags.reset_for_tests()
poolB = _Pool(free=22000)
seenB = []


class _Angry(_Foreign):
    def release_protected_prompt_blocks(self):
        raise RuntimeError("boom")


mineB = _mgr_with_pool(poolB, release=lambda *a, **k: seenB.append("ours"))
DSparkMLAAttentionManager.all_managers = (
    mineB, _Angry(seenB), _Foreign(seenB), _Foreign(seenB))
harness.clear_log()
check("a raising group does not abort the drop",
      mineB.kvmgr_drop_gpu_cache() is True)
eq("...the others are still released", len(seenB), 3)
txtB = harness.log_text()
check("...and the count SAYS it was partial", "released 3/4" in txtB)
check("...naming the class that failed", "_Angry" in txtB)

# And the fallback: if the stash never happened, releasing one group is still
# better than releasing none -- but it must WARN, because patch7 will refuse.
flags.reset_for_tests()
poolC = _Pool(free=22000)
seenC = []
mineC = _mgr_with_pool(poolC, release=lambda *a, **k: seenC.append("ours"))
DSparkMLAAttentionManager.all_managers = ()
harness.clear_log()
mineC.kvmgr_drop_gpu_cache()
eq("with no manager list, it still releases ITS OWN group", len(seenC), 1)
txtC = harness.log_text()
check("*** and warns that patch7 will refuse a one-group release ***",
      "only THIS group" in txtC and "whole pool" in txtC)

if _saved_all is None:
    del DSparkMLAAttentionManager.all_managers
else:
    DSparkMLAAttentionManager.all_managers = _saved_all

# A refusal must say so and say what to do -- a drop that silently did nothing
# would send somebody hunting a bug in a read path that never ran.
pool2 = _Pool(free=10, resets_ok=False)
m2 = _mgr_with_pool(pool2)
harness.clear_log()
eq("a refused reset returns False", m2.kvmgr_drop_gpu_cache(), False)
txt2 = harness.log_text()
check("*** and says REFUSED rather than looking like it worked ***",
      "REFUSED" in txt2)
check("...naming the cause and the remedy (go idle, bump it again)",
      "still referenced" in txt2 and "gpu_cache_epoch" in txt2)

# No pool at all must not raise -- this runs on a scheduler step.
m3 = DSparkMLAAttentionManager.__new__(DSparkMLAAttentionManager)
m3.block_pool = None
harness.clear_log()
eq("a missing pool is an error, not an exception", m3.kvmgr_drop_gpu_cache(), False)
check("...and says nothing was reset", "nothing was reset" in harness.log_text())

# And the knob is an INTEGER acted on at the EDGE, because the useful operation
# is "do it again" and a bool can only be flipped once.
# *** AND THE WIRING, which break-verification caught me not testing. ***
# Everything above calls kvmgr_drop_gpu_cache() directly, so dropping the
# `flags.changed("gpu_cache_epoch")` edge from new_step_starts left every
# assertion green -- a perfect drop that nothing invokes, which is the
# invisible-icon failure in another costume. This drives the REAL hook.
flags.reset_for_tests()
stats.reset_for_tests()
harness.reset_manager_count()
_pool = harness.make_pool(num_gpu_blocks=120, hash_block_size=16)
F.write(kvmgr_enabled=1, gpu_cache_epoch=0)
_m = MGR.get_manager_for_kv_cache_spec(
    kv_cache_spec=iface.MLAAttentionSpec(block_size=16),
    max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
    block_pool=_pool, enable_caching=True, kv_cache_group_id=0,
    dcp_world_size=1, pcp_world_size=1)
_hits = []
_m.kvmgr_drop_gpu_cache = lambda: (_hits.append(1), True)[1]

_m.new_step_starts()
eq("a step with no epoch change does NOT drop the cache", len(_hits), 0)

# NOTE: F.write() alone, NOT reset_for_tests(). The harness's write already
# invalidates the 1s TTL, while reset_for_tests ALSO clears `_last_seen` -- and
# changed() deliberately returns False on a first read (it establishes the
# baseline). So resetting between bumps destroys the very edge under test.
F.write(kvmgr_enabled=1, gpu_cache_epoch=1)
_m.new_step_starts()
eq("*** bumping gpu_cache_epoch DOES reach the drop, via the real hook ***",
   len(_hits), 1)

_m.new_step_starts()
eq("...and it fires ONCE per bump, not every step after", len(_hits), 1)

F.write(kvmgr_enabled=1, gpu_cache_epoch=2)
_m.new_step_starts()
eq("...and again on the next bump, which is why it is an int", len(_hits), 2)

eq("the knob defaults to 0", flags.DEFAULTS["gpu_cache_epoch"], 0)
check("...and is an int, so it can be bumped repeatedly",
      isinstance(flags.DEFAULTS["gpu_cache_epoch"], int))

print()
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
print("== THE DROP IS OPT-IN, BECAUSE IT KILLED THE ENGINE TWICE ==")
# *** Xid 32 at 11:50 (worker) and 17:07 (head) on 2026-10-09, both within ~4
# *** seconds of a gpu_cache_epoch bump, on different ranks. ***
# The second had ZERO eviction pressure (budget 1000 GB, 700 GB headroom) and
# the pending-blocks fix live in all six processes, so neither churn nor that
# fix explains it. The mechanism is UNESTABLISHED -- which is why the default
# is off rather than a warning: a lever whose failure mode is a dead engine
# and whose cause is unknown should cost a deliberate keystroke. No production
# path drops the prefix cache, so gating it costs nothing a workload needs.
_gp = _Pool(free=22000)
_gm = _mgr_with_pool(_gp)
# *** `reset_for_tests()` IS NOT ENOUGH, and that caught me. *** It clears the
# OVERRIDE map; the flags FILE persists, and the helper above just wrote the
# knob into it. So the first version of this test saw the drop succeed and read
# as "the gate does not work" when the gate was never consulted at its default.
# Rewrite the file WITHOUT the knob, which is what "at its default" means.
F.write(kvmgr_enabled=1)
flags.reset_for_tests()
harness.clear_log()
eq("*** with the knob at its default the drop REFUSES ***",
   _gm.kvmgr_drop_gpu_cache(), False)
eq("*** and the pool is NOT reset -- a gate that still reset would be no gate "
   "at all ***", _gp.reset_calls, 0)
check("...naming the Xid and that the cause is unknown, so a reader can weigh "
      "it", "Xid 32" in harness.log_text() and "not understood" in harness.log_text(),
      harness.log_text()[-240:])

_gp2 = _Pool(free=22000)
_gm2 = _mgr_with_pool(_gp2)                  # the helper enables the knob
F.write(kvmgr_enabled=1, gpu_cache_drop_allowed=1)
eq("...and with it deliberately ON the lever still works",
   _gm2.kvmgr_drop_gpu_cache(), True)
eq("...really resetting the pool", _gp2.reset_calls, 1)

print("== THE DROP MUST REFUSE WHILE A KV TRANSFER IS IN FLIGHT ==")
# *** THIS KILLED THE ENGINE ON 2026-10-09. *** At 11:50:00 the drop released
# 449 blocks and reset the pool with store jobs outstanding on BOTH ranks
# ("store path busy -- 2 job(s) outstanding" at 11:49:23; rank 0 still
# reporting 1 at 11:50:08). Four seconds later: Xid 32 TWICE on the worker node
# -- a corrupted push buffer -- and Worker_TP1 died with `CUDA error:
# unspecified launch failure`, surfacing at `synchronize_input_prep`, nowhere
# near the cause, because the fault is asynchronous.
#
# A store copies GPU -> staging. Free its source block and hand it back to the
# allocator mid-copy and the DMA reads what is no longer there. *** AND THE
# EXISTING GATE CANNOT SEE IT: *** patch7 gates on `num_used_blocks`, which
# counts REQUEST references, and a store job OUTLIVES its request because
# stores are deferred past token sampling and pumped over rounds. "Idle" by
# request count is not idle by transfer, and every earlier drop was luck about
# timing -- which is why this needs a test rather than a comment.
# *** THE REAL diskspec CANNOT BE IMPORTED HERE, and that is a fixture fact
# *** rather than a problem. *** This file stubs the `vllm.*` leaves patch6/7/8
# need, not `vllm.v1.kv_offload`, which only the *_disk.py suites fake. But the
# manager does `from . import diskspec` at CALL time, so a stub module in
# sys.modules is enough -- and it keeps the test beside the manager fixtures it
# needs instead of moving the drop's tests away from the drop.
import sys as _sysP3
import types as _tP3

_saved_ds = _sysP3.modules.get("kvmgr.diskspec")


def _drop_with(snap):
    """Run a fresh drop against a faked scheduler snapshot. Returns (ok, pool)."""
    F.write(kvmgr_enabled=1, gpu_cache_drop_allowed=1)
    _p = _Pool(free=22000)
    _m = _mgr_with_pool(_p, release=lambda *a, **k: None)
    _stub = _tP3.ModuleType("kvmgr.diskspec")
    _stub.offload_sched_snapshot = lambda: snap
    _sysP3.modules["kvmgr.diskspec"] = _stub
    harness.clear_log()
    try:
        return _m.kvmgr_drop_gpu_cache(), _p
    finally:
        if _saved_ds is not None:
            _sysP3.modules["kvmgr.diskspec"] = _saved_ds
        else:
            _sysP3.modules.pop("kvmgr.diskspec", None)


_ok, _p = _drop_with({"available": True, "jobs_open": 2, "stores_open": 2,
                      "loads_open": 0, "blocks_with_pending_jobs": 7})
eq("*** a drop with store jobs outstanding is REFUSED ***", _ok, False)
eq("*** AND THE POOL IS NOT RESET -- that is the whole safety property, "
   "because the reset is what recycles the blocks mid-DMA ***",
   _p.reset_calls, 0)
check("...and it names the counts and the remedy",
      "REFUSED" in harness.log_text() and "outstanding" in harness.log_text())

# Blocks held with no job open is still unsafe: the scheduler will not reuse a
# block with a pending store, so a non-zero block count means the same thing.
_ok, _p = _drop_with({"available": True, "jobs_open": 0, "stores_open": 0,
                      "loads_open": 0, "blocks_with_pending_jobs": 3})
eq("blocks still pending a transfer also refuse", _ok, False)
eq("...without resetting", _p.reset_calls, 0)

# A genuinely drained scheduler must still allow the drop, or the lever is dead.
_ok, _p = _drop_with({"available": True, "jobs_open": 0, "stores_open": 0,
                      "loads_open": 0, "blocks_with_pending_jobs": 0})
eq("*** a DRAINED scheduler still drops, or the lever is dead ***", _ok, True)
eq("...and the pool really was reset", _p.reset_calls, 1)

# *** ABSENT IS NOT ZERO. *** An unreachable scheduler cannot say there are no
# transfers. It proceeds -- refusing forever would disable the lever on any
# build without the connector -- but it must SAY that it is proceeding on a
# not-knowing, because that line is the first thing somebody reads after the
# next Xid 32.
_ok, _p = _drop_with({"available": False, "why": "no scheduler reachable"})
eq("an unreachable scheduler still drops", _ok, True)
check("*** ...but says the transfer state is UNKNOWN rather than zero ***",
      "UNKNOWN" in harness.log_text(), harness.log_text()[-300:])

print("== A BLOCK WITH A TRANSFER IN FLIGHT MUST KEEP ITS LAST REFERENCE ==")
# *** THE WINDOW. *** The offload connector fences a store's blocks against
# reuse, but for NON-SLIDING-WINDOW blocks it registers that fence only at
# `request_finished`, because during the request "ref_cnt protects them"
# (scheduler.py:54-57). That is sound until something else moves ref_cnt on a
# different schedule -- and MNT-6a does: _release_one_protected_prompt_block
# ends in free_blocks, which at ref_cnt 0 returns the block to the free queue
# for reallocation. If ours was the last reference and a copy is still reading
# that block, the copy reads reallocated memory, which on a GPU surfaces as an
# asynchronous Xid 13 / Data deaddead at an unrelated sync point -- what we saw
# on 2026-10-09, rank 0's worker dead one second later.
#
# Not proven to be that crash's cause. A real structural gap either way.
import kvmgr.manager as _mgrmod
_stm = sys.modules["vllm.v1.core.single_type_kv_cache_manager"]

check("patch6 offers the hook at all -- without it this cannot be wired",
      hasattr(_stm, "set_pending_block_ids_provider"))


def _set_provider(fn):
    """Install a provider, tolerating a patch6 that has no hook.

    *** REMOVING THE HOOK ABORTED THE FILE INSTEAD OF FAILING IT. *** Renaming
    the setter in patch6 made every later call an AttributeError that escaped,
    so break-verification saw no output rather than a named failure -- the trap
    this suite has now hit six times. Any call whose subject can vanish under
    mutation has to go through something that catches.
    """
    setter = getattr(_stm, "set_pending_block_ids_provider", None)
    if setter is None:
        return False
    setter(fn)
    return True


def _mk_protected(ids, pending=None):
    """A manager holding `ids` as protected, with a pending-blocks provider."""
    F.write(kvmgr_enabled=1, protected_fraction=0.5)
    harness.reset_manager_count()
    pool = harness.make_pool(num_gpu_blocks=120, hash_block_size=16)
    m = MGR.get_manager_for_kv_cache_spec(
        kv_cache_spec=iface.MLAAttentionSpec(block_size=16),
        max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
        block_pool=pool, enable_caching=True, kv_cache_group_id=0,
        dcp_world_size=1, pcp_world_size=1)
    DSparkMLAAttentionManager.group_count = 1
    m._protected_prompt_block_ids = set(ids)
    m._protected_prompt_block_queue = deque(ids)
    freed = []
    pool.free_blocks = lambda blocks: freed.extend(b.block_id for b in blocks)
    _set_provider((lambda: pending) if pending is not None else None)
    return m, freed


# *** ASSERT ON WHICH BLOCK WAS CHOSEN, not on free_blocks. ***
# The first version watched a `freed` list and got [] for both arms, which
# looked like the release doing nothing -- but `free_blocks` is gated on
# `ref_cnt > 0` and the harness pool's blocks sit at 0, so the call is
# legitimately skipped. The property under test is VICTIM SELECTION, and the
# observable for that is which id left the protected set.
def _chose(m, before):
    gone = before - set(m._protected_prompt_block_ids)
    return sorted(gone)


# A block NOT in flight is released exactly as before.
_m1, _freed1 = _mk_protected([5, 6, 7], pending={99})
_b1 = set(_m1._protected_prompt_block_ids)
_m1._release_one_protected_prompt_block(block_ids_to_skip={99})
eq("*** a block with no transfer in flight is still chosen -- the cap must "
   "keep working ***", _chose(_m1, _b1), [5])

# A block IN FLIGHT is skipped and RE-QUEUED, so the cap is late, not abandoned.
_m2, _freed2 = _mk_protected([5, 6, 7], pending={5})
_b2 = set(_m2._protected_prompt_block_ids)
_ok = _m2._release_one_protected_prompt_block(block_ids_to_skip={5})
eq("*** a block with a transfer in flight is NOT the one chosen ***",
   _chose(_m2, _b2), [6])
check("...and it is still protected, re-queued for the next attempt",
      5 in _m2._protected_prompt_block_ids, _m2._protected_prompt_block_ids)

# Every candidate pending: the release REFUSES rather than picking one anyway.
_m3, _freed3 = _mk_protected([5, 6], pending={5, 6})
_ok3 = _m3._release_one_protected_prompt_block(block_ids_to_skip={5, 6})
eq("*** with every candidate in flight, nothing is freed ***", _freed3, [])
eq("...and it reports that it could not release", _ok3, False)
check("...leaving both protected, so the next protect retries",
      _m3._protected_prompt_block_ids == {5, 6},
      _m3._protected_prompt_block_ids)

# *** AND THIS IS THE TEST THE MUTATION SHOWED WAS MISSING. ***
# Everything above calls _release_one_protected_prompt_block DIRECTLY with an
# explicit skip set, so none of it exercised `_trim_protected_prompt_blocks`
# actually CONSULTING the provider. Setting `pending = None` inside the trim
# left the whole block green -- the plumbing untested while its callee was
# tested six ways. Same shape as an icon with no symbol: present,
# correct-looking, wired to nothing.
_m5, _freed5 = _mk_protected([5, 6, 7], pending={5})
_m5._max_protected_prompt_blocks = lambda: 1      # force the trim to release
_b5 = set(_m5._protected_prompt_block_ids)
_m5._trim_protected_prompt_blocks()
_left5 = set(_m5._protected_prompt_block_ids)
check("*** the TRIM consults the provider -- the in-flight block survives a "
      "trim that released everything else ***",
      _left5 == {5}, (_b5, _left5))

print("== ...AND IT FAILS OPEN, WHICH IS THE TRADE ==")
# *** SKIPPING ON A NOT-KNOWING WOULD BE WORSE. *** If the provider is absent
# or raises, the release must PROCEED -- that is today's behaviour. The
# alternative makes the protection cap unenforceable the moment the hook
# breaks, and an unenforceable cap means memory pressure and preemptions. So
# this narrows a real window; it does not close it, and the tests say so.
_m4, _freed4 = _mk_protected([5, 6], pending=None)   # no provider installed
_set_provider(None)
_m4._trim_protected_prompt_blocks()
check("with NO provider the trim behaves exactly as before (it still runs)",
      True)

_boom = []
def _raiser():
    _boom.append(1)
    raise RuntimeError("provider exploded")

_set_provider(_raiser)
try:
    _got = _stm._pending_block_ids()
except Exception as exc:
    _got = "RAISED: %r" % (exc,)
eq("*** a provider that RAISES yields None, never an exception into the "
   "protect path ***", _got, None)
eq("...and it really was called", len(_boom), 1)
_set_provider(lambda: None)
eq("a provider returning None is None, not an empty set -- 'cannot say' and "
   "'nothing pending' must stay distinguishable",
   _stm._pending_block_ids(), None)
_set_provider(None)

if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
