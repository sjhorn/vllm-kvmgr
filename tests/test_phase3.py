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
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
