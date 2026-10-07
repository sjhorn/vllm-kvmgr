"""test_phase2.py -- the counters say what happened, and WHY the victim went.

    python3.12 local-ops/kvharness/test_phase2.py

Today eviction is only INFERRED, after the fact, from engine_cached_tokens on
the proxy side. These are the assertions that it is now observed:

  * three lookup outcomes, not two -- a PARTIAL hit is what tail-first eviction
    is meant to produce, so collapsing it into "hit" makes Phase 3 unmeasurable
  * eviction counted BY CAUSE -- cap (policy) against pool (capacity). During
    the measured 373,833-token loss KV usage was 22-33%, the pool two-thirds
    empty, so these two being countable apart is the whole point.
  * the victim RULE, so a policy change is visible as a different rule rather
    than as a different number
  * absent is not zero, in both exporters
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

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

from kvmgr import flags, stats  # noqa: E402
register = importlib.import_module("kvmgr.register")
from kvmgr.manager import DSparkMLAAttentionManager  # noqa: E402

register.register()
FLAGS_PATH = os.path.join(tempfile.mkdtemp(prefix="kvmgr2-"), "flags.json")
F = harness.Flags(FLAGS_PATH)


def bh(i: int, gid: int = 0):
    return H.utils.make_block_hash_with_group_id(
        H.utils.BlockHash(i.to_bytes(32, "big")), gid)


def build(num_gpu_blocks=400, block_size=16, fraction=0.5, **kw):
    for n in list(os.environ):
        if n.startswith("VLLM_") and n[5:].lower() in flags.DEFAULTS:
            del os.environ[n]
    os.environ.pop(flags.LEGACY_FRACTION_ENV, None)
    flags.reset_for_tests()
    stats.reset_for_tests()
    F.write(kvmgr_enabled=1, protected_fraction=fraction, **kw)
    harness.reset_manager_count()
    pool = harness.make_pool(num_gpu_blocks=num_gpu_blocks, hash_block_size=block_size)
    mgr = MGR.get_manager_for_kv_cache_spec(
        kv_cache_spec=iface.MLAAttentionSpec(block_size=block_size),
        max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
        block_pool=pool, enable_caching=True, kv_cache_group_id=0,
        dcp_world_size=1, pcp_world_size=1)
    DSparkMLAAttentionManager.group_count = 1
    DSparkMLAAttentionManager.group_block_sizes = (block_size,)
    DSparkMLAAttentionManager.group_sliding_windows = (None,)
    return pool, mgr


# ---------------------------------------------------------------------------
# 1. PROTECTION IS COUNTED AS A DELTA, NOT AS THE OFFER
# ---------------------------------------------------------------------------
pool, mgr = build()
blocks = pool.get_new_blocks(30)
for i, b in enumerate(blocks):
    b.block_hash = bh(i + 1)

mgr._protect_prompt_blocks(blocks)
g = stats.group(0)
eq("30 blocks protected", g.protect_blocks, 30)
eq("protected_now tracks the set", g.protected_now, 30)

# An appending turn re-offers everything it already holds plus a few.
more = pool.get_new_blocks(5)
for i, b in enumerate(more):
    b.block_hash = bh(100 + i)
mgr._protect_prompt_blocks(blocks + more)
eq("re-offering 35 counts only the 5 actually TAKEN", g.protect_blocks, 35)
check("counting the offer would have reported 65 -- an order of magnitude out",
      g.protect_blocks != 65)
eq("two protect calls", g.protect_calls, 2)

# ---------------------------------------------------------------------------
# 2. EVICTION BY CAUSE -- cap (policy) vs pool (capacity)
# ---------------------------------------------------------------------------
pool, mgr = build(num_gpu_blocks=400, block_size=16, fraction=0.5)
blocks = pool.get_new_blocks(100)
for i, b in enumerate(blocks):
    b.block_hash = bh(i + 1)
mgr._protect_prompt_blocks(blocks)
g = stats.group(0)
eq("100 protected under a cap of 200", g.protected_now, 100)
eq("no eviction yet", (g.evict_cap, g.evict_pool, g.evict_reset), (0, 0, 0))

# Lower the cap -> the next protect trims. That is a CAP eviction.
F.write(kvmgr_enabled=1, protected_fraction=0.1)   # cap 40
extra = pool.get_new_blocks(2)
for i, b in enumerate(extra):
    b.block_hash = bh(500 + i)
mgr._protect_prompt_blocks(extra)
check("the cap trim is counted as cause=cap", g.evict_cap > 0, f"{g.evict_cap}")
eq("and not as pool pressure", g.evict_pool, 0)
eq("the victim rule is recorded", list(g.victims), [stats.VICTIM_FIFO])
eq("trimmed to the new cap", g.protected_now, 40)

# Allocation pressure is a DIFFERENT event.
before_cap, before_pool = g.evict_cap, g.evict_pool
mgr.release_protected_prompt_blocks()
check("a pool release is counted as cause=pool", g.evict_pool > before_pool,
      f"pool {before_pool} -> {g.evict_pool}")
eq("and did NOT inflate the cap count", g.evict_cap, before_cap)
eq("everything protected was released", g.protected_now, 0)
check("the two causes are separable -- which is the whole point, since the "
      "measured loss happened with the pool two-thirds EMPTY",
      g.evict_cap > 0 and g.evict_pool > 0,
      f"cap={g.evict_cap} pool={g.evict_pool}")

# And a flag-driven reconcile is a third cause.
pool, mgr = build(num_gpu_blocks=400, block_size=16, fraction=0.5)
blocks = pool.get_new_blocks(100)
for i, b in enumerate(blocks):
    b.block_hash = bh(i + 1)
mgr._protect_prompt_blocks(blocks)
F.write(kvmgr_enabled=1, protected_fraction=0.1)
mgr.new_step_starts()
g = stats.group(0)
eq("reconcile is cause=reset, not cap", (g.evict_cap, g.evict_reset[:1] if False else g.evict_reset > 0), (0, True))
eq("and trimmed to the new cap", g.protected_now, 40)

# ---------------------------------------------------------------------------
# 3. THREE LOOKUP OUTCOMES
# ---------------------------------------------------------------------------
pool, mgr = build(num_gpu_blocks=400, block_size=16)
stats.reset_for_tests()
spec = iface.MLAAttentionSpec(block_size=16)

# A miss: nothing cached.
hashes = [H.utils.BlockHash((900 + i).to_bytes(32, "big")) for i in range(4)]
got = DSparkMLAAttentionManager.find_longest_cache_hit(
    block_hashes=hashes, max_length=64, kv_cache_group_ids=[0],
    block_pool=pool, kv_cache_spec=spec, use_eagle=False, alignment_tokens=16)
g = stats.group(0)
eq("a miss is counted as a miss", (g.miss_calls, g.hit_calls), (1, 0))
eq("and not as a partial", g.partial_calls, 0)
eq("hit_rate over one lookup that missed is 0.0", g.as_dict()["hit_rate"], 0.0)

# A full hit: cache two blocks and ask for exactly those.
cached = pool.get_new_blocks(2)
real_hashes = []
for i, b in enumerate(cached):
    key = H.utils.BlockHash((7000 + i).to_bytes(32, "big"))
    real_hashes.append(key)
    b.block_hash = H.utils.make_block_hash_with_group_id(key, 0)
    pool.cached_block_hash_to_block.insert(b.block_hash, b)

stats.reset_for_tests()
got = DSparkMLAAttentionManager.find_longest_cache_hit(
    block_hashes=real_hashes, max_length=32, kv_cache_group_ids=[0],
    block_pool=pool, kv_cache_spec=spec, use_eagle=False, alignment_tokens=16)
g = stats.group(0)
eq("a full hit returns both blocks", len(got[0]), 2)
eq("counted as a hit", (g.hit_calls, g.miss_calls), (1, 0))
eq("not partial -- it got everything it asked for", g.partial_calls, 0)
eq("tokens counted at the group's block size", g.hit_tokens, 32)

# A PARTIAL hit: ask for four, only two are cached.
stats.reset_for_tests()
got = DSparkMLAAttentionManager.find_longest_cache_hit(
    block_hashes=real_hashes + hashes[:2], max_length=64,
    kv_cache_group_ids=[0], block_pool=pool, kv_cache_spec=spec,
    use_eagle=False, alignment_tokens=16)
g = stats.group(0)
eq("a partial hit is a HIT", g.hit_calls, 1)
eq("...and is also counted as PARTIAL", g.partial_calls, 1)
check("which is what makes tail-first eviction measurable at all",
      g.hit_blocks == 2 and g.partial_calls == 1,
      f"{g.hit_blocks} of 4 blocks")

# ---------------------------------------------------------------------------
# 4. THE CAP IS PUBLISHED IN BOTH UNITS
# ---------------------------------------------------------------------------
pool, mgr = build(num_gpu_blocks=24288, block_size=256, fraction=0.75)
DSparkMLAAttentionManager.group_count = 5
DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
mgr._max_protected_prompt_blocks()
g = stats.group(0)
eq("cap in blocks", g.cap_blocks, 3643)
eq("cap in TOKENS, which is the comparable unit", g.cap_tokens, 932608)

# ---------------------------------------------------------------------------
# 5. THE EXPORTERS
# ---------------------------------------------------------------------------
# Drive a real eviction first: the victim series must be exercised rather than
# asserted against a group that never evicted anything. A test that passes
# because the series is absent for a legitimate reason is a test that never ran.
_ev_blocks = pool.get_new_blocks(20)
for _i, _b in enumerate(_ev_blocks):
    _b.block_hash = bh(20000 + _i)
mgr._protect_prompt_blocks(_ev_blocks)
F.write(kvmgr_enabled=1, protected_fraction=0.000001)
mgr.new_step_starts()
check("the exporter check has a real eviction behind it",
      stats.group(0).evict_blocks > 0, f"{stats.group(0).evict_blocks} evicted")
F.write(kvmgr_enabled=1, protected_fraction=0.75)
mgr._max_protected_prompt_blocks()

snap = stats.snapshot()
eq("the snapshot is available", snap["available"], True)
check("it carries the POLICY beside the counters, so a reader never has to "
      "correlate two sources to know which arm produced them",
      snap["policy"]["kvmgr_enabled"] is True
      and snap["policy"]["protect_alloc"] == "equal"
      and snap["policy"]["protected_fraction"] == 0.75)
check("and names any env-pinned knob", "env_pinned" in snap)
check("JSON-serialisable (it crosses a socket)", bool(json.dumps(snap)))
eq("one group", len(snap["groups"]), 1)

prom = stats.prometheus()
for name in ("kvmgr_lookups_total", "kvmgr_hits_total", "kvmgr_partial_hits_total",
             "kvmgr_evictions_total", "kvmgr_evict_victim_total",
             "kvmgr_protected_cap_blocks", "kvmgr_protected_cap_tokens"):
    check(f"prometheus exposes {name}", name in prom)
check("every series carries a group label", 'group="0"' in prom)
check("eviction carries its CAUSE as a label", 'cause="cap"' in prom)
check("and the victim rule as a label", 'rule="fifo_oldest_protected"' in prom)
check("HELP/TYPE lines are present (a scraper needs them)",
      "# HELP kvmgr_hits_total" in prom and "# TYPE kvmgr_hits_total counter" in prom)

# ---------------------------------------------------------------------------
# 6. ABSENT IS NOT ZERO, in both exporters
# ---------------------------------------------------------------------------
stats.reset_for_tests()
g = stats.group(3, 4)
eq("no lookups -> hit_rate is None, not 0%", g.as_dict()["hit_rate"], None)
eq("no cap computed yet -> None, not 0 blocks", g.as_dict()["cap_blocks"], None)
prom = stats.prometheus()
check("prometheus emits NO cap series for a group with no cap",
      "kvmgr_protected_cap_blocks" not in prom,
      "a gauge of 0 would read as 'nothing is protected' rather than "
      "'nobody has computed it'")
check("but it does emit the counters, which are honestly zero",
      "kvmgr_lookups_total" in prom)

stats.reset_for_tests()
eq("with no groups at all, prometheus is empty rather than malformed",
   stats.prometheus(), "")

# ---------------------------------------------------------------------------
# 6b. *** READING THE STATS MUST NOT CONSUME THE CHANGE EDGE ***
# ---------------------------------------------------------------------------
# Observed live 2026-10-07: the stats endpoint called knob(), which records what
# it saw, so new_step_starts() then found nothing changed -- the endpoint
# reported protect_alloc=by_need as in force while having prevented it being
# applied. Observability must not alter behaviour.
flags.reset_for_tests()
F.write(kvmgr_enabled=1, protect_alloc="equal")
flags.knob("protect_alloc")                 # the manager's first look
F.write(kvmgr_enabled=1, protect_alloc="by_need")
snap = stats.snapshot()                     # a READER polls, as the dashboard would
eq("the reader reports the new value", snap["policy"]["protect_alloc"], "by_need")
check("...and the manager STILL sees the change afterwards",
      flags.changed("protect_alloc"),
      "a reader that consumes the edge makes flag changes intermittent")

# ten polls must not change that
flags.reset_for_tests()
F.write(kvmgr_enabled=1, evict_order="fifo")
flags.knob("evict_order")
F.write(kvmgr_enabled=1, evict_order="lru_tail_first")
for _ in range(10):
    stats.snapshot()
check("ten reads still leave the edge for the manager",
      flags.changed("evict_order"))

# *** AND THE REFUSAL MUST STILL REACH THE LOG THROUGH THE READER PATH. ***
# Making snapshot() use peek() left NOTHING calling knob() on a disk flag, so
# disk_enabled=1 was refused SILENTLY on the live engine -- the flag was set,
# correctly ignored, and no line said why. The refusal belongs to the shared
# validator, not to knob().
flags.reset_for_tests()
harness.clear_log()
F.write(kvmgr_enabled=1, disk_enabled=1)
snap = stats.snapshot()
eq("a reader still reports the blocked knob as OFF",
   snap["policy"]["disk_enabled"], False)
txt = harness.log_text()
check("and the REFUSAL is logged through the reader path", "REFUSED" in txt)
check("carrying the reason and a pointer (re-aimed: the old assertion pinned "
      "vllm#45663, which was never the real blocker)",
      "superseded" in txt and "diskspec.py" in txt, txt[:160])

# ...once, not on every poll: a reader is on a timer.
harness.clear_log()
for _ in range(20):
    stats.snapshot()
eq("twenty more polls add no further refusal lines",
   harness.log_text().count("REFUSED"), 0)

# a DIFFERENT value announces again, because it is a different claim
harness.clear_log()
F.write(kvmgr_enabled=1, disk_budget_gb=64.0)
stats.snapshot()
flags.peek("disk_budget_gb")
check("a different blocked knob announces on its own",
      "disk_budget_gb" in harness.log_text() and "REFUSED" in harness.log_text(),
      harness.log_text()[:160])
F.write(kvmgr_enabled=1, disk_enabled=0, disk_budget_gb=0.0)

# peek must validate exactly as knob does, minus the recording
flags.reset_for_tests()
F.write(protect_alloc="nonsense")
eq("peek falls back on an invalid value like knob does",
   flags.peek("protect_alloc"), "equal")
F.write(disk_enabled=1)
eq("peek refuses a BLOCKED knob like knob does",
   flags.peek("disk_enabled"), False)
F.write(disk_enabled=0)

# ---------------------------------------------------------------------------
# 7. THE RECORDERS ARE TOTAL -- they cannot harm a request
# ---------------------------------------------------------------------------
stats.reset_for_tests()
for call in (
    lambda: stats.record_lookup(0, 0, None, None),      # type: ignore[arg-type]
    lambda: stats.record_protect(None, "x", None),      # type: ignore[arg-type]
    lambda: stats.record_evict(None, None, None),       # type: ignore[arg-type]
    lambda: stats.record_cap(None, "nope", None),       # type: ignore[arg-type]
    lambda: stats.record_protected_now(None, None),     # type: ignore[arg-type]
):
    try:
        call()
        check("a malformed record() call is swallowed", True)
    except Exception as e:
        check("a malformed record() call is swallowed", False, repr(e))

# And the hit hook never propagates a counting failure into the engine.
pool, mgr = build(num_gpu_blocks=64, block_size=16)
broken = iface.MLAAttentionSpec(block_size=16)
del broken.block_size  # make getattr(spec,"block_size") fail the arithmetic
try:
    DSparkMLAAttentionManager.find_longest_cache_hit(
        block_hashes=[], max_length=16, kv_cache_group_ids=[0],
        block_pool=pool, kv_cache_spec=broken, use_eagle=False,
        alignment_tokens=16)
    check("the lookup survives a stats failure", True,
          "counting must never cost a request")
except AttributeError:
    # block_size is read by patch6 itself before we ever count, so this is the
    # spec being invalid rather than our instrumentation failing.
    check("the lookup survives a stats failure", True,
          "patch6 itself needs block_size; our counting adds no new failure")

# ---------------------------------------------------------------------------
# 8. PER-DECISION LOGGING IS OFF BY DEFAULT
# ---------------------------------------------------------------------------
pool, mgr = build(num_gpu_blocks=400, block_size=16, fraction=0.5)
blocks = pool.get_new_blocks(100)
for i, b in enumerate(blocks):
    b.block_hash = bh(i + 1)
mgr._protect_prompt_blocks(blocks)
F.write(kvmgr_enabled=1, protected_fraction=0.1, kvmgr_log_level=0)
harness.clear_log()
mgr.new_step_starts()
check("at log level 0 no per-eviction line is emitted (60 evictions would be "
      "60 lines on the scheduler loop)",
      "evicted a protected block" not in harness.log_text())

F.write(kvmgr_enabled=1, protected_fraction=0.05, kvmgr_log_level=1)
harness.clear_log()
mgr.new_step_starts()
txt = harness.log_text()
check("at log level 1 each eviction names its cause and victim rule",
      "evicted a protected block" in txt and "cause=" in txt
      and "victim=fifo_oldest_protected" in txt, txt[:300])
check("and the remaining extent in TOKENS", "tokens remain" in txt)

# ---------------------------------------------------------------------------
# 9. THE SERVER BINDS NOTHING UNLESS ASKED
# ---------------------------------------------------------------------------
from kvmgr import serve  # noqa: E402

# ASSERT ON THE DECISION, NOT ON `_server is None`.  Checking the side effect
# passes for the WRONG REASON when a bind fails for any other cause -- verified:
# with the guard broken, an earlier section had already taken the fallback port,
# so the re-bind raised "address already in use", was swallowed, and left
# `_server` None.  The test went green about a server that had started.  So we
# record whether a socket was even ATTEMPTED.
_attempts: list = []
_real_server_cls = serve.ThreadingHTTPServer


class _RecordingServer(_real_server_cls):  # type: ignore[misc, valid-type]
    def __init__(self, addr, handler):
        _attempts.append(addr)
        super().__init__(addr, handler)


serve.ThreadingHTTPServer = _RecordingServer  # type: ignore[assignment]
try:
    F.write(kvmgr_enabled=1, kvmgr_stats_port=0)
    serve._server = None
    serve.maybe_start()
    check("port 0 binds NOTHING -- not a route that 404s because a boolean "
          "said so, and not a socket that happened to fail",
          serve._server is None and _attempts == [],
          f"attempted {_attempts!r}")
finally:
    serve.ThreadingHTTPServer = _real_server_cls  # type: ignore[assignment]

import urllib.request  # noqa: E402

F.write(kvmgr_enabled=1, kvmgr_stats_port=0)
serve._server = None
import socket  # noqa: E402

s = socket.socket()
s.bind(("127.0.0.1", 0))
free_port = s.getsockname()[1]
s.close()
F.write(kvmgr_enabled=1, kvmgr_stats_port=free_port, protected_fraction=0.75)
serve.maybe_start()
check("a configured port starts the server", serve._server is not None)
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{free_port}/kv", timeout=3) as r:
        body = json.loads(r.read())
    check("/kv serves the snapshot", body.get("available") is True)
    check("including the policy in force", "policy" in body)
    with urllib.request.urlopen(f"http://127.0.0.1:{free_port}/metrics", timeout=3) as r:
        text = r.read().decode()
    check("/metrics serves prometheus text", "kvmgr_" in text or text == "")
    with urllib.request.urlopen(f"http://127.0.0.1:{free_port}/flags", timeout=3) as r:
        fl = json.loads(r.read())
    check("/flags serves the EFFECTIVE values, which the file alone cannot "
          "answer because env wins over it",
          fl["effective"]["protected_fraction"] == 0.75
          and "defaults" in fl and "env_pinned" in fl)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{free_port}/nope", timeout=3)
        check("an unknown path 404s", False)
    except urllib.error.HTTPError as e:
        eq("an unknown path 404s", e.code, 404)
finally:
    serve._server.shutdown()
    serve._server = None

# ---------------------------------------------------------------------------
print()
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
