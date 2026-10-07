"""test_phase1.py -- our manager is behaviour-identical, and every knob is HOT.

    python3.12 local-ops/kvharness/test_phase1.py

Two jobs, and the second is the one that catches the failure this work is most
likely to ship:

  PARITY -- our cap must agree with patch6's block for block at `equal`/`fifo`,
  and flipping `kvmgr_enabled` off must reach patch6's own code path rather than
  "our policy with today's defaults".

  HOT CONFIG -- for EVERY knob in the table, the value must be RE-READ rather
  than captured at import or in a constructor.  Each of these is verified by
  breaking it; a hot-config test that passes against a cached constant is the
  whole failure mode.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(harness_dir := os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "patches", "kvmgr_plugin"))

import harness  # noqa: E402

PASS = 0
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL.append(name)
        print(f"FAIL  {name}  {detail}")


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r} want {want!r}")


H = harness.load()
MGR = H.mgr_mod
iface = sys.modules["vllm.v1.kv_cache_interface"]

# Our plugin imports `vllm.*`, which the harness has already faked/loaded.
import importlib  # noqa: E402

from kvmgr import flags, policy  # noqa: E402

# `kvmgr.__init__` re-exports the FUNCTION as `register`, which shadows the
# submodule for `from kvmgr import register`. Reach the module explicitly.
register = importlib.import_module("kvmgr.register")
from kvmgr.manager import DSparkMLAAttentionManager  # noqa: E402

FLAGS_DIR = tempfile.mkdtemp(prefix="kvmgr-flags-")
FLAGS_PATH = os.path.join(FLAGS_DIR, "flags.json")
F = harness.Flags(FLAGS_PATH)


def fresh(fraction_env: str | None = None, **flag_kw):
    """A manager with a clean flag state. Resets both caches."""
    os.environ.pop(flags.LEGACY_FRACTION_ENV, None)
    for name in list(os.environ):
        if name.startswith("VLLM_") and name[5:].lower() in flags.DEFAULTS:
            del os.environ[name]
    if fraction_env is not None:
        os.environ[flags.LEGACY_FRACTION_ENV] = fraction_env
    flags.reset_for_tests()
    F.write(**flag_kw)
    harness.reset_manager_count()
    pool = harness.make_pool(num_gpu_blocks=harness.LIVE_NUM_GPU_BLOCKS)
    mgrs = harness.make_live_groups(pool, reset=False)
    return pool, mgrs


# ---------------------------------------------------------------------------
# 1. REGISTRATION
# ---------------------------------------------------------------------------
register.register()
eq("our class is registered for MLAAttentionSpec",
   MGR.spec_manager_map[iface.MLAAttentionSpec], DSparkMLAAttentionManager)
check("register() is idempotent", (register.register() or True))
eq("still registered once", MGR.spec_manager_map[iface.MLAAttentionSpec],
   DSparkMLAAttentionManager)

# MNT-11: patch6's own lazy hook, which is what guarantees registration in the
# process that builds managers -- the entry point cannot be verified to load in
# the engine core without the cluster.
check("patch6 carries the MNT-11 lazy registration hook",
      hasattr(MGR, "_maybe_register_kvmgr"),
      "belt and braces: a silent no-op is the failure this stack keeps paying for")
MGR.spec_manager_map[iface.MLAAttentionSpec] = MGR.MLAAttentionManager
MGR._kvmgr_tried = False
register._REGISTERED = False
harness.reset_manager_count()
_hook_pool = harness.make_pool(num_gpu_blocks=64, hash_block_size=16)
_built = MGR.get_manager_for_kv_cache_spec(
    kv_cache_spec=iface.MLAAttentionSpec(block_size=16),
    max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
    block_pool=_hook_pool, enable_caching=True, kv_cache_group_id=0,
    dcp_world_size=1, pcp_world_size=1)
check("the hook registers kvmgr on the FIRST manager built, with no entry point",
      isinstance(_built, DSparkMLAAttentionManager),
      "so Phase 1 cannot silently do nothing if general plugins skip engine core")

# And it is tried ONCE, not per manager: this is on the construction path.
MGR.spec_manager_map[iface.MLAAttentionSpec] = MGR.MLAAttentionManager
MGR.get_manager_for_kv_cache_spec(
    kv_cache_spec=iface.MLAAttentionSpec(block_size=16),
    max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
    block_pool=_hook_pool, enable_caching=True, kv_cache_group_id=0,
    dcp_world_size=1, pcp_world_size=1)
check("and only tries once -- a retry per manager would re-import on a hot path",
      MGR.spec_manager_map[iface.MLAAttentionSpec] is MGR.MLAAttentionManager)
register._REGISTERED = False
register.register()

check("the other four spec types are untouched (group 0 only)",
      MGR.spec_manager_map[iface.SlidingWindowMLASpec] is MGR.SlidingWindowMLAManager
      and MGR.spec_manager_map[iface.FullAttentionSpec] is MGR.FullAttentionManager,
      "the narrow rollout, and the fallback if this is wrong")

pool, mgrs = fresh(kvmgr_enabled=1, protected_fraction=0.75)
eq("the factory builds OUR class for group 0", type(mgrs[0]), DSparkMLAAttentionManager)
check("and upstream's for the windowed groups",
      all(type(m) is MGR.SlidingWindowMLAManager for m in mgrs[1:]))

# The geometry hook: wired by register(), fed by a real KVCacheConfig.
cfg = iface.KVCacheConfig(
    num_blocks=harness.LIVE_NUM_GPU_BLOCKS,
    kv_cache_groups=[iface.KVCacheGroupSpec(kv_cache_spec=s) for s in harness.live_specs()],
)
harness.reset_manager_count()
coord_mod = sys.modules.get("vllm.v1.core.kv_cache_coordinator")
check("the coordinator wrap is installed",
      coord_mod is not None and getattr(coord_mod.KVCacheCoordinator.__init__, "_kvmgr_wrapped", False)
      if coord_mod else False, "stamps group_count/group_block_sizes")

DSparkMLAAttentionManager.group_count = len(cfg.kv_cache_groups)
DSparkMLAAttentionManager.group_block_sizes = tuple(
    g.kv_cache_spec.block_size for g in cfg.kv_cache_groups)
eq("group geometry read off the config", DSparkMLAAttentionManager.group_block_sizes,
   harness.LIVE_BLOCK_SIZES)
DSparkMLAAttentionManager.group_sliding_windows = tuple(
    getattr(g.kv_cache_spec, "sliding_window", None) for g in cfg.kv_cache_groups)
eq("and the per-group windows, which by_need needs",
   DSparkMLAAttentionManager.group_sliding_windows, (None, 128, 128, 128, 128))

# ---------------------------------------------------------------------------
# 2. PARITY -- the cap agrees with patch6, block for block
# ---------------------------------------------------------------------------
for frac in (0.30, 0.50, 0.75, 1.00, 0.0):
    pool, mgrs = fresh(kvmgr_enabled=1, protected_fraction=frac)
    DSparkMLAAttentionManager.group_count = 5
    DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
    DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
    ours = mgrs[0]._max_protected_prompt_blocks()

    # patch6's own answer, from the unmodified parent, via the legacy env.
    os.environ[flags.LEGACY_FRACTION_ENV] = str(frac)
    theirs = MGR.MLAAttentionManager._max_protected_prompt_blocks(mgrs[0])
    eq(f"cap parity at fraction {frac}", ours, theirs)

# negative fraction: patch6's "keep only the legacy cap" contract
pool, mgrs = fresh(kvmgr_enabled=1, protected_fraction=-1.0)
DSparkMLAAttentionManager.group_count = 5
DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
eq("a negative fraction falls back to the legacy cap only",
   mgrs[0]._max_protected_prompt_blocks(), 8192)

# the live configuration, against the numbers in the plan
pool, mgrs = fresh(kvmgr_enabled=1, protected_fraction=0.75)
DSparkMLAAttentionManager.group_count = 5
DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
eq("live config reproduces 3,643 blocks", mgrs[0]._max_protected_prompt_blocks(), 3643)
eq("= 932,608 tokens", policy.cap_tokens(3643, 256), 932608)

# ---------------------------------------------------------------------------
# 3. THE KILL SWITCH IS HOT, AND REACHES UPSTREAM'S CODE PATH
# ---------------------------------------------------------------------------
pool, mgrs = fresh(kvmgr_enabled=0, protected_fraction=0.75)
DSparkMLAAttentionManager.group_count = 5
DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
m = mgrs[0]

# With our policy OFF, patch6 reads os.environ -- so the flags file is ignored,
# which is the observable signature of being on the upstream path.
eq("off: the cap is patch6's, from os.environ (default 0.30)",
   m._max_protected_prompt_blocks(), int(24288 * 0.30 / 5))
F.write(kvmgr_enabled=1, protected_fraction=0.75)
eq("on: the cap is ours, from the flags file -- live, no restart",
   m._max_protected_prompt_blocks(), 3643)
F.write(kvmgr_enabled=0, protected_fraction=0.75)
eq("off again: back to patch6 in the same process",
   m._max_protected_prompt_blocks(), int(24288 * 0.30 / 5))
F.write(kvmgr_enabled=1, protected_fraction=0.75)
eq("and on again", m._max_protected_prompt_blocks(), 3643)

harness.clear_log()
F.write(kvmgr_enabled=0, protected_fraction=0.75)
m._max_protected_prompt_blocks()
check("a kvmgr_enabled flip LOGS both the old and new value",
      "kvmgr_enabled CHANGED" in harness.log_text(),
      "a silent config change is the failure this repo keeps paying for")

# ---------------------------------------------------------------------------
# 4. HOT CONFIG, PER KNOB -- re-read, never captured
# ---------------------------------------------------------------------------
pool, mgrs = fresh(kvmgr_enabled=1)
PROBE = {
    "kvmgr_enabled": (0, 1),
    "protected_fraction": (0.10, 0.80),
    "protect_alloc": ("equal", "by_need"),
    "evict_order": ("fifo", "lru"),
    "disk_enabled": (0, 1),
    "disk_budget_gb": (0.0, 64.0),
    "disk_restore": (0, 1),
    "kvmgr_log_level": (0, 2),
    "kvmgr_stats_port": (0, 9107),
    "swa_recycle": (1, 0),
    "swa_recycle_cached": (0, 1),
    "free_queue_prepend_uncached": (0, 1),
    "pin_static_head": (0, 1),
    "head_pin_fraction": (0.05, 0.25),
}
eq("the probe covers every knob in the table",
   sorted(PROBE), sorted(flags.DEFAULTS))

for name, (lo, hi) in PROBE.items():
    F.write(**{name: lo})
    a = flags.raw(name)
    F.write(**{name: hi})
    b = flags.raw(name)
    check(f"{name} is re-read from the file with no restart", a != b,
          f"{a!r} -> {b!r}")

for name in flags.DEFAULTS:
    F.remove()
    flags.reset_for_tests()
    eq(f"{name}: absent file -> today's default", flags.raw(name), flags.DEFAULTS[name])
    F.write_raw("{broken")
    eq(f"{name}: bad JSON -> today's default", flags.raw(name), flags.DEFAULTS[name])

# env wins, per knob, including the legacy name
for name in flags.DEFAULTS:
    F.write(**{name: PROBE[name][1]})
    env_name = (flags.LEGACY_FRACTION_ENV if name == "protected_fraction"
                else "VLLM_" + name.upper())
    os.environ[env_name] = str(PROBE[name][0])
    flags.reset_for_tests()
    F.invalidate()
    got = flags.raw(name)
    want = type(flags.DEFAULTS[name])(PROBE[name][0]) if not isinstance(
        flags.DEFAULTS[name], bool) else bool(int(PROBE[name][0]))
    check(f"{name}: VLLM env wins over the file", got == want,
          f"got {got!r} want {want!r} via {env_name}")
    del os.environ[env_name]
flags.reset_for_tests()

check("protected_fraction honours the LEGACY env name for boot-pinning",
      flags.LEGACY_FRACTION_ENV == "VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION",
      "renaming it would silently drop the 0.75 live on both nodes")

os.environ[flags.LEGACY_FRACTION_ENV] = "0.60"
F.write(protected_fraction=0.10)
eq("the legacy env pins the fraction", flags.raw("protected_fraction"), 0.60)
check("and audit_env() REPORTS the pin, so a dead flag is visible",
      any("protected_fraction" in e for e in flags.audit_env()),
      "env silently winning is Trap 1; this is how it stops being silent")
harness.clear_log()
flags.log_env_audit()
check("the audit warns for a knob that must not be env-pinned",
      "CANNOT move this knob" in harness.log_text())
del os.environ[flags.LEGACY_FRACTION_ENV]

# ---------------------------------------------------------------------------
# 5. AN UNBUILT POLICY VALUE IS REFUSED, NOT SILENTLY DOWNGRADED
# ---------------------------------------------------------------------------
flags.reset_for_tests()
harness.clear_log()
F.write(kvmgr_enabled=1, protect_alloc="by_need")
eq("by_need is recognised but refused in Phase 1",
   flags.knob("protect_alloc"), "equal")
check("and says so, naming the knob", "NOT YET IMPLEMENTED" in harness.log_text())

flags.reset_for_tests()
harness.clear_log()
F.write(protect_alloc="nonsense")
eq("an invalid value falls back to the default", flags.knob("protect_alloc"), "equal")
check("and is reported as invalid", "is not one of" in harness.log_text())

flags.reset_for_tests()
harness.clear_log()
F.write(evict_order="lru_tail_first")
eq("lru_tail_first likewise refused for now", flags.knob("evict_order"), "fifo")

try:
    flags.raw("knob_that_does_not_exist")
    check("an unknown knob raises", False)
except KeyError:
    check("an unknown knob raises rather than defaulting silently", True,
          "a knob outside the table is one no hot-config test covers")

# ---------------------------------------------------------------------------
# 6. THE CAP CHANGE IS LOGGED, IN BLOCKS AND TOKENS
# ---------------------------------------------------------------------------
pool, mgrs = fresh(kvmgr_enabled=1, protected_fraction=0.30)
DSparkMLAAttentionManager.group_count = 5
DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
m = mgrs[0]
m._max_protected_prompt_blocks()
harness.clear_log()
F.write(kvmgr_enabled=1, protected_fraction=0.75)
m._max_protected_prompt_blocks()
txt = harness.log_text()
check("the new cap is logged on change", "protected cap for KV group 0" in txt)
check("in BLOCKS", "3643 blocks" in txt)
check("and in TOKENS -- the figure an operator can act on", "932608 tokens" in txt)
check("naming the previous value", "was 1457 blocks" in txt, txt[-200:])

harness.clear_log()
m._max_protected_prompt_blocks()
check("an unchanged cap logs NOTHING (no per-call spam on the hot path)",
      "protected cap" not in harness.log_text())

# ---------------------------------------------------------------------------
# 7. RECONCILE ON THE FLAG-CHANGE EDGE
# ---------------------------------------------------------------------------
harness.reset_manager_count()
small_pool = harness.make_pool(num_gpu_blocks=400, hash_block_size=16)
flags.reset_for_tests()
F.write(kvmgr_enabled=1, protected_fraction=0.50)
mgr = MGR.get_manager_for_kv_cache_spec(
    kv_cache_spec=iface.MLAAttentionSpec(block_size=16),
    max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
    block_pool=small_pool, enable_caching=True, kv_cache_group_id=0,
    dcp_world_size=1, pcp_world_size=1)
DSparkMLAAttentionManager.group_count = 1
DSparkMLAAttentionManager.group_block_sizes = (16,)
eq("a 400-block pool at 0.50 caps at 200", mgr._max_protected_prompt_blocks(), 200)

blocks = small_pool.get_new_blocks(120)
for i, b in enumerate(blocks):
    b.block_hash = H.utils.make_block_hash_with_group_id(
        H.utils.BlockHash((i + 1).to_bytes(32, "big")), 0)
mgr._protect_prompt_blocks(blocks)
eq("120 blocks protected under a cap of 200", len(mgr._protected_prompt_block_ids), 120)

# Lower the cap. Nothing has called protect since, so patch6 alone would hold
# all 120 indefinitely -- a quiet group with stale protection.
F.write(kvmgr_enabled=1, protected_fraction=0.10)
eq("the cap drops to 40", mgr._max_protected_prompt_blocks(), 40)
eq("...but nothing is released until something runs",
   len(mgr._protected_prompt_block_ids), 120)

harness.clear_log()
released = mgr.kvmgr_reconcile()
eq("reconcile releases down to the new cap", len(mgr._protected_prompt_block_ids), 40)
eq("and reports how many went", released, 80)
check("and logs it", "released 80 protected block(s)" in harness.log_text())

# And the per-step hook does it without being asked.
F.write(kvmgr_enabled=1, protected_fraction=0.05)
harness.clear_log()
mgr.new_step_starts()
eq("new_step_starts reconciles on the edge", len(mgr._protected_prompt_block_ids), 20)
check("logging the new cap and the release",
      "protected cap" in harness.log_text() and "released" in harness.log_text())

harness.clear_log()
mgr.new_step_starts()
check("a step with no flag change does nothing and logs nothing",
      harness.log_text() == "", harness.log_text()[:200])

# ---------------------------------------------------------------------------
# 8. POLICY ARITHMETIC, INCLUDING THE by_need SHAPE PHASE 3 WILL TURN ON
# ---------------------------------------------------------------------------
eq("equal split is today's formula",
   policy.pool_share_equal(24288, 0.75, 5), 3643)
eq("legacy cap formula agrees with patch6's",
   policy.legacy_cap(1048576, 256), 8192)

# The NAIVE reading -- equal token extent -- is a regression, and the arithmetic
# is kept here so nobody re-derives it as an improvement.
naive_e = int(24288 * 0.75) / sum(1.0 / b for b in harness.LIVE_BLOCK_SIZES)
check("equal-token-extent by_need would be a 21x REGRESSION, not a fix",
      naive_e < 50_000,
      f"{int(naive_e)} tokens per group against 932,608 for group 0 today")

eq("window_need_blocks matches patch6's MNT-9 formula",
   [policy.window_need_blocks(w, bs)
    for w, bs in zip(harness.LIVE_SLIDING_WINDOWS, harness.LIVE_BLOCK_SIZES)],
   [None, 3, 3, 33, 17])

# ...and against patch6's own code, so the two definitions cannot drift.
harness.reset_manager_count()
probe_pool = harness.make_pool(num_gpu_blocks=400, hash_block_size=4)
swa = MGR.get_manager_for_kv_cache_spec(
    kv_cache_spec=iface.SlidingWindowMLASpec(block_size=4, sliding_window=128),
    max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
    block_pool=probe_pool, enable_caching=True, kv_cache_group_id=3,
    dcp_world_size=1, pcp_world_size=1)
fake_blocks = [None] * 500
patch6_need = len(swa._head_pin_candidates(fake_blocks, 400))
eq("patch6 agrees that a 128-token window over 4-token blocks is 33 blocks",
   patch6_need, policy.window_need_blocks(128, 4))

caps_by_need = [
    policy.protected_cap(num_gpu_blocks=24288, fraction=0.75, num_groups=5,
                         block_size=bs, max_model_len=1048576, mode="by_need",
                         block_sizes=harness.LIVE_BLOCK_SIZES, group_index=i,
                         sliding_windows=harness.LIVE_SLIDING_WINDOWS)
    for i, bs in enumerate(harness.LIVE_BLOCK_SIZES)
]
eq("the four windowed groups get exactly their window",
   caps_by_need[1:], [3, 3, 33, 17])
eq("under by_need group 0 is clamped by the LEGACY cap, not the fraction",
   caps_by_need[0], 8192)
g0_tokens = policy.cap_tokens(caps_by_need[0], 256)
eq("so group 0 reaches 2,097,152 tokens", g0_tokens, 2097152)
check("which is 2.25x today, NOT the plan's 4.2x",
      abs(g0_tokens / 932608 - 2.25) < 0.01,
      f"{g0_tokens/932608:.2f}x -- Phase 0 finding 2, re-asserted here")
check("and still 2x the 1M context ceiling, so the 740,315-token session fits",
      g0_tokens > 2 * 740315)
check("and the windowed groups together cost 56 blocks, 0.2% of the pool",
      sum(caps_by_need[1:]) == 56)

# ---------------------------------------------------------------------------
print()
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
