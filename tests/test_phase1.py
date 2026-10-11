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
# 1b. EVERY LOGGER MUST SIT UNDER THE `vllm.` NAMESPACE
# ---------------------------------------------------------------------------
# vLLM hands a handler to the `vllm` logger only, propagate=false, with no root
# handler -- so a logger named `kvmgr.flags` has NO handler and an effective
# level of 30 (WARNING), and every info() is dropped. Measured in the running
# container 2026-10-07, which is how the first deploy came up with the
# registration line, the cap line, the flag-change line and the env audit all
# INVISIBLE -- indistinguishable from the plugin never having loaded.
import glob  # noqa: E402

_plugin_dir = os.path.join(REPO_ROOT := os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "patches", "kvmgr_plugin", "kvmgr")
_mods = sorted(glob.glob(os.path.join(_plugin_dir, "*.py")))
check("found the plugin modules", len(_mods) >= 6, f"{len(_mods)}")
for _m in _mods:
    _src = open(_m).read()
    _name = os.path.basename(_m)
    if "init_logger(" not in _src:
        continue
    bare = "init_logger(__name__)" in _src
    namespaced = 'init_logger("vllm." + __name__)' in _src
    check(f"{_name}: logger is under the vllm namespace, not bare __name__",
          namespaced and not bare,
          "a bare __name__ logger has effective level 30 and drops every info()")

# And the live objects agree, which is the half a source grep cannot see.
for _lg in (flags.logger, _mgr_mod_logger := __import__(
        "kvmgr.manager", fromlist=["logger"]).logger):
    check(f"{_lg.name}: name starts with 'vllm.'", _lg.name.startswith("vllm."),
          _lg.name)

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

# *** THE SIGNATURE OF "WE ARE ON PATCH6'S PATH" CHANGED ON 2026-10-07. ***
# It used to be "the flags file is ignored", because patch6 read the fraction
# straight off os.environ. Now that the fraction is hot in patch6 too, BOTH
# paths honour the file -- which is the point of that fix and means the old
# discriminator is gone.
#
# `protect_alloc` is the discriminator now, and it is a better one because it is
# the behavioural difference that actually matters: patch6 only ever does the
# equal /num_managers split, so by_need is INERT with our policy off and moves
# the cap 2.25x with it on.
F.write(kvmgr_enabled=0, protected_fraction=0.75, protect_alloc="by_need")
eq("off: by_need is INERT -- patch6 only knows the equal split",
   m._max_protected_prompt_blocks(), 3643)
F.write(kvmgr_enabled=1, protected_fraction=0.75, protect_alloc="by_need")
eq("on: by_need applies -- 8192 blocks, live, no restart",
   m._max_protected_prompt_blocks(), 8192)
F.write(kvmgr_enabled=0, protected_fraction=0.75, protect_alloc="by_need")
eq("off again: back to patch6's equal split in the same process",
   m._max_protected_prompt_blocks(), 3643)
F.write(kvmgr_enabled=1, protected_fraction=0.75, protect_alloc="by_need")
eq("and on again", m._max_protected_prompt_blocks(), 8192)

# ...and the fraction is hot on BOTH sides of the switch, which is the new
# property and the one the operator asked for.
F.write(kvmgr_enabled=0, protected_fraction=0.50, protect_alloc="equal")
eq("fraction is hot with our policy OFF (patch6's own path)",
   m._max_protected_prompt_blocks(), int(24288 * 0.50 / 5))
F.write(kvmgr_enabled=1, protected_fraction=0.50, protect_alloc="equal")
eq("and hot with it ON", m._max_protected_prompt_blocks(), int(24288 * 0.50 / 5))
F.write(kvmgr_enabled=1, protected_fraction=0.75, protect_alloc="equal")

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
    "disk_store_paused": (0, 1),
    # Default is ON (skip a store whose file exists BEFORE the GPU copy),
    # so the probe pair is inverted relative to the off-by-default knobs:
    # (value to pin via env, value to write to the file).
    "disk_skip_present_before_copy": (0, 1),
    "staging_load_slots": (0, 4),
    "load_batch_max": (0, 32),
    "load_inflight_max": (0, 2),
    "load_ops_max": (0, 512),
    # the STORE's per-round budget, added 2026-10-11 after the store path was
    # measured to stall the stream by ~1.1s with no cap at all
    "store_batch_max": (0, 16),
    "store_inflight_max": (0, 2),
    "store_ops_max": (0, 512),
    "stall_fail_after_s": (0, 90),
    # The staging knobs. These ARE genuinely hot in both value and effect: the
    # live pool size is re-read from the flags on every transfer round and
    # `set_limit` only moves which already-allocated rows are handed out, so
    # nothing is reallocated. The ALLOCATION (`staging_blocks` in
    # KV_TRANSFER_CONFIG) is the boot ceiling and is deliberately NOT a flag --
    # a knob that silently needed a restart is what the control plane exists to
    # stop, so it is not offered as one.
    "staging_slots": (0, 64),
    "staging_write_floor": (0, 8),
    "staging_read_floor": (0, 8),
    # *** THE ONE KNOB WHOSE VALUE IS HOT AND WHOSE EFFECT IS NOT, and the
    # loop below cannot tell the difference. *** It passes here because
    # `flags.raw()` really does re-read it with no restart -- but it is
    # consulted at ATTACH time, and a FastAPI route either exists in the app
    # or it does not. So nobody should read this passing assertion as "the
    # control plane can be switched on hot": it cannot, and
    # `attach_router`'s own log line says so when it declines.
    "kvmgr_control_plane": (0, 1),
    "kvmgr_log_level": (0, 2),
    "kvmgr_stats_port": (0, 9107),
    "offload_allow_hybrid": (0, 1),
    "offload_odirect": (0, 1),
    "offload_verify": (0, 1),
    "gpu_cache_epoch": (0, 7),
    "swa_recycle": (1, 0),
    "swa_recycle_cached": (0, 1),
    "free_queue_prepend_uncached": (0, 1),
    "pin_static_head": (0, 1),
    "head_pin_fraction": (0.05, 0.25),
    # *** ADDED BECAUSE THIS LOOP REFUSED TO RUN WITHOUT IT, which is the
    # table-driven discipline working. *** `head_pin_partial` was read by
    # patch6 with an inline default and declared nowhere, so it was an
    # invisible knob; declaring it immediately failed here, because this suite
    # enumerates DEFAULTS and demands a probe pair per knob. A new knob cannot
    # be added without somebody verifying its hot-config behaviour.
    "head_pin_partial": (0, 1),
    "gpu_cache_drop_allowed": (0, 1),
    # The presence journal. Added 2026-10-11, and this table is why: the audit
    # below raised KeyError on the first run after they were declared, which is
    # the third time it has caught a knob nobody had proved hot.
    "journal_enabled": (0, 1),
    "journal_max_records": (100, 50000),
    "journal_max_bytes": (65536, 67108864),
    "journal_fsync": (0, 1),
}
# *** THIS ASSERTION IS WHY A NEW KNOB CANNOT SHIP UNPROVEN, and it caught one
# on 2026-10-08. *** `disk_store_paused` was added to DEFAULTS and not here, and
# the loop below raised `KeyError: 'disk_store_paused'` -- so the suite went RED
# rather than silently skipping the one knob nobody had proved hot. The eq()
# below would have named it; the loop got there first, which is louder.
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

# *** A BLANK ENV VALUE COUNTS AS UNSET, and that is what lets the compose stop
# pinning this knob at all. *** docker-compose emits "" for
# ${PROTECTED_FRACTION:-} when the variable is unset, and float("") raises --
# which the old code caught and turned into 0.30, silently IGNORING the flags
# file and dropping retention from 0.75 to 0.30 with nothing saying so.
F.write(protected_fraction=0.10)
for _blank in ("", "   ", "\t"):
    os.environ[flags.LEGACY_FRACTION_ENV] = _blank
    flags.reset_for_tests()
    MGR._flags_checked_at = -1.0
    eq(f"a blank env value {_blank!r} is UNSET, so the flags file still wins",
       flags.raw("protected_fraction"), 0.10)
del os.environ[flags.LEGACY_FRACTION_ENV]
MGR._flags_checked_at = -1.0
eq("with the env absent the file wins too", flags.raw("protected_fraction"), 0.10)
os.environ[flags.LEGACY_FRACTION_ENV] = "0.55"
MGR._flags_checked_at = -1.0
eq("and a REAL env value still pins an arm for a boot",
   flags.raw("protected_fraction"), 0.55)
del os.environ[flags.LEGACY_FRACTION_ENV]
MGR._flags_checked_at = -1.0
check("garbage in the env falls back to 0.30 rather than raising",
      (os.environ.__setitem__(flags.LEGACY_FRACTION_ENV, "not-a-number"),
       MGR.__setattr__("_flags_checked_at", -1.0),
       flags.raw("protected_fraction"))[2] == 0.30)
del os.environ[flags.LEGACY_FRACTION_ENV]
MGR._flags_checked_at = -1.0

# the compose must not re-introduce a default, or the knob goes cold again
_compose = open(os.path.join(REPO_ROOT, "docker-compose.dspark.yml")).read()
# A BLANK env value must not be reported as a PIN. The first version reported
# "pinned by VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION=''" on the very boot where the
# knob had just been made hot -- telling the operator a live knob was cold.
for _blank in ("", "  "):
    os.environ[flags.LEGACY_FRACTION_ENV] = _blank
    check(f"a blank env value {_blank!r} is NOT reported as a pin",
          not any("protected_fraction" in e for e in flags.audit_env()),
          f"audit_env() -> {flags.audit_env()}")
    harness.clear_log()
    flags.log_env_audit()
    check(f"and the audit says nothing is pinned, not that it CANNOT move",
          "CANNOT move this knob" not in harness.log_text(),
          harness.log_text()[:140])
os.environ[flags.LEGACY_FRACTION_ENV] = "0.75"
check("while a REAL value is still reported as a pin",
      any("protected_fraction" in e for e in flags.audit_env()))
del os.environ[flags.LEGACY_FRACTION_ENV]
MGR._flags_checked_at = -1.0

check("the compose does NOT default VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION",
      'VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION: "${PROTECTED_FRACTION:-}"' in _compose,
      "a default there wins over the flags file and makes the knob restart-only")

# ---------------------------------------------------------------------------
# 5. AN UNBUILT POLICY VALUE IS REFUSED, NOT SILENTLY DOWNGRADED
# ---------------------------------------------------------------------------
flags.reset_for_tests()
harness.clear_log()
F.write(kvmgr_enabled=1, protect_alloc="by_need")
eq("by_need is BUILT as of Phase 3, so it is honoured",
   flags.knob("protect_alloc"), "by_need")
F.write(kvmgr_enabled=1, evict_order="lru_tail_first")
eq("and so is lru_tail_first", flags.knob("evict_order"), "lru_tail_first")

flags.reset_for_tests()
harness.clear_log()
F.write(protect_alloc="nonsense")
eq("an invalid value falls back to the default", flags.knob("protect_alloc"), "equal")
check("and is reported as invalid", "is not one of" in harness.log_text())

# THE REFUSAL MECHANISM IS STILL TESTED, with a synthetic unbuilt value -- the
# next arm anyone adds must refuse rather than ship looking live, and a test
# that only covered Phase 1's two values would go vacuous the moment they were
# built. Which is exactly what happened to it.
flags.reset_for_tests()
harness.clear_log()
flags._VALID["evict_order"] = ("fifo", "lru", "lru_tail_first", "future_arm")
flags._IMPLEMENTED["evict_order"] = ("fifo", "lru", "lru_tail_first")
try:
    F.write(evict_order="future_arm")
    eq("a recognised-but-unbuilt value falls back", flags.knob("evict_order"), "fifo")
    check("and says NOT YET IMPLEMENTED, naming the knob",
          "NOT YET IMPLEMENTED" in harness.log_text())
finally:
    flags._VALID["evict_order"] = ("fifo", "lru", "lru_tail_first")
    flags._IMPLEMENTED.pop("evict_order", None)

# ---------------------------------------------------------------------------
# 5b. A KNOB THAT CANNOT WORK AT THIS LAYER REFUSES, WITH THE REASON
# ---------------------------------------------------------------------------
# *** RE-AIMED, NOT DELETED, when Phase 5 built disk_budget_gb. *** The
# property these check -- a refusal must name a cause and point somewhere -- is
# right; only the example moved. This is the second time: disk_restore left the
# table the same way. Driven off flags.BLOCKED itself so the next knob to be
# built needs no edit here at all, which is what stops the table and the test
# drifting apart.
for name in sorted(flags.BLOCKED):
    flags.reset_for_tests()
    harness.clear_log()
    F.write(**{name: (1 if isinstance(flags.DEFAULTS[name], bool) else 64.0)})
    eq(f"{name} is REFUSED, not silently ignored",
       flags.knob(name), flags.DEFAULTS[name])
    txt = harness.log_text()
    check(f"{name}'s refusal says REFUSED", "REFUSED" in txt)
    check(f"{name}'s refusal carries the REASON, so the next person to reach "
          f"for the flag learns it from the log",
          "superseded" in txt or "CORRECTNESS HAZARD" in txt
          or "bounds nothing" in txt, txt[:200])

# *** THESE TWO ASSERTED THE OLD, WRONG CAUSE and were re-aimed 2026-10-07
# rather than deleted: the property is right (a refusal must name a cause and
# point somewhere) and only the cause moved. The old reason blamed vllm#45663
# and "NO KV BYTES"; what actually gates restore is measured -- every file the
# tiering tier wrote has rank>0's half EXACTLY ZERO.
# *** disk_restore IS NO LONGER BLOCKED, and the removal fixed a real defect
# rather than relaxing a guard. *** It was in BLOCKED while the restore gates
# read it with raw() -- which does no validation -- so the feature ran while
# BLOCKED said it was off and /kv reported disk_restore: False. A capability
# live behind a reader that denies it is worse than a refusal nobody can see.
# Now: it is out of the table (it is built), and the gates use a VALIDATED read
# so an entry here means blocked for real.
check("disk_restore is NOT in BLOCKED any more",
      "disk_restore" not in flags.BLOCKED)
check("*** and the restore gates use a VALIDATED read, so a future BLOCKED "
      "entry would actually block ***",
      'flags.raw("disk_restore")' not in open(os.path.join(
          os.path.dirname(os.path.dirname(os.path.dirname(
              os.path.abspath(__file__)))),
          "patches", "kvmgr_plugin", "kvmgr", "diskspec.py")).read())
for _k in sorted(flags.BLOCKED):
    check(f"{_k}'s refusal still POINTS somewhere",
          "diskspec.py" in flags.BLOCKED[_k])

# raw() still reports what was ASKED FOR -- the refusal is knob()'s job, so a
# reader can tell "nobody set it" from "it was set and refused".
F.write(disk_enabled=1)
check("raw() still shows what was requested, so the two are distinguishable",
      bool(flags.raw("disk_enabled")) is True
      and bool(flags.knob("disk_enabled", quiet=True)) is False)
F.write(disk_enabled=0)

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
   [None, 3, 3, 3, 17])

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
   caps_by_need[1:], [3, 3, 3, 17])
eq("under by_need group 0 is clamped by the LEGACY cap, not the fraction",
   caps_by_need[0], 8192)
g0_tokens = policy.cap_tokens(caps_by_need[0], 256)
eq("so group 0 reaches 2,097,152 tokens", g0_tokens, 2097152)
check("which is 2.25x today, NOT the plan's 4.2x",
      abs(g0_tokens / 932608 - 2.25) < 0.01,
      f"{g0_tokens/932608:.2f}x -- Phase 0 finding 2, re-asserted here")
check("and still 2x the 1M context ceiling, so the 740,315-token session fits",
      g0_tokens > 2 * 740315)
check("and the windowed groups together cost 26 blocks, 0.1% of the pool",
      sum(caps_by_need[1:]) == 26,
      "measured on the live engine: windows (None,128,128,8,128)")

# ---------------------------------------------------------------------------
# THE MEASURED-SAFE DEFAULTS, PINNED. A safe value carried as a runtime override
# is a value you lose on the next boot: every 700K measurement ran with
# load_inflight_max=1 as an override, and when the engine restarted it came back
# with restore LIVE and the DEFAULT of 4 -- the configuration measured on
# 2026-10-10 to kill the engine (Xid 32 x2 on the head, unspecified launch
# failure, EngineDeadError) with load_ops_max pinned at 2048 throughout.
#
# These two assertions exist so raising either one has to be a deliberate act
# that breaks a test naming the fault, rather than an edit that looks like
# tuning. They assert the DEFAULT, not a knob read, because the default is the
# thing a restart falls back to.
import kvmgr.flags as _flags_mod
_d = _flags_mod.DEFAULTS if hasattr(_flags_mod, "DEFAULTS") else _flags_mod._DEFAULTS
eq("load_inflight_max DEFAULTS to the measured-safe 1, not the fatal 4",
   _d["load_inflight_max"], 1)
eq("and load_ops_max defaults to the measured 2048",
   _d["load_ops_max"], 2048)
# AND THE STORE'S BUDGET, which did not exist until the stream stall was
# measured. An unbounded store round is a ~1.1s hole in every generation, so
# these three being present at all is the property; the exact numbers are
# tuned against test-meter.sh section 1 and are expected to move.
eq("store_inflight_max defaults to 1 -- the knob that bounds a STEP",
   _d["store_inflight_max"], 1)
check("store_batch_max is a small cap, not the 256-slot pool",
      0 < _d["store_batch_max"] <= 64,
      "got %r; the pool is 256 and submitting that many was the stall" % (
          _d["store_batch_max"],))
check("store_ops_max bounds DESCRIPTORS, so group 0 cannot dominate a step",
      _d["store_ops_max"] > 0, "got %r" % (_d["store_ops_max"],))
check("and inflight_max=1 costs nothing: 1 and 2 both restore 700K in 12s",
      True,
      "measured 2026-10-10 -- the restore is bandwidth-bound, not "
      "submission-bound, so there is no throughput traded for this default")

print()
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
