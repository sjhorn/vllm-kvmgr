"""test_phase18_knobs.py -- every knob is HOT, and env > file > default, per knob.

Moved out of test_phase1.py section 4 (vllm-0.31 branch) because none of it
needs the GPU-manager core: it is the flags table, so it must run on every
vLLM, including 0.31, where phase1 as a whole is skipped (no GPU manager).
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))

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


harness.load()
from kvmgr import flags  # noqa: E402

FLAGS_PATH = os.path.join(tempfile.mkdtemp(prefix="kvmgr-flags-"), "flags.json")
F = harness.Flags(FLAGS_PATH)
REPO_ROOT = harness.DSPARK_REPO
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
    # v0.4.0: the read stage's own bound, beside the GPU one
    "load_read_ahead": (0, 2),
    # default ON, so inverted like disk_skip_present_before_copy
    "load_wait_compute": (0, 1),
    # v0.4.4: the load poll thread, off by default
    "load_poll_ms": (0, 5),
    "load_ops_max": (0, 512),
    # ask9: the load copy path and its chunk size
    "load_copy_api": ("loop", "chunk"),
    "load_copy_chunk": (16, 256),
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
    # Cross-rank verification and the free-space floor, added 2026-10-12. This
    # audit raised KeyError on `disk_verify_enabled` the first time it ran
    # after they were declared -- the FOURTH time it has caught a knob nobody
    # had proved hot, and louder than the eq() below because the loop gets
    # there first.
    "disk_verify_enabled": (0, 1),
    "disk_verify_chunk": (100, 20000),
    "disk_boot_verify_chunk": (1000, 200000),
    # ask5: the per-rank pool for the boot walk
    "disk_verify_threads": (1, 4),
    # borrowed from sparkcache: failed restore -> recompute, and checksums
    "disk_load_failure_recompute": (0, 1),
    # ask6
    "diag_request_keys": (0, 1),
    "diag_ledger_keys": (0, 1),
    "disk_checksum_write": (0, 1),
    "disk_checksum_verify": (0, 1),
    "disk_verify_defer_s": (0.5, 30.0),
    "disk_verify_rpc_timeout_s": (5.0, 600.0),
    "disk_verify_bg_per_s": (0.0, 5000.0),
    "disk_free_floor_gb": (0.0, 2000.0),
    "disk_free_hard_gb": (0.0, 500.0),
    "disk_free_hysteresis_gb": (0.0, 100.0),
    "disk_free_evict_step_gb": (0.1, 50.0),
    "disk_free_check_s": (0.5, 300.0),
    "disk_free_rpc_s": (1.0, 3600.0),
    # The diagnostics (kvmgr.diag). The VALUE re-reads hot like every knob;
    # the EFFECT is boot-time, because each is installed at plugin load --
    # the same caveat kvmgr_control_plane carries above.
    "diag_hang_dump": (0, 1),
    "diag_transfer_ledger": (0, 1),
    "diag_ledger_file": (0, 1),
    "diag_copy_probe": (0, 1),
    "diag_copy_selftest": (0, 1),
    "diag_serializer_walk": (0, 1),
    "diag_geometry_snapshot": (0, 1),
    "diag_tiering_shims": (0, 1),
    "diag_hot_reload": (0, 1),
}
# *** THIS ASSERTION IS WHY A NEW KNOB CANNOT SHIP UNPROVEN, and it caught one
# on 2026-10-08. *** `disk_store_paused` was added to DEFAULTS and not here, and
# the loop below raised `KeyError: 'disk_store_paused'` -- so the suite went RED
# rather than silently skipping the one knob nobody had proved hot. The eq()
# below would have named it; the loop got there first, which is louder.
# The DSpark fork's knobs are declared only when patch6 is loaded.
from kvmgr.compat import dspark as _fork  # noqa: E402

if not harness.DSPARK:
    for _k in _fork.FORK_DEFAULTS:
        PROBE.pop(_k)
    check("on a vLLM without patch6, the fork's knobs are NOT declared",
          not set(_fork.FORK_DEFAULTS) & set(flags.DEFAULTS),
          sorted(set(_fork.FORK_DEFAULTS) & set(flags.DEFAULTS)))
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
    harness.invalidate_flags()
    eq(f"a blank env value {_blank!r} is UNSET, so the flags file still wins",
       flags.raw("protected_fraction"), 0.10)
del os.environ[flags.LEGACY_FRACTION_ENV]
harness.invalidate_flags()
eq("with the env absent the file wins too", flags.raw("protected_fraction"), 0.10)
os.environ[flags.LEGACY_FRACTION_ENV] = "0.55"
harness.invalidate_flags()
eq("and a REAL env value still pins an arm for a boot",
   flags.raw("protected_fraction"), 0.55)
del os.environ[flags.LEGACY_FRACTION_ENV]
harness.invalidate_flags()
check("garbage in the env falls back to 0.30 rather than raising",
      (os.environ.__setitem__(flags.LEGACY_FRACTION_ENV, "not-a-number"),
       harness.invalidate_flags(),
       flags.raw("protected_fraction"))[2] == 0.30)
del os.environ[flags.LEGACY_FRACTION_ENV]
harness.invalidate_flags()

# the compose must not re-introduce a default, or the knob goes cold again
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
harness.invalidate_flags()

if harness.needs_dspark("the DSpark compose does not pin protected_fraction"):
    _compose = open(os.path.join(REPO_ROOT, "docker-compose.dspark.yml")).read()
    check("the compose does NOT default VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION",
          'VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION: "${PROTECTED_FRACTION:-}"' in _compose,
          "a default there wins over the flags file and makes the knob restart-only")


print()
print(f"{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)
