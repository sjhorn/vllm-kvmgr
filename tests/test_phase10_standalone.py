"""test_phase10_standalone.py -- kvmgr needs no patch of ours, and the fork
bridge does what it claims when the fork IS there.

    python3.12 test_phase10_standalone.py                        # pristine vLLM
    python3.12 test_phase10_standalone.py --with-dspark-patches  # the DSpark fork

Until 2026-10-13 `kvmgr.flags` did `from ...single_type_kv_cache_manager
import _flag` at module top, so kvmgr could not even IMPORT on a vLLM without
the DSpark recipe's patch6. These are the assertions that make "standalone" a
tested claim rather than a README sentence, in both directions:

  pristine  kvmgr imports and registers with no patch6 symbol reachable; the
            kill switch reaches upstream's own cap; an unknown group count
            refuses to guess.
  dspark    the bridge shares one flags cache, carries an override into the
            fork's own readers (and is PROVEN to be what carries it, by
            uninstalling it), declares the fork's knobs, and does not stack.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

PASS = 0
FAIL: list[str] = []


def check(name: str, cond: bool, detail="") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"PASS  {name}" + (f"  [{detail}]" if detail != "" else ""))
    else:
        FAIL.append(name)
        print(f"FAIL  {name}  {detail}")


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r} want {want!r}")


H = harness.load()
stm = H.mgr_mod
iface = sys.modules["vllm.v1.kv_cache_interface"]
F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv10-"), "flags.json"))

import importlib  # noqa: E402

from kvmgr import flags  # noqa: E402

# NOT `from kvmgr import register`: the package re-exports the FUNCTION under
# that name, so the attribute is not the module.
register = importlib.import_module("kvmgr.register")
from kvmgr.compat import dspark as fork  # noqa: E402
from kvmgr.manager import DSparkMLAAttentionManager  # noqa: E402

PATCH6_ONLY = ("_flag", "_flags", "_protected_fraction", "_FLAGS_PATH",
               "set_pending_block_ids_provider")

print(f"== mode: {harness.MODE} ==")

# ---------------------------------------------------------------------------
print("\n== (1) no module of kvmgr's CORE imports the manager module's privates ==")
# A source scan, so it holds in BOTH modes: the core must never reach for a
# patch6 symbol, whichever vLLM is loaded. compat/ is the one place allowed to.
import glob  # noqa: E402

_core = [p for p in glob.glob(os.path.join(harness.KVMGR_DIR, "**", "*.py"),
                              recursive=True)
         if os.sep + "compat" + os.sep not in p]
check("found the core modules", len(_core) >= 10, len(_core))
_offenders = []
for _p in _core:
    _src = open(_p).read()
    for _sym in PATCH6_ONLY:
        if f"single_type_kv_cache_manager import {_sym}" in _src or \
           f"stm.{_sym}" in _src:
            _offenders.append(f"{os.path.relpath(_p, harness.KVMGR_DIR)}:{_sym}")
eq("*** no core module names a patch6-only symbol ***", _offenders, [])

if not harness.DSPARK:
    # -----------------------------------------------------------------------
    print("\n== (2) PRISTINE: the claim itself ==")
    _reachable = [s for s in PATCH6_ONLY if hasattr(stm, s)]
    eq("*** no patch6 symbol is reachable on the loaded manager module ***",
       _reachable, [])
    check("the bridge finds no fork", fork.detect() is None)

    register._REGISTERED = False
    register.register()
    check("*** register() succeeds with no patch6 ***",
          stm.spec_manager_map[iface.MLAAttentionSpec] is DSparkMLAAttentionManager,
          stm.spec_manager_map[iface.MLAAttentionSpec])
    eq("...and still no patch6 symbol appeared", [s for s in PATCH6_ONLY
                                                 if hasattr(stm, s)], [])
    _fork_declared = sorted(set(fork.FORK_DEFAULTS) & set(flags.DEFAULTS))
    eq("the fork's knobs are NOT declared -- they would govern nothing here",
       _fork_declared, [])

    # -----------------------------------------------------------------------
    print("\n== (3) PRISTINE: the kill switch reaches UPSTREAM's own cap ==")
    pool = harness.make_pool()
    mgrs = harness.make_live_groups(pool)
    m0 = mgrs[0]
    check("group 0 is our manager", isinstance(m0, DSparkMLAAttentionManager),
          type(m0).__name__)
    # upstream: 2 x cdiv(max_model_len, block_size), no fraction at all
    _upstream = 2 * -(-harness.LIVE_MAX_MODEL_LEN // 256)
    F.write(kvmgr_enabled=0, protected_fraction=0.10)
    eq("*** off: upstream's cap exactly, the fraction ignored ***",
       m0._max_protected_prompt_blocks(), _upstream)

    DSparkMLAAttentionManager.group_count = 5
    DSparkMLAAttentionManager.group_block_sizes = harness.LIVE_BLOCK_SIZES
    DSparkMLAAttentionManager.group_sliding_windows = harness.LIVE_SLIDING_WINDOWS
    F.write(kvmgr_enabled=1, protected_fraction=0.75, protect_alloc="equal")
    eq("on, geometry known: kvmgr's equal split works on pristine too",
       m0._max_protected_prompt_blocks(), 3643)
    F.write(kvmgr_enabled=1, protected_fraction=0.75, protect_alloc="by_need")
    eq("...and by_need", m0._max_protected_prompt_blocks(), 8192)

    # -----------------------------------------------------------------------
    print("\n== (4) PRISTINE: an unknown group count REFUSES to guess ==")
    DSparkMLAAttentionManager.group_count = None
    if hasattr(stm.SingleTypeKVCacheManager, "_num_managers"):
        delattr(stm.SingleTypeKVCacheManager, "_num_managers")
    m0._kvmgr_geometry_refused = False
    harness.clear_log()
    F.write(kvmgr_enabled=1, protected_fraction=0.10, protect_alloc="equal")
    eq("*** no geometry and no counter: upstream's cap, not a guessed one ***",
       m0._max_protected_prompt_blocks(), _upstream)
    check("...and it says so", "group count unknown" in harness.log_text(),
          harness.log_text()[-200:])
    harness.clear_log()
    m0._max_protected_prompt_blocks()
    check("...once, not per call", "group count unknown" not in harness.log_text())

else:
    # -----------------------------------------------------------------------
    print("\n== (2) DSPARK: the bridge is installed and shares one cache ==")
    check("the bridge finds patch6", fork.detect() is stm)
    check("*** patch6's _flags IS kvmgr's reader -- one cache over one file ***",
          stm._flags is flags.file_flags)
    check("the fork's knobs are declared",
          set(fork.FORK_DEFAULTS) <= set(flags.DEFAULTS),
          sorted(set(fork.FORK_DEFAULTS) - set(flags.DEFAULTS)))

    # -----------------------------------------------------------------------
    print("\n== (3) DSPARK: an override REACHES the fork's own reader ==")
    flags.reset_for_tests()
    F.write(pin_static_head=0, head_pin_fraction=0.05)
    eq("file says off, patch6 reads off", stm._flag("pin_static_head", False), False)
    flags.set_override("pin_static_head", True)
    eq("*** an override reaches patch6's _flag (MNT-9's reader) ***",
       stm._flag("pin_static_head", False), True)
    flags.set_override("head_pin_fraction", "0.25")
    eq("...coerced to the type patch6 asks for", stm._flag("head_pin_fraction", 0.05), 0.25)
    flags.set_override("protected_fraction", 0.42)
    eq("*** and to MNT-6a's own fraction, so policy-off honours it too ***",
       stm._protected_fraction(), 0.42)

    # -----------------------------------------------------------------------
    print("\n== (4) DSPARK: the bridge is WHAT carries it -- remove it and it stops ==")
    fork.uninstall()
    eq("*** unbridged, patch6 cannot see the override (the old defect) ***",
       stm._flag("pin_static_head", False), False)
    check("...and has its own reader back", stm._flags is not flags.file_flags)
    check("reinstall", fork.install())
    eq("bridged again, it sees it", stm._flag("pin_static_head", False), True)
    flags.clear_override()
    eq("cleared: the file governs again", stm._flag("pin_static_head", False), False)

    # -----------------------------------------------------------------------
    print("\n== (5) DSPARK: install is idempotent and never stacks ==")
    _first = stm._kvmgr_originals["_flag"]
    fork.install()
    fork.install()
    check("*** the saved original is still patch6's, not a bridge ***",
          stm._kvmgr_originals["_flag"] is _first
          and "bridge" not in (_first.__doc__ or ""), _first.__doc__)
    flags.set_override("pin_static_head", True)
    fork.uninstall()
    eq("one uninstall fully unwinds three installs",
       stm._flag("pin_static_head", False), False)
    fork.install()
    flags.clear_override()

# ---------------------------------------------------------------------------
print("\n== (6) BOTH modes: the frozen spec name and its generic alias ==")
harness.load_kv_offload()
from vllm.v1.kv_offload.factory import OffloadingSpecFactory  # noqa: E402

from kvmgr import diskspec, stats  # noqa: E402

# register() returns early once our manager is in the map (it is, from (2)),
# so start from the stock map -- what a fresh process with kv_offload sees.
stm.spec_manager_map[iface.MLAAttentionSpec] = stm.MLAAttentionManager
register._REGISTERED = False
register.register()
_reg_specs = OffloadingSpecFactory._registry
check("the frozen name is registered", "DSparkDiskOffloadingSpec" in _reg_specs)
check("the generic alias is registered", "KvmgrDiskOffloadingSpec" in _reg_specs)
if all(n in _reg_specs for n in diskspec.SPEC_NAMES):
    check("*** both resolve to the SAME class ***",
          _reg_specs["KvmgrDiskOffloadingSpec"]() is
          _reg_specs["DSparkDiskOffloadingSpec"]() is diskspec.DSparkDiskOffloadingSpec)
_saved_ktc = os.environ.get("KV_TRANSFER_CONFIG")
for _n in diskspec.SPEC_NAMES:
    os.environ["KV_TRANSFER_CONFIG"] = (
        '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":'
        '{"spec_name":"%s"}}' % _n)
    check(f"/kv reports the tier configured when spec_name is {_n}",
          stats._disk_spec_is_live() is True)
os.environ["KV_TRANSFER_CONFIG"] = (
    '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":'
    '{"spec_name":"CPUOffloadingSpec"}}')
check("...and NOT for somebody else's spec", stats._disk_spec_is_live() is False)
os.environ.pop("KV_TRANSFER_CONFIG", None)
if _saved_ktc is not None:
    os.environ["KV_TRANSFER_CONFIG"] = _saved_ktc

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
