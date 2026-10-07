"""test_phase4.py -- the hybrid-offload shim, exercised against a stub that
asserts exactly what the real TieringOffloadingSpec asserts.

    python3.12 local-ops/kvharness/test_phase4.py

The real offload machinery needs CUDA, so the stub reproduces the ONE line that
blocks us (`assert len(self.gpu_block_size) == 1`) plus the two fields the shim
reasons about. That is enough to pin every branch, because the shim's whole job
is to decide WHETHER to call through.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

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


harness.load()

# --- a stub standing in for vllm.v1.kv_offload.tiering.spec ----------------
CALLS: list[tuple] = []


class StubSpec:
    """Asserts exactly what the real get_manager asserts, at the same point."""

    def __init__(self, sizes, factor=1):
        self.gpu_block_size = tuple(sizes)
        self.block_size_factor = factor

    def get_manager(self):
        # THE REAL LINE, tiering/spec.py:113
        assert len(self.gpu_block_size) == 1
        CALLS.append(("built", self.gpu_block_size))
        return "manager"


def install_stub():
    for name in ("vllm.v1.kv_offload", "vllm.v1.kv_offload.tiering",
                 "vllm.v1.kv_offload.tiering.spec"):
        m = types.ModuleType(name)
        sys.modules[name] = m
        parent, _, leaf = name.rpartition(".")
        setattr(sys.modules[parent], leaf, m)
    sys.modules["vllm.v1.kv_offload.tiering.spec"].TieringOffloadingSpec = StubSpec


install_stub()

from kvmgr import flags  # noqa: E402
import importlib  # noqa: E402

offload = importlib.import_module("kvmgr.offload")

F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kvmgr4-"), "flags.json"))
LIVE = (256, 64, 64, 4, 8)

# ---------------------------------------------------------------------------
# 1. WITHOUT the shim, our real layout raises -- the baseline this exists for
# ---------------------------------------------------------------------------
try:
    StubSpec(LIVE).get_manager()
    check("the stub reproduces the blocker", False, "it did NOT raise")
except AssertionError:
    check("the stub reproduces the blocker: our (256,64,64,4,8) raises", True)
eq("a uniform layout is fine without any shim", StubSpec((256,)).get_manager(),
   "manager")

# ---------------------------------------------------------------------------
# 2. INSTALLING the shim is idempotent and total
# ---------------------------------------------------------------------------
check("install() reports success", offload.install() is True)
check("install() is idempotent", offload.install() is True)
check("it marks the wrapper so it is not double-wrapped",
      getattr(StubSpec.get_manager, "_kvmgr_hybrid", False) is True)

# ---------------------------------------------------------------------------
# 3. OFF BY DEFAULT -- the assert still bites
# ---------------------------------------------------------------------------
flags.reset_for_tests()
F.remove()
eq("offload_allow_hybrid defaults to False", flags.raw("offload_allow_hybrid"),
   False)
try:
    StubSpec(LIVE).get_manager()
    check("with the flag OFF the assert still raises", False, "it did not")
except AssertionError:
    check("with the flag OFF the assert still raises -- opt-in means opt-in",
          True)

# ---------------------------------------------------------------------------
# 4. ON: it relaxes, and RESTORES the tuple afterwards
# ---------------------------------------------------------------------------
CALLS.clear()
flags.reset_for_tests()
F.write(offload_allow_hybrid=1)
spec = StubSpec(LIVE)
eq("with the flag ON the manager is built", spec.get_manager(), "manager")
eq("the original was called with a 1-tuple", CALLS[-1], ("built", (256,)))
eq("*** and gpu_block_size is RESTORED afterwards ***", spec.gpu_block_size,
   LIVE)
check("the first element is what is presented (group 0, the one that binds)",
      CALLS[-1][1] == (LIVE[0],))

# *** A DOWNSTREAM FAILURE MUST PROPAGATE, AND original() MUST BE CALLED ONCE. ***
# The first version of the shim wrapped the whole body in `except Exception` and
# called original() from the handler -- so anything the real get_manager raised
# was caught, logged as "the shim failed", and get_manager was called a SECOND
# time. That double-calls an allocator and buries the real error under whatever
# the retry raises. This is the assertion that found it.
CALLS.clear()
_depth = {"n": 0}


class Raiser(StubSpec):
    def get_manager(self):
        assert len(self.gpu_block_size) == 1
        _depth["n"] += 1
        raise RuntimeError("downstream failure")


sys.modules["vllm.v1.kv_offload.tiering.spec"].TieringOffloadingSpec = Raiser
import kvmgr.offload as _off  # noqa: E402

_off.install()
r = Raiser(LIVE)
flags.reset_for_tests()
F.write(offload_allow_hybrid=1)
try:
    r.get_manager()
    check("a downstream failure PROPAGATES", False, "it was swallowed")
except RuntimeError as e:
    check("*** a downstream failure PROPAGATES, unchanged ***",
          str(e) == "downstream failure", f"got {e!r}")
except AssertionError:
    check("a downstream failure PROPAGATES, unchanged", False,
          "it was retried and surfaced as AssertionError -- the double-call bug")
eq("*** original() was called EXACTLY ONCE ***", _depth["n"], 1)
eq("and gpu_block_size is still restored after a raise", r.gpu_block_size, LIVE)
sys.modules["vllm.v1.kv_offload.tiering.spec"].TieringOffloadingSpec = StubSpec
_off.install()

# ---------------------------------------------------------------------------
# 5. COARSENING IS REFUSED -- base.py's own rule, which we do NOT override
# ---------------------------------------------------------------------------
flags.reset_for_tests()
harness.clear_log()
F.write(offload_allow_hybrid=1)
try:
    StubSpec(LIVE, factor=4).get_manager()
    check("block_size_factor != 1 is REFUSED", False, "it was relaxed anyway")
except AssertionError:
    check("*** block_size_factor != 1 is REFUSED, not relaxed ***", True,
          "coarsening genuinely needs one block size")
txt = harness.log_text()
check("and it says why, naming the knob",
      "NOT relaxing" in txt and "block_size_factor=4" in txt
      and "kv_connector_extra_config" in txt, txt[:200])

# ---------------------------------------------------------------------------
# 6. IT ANNOUNCES ITSELF, and names OUTPUT EQUIVALENCE as the gate
# ---------------------------------------------------------------------------
flags.reset_for_tests()
harness.clear_log()
F.write(offload_allow_hybrid=1)
StubSpec(LIVE).get_manager()
txt = harness.log_text()
check("the relaxation is logged at WARNING with the block sizes",
      "relaxing" in txt and "256" in txt, txt[:200])
check("*** and the log names OUTPUT EQUIVALENCE as the gate, not the boot ***",
      "OUTPUT EQUIVALENCE" in txt,
      "a successful boot does not show the per-group slicing is right")

# a uniform layout is passed straight through, no log, no relaxation
harness.clear_log()
eq("a uniform layout still just works", StubSpec((256,)).get_manager(),
   "manager")
check("and nothing is logged about relaxing it",
      "relaxing" not in harness.log_text())

# ---------------------------------------------------------------------------
# 7. IT CANNOT TAKE THE ENGINE DOWN
# ---------------------------------------------------------------------------
flags.reset_for_tests()
F.write(offload_allow_hybrid=1)


class Weird:
    """No gpu_block_size, no block_size_factor at all."""

    def get_manager(self):
        return "manager"


sys.modules["vllm.v1.kv_offload.tiering.spec"].TieringOffloadingSpec = Weird
_off.install()
eq("a spec missing both attributes falls through cleanly",
   Weird().get_manager(), "manager")
sys.modules["vllm.v1.kv_offload.tiering.spec"].TieringOffloadingSpec = StubSpec

# and with no offload machinery at all, install() is a quiet no-op
for name in ("vllm.v1.kv_offload.tiering.spec", "vllm.v1.kv_offload.tiering",
             "vllm.v1.kv_offload"):
    sys.modules.pop(name, None)
harness.clear_log()
eq("install() returns False when kv_offload is absent", _off.install(), False)
eq("and logs nothing about it (it is the normal case)",
   harness.log_text(), "")

# ---------------------------------------------------------------------------
# 8. THE KNOB IS IN THE TABLE, so the hot-config suite covers it
# ---------------------------------------------------------------------------
check("offload_allow_hybrid is in DEFAULTS",
      "offload_allow_hybrid" in flags.DEFAULTS)
eq("and defaults to False", flags.DEFAULTS["offload_allow_hybrid"], False)
check("it is NOT in BLOCKED (it is buildable, unlike the disk_* knobs)",
      "offload_allow_hybrid" not in flags.BLOCKED)

print()
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
