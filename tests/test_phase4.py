"""test_phase4.py -- the hybrid-offload shim, exercised against a stub that
asserts exactly what the real TieringOffloadingSpec asserts.

    python3.12 packages/vllm-kvmgr/tests/test_phase4.py

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
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the PACKAGE root
sys.path.insert(0, os.path.join(REPO, "src"))

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

offload = importlib.import_module("kvmgr.diag.tiering")
stats = importlib.import_module("kvmgr.stats")

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
import kvmgr.diag.tiering as _off  # noqa: E402

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

# ---------------------------------------------------------------------------
# 9. THE O_DIRECT SHIM -- what actually stopped anything reaching disk
# ---------------------------------------------------------------------------
# Measured: our per-block CPU page is 2,103,552 bytes; mod 512 = 256, and the
# device logical sector is 512. O_DIRECT needs an aligned length, so every write
# failed with EINVAL -- 13,519 times in one boot, against 1 file on disk.
eq("the real block size is NOT 512-aligned (the whole cause)",
   2103552 % 512, 256)
eq("nor 4096-aligned", 2103552 % 4096, 2304)

# Section 7 removed the kv_offload chain from sys.modules, so rebuild it --
# otherwise this section fails for the wrong reason.
for _n in ("vllm.v1.kv_offload", "vllm.v1.kv_offload.tiering",
           "vllm.v1.kv_offload.tiering.fs", "vllm.v1.kv_offload.tiering.fs.io"):
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        sys.modules[_n] = _m
        _par, _, _leaf = _n.rpartition(".")
        setattr(sys.modules[_par], _leaf, _m)
_fsio = sys.modules["vllm.v1.kv_offload.tiering.fs.io"]

PLATFORM_ODIRECT = 0x4000          # what Linux os.O_DIRECT actually is
_fsio.O_DIRECT = PLATFORM_ODIRECT
_opens: list = []


def _store_block(*a, **kw):
    _opens.append(("store", _fsio.O_DIRECT))
    return "stored"


def _load_block(*a, **kw):
    _opens.append(("load", _fsio.O_DIRECT))
    return "loaded"


_fsio.store_block = _store_block
_fsio.load_block = _load_block

offload._real_odirect = None
check("install_io() reports success", offload.install_io() is True)
check("install_io() is idempotent", offload.install_io() is True)

flags.reset_for_tests()
F.write(offload_odirect=0)
_opens.clear()
eq("a store still returns normally", _fsio.store_block("x"), "stored")
eq("*** and O_DIRECT was 0 for that write -- which is what makes a "
   "2,103,552-byte block writable at all ***", _opens[-1], ("store", 0))
eq("the load path too", _fsio.load_block("x"), "loaded")
eq("also with O_DIRECT 0", _opens[-1], ("load", 0))

# HOT: flip it back on without a restart
F.write(offload_odirect=1)
_opens.clear()
_fsio.store_block("x")
eq("*** offload_odirect is HOT -- O_DIRECT comes back with no restart ***",
   _opens[-1], ("store", PLATFORM_ODIRECT))
F.write(offload_odirect=0)
_fsio.store_block("x")
eq("and goes away again", _opens[-1], ("store", 0))

eq("the default is OFF, i.e. dropped -- the engine cannot write otherwise",
   flags.DEFAULTS["offload_odirect"], False)

# it must never fail a transfer over a flag read
_bad = {"n": 0}
_orig_raw = flags.raw


def _boom(name):
    if name == "offload_odirect":
        _bad["n"] += 1
        raise RuntimeError("flags exploded")
    return _orig_raw(name)


flags.raw = _boom  # type: ignore[assignment]
try:
    eq("a flag-read failure does NOT fail the block transfer",
       _fsio.store_block("x"), "stored")
    check("and it was actually exercised", _bad["n"] > 0)
finally:
    flags.raw = _orig_raw  # type: ignore[assignment]

# absent fs io module -> quiet no-op
for _n in ("vllm.v1.kv_offload.tiering.fs.io", "vllm.v1.kv_offload.tiering.fs"):
    sys.modules.pop(_n, None)
offload._real_odirect = None
harness.clear_log()
eq("install_io() returns False when the fs tier is absent",
   offload.install_io(), False)

# --------------------------------------------------------------------------
# (4c) the disk ROUND-TRIP VERIFIER
#
# WHY IT EXISTS: the output-equivalence gate was worthless. Four runs of one
# prompt on ONE unchanged config gave two different answers (156 vs 157 tokens,
# differing in capitalisation), because num_spec_tokens=5 plus async scheduling
# moves the reduction order and flips a near-tie argmax. A byte compare needs no
# determinism.
#
# Both directions are exercised, because a verifier that cannot FAIL is the
# vacuous test this suite has already caught twice.
# --------------------------------------------------------------------------
print()
print("== (4c) the disk round-trip verifier ==")

for _n in ("vllm.v1.kv_offload", "vllm.v1.kv_offload.tiering",
           "vllm.v1.kv_offload.tiering.fs", "vllm.v1.kv_offload.tiering.fs.io"):
    if _n not in sys.modules:
        _m = types.ModuleType(_n)
        sys.modules[_n] = _m
        _par, _, _leaf = _n.rpartition(".")
        setattr(sys.modules[_par], _leaf, _m)
_fsio = sys.modules["vllm.v1.kv_offload.tiering.fs.io"]
_fsio.O_DIRECT = 0

# A byte-faithful fake disk, and a corrupting one. `store_block` is handed a
# buffer plus an OFFSET and a LENGTH, exactly as the real fs tier is.
_disk: dict = {}
_corrupt = {"on": False}


def _store_block(path, buffer, offset, block_size, *a, **kw):
    b = bytes(buffer.cast("B")[offset:offset + block_size])
    if _corrupt["on"]:
        b = b[:-1] + bytes([(b[-1] + 1) & 0xFF])     # flip the LAST byte
    _disk[path] = b
    return None


def _load_block(path, view, offset, block_size, *a, **kw):
    view.cast("B")[offset:offset + block_size] = _disk[path]
    return None


_fsio.store_block = _store_block
_fsio.load_block = _load_block
_fsio._kvmgr_verify_wrapped = False

check("install_verify() reports success", offload.install_verify() is True)
check("install_verify() is idempotent", offload.install_verify() is True)

import array
_src = array.array("B", bytes(range(256)) * 16)      # 4096 bytes
_buf = memoryview(_src)

flags.reset_for_tests()
F.write(offload_verify=0)
stats.reset_for_tests()
_fsio.store_block("/d/a.bin", _buf, 0, 4096)
eq("OFF by default: nothing verified", stats.group(0).verify_ok, 0)
eq("and no mismatch recorded either", stats.group(0).verify_mismatch, 0)
check("but the block still reached the fake disk", "/d/a.bin" in _disk)

F.write(offload_verify=1)
stats.reset_for_tests()
_fsio.store_block("/d/b.bin", _buf, 0, 4096)
eq("*** HOT: a faithful round trip VERIFIES, with no restart ***",
   stats.group(0).verify_ok, 1)
eq("and nothing is reported as a mismatch", stats.group(0).verify_mismatch, 0)

# a NON-ZERO offset, because the real tier slices into one big page and an
# off-by-one there is exactly the class of bug this is for
stats.reset_for_tests()
_fsio.store_block("/d/c.bin", _buf, 1024, 2048)
eq("a mid-buffer slice verifies too", stats.group(0).verify_ok, 1)
eq("and the right 2,048 bytes landed", _disk["/d/c.bin"], bytes(_src[1024:3072]))

# *** THE DIRECTION THAT MATTERS: it must be able to FAIL. ***
_corrupt["on"] = True
stats.reset_for_tests()
harness.clear_log()
_fsio.store_block("/d/d.bin", _buf, 0, 4096)
_corrupt["on"] = False
eq("*** a ONE-BYTE corruption is CAUGHT ***",
   stats.group(0).verify_mismatch, 1)
eq("and is not also counted as a success", stats.group(0).verify_ok, 0)
check("the mismatch is LOGGED, loudly",
      "DISK ROUND TRIP MISMATCH" in harness.log_text())
check("and the log names where it diverged",
      "first differing byte at 4095" in harness.log_text())

# a read that EXPLODES must not break the write it was only observing
def _boom_load(*a, **kw):
    raise OSError("read failed")


_keep = _fsio.load_block
_fsio.load_block = _boom_load
stats.reset_for_tests()
try:
    _fsio.store_block("/d/e.bin", _buf, 0, 4096)
    check("a failing read-back does NOT break the store", "/d/e.bin" in _disk)
    eq("and records neither outcome", (stats.group(0).verify_ok,
                                      stats.group(0).verify_mismatch), (0, 0))
finally:
    _fsio.load_block = _keep

check("offload_verify is in DEFAULTS so the hot-config suite covers it",
      "offload_verify" in flags.DEFAULTS)
eq("and it is OFF by default -- it doubles store I/O",
   flags.DEFAULTS["offload_verify"], False)

check("offload_odirect is in DEFAULTS so the hot-config suite covers it",
      "offload_odirect" in flags.DEFAULTS)

print()
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
