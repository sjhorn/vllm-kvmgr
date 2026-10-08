"""test_phase6.py -- the HTTP control plane's foundation: the in-memory override.

    python3.12 local-ops/kvharness/test_phase6.py

*** THE CONSTRAINT THIS FILE ENFORCES: a hot knob must not DEPEND on disk or
container injection. *** Every knob today is reached through
`/var/tmp/vllm-kv-flags/flags.json`, which works and is why they are all hot --
but as the only way in it is a bind mount (so the control plane depends on the
container spec), it is PER NODE (writing it on the head does not change the
worker), it returns no acknowledgement, and it is not atomic across the three
processes that re-stat it on independent 1s TTLs.

So an explicit set lands in memory, above the env and the file, and the file is
demoted to the boot-time default. Everything here is pure: no vLLM routes, no
HTTP, no box.
"""

from __future__ import annotations

import os
import sys

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


import tempfile  # noqa: E402

harness.load()
from kvmgr import flags  # noqa: E402

FLAGS_DIR = tempfile.mkdtemp(prefix="kvmgr-p6-")
FLAGS_PATH = os.path.join(FLAGS_DIR, "flags.json")
F = harness.Flags(FLAGS_PATH)

print("\n== (1) precedence: override > env > file > default ==")

flags.reset_for_tests()
F.write()
eq("with nothing set, the DEFAULT wins", flags.raw("evict_order"), "fifo")
eq("...and says so", flags.source_of("evict_order"), "default")

F.write(evict_order="lru")
eq("the FILE beats the default", flags.raw("evict_order"), "lru")
eq("...and says so", flags.source_of("evict_order"), "file")

os.environ["VLLM_EVICT_ORDER"] = "lru_tail_first"
flags.reset_for_tests()
eq("the ENV beats the file", flags.raw("evict_order"), "lru_tail_first")
eq("...and says so", flags.source_of("evict_order"), "env")

act = flags.set_override("evict_order", "fifo")
eq("*** the OVERRIDE beats the env pin, which is the reversal Stage 6 is for ***",
   flags.raw("evict_order"), "fifo")
eq("...and says so", flags.source_of("evict_order"), "override")
eq("...and the act reports what it displaced", act["was"], "lru_tail_first")
eq("...and what it now reads", act["now"], "fifo")
eq("...naming the winning source", act["source"], "override")

flags.clear_override("evict_order")
eq("cleared, the env pin governs again", flags.raw("evict_order"), "lru_tail_first")
eq("...and says so", flags.source_of("evict_order"), "env")
del os.environ["VLLM_EVICT_ORDER"]
flags.reset_for_tests()

print("\n== (2) bool('false') IS TRUE, and that would invert the request ==")

# The single most likely way this plane could do the OPPOSITE of what it was
# asked. `?on=false` arrives as the string "false", and truthiness says True.
for word, want in (("true", True), ("True", True), ("1", True), ("yes", True),
                   ("on", True), ("false", False), ("False", False),
                   ("0", False), ("no", False), ("off", False), ("  OFF  ", False)):
    flags.reset_for_tests()
    flags.set_override("disk_store_paused", word)
    eq(f"{word!r} -> {want}", flags.raw("disk_store_paused"), want)

flags.reset_for_tests()
eq("a real bool passes through", flags._coerce("disk_store_paused", True), True)
eq("an int coerces", flags._coerce("disk_store_paused", 0), False)
try:
    flags._coerce("disk_store_paused", "maybe")
    check("*** a word that is not a boolean is REFUSED, not coerced ***", False,
          "it was accepted")
except ValueError as e:
    check("*** a word that is not a boolean is REFUSED, not coerced ***",
          "not a boolean" in str(e) and "invert" in str(e), str(e)[:80])

print("\n== (3) types are coerced to the knob's DECLARED type ==")

flags.reset_for_tests()
flags.set_override("kvmgr_stats_port", "9199")
eq("a numeric string becomes an int", flags.raw("kvmgr_stats_port"), 9199)
check("...really an int, not a string",
      isinstance(flags.raw("kvmgr_stats_port"), int))
flags.set_override("head_pin_fraction", "0.25")
eq("...and a float a float", flags.raw("head_pin_fraction"), 0.25)
flags.set_override("protect_alloc", "by_need")
eq("a string stays a string", flags.raw("protect_alloc"), "by_need")

print("\n== (4) an unknown knob is refused BY NAME, with the list ==")

for fn, label in ((lambda: flags.set_override("nope", 1), "set"),
                  (lambda: flags.clear_override("nope"), "clear")):
    try:
        fn()
        check(f"{label} of an unknown knob is refused", False, "it was accepted")
    except KeyError as e:
        check(f"{label} of an unknown knob is refused", "nope" in str(e))
        check(f"...and {label} lists the knobs that DO exist",
              "evict_order" in str(e), "so the answer is actionable")

print("\n== (5) EVERY knob must be settable through the plane ==")

# The constraint in one assertion: if a knob cannot be set in memory then it
# depends on the file, which is the thing Stage 6 exists to stop.
flags.reset_for_tests()
F.write()
unsettable = []
for name, default in flags.DEFAULTS.items():
    try:
        probe = {bool: True, int: 7, float: 0.5, str: "x"}[type(default)]
        if name == "protect_alloc":
            probe = "by_need"
        elif name == "evict_order":
            probe = "lru"
        flags.set_override(name, probe)
        if flags.source_of(name) != "override":
            unsettable.append((name, "source is not override"))
    except Exception as e:
        unsettable.append((name, str(e)[:40]))
    finally:
        flags.clear_override(name)
eq("*** every knob in DEFAULTS is settable in memory ***", unsettable, [])
eq("...and that is all of them", len(flags.DEFAULTS) > 15, True)

print("\n== (6) clearing, and not leaking between tests ==")

flags.reset_for_tests()
flags.set_override("kvmgr_log_level", 3)
flags.set_override("disk_restore", 1)
eq("clear_override() with no name drops them all",
   sorted(flags.clear_override()), ["disk_restore", "kvmgr_log_level"])
eq("...leaving nothing", flags._OVERRIDE, {})
eq("clearing a knob that was never set reports nothing dropped",
   flags.clear_override("disk_restore"), [])

flags.set_override("kvmgr_log_level", 5)
flags.reset_for_tests()
eq("*** reset_for_tests clears overrides, so a test cannot leak into the next ***",
   flags._OVERRIDE, {})

print("\n== (7) protected_fraction keeps ONE definition, and the override still wins ==")

# This knob is special: raw() delegates to patch6's `_protected_fraction()` so
# there is one precedence implementation rather than two that drift. The
# override has to sit ABOVE that delegation, not inside it.
flags.reset_for_tests()
F.write(protected_fraction=0.75)
eq("the file governs it as before", flags.raw("protected_fraction"), 0.75)
flags.set_override("protected_fraction", "0.40")
eq("*** and an override still outranks it ***",
   flags.raw("protected_fraction"), 0.40)
eq("...reported as an override", flags.source_of("protected_fraction"), "override")
flags.clear_override()
eq("...and clearing returns the delegation", flags.raw("protected_fraction"), 0.75)

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
