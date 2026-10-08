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

print("\n== (8) apply_knob / clear_knob: the per-process RPC body ==")

from kvmgr import control  # noqa: E402

flags.reset_for_tests()
F.write()
a = control.apply_knob("evict_order", "lru_tail_first")
check("a good set reports ok", a["ok"] is True)
eq("...the value it set", a["set_to"], "lru_tail_first")
eq("...what it displaced", a["was"], "fifo")
eq("...and the winning source", a["source"], "override")

bad = control.apply_knob("not_a_knob", 1)
check("*** an unknown knob does NOT raise across the RPC boundary ***",
      bad["ok"] is False, "a raise there is an engine-core traceback")
check("...it names the knob", "not_a_knob" in bad["error"])
inv = control.apply_knob("disk_store_paused", "maybe")
check("a bad boolean is refused the same way, not coerced",
      inv["ok"] is False and "not a boolean" in inv["error"])

c = control.clear_knob("evict_order")
check("clear reports ok", c["ok"] is True)
eq("...naming what it dropped", c["cleared"], ["evict_order"])
eq("...and what governs now", c["source"], "default")
check("clearing an unknown knob reports, does not raise",
      control.clear_knob("nope")["ok"] is False)

print("\n== (9) THE SPLIT VIEW is the dangerous failure, so it is reported ==")

# The engine core applies a kvmgr_* knob inside the wrap and the workers apply
# it over RPC. If the fan-out fails after the local apply, the knob is set HERE
# and not THERE -- and a 200 with a local success in it would hide exactly
# that. Modelled directly, because no offline harness has an executor.
calls = []


class _FakeClient:
    def __init__(self, fail=False):
        self.fail = fail

    async def collective_rpc(self, method, args=(), kwargs=None):
        calls.append((method, kwargs))
        if self.fail:
            raise RuntimeError("worker handshake timed out")
        return [{"ok": True, "rank": 0}, {"ok": True, "rank": 1}]


import asyncio  # noqa: E402


async def _drive(client, method, kwargs):
    """The route's fan-out, lifted: apply locally, then fan out, then report."""
    local = control._apply_locally_reported(method, kwargs)
    try:
        ranks = await client.collective_rpc(method=method, args=(), kwargs=kwargs)
    except Exception as e:
        return {"ok": False, "api_server": local, "error": str(e)}
    return {"ok": True, "api_server": local, "ranks": ranks}


flags.reset_for_tests()
ok = asyncio.get_event_loop().run_until_complete(
    _drive(_FakeClient(), "kvmgr_set_knob",
           {"name": "protect_alloc", "value": "by_need"}))
check("a good fan-out is ok", ok["ok"] is True)
eq("...and reports BOTH ranks", len(ok["ranks"]), 2)
eq("...having applied locally too", ok["api_server"]["set_to"], "by_need")
eq("...and the RPC carried the method vLLM resolves on the Worker",
   calls[-1][0], "kvmgr_set_knob")

flags.reset_for_tests()
split = asyncio.get_event_loop().run_until_complete(
    _drive(_FakeClient(fail=True), "kvmgr_set_knob",
           {"name": "protect_alloc", "value": "by_need"}))
check("*** a FAILED fan-out reports NOT ok, never a 200 with a local win ***",
      split["ok"] is False)
check("...naming the cause", "handshake" in split["error"])
check("...while still saying what the local apply did, so the split is visible",
      split["api_server"]["set_to"] == "by_need")

print("\n== (10) _apply_locally accepts both positional and keyword args ==")

# The engine-core wrap sees whatever the caller passed. Upstream's own
# /collective_rpc route sends args as a LIST and kwargs as a dict, and our
# route sends kwargs -- so both shapes have to work or the core silently skips
# its own apply and we get the split view we just tested for.
for label, args, kwargs in (("kwargs", (), {"name": "kvmgr_log_level", "value": 4}),
                            ("positional", ("kvmgr_log_level", 4), {})):
    flags.reset_for_tests()
    control._apply_locally("kvmgr_set_knob", args, kwargs)
    eq(f"the core applies a {label} call", flags.raw("kvmgr_log_level"), 4)

print("\n== (11) rebind: __class__ swap moves EVERY method, including new ones ==")

# Per-method rebinding silently misses a method the reload ADDED, which is
# exactly the edit a reload is usually for. The __class__ swap cannot.
import types  # noqa: E402

mod = types.ModuleType("fake_live_mod")
code_v1 = "class Thing:\n    def verdict(self):\n        return 'v1'\n"
exec(compile(code_v1, "<v1>", "exec"), mod.__dict__)
sys.modules["fake_live_mod"] = mod
obj = mod.Thing()
obj.state = "kept"
eq("v1 behaviour", obj.verdict(), "v1")

# a reload that CHANGES one method and ADDS another
code_v2 = ("class Thing:\n    def verdict(self):\n        return 'v2'\n"
           "    def added_later(self):\n        return 'new'\n")
exec(compile(code_v2, "<v2>", "exec"), mod.__dict__)
obj.__class__ = mod.Thing
eq("*** the changed method moved ***", obj.verdict(), "v2")
eq("*** and the ADDED one exists, which per-method rebinding would miss ***",
   obj.added_later(), "new")
eq("...and the instance kept its state", obj.state, "kept")
del sys.modules["fake_live_mod"]

print("\n== (12) the attach gate: OFF means ABSENT, not refusing ==")

flags.reset_for_tests()
F.write()


class _App:
    def __init__(self):
        self.included = []

    def include_router(self, r):
        self.included.append(r)


app = _App()
control.attach_router(app)
eq("*** off by default: nothing is attached, so the routes are ABSENT ***",
   app.included, [])
eq("...and the default is off, because the plane shares the engine's port",
   bool(flags.raw("kvmgr_control_plane")), False)

eq("the gate knob is in DEFAULTS, so the hot-config suite covers it",
   "kvmgr_control_plane" in flags.DEFAULTS, True)

print("\n== (13) every ROUTE ANNOTATION must resolve from MODULE globals ==")

# *** THE BUG THIS CATCHES COST A WHOLE DEPLOY, and nothing else could see it:
# all five routes registered and every one answered 400. ***
#
#   {'loc': ('query', 'raw_request'), 'msg': 'Field required'}
#
# `from __future__ import annotations` turns every annotation into a STRING,
# and FastAPI resolves it with get_type_hints() against the function's MODULE
# GLOBALS -- not its closure. `Request` was imported inside `_router()`, so
# `raw_request: Request` read as an unresolvable name and FastAPI fell back to
# treating it as a query parameter. Routes that exist and cannot be called: a
# mechanism present, correct-looking, producing nothing.
#
# So: for every annotated parameter of every nested route handler, the
# annotation's root name must be bound at MODULE level. Checked with ast, so it
# needs no fastapi and no box.
import ast as _ast  # noqa: E402

_cpath = os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr", "control.py")
_tree = _ast.parse(open(_cpath).read())

_module_names = set()
for node in _tree.body:
    if isinstance(node, (_ast.Import, _ast.ImportFrom)):
        for a in node.names:
            _module_names.add((a.asname or a.name).split(".")[0])
    elif isinstance(node, _ast.Assign):
        for t in node.targets:
            if isinstance(t, _ast.Name):
                _module_names.add(t.id)
    elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
        _module_names.add(node.name)
    elif isinstance(node, _ast.Try):
        # the guarded fastapi import lives in a try/except at module level,
        # and both branches bind the names -- which is the whole point of it
        for sub in list(node.body) + [x for h in node.handlers for x in h.body]:
            if isinstance(sub, (_ast.Import, _ast.ImportFrom)):
                for a in sub.names:
                    _module_names.add((a.asname or a.name).split(".")[0])
            elif isinstance(sub, _ast.Assign):
                for t in sub.targets:
                    if isinstance(t, _ast.Name):
                        _module_names.add(t.id)

check("the guarded fastapi import binds Request at MODULE level",
      "Request" in _module_names,
      "if this fails, every route 400s on a missing query param")
for n in ("APIRouter", "JSONResponse"):
    check(f"...and {n}", n in _module_names)


def _root(ann):
    while isinstance(ann, _ast.Subscript):
        ann = ann.value
    while isinstance(ann, _ast.Attribute):
        ann = ann.value
    return ann.id if isinstance(ann, _ast.Name) else None


_handlers, _bad = [], []
for node in _ast.walk(_tree):
    if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
        continue
    if not any(isinstance(d, _ast.Attribute) and d.attr in ("get", "post")
               or (isinstance(d, _ast.Call) and isinstance(d.func, _ast.Attribute)
                   and d.func.attr in ("get", "post"))
               for d in node.decorator_list):
        continue
    _handlers.append(node.name)
    for arg in node.args.args + node.args.kwonlyargs:
        if arg.annotation is None:
            continue
        r = _root(arg.annotation)
        if r and r not in _module_names:
            _bad.append(f"{node.name}({arg.arg}: {r})")

check("the lint actually found the route handlers", len(_handlers) >= 5,
      f"{len(_handlers)}: {_handlers}")
eq("*** every route annotation resolves from module globals ***", _bad, [])

# And prove the lint can FAIL, because a lint that cannot is coverage theatre.
_canary = _ast.parse(
    "from __future__ import annotations\n"
    "def _r():\n"
    "    from fastapi import Request\n"
    "    @router.get('/x')\n"
    "    async def h(raw: Request):\n"
    "        return 1\n")
_cnames = {n.name for n in _canary.body
           if isinstance(n, (_ast.FunctionDef, _ast.AsyncFunctionDef))}
_cbad = []
for node in _ast.walk(_canary):
    if isinstance(node, _ast.AsyncFunctionDef) and node.decorator_list:
        for arg in node.args.args:
            if arg.annotation is not None and _root(arg.annotation) not in _cnames:
                _cbad.append(arg.arg)
check("*** ...and the lint CATCHES a locally-imported annotation ***",
      _cbad == ["raw"], f"{_cbad} -- a lint that cannot fail is theatre")

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
