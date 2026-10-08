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

print("\n== (14) the reload must not ERASE THE REGISTRY IT READS ==")

# *** MEASURED ON THE BOX 2026-10-08: the first live /kv/reload reported
# `instances: []` on both ranks while a store handler was demonstrably
# running. *** `importlib.reload` RE-EXECUTES the module, so diskio's
# `_LIVE_HANDLERS: list = []` runs again and the list is empty by the time
# live_instances() looks -- the handler objects are still alive and serving,
# and nothing can find them to re-point. Same class as the dashboard's priming
# code being a destructive overwrite: a step that rebuilds what it then
# consumes.
from kvmgr import diskio as _dio  # noqa: E402

_saved_h = list(getattr(_dio, "_LIVE_HANDLERS", []))
try:
    # A REAL module on disk, so the reload path is the real one. A class
    # defined in this test file lives in `__main__`, which importlib cannot
    # re-import -- and that case is asserted separately below, because it is a
    # legitimate one that must DEGRADE rather than fail the whole reload.
    _tmpdir = tempfile.mkdtemp(prefix="kvreload-")
    sys.path.insert(0, _tmpdir)
    open(os.path.join(_tmpdir, "fake_handler_mod.py"), "w").write(
        "class H:\n    def verdict(self):\n        return 'v1'\n")
    import fake_handler_mod  # noqa: E402

    h1, h2 = fake_handler_mod.H(), fake_handler_mod.H()

    class _Unreloadable:  # lives in __main__
        pass

    _dio._LIVE_HANDLERS[:] = [h1, h2, _Unreloadable()]
    eq("three handlers registered before the reload, one unreloadable",
       len(_dio._LIVE_HANDLERS), 3)

    # Change the module on disk, so the rebind has something to move to.
    # *** THE NEW SOURCE MUST DIFFER IN LENGTH, and that cost a confusing
    # failure. *** Python validates its bytecode cache on (mtime, SIZE), and
    # `return 'v1'` and `return 'v2'` are the same number of bytes -- written
    # in the same second, the cache was reused, reload re-executed the OLD
    # bytecode, and the test read `rebound` with v1 behaviour. A test artefact
    # rather than a code bug, but an instructive one: a reload is only as fresh
    # as the import system thinks the file is.
    #
    # The replacement also ADDS a method, so this exercises on the real import
    # path the thing section (11) proves on a synthetic one.
    open(os.path.join(_tmpdir, "fake_handler_mod.py"), "w").write(
        "class H:\n"
        "    def verdict(self):\n"
        "        return 'v2-and-longer-so-the-bytecode-cache-is-invalidated'\n"
        "    def added_later(self):\n"
        "        return 'new'\n")

    rep = control.reload_modules()
    eq("*** it FOUND them all, because it snapshots before reloading ***",
       rep.get("found", {}).get("handlers"), 3)
    eq("...re-pointing all three", len([r for r in rep["rebound"]
                                       if r.startswith("handlers/")]), 3)
    check("*** and the reloadable ones really moved ***",
          h1.verdict().startswith("v2"), h1.verdict())
    eq("...including a method the reload ADDED, on the real import path",
       h1.added_later(), "new")
    check("*** while the unreloadable one DEGRADES and is NAMED ***",
          any("NOT reloadable" in r for r in rep["rebound"]),
          [r for r in rep["rebound"] if "handlers/" in r])
    check("...without failing the whole reload, which is the wrong scope",
          rep.get("ok") is True, str(rep.get("errors"))[:90])
    sys.path.remove(_tmpdir)
    # and the registry must still hold them afterwards, or the NEXT reload
    # finds nothing -- the bug one step along.
    from kvmgr import diskio as _dio2  # re-imported after the reload

    eq("*** and the registry is restored, so a SECOND reload still finds them ***",
       len(_dio2._LIVE_HANDLERS), 3)
    rep2 = control.reload_modules()
    eq("...proved by doing it twice", rep2.get("found", {}).get("handlers"), 3)
finally:
    from kvmgr import diskio as _dio3

    _dio3._LIVE_HANDLERS[:] = _saved_h

# `found` must be in the response at all: `instances: []` was indistinguishable
# from "there was nothing to rebind", which is why the live bug read as normal.
rep3 = control.reload_modules()
check("the response always says what it FOUND, so empty is not ambiguous",
      "found" in rep3 and isinstance(rep3["found"], dict), str(rep3)[:90])

print("\n== (15) the boot sync: the HEAD's file is the cluster's source ==")

# *** MEASURED, NOT FEARED: the flags file is PER NODE and the worker's goes
# stale. *** 2026-10-08, after a /kv/knob/clear, rank 0 reported source=file
# and rank 1 reported source=DEFAULT --
#     head    Oct 8 07:59  12 keys  disk_store_paused=1
#     worker  Oct 7 20:29   9 keys  disk_store_paused ABSENT
# -- so every knob set through the file reached rank 0 only, and a deploy made
# "safe" by pausing the disk store had it paused on rank 0 and RUNNING on rank
# 1. The plane fixes a knob set over HTTP, because that fans out; it does not
# fix the BOOT defaults, which each node reads from its own file.
flags.reset_for_tests()
F.write(evict_order="lru_tail_first", kvmgr_log_level=2)

sent = []


class _Core:
    def collective_rpc(self, method, args=(), kwargs=None):
        sent.append((method, dict(kwargs or {})))
        return [{"ok": True, "rank": 0}, {"ok": True, "rank": 1}]


rep = control.sync_knobs_to_workers(_Core())
pushed = {k["name"]: k["value"] for _, k in sent}
eq("no errors", rep["errors"], [])
eq("*** a knob the FILE set is pushed ***", pushed.get("evict_order"),
   "lru_tail_first")
eq("...and another", pushed.get("kvmgr_log_level"), 2)
check("*** a knob at its DEFAULT is NOT pushed ***",
      "swa_recycle" not in pushed,
      "pushing it would make source_of() say 'override' for a knob nobody set")
eq("...so only what was really set crosses", sorted(pushed), 
   ["evict_order", "kvmgr_log_level"])
check("every push is a kvmgr_set_knob, which is what the Worker resolves",
      all(m == "kvmgr_set_knob" for m, _ in sent))

# an ENV pin must be pushed too, or the worker starts without it
os.environ["VLLM_PROTECT_ALLOC"] = "by_need"
flags.reset_for_tests()
F.write()
sent.clear()
control.sync_knobs_to_workers(_Core())
pushed = {k["name"]: k["value"] for _, k in sent}
eq("*** an ENV pin is pushed as well, or the worker boots without it ***",
   pushed.get("protect_alloc"), "by_need")
del os.environ["VLLM_PROTECT_ALLOC"]
flags.reset_for_tests()

# and a failing RPC must not take the boot down
class _AngryCore:
    def collective_rpc(self, method, args=(), kwargs=None):
        raise RuntimeError("workers not up yet")


F.write(evict_order="lru")
rep2 = control.sync_knobs_to_workers(_AngryCore())
check("*** a failed sync REPORTS rather than raising, so a boot survives it ***",
      rep2["pushed"] == [] and len(rep2["errors"]) >= 1,
      str(rep2["errors"])[:70])

# the wrap must be idempotent -- plugins load more than once per process
_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                         "control.py")).read()
# *** SLICE TO THE REAL FUNCTION BOUNDARY, not a fixed character window. ***
# This read `_src[i:i+1400]` and passed until the function grew a longer
# docstring, at which point it failed for being too short rather than for
# anything being wrong -- a test that fails for the wrong reason is as
# unhelpful as one that passes for it, which this repo has now paid for four
# times today.
def _fn_body(src, name):
    i = src.index(f"def {name}(")
    j = src.find("\ndef ", i + 10)
    return src[i:j if j != -1 else len(src)]


# Asserted as a PROPERTY, not a spelling: the first version matched the literal
# `if getattr` and broke when the function was rewritten to check per
# candidate with a `continue`. A guard tied to how code is phrased fails on a
# refactor that changed nothing about the behaviour -- the fifth time today a
# test of mine failed for the wrong reason.
for fn in ("wrap_engine_core_init", "wrap_engine_core_rpc"):
    body = _fn_body(_src, fn)
    # Either marker name is fine: what must hold is that one is SET and
    # CHECKED. Pinning the exact name is pinning a spelling, which is what
    # broke the previous version of this assertion.
    marker = next((m for m in ("_kvmgr_sync", "_kvmgr_wrapped") if m in body),
                  None)
    check(f"{fn} marks what it wrapped, so a double load cannot double-wrap",
          marker is not None, "a double wrap would double every call")
    check(f"...and {fn} CHECKS that mark before wrapping",
          marker is not None and "getattr(original" in body
          and f'"{marker}"' in body)

print("\n== (16) the sync hook: EVERY step entry point, ONE shot ==")

# *** TWO HOOKS WRONG IN A ROW, both with the evidence already in hand. ***
# Attempt 1 wrapped EngineCore.__init__, whose first statement is
# load_general_plugins() -- us -- so it could never fire. Attempt 2 wrapped
# EngineCore.step, which this engine never calls: the busy loop goes through
# self.step_fn(), a reference CAPTURED at construction and set to
# step_with_batch_queue under async scheduling. The morning's crash traceback
# showed `_process_engine_step()` calling `self.step_fn()` in as many words.
#
# So every candidate is wrapped, sharing ONE one-shot flag. A scheduling-mode
# change must not be able to silently disable this -- the symptom when it does
# is behaviour differing by node with nothing in any log.
_src6 = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "control.py")).read()


def _fnbody(src, name):
    i = src.index(f"def {name}(")
    j = src.find("\ndef ", i + 10)
    return src[i:j if j != -1 else len(src)]


_body6 = _fnbody(_src6, "wrap_engine_core_init")
for cand in ('"step"', '"step_with_batch_queue"', '"_process_engine_step"'):
    check(f"{cand} is among the wrapped candidates", cand in _body6,
          "one name is a scheduling-mode change away from silent")
check("*** and __init__ is NOT wrapped, because it is what loads us ***",
      'EngineCore.__init__ =' not in _body6 and '"__init__"' not in _body6)
# *** NORMALISE WHITESPACE BEFORE MATCHING A PHRASE IN SOURCE. *** This read
# the raw body and failed, because the docstring wraps the sentence mid-phrase
# ("cannot wrap the\n    function that loads you"). Searching formatted prose
# for a multi-word phrase without collapsing whitespace can only work by luck.
import re as _re6  # noqa: E402

_flat6 = _re6.sub(r"\s+", " ", _body6).lower()
check("...with the reason recorded where somebody would re-break it",
      "cannot wrap the function that loads you" in _flat6)
check("...and why `step` alone is not enough",
      "step_fn" in _body6 and "async scheduling" in _body6)

# ONE shared flag, or two hooks both fire and the sync runs twice.
eq("the one-shot flag is shared, so only the FIRST entry point syncs",
   _body6.count("_kvmgr_synced"), 2)
check("the flag is set BEFORE the sync is attempted",
      _body6.index("_kvmgr_synced = True") < _body6.index("sync_knobs_to_workers(self)"),
      "otherwise a raising sync retries forever on the hot path")

# Behaviour, modelled: three wrapped entry points, one sync.
_fired = []


def _mk(name):
    def original(self, *a, **kw):
        return f"{name}-ran"
    return original


class _FakeCore:
    pass


def _oneshot(original):
    def wrapper(self, *a, **kw):
        if not getattr(self, "_kvmgr_synced", False):
            self._kvmgr_synced = True
            _fired.append("sync")
        return original(self, *a, **kw)
    return wrapper


for n in ("step", "step_with_batch_queue", "_process_engine_step"):
    setattr(_FakeCore, n, _oneshot(_mk(n)))
_c = _FakeCore()
eq("whichever entry point runs first still works", _c.step_with_batch_queue(),
   "step_with_batch_queue-ran")
_c.step()
_c._process_engine_step()
_c.step_with_batch_queue()
eq("*** and the sync happens exactly ONCE across all of them ***",
   _fired, ["sync"])

print("\n== (17) A RELOAD MUST NOT DISCARD THE KNOBS IT WAS ASKED TO KEEP ==")

# *** MEASURED ON THE BOX 2026-10-08, and it is the same bug twice in one
# function. *** The boot sync pushed 12 knobs to both workers; `POST
# /kv/reload` ran; and the ranks were SPLIT again -- rank 1 back on its stale
# file, because `importlib.reload` re-executes `flags._OVERRIDE: dict = {}` and
# threw away the override that had fixed it. I had fixed exactly this for
# `_LIVE_HANDLERS` one line earlier and not carried it across.
#
# The rule for this module: a reload must restore every module-level piece of
# STATE it re-initialises.
flags.reset_for_tests()
# TWO knobs in the file and only ONE overridden, so the "restoring must not
# invent overrides for everything" check below has something real to test.
# The first version wrote one key and then asserted another was file-governed,
# which was simply false -- my assertion, not the code.
F.write(evict_order="lru", protect_alloc="by_need")
flags.set_override("evict_order", "lru_tail_first")
flags.set_override("kvmgr_log_level", 4)
eq("two overrides set before the reload", sorted(flags._OVERRIDE),
   ["evict_order", "kvmgr_log_level"])
eq("...and winning", flags.raw("evict_order"), "lru_tail_first")

rep17 = control.reload_modules()
check("the reload reports ok", rep17.get("ok") is True, str(rep17.get("errors"))[:80])

# re-import, because the reload replaced the module object's contents
from kvmgr import flags as _f17  # noqa: E402

eq("*** the overrides SURVIVE the reload ***", sorted(_f17._OVERRIDE),
   ["evict_order", "kvmgr_log_level"])
eq("...and still win over the file", _f17.raw("evict_order"), "lru_tail_first")
eq("...reported as an override", _f17.source_of("evict_order"), "override")
eq("...and the response NAMES what it restored",
   sorted(rep17.get("overrides_restored") or []),
   ["evict_order", "kvmgr_log_level"])

# the file must still govern a knob nobody overrode -- restoring must not
# invent overrides for everything
eq("a knob in the FILE that nobody overrode is still file-governed",
   _f17.source_of("protect_alloc"), "file")
eq("...with the file's value", _f17.raw("protect_alloc"), "by_need")
eq("a knob in NEITHER is still default", _f17.source_of("swa_recycle"), "default")

_f17.clear_override()

# And the general property, asserted so the NEXT module-level state is not
# forgotten: every mutable module-level container in kvmgr must be either
# restored by reload_modules or provably not state.
_csrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "control.py")).read()
# *** AUDIT THE MODULES AGAINST THE TABLE, DO NOT NAME THE FIELDS. *** The
# first version listed two names by hand and passed while three more stateful
# globals were silently not restored -- `diskspec._LIVE_MANAGER`,
# `stats._groups` (every observability counter) and `offload._real_odirect`.
# A hand-written list of things to remember is the thing that needs
# remembering, so control._PRESERVE is the table and this parses the kvmgr
# package to check nothing is missing from it.
import ast as _ast6  # noqa: E402

_preserved = {f"{m.split('.')[-1]}.{a}" for m, a, _ in control._PRESERVE}

# Constants and caches are NOT state: re-executing them restores the same
# value, or losing them costs at most one log line. Named so the exemption is
# a decision rather than an omission.
_EXEMPT = {
    "flags._VALID": "a validation table -- re-executing yields the same table",
    "flags._IMPLEMENTED": "a capability table, empty by design since Phase 3",
    "flags._last_seen": "flag-change edge cache; losing it costs one edge",
    "flags._announced": "log de-duplication; losing it costs a repeated line",
    "__init__.__all__": "an export list, not state",
}

_kvdir = os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr")
_missing = []
for _f in sorted(os.listdir(_kvdir)):
    if not _f.endswith(".py"):
        continue
    _mod = _f[:-3]
    for _node in _ast6.parse(open(os.path.join(_kvdir, _f)).read()).body:
        if isinstance(_node, _ast6.Assign):
            _names = [t.id for t in _node.targets if isinstance(t, _ast6.Name)]
            _val = _node.value
        elif isinstance(_node, _ast6.AnnAssign) and isinstance(_node.target, _ast6.Name):
            _names, _val = [_node.target.id], _node.value
        else:
            continue
        # *** A ONE-SHOT BOOL IS STATE, AND THIS AUDIT COULD NOT SEE ONE. ***
        # The first version recognised only container literals and None, so
        # `_watchdog_started = False` was invisible to it -- and a reload
        # resetting that flag leaks one more watchdog thread per reload, each
        # dumping the same stacks. Found by break-verification: deleting the
        # _PRESERVE entry left this assertion GREEN and only a hardcoded count
        # elsewhere complained, which is "a lint that cannot fail reads as
        # coverage" with the lint being mine.
        #
        # So a `False` / `0` initialiser counts too -- that is the shape of
        # every one-shot flag and every accumulator. ALL-CAPS names are exempt
        # from the NEW shapes only, because by convention they are tunables
        # (_WATCHDOG_STALL_S = 120.0) and re-executing one restores the same
        # value. The None rule stays unconditional, so this cannot weaken what
        # the audit already caught.
        _container_or_none = isinstance(
            _val, (_ast6.Dict, _ast6.List, _ast6.Set)) or (
            isinstance(_val, _ast6.Constant) and _val.value is None)
        _oneshot = (isinstance(_val, _ast6.Constant)
                    and _val.value in (False, 0))
        for _n in _names:
            if not _n.startswith("_"):
                continue
            _is_state = _container_or_none or (
                _oneshot and not _n.lstrip("_").isupper())
            if not _is_state:
                continue
            _key = f"{_mod}.{_n}"
            if _key in _EXEMPT or _key in _preserved:
                continue
            _missing.append(_key)

eq("*** every stateful module global is in _PRESERVE or explicitly exempt ***",
   sorted(set(_missing)), [])
# A pinned COUNT is the brittle assertion here and it is kept deliberately
# small in ambition: it only guards against an entry being DELETED, which is
# how three real ones came to be unprotected. The assertion that carries the
# weight is the package audit immediately above -- it scans the tree, so it
# grows with the code, whereas a magic number has to be edited every time the
# table legitimately changes (as it just was, for _watchdog_started). Asserting
# a floor rather than equality keeps the guard without the chore.
check("...and _PRESERVE names at least the seven we know about",
      len(control._PRESERVE) >= 7)
check("...and names no duplicates (a dup would silently restore twice)",
      len({(m, a) for m, a, _ in control._PRESERVE}) == len(control._PRESERVE))
check("...every entry carrying one",
      all(isinstance(w, str) and w for _, _, w in control._PRESERVE))
for name in ("_LIVE_HANDLERS", "_OVERRIDE", "_LIVE_MANAGER", "_groups"):
    check(f"_PRESERVE covers {name}",
          any(a == name for _, a, _ in control._PRESERVE))

# *** `from kvmgr import <submodule>` IS A TRAP IN THIS PACKAGE. ***
# __init__.py does `from .register import register`, so the package attribute
# `register` is the entry-point FUNCTION and shadows the submodule. The reload
# path used the obvious spelling and raised AttributeError on all three
# processes live. Any module that re-resolves a kvmgr submodule at runtime must
# use importlib.import_module, which ignores what the package re-exports.
_ctl_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                             "control.py")).read()
_shadowed = {"register"}  # the names __init__.py re-exports
_bad = []
for _line in _ctl_src.splitlines():
    _t = _line.strip()
    if _t.startswith("#"):
        continue
    for _name in _shadowed:
        if _t.startswith(f"from kvmgr import {_name}"):
            _bad.append(_t)
eq("control.py never does `from kvmgr import <a re-exported name>`", _bad, [])
check("...and it reaches submodules via import_module instead",
      'import_module("kvmgr.register")' in _ctl_src)

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
