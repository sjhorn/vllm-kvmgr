"""test_phase11_diag.py -- the diagnostics are isolated, opt-in, and inert when off.

    python3.12 test_phase11_diag.py [--with-dspark-patches]

kvmgr.diag holds what helps diagnose the tier and is not needed to run it:
the hang dump, the transfer ledger, the numpy-producer walk, the geometry
snapshot, the tiering shims and hot reload. The claims, each asserted:

  1. NO FUNCTIONAL PATH IMPORTS diag. One function may --
     register._install_diagnostics -- and only behind each module's knob.
  2. The defaults are the stated ones: hang dump and ledger ON, the rest OFF.
  3. With every diag knob off, register() imports NO diag module at all
     (checked in a fresh process, since this one imports them on purpose).
  4. Off means inert: the ledger reports an ABSENCE, a reload RPC refuses
     with its reason, `/kv/reload` is not routed, no Worker method appears.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import types

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
from kvmgr import flags, hooks  # noqa: E402

print(f"== mode: {harness.MODE} ==")

# ---------------------------------------------------------------------------
print("\n== (1) no functional path imports kvmgr.diag ==")
ALLOWED = {("register.py", "_install_diagnostics")}


def _diag_refs(path: str) -> list[tuple[str, int]]:
    """(enclosing function, line) of every reference to kvmgr.diag in `path`:
    an import of it, or an import_module() whose argument names it."""
    src = open(path).read()
    tree = ast.parse(src)
    out = []

    def names_diag(node) -> bool:
        if isinstance(node, ast.ImportFrom):
            mod = ("." * node.level) + (node.module or "")
            return (mod.startswith(".diag") or mod.startswith("kvmgr.diag")
                    or (node.level >= 1 and not node.module
                        and any(a.name == "diag" for a in node.names)))
        if isinstance(node, ast.Import):
            return any(a.name.startswith("kvmgr.diag") for a in node.names)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(
                node.func, "id", "")) == "import_module" and node.args:
            seg = ast.get_source_segment(src, node.args[0]) or ""
            return "kvmgr.diag" in seg
        return False

    def walk(node, fn):
        for child in ast.iter_child_nodes(node):
            f = child.name if isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if names_diag(child):
                out.append((fn, child.lineno))
            walk(child, f)

    walk(tree, "<module>")
    return out


_core = []
for _root, _dirs, _files in os.walk(harness.KVMGR_DIR):
    _dirs[:] = [d for d in _dirs if d not in ("__pycache__", "diag")]
    _core += [os.path.join(_root, f) for f in _files if f.endswith(".py")]
check("found the core modules (diag/ excluded)", len(_core) >= 12, len(_core))
_hits = {os.path.relpath(p, harness.KVMGR_DIR): _diag_refs(p) for p in _core}
_bad = sorted(f"{f}:{fn}:{ln}" for f, refs in _hits.items()
              for fn, ln in refs if (f, fn) not in ALLOWED)
eq("*** no core module imports kvmgr.diag outside _install_diagnostics ***",
   _bad, [])
check("...and the lint can SEE the one allowed site -- a lint that finds "
      "nothing anywhere cannot fail", bool(_hits.get("register.py")),
      _hits.get("register.py"))

_reg_src = harness.kvmgr_source("register")
_reg = importlib = __import__("importlib")
R = importlib.import_module("kvmgr.register")
_mods = {m for m, _ in R._DIAGNOSTICS}
_files = {f[:-3] for f in os.listdir(os.path.join(harness.KVMGR_DIR, "diag"))
          if f.endswith(".py") and f != "__init__.py"}
eq("every diag module is in _DIAGNOSTICS, and nothing else is", _mods, _files)
eq("every _DIAGNOSTICS knob is declared",
   sorted(k for _, k in R._DIAGNOSTICS if k not in flags.DEFAULTS), [])

# ---------------------------------------------------------------------------
print("\n== (2) the defaults: zero-cost witnesses ON, everything else OFF ==")
WANT = {"diag_hang_dump": True, "diag_transfer_ledger": True,
        "diag_serializer_walk": False, "diag_geometry_snapshot": False,
        "diag_tiering_shims": False, "diag_hot_reload": False}
eq("diag knob defaults", {k: flags.DEFAULTS.get(k) for k in WANT}, WANT)

# ---------------------------------------------------------------------------
print("\n== (3) all off: register() imports NO diag module (fresh process) ==")
_probe = r'''
import json, os, sys
sys.path.insert(0, %(tests)r)
import harness
harness.load()
from kvmgr import flags
harness.Flags(%(fp)r).write(**%(knobs)r)
import importlib
R = importlib.import_module("kvmgr.register")
R._REGISTERED = False
R.register()
from kvmgr import hooks
print("RESULT " + json.dumps({
    "diag": sorted(m for m in sys.modules if m.startswith("kvmgr.diag")),
    "hooks": sorted(k for k, v in hooks._HOOKS.items() if v)}))
'''


def _run(knobs: dict) -> dict:
    fp = os.path.join(tempfile.mkdtemp(prefix="kv11-"), "flags.json")
    code = _probe % {"tests": os.path.dirname(os.path.abspath(__file__)),
                     "fp": fp, "knobs": knobs}
    env = dict(os.environ)
    if harness.DSPARK:
        env["KVMGR_WITH_DSPARK_PATCHES"] = "1"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=env, timeout=120)
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("RESULT ")]
    if not line:
        return {"error": (out.stderr or out.stdout)[-400:]}
    return json.loads(line[-1][len("RESULT "):])


_off = _run({k: 0 for k in WANT})
eq("*** every diag knob off: no kvmgr.diag module is imported ***",
   _off.get("diag"), [])
eq("...and no diagnostic hook is registered", _off.get("hooks"), [])
_def = _run({})
eq("defaults: exactly the hang dump and the ledger are imported",
   _def.get("diag"), ["kvmgr.diag", "kvmgr.diag.hangdump", "kvmgr.diag.ledger"])
eq("...and only the ledger's hooks are registered (the hang dump needs none)",
   _def.get("hooks"), ["ledger_dump", "ledger_note", "ledger_snapshot"])

# ---------------------------------------------------------------------------
print("\n== (4) off means inert, and says so ==")
from kvmgr import control, diskio  # noqa: E402

_saved = {k: list(v) for k, v in hooks._HOOKS.items()}
hooks._HOOKS.clear()
try:
    h = diskio.DiskStoreHandler.__new__(diskio.DiskStoreHandler)
    snap = h._ledger_snapshot()
    eq("*** ledger off: the snapshot says DISABLED, not an empty range ***",
       snap.get("enabled"), False)
    check("...with the reason", "ABSENCE" in snap.get("why", ""), snap)
    h._ledger_note("store", 0, [1, 2], [0, 1])  # must be a quiet no-op
    check("...and noting a transfer is a no-op, not an error",
          getattr(h, "_ledger", None) is None)
    r = control._apply_locally_reported("kvmgr_reload", {})
    eq("*** reload off: the RPC refuses ***", r.get("ok"), False)
    check("...naming the knob", "diag_hot_reload" in r.get("error", ""), r)
    check("hooks.active('reload') is false, which is what gates the route",
          not hooks.active("reload"))
finally:
    hooks._HOOKS.clear()
    hooks._HOOKS.update(_saved)

# The route itself needs fastapi, which the harness does not have, so check
# its STRUCTURE: the decorator sits inside the `if hooks.active("reload"):`.
_ctl = harness.kvmgr_source("control")
_tree = ast.parse(_ctl)
_gated = False
for node in ast.walk(_tree):
    if isinstance(node, ast.If) and "hooks.active(\"reload\")" in (
            ast.get_source_segment(_ctl, node.test) or ""):
        # A function's source segment starts at `def`; the route path is in
        # its DECORATOR, so read those.
        _gated = any('"/kv/reload"' in (ast.get_source_segment(_ctl, d) or "")
                     for n in node.body
                     for d in getattr(n, "decorator_list", ()))
eq("*** the /kv/reload route is defined ONLY under hooks.active('reload') ***",
   _gated, True)
check("...and nowhere else",
      _ctl.count('"/kv/reload"') == 1, _ctl.count('"/kv/reload"'))

# The Worker method appears only when the diagnostic installs.
_wmod = types.ModuleType("vllm.v1.worker.gpu_worker")


class _Worker:
    pass


_wmod.Worker = _Worker
for _n in ("vllm.v1.worker",):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["vllm.v1.worker.gpu_worker"] = _wmod
check("no kvmgr_reload on a Worker before the diagnostic installs",
      not hasattr(_Worker, "kvmgr_reload"))
importlib.import_module("kvmgr.diag.reload").install()
check("...and it appears when it does", hasattr(_Worker, "kvmgr_reload"))
check("...with the reload hook registered", hooks.active("reload"))
hooks.unregister("reload")

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
