"""test_phase12_packaging.py -- the package is what pip and the box both see.

    python3.12 test_phase12_packaging.py
    KVMGR_BUILD_PY=/path/to/python-with-setuptools-and-build python3.12 test_phase12_packaging.py

The old dist-info was written by hand and its RECORD listed 5 of 11 modules.
Two deliveries now, both checked:

  * the BIND-MOUNTED dist-info the DSpark compose uses, generated from
    pyproject.toml by deploy/gen-dist-info.py -- in sync, every module listed,
    the entry point vLLM looks for;
  * the WHEEL, built and pip-installed (--no-deps) into a scratch venv, where
    kvmgr must import, register and be discoverable by its entry point with
    no src/ on the path. Needs KVMGR_BUILD_PY; without it, a named SKIP.
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile

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


PKG = harness.PKG
with open(os.path.join(PKG, "pyproject.toml"), "rb") as _fh:
    PROJ = tomllib.load(_fh)["project"]
MODULES = sorted(
    os.path.relpath(p, harness.SRC).replace(os.sep, "/")
    for p in glob.glob(os.path.join(harness.KVMGR_DIR, "**", "*.py"), recursive=True))
EP = PROJ["entry-points"]["vllm.general_plugins"]

print("== (1) the bind-mounted dist-info ==")
check("found the package modules", len(MODULES) >= 15, len(MODULES))
eq("the entry point is kvmgr:register", EP, {"kvmgr": "kvmgr:register"})
r = subprocess.run([sys.executable, os.path.join(PKG, "deploy", "gen-dist-info.py"),
                    "--check"], capture_output=True, text=True)
eq("*** the committed dist-info is in sync with pyproject + src ***",
   r.returncode, 0)
_di = glob.glob(os.path.join(PKG, "deploy", "*.dist-info"))
eq("exactly one dist-info is committed", len(_di), 1)
if _di:
    _rec = {ln.split(",")[0] for ln in open(os.path.join(_di[0], "RECORD"))}
    eq("*** RECORD lists EVERY module ***", sorted(set(MODULES) - _rec), [])
    check("...and the entry point file names kvmgr:register",
          "kvmgr = kvmgr:register" in open(os.path.join(_di[0], "entry_points.txt")).read())
    if harness.needs_dspark("the DSpark compose mounts this dist-info"):
        _comp = open(os.path.join(harness.DSPARK_REPO, "docker-compose.dspark.yml")).read()
        _name = os.path.basename(_di[0])
        check("*** the compose mounts the GENERATED dist-info, by its name ***",
              f"deploy/{_name}:/opt/env/lib/python3.12/site-packages/{_name}" in _comp, _name)

print("\n== (2) the wheel ==")
BUILD_PY = os.environ.get("KVMGR_BUILD_PY")
if not BUILD_PY:
    print("SKIP  the wheel build and install -- set KVMGR_BUILD_PY to a python "
          "with setuptools and build")
else:
    work = tempfile.mkdtemp(prefix="kv12-")
    try:
        src_copy = os.path.join(work, "pkg")
        shutil.copytree(PKG, src_copy, ignore=shutil.ignore_patterns(
            "__pycache__", "build", "*.egg-info", "tests"))
        b = subprocess.run([BUILD_PY, "-m", "build", "--wheel", "--no-isolation",
                            "--outdir", os.path.join(work, "dist"), src_copy],
                           capture_output=True, text=True)
        eq("the wheel builds", b.returncode, 0)
        whl = glob.glob(os.path.join(work, "dist", "*.whl"))
        eq("one wheel", len(whl), 1)
        z = zipfile.ZipFile(whl[0])
        rec = [n for n in z.namelist() if n.endswith(".dist-info/RECORD")]
        listed = {ln.split(",")[0] for ln in z.read(rec[0]).decode().splitlines()}
        eq("*** the wheel's RECORD lists every module ***",
           sorted(set(MODULES) - listed), [])
        eps = [n for n in z.namelist() if n.endswith("entry_points.txt")]
        check("...and declares the general-plugin entry point",
              "kvmgr = kvmgr:register" in z.read(eps[0]).decode())
        meta = z.read([n for n in z.namelist() if n.endswith("METADATA")][0]).decode()
        check("...and the vLLM floor it is proven on",
              "Requires-Dist: vllm" in meta, [l for l in meta.splitlines() if "Requires" in l])

        venv = os.path.join(work, "venv")
        v = subprocess.run([BUILD_PY, "-m", "venv", venv], capture_output=True, text=True)
        vpy = os.path.join(venv, "bin", "python")
        i = subprocess.run([vpy, "-m", "pip", "install", "-q", "--no-deps",
                            "--no-index", whl[0]], capture_output=True, text=True)
        eq("pip install --no-deps into a scratch venv", i.returncode, 0)
        probe = (
            "import os, sys; sys.path.insert(0, %r); import harness; harness.load();"
            "import kvmgr, importlib, importlib.metadata as md;"
            "assert 'site-packages' in kvmgr.__file__, kvmgr.__file__;"
            "assert harness.SRC not in sys.path;"
            "ep=[e for e in md.entry_points(group='vllm.general_plugins') if e.name=='kvmgr'];"
            "assert ep and ep[0].load() is kvmgr.register, ep;"
            "R=importlib.import_module('kvmgr.register'); R._REGISTERED=False; ep[0].load()();"
            "print('INSTALLED-OK', kvmgr.__file__)"
        ) % os.path.dirname(os.path.abspath(__file__))
        env = dict(os.environ, KVMGR_TEST_INSTALLED="1")
        if harness.DSPARK:
            env["KVMGR_WITH_DSPARK_PATCHES"] = "1"
        p = subprocess.run([vpy, "-c", probe], capture_output=True, text=True, env=env)
        check("*** the INSTALLED kvmgr imports from site-packages, is found by its "
              "entry point, and registers -- no src/ on the path ***",
              "INSTALLED-OK" in p.stdout, (p.stdout + p.stderr)[-300:])
    finally:
        shutil.rmtree(work, ignore_errors=True)

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
