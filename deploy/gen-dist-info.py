#!/usr/bin/env python3
"""Generate the .dist-info the DSpark compose bind-mounts beside src/kvmgr.

    python3 deploy/gen-dist-info.py           # write deploy/<name>-<ver>.dist-info
    python3 deploy/gen-dist-info.py --check   # exit 1 if the committed one is stale

WHY IT EXISTS. On the box kvmgr is not pip-installed: compose mounts the
package directory and a dist-info into site-packages, because vLLM finds a
general plugin through its ENTRY POINT, and an entry point lives in a
dist-info. The old one was written by hand and its RECORD listed 5 of 11
modules. This derives every file from pyproject.toml and the src tree, and
--check (run by the tests) fails the moment either moves.

RECORD rows carry no hashes: this dist-info is never pip-uninstalled (it is a
read-only bind mount), and hashes would make every source edit a regen.
Requires Python 3.11+ (tomllib).
"""
from __future__ import annotations

import os
import shutil
import sys
import tomllib

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render() -> tuple[str, dict[str, str]]:
    with open(os.path.join(PKG, "pyproject.toml"), "rb") as fh:
        proj = tomllib.load(fh)["project"]
    dist = proj["name"].replace("-", "_")
    name = f"{dist}-{proj['version']}.dist-info"
    meta = ["Metadata-Version: 2.1", f"Name: {proj['name']}",
            f"Version: {proj['version']}", f"Summary: {proj['description']}",
            f"Requires-Python: {proj['requires-python']}"]
    meta += [f"Requires-Dist: {d}" for d in proj.get("dependencies", [])]
    eps = []
    for group, entries in proj.get("entry-points", {}).items():
        eps.append(f"[{group}]")
        eps += [f"{k} = {v}" for k, v in entries.items()]
    files = {
        "METADATA": "\n".join(meta) + "\n",
        "entry_points.txt": "\n".join(eps) + "\n",
        "top_level.txt": "kvmgr\n",
        "INSTALLER": "dspark-bind-mount\n",
    }
    src = os.path.join(PKG, "src")
    mods = []
    for root, dirs, fs in os.walk(os.path.join(src, "kvmgr")):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        mods += [os.path.relpath(os.path.join(root, f), src)
                 for f in fs if f.endswith(".py")]
    rows = sorted(m.replace(os.sep, "/") for m in mods)
    rows += [f"{name}/{f}" for f in sorted(files)] + [f"{name}/RECORD"]
    files["RECORD"] = "".join(f"{r},,\n" for r in rows)
    return name, files


def main() -> int:
    name, files = render()
    out = os.path.join(PKG, "deploy", name)
    stale = [d for d in os.listdir(os.path.join(PKG, "deploy"))
             if d.endswith(".dist-info") and d != name]
    if "--check" in sys.argv:
        bad = list(stale)
        for f, text in files.items():
            p = os.path.join(out, f)
            if not os.path.isfile(p) or open(p).read() != text:
                bad.append(f"{name}/{f}")
        if bad:
            print("STALE dist-info -- run deploy/gen-dist-info.py:", bad)
            return 1
        print(f"dist-info in sync: {name}")
        return 0
    for d in stale:
        shutil.rmtree(os.path.join(PKG, "deploy", d))
    os.makedirs(out, exist_ok=True)
    for f, text in files.items():
        with open(os.path.join(out, f), "w") as fh:
            fh.write(text)
    print(f"wrote {out} ({len(files)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
