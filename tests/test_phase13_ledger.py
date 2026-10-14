"""test_phase13_ledger.py -- the write-ahead transfer ledger (v0.3.0).

    python3.12 test_phase13_ledger.py [--with-dspark-patches]

The 2026-10-14 Xid 32 on the worker landed in the same second as a ledger
SUMMARY covering 205 transfers, with 50 stores outstanding, and nothing could
say which transfer faulted: the ring that could died with the process. So
kvmgr.diag.ledger now writes one line per GPU transfer BEFORE the submit and
one on its observed completion, and checks every submit against the in-flight
set. The claims, each asserted:

  1. the record format: one JSON line, block ids as runs, ints not numpy.
  2. the open entries ARE the in-flight set -- per pid, tolerant of a torn line.
  3. overlap detection: load<->store and load<->load flagged, store<->store not,
     disjoint sets never; counted and logged, and the transfer unchanged.
  4. rotation at the byte bound, keeping exactly two files.
  5. diag_ledger_file off means NO file (and says so); the registry still runs.
  6. diskio calls submit BEFORE transfer_async on both paths, and done on every
     observed completion and every failed submit.
"""

from __future__ import annotations

import json
import os
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
import importlib  # noqa: E402

from kvmgr import flags  # noqa: E402

L = importlib.import_module("kvmgr.diag.ledger")
F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv13f-"), "flags.json"))
print(f"== mode: {harness.MODE} ==")


def fresh(tmp=None):
    """Reset the module state and point the file at a new directory."""
    for fd in [L._WAL.get("fd")]:
        if fd is not None:
            try:
                os.close(fd)
            except OSError:
                pass
    L._INFLIGHT.clear()
    L._WAL.update(fd=None, path=None, bytes=0, why="nothing submitted yet")
    for k in L._STATS:
        L._STATS[k] = None if k == "overlap_last" else 0
    tmp = tmp or tempfile.mkdtemp(prefix="kv13-")
    os.environ["KVMGR_LEDGER_DIR"] = tmp
    return tmp


def handler(rank=0):
    return types.SimpleNamespace(_spec=types.SimpleNamespace(
        rank=rank, root_dir="/nonexistent/dspark-kv-v1"))


def lines(path):
    try:
        with open(path) as f:
            return f.read().splitlines()
    except OSError:
        return []


def first(path, i=0) -> dict:
    """Line i as a dict, or {} -- so a mutation that writes nothing FAILS the
    assertions below instead of aborting the suite on an IndexError, which
    would detect the bug while saying nothing about it."""
    ls = lines(path)
    try:
        return json.loads(ls[i])
    except (IndexError, ValueError):
        return {}


# ---------------------------------------------------------------------------
print("\n== (1) the record format ==")
eq("ranges: runs, sorted, deduplicated", L.ranges([7, 3, 4, 5, 9, 4]),
   [[3, 5], [7, 7], [9, 9]])
eq("ranges: empty is empty, not [[None, None]]", L.ranges([]), [])
eq("expand inverts ranges", L.expand(L.ranges([1, 2, 3, 10])), {1, 2, 3, 10})
r = json.loads(L.record("submit", "load", 4, ts=1.5, pid=77, job=9, group=2,
                        bids=[[1, 3]]))
eq("one JSON object with the fields asked for",
   sorted(r), ["bids", "ev", "group", "inner", "job", "kind", "pid", "ts"])
eq("...values intact", (r["ev"], r["kind"], r["inner"], r["job"], r["pid"]),
   ("submit", "load", 4, 9, 77))
check("a record is exactly ONE line",
      L.record("done", "store", 1).count("\n") == 1)

tmp = fresh()
h = handler(rank=1)
F.write(diag_ledger_file=1)


class NpInt:
    """Stands in for numpy.int64: NOT an int, so json.dumps refuses it exactly
    as it refuses a numpy scalar -- and that refusal would happen inside
    submit's except, losing the line silently. (An int SUBCLASS would pass
    json and test nothing; the first version of this did.)"""

    def __init__(self, v):
        self.v = v

    def __int__(self):
        return self.v

    __index__ = __int__

    def __hash__(self):
        return hash(self.v)

    def __eq__(self, o):
        return int(self) == int(o)

    def __lt__(self, o):
        return int(self) < int(o)


L.submit(h, "store", 3, NpInt(41), NpInt(0), [NpInt(5), NpInt(6)], [0, 1])
path = os.path.join(tmp, "ledger-r1.jsonl")
check("*** the file is ledger-r<rank>.jsonl in KVMGR_LEDGER_DIR ***",
      os.path.exists(path), os.listdir(tmp))
rec = first(path)
eq("the submit line carries kind/inner/job/group/n/min/max/runs/slots",
   tuple(rec.get(k) for k in ("ev", "kind", "inner", "job", "group", "n",
                              "bid_min", "bid_max", "bids", "slots")),
   ("submit", "store", 3, 41, 0, 2, 5, 6, [[5, 6]], [[0, 1]]))
check("...and a wall-clock ts (lined up against journalctl -k)",
      abs((rec.get("ts") or 0) - __import__("time").time()) < 60, rec.get("ts"))
L.done(h, "store", 3, True)
eq("done appends a `done` line for the same (kind, inner)",
   [json.loads(x)["ev"] for x in lines(path)], ["submit", "done"])
eq("the default dir is the tier root's PARENT, outside the walked tree",
   (os.environ.pop("KVMGR_LEDGER_DIR"), L.ledger_dir(h))[1],
   "/nonexistent/diag")
os.environ["KVMGR_LEDGER_DIR"] = tmp

# ---------------------------------------------------------------------------
print("\n== (2) the open entries are the in-flight set ==")
tmp = fresh()
h = handler()
L.submit(h, "store", 0, 1, 0, [1, 2], [0, 1])
L.submit(h, "load", 0, 2, 0, [10], [0])      # same inner id, other direction
L.submit(h, "store", 1, 1, 0, [3], [2])
L.done(h, "store", 0, True)
p = os.path.join(tmp, "ledger-r0.jsonl")
op = L.open_entries(lines(p))
eq("*** submits with no done are exactly what is open ***",
   sorted((k[1], k[2]) for k in op), [("load", 0), ("store", 1)])
eq("...and equal the live registry", sorted(op and [(k[1], k[2]) for k in op]),
   sorted(L._INFLIGHT))
# a restarted process numbers its inner ids from 0 again
other = L.record("done", "load", 0, pid=os.getpid() + 1)
eq("a `done` from ANOTHER pid does not close this pid's entry",
   len(L.open_entries(lines(p) + [other.strip()])), 2)
eq("a torn final line (a death mid-write) is skipped, not fatal",
   len(L.open_entries(lines(p) + ['{"ts":1,"ev":"sub'])), 2)

# ---------------------------------------------------------------------------
print("\n== (3) overlap detection ==")
inflight = {("store", 1): {"bids": frozenset({5, 6, 7}), "group": 0, "job": 1},
            ("load", 2): {"bids": frozenset({20}), "group": 1, "job": 2}}
o = L.overlaps("load", [7, 8], inflight)
eq("*** load<->store on a shared block is an overlap ***",
   [(k, i, c) for k, i, _j, _g, c in o], [("store", 1, [7])])
eq("load<->load (two writers) is an overlap",
   [x[0] for x in L.overlaps("load", [20], inflight)], ["load"])
eq("store vs in-flight load is an overlap too (either side a load)",
   [x[0] for x in L.overlaps("store", [20], inflight)], ["load"])
eq("*** store<->store is NOT: two readers are benign ***",
   L.overlaps("store", [5], inflight), [])
eq("*** disjoint sets are never an overlap ***",
   L.overlaps("load", [1, 2, 3, 100], inflight), [])

tmp = fresh()
h = handler()


class Cap:
    def __init__(self):
        self.lines = []

    def warning(self, fmt, *a):
        self.lines.append(fmt % a)

    info = error = critical = exception = warning


cap = Cap()
_real_logger, L.logger = L.logger, cap
try:
    L.submit(h, "store", 0, 1, 0, [5, 6, 7], [0, 1, 2])
    L.submit(h, "load", 0, 2, 3, [7, 8], [0, 1])
    L.submit(h, "load", 1, 3, 0, [99], [2])
finally:
    L.logger = _real_logger
eq("*** exactly one overlap counted ***", L._STATS["overlaps"], 1)
eq("...not same-group (0 vs 3)", L._STATS["overlap_same_group"], 0)
check("...and logged as LEDGER OVERLAP load<->store",
      any("LEDGER OVERLAP load<->store blocks=[7]" in x for x in cap.lines),
      cap.lines)
recs = [json.loads(x) for x in lines(os.path.join(tmp, "ledger-r0.jsonl"))]
eq("the overlapping submit's line names the transfer it overlapped",
   [r.get("overlap") for r in recs], [None, [["store", 0]], None])
eq("...and the transfer is still registered: observed, not refused",
   sorted(L._INFLIGHT), [("load", 0), ("load", 1), ("store", 0)])
snap = L.snapshot(types.SimpleNamespace())
eq("/kv/workers carries the counter", snap.get("overlaps"), 1)
check("...and the last overlap", (snap.get("overlap_last") or {}).get("blocks") == 1,
      snap.get("overlap_last"))
eq("...and an untouched handler's snapshot is still an ABSENCE",
   snap["transfers"], 0)

# ---------------------------------------------------------------------------
print("\n== (4) rotation ==")
tmp = fresh()
h = handler()
_saved_rot, L._WAL_ROTATE_BYTES = L._WAL_ROTATE_BYTES, 600
try:
    for i in range(40):
        L.submit(h, "store", i, i, 0, [i], [0])
        L.done(h, "store", i, True)
finally:
    L._WAL_ROTATE_BYTES = _saved_rot
files = sorted(os.listdir(tmp))
eq("*** two files, the live one and .1 -- never a third ***", files,
   ["ledger-r0.jsonl", "ledger-r0.jsonl.1"])
check("rotations were counted", L._STATS["wal_rotations"] >= 2,
      L._STATS["wal_rotations"])
check("the live file is under the bound plus one line",
      os.path.getsize(os.path.join(tmp, files[0])) < 600 + 400,
      os.path.getsize(os.path.join(tmp, files[0])))
last = first(os.path.join(tmp, files[0]), -1)
eq("...and holds the newest record", (last.get("ev"), last.get("inner")),
   ("done", 39))

# ---------------------------------------------------------------------------
print("\n== (5) diag_ledger_file off: no file, and it says so ==")
tmp = fresh()
h = handler()
F.write(diag_ledger_file=0)
L.submit(h, "store", 0, 1, 0, [1], [0])
L.submit(h, "load", 0, 2, 0, [1], [0])
eq("*** knob off: NO file is created ***", os.listdir(tmp), [])
eq("...the snapshot says why rather than reporting an empty file",
   (L.wal_snapshot()["ledger_file"], L.wal_snapshot()["ledger_file_why"]),
   (None, "diag_ledger_file is off"))
eq("...and the overlap check still runs (it costs no disk)",
   L._STATS["overlaps"], 1)
F.write(diag_ledger_file=1)
L.submit(h, "store", 1, 3, 0, [2], [1])
check("turning it back on is HOT: the next submit opens the file",
      os.path.exists(os.path.join(tmp, "ledger-r0.jsonl")))
eq("the default is ON", flags.DEFAULTS.get("diag_ledger_file"), True)

tmp = fresh("/dev/null/cannot")              # a directory that cannot exist
L.submit(h, "store", 0, 1, 0, [1], [0])
eq("an unwritable directory is counted, never raised", L._STATS["wal_errors"], 1)
eq("...and the registry still holds the transfer", list(L._INFLIGHT),
   [("store", 0)])

# ---------------------------------------------------------------------------
print("\n== (6) diskio: submit BEFORE transfer_async; done on every exit ==")
src = harness.kvmgr_source("diskio")


def method(cls: str, name: str) -> str:
    i = src.index(f"class {cls}")
    j = src.index(f"    def {name}(", i)
    k = src.find("\n    def ", j + 1)
    return src[j:k]


for cls, meth, kind in (("DiskStoreHandler", "_pump", "store"),
                        ("DiskLoadHandler", "_collect", "load")):
    body = method(cls, meth)
    sub = body.find(f'self._ledger_submit("{kind}", inner,')
    ta = body.find("self._inner.transfer_async(")
    check(f"*** {cls}: _ledger_submit precedes transfer_async (write-ahead) ***",
          0 <= sub < ta, (sub, ta))
    fail = body.find(f'self._ledger_done("{kind}", inner, False)', ta)
    check(f"{cls}: a refused submit closes its entry", fail > ta, fail)

for cls, meth, kind in (("DiskStoreHandler", "_collect", "store"),
                        ("DiskLoadHandler", "_collect", "load")):
    body = method(cls, meth)
    pop = body.find("ent = self._inner_to_job.pop(res.job_id, None)")
    dn = body.find(f'self._ledger_done("{kind}", res.job_id, res.success)', pop)
    check(f"*** {cls}: every observed completion appends done ***",
          0 <= pop < dn, (pop, dn))

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("FAILED:", f)
sys.exit(1 if FAIL else 0)
