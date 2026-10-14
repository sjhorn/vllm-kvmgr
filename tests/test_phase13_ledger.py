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
  7. (v0.4.0) burst depth and copy size on every submit line, the high-water
     marks, and the LOAD BURST canary.
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


H = harness.load_kv_offload()  # 0.31: the ledger needs the disk tier only
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
   sorted(r), ["bids", "boot", "ev", "group", "inner", "job", "kind", "pid",
               "ts"])
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
# a restarted process numbers its inner ids from 0 again -- and in a container
# it has the SAME pid, so only the boot token tells the two apart
other = L.record("done", "load", 0, boot="1-deadbeef")
eq("*** a `done` from ANOTHER boot with the SAME pid does not close this "
   "boot's entry ***", len(L.open_entries(lines(p) + [other.strip()])), 2)
same = L.record("done", "load", 0)
eq("...while a `done` from this boot does",
   len(L.open_entries(lines(p) + [same.strip()])), 1)
eq("open entries are keyed (boot, kind, inner)",
   sorted(k[0] for k in L.open_entries(lines(p))), [L.boot_id()] * 2)
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
                        ("DiskLoadHandler", "_collect_locked", "load")):
    body = method(cls, meth)
    sub = body.find(f'self._ledger_submit("{kind}", inner,')
    ta = body.find("self._inner.transfer_async(")
    check(f"*** {cls}: _ledger_submit precedes transfer_async (write-ahead) ***",
          0 <= sub < ta, (sub, ta))
    fail = body.find(f'self._ledger_done("{kind}", inner, False)', ta)
    check(f"{cls}: a refused submit closes its entry", fail > ta, fail)

for cls, meth, kind in (("DiskStoreHandler", "_collect", "store"),
                        ("DiskLoadHandler", "_collect_locked", "load")):
    body = method(cls, meth)
    pop = body.find("ent = self._inner_to_job.pop(res.job_id, None)")
    dn = body.find(f'self._ledger_done("{kind}", res.job_id, res.success)', pop)
    check(f"*** {cls}: every observed completion appends done ***",
          0 <= pop < dn, (pop, dn))

# ---------------------------------------------------------------------------
print("\n== (7) v0.4.0: burst depth, copy size, high-water marks, canary ==")
# The 10:12:17 Xid had three loads open, submitted within 3 ms, two of them at
# 2,046 descriptors -- and both facts had to be rebuilt by hand from timestamps
# and refs. They go on the line now.
tmp = fresh()
F.write(diag_ledger_file=1, load_inflight_max=1)
flags.reset_for_tests()


def rhandler(refs, rank=0):
    h = handler(rank)
    h._refs = refs
    return h


hr = rhandler([[0] * 62, [0] * 2])
cap = Cap()
_real_logger, L.logger = L.logger, cap
try:
    L.submit(hr, "store", 0, 1, 0, [1, 2], [0, 1])
    L.submit(hr, "load", 0, 2, 0, list(range(100, 133)), list(range(33)))
    L.submit(hr, "load", 1, 3, 1, [200], [40])
    L.submit(hr, "load", 2, 4, 1, [201], [41])
finally:
    L.logger = _real_logger
p = os.path.join(tmp, "ledger-r0.jsonl")
recs = [json.loads(x) for x in lines(p)]
eq("*** each submit line carries gpu_loads / gpu_stores ALREADY in flight ***",
   [(r.get("gpu_loads"), r.get("gpu_stores")) for r in recs],
   [(0, 0), (0, 1), (1, 1), (2, 1)])
eq("*** and ops = blocks x refs for the group: 33 x 62 = 2,046 ***",
   [r.get("ops") for r in recs], [124, 2046, 2, 2])
eq("high-water marks INCLUDE the new transfer",
   (L._STATS["max_gpu_loads"], L._STATS["max_gpu_stores"]), (3, 1))
snap = L.wal_snapshot()
eq("...and reach /kv/workers as ints",
   (snap.get("max_gpu_loads"), snap.get("max_gpu_stores")), (3, 1))
check("...builtin ints", type(snap.get("max_gpu_loads")) is int)
eq("*** the canary counts every load submitted at or over the bound ***",
   L._STATS["burst_canary"], 2)
eq("...but logs ONCE per process",
   sum("LOAD BURST" in x for x in cap.lines), 1)
check("...naming the bound", any("load_inflight_max=1" in x for x in cap.lines),
      cap.lines)

# a correctly bounded engine: one load at a time reads 1 and never fires
tmp = fresh()
eq("fresh() resets the high-water marks and the canary",
   (L._STATS["max_gpu_loads"], L._STATS["max_gpu_stores"],
    L._STATS["burst_canary"]), (0, 0, 0))
cap = Cap()
_real_logger, L.logger = L.logger, cap
try:
    for i in range(5):
        L.submit(hr, "load", i, i, 0, [i], [i])
        L.done(hr, "load", i, True)
finally:
    L.logger = _real_logger
eq("*** serial loads read max_gpu_loads == 1 -- the deploy check ***",
   L._STATS["max_gpu_loads"], 1)
eq("...and the canary stays silent", (L._STATS["burst_canary"],
   sum("LOAD BURST" in x for x in cap.lines)), (0, 0))

# hostile refs: ops is an ABSENCE, and the line is still written
for label, refs in (("None", None), ("short", [[0]]), ("non-list", [7, 7])):
    tmp = fresh()
    L.submit(rhandler(refs), "load", 0, 1, 1, [5, 6], [0, 1])
    rec = first(os.path.join(tmp, "ledger-r0.jsonl"))
    eq(f"refs {label}: ops is None, never a guess", rec.get("ops", "MISSING"),
       None)
    eq(f"...and the line is still written", rec.get("ev"), "submit")
eq("submit_ops on a handler with no _refs at all", L.submit_ops(handler(), 0, 3),
   None)

# numpy-scalar bids and group (the stand-in from section 1)
tmp = fresh()
L.submit(hr, "load", NpInt(7), NpInt(9), NpInt(0), [NpInt(5), NpInt(6)],
         [NpInt(0), NpInt(1)])
rec = first(os.path.join(tmp, "ledger-r0.jsonl"))
eq("*** numpy-like bids and group: ops computed, line written ***",
   (rec.get("ev"), rec.get("ops"), rec.get("gpu_loads")), ("submit", 124, 0))

# a hot reload preserves the v0.3.0 _STATS dict, which has none of the new keys
tmp = fresh()
for k in ("max_gpu_loads", "max_gpu_stores", "burst_canary"):
    L._STATS.pop(k, None)
L.submit(hr, "load", 0, 1, 0, [5], [0])
L.submit(hr, "load", 1, 2, 0, [6], [1])
eq("*** a reload-preserved v0.3.0 _STATS still writes every submit line ***",
   [json.loads(x).get("ev") for x in lines(os.path.join(tmp, "ledger-r0.jsonl"))],
   ["submit", "submit"])
eq("...and the marks start from it",
   (L._STATS.get("max_gpu_loads"), L.wal_snapshot().get("burst_canary")), (2, 1))
F.write(diag_ledger_file=1)
flags.reset_for_tests()

# ---------------------------------------------------------------------------
print("\n== (8) v0.4.1: boot identity, and the DESTINATION blocks ==")
# The 12:43 Xid 32 hit both ranks 3 ms apart on the same load, identical GPU
# block ids, alone on the head's GPU. What both ranks share is the destination
# blocks and the lockstep forward pass, so each load submit now records what
# the current batch is doing with them.

# (A) two boots, one pid, one file: the 10:12 contamination, reproduced
tmp = fresh()
h = handler()
L._ID.update(boot="100-aaaaaaaa", pid=os.getpid())
L.submit(h, "load", 0, 1, 0, [10], [0])
L.submit(h, "load", 1, 2, 0, [11], [1])      # boot 1 dies with both open
L._ID.update(boot="200-bbbbbbbb", pid=os.getpid())
L._INFLIGHT.clear()
L.submit(h, "load", 0, 1, 0, [10], [0])      # boot 2 reuses pid AND inner ids
L.done(h, "load", 0, True)
recs = [json.loads(x) for x in lines(os.path.join(tmp, "ledger-r0.jsonl"))]
eq("every record carries the boot that wrote it",
   [r.get("boot") for r in recs],
   ["100-aaaaaaaa", "100-aaaaaaaa", "200-bbbbbbbb", "200-bbbbbbbb"])
eq("...and the same pid throughout -- the container case",
   len({r.get("pid") for r in recs}), 1)
op = L.open_entries(lines(os.path.join(tmp, "ledger-r0.jsonl")))
eq("*** boot 2's done does NOT close boot 1's inner 0: the dead boot keeps "
   "both its open entries ***", sorted(op), [("100-aaaaaaaa", "load", 0),
                                           ("100-aaaaaaaa", "load", 1)])
eq("a pre-v0.4.1 line (no boot) still keys by pid",
   list(L.open_entries(['{"ev":"submit","kind":"load","inner":3,"pid":9}'])),
   [(("pid", 9), "load", 3)])
L._ID.update(boot=None, pid=None)
b1 = L.boot_id()
check("a fresh token is <epoch>-<8 hex>",
      len(b1.split("-")) == 2 and len(b1.split("-")[1]) == 8, b1)
eq("...stable within the process", L.boot_id(), b1)
L._ID["pid"] = -1                             # as if forked from a parent
check("*** a forked child mints its OWN token, not the parent's ***",
      L.boot_id() != b1)


# (B) the destination blocks against a fake block table
class Buf:
    def __init__(self, rows):
        self.np = rows


class BT:
    def __init__(self, rows, div=None):
        self.block_table = Buf([list(r) + [0] * 4 for r in rows])
        # the stale tail of row 0 is ids 900.. -- must NOT count
        self.block_table.np[0] = list(rows[0]) + [900, 901, 902, 903]
        self.num_blocks_per_row = [len(r) for r in rows]
        if div:
            self.blocks_per_kv_block = div


class IB:
    def __init__(self, tables, req_ids):
        self.block_table = tables
        self.req_ids = req_ids
        self.num_reqs = len(req_ids)
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}


ib = IB([BT([[1988, 1989, 5], [40, 41]]), BT([[7], [8]])], ["reqA", "reqB"])
eq("*** a load whose ids intersect the step's table: in_step > 0 ***",
   L.step_hits(ib, 0, range(1988, 2111)),
   {"in_step": 2, "in_step_ids": [1988, 1989], "in_step_reqs": ["reqA"],
    "in_step_div": 1})
eq("*** a disjoint one: in_step == 0, a MEASURED zero ***",
   L.step_hits(ib, 0, [3000, 3001])["in_step"], 0)
eq("ids past num_blocks_per_row (a previous occupant's) are not counted",
   L.step_hits(ib, 0, [900, 901])["in_step"], 0)
eq("only THIS group's table is read", L.step_hits(ib, 1, [1988])["in_step"], 0)
eq("kernel-block ids are divided back to KV-manager ids",
   L.step_hits(IB([BT([[40, 41, 42, 43]], div=4)], ["r"]), 0, [10])["in_step"],
   1)
for label, bad in (("absent table", IB(None, ["r"])),
                   ("short table", IB([], ["r"])),
                   ("hostile num_reqs", types.SimpleNamespace(num_reqs="x")),
                   ("None", None)):
    got = L.step_hits(bad, 0, [1988])
    check(f"*** {label}: in_step is None with a reason -- ABSENT IS NOT ZERO ***",
          got.get("in_step") is None and "unreadable" in got.get("in_step_why", ""),
          got)


# end to end: submit() finds the runner on the stack and the request per job
class Runner:
    def __init__(self, ib):
        self.input_batch = ib

    def execute_model(self, fn):
        return fn()


class CW:
    pass


from kvmgr import diskio as _dio  # noqa: E402
_saved_cw = list(_dio._LIVE_CONN_WORKER)
cw = CW()
cw._load_jobs = {902: "req-902", 903: "reqB"}
_dio._LIVE_CONN_WORKER[:] = [cw]
try:
    tmp = fresh()
    L._RUNNER.clear()
    hr = rhandler([[0] * 62])
    eq("with no runner on the stack, a load records in_step=None, not 0",
       (L.submit(hr, "load", 50, 902, 0, [1988], [0]),
        first(os.path.join(tmp, "ledger-r0.jsonl")).get("in_step", "MISSING"))[1],
       None)
    eq("...counted as UNREADABLE, apart from the zeroes",
       (L.wal_snapshot()["loads_in_step_unreadable"],
        L.wal_snapshot()["loads_not_in_step"]), (1, 0))
    run = Runner(ib)
    run.execute_model(lambda: L.submit(hr, "load", 51, 902, 0,
                                       list(range(1988, 2021)), [1]))
    run.execute_model(lambda: L.submit(hr, "load", 52, 903, 0, [3000], [2]))
    L.submit(hr, "store", 9, 1, 0, [1988], [3])
    recs = [json.loads(x) for x in lines(os.path.join(tmp, "ledger-r0.jsonl"))]
    eq("*** the submit line carries in_step, examples, req, req_in_step ***",
       [(r.get("in_step"), r.get("in_step_ids"), r.get("req"),
         r.get("req_in_step")) for r in recs[1:3]],
       [(2, [1988, 1989], "req-902", False), (0, [], "reqB", True)])
    check("a STORE line carries none of it",
          "in_step" not in recs[3] and "req" not in recs[3], recs[3])
    snap = L.wal_snapshot()
    eq("*** /kv/workers counts loads by in_step, absence kept apart ***",
       tuple(snap.get(k) for k in ("loads_in_step", "loads_not_in_step",
                                   "loads_in_step_unreadable",
                                   "loads_req_in_step")), (1, 1, 1, 1))
    eq("...and names this process's boot", snap.get("boot"), L.boot_id())
    check("the runner is cached once found", L._RUNNER == [run])
    # a request the connector worker does not know: absent, not guessed
    L.submit(hr, "load", 53, 777, 0, [3000], [4])
    rec = first(os.path.join(tmp, "ledger-r0.jsonl"), -1)
    eq("an unknown job's req is None", (rec.get("req"), rec.get("req_in_step")),
       (None, None))
    json.dumps(L.wal_snapshot())
    check("every new field is JSON-plain (the msgpack rule)", True)
finally:
    _dio._LIVE_CONN_WORKER[:] = _saved_cw
    L._RUNNER.clear()

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("FAILED:", f)
sys.exit(1 if FAIL else 0)
