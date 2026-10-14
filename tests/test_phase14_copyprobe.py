"""test_phase14_copyprobe.py -- ask3: the copy probe (A), the dump (B), the boot
self-test (C) and the ledger's load context (D). No GPU: tensors, events and
the inner copy handlers are fakes that follow 0.21's addressing.

    python3.12 test_phase14_copyprobe.py [--with-dspark-patches]
"""

from __future__ import annotations

import importlib
import json
import os
import sys
import tempfile

import harness

PASS = 0
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL.append(name)
        print(f"FAIL  {name}  {detail}")


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r} want {want!r}")


base = harness.load_kv_offload()
from kvmgr import flags, hooks  # noqa: E402

L = importlib.import_module("kvmgr.diag.ledger")
CP = importlib.import_module("kvmgr.diag.copyprobe")
ST = importlib.import_module("kvmgr.diag.selftest")
TMP = tempfile.mkdtemp(prefix="kv14-")
os.environ["KVMGR_LEDGER_DIR"] = TMP
F = harness.Flags(os.path.join(TMP, "flags.json"))
F.write(diag_ledger_file=1)
print(f"== mode: {harness.MODE} ==")

INCIDENT = [4892, 16178, 16524, 18563, 18924, 19257, 19329, 19376, 19861,
            20196, 20805, 20964, 20967, 21069, 21440, 21696]


# ---------------------------------------------------------------------------
# fakes
# ---------------------------------------------------------------------------
class Storage:
    def __init__(self, ptr, nbytes):
        self._p, self._n = ptr, nbytes

    def data_ptr(self):
        return self._p

    def nbytes(self):
        return self._n


class T:
    """A 2-D int8 tensor: address arithmetic, plus row bytes for C."""

    def __init__(self, base_ptr, rows, row_bytes, alloc_bytes=None):
        self._p, self.shape = base_ptr, (rows, row_bytes)
        self._alloc = alloc_bytes if alloc_bytes is not None else rows * row_bytes
        self.rows: dict = {}

    def data_ptr(self):
        return self._p

    def stride(self, d):
        return self.shape[1] if d == 0 else 1

    def untyped_storage(self):
        return Storage(self._p, self._alloc)


class Ref:
    def __init__(self, t, size):
        self.tensor_idx, self.page_size_bytes = t, size


class Ev:
    def __init__(self, state):
        self.state = state

    def query(self):
        if isinstance(self.state, Exception):
            raise self.state
        return self.state


class Stream:
    def __init__(self, h):
        self.cuda_stream = h


def mk_handler(ntensors=3, rows=22756, row_bytes=64, staging=40, refs=None,
               rank=0):
    h = type("H", (), {})()
    h._spec = type("S", (), {"rank": rank, "root_dir": "/kvoffload/dspark-kv-v1"})()
    h._gpu_tensors = [T(0x7f0000000000 + i * (1 << 32), rows, row_bytes)
                      for i in range(ntensors)]
    h._staging = [T(0x100000000 + i * (1 << 24), staging, row_bytes)
                  for i in range(ntensors)]
    h._refs = [refs if refs is not None else
               [Ref(i, row_bytes) for i in range(ntensors)], [Ref(0, row_bytes)]]
    h._inner = type("I", (), {})()
    h._inner._transfer_events = {}
    h._inner._stream_pool = []
    return h


def wal_lines():
    p = os.path.join(TMP, "ledger-r0.jsonl")
    if not os.path.exists(p):
        return []
    return [json.loads(x) for x in open(p).read().splitlines() if x.strip()]


# ---------------------------------------------------------------------------
print("\n== (D) the ledger's load context ==")
eq("*** bids_sha reproduces the incident's own id for job 5490 ***",
   L.bids_sha(set(INCIDENT)), "cadfed0fc6ce")
h = mk_handler(refs=[Ref(0, 64), Ref(1, 64), Ref(1, 32)])
d = L.dest_shape(h, 0, {10, 11, 12, 40})
eq("dst_span is max-min", d["dst_span"], 30)
eq("dst_contig_runs counts runs", d["dst_contig_runs"], 2)
eq("ref_tensors", d["ref_tensors"], 2)
eq("*** ref_dup_tensors counts refs sharing a tensor ***", d["ref_dup_tensors"], 1)
check("tensor_idx_set_hash is 12 hex", len(d["tensor_idx_set_hash"]) == 12)
check("no torch.cuda: compute_stream is null WITH a reason, never 0",
      d["compute_stream"] is None and d.get("compute_stream_why"))
eq("an empty stream pool says why", d["copy_stream_why"],
   "pool empty: upstream creates a new stream")
import torch as _torch  # the stub  # noqa: E402
_had = hasattr(_torch, "cuda")
_torch.cuda = type("C", (), {"current_stream": staticmethod(lambda: Stream(111))})
h._inner._stream_pool.extend([Stream(333), Stream(222)])
d2 = L.stream_context(h)
eq("compute stream handle", d2["compute_stream"], 111)
eq("*** copy stream is the TOP of the pool (what upstream pops) ***",
   d2["copy_stream"], 222)
L.submit(h, "load", 1000051, 902, 0, [1988, 1990], [3, 4])
sub = [x for x in wal_lines() if x["ev"] == "submit" and x["inner"] == 1000051]
check("the submit line carries the D fields",
      sub and all(k in sub[-1] for k in ("dst_span", "dst_contig_runs", "bids_sha",
                                         "tensor_idx_set_hash", "ref_dup_tensors",
                                         "compute_stream", "copy_stream")),
      repr(sub[-1] if sub else None))
L.submit(h, "store", 5, 1, 0, [7], [0])
st_line = [x for x in wal_lines() if x["ev"] == "submit" and x["inner"] == 5]
check("...and store lines do not", st_line and "dst_span" not in st_line[-1])
L.done(h, "load", 1000051, True)
L.done(h, "store", 5, True)

# ---------------------------------------------------------------------------
print("\n== (B) the descriptor dump ==")
h = mk_handler(ntensors=2, rows=100, row_bytes=64)
d = CP.describe_copy("load", h._gpu_tensors, h._staging, h._refs[0],
                     [5, 9], [0, 1])
eq("one descriptor per ref per block", d["n_descriptors"], 4)
r0 = d["descriptors"][0]
eq("*** dst = gpu base + block * stride(0) (0.21's addressing) ***",
   r0[5], h._gpu_tensors[0].data_ptr() + 5 * 64)
eq("...src = staging base + slot * stride(0)", r0[4], h._staging[0].data_ptr())
eq("...size = the ref's page_size_bytes", r0[6], 64)
check("in view and in allocation", d["all_in_view"] and d["all_in_alloc"])
eq("no overlaps in a clean copy", d["dst_overlaps"], 0)
eq("alignment reported", d["min_align"], 64)
ds = CP.describe_copy("store", h._gpu_tensors, h._staging, h._refs[0], [5], [0])
eq("a store's dst is the staging side", ds["descriptors"][0][5],
   h._staging[0].data_ptr())

hd = mk_handler(ntensors=2, rows=100, row_bytes=64,
                refs=[Ref(0, 64), Ref(0, 64), Ref(1, 64)])
dd = CP.describe_copy("load", hd._gpu_tensors, hd._staging, hd._refs[0], [5], [0])
check("*** two refs on ONE tensor write the same range: overlap found ***",
      dd["dst_overlaps"] == 1 and dd["ref_dup_tensors"] == 1, repr(dd["dst_overlaps"]))
do = CP.describe_copy("load", h._gpu_tensors, h._staging, h._refs[0], [100], [0])
check("*** a block past the tensor's rows is OUT OF VIEW ***",
      not do["all_in_view"] and do["out_of_view"][0] == [0, 100])
ha = mk_handler(ntensors=1, rows=100, row_bytes=64)
ha._gpu_tensors[0]._alloc = 64 * 50
da = CP.describe_copy("load", ha._gpu_tensors, ha._staging, ha._refs[0], [60], [0])
check("*** a range past the ALLOCATION is caught even when the view allows it ***",
      da["all_in_view"] and not da["all_in_alloc"])
hb = mk_handler(ntensors=1, rows=100, row_bytes=64, refs=[Ref(0, 80)])
db = CP.describe_copy("load", hb._gpu_tensors, hb._staging, hb._refs[0], [1], [0])
eq("a ref bigger than the row is named", db["ref_bigger_than_row"], [0])

# ---------------------------------------------------------------------------
print("\n== (A) the probe asks the GPU, from the watchdog thread ==")
harness.clear_log()
h = mk_handler()
CP._OPEN.clear()
CP.on_submit(h, "load", 1000447, 5489, 0, [10, 11], [0, 1])
CP.on_submit(h, "load", 1000448, 5490, 0, INCIDENT, list(range(16)))
CP.on_submit(h, "store", 77, 12, 0, [30], [2])
h._inner._transfer_events.update({1000447: Ev(True), 1000448: Ev(False)})
CP.tick("busy", 0.0)
eq("young copies are probed, not flagged", CP._STATS["stuck"], 0)
eq("...and the probe recorded the GPU's answer",
   CP._OPEN[("load", 1000448)]["gpu_done"], False)
CP._OPEN[("load", 1000448)]["mono"] -= CP.STUCK_S + 1
CP.tick("busy", 0.0)
eq("*** a copy open past STUCK_S is flagged ***", CP._STATS["stuck"], 1)
check("...logged with copy_done_on_gpu=False and what that means",
      "copy_done_on_gpu=False -- the GPU has NOT finished this copy"
      in harness.log_text())
stuck = [x for x in wal_lines() if x["ev"] == "stuck"]
check("...written to the ledger as ev=stuck",
      stuck and stuck[-1]["inner"] == 1000448
      and stuck[-1]["copy_done_on_gpu"] is False, repr(stuck[-1:]))
dump_path = os.path.join(TMP, "copydump-r0-load-1000448.json")
check("*** and its descriptors dumped ***", os.path.exists(dump_path))
dj = json.load(open(dump_path)) if os.path.exists(dump_path) else {}
eq("...the dump names the incident's set", dj.get("bids_sha"), "cadfed0fc6ce")
eq("...with every descriptor", dj.get("n_descriptors"), 16 * 3)
check("...and a copydump ledger line points at it",
      any(x["ev"] == "copydump" and x["path"] == dump_path for x in wal_lines()))
CP.tick("busy", 0.0)
eq("flagged ONCE, not every tick", CP._STATS["stuck"], 1)
CP.tick("stalled", 40.0)
check("*** while STALLED, a finished-but-unobserved copy says so ***",
      "the GPU FINISHED it but the worker never observed it" in harness.log_text())
check("...and the store with no event says it could not be asked",
      "could not be asked: no event" in harness.log_text())
h._inner._transfer_events[1000447] = Ev(RuntimeError("CUDA error: an illegal memory access"))
eq("a query that raises is recorded, not raised",
   CP.query_copy(h, 1000447)[0], None)
check("...with the CUDA error as the reason",
      "illegal memory access" in CP.query_copy(h, 1000447)[1])
CP.on_done(h, "load", 1000448, True)
check("a completed copy leaves the open set", ("load", 1000448) not in CP._OPEN)
CP.on_submit(h, "load", 1000500, 6000, 0, [40], [5])
CP.on_done(h, "load", 1000500, False)
check("*** a FAILED submit is dumped ***",
      os.path.exists(os.path.join(TMP, "copydump-r0-load-1000500.json")))
snap = CP.snapshot()
check("the snapshot is plain ints/strings", isinstance(snap["copy_stuck"], int)
      and isinstance(snap["copy_dumps"], int))
CP._OPEN.clear()

# ---------------------------------------------------------------------------
print("\n== (C) the boot self-test, over fake copy handlers ==")


class Ops:
    """Bytes in fake tensors' `rows`, keyed by row index."""

    def fill_pattern(self, t, rows, seed):
        for i in range(rows):
            t.rows[i] = bytes(((seed * 31 + i * 7 + j) % 251) for j in range(t.shape[1]))

    def poison(self, t, rows):
        for i in range(rows):
            t.rows[i] = b"\x5a" * t.shape[1]

    def equal(self, a, ai, b, bi, n):
        return a.rows.get(ai, b"")[:n] == b.rows.get(bi, b"")[:n]


class Inner:
    """0.21's inner handler, simulated: copies rows per ref, records an event."""

    def __init__(self, gpu, host, refs, to_gpu, mode="ok"):
        self.gpu, self.host, self.refs, self.to_gpu, self.mode = gpu, host, refs, to_gpu, mode
        self._transfer_events, self.calls = {}, []

    def transfer_async(self, iid, src, dst):      # vLLM 0.31's signature
        gspec, cspec = (dst, src) if self.to_gpu else (src, dst)
        self.calls.append((iid, list(gspec.block_ids)))
        for r in self.refs[0]:
            t = r.tensor_idx
            for b, s in zip(gspec.block_ids, cspec.block_ids):
                b = int(b) + (1 if self.mode == "misdirect" and self.to_gpu else 0)
                if self.to_gpu:
                    self.gpu[t].rows[b] = self.host[t].rows.get(int(s), b"")
                else:
                    self.host[t].rows[int(s)] = self.gpu[t].rows.get(b, b"")
        self._transfer_events[iid] = Ev(
            False if self.mode == "hang" else
            RuntimeError("CUDA error: unspecified launch failure")
            if self.mode == "fault" else True)
        return True

    def get_finished(self):
        out = [k for k, e in self._transfer_events.items() if e.state is True]
        for k in out:
            self._transfer_events.pop(k)
        return out


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += 0.5


def mk_pair(mode="ok", load_rows=40, store_rows=8):
    h = mk_handler(ntensors=2, rows=22756, row_bytes=16, staging=load_rows)
    store = type("St", (), {})()
    store._staging = [T(0x200000000 + i * (1 << 24), store_rows, 16) for i in range(2)]
    h._inner = Inner(h._gpu_tensors, h._staging, h._refs, True, mode)
    store._inner = Inner(h._gpu_tensors, store._staging, h._refs, False)
    return h, store


ck = Clock()
h, store = mk_pair()
v = ST.run(h, store, ops=Ops(), clock=ck, sleep=ck.sleep, dirp=TMP)
names = [r["name"] for r in v["results"]]
eq("*** the control runs FIRST, then the incident set ***", names,
   ["control-low-16", "job5490-hang"])
eq("the control is blocks 1..n (never the null block 0)",
   h._inner.calls[0][1], list(range(1, 17)))
eq("*** the incident set is replayed into EXACTLY its ids, in ONE copy ***",
   h._inner.calls[1][1], INCIDENT)
check("both completed and read back equal",
      all(r["result"] == "completed" and r["match"] for r in v["results"]),
      repr(v["results"]))
eq("the readback is chunked to the store's 8 staging rows",
   len(store._inner.calls), 4)
eq("verdict state", v["state"], "done")
vf = json.load(open(os.path.join(TMP, "selftest-r0.json")))
eq("...and written to selftest-r<rank>.json", vf["failed"], [])

h, store = mk_pair(mode="misdirect")
v = ST.run(h, store, ops=Ops(), clock=Clock(), sleep=Clock().sleep, dirp=TMP)
check("*** a copy that lands in the wrong block is a MISMATCH, not a pass ***",
      v["results"][0]["match"] is False and v["results"][0]["mismatches"] > 0
      and v["failed"], repr(v["results"][0]))

ck = Clock()
h, store = mk_pair(mode="hang")
v = ST.run(h, store, ops=Ops(), clock=ck, sleep=ck.sleep, dirp=TMP)
eq("*** a copy that never completes TIMES OUT (polled, never synced) ***",
   v["results"][0]["result"], "timed_out")
check("...within the copy timeout", v["results"][0]["copy_ms"]
      <= (ST.COPY_TIMEOUT_S + 1) * 1e3)
check("...and the run stops: the next set is skipped, saying why",
      v["results"][1]["result"] == "skipped"
      and "timed_out" in v["results"][1]["why"])

h, store = mk_pair(mode="fault")
v = ST.run(h, store, ops=Ops(), clock=Clock(), sleep=Clock().sleep, dirp=TMP)
check("*** a fault is recorded with the CUDA error ***",
      v["results"][0]["result"] == "faulted"
      and "unspecified launch failure" in v["results"][0]["why"])

h, store = mk_pair(load_rows=8)
v = ST.run(h, store, ops=Ops(), clock=Clock(), sleep=Clock().sleep, dirp=TMP)
check("a set larger than load staging is SKIPPED, not split",
      v["results"][1]["result"] == "skipped"
      and "ONE copy" in v["results"][1]["why"])

json.dump({"sets": [{"name": "wrong-ids", "bids": [1988, 1990], "sha": "7513f576f389"},
                    {"name": "job902-like", "bids": list(range(1988, 2021))}]},
          open(os.path.join(TMP, ST.SETS_FILE), "w"))
h, store = mk_pair()
v = ST.run(h, store, ops=Ops(), clock=Clock(), sleep=Clock().sleep, dirp=TMP)
by = {r["name"]: r for r in v["results"]}
check("*** the operator's file adds sets; a sha mismatch is SKIPPED ***",
      by["wrong-ids"]["result"] == "skipped" and "sha mismatch" in by["wrong-ids"]["why"])
check("...a 33-block set gets its own 33-block control",
      "control-low-33" in by and by["job902-like"]["result"] == "completed")
os.unlink(os.path.join(TMP, ST.SETS_FILE))

ck = Clock()
ck.t = 0.0
h, store = mk_pair()
_orig = ST.TOTAL_BUDGET_S
ST.TOTAL_BUDGET_S = 0.0
v = ST.run(h, store, ops=Ops(), clock=ck, sleep=ck.sleep, dirp=TMP)
ST.TOTAL_BUDGET_S = _orig
check("the total budget bounds the run", all(r["result"] == "skipped"
                                             for r in v["results"]))

# ---------------------------------------------------------------------------
print("\n== (wiring) the hooks core calls, and the install ==")
_dio = harness.kvmgr_source("diskio")
check("the watchdog tick calls watchdog_tick",
      'hooks.call("watchdog_tick", verdict,' in _dio)
check("the load handler's constructor ends by calling disk_load_handler_built",
      'hooks.call("disk_load_handler_built", self, store)' in _dio)
check("both handler snapshots carry copy_probe",
      _dio.count("copy_probe=_copy_probe_snapshot(self)") == 2)
hooks._HOOKS.clear()
CP.install()
ST.install()
eq("copyprobe + selftest register exactly these hooks", sorted(hooks._HOOKS),
   ["copy_probe_snapshot", "disk_load_handler_built", "ledger_done",
    "ledger_dump", "ledger_submit", "watchdog_tick"])
check("selftest is OFF by default; the probe is ON",
      flags.DEFAULTS["diag_copy_selftest"] is False
      and flags.DEFAULTS["diag_copy_probe"] is True)

if not _had:
    del _torch.cuda
print()
print(f"{PASS} passed, {len(FAIL)} failed")
if harness.SKIPPED:
    print("SKIPPED:", harness.SKIPPED)
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
