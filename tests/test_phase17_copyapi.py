"""Phase 17: load_copy_api -- how a load's descriptors reach the driver (ask9).

Class B is a fault in libcuda's cuMemcpyBatchAsync on GB10. `chunk` and `loop`
must issue exactly the batch's (src, dst, size) triples, in order, all of
them; `chunk` never more than the cap per call; every call between the copy's
start and end events; and only for LOADS.
"""

from __future__ import annotations

import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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


harness.load_kv_offload()
from kvmgr import copyapi as CA  # noqa: E402
from kvmgr import flags as KF  # noqa: E402

F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv17-"), "flags.json"))

# -- a copy handler shaped like upstream's: events around ONE ops call -------
LOG: list = []


class RealOps:
    """Stands in for vllm._custom_ops: records every batch call it gets."""
    other = "passed through"

    def swap_blocks_batch(self, src, dst, sizes, is_src_access_order_any=False):
        LOG.append(("batch", list(src), list(dst), list(sizes),
                    is_src_access_order_any))


_mod = types.ModuleType("fake_gpu_worker")
_mod.ops = RealOps()
exec(compile('''
class Handler:
    def __init__(self, gpu_to_cpu):
        self.gpu_to_cpu = gpu_to_cpu

    def transfer_async(self, job_id, src, dst, sizes):
        LOG.append(("start", job_id))
        ops.swap_blocks_batch(src, dst, sizes,
                              is_src_access_order_any=not self.gpu_to_cpu)
        LOG.append(("end", job_id))
        return True
''', "fake_gpu_worker", "exec"), _mod.__dict__)
_mod.LOG = LOG
LoadH, StoreH = _mod.Handler(False), _mod.Handler(True)

# loop: record instead of calling libcuda
CA._stream_handle = lambda: "the-transfer-stream"
CA.memcpy_async = lambda d, s, n, st: LOG.append(("memcpy", d, s, n, st))

N = 150
SRC = list(range(1000, 1000 + N))
DST = list(range(5000, 5000 + N))
SZ = [37376] * N
TRIPLES = list(zip(SRC, DST, SZ))


def triples(log):
    out = []
    for e in log:
        if e[0] == "batch":
            out += list(zip(e[1], e[2], e[3]))
        elif e[0] == "memcpy":
            out.append((e[2], e[1], e[3]))
    return out


def run(mode, chunk=64, h=LoadH, job=1):
    LOG.clear()
    with CA.active(mode, chunk) as rec:
        h.transfer_async(job, SRC, DST, SZ)
    return rec


print("== (17a) install: the proxy goes where the handler looks up `ops` ==")
eq("installs", CA.install_on(LoadH), None)
eq("...idempotent", CA.install_on(LoadH), None)
check("the module's ops is now the proxy, over the real one",
      getattr(_mod.ops, "_kvmgr_real", None).__class__ is RealOps)
eq("other attributes pass through", _mod.ops.other, "passed through")
_bad = types.ModuleType("nobatch")
exec("class H:\n    def transfer_async(self): pass\n", _bad.__dict__)
check("a module with no ops.swap_blocks_batch is refused, with a reason",
      "no ops.swap_blocks_batch" in (CA.install_on(_bad.H()) or ""))

print("\n== (17b) batch, and every caller outside a load: upstream's one call ==")
LOG.clear()
StoreH.transfer_async(9, SRC, DST, SZ)
eq("*** a store (no mode set) makes ONE batch call ***",
   [e[0] for e in LOG], ["start", "batch", "end"])
eq("...with its own access order untouched", LOG[1][4], False)
rec = run("batch")
eq("batch: one call, every triple", (rec["calls"], triples(LOG)), (1, TRIPLES))
eq("...the load's access order passed through", LOG[1][4], True)

print("\n== (17c) chunk: the same triples, in order, never more than the cap ==")
for c in (64, 7, 1, 150, 1000):
    rec = run("chunk", c)
    calls = [e for e in LOG if e[0] == "batch"]
    eq("chunk %d: every triple, in order" % c, triples(LOG), TRIPLES)
    check("chunk %d: no call over the cap" % c,
          all(len(e[3]) <= c for e in calls), [len(e[3]) for e in calls])
    eq("chunk %d: calls = ceil(N/cap), as calls_for says" % c,
       (len(calls), rec["calls"]), (CA.calls_for("chunk", N, c),) * 2)
    check("chunk %d: each call keeps the access order" % c,
          all(e[4] is True for e in calls))

print("\n== (17d) loop: one cuMemcpyAsync per descriptor, on the copy stream ==")
rec = run("loop")
mc = [e for e in LOG if e[0] == "memcpy"]
eq("every triple, in order, as (dst, src, n)", triples(LOG), TRIPLES)
eq("one call each", (len(mc), rec["calls"]), (N, N))
check("*** never the batch API ***", not any(e[0] == "batch" for e in LOG))
check("...all on the stream current inside the handler",
      all(e[4] == "the-transfer-stream" for e in mc))

print("\n== (17e) the events bracket every call ==")
for mode in ("batch", "chunk", "loop"):
    run(mode, 16)
    kinds = [e[0] for e in LOG]
    check("*** %s: start first, end last, every copy between ***" % mode,
          kinds[0] == "start" and kinds[-1] == "end"
          and "start" not in kinds[1:] and "end" not in kinds[:-1], kinds[:4])

print("\n== (17f) the mode is per thread and per call ==")
LOG.clear()
with CA.active("loop", 64):
    StoreH.transfer_async(2, SRC[:3], DST[:3], SZ[:3])
check("inside a load's scope even the store handler loops (scope is the call)",
      [e[0] for e in LOG].count("memcpy") == 3)
LOG.clear()
StoreH.transfer_async(3, SRC[:3], DST[:3], SZ[:3])
eq("...and after it, batch again", [e[0] for e in LOG], ["start", "batch", "end"])
import threading  # noqa: E402
seen = []
with CA.active("loop", 64):
    t = threading.Thread(target=lambda: (LoadH.transfer_async(4, SRC[:2], DST[:2],
                                                              SZ[:2]),
                                         seen.extend(LOG)))
    LOG.clear()
    t.start()
    t.join()
check("*** another thread is not affected by this thread's mode ***",
      any(e[0] == "batch" for e in seen) and not any(e[0] == "memcpy" for e in seen))

print("\n== (17g) the knobs: hot, validated, defaulted ==")
F.write()
KF.reset_for_tests()
eq("defaults: batch, 64", CA.selected(), ("batch", 64))
F.write(load_copy_api="chunk", load_copy_chunk=256)
KF.reset_for_tests()
eq("read from the flags", CA.selected(), ("chunk", 256))
F.write(load_copy_api="LOOP ")
KF.reset_for_tests()
eq("case and spaces forgiven", CA.selected()[0], "loop")
F.write(load_copy_api="turbo", load_copy_chunk=0)
KF.reset_for_tests()
harness.clear_log()
eq("*** an unknown mode is batch, never a guess ***", CA.selected(), ("batch", 64))
check("...said", "not one of" in harness.log_text())
eq("calls_for", [CA.calls_for(m, n, 64) for m, n in
                 (("batch", 2046), ("chunk", 2046), ("loop", 2046), ("chunk", 0))],
   [1, 32, 2046, 0])
F.write()
KF.reset_for_tests()

print("\n== (17h) wiring: the load handler, the ledger, /kv/workers, upstream ==")
_d = open(os.path.join(REPO, "src", "kvmgr", "diskio.py")).read()
_load = _d[_d.index("class DiskLoadHandler"):]
check("the load handler installs the proxy on its inner handler",
      "copyapi.install_on(self._inner)" in _load)
_sub = _load.index("self._ledger_submit(\"load\"")
_call = _load.index("ok = self._inner.transfer_async(", _sub)
check("*** the path is chosen and named BEFORE the ledger line ***",
      _load.index("self._copy_last = {") < _sub)
check("*** the inner call runs inside copyapi.active ***",
      _load.rindex("with copyapi.active(_mode, _chunk)", 0, _call) > _sub)
_store = _d[_d.index("class DiskStoreHandler"):_d.index("class DiskLoadHandler")]
check("*** the store handler never sets a mode ***", "copyapi" not in _store)
check("/kv/workers shows both knobs", "copy_api=" in _load and "copy_chunk=" in _load)
_l = open(os.path.join(REPO, "src", "kvmgr", "diag", "ledger.py")).read()
check("the ledger's load submit line carries copy_api and copy_calls",
      '"_copy_last"' in _l and '"copy_calls": None' in _l)
_up = open(os.path.join(REPO, "tests", "upstream", "vllm", "v1", "kv_offload",
                        "cpu", "gpu_worker.py")).read()
check("UPSTREAM'S SHAPE: the copy is `ops` from `vllm._custom_ops`",
      "from vllm import _custom_ops as ops" in _up)
_ta = _up[_up.index("    def transfer_async("):]
_ta = _ta[:_ta.index("\n    def ", 10)]
check("UPSTREAM'S SHAPE: ONE ops.swap_blocks_batch, between the start and end "
      "events, inside the copy stream",
      _ta.count("ops.swap_blocks_batch(") == 1
      and _ta.index("start_event.record(stream)")
      < _ta.index("ops.swap_blocks_batch(")
      < _ta.index("end_event.record(stream)")
      and _ta.index("with torch.cuda.stream(stream):")
      < _ta.index("start_event.record(stream)"))
_r = open(os.path.join(REPO, "src", "kvmgr", "diag", "reload.py")).read()
check("a reload keeps the thread-local the installed proxy reads",
      '("kvmgr.copyapi", "_TLS"' in _r)

print()
print(f"{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)
