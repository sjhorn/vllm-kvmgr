"""Phase 15: a failed restore becomes a recompute (kvmgr.loadfail).

Borrowed from sparkcache (FujitsuPolycom). Offline: the scheduler function under
test is the REAL one, exec'd out of the vendored vLLM source (in dspark mode,
DSpark's mnt10 scheduler, which is what the cluster runs), against fakes for the
objects it touches. The worker side (the handler reporting invalid blocks) is in
test_phase1b_disk.py section C9.
"""

from __future__ import annotations

import ast
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


base = harness.load_kv_offload()
from kvmgr import diskspec as D  # noqa: E402
from kvmgr import flags as KF  # noqa: E402
from kvmgr import loadfail as LF  # noqa: E402

K = base.make_offload_key
F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv15-"), "flags.json"))

DSPARK = bool(os.environ.get("KVMGR_WITH_DSPARK_PATCHES"))
if DSPARK:
    SCHED_SRC = os.path.join(harness.PATCHES, "mnt10-scheduler.py")
else:
    SCHED_SRC = os.path.join(REPO, "tests", "upstream", "vllm", "v1", "core",
                             "sched", "scheduler.py")
print("scheduler under test:", SCHED_SRC)

print("\n== (15a) the defect, on this build: one-group unpack on five groups ==")
_src = open(SCHED_SRC).read()
_fn = next(n for n in ast.walk(ast.parse(_src))
           if isinstance(n, ast.FunctionDef)
           and n.name == "_update_requests_with_invalid_blocks")
_code = "from typing import Iterable\n" + ast.get_source_segment(_src, _fn)
_ns: dict = {}
exec(compile(_code.replace("\n    def ", "\ndef ").replace("\n        ", "\n    "),
             SCHED_SRC, "exec"), _ns)
UPSTREAM = _ns["_update_requests_with_invalid_blocks"]
check("the function still unpacks ONE group (the line sparkcache 031 patches)",
      "(req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)"
      in ast.get_source_segment(_src, _fn))


class Req:
    def __init__(self, rid, computed):
        self.request_id = rid
        self.num_computed_tokens = computed


class KCM:
    def __init__(self, blocks):
        self.blocks = blocks           # rid -> tuple of per-group lists

    def get_block_ids(self, rid):
        return self.blocks[rid]


class Sched:
    def __init__(self, blocks, ngroups, block_size=4):
        self.kv_cache_manager = KCM(blocks)
        self.kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=[object()] * ngroups)
        self.block_size = block_size


FIVE = {"a": ([1, 2, 3], [11, 12], [21], [31], [41]),
        "b": ([4, 5], [13], [22], [32], [42])}
s5 = Sched(FIVE, 5)
try:
    UPSTREAM(s5, [Req("a", 12)], {12}, {})
    check("*** stock 0.21 RAISES on a five-group request ***", False)
except ValueError as e:
    check("*** stock 0.21 RAISES on a five-group request ***",
          "unpack" in str(e), str(e))

print("\n== (15b) the patched Scheduler: sparkcache 031 semantics on >1 group ==")
_sm = types.ModuleType("vllm.v1.core.sched.scheduler")
_sm.Scheduler = type("Scheduler", (), {
    "_update_requests_with_invalid_blocks": UPSTREAM,
    "__init__": lambda self, blocks, n, bs=4: Sched.__init__(self, blocks, n, bs)})
for _n in ("vllm.v1.core", "vllm.v1.core.sched"):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules["vllm.v1.core.sched"].scheduler = _sm
sys.modules["vllm.v1.core.sched.scheduler"] = _sm
LF._install_scheduler()
LF._install_scheduler()        # idempotent
check("installed once", getattr(_sm.Scheduler._update_requests_with_invalid_blocks,
                                "_kvmgr_wrapped", False))

# the index hook: a published offload scheduler with one load and one store job
_forgot: list = []
_KA, _KB, _KS = K(b"\x01" * 32, 0), K(b"\x02" * 32, 1), K(b"\x03" * 32, 0)
_osched = types.SimpleNamespace(
    _jobs={7: types.SimpleNamespace(req_id="a", keys={_KA, _KB}, is_store=False),
           8: types.SimpleNamespace(req_id="a", keys={_KS}, is_store=True),
           9: types.SimpleNamespace(req_id="b", keys={_KS}, is_store=False)},
    manager=types.SimpleNamespace(
        forget_failed_load=lambda keys: (_forgot.append(set(keys)), len(keys))[1]))
D._LIVE_OFFLOAD_SCHED[:] = [_osched]

S = _sm.Scheduler(FIVE, 5)
ra, rb = Req("a", 12), Req("b", 8)
harness.clear_log()
aff, tok, ev = S._update_requests_with_invalid_blocks([ra, rb], {31}, {"a": 0},
                                                     True)
eq("*** a bad block in ANY group (here group 3) affects the request ***", aff, {"a"})
eq("...which restarts from token 0", ra.num_computed_tokens, 0)
eq("...counting every computed token as affected", tok, 12)
eq("...evicting every group's blocks when asked", ev, {1, 2, 3, 11, 12, 21, 31, 41})
eq("an untouched request is left alone", rb.num_computed_tokens, 8)
eq("*** the failed request's LOAD keys leave the index (not its store's, not "
   "another request's) ***", _forgot, [{_KA, _KB}])
check("...and it is logged", "failed to load" in harness.log_text())
ra2 = Req("a", 12)
_, _, ev2 = S._update_requests_with_invalid_blocks([ra2], {1}, {"a": 4}, False)
eq("async (evict_blocks=False): nothing to evict", ev2, set())
eq("...and tokens already scheduled this step are not counted as computed",
   LF.STATS["hma_tokens"], 12 + 8)

print("\n== (15c) one group: upstream's code, untouched ==")
ONE = {"c": ([1, 2, 3, 4],)}
S1 = _sm.Scheduler(ONE, 1)
rc = Req("c", 16)
_forgot.clear()
aff1, tok1, ev1 = S1._update_requests_with_invalid_blocks([rc], {3}, {}, True)
eq("single group: truncated at the first bad block, as upstream does",
   rc.num_computed_tokens, 8)
eq("...affected", aff1, {"c"})
eq("...evicting the bad block and everything after it", ev1, {3, 4})

print("\n== (15d) the connector reports what the worker noted ==")
_cm = types.ModuleType(LF._CONNECTOR_MODULES[0])


class OffloadingConnector:
    def get_block_ids_with_load_errors(self):
        return {99}


_cm.OffloadingConnector = OffloadingConnector
sys.modules[LF._CONNECTOR_MODULES[0]] = _cm
LF._install_connector()
LF._install_connector()
c = OffloadingConnector()
eq("nothing failed: only what upstream reports", c.get_block_ids_with_load_errors(),
   {99})
LF.note_failed([5, 6])
LF.note_failed([6, 7])
eq("*** a failed load's blocks are reported, with upstream's ***",
   c.get_block_ids_with_load_errors(), {5, 6, 7, 99})
eq("...once: the report drains them", c.get_block_ids_with_load_errors(), {99})
eq("counted", (LF.STATS["noted_jobs"], LF.STATS["drained_blocks"]), (2, 3))

print("\n== (15d2) 0.31: a failed load is reported by REQUEST, never by block ==")
_hib = next(n for n in ast.walk(ast.parse(_src))
            if isinstance(n, ast.FunctionDef) and n.name == "_handle_invalid_blocks")
check("UPSTREAM'S SHAPE: _handle_invalid_blocks raises on >1 group before our "
      "multi-group fix is reached (found on the spike, Gemma 3)",
      "if len(self.kv_cache_config.kv_cache_groups) > 1:"
      in ast.get_source_segment(_src, _hib)
      and "raise RuntimeError" in ast.get_source_segment(_src, _hib))
_hfr = next(n for n in ast.walk(ast.parse(_src))
            if isinstance(n, ast.FunctionDef) and n.name == "_handle_failed_recving")
check("...and _handle_failed_recving restarts the request from token 0",
      "request.num_computed_tokens = 0" in ast.get_source_segment(_src, _hfr))


class _Results:
    def __init__(self, fin):
        self.finished_sending, self.finished_recving = set(), set(fin)
        self.failed_recving = set()


class OffloadingConnector31:
    def __init__(self):
        self.connector_worker = types.SimpleNamespace(_load_jobs={3: "r3", 4: "r4"})
        self.finish = set()

    def get_block_ids_with_load_errors(self):
        return set()

    def get_transfer_results(self, finished_req_ids):
        # upstream pops a finished job's mapping, and the handler's
        # get_finished (inside this call) is where the failure is noted
        w = self.connector_worker
        for jid in [j for j, r in w._load_jobs.items() if r in self.finish]:
            if jid == 3:
                LF.note_failed([10, 11, 12], job_id=3)
            del w._load_jobs[jid]
        return _Results(self.finish)


_cm.OffloadingConnector = OffloadingConnector31
LF._install_connector()
LF._install_connector()
c31 = OffloadingConnector31()
c31.finish = {"r4"}
r = c31.get_transfer_results(set())
eq("a good load finishing: no failures", (r.finished_recving, r.failed_recving),
   ({"r4"}, set()))
c31.finish = {"r3"}
r = c31.get_transfer_results(set())
eq("*** a failed job's request is in failed_recving, in the same result that "
   "finishes it ***", (r.finished_recving, r.failed_recving), ({"r3"}, {"r3"}))
eq("*** ...and NO block ids: they raise on a multi-group model ***",
   c31.get_block_ids_with_load_errors(), set())
eq("...reported once", LF.snapshot()["pending_jobs"], 0)
check("request-level mode is on", LF.snapshot()["request_level"])

_sm.Scheduler._handle_failed_recving = (
    lambda self, ids: (setattr(self, "_seen", set(ids)), set())[1])
del _sm.Scheduler._update_requests_with_invalid_blocks._kvmgr_wrapped
LF._install_scheduler()
check("the scheduler's failed_recving handler is wrapped",
      getattr(_sm.Scheduler._handle_failed_recving, "_kvmgr_wrapped", False))
_forgot.clear()
harness.clear_log()
S = _sm.Scheduler(FIVE, 5)
eq("upstream's handler runs and its answer is passed through",
   (S._handle_failed_recving({"a"}), S._seen), (set(), {"a"}))
eq("*** the failed request's LOAD keys leave the index ***", _forgot, [{_KA, _KB}])
check("...and it is logged", "request-level" in harness.log_text())
_dsrc = open(os.path.join(REPO, "src", "kvmgr", "diskio.py")).read()
check("the load handler names the job when it notes a failure",
      'loadfail.note_failed(st.get("bids_all") or (), job_id=job_id)' in _dsrc)
LF._REQUEST_LEVEL[0] = False

print("\n== (15e) the presence manager drops a failed load's keys, journalled ==")


class FakeMapper:
    def __init__(self, root):
        self.root = root

    def get_file_name(self, key):
        return os.path.join(self.root, bytes(key).hex() + ".bin")


tmp = tempfile.mkdtemp(prefix="kv15-mgr-")
F.write(disk_restore=1, disk_verify_enabled=0, journal_enabled=1)
KF.reset_for_tests()
m = D.DiskPresenceManager(FakeMapper(tmp), num_workers=2,
                          manifest_path=os.path.join(tmp, "presence.json"))
keys = [K(bytes([i]) * 32, 0) for i in range(1, 5)]
for k in keys:
    open(FakeMapper(tmp).get_file_name(k), "wb").write(b"x")
    m._remember(k, 0)
harness.clear_log()
n = m.forget_failed_load(keys[:2] + [K(b"\x09" * 32, 0)])
eq("the keys present are dropped (one was never there)", n, 2)
check("...gone from the index", all(bytes(k) not in m._present for k in keys[:2])
      and all(bytes(k) in m._present for k in keys[2:]))
eq("...counted", m._stats.get("load_failed_forgot"), 2)
_j = m._journal_names()[0]
check("*** ...and journalled, so a reboot cannot resurrect them ***",
      _j and os.path.exists(_j) and '"r"' in open(_j).read(),
      open(_j).read() if _j and os.path.exists(_j) else "no journal")
check("...and logged", "turned into a recompute" in harness.log_text())

print("\n== (15f) wiring ==")
_reg = open(os.path.join(REPO, "src", "kvmgr", "register.py")).read()
check("register() installs it", "loadfail.install()" in _reg)
check("the knob is declared, default OFF",
      KF.DEFAULTS.get("disk_load_failure_recompute") is False)
F.write(disk_load_failure_recompute=1)
KF.reset_for_tests()
check("...and read hot", LF.enabled())
F.write()
KF.reset_for_tests()
check("...off again", not LF.enabled())

print()
print(f"{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)
