"""test_phase1_disk.py -- the scheduler-side half of our disk spec.

    python3.12 local-ops/kvharness/test_phase1_disk.py

Phase 1a is the SPEC, the MANAGER and the path arithmetic -- everything that
decides *what* gets written and *where*. The transfer itself (Phase 1b) is the
only part that needs a GPU, so it is a loud refusal here rather than a no-op.

The assertions worth reading are (1e), which pins that this phase is write-only
BY CONSTRUCTION rather than by omission, and (1c), which pins the per-worker
slice -- the number whose absence is the entire reason this module exists.
"""

from __future__ import annotations

import os
import sys
import tempfile

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


base = harness.load_kv_offload()
from kvmgr import diskspec, flags  # noqa: E402

F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kvdisk-"), "flags.json"))

# The live shape, so the arithmetic is checked against the box and not a toy.
WORLD = 2
NUM_BLOCKS = 22756            # cache_config_info on the running engine
PER_RANK = 1_051_776          # measured: the non-zero half of every file
WHOLE_PAGE = PER_RANK * WORLD  # 2,103,552, also measured


class _T:
    def __init__(self, size):
        self.size = size


class _KVCfg:
    num_blocks = NUM_BLOCKS
    # total GPU bytes such that (total // num_blocks) == PER_RANK exactly
    kv_cache_tensors = [_T(PER_RANK * NUM_BLOCKS)]

    def __init__(self):
        g = type("G", (), {})()
        g.kv_cache_spec = type("S", (), {"block_size": 256})()
        g.layer_names = ["l0"]
        self.kv_cache_groups = [g]


class _Cfg:
    def __init__(self, world=WORLD, rank=0):
        self.parallel_config = type("P", (), {
            "world_size": world, "rank": rank,
            "decode_context_parallel_size": 1,
            "prefill_context_parallel_size": 1,
            "tensor_parallel_size": world, "pipeline_parallel_size": 1,
        })()
        self.cache_config = type("C", (), {"block_size": 256,
                                           "cache_dtype": "nvfp4_ds_mla"})()
        self.model_config = type("M", (), {
            "model": "deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"})()
        self.kv_transfer_config = type("K", (), {
            "kv_connector_extra_config": {"root_dir": "/kvoffload"}})()


def mkspec(world=WORLD, rank=0):
    flags.reset_for_tests()
    return diskspec.DSparkDiskOffloadingSpec(_Cfg(world, rank), _KVCfg())


print("== (1a) the medium, and that it cannot collide ==")
eq("our medium", diskspec.DISK_MEDIUM, "DSPARK_DISK")
check("it is not one of upstream's", diskspec.DISK_MEDIUM not in ("CPU", "GPU"))
eq("DiskLoadStoreSpec reports it", diskspec.DiskLoadStoreSpec([]).medium(),
   "DSPARK_DISK")
check("DiskLoadStoreSpec IS a LoadStoreSpec",
      issubclass(diskspec.DiskLoadStoreSpec, base.LoadStoreSpec))
check("...and is concrete (medium implemented)",
      not diskspec.DiskLoadStoreSpec.__abstractmethods__)

print()
print("== (1b) the spec satisfies the contract Phase 0 pinned ==")
sp = mkspec()
check("DSparkDiskOffloadingSpec IS an OffloadingSpec",
      isinstance(sp, base.OffloadingSpec))
check("...and is instantiable, i.e. implements every abstract method",
      not type(sp).__abstractmethods__, repr(type(sp).__abstractmethods__))
eq("block_size_factor stays 1 (we never set a uniform block size)",
   sp.block_size_factor, 1)
mgr = sp.get_manager()
check("get_manager returns an OffloadingManager",
      isinstance(mgr, base.OffloadingManager))
check("...and the same one twice (the scheduler holds one)",
      sp.get_manager() is mgr)

print()
print("== (1c) THE PER-WORKER SLICE -- the number this module exists for ==")
eq("whole page, as the tiering tier computes it", sp.whole_page_bytes, WHOLE_PAGE)
eq("*** per-worker slice, what WE write ***", sp.page_size_per_worker, PER_RANK)
check("they differ by exactly world_size -- the half-empty cause in one line",
      sp.whole_page_bytes == sp.page_size_per_worker * WORLD)
# A single-node TP must collapse to the same number, or the arithmetic is
# accidental rather than derived.
sp1 = mkspec(world=1)
eq("at world_size=1 the two coincide", sp1.whole_page_bytes,
   sp1.page_size_per_worker)

print()
print("== (1d) the on-disk root is VERSIONED, which is correctness ==")
# 6,141 half-empty files from the tiering era are still on disk under the
# fingerprint FileMapper computes. Sharing a root would let exists() find one.
check("our root carries a format segment", diskspec.DISK_FORMAT_ROOT in sp.root_dir)
check("...and it is not the bare mount", sp.root_dir != "/kvoffload")
check("the tiering tier's own path could not match ours",
      not sp.root_dir.rstrip("/").endswith("kvoffload"))

print()
print("== (1e) WRITE-ONLY BY CONSTRUCTION, not by omission ==")
ctx = base.ReqContext(req_id="r1")
k1 = base.make_offload_key(b"\x11" * 32, 0)
k2 = base.make_offload_key(b"\x22" * 32, 1)

F.write(disk_restore=0)
flags.reset_for_tests()
eq("lookup refuses while disk_restore is off", mgr.lookup(k1, ctx), False)
mgr.complete_store([k1], ctx)      # even for a key we DO hold
eq("...even for a key the manager holds", mgr.lookup(k1, ctx), False)
refused = False
try:
    mgr.prepare_load([k1], ctx)
except RuntimeError:
    refused = True
check("*** prepare_load RAISES rather than best-efforting ***", refused)

F.write(disk_restore=1)
flags.reset_for_tests()
eq("with the flag on, a held key is a hit", mgr.lookup(k1, ctx), True)
eq("...and an unheld one is not", mgr.lookup(k2, ctx), False)
spec = mgr.prepare_load([k1], ctx)
check("prepare_load now returns our disk spec",
      isinstance(spec, diskspec.DiskLoadStoreSpec))
eq("...carrying the keys verbatim", list(spec.keys), [k1])
F.write(disk_restore=0)
flags.reset_for_tests()

print()
print("== (1f) prepare_store: the dedup that made boot 2 write zero files ==")
m2 = diskspec.DiskPresenceManager(sp.file_mapper(), num_workers=WORLD)
out = m2.prepare_store([k1, k2], ctx)
eq("a cold pair is all to store", sorted(out.keys_to_store), sorted([k1, k2]))
check("store_spec is ours", isinstance(out.store_spec, diskspec.DiskLoadStoreSpec))
eq("nothing is displaced yet (eviction is Phase 5)", out.evicted_keys, [])
m2.complete_store([k1], ctx)
out2 = m2.prepare_store([k1, k2], ctx)
eq("*** a key already held is SKIPPED ***", out2.keys_to_store, [k2])
m2.complete_store([k2], ctx)
# *** THIS ASSERTED None AND THAT WAS THE BUG. *** scheduler.py treats None as
# "I cannot store" and does NOT advance the request's stored index, so the very
# same blocks are attempted again every step, indefinitely -- measured on the
# box 2026-10-08 as 3,309 "cannot store blocks" warnings at ~20/second. An
# EMPTY output means "nothing new here" and lets the request progress. The two
# are six lines apart in upstream's own code and are not interchangeable.
_held = m2.prepare_store([k1, k2], ctx)
check("*** when everything is held, prepare_store returns an EMPTY OUTPUT, "
      "never None -- None makes the scheduler retry forever ***",
      _held is not None and list(_held.keys_to_store) == [],
      repr(_held)[:70])

print()
print("== (1g) complete_store is the ALL-RANKS edge, and it is free ==")
# OffloadingWorkerMetadata: workers report {job_id: 1}, aggregate() sums across
# workers, and the scheduler "processes a transfer completion only when count
# reaches num_workers". So presence recorded HERE is all-ranks correct with no
# protocol of ours. Pinned against the pristine source, not just asserted.
common = open(os.path.join(REPO, "patches", "pristine", "vllm",
                           "kv_connector_offloading", "common.py")).read()
check("the framework really does gate on num_workers",
      "only when count reaches num_workers" in common)
check("...by summing per-worker reports", "aggregate" in common)
sched = open(os.path.join(REPO, "patches", "pristine", "vllm",
                          "kv_connector_offloading", "scheduler.py")).read()
check("...and the scheduler sets pending_count from num_workers",
      "pending_count=self.config.num_workers" in sched)
_flat = "".join(sched.split())
check("...and num_workers IS world_size, so 'all ranks' means both nodes",
      "num_workers=spec.vllm_config.parallel_config.world_size" in _flat)
eq("our manager records the rank count it was built with",
   m2.snapshot()["num_workers"], WORLD)

print()
print("== (1h) events, so a write is observable ==")
m3 = diskspec.DiskPresenceManager(sp.file_mapper(), num_workers=WORLD)
m3.complete_store([k1, k2], ctx)
evs = list(m3.take_events())
eq("one event for the batch", len(evs), 1)
eq("...on our medium", evs[0].medium, "DSPARK_DISK")
check("...marked stored, not removed", evs[0].removed is False)
eq("events are TAKEN, not re-reported", list(m3.take_events()), [])
m3.complete_store([k1], ctx)
eq("a re-store of a held key emits nothing", list(m3.take_events()), [])

print()
print("== (1i) get_handlers registers BOTH directions, ALWAYS ==")
# The handlers need a GPU, but get_handlers imports them INSIDE the function, so
# patching the module attributes gives a real test of the registration itself --
# which is the part that decides whether a restore is even routable.
from kvmgr import diskio  # noqa: E402

_built = {}


class _FakeHandler:
    def __init__(self, **kw):
        _built.update(kw)


_realS, _realL = diskio.DiskStoreHandler, diskio.DiskLoadHandler


def _triples():
    diskio.DiskStoreHandler = diskio.DiskLoadHandler = _FakeHandler
    try:
        return list(sp.get_handlers(kv_caches=object()))
    finally:
        diskio.DiskStoreHandler, diskio.DiskLoadHandler = _realS, _realL


# *** THIS ASSERTED "STORE ONLY" AND THAT DESIGN KILLED THE ENGINE. ***
# Withholding the DISK->GPU handler when disk_restore was off was written up as
# a second gate -- "done twice so a later optimistic lookup() still cannot
# restore". It was wrong about the FAILURE MODE. The scheduler reads
# disk_restore HOT, so flipping it on a running engine makes lookup() claim keys
# and prepare_load issue load jobs while the worker still has no handler:
#     worker.py:133  assert handler is not None   -> AssertionError
#     RuntimeError: Worker failed with error ''
# A bare assert in a worker, which took the engine core with it. Measured
# 2026-10-08 at 205K tokens.
#
# So the property to assert is the opposite: the HANDLER SET MUST NOT DEPEND ON
# THE FLAG. lookup() is the only gate, and it is structural -- the scheduler
# only calls prepare_load for keys lookup claimed.
for _flag in (0, 1):
    F.write(kvmgr_enabled=1, disk_restore=_flag)
    flags.reset_for_tests()
    _t = _triples()
    eq(f"with disk_restore={_flag}: BOTH directions registered", len(_t), 2)
    _pairs = sorted((a.medium(), b.medium()) for a, b, _ in _t)
    eq(f"...the pairs, disk_restore={_flag}", _pairs,
       [("DSPARK_DISK", "GPU"), ("GPU", "DSPARK_DISK")])

F.write(kvmgr_enabled=1, disk_restore=0)
flags.reset_for_tests()
_off = sorted((a.medium(), b.medium()) for a, b, _ in _triples())
F.write(kvmgr_enabled=1, disk_restore=1)
flags.reset_for_tests()
_on = sorted((a.medium(), b.medium()) for a, b, _ in _triples())
eq("*** the handler set is IDENTICAL either way -- which is also why the two "
   "ranks can no longer disagree about it ***", _off, _on)

# and the gate that remains must still refuse
F.write(kvmgr_enabled=1, disk_restore=0)
flags.reset_for_tests()
_m = sp.get_manager()
check("*** with restore off, lookup() still refuses -- the ONE gate, and it is "
      "structural ***", _m.lookup(k1, ctx) is False)

eq("the handler is built with our staging size", _built.get("staging_blocks"), 32)
check("...and gets the spec, so it can reach file_mapper()", _built.get("spec") is sp)

# ...and with the flag ON, BOTH directions appear. This is the one knob that
# needs a restart -- handlers are registered once at boot -- so it is read once
# here rather than per transfer, and the code says so rather than offering a
# hot flip that could not work.
F.write(disk_restore=1)
flags.reset_for_tests()


class _FakeLoad:
    def __init__(self, **kw):
        pass


_rl = diskio.DiskLoadHandler
diskio.DiskLoadHandler = _FakeLoad
diskio.DiskStoreHandler = _FakeHandler
try:
    both = list(mkspec().get_handlers(kv_caches=object()))
finally:
    diskio.DiskLoadHandler = _rl
    diskio.DiskStoreHandler = _realS
    F.write(disk_restore=0)
    flags.reset_for_tests()

eq("with disk_restore ON there are two handlers", len(both), 2)
eq("...the store direction", (both[0][0].medium(), both[0][1].medium()),
   ("GPU", "DSPARK_DISK"))
eq("...and the load direction", (both[1][0].medium(), both[1][1].medium()),
   ("DSPARK_DISK", "GPU"))

print()
print("== (1j) the file path: per-rank, per-group, under our root ==")
fm = sp.file_mapper()
n1 = fm.get_file_name(k1)
check("the path is under our versioned root", diskspec.DISK_FORMAT_ROOT in n1)
check("...carries this rank", "_r0" in n1)
check("...and the key's group", "_g0" in n1)
eq("...and ends in .bin", os.path.splitext(n1)[1], ".bin")
check("a different group gives a different path", fm.get_file_name(k2) != n1)
check("...and group 1 is named as such", "_g1" in fm.get_file_name(k2))
# rank is in the PATH and outside the fingerprint, which is what makes _r0/_r1
# siblings under ONE fingerprint -- each rank restoring its own half.
fm1 = mkspec(rank=1).file_mapper()
eq("rank 1's path differs only in the rank segment",
   fm1.get_file_name(k1).replace("_r1", "_rX"), n1.replace("_r0", "_rX"))

print()
print("== (1z) A REBOUND INSTANCE: a reload never runs __init__ ==")
# *** THIS EXACT GAP CRASHED THE ENGINE ON 2026-10-08. ***
#
#   AttributeError: 'DiskPresenceManager' object has no attribute '_gcounts'
#       in prepare_store  ->  fatal to the scheduler
#
# `POST /kv/reload` re-points LIVE objects with `obj.__class__ = new`. The
# instance therefore gets every new METHOD and never the new `__init__`, so any
# attribute introduced alongside a method is absent on exactly the objects that
# matter -- the ones already serving. Adding per-group counters created
# `_gcounts` in __init__ and read it in prepare_store, and the live manager had
# the reader without the attribute.
#
# The property, stated so it survives the next instrumentation attribute:
# EVERY ATTRIBUTE ADDED FOR OBSERVABILITY MUST BE OPTIONAL. The data-path
# attributes (_present, _mapper) are legitimately required -- a manager without
# them is broken, not degraded -- so they are not in this set.
_OPTIONAL_ATTRS = ("_gcounts",)

for _attr in _OPTIONAL_ATTRS:
    m_rb = diskspec.DiskPresenceManager(sp.file_mapper(), num_workers=WORLD)
    check(f"{_attr} is created by __init__ (so a FRESH instance has it)",
          hasattr(m_rb, _attr))
    delattr(m_rb, _attr)          # <- precisely what a rebind leaves behind
    _ok, _why = True, ""
    try:
        m_rb.prepare_store([k1, k2], ctx)
        m_rb.complete_store([k1, k2], ctx)
        m_rb.lookup(k1, ctx)
        _snap = m_rb.snapshot()
    except Exception as _exc:
        _ok, _why = False, repr(_exc)
    check(f"*** a REBOUND manager missing {_attr} still serves the data path "
          f"*** {_why}", _ok)
    check(f"...and snapshot() still answers, with by_group present",
          _ok and "by_group" in _snap)
    check(f"...and the counter re-initialises itself rather than staying gone",
          _ok and hasattr(m_rb, _attr))

# The lint that generalises: every read of an optional attribute must go
# through getattr, because a direct `self._x` is what fails on a rebind. A
# test of one attribute would pass again while the next one broke the engine.
_dsrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskspec.py")).read()
for _attr in _OPTIONAL_ATTRS:
    # Exclude the declaration itself. It is an ANNOTATED assignment
    # (`self._gcounts: dict[...] = {}`), so a plain `self._x =` test misses it
    # and the lint failed for the wrong reason on its first run -- which is its
    # own lesson in this repo and worth not repeating silently.
    def _is_decl(line, attr):
        t = line.strip()
        return t.startswith(f"self.{attr} =") or t.startswith(f"self.{attr}:")

    _direct = [ln.strip() for ln in _dsrc.splitlines()
               if f"self.{_attr}" in ln and "getattr" not in ln
               and not ln.strip().startswith("#")
               and not _is_decl(ln, _attr)]
    eq(f"every read of self.{_attr} outside __init__ uses getattr", _direct, [])

print("== (1x) capturing the connector scheduler off the CALL STACK ==")
# A gc scan cannot find it: EngineCore.__init__ calls freeze_gc_heap() ->
# gc.freeze(), and gc.get_objects() does not return frozen objects. The
# connector scheduler is built at boot, so it is permanently invisible to a
# scan -- measured live, "no live instance after 17,003 objects", which is
# implausibly few for an engine core and is the tell.
#
# It is, however, the thing that CALLS us, so its `self` is in a caller frame.
del diskspec._LIVE_OFFLOAD_SCHED[:]


class OffloadingConnectorScheduler:  # name is what is matched, deliberately
    def call_it(self, mgr, key, ctx):
        return mgr.lookup(key, ctx)


class SomethingElse:
    def call_it(self, mgr, key, ctx):
        return mgr.lookup(key, ctx)


_m = diskspec.DiskPresenceManager(sp.file_mapper(), num_workers=WORLD)
check("nothing captured before any call", not diskspec._LIVE_OFFLOAD_SCHED)
_wrong = SomethingElse()
_wrong.call_it(_m, k1, ctx)
check("*** a caller of the WRONG type is not captured ***",
      not diskspec._LIVE_OFFLOAD_SCHED)
_right = OffloadingConnectorScheduler()
_right.call_it(_m, k1, ctx)
eq("*** the connector scheduler IS captured, from the call stack ***",
   diskspec._LIVE_OFFLOAD_SCHED and diskspec._LIVE_OFFLOAD_SCHED[0] is _right,
   True)
# *** AND "len == 1" WAS A VACUOUS ONE-SHOT TEST. *** The capture clears the
# list before appending, so removing the early return leaves the length at 1 and
# the assertion passes while the stack walk runs on every single lookup.
# Break-verification caught it. The behavioural difference is STABILITY: once a
# handle is captured it must not be replaced, because a later caller is just as
# valid a match and the handle would flap between request-scoped objects.
_right2 = OffloadingConnectorScheduler()
_right2.call_it(_m, k2, ctx)
eq("*** the captured handle is STABLE -- a later caller does not replace it ***",
   diskspec._LIVE_OFFLOAD_SCHED[0] is _right, True)
eq("...and there is still exactly one", len(diskspec._LIVE_OFFLOAD_SCHED), 1)

# prepare_store is the SECOND capture point, and the one that matters in
# practice: `lookup` is only reached on an external prefix-cache path and may
# not run for hours, while prepare_store is called on every request that fills
# a block -- even with the tier paused, because our pause gate sits INSIDE the
# call rather than preventing it. Verified with the pause on, which is the
# state the box actually sits in.
del diskspec._LIVE_OFFLOAD_SCHED[:]
flags.set_override("disk_store_paused", True)
try:
    class _SchedStore:
        pass
    _SchedStore.__name__ = "OffloadingConnectorScheduler"

    def _call_store(sched, mgr):
        self = sched           # noqa: F841  -- the frame local the walk reads
        return mgr.prepare_store([k1, k2], ctx)

    _ss = _SchedStore()
    _out = _call_store(_ss, _m)
    eq("*** prepare_store captures it even with the store path PAUSED ***",
       diskspec._LIVE_OFFLOAD_SCHED and diskspec._LIVE_OFFLOAD_SCHED[0] is _ss,
       True)
    check("...and the pause still returns an empty output, not None",
          _out is not None and not _out.keys_to_store)
finally:
    flags.clear_override("disk_store_paused")
    del diskspec._LIVE_OFFLOAD_SCHED[:]

# A frame held after the walk pins a request's locals and builds a cycle on the
# scheduler's hot path, so the function must not retain one.
#
# *** THE FIRST VERSION OF THIS ASSERTION WAS VACUOUS. *** It scanned gc for a
# frame whose code name contained "capture_offload_sched" -- but the walk ends
# holding the CALLER's frame, so the name to look for was "call_it", and the
# filter could never match. Break-verification proved it: the mutation
# (`_LEAKED_FRAMES.append(f)` instead of `del f`) applied cleanly and every
# assertion stayed green.
#
# What is actually checkable is RETENTION BY OUR MODULE: a leaked frame has to
# be reachable from diskspec's own globals to survive, so look there rather
# than guessing at a code name.
def _frames_held_by(mod):
    found = []
    for name, val in vars(mod).items():
        if type(val).__name__ == "frame":
            found.append(name)
        elif isinstance(val, (list, tuple, set)):
            for item in val:
                if type(item).__name__ == "frame":
                    found.append(name)
        elif isinstance(val, dict):
            for item in val.values():
                if type(item).__name__ == "frame":
                    found.append(name)
    return sorted(set(found))

eq("*** the capture retains no frame in the module's globals ***",
   _frames_held_by(diskspec), [])
del diskspec._LIVE_OFFLOAD_SCHED[:]

print("== (1w) the completion-ARRIVAL counter ==")
# Every guard on the completion path is satisfied by inspection and the
# completions still do not arrive, so the arrival point has to be counted. The
# three outcomes it separates are in the function's docstring; the test's job is
# that the wrap is idempotent, passes the call through unchanged, and cannot
# break the scheduler if the counting itself raises.
import sys as _sys  # noqa: E402
import types as _types  # noqa: E402

_seen = []


class _FakeSched:
    def __init__(self):
        self._stale_job_threshold = 5

    def update_connector_output(self, out):
        _seen.append(out)
        return "passed-through"


_fake_mod = _types.ModuleType(
    "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler")
_fake_mod.OffloadingConnectorScheduler = _FakeSched
_key = "vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler"
_saved_mod = _sys.modules.get(_key)
_sys.modules[_key] = _fake_mod
try:
    diskspec._ARRIVAL.update(calls=0, calls_with_meta=0, job_ids_seen=0,
                             stale_skipped=0)
    # *** THE ZEROS MUST NOT BE READABLE AS A MEASUREMENT. *** Gating the
    # installer behind the capture one-shot meant it never ran after a reload,
    # and `calls: 0` then read as "update_connector_output is never reached"
    # when the truth was "nobody was counting".
    diskspec._ARRIVAL["installed"] = False
    _sn_before = diskspec.offload_sched_snapshot()
    eq("the counter installs", diskspec.install_arrival_counter(), True)
    eq("...and installing twice is idempotent",
       diskspec.install_arrival_counter(), True)
    check("...leaving ONE wrap, not two",
          getattr(_FakeSched.update_connector_output, "_kvmgr_wrapped", False))

    class _Meta:
        completed_jobs = {3: 1, 7: 1, 9: 1}   # 3 and ... below the threshold 5

    class _Out:
        kv_connector_worker_meta = _Meta()

    _s = _FakeSched()
    eq("*** the call is passed through UNCHANGED ***",
       _s.update_connector_output(_Out()), "passed-through")
    eq("...and the original ran", len(_seen), 1)
    eq("calls counted", diskspec._ARRIVAL["calls"], 1)
    eq("calls_with_meta counted", diskspec._ARRIVAL["calls_with_meta"], 1)
    eq("job ids seen counted", diskspec._ARRIVAL["job_ids_seen"], 3)
    eq("*** stale-skipped counted separately -- that branch drops completions "
       "SILENTLY at debug level upstream ***",
       diskspec._ARRIVAL["stale_skipped"], 1)

    # a meta of None must count the call but not pretend to have seen jobs
    class _OutNoMeta:
        kv_connector_worker_meta = None

    _s.update_connector_output(_OutNoMeta())
    eq("a call with NO meta still counts as a call",
       diskspec._ARRIVAL["calls"], 2)
    eq("...and does not inflate calls_with_meta",
       diskspec._ARRIVAL["calls_with_meta"], 1)

    # and the counting must never break the scheduler
    class _Hostile:
        @property
        def kv_connector_worker_meta(self):
            raise RuntimeError("boom")

    # Catch it HERE rather than letting it escape: with the guard removed this
    # raises, and an escaping exception aborts the whole file with a traceback
    # instead of reporting a named failure. A test that fails for the wrong
    # reason is as unhelpful as one that passes for it.
    try:
        _hostile_result = _s.update_connector_output(_Hostile())
    except Exception as _exc:
        _hostile_result = "RAISED: %r" % (_exc,)
    eq("*** a raising output still passes through -- the instrument cannot "
       "break the path it measures ***",
       _hostile_result, "passed-through")

    # and the snapshot must refuse to show counts it has not been collecting
    diskspec._ARRIVAL["installed"] = False
    del diskspec._LIVE_OFFLOAD_SCHED[:]

    # The snapshot bails before `arrival` unless the handle looks like a real
    # scheduler, so the fake needs _jobs -- otherwise this section would test
    # the early return instead of the thing it is about.
    class _SchedWithJobs(_FakeSched):
        def __init__(self):
            _FakeSched.__init__(self)
            self._jobs = {}
            self._job_counter = 0
            self._block_id_to_pending_jobs = {}
            self._current_batch_jobs_to_flush = set()
            self.config = None

    diskspec._LIVE_OFFLOAD_SCHED.append(_SchedWithJobs())
    _a = diskspec.offload_sched_snapshot().get("arrival", {})
    eq("*** uninstalled reports installed=False ***", _a.get("installed"), False)
    check("...and withholds the counts entirely", "calls" not in _a)
    check("...and says why", bool(_a.get("why")))
    diskspec._ARRIVAL["installed"] = True
    _a = diskspec.offload_sched_snapshot().get("arrival", {})
    check("...while installed DOES report calls", "calls" in _a)
    del diskspec._LIVE_OFFLOAD_SCHED[:]
finally:
    if _saved_mod is not None:
        _sys.modules[_key] = _saved_mod
    else:
        _sys.modules.pop(_key, None)

print("== (1y) the scheduler-side job set: ABSENT IS NOT ZERO ==")
# The worker reports jobs_open: 0 while the scheduler can hold a hundred jobs,
# and the scheduler's _jobs is what `jobs_to_flush` names -- so this field is
# the fence, made visible. The dangerous failure is reporting jobs_open: 0 when
# the hook never fired, because that reads as "the fence is clear" and is the
# one wrong answer.
del diskspec._LIVE_OFFLOAD_SCHED[:]
_s = diskspec.offload_sched_snapshot()
eq("*** with no hook fired, available is False ***", _s.get("available"), False)
check("...and jobs_open is ABSENT, not 0", "jobs_open" not in _s)
check("...and it says why", bool(_s.get("why")))


class _FakeJob:
    def __init__(self, pc, store=True):
        self.pending_count = pc
        self.is_store = store


class _FakeSchedCfg:
    num_workers = 2


class _FakeSched:
    def __init__(self):
        # two jobs with ONE rank still outstanding, one with both -- the exact
        # shape that distinguishes "a rank's completion was lost" from
        # "neither arrived"
        self._jobs = {7: _FakeJob(1), 9: _FakeJob(1), 11: _FakeJob(2, False)}
        self._job_counter = 40
        self._stale_job_threshold = 0
        self._block_id_to_pending_jobs = {3: (7,), 4: (9,)}
        self._current_batch_jobs_to_flush = {7, 9}
        self.config = _FakeSchedCfg()


diskspec._LIVE_OFFLOAD_SCHED.append(_FakeSched())
_s = diskspec.offload_sched_snapshot()
eq("available with the hook fired", _s.get("available"), True)
eq("jobs_open counts them", _s.get("jobs_open"), 3)
eq("stores and loads are split", (_s.get("stores_open"), _s.get("loads_open")), (2, 1))
eq("the oldest job id is the LOWEST, which is the one wedging the fence",
   _s.get("oldest_job_id"), 7)
eq("*** the pending_count histogram is the diagnosis ***",
   _s.get("pending_count_hist"), {"1": 2, "2": 1})
eq("num_workers is carried, so the histogram can be read",
   _s.get("num_workers"), 2)
eq("blocks still fenced", _s.get("blocks_with_pending_jobs"), 2)
eq("and this batch's flush set", _s.get("batch_jobs_to_flush"), 2)


class _HostileSched:
    """Upstream renamed something, or an attribute raises on access."""

    @property
    def _jobs(self):
        raise RuntimeError("upstream changed shape")


del diskspec._LIVE_OFFLOAD_SCHED[:]
diskspec._LIVE_OFFLOAD_SCHED.append(_HostileSched())
_s = diskspec.offload_sched_snapshot()
eq("*** an upstream change is an ABSENCE, never an exception ***",
   _s.get("available"), False)
check("...and never a zero that reads as a clear fence",
      "jobs_open" not in _s)
check("...and it names what went wrong", "raised" in (_s.get("why") or ""))


class _NoJobs:
    pass


del diskspec._LIVE_OFFLOAD_SCHED[:]
diskspec._LIVE_OFFLOAD_SCHED.append(_NoJobs())
eq("a renamed _jobs is absent, not empty",
   diskspec.offload_sched_snapshot().get("available"), False)
del diskspec._LIVE_OFFLOAD_SCHED[:]

print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
