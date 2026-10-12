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
import json
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

# *** THE LOAD HAS ITS OWN ROW COUNT NOW, and this assertion caught the change
# *** rather than being quietly updated to match it.
# It used to read `== 32` and recorded only the LAST handler built, so it could
# not tell the two apart at all. Sharing one pinned buffer is what faulted the
# worker with Xid 13 five times (the load reads it under
# CU_MEMCPY_SRC_ACCESS_ORDER_ANY, whose precondition is that no concurrent GPU
# stream writes the source, and the store's copy is exactly such a write), so
# the two sizes being DIFFERENT is now the property worth pinning.
eq("the LOAD handler is built with its own staging size, a quarter of the "
   "store's", _built.get("staging_blocks"), 8)
eq("...and the spec derived it from the store's, not a constant",
   sp.staging_load_blocks, max(1, sp.staging_blocks // 4))
check("*** the two counts DIFFER, which is the whole point of the split ***",
      sp.staging_load_blocks != sp.staging_blocks)
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

    # The same word boundary the lazy tier needs: an accessor named after the
    # attribute (`self._verified_set()`) CONTAINS `self._verified`, so a
    # substring test reports the correct spelling as a violation. This lint
    # escaped it only because `_gcounts` has no such accessor.
    import re as _reO
    _patO = _reO.compile(r"self\." + _attr + r"(?![A-Za-z0-9_])")
    _direct = [ln.strip() for ln in _dsrc.splitlines()
               if _patO.search(ln) and "getattr" not in ln
               and not ln.strip().startswith("#")
               and not _is_decl(ln, _attr)]
    eq(f"every read of self.{_attr} outside __init__ uses getattr", _direct, [])

# *** AND A SECOND TIER THE AUDIT ABOVE COULD NOT SEE: PURELY LAZY STATE. ***
# `_OPTIONAL_ATTRS` asserts the attribute EXISTS on a fresh instance, so it can
# only cover things __init__ creates. State created on first use and never in
# __init__ -- which is the strictly safer pattern, and what `_loading_map` has
# always done -- was therefore outside the audit entirely, including its getattr
# LINT. That gap is why this tier exists: `_loading` is the eviction hold, and a
# direct `self._loading` read on a rebound manager is the same fatal
# AttributeError in the same function as `_gcounts` was.
#
# The contract is the inverse of the one above: ABSENT on a fresh instance (no
# __init__ assignment to drift), present after use, and every read through the
# one accessor.
# `_loading` is in here, and this audit is how that was settled: it asked
# whether __init__ creates it and the answer was YES -- so it is a HYBRID,
# assigned in __init__ and still read through `_loading_map()`. The braces are
# what matter, because a rebound instance is the case that crashes, and that is
# exactly what this tier checks. Sorting attributes by whether __init__
# happens to touch them would have been a category judgment nobody can keep
# current; "survive the attribute being gone" is the property either way.
_LAZY_ATTRS = ("_loading", "_verified", "_wanted", "_vst", "_fst")
_ACCESSOR = {"_loading": "_loading_map", "_verified": "_verified_set",
             "_wanted": "_wanted_map", "_vst": "_verify_state",
             "_fst": "_free_state"}

for _attr in _LAZY_ATTRS:
    m_lz = diskspec.DiskPresenceManager(sp.file_mapper(), num_workers=WORLD)
    # Whether __init__ created it or not, a REBIND leaves it gone. That is the
    # state to test, so make it unconditionally.
    if hasattr(m_lz, _attr):
        delattr(m_lz, _attr)
    _ok, _why = True, ""
    try:
        m_lz.prepare_store([k1, k2], ctx)
        m_lz.complete_store([k1, k2], ctx)
        m_lz.lookup(k1, ctx)
        _snap2 = m_lz.snapshot()
    except Exception as _exc:
        _ok, _why = False, repr(_exc)
    check(f"*** a manager that never had {_attr} still serves the data path "
          f"*** {_why}", _ok)
    check(f"...and snapshot() still answers", _ok and isinstance(_snap2, dict))
    # And the accessor must SUPPLY it, not merely tolerate its absence: a
    # getattr-with-default that never stores anything back would pass the check
    # above while silently losing every write.
    getattr(m_lz, _ACCESSOR[_attr])()
    check(f"...and {_ACCESSOR[_attr]}() creates it on the INSTANCE",
          hasattr(m_lz, _attr))

# *** THE MATCHER NEEDS A WORD BOUNDARY, AND ITS FIRST VERSION FAILED FOR THE
# *** WRONG REASON BECAUSE IT DID NOT HAVE ONE. *** `self._verified_set()` -- the
# accessor, which is the CORRECT spelling -- contains `self._verified`, so a
# substring test reported every proper call as a violation. The `_gcounts` lint
# above never hit this only because its accessor is not named after it. A lint
# that fails for the wrong reason is as unhelpful as one that passes for it.
import re as _reL
for _attr in _LAZY_ATTRS:
    _pat = _reL.compile(r"self\." + _attr + r"(?![A-Za-z0-9_])")
    _direct = [ln.strip() for ln in _dsrc.splitlines()
               if _pat.search(ln) and "getattr" not in ln
               and not ln.strip().startswith("#")
               and not _is_decl(ln, _attr)]
    eq(f"every read of self.{_attr} goes through its accessor", _direct, [])

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

print("== (5a) PHASE 5: the victim rule, tail-first within an age ==")
# *** PLAIN LRU IS THE WRONG ANSWER AND THAT IS THE WHOLE POINT. *** External
# reuse is a CONTIGUOUS RUN FROM TOKEN 0, so evicting a prefix's TAIL leaves a
# shorter prefix that still hits, while evicting its HEAD writes off every block
# behind it -- those bytes stay on disk and can never be read again. Measured on
# the GPU side already: head-lost left 228 blocks reusable as 0, tail-first kept
# 495 of 495.
cv = diskspec.choose_victims

# (key, nbytes, pos, recency) -- smaller recency is older
_prefix_old = [(b"o%d" % i, 100, i, 1) for i in range(5)]   # one cold prefix
_prefix_hot = [(b"h%d" % i, 100, i, 9) for i in range(5)]   # one hot prefix

eq("nothing to free means no victims", cv(_prefix_old + _prefix_hot, 0), [])
eq("...and a negative need is not an invitation to evict",
   cv(_prefix_old + _prefix_hot, -1), [])

_v = cv(_prefix_old + _prefix_hot, 250)
eq("*** the COLD prefix is evicted, not the hot one ***",
   all(k.startswith(b"o") for k in _v), True)
eq("*** and TAIL-FIRST within it: highest block index goes first ***",
   _v, [b"o4", b"o3", b"o2"])
check("...which is what stops a HEAD being orphaned: o0 survives",
      b"o0" not in _v)

# recency must beat position: a hot TAIL must outlive a cold HEAD
_v2 = cv([(b"cold-head", 100, 0, 1), (b"hot-tail", 100, 99, 9)], 100)
eq("*** recency is PRIMARY -- a cold head goes before a hot tail ***",
   _v2, [b"cold-head"])

eq("it frees AT LEAST what was asked, and stops there",
   len(cv(_prefix_old, 150)), 2)
eq("...and will take everything if that is what the budget needs",
   len(cv(_prefix_old, 10_000)), 5)
eq("an empty index yields no victims", cv([], 1000), [])

# differing sizes must be respected, not assumed uniform
_mixed = [(b"big", 900, 1, 1), (b"small", 100, 0, 1)]
eq("the TAIL is taken first even when it is the big one, and one suffices",
   cv(_mixed, 500), [b"big"])

print("== (5b) the budget conversion, where 0 must mean UNBOUNDED ==")
bb = diskspec.budget_bytes
eq("GB to bytes", bb(1.5), 1_500_000_000)
eq("*** 0 means NO BUDGET, not 'evict everything' ***", bb(0), 0)
eq("...and so does a negative, which is the same misread", bb(-5), 0)
eq("...and so does garbage, rather than raising into prepare_store", bb("x"), 0)
check("*** that direction matters: an unset knob must not wipe a 518 GB index "
      "***", bb(0) == 0 and cv([(b"k", 1, 0, 1)], bb(0)) == [])

print("== (5c) the manifest must read BOTH formats -- 520,446 keys depend on it ==")
# Phase 5 added a block POSITION per key, so entries became [hex, pos]. A reader
# that only understood the new shape would discard the ENTIRE existing index --
# 518 GB of still-restorable files -- and the loss would look like a cache that
# merely went cold. So the old bare-string form must still load.
_mdir = tempfile.mkdtemp(prefix="kvmanifest-")
_troot = tempfile.mkdtemp(prefix="kvroot-")


# root_dir lives on _Cfg's kv_transfer_config, not on _KVCfg -- so the WRITABLE
# root goes there. The default is /kvoffload, the real mount on the box and
# read-only here, and testing the manifest means actually creating files.
class _CfgTmp(_Cfg):
    def __init__(self, world=WORLD, rank=0):
        _Cfg.__init__(self, world, rank)
        self.kv_transfer_config = type("K", (), {
            "kv_connector_extra_config": {"root_dir": _troot}})()


flags.reset_for_tests()
_tmpspec = diskspec.DSparkDiskOffloadingSpec(_CfgTmp(WORLD, 0), _KVCfg())
_k1 = base.make_offload_key(b"\xa1" * 32, 0)
_k2 = base.make_offload_key(b"\xa2" * 32, 1)

def _mkmanifest(entries):
    path = os.path.join(_mdir, "presence-%d.json" % len(os.listdir(_mdir)))
    with open(path, "w") as f:
        json.dump({"keys": entries}, f)
    # The files must exist or the loader's pruner drops the keys -- which is
    # itself the behaviour that makes an unreadable manifest safe. Paths come
    # from the real FileMapper, so this exercises the actual layout rather than
    # a guess at it.
    # A spec rooted in a WRITABLE temp dir: the default root is /kvoffload,
    # which is the real mount on the box and read-only here. Testing the
    # manifest means actually creating files, so the fixture has to own a root.
    _fm = _tmpspec.file_mapper()
    for e in entries:
        h = e if isinstance(e, str) else e[0]
        fp = _fm.get_file_name(bytes.fromhex(h))
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        open(fp, "wb").close()
    return path

_old_fmt = _mkmanifest([_k1.hex(), _k2.hex()])
_m_old = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD,
                                      manifest_path=_old_fmt)
eq("*** the OLD bare-string manifest still loads every key ***",
   len(_m_old._present), 2)
eq("...and an unknown position is -1, not 0",
   sorted(p for _nb, p in _m_old._present.values()), [-1, -1])
check("*** -1 sorts BEFORE 0 under the -pos tie-break, so unknown-position "
      "keys are the LAST victims of their age -- the conservative direction, "
      "because one of them might be a head ***",
      diskspec.choose_victims([(b"unknown", 10, -1, 1), (b"known-tail", 10, 5, 1)],
                              10) == [b"known-tail"])

_new_fmt = _mkmanifest([[_k1.hex(), 0], [_k2.hex(), 7]])
_m_new = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD,
                                      manifest_path=_new_fmt)
eq("the NEW [hex, pos] manifest loads too", len(_m_new._present), 2)
eq("...with the positions preserved",
   sorted(p for _nb, p in _m_new._present.values()), [0, 7])

print("== (5d) touch() is the LRU, and it was a no-op ==")
_m = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD)
_m._remember(b"a", 0); _m._remember(b"b", 1); _m._remember(b"c", 2)
eq("insertion order is the initial recency", list(_m._present), [b"a", b"b", b"c"])
_m.touch([b"a"], ctx)
eq("*** touch moves a key to MOST recent, so a hot prefix outlives a cold one "
   "***", list(_m._present), [b"b", b"c", b"a"])
_m.touch([b"zzz-not-present"], ctx)
eq("...and touching an absent key is a no-op, not an insertion",
   list(_m._present), [b"b", b"c", b"a"])

print("== (5e) the budget: 0 is unbounded, and the delete reaches BOTH ranks ==")
flags.set_override("disk_budget_gb", 0)
try:
    _out = _m_new.prepare_store([_k1], ctx)
    check("*** with no budget set, nothing is ever evicted ***",
          _out is None or not getattr(_out.store_spec, "to_delete", []))
finally:
    flags.clear_override("disk_budget_gb")

check("*** the delete rides the spec, which serialises to every rank ***",
      hasattr(diskspec.DiskLoadStoreSpec([], to_delete=[b"x"]), "to_delete"))
eq("...carrying plain bytes, because a numpy scalar in a serialised structure "
   "killed the output thread tonight",
   [type(x) for x in diskspec.DiskLoadStoreSpec([], to_delete=[b"x"]).to_delete],
   [bytes])
eq("...and an empty delete list is empty, not None",
   diskspec.DiskLoadStoreSpec([]).to_delete, [])

print("== (5f) A LEGACY-SHAPED MANAGER: every public method must survive ==")
# *** THE REBIND TRAP, SIXTH TIME, AND THIS ONE CHANGED A TYPE. *** A reload
# never runs __init__, so a live manager keeps whatever its OLD constructor
# built. The first five instances were MISSING attributes; Phase 5 turned
# `_present` from a set into an OrderedDict, so a rebound manager had a set
# (no move_to_end) and no `_bytes` at all:
#
#   AttributeError: 'DiskPresenceManager' object has no attribute '_bytes'
#     -- fatal to the scheduler, measured on the box
#
# Five getattrs would have been five more chances to miss one. This is the
# GENERAL guard I should have written after the first instance: build the worst
# legacy shape and drive EVERY public entry point through it.
_legacy = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD)
_lk = base.make_offload_key(b"\xc3" * 32, 0)
# the worst case: a bare set, and none of the Phase 5 attributes
_legacy._present = {bytes(_lk)}
for _attr in ("_bytes", "_pos_hint", "_evicted_pending", "_gcounts"):
    if hasattr(_legacy, _attr):
        delattr(_legacy, _attr)

_failures = []
for _name, _call in (
    ("lookup", lambda: _legacy.lookup(_lk, ctx)),
    ("touch", lambda: _legacy.touch([_lk], ctx)),
    ("prepare_store", lambda: _legacy.prepare_store([_lk], ctx)),
    ("complete_store", lambda: _legacy.complete_store([_lk], ctx)),
    ("snapshot", lambda: _legacy.snapshot()),
    ("take_events", lambda: list(_legacy.take_events())),
):
    try:
        _call()
    except Exception as _exc:
        _failures.append("%s: %r" % (_name, _exc))
eq("*** every public method survives a legacy-shaped (rebound) manager ***",
   _failures, [])
_migrated = isinstance(_legacy._present, type(diskspec.OrderedDict()))
check("...and the set was migrated to the ordered index, not discarded",
      _migrated)
check("...keeping the key it already held", bytes(_lk) in _legacy._present)
# Guarded: without the migration `_present` is still a set, and subscripting it
# raises TypeError which would ABORT the file rather than report. The mutation
# is already caught by the assertion above; this must not turn a caught
# mutation into a crash.
eq("...with position UNKNOWN (-1), not a fabricated 0",
   (_legacy._present[bytes(_lk)][1] if _migrated else "NOT MIGRATED"), -1)
check("...and a byte total exists afterwards", hasattr(_legacy, "_bytes"))

# migration must be idempotent -- it runs at the top of every entry point
_before = dict(_legacy._present) if _migrated else None
try:
    _legacy.lookup(_lk, ctx); _legacy.snapshot(); _legacy.touch([_lk], ctx)
    _after = dict(_legacy._present) if _migrated else None
except Exception as _exc:
    _after = "RAISED: %r" % (_exc,)
eq("*** the migration is idempotent, so calling it per-entry costs nothing ***",
   _after, _before)

print("== (5f2) a migration must repair bad VALUES, not just a bad shape ==")
# *** THE FIRST ACCOUNTING FIX DID NOTHING ON THE BOX. *** It was deployed hot
# onto an index that was ALREADY an OrderedDict -- built by the previous, broken
# version -- so the type migration found nothing to convert and every entry kept
# the cost 0 it had been given by code that could not compute one. Measured:
# 523,194 keys, held 0.0 GB, byte_accounting_ok False, AFTER the fix.
_vm = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD)
_vk = base.make_offload_key(b"\xe6" * 32, 1)
_vpath = _tmpspec.file_mapper().get_file_name(_vk)
os.makedirs(os.path.dirname(_vpath), exist_ok=True)
with open(_vpath, "wb") as _f:
    _f.write(b"\0" * 2048)
# the exact broken state: right TYPE, zero COSTS
_vm._present[bytes(_vk)] = (0, 4)
_vm._bytes = 0
_vm.snapshot()                      # any entry point triggers the repair
eq("*** zero-cost entries are RE-COSTED from the files ***",
   _vm._present[bytes(_vk)][0], 2048)
eq("...and the total is rebuilt", _vm._bytes, 2048)
eq("...while the POSITION is preserved, not reset",
   _vm._present[bytes(_vk)][1], 4)
eq("...and the accounting now reads sound",
   _vm.snapshot()["byte_accounting_ok"], True)
os.unlink(_vpath)

print("== (5g) byte accounting comes from a STAT, not from worker state ==")
# *** THE FIRST VERSION ASKED THE LIVE HANDLER, WHICH LIVES IN THE WORKER. ***
# This manager runs in the ENGINE CORE, so `_LIVE_HANDLERS` is always empty
# here and every key costed 0. Measured on the box: 521,628 keys reporting
# "0.0 GB held" with a 1 GB budget and over_budget=False -- the accounting was
# a fiction and the budget could never fire. One stat per GROUP, cached.
_bm = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD)
_bk = base.make_offload_key(b"\xd4" * 32, 2)
_bpath = _tmpspec.file_mapper().get_file_name(_bk)
os.makedirs(os.path.dirname(_bpath), exist_ok=True)
with open(_bpath, "wb") as _f:
    _f.write(b"\0" * 4096)
eq("*** the cost is the file's real size ***", _bm._key_bytes(_bk), 4096)
eq("...cached per group, so it is one stat ever",
   sorted(_bm._group_bytes), [2])
os.unlink(_bpath)
eq("...and the cache survives the file going away (it is a group constant)",
   _bm._key_bytes(_bk), 4096)

_bk_absent = base.make_offload_key(b"\xd5" * 32, 3)
eq("a group with no file yet costs 0 rather than raising",
   _bm._key_bytes(_bk_absent), 0)

print("== (5h) 0 bytes held with keys present is an ABSENCE, not a measurement ==")
_am = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD)
eq("an empty index has sound accounting", _am.snapshot()["byte_accounting_ok"],
   True)
# the exact broken state seen live: keys present, nothing sized
_am._present[b"ghost"] = (0, -1)
_am._bytes = 0
check("*** keys present at 0 bytes is flagged, because a budget computed from "
      "it silently never fires ***",
      _am.snapshot()["byte_accounting_ok"] is False)
_am._group_bytes = {0: 1000}
_am._present[b"real"] = (1000, 0)
_am._bytes = 1000
check("...and sound accounting reports True",
      _am.snapshot()["byte_accounting_ok"] is True)

print("== (5i) the eviction log must be THROTTLED at steady state ==")
# *** MEASURED ON THE BOX: ONE LINE PER STORE ROUND, ABOUT ONE A SECOND. ***
# Once the index sits at budget, every store evicts ~90 keys to make room for
# the same bytes, so the unthrottled line never stops -- a ROUTINE event logged
# as though it were notable, which trains a reader to skip the very channel the
# interesting first eviction (100,665 keys / 71.8 GB) arrives on.
check("there is a throttle interval, and it is not sub-second",
      diskspec.DiskPresenceManager.EVICT_LOG_S >= 10)
_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                         "diskspec.py")).read()
check("*** the first eviction still logs in full -- dropping 71.8 GB IS an "
      "event ***", "first = not getattr(self, \"_evict_logged\"" in _src)
check("...and later lines carry the TOTALS, which is what a steady-state "
      "reader wants", "evicted in total" in _src)

_dsrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskio.py")).read()
check("the worker's unlink line is throttled too", "_del_log_at" in _dsrc)
check("*** but an unlink FAILURE is never throttled -- that is the thing "
      "somebody needs the moment it happens ***",
      "FAILED to unlink" in _dsrc and _dsrc.index("FAILED to unlink")
      < _dsrc.index("_del_logged = True"))

print("== (5j) A KEY WITH A LOAD IN FLIGHT MUST NEVER BE A VICTIM ==")
# *** THE FRAMEWORK CANNOT SURVIVE A FAILED TRANSFER. ***
# `offloading/worker.py:273` is a bare `assert transfer_result.success`, under
# the comment "we currently do not support job failures" -- confirmed in the
# INSTALLED tree on the box, not merely in patches/pristine/. So a load that
# cannot find its file does not degrade to a miss, it raises AssertionError in
# the worker's output path and takes the engine with it.
#
# Phase 5's eviction made that reachable for the first time: lookup() says
# present, prepare_load issues the load, a concurrent store's budget picks the
# same key as a victim, _apply_deletes unlinks it on BOTH ranks, and the read
# hits FileNotFoundError. The LRU touch makes it unlikely -- a just-looked-up
# key is the most recently used and therefore the last victim -- but UNLIKELY
# IS NOT A SAFETY PROPERTY when the failure is a dead engine.
F.write(kvmgr_enabled=1, disk_restore=1)   # prepare_load refuses otherwise
_lm = diskspec.DiskPresenceManager(_tmpspec.file_mapper(), num_workers=WORLD)
_lk = base.make_offload_key(b"\xe1" * 32, 0)
_lk2 = base.make_offload_key(b"\xe2" * 32, 0)

# *** AN ASSERTION WHOSE SUBJECT CAN RAISE MUST CATCH. *** Fifth time this
# suite has needed the correction: under mutation (dropping the lazy init from
# _loading_map) these calls raise AttributeError, the exception escapes, and
# the FILE ABORTS before printing its summary -- so break-verification reads a
# crash as "no output" rather than as a named failure. Every driver below goes
# through these.
def _lm_load(keys):
    try:
        _lm.prepare_load(keys, None)
        return "ok"
    except Exception as exc:
        return "RAISED: %r" % (exc,)


def _lm_done(keys):
    try:
        _lm.complete_load(keys, None)
        return "ok"
    except Exception as exc:
        return "RAISED: %r" % (exc,)


def _lm_holds():
    try:
        return dict(_lm._loading_map())
    except Exception as exc:
        return "RAISED: %r" % (exc,)


eq("no holds to begin with", _lm_holds(), {})
_lm_load([_lk])
check("*** prepare_load takes an eviction hold ***",
      bytes(_lk) in (_lm_holds() or {}), _lm_holds())
_lm_done([_lk])
eq("...and complete_load releases it", _lm_holds(), {})

# Counted, not a set: two requests can be loading the same key, and the first
# completion must not release the second's hold.
_lm_load([_lk])
_lm_load([_lk])
_lm_done([_lk])
check("*** the hold is COUNTED -- one of two concurrent loads completing does "
      "NOT release it ***", bytes(_lk) in (_lm_holds() or {}),
      _lm_holds())
_lm_done([_lk])
eq("...and the second completion does", _lm_holds(), {})

# The hold must actually reach victim selection. Index holds two keys; the
# budget needs room for one; the held one must survive even though it is the
# OLDEST and so the natural victim.
# *** THE FIRST VERSION OF THIS ASSERTION PASSED VACUOUSLY. *** With no
# budget set, _enforce_budget returns before choosing anything, so "the held
# key is not among the victims" was true of an EMPTY victim list. Sized in real
# bytes against a 1 GB budget so eviction genuinely fires -- and the companion
# assertion below ("the unheld one IS chosen") is what caught it, which is the
# reason a guard needs a partner that fails when nothing happens at all.
F.write(kvmgr_enabled=1, disk_restore=1, disk_budget_gb=1)
_lm._group_bytes = {0: 600_000_000}
_lm._present[bytes(_lk)] = (600_000_000, 0)   # oldest -> first victim by LRU
_lm._present[bytes(_lk2)] = (600_000_000, 1)
_lm._bytes = 1_200_000_000
_lm_load([_lk])
_incoming = [base.make_offload_key(b"\xe3" * 32, 0)]
try:
    _victims = [bytes(v) for v in (_lm._enforce_budget(_incoming) or [])]
except Exception as _exc:
    _victims = ["RAISED: %r" % (_exc,)]
check("the budget actually fired, so the next two are not vacuous",
      len(_victims) > 0, _victims)
check("*** the held key is NOT chosen, though it is the oldest ***",
      bytes(_lk) not in _victims, _victims)
check("...and the unheld one is, so the budget still works",
      bytes(_lk2) in _victims, _victims)

# The rebind trap, for the SEVENTH time. A reload re-points a live manager with
# `obj.__class__ = new`, so it gets every new METHOD and no new attribute.
_orphan = diskspec.DiskPresenceManager.__new__(diskspec.DiskPresenceManager)
try:
    _got = dict(diskspec.DiskPresenceManager._loading_map(_orphan))
except Exception as exc:
    _got = "RAISED: %r" % (exc,)
eq("*** _loading_map works on an instance that never ran __init__ -- a reload "
   "gives new methods and no new attributes ***", _got, {})
_a = diskspec.DiskPresenceManager.__new__(diskspec.DiskPresenceManager)
_b = diskspec.DiskPresenceManager.__new__(diskspec.DiskPresenceManager)
def _orphan_map(o):
    try:
        return diskspec.DiskPresenceManager._loading_map(o)
    except Exception as exc:
        return {"RAISED": repr(exc)}


_orphan_map(_a)["x"] = 1
eq("...and the map is per INSTANCE, not shared on the class",
   dict(_orphan_map(_b)), {})

print("== (P4) THE ON-DISK LAYOUT CANNOT CHANGE WITHOUT THE VERSION MOVING ==")
# *** THE DANGEROUS FORMAT CHANGE IS THE ACCIDENTAL ONE. *** Anybody bumping
# DISK_FORMAT_VERSION has already thought about it. The failure to design
# against is somebody editing `file_plan`, `write_chunks` or `plan_nbytes` and
# NOT bumping it -- after which every pre-change file restores silently as
# wrong KV, which is the F8_E8M0 class: plausible tensors, nothing crashing.
#
# *** AND THE GUARD HAS TO BE OFFLINE, NOT A RUNTIME HASH. *** Hashing the
# writer's source at boot would catch it -- and would also orphan 400 GB the
# first time anybody fixed a typo in a comment. So: comments and docstrings are
# stripped, whitespace runs collapsed, and the hash is pinned HERE. A real
# layout change fails this test on a laptop; a cosmetic edit fails it too, at
# which point you look, see it is cosmetic, and re-pin without bumping.
import re as _reP4
import tokenize as _tokP4
import io as _ioP4
import hashlib as _hlP4

# *** BLANK THE REGIONS IN PLACE, BY POSITION. *** This repo has already paid
# for the alternative: a lint that stripped with `tokenize` and then JOINED the
# surviving tokens turned `.data[:] =` into `. data [ : ] =`, so its pattern
# could never match and it passed however broken the module was. Found only by
# break-verification.
def _normalise_p4(src):
    out = list(src)
    try:
        toks = list(_tokP4.generate_tokens(_ioP4.StringIO(src).readline))
    except Exception:
        return src
    starts = [0]
    for _l in src.splitlines(keepends=True):
        starts.append(starts[-1] + len(_l))
    prev = None
    for t in toks:
        kill = t.type == _tokP4.COMMENT or (
            t.type == _tokP4.STRING and prev in
            (_tokP4.INDENT, _tokP4.NEWLINE, _tokP4.NL, None))
        if kill:
            a = starts[t.start[0] - 1] + t.start[1]
            b = starts[t.end[0] - 1] + t.end[1]
            for k in range(a, min(b, len(out))):
                if out[k] != "\n":
                    out[k] = " "
        if t.type not in (_tokP4.NL, _tokP4.NEWLINE):
            prev = t.type
    return _reP4.sub(r"[ \t]+", " ", "".join(out))


_diskio_src = open(os.path.join(REPO, "patches", "kvmgr_plugin",
                                   "kvmgr", "diskio.py")).read()


def _layout_hash():
    h = _hlP4.sha256()
    missing = []
    for n in diskspec.DISK_LAYOUT_FUNCTIONS:
        m = _reP4.search(r"^def %s\(.*?(?=\n(?:def |class |@)|\Z)" % n,
                         _diskio_src, _reP4.S | _reP4.M)
        if not m:
            missing.append(n)
            continue
        h.update(_normalise_p4(m.group(0)).encode())
    return h.hexdigest()[:16], missing


# *** PINNED. Bump DISK_FORMAT_VERSION and re-pin only when the LAYOUT changed.
LAYOUT_HASH_PINNED = "b961b152f2676405"
LAYOUT_VERSION_PINNED = 1

_lh, _miss = _layout_hash()
eq("*** every declared layout function was FOUND -- a guard that hashes "
   "nothing passes forever ***", _miss, [])
check("...and there are several of them", len(diskspec.DISK_LAYOUT_FUNCTIONS) >= 3,
      diskspec.DISK_LAYOUT_FUNCTIONS)
eq("*** the on-disk layout is unchanged; if this fails, either BUMP "
   "DISK_FORMAT_VERSION (a real layout change) or re-pin (a cosmetic edit) ***",
   _lh, LAYOUT_HASH_PINNED)
eq("...and the pinned hash belongs to the current format version",
   diskspec.DISK_FORMAT_VERSION, LAYOUT_VERSION_PINNED)

# The normaliser must actually normalise, or the pin tracks prose.
_probe = "def f(x):\n    \"\"\"doc.\"\"\"\n    # note\n    return x[:] = 1\n"
_probe2 = "def f(x):\n    \"\"\"TOTALLY different doc.\"\"\"\n    # other\n    return x[:] = 1\n"
eq("*** comments and docstrings do NOT move the hash -- otherwise every "
   "documentation edit would read as a format change ***",
   _normalise_p4(_probe), _normalise_p4(_probe2))
check("*** but the CODE does -- the subscript survives normalisation, which is "
   "the join-the-tokens bug this repo already paid for ***",
      "[:]" in _normalise_p4(_probe), _normalise_p4(_probe))
check("...and a real code change moves it",
      _normalise_p4(_probe) != _normalise_p4(_probe.replace("x[:]", "x[1:]")))

print("== (P4) the format version is INSIDE the fingerprint, not beside it ==")
eq("the root directory carries the version", diskspec.DISK_FORMAT_ROOT,
   "dspark-kv-v%d" % diskspec.DISK_FORMAT_VERSION)
_fp_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                               "diskspec.py")).read()
check("*** and so does the HASH -- a root name alone can be sidestepped by "
      "anything that computes a base path itself ***",
      'inference_engine="vllm+dspark-kv/v%d" % DISK_FORMAT_VERSION' in _fp_src)

print("== (P4) the model identity is CONTENT, not the HF commit ==")
# *** THE HF COMMIT IS THE WRONG THING TO FINGERPRINT, AND THIS BOX PROVES IT.
# *** Measured 2026-10-09: `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` has TWO
# snapshots in the hub cache, and they share ALL 49 safetensors blob targets
# with a byte-identical config.json -- they differ only in LICENSE, README.md,
# encoding/ and inference/. So the KV written under one is byte-valid under the
# other, and a commit-keyed fingerprint would have orphaned 400 GB of warm
# cache to protect against a docs-only revision bump.
#
# This fixture IS that measurement, so the reasoning cannot quietly be undone.
import tempfile as _tfP
import os as _osP

_hub = tempfile.mkdtemp(prefix="kvhub-")
_repo = _osP.path.join(_hub, "models--org--mdl")
_blobs = _osP.path.join(_repo, "blobs")
_osP.makedirs(_blobs)


def _snap(commit, extra_files, blob_map):
    d = _osP.path.join(_repo, "snapshots", commit)
    _osP.makedirs(d, exist_ok=True)
    with open(_osP.path.join(d, "config.json"), "w") as f:
        f.write('{"hidden_size": 4096}')
    for name, blob in blob_map.items():
        bp = _osP.path.join(_blobs, blob)
        if not _osP.path.exists(bp):
            open(bp, "wb").write(b"x")
        _osP.symlink(_osP.path.join("..", "..", "blobs", blob),
                     _osP.path.join(d, name))
    for extra in extra_files:
        open(_osP.path.join(d, extra), "w").write("doc")
    return d


_BLOBS = {"model-00001.safetensors": "aaa", "model-00002.safetensors": "bbb"}
_A = _snap("c0mmit_aaaaaa", ["README.md"], _BLOBS)
_B = _snap("c0mmit_bbbbbb", ["README.md", "LICENSE", "inference"], _BLOBS)

_ida = diskspec.model_content_id(_A)
_idb = diskspec.model_content_id(_B)
check("the identity is a real 12-hex id, not a sentinel",
      _ida != "unknown" and len(_ida) == 12, _ida)
eq("*** two revisions that share weights AND config get the SAME identity -- "
   "a docs-only bump must not orphan the tier ***", _ida, _idb)

# And it must change when the thing that actually matters changes.
_C = _snap("c0mmit_cccccc", [], {"model-00001.safetensors": "aaa",
                                 "model-00002.safetensors": "DIFFERENT"})
check("*** but a CHANGED WEIGHT BLOB changes it -- that is the whole point ***",
      diskspec.model_content_id(_C) != _ida,
      (diskspec.model_content_id(_C), _ida))

_D = _snap("c0mmit_dddddd", [], _BLOBS)
with open(_osP.path.join(_D, "config.json"), "w") as _f:
    _f.write('{"hidden_size": 8192}')
check("...and so does a changed config.json",
      diskspec.model_content_id(_D) != _ida)

print("== (P4) model_dir resolves the hub the way the hub does ==")
_osP.makedirs(_osP.path.join(_repo, "refs"), exist_ok=True)
with open(_osP.path.join(_repo, "refs", "main"), "w") as _f:
    _f.write("c0mmit_aaaaaa\n")
eq("a bare model id resolves through refs/main to a snapshot",
   diskspec.model_dir("org/mdl", None, _hub), _A)
eq("...and an explicitly pinned commit wins over refs/main",
   diskspec.model_dir("org/mdl", "c0mmit_bbbbbb", _hub), _B)
eq("a plain directory is used as-is", diskspec.model_dir(_A, None, _hub), _A)

print("== (P4) an unidentifiable model DEGRADES, it does not pretend ==")
# *** `unknown` IS NOT A REFUSAL, DELIBERATELY. *** If the layout cannot be
# read the identity falls back to the model NAME alone -- exactly today's
# behaviour, so the change is never a regression. The honest claim is "no
# better protected than before", which is why it is logged and exposed in /kv
# rather than hidden behind a reassuring hash.
eq("an absent directory is unknown", diskspec.model_content_id(None), "unknown")
eq("a missing path is unknown",
   diskspec.model_content_id("/nonexistent/kvmgr/model"), "unknown")
_empty = tempfile.mkdtemp(prefix="kvempty-")
eq("*** a directory with NO safetensors is unknown rather than a hash of "
   "nothing -- which two different empty dirs would have SHARED ***",
   diskspec.model_content_id(_empty), "unknown")
eq("an unresolvable model id is unknown",
   diskspec.model_dir("no/such-model", None, _hub), None)

print("== (P4) the hub lookup must not assume THIS box's layout ==")
# *** MY FIRST VERSION ENDED AT "/models/hub", WHICH IS THIS DEPLOYMENT'S BIND
# *** MOUNT. *** On any other machine the lookup would miss and the fingerprint
# would quietly degrade to `unknown` -- the protection silently absent rather
# than announced. huggingface_hub's own default is ~/.cache/huggingface/hub, so
# that has to be tried before the local convention.
_pt_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                            "diskspec.py")).read()
# *** AND THE FIRST VERSION OF THIS CHECK FAILED FOR THE WRONG REASON. ***
# It compared raw substring positions, and "/models/hub" appears in the COMMENT
# above the code -- earlier in the file than the line it describes -- so the
# lint read prose as code and failed against correct code. The digest already
# records this one: strip comments first, or anchor on something only code can
# produce. Here the honest anchor is the `candidates.append(...)` CALLS, which
# are the order being asserted.
_appends = [m.group(1) for m in _reP4.finditer(
    r"candidates\.append\(\s*(.*?)\)\n", _pt_src, _reP4.S)]
check("the append calls were located at all -- a regex that matches nothing "
      "cannot fail", len(_appends) >= 2, _appends)
_std_at = next((i for i, a in enumerate(_appends) if ".cache" in a), None)
_box_at = next((i for i, a in enumerate(_appends) if "/models/hub" in a), None)
check("*** the standard HF cache is appended BEFORE this box's convention, so "
      "the local bind mount is only ever a last resort ***",
      _std_at is not None and _box_at is not None and _std_at < _box_at,
      (_std_at, _box_at, _appends))
check("HF_HUB_CACHE is honoured", 'environ.get("HF_HUB_CACHE")' in _pt_src)
check("...and HF_HOME", '"HF_HOME"' in _pt_src)
# And the search must keep looking rather than stopping at the first ROOT that
# does not hold this repo -- a single-root check would miss a model present in
# a later candidate.
_other = tempfile.mkdtemp(prefix="kvhub2-")
eq("*** a candidate root without the repo is skipped, not fatal -- the search "
   "continues to the next one ***",
   diskspec.model_dir("org/mdl", None, _other), None)

print("== (P4) the migration keeps the warm tier, and REFUSES past v1 ==")
# *** STRENGTHENING A FINGERPRINT ORPHANS EVERYTHING UNDER THE OLD ONE, and on
# *** this box that is 400 GB and 560,000 keys. *** The bytes are not wrong --
# the LAYOUT did not change, only the identity string -- so discarding them
# would be a pure loss paid in 430-795 s cold prefills. A directory rename is
# O(1) and keeps the manifest too, because presence.json sits at the format
# root and is keyed by block hash rather than by path.
_mg = diskspec.DSparkDiskOffloadingSpec(_CfgTmp(WORLD, 0), _KVCfg())
_newbase = "%s_r%d" % (_mg.file_mapper().base_path, _mg.rank)

# Build the pre-Phase-4 path the same way the migration does, and plant a file.
_legacy_dir = None
for _cand in os.listdir(_mg.root_dir) if os.path.isdir(_mg.root_dir) else []:
    pass
import shutil as _shP
_old_fake = os.path.join(_mg.root_dir, "legacy-fingerprint_r0")
os.makedirs(_old_fake, exist_ok=True)
open(os.path.join(_old_fake, "block.bin"), "wb").write(b"kv")

# The real test drives the method, with the legacy mapper monkeypatched to
# point at our planted directory -- otherwise the assertion only ever exercises
# a path that does not exist, which is a guard that cannot fail.
_realFM = diskspec.FileMapper


class _FakeFM:
    def __init__(self, *a, **kw):
        self.base_path = os.path.join(_mg.root_dir, "legacy-fingerprint")


_shP.rmtree(_newbase, ignore_errors=True)
diskspec.FileMapper = _FakeFM
try:
    _mg._migrate_legacy_base(_CfgTmp(WORLD, 0).parallel_config)
finally:
    diskspec.FileMapper = _realFM

check("*** the legacy directory is RENAMED onto the new fingerprint, so the "
      "warm tier survives a fingerprint change ***",
      os.path.isdir(_newbase) and not os.path.isdir(_old_fake),
      (os.path.isdir(_newbase), os.path.isdir(_old_fake)))
check("...and the files came with it",
      os.path.isfile(os.path.join(_newbase, "block.bin")))

# *** THE VERSION GUARD IS NOT OPTIONAL. *** Past v1 the old directory was
# written by a DIFFERENT layout, so adopting it is the exact hazard the
# version exists to prevent. A migration that outlives its own precondition is
# worse than no migration.
os.makedirs(_old_fake, exist_ok=True)
open(os.path.join(_old_fake, "block.bin"), "wb").write(b"kv")
_shP.rmtree(_newbase, ignore_errors=True)
_realV = diskspec.DISK_FORMAT_VERSION
diskspec.DISK_FORMAT_VERSION = 2
diskspec.FileMapper = _FakeFM
try:
    _mg._migrate_legacy_base(_CfgTmp(WORLD, 0).parallel_config)
finally:
    diskspec.DISK_FORMAT_VERSION = _realV
    diskspec.FileMapper = _realFM
check("*** at format v2 the migration REFUSES -- adopting v1 bytes under a v2 "
      "layout is plausible-but-wrong KV, the whole thing the version prevents "
      "***", os.path.isdir(_old_fake) and not os.path.isdir(_newbase),
      (os.path.isdir(_old_fake), os.path.isdir(_newbase)))

# *** AND THE "never clobber" GUARD HAD TO BE TESTED ON AN EMPTY TARGET. ***
# The first version planted a file in the new directory and asserted it
# survived -- which it does EITHER WAY, because POSIX `rename` refuses to
# replace a NON-EMPTY directory (ENOTEMPTY), so the mutation was caught by the
# kernel rather than by the guard and the assertion could not fail.
# Break-verification found that. An EMPTY target is where behaviour genuinely
# differs: there the rename would succeed, so the explicit check is what makes
# the outcome a predictable decline with a log instead of a silent adoption.
os.makedirs(_newbase, exist_ok=True)
assert not os.listdir(_newbase), "the target must be EMPTY for this to bite"
diskspec.FileMapper = _FakeFM
try:
    _mg._migrate_legacy_base(_CfgTmp(WORLD, 0).parallel_config)
finally:
    diskspec.FileMapper = _realFM
check("*** an EXISTING new-format directory is never adopted into, even when "
      "empty -- a rename cannot merge two caches, so it declines and says so "
      "***", os.path.isdir(_old_fake) and os.path.isdir(_newbase),
      (os.path.isdir(_old_fake), os.path.isdir(_newbase)))
_shP.rmtree(_old_fake, ignore_errors=True)

print("== (P4) the fingerprint is PUBLISHED, so it can be checked ==")
# *** A FINGERPRINT NOBODY CAN READ IS A CLAIM NOBODY CAN CHECK. *** The
# 2026-10-07 failure was the worker and the engine core computing DIFFERENT
# base paths with nothing saying so: 552 keys pruned as "the file is gone"
# with the files right there, one boot per guess until the fields were printed.
_fpm = _mg.get_manager()
_fp = _fpm.snapshot().get("fingerprint") or {}
eq("the format version is published", _fp.get("format_version"),
   diskspec.DISK_FORMAT_VERSION)
eq("...and the root", _fp.get("format_root"), diskspec.DISK_FORMAT_ROOT)
check("the base path is published, which is what two processes must AGREE on",
      bool(_fp.get("base_path")), _fp)
check("*** the model identity is published separately from the name, because "
      "`unknown` is a fact about the deployment worth seeing ***",
      "model_identity" in _fp, _fp)
check("...and the writer string carries our format version",
      "dspark-kv" in (_fp.get("writer") or ""), _fp.get("writer"))

# It must survive a manager whose __init__ never ran -- the rebind trap, which
# has now bitten seven times in this module.
_orph = diskspec.DiskPresenceManager.__new__(diskspec.DiskPresenceManager)
try:
    _got = diskspec.DiskPresenceManager._fingerprint(_orph)
except Exception as exc:
    _got = "RAISED: %r" % (exc,)
check("*** _fingerprint works on an instance that never ran __init__ ***",
      isinstance(_got, dict) and _got.get("format_version") is not None, _got)

print("== (P5) pos MUST be the position in the PREFIX, not in the batch ==")
# *** THIS IS THE BUG THAT STRANDED A 300 GB CACHE. ***
# `todo` is the NOT-PRESENT subset of `keys`, so indexing `todo` numbers keys
# within THIS BATCH. For an incremental store -- a conversation growing, the
# common case -- blocks 3..4 of a prefix are the only new ones and get pos
# 0..1, labelling a block deep in the prefix as a HEAD. Tail-first eviction
# then takes heads, and reuse is a contiguous run from token 0, so everything
# behind a lost head becomes unreachable.
#
# Measured on the box: after evicting 150,358 keys, 41% of offered blocks were
# already on disk and 27 external lookups produced ZERO hits.
#
# No existing test covered the incremental case, which is exactly why it
# shipped: every other prepare_store test stores a whole run at once, where
# indexing `todo` and indexing `keys` give the same answer.
F.write(kvmgr_enabled=1, disk_restore=1, disk_budget_gb=0)
_pm = diskspec.DSparkDiskOffloadingSpec(_CfgTmp(WORLD, 0), _KVCfg()).get_manager()

# One group, five blocks, offered in prefix order.
_run = [base.make_offload_key(bytes([0xA0 + n]) * 32, 0) for n in range(5)]

# First store: the whole run is new, so batch position == prefix position and
# the OLD code would also have been right here. This arm exists to prove the
# fixture can express the RIGHT answer, not only the wrong one.
_pm.prepare_store(_run, ctx)
_pm.complete_store(_run, ctx)
eq("a whole-run store records 0..4",
   [_pm._present[bytes(k)][1] for k in _run], [0, 1, 2, 3, 4])

# Now the incremental case: blocks 0..2 are already present, and only 3..4 are
# new. The full run is still offered -- that is what `skipped_present` counts.
_pm2 = diskspec.DSparkDiskOffloadingSpec(_CfgTmp(WORLD, 0), _KVCfg()).get_manager()
_pm2.prepare_store(_run[:3], ctx)
_pm2.complete_store(_run[:3], ctx)
_pm2.prepare_store(_run, ctx)          # all five offered, three already held
_pm2.complete_store(_run, ctx)
_got = [_pm2._present[bytes(k)][1] for k in _run]
eq("*** the LATER blocks keep their PREFIX positions 3 and 4 -- indexing the "
   "batch would have labelled them 0 and 1, i.e. as heads ***",
   _got[3:], [3, 4])
eq("...and the whole run reads as a prefix, 0..4", _got, [0, 1, 2, 3, 4])

# And the consequence that actually matters: with correct positions, the victim
# order takes the TAIL of the prefix, so the head survives and the prefix still
# hits. That is the entire point of tail-first.
_items = [(bytes(k), 1000, _got[n], 0) for n, k in enumerate(_run)]
_victims = diskspec.choose_victims(_items, 2000)
eq("*** tail-first evicts the LAST blocks, leaving the head intact ***",
   _victims, [bytes(_run[4]), bytes(_run[3])])
check("...so block 0 -- the head -- is never among the first victims",
      bytes(_run[0]) not in _victims, _victims)

# *** AND pos MUST ONLY BE RECORDED FOR KEYS WE ARE STORING, or `_pos_hint`
# *** LEAKS. *** Entries leave it only via the `pop` in complete_store, so an
# entry written for an ALREADY-PRESENT key is never removed. The scheduler
# re-offers present keys constantly -- 6,801 of them in half an hour on the
# box -- so this would grow without bound on the scheduler's own path. Not a
# correctness bug, which is why the position assertions above cannot see it:
# a memory one, found by break-verification asking what the `if` was for.
# *** AND THE SCENARIO HAS TO BE THE MIXED ONE. *** My first attempt offered a
# run whose keys were ALL present, and `prepare_store` returns at
# `if not todo` BEFORE the position loop -- so the loop never ran and the test
# could not see the leak however broken the code was. The leak needs `todo`
# non-empty AND some keys present, which is the incremental case again.
_pm3 = diskspec.DSparkDiskOffloadingSpec(_CfgTmp(WORLD, 0), _KVCfg()).get_manager()
_pm3.prepare_store(_run[:3], ctx)
_pm3.complete_store(_run[:3], ctx)
eq("after a completed store, no position hints are left over",
   len(_pm3._pos_hint), 0)
for _ in range(5):
    # Three present, two new: the loop runs, and only the two NEW keys may be
    # recorded. Recording all five would leave three behind on every pass.
    _pm3.prepare_store(_run, ctx)
check("*** a mixed store records hints only for the keys being STORED -- five "
      "passes and at most the two new ones are outstanding ***",
      len(_pm3._pos_hint) <= 2, len(_pm3._pos_hint))
_pm3.complete_store(_run, ctx)
eq("...and completing the store leaves the map empty, so nothing accumulates",
   len(_pm3._pos_hint), 0)

print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
