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
print("== (1i) get_handlers registers the STORE direction ONLY ==")
# The handler needs a GPU, but get_handlers imports it INSIDE the function, so
# patching the module attribute gives a real test of the registration itself --
# which is the part that decides whether a restore is even routable.
from kvmgr import diskio  # noqa: E402

_built = {}


class _FakeHandler:
    def __init__(self, **kw):
        _built.update(kw)


_real = diskio.DiskStoreHandler
diskio.DiskStoreHandler = _FakeHandler
try:
    triples = list(sp.get_handlers(kv_caches=object()))
finally:
    diskio.DiskStoreHandler = _real

eq("exactly one handler is registered", len(triples), 1)
src_cls, dst_cls, _h = triples[0]
eq("...src is GPU", src_cls.medium(), "GPU")
eq("...dst is our disk medium", dst_cls.medium(), "DSPARK_DISK")
# *** THE LOAD DIRECTION MUST NOT EXIST. *** This is the same refusal lookup()
# makes, done a second time on purpose: with no (DISK -> GPU) handler
# registered, OffloadingWorker has nothing to route a load to, so a restore
# cannot happen even if a later edit made lookup() optimistic.
check("*** with disk_restore OFF, no (DISK -> GPU) handler exists at all, so "
      "a restore is not even routable ***",
      not any(s.medium() == "DSPARK_DISK" and d.medium() == "GPU"
              for s, d, _ in triples))

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
    diskio.DiskStoreHandler = _real
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
print(f"{PASS} passed, {len(FAIL)} failed")
F.remove()
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
