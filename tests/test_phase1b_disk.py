"""test_phase1b_disk.py -- the pure core of the disk write path.

    python3.12 local-ops/kvharness/test_phase1b_disk.py

Everything here runs with no torch and no GPU, because everything here is where
a SILENT KV corruption would live: which bytes go in which file, which staging
slot pairs with which key, and whether a half-written file can ever be mistaken
for a complete one. The torch wiring in diskio.DiskStoreHandler is a thin shell
over these functions and is covered only by the box gate.
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
from kvmgr import diskio, diskspec, flags  # noqa: E402

F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv1b-"), "flags.json"))


class Ref:
    def __init__(self, tensor_idx, page_size_bytes):
        self.tensor_idx = tensor_idx
        self.page_size_bytes = page_size_bytes


# The live five-group shape. Padded tensor pages vs un-padded ref pages is the
# distinction that decides whether we write slack onto disk.
REFS = [
    [Ref(0, 4096), Ref(1, 4096)],   # group 0, two layers
    [Ref(2, 1024)],                 # group 1
    [Ref(2, 1024)],                 # group 2, SAME tensor as group 1
    [Ref(3, 64)],                   # group 3
    [Ref(4, 128)],                  # group 4
]

print("== (b1) file_plan: which bytes belong in which file ==")
p0 = diskio.file_plan(REFS, 0, slot=7)
eq("group 0 plans one chunk per LAYER", len(p0), 2)
eq("...from the tensors its refs name", [c.tensor_idx for c in p0], [0, 1])
eq("...all from the one staging slot", [c.slot for c in p0], [7, 7])
eq("...sized by the UN-PADDED ref page, not the tensor's padded one",
   [c.nbytes for c in p0], [4096, 4096])
eq("group 0's file is the sum", diskio.plan_nbytes(p0), 8192)
eq("a single-layer group is one chunk", len(diskio.file_plan(REFS, 3, 0)), 1)
eq("...of that group's size", diskio.plan_nbytes(diskio.file_plan(REFS, 3, 0)), 64)
# Two groups sharing a tensor must still get DIFFERENT files, and the group
# index is the only thing distinguishing them.
eq("groups 1 and 2 share tensor 2", diskio.file_plan(REFS, 1, 0)[0].tensor_idx,
   diskio.file_plan(REFS, 2, 0)[0].tensor_idx)

print()
print("== (b2) the per-group sizes, and the invariant the tiering tier broke ==")
sizes = diskio.group_file_sizes(REFS)
eq("one size per group", len(sizes), 5)
eq("the sizes", sizes, [8192, 1024, 1024, 64, 128])
# THE INVARIANT: the five files sum to the per-rank page. The tiering tier wrote
# the WHOLE page into EVERY group's file, so its total was 5x the page and
# half of each file was zero.
eq("*** the five files SUM to the per-rank page ***", sum(sizes), 10432)
check("...which is far less than 5 x the page, the tiering tier's total",
      sum(sizes) < 5 * sum(sizes) / 2)

bad = False
try:
    diskio.file_plan(REFS, 5, 0)
except IndexError:
    bad = True
check("a key naming a group this engine lacks RAISES rather than writing "
      "something", bad)

print()
print("== (b3) check_pairing -- the silent-corruption guard ==")
gidx = base.get_offload_group_idx
K = base.make_offload_key


def keys(*gs):
    return [K(bytes([i + 1]) * 32, g) for i, g in enumerate(gs)]


# The good case: keys in the scheduler's group-major order.
ks = keys(0, 0, 1, 2)
check("a correctly ordered batch passes",
      diskio.check_pairing(ks, [10, 11, 12, 13], [2, 1, 1, 0, 0], gidx) is None)

# 1. the off-by-one the scheduler's `block_id == 0` skip would produce.
why = diskio.check_pairing(ks, [10, 11, 12], [2, 1, 0, 0, 0], gidx)
check("*** a key/block count mismatch is CAUGHT ***", why is not None)
check("...and the reason names the real consequence", "another block's file" in (why or ""))

# 2. reordering: same length, wrong groups.
why2 = diskio.check_pairing(keys(0, 1, 1, 2), [10, 11, 12, 13], [2, 1, 1, 0, 0], gidx)
check("*** a per-group count mismatch is CAUGHT ***", why2 is not None)
check("...and names the group", "group 0" in (why2 or ""), why2)

# 3. group_sizes that do not sum to the block count.
check("group_sizes disagreeing with the block count is caught",
      diskio.check_pairing(ks, [10, 11, 12, 13], [1, 1, 1, 0, 0], gidx) is not None)

# And the honest limit, asserted so nobody mistakes this for a proof: two keys
# transposed WITHIN one group pass both checks. Only Phase 3's content compare
# can catch that.
a, b = keys(0, 0)
check("a transposition WITHIN a group passes -- the stated limit, not a bug",
      diskio.check_pairing([b, a], [10, 11], [2, 0, 0, 0, 0], gidx) is None)

print()
print("== (b4) SlotPool: staging is small, so exhaustion is normal ==")
pool = diskio.SlotPool(4)
eq("starts full", pool.available, 4)
s1 = pool.take(3)
eq("hands out distinct slots", len(set(s1)), 3)
eq("...and accounts for them", pool.available, 1)
# *** THIS ASSERTION USED TO READ "the connector retries", WHICH IS FALSE AND
# IS WHY THE FIRST DEPLOY KILLED THE ENGINE. *** take() is all-or-nothing and
# that boundary is still worth pinning, but refusing a transfer because the pool
# is full is fatal -- see (b6). take_upto is what the handler uses.
check("take() is all-or-nothing and returns None rather than raising",
      pool.take(2) is None)
eq("...and nothing was consumed by the refusal", pool.available, 1)
part = pool.take_upto(3)
eq("*** take_upto gives what it CAN -- the interface chunking needs ***",
   len(part), 1)
pool.give(part)
pool.give(s1)
eq("returning restores them", pool.available, 4)
eq("high water counts BOTH take paths (3 + the partial 1)",
   pool.snapshot()["high_water"], 4)
twice = False
try:
    pool.give([s1[0]])
except AssertionError:
    twice = True
check("*** a double free RAISES -- it would alias two blocks onto one slot ***",
      twice)
empty = False
try:
    diskio.SlotPool(0)
except ValueError:
    empty = True
check("a zero-size pool is refused at construction", empty)

print()
print("== (b5) write_chunks is ATOMIC, and that IS the presence check ==")
d = tempfile.mkdtemp(prefix="kvdisk1b-")
rows = {(0, 0): b"A" * 4096, (1, 0): b"B" * 4096}


def read_row(tidx, slot, nbytes):
    return rows[(tidx, slot)][:nbytes]


path = os.path.join(d, "sub", "dir", "block.bin")
n = diskio.write_chunks(path, diskio.file_plan(REFS, 0, 0), read_row)
eq("writes the planned byte count", n, 8192)
check("...creating the directory tree", os.path.exists(path))
eq("...at the planned size", os.path.getsize(path), 8192)
import hashlib  # noqa: E402
eq("...with the chunks in REF ORDER, which the read path must replay",
   hashlib.sha256(open(path, "rb").read()).hexdigest()[:16],
   hashlib.sha256(b"A" * 4096 + b"B" * 4096).hexdigest()[:16])
check("no .part file is left behind", not os.path.exists(path + ".part"))

# *** THE ATOMICITY TEST PROPER, and the first version of it was vacuous. ***
# Asserting "no file at the real path after a failure" passes even for a direct
# write, because the failure handler's unlink removes it -- so that assertion
# tested the CLEANUP, not the atomicity. Break-verification caught it: making
# the write non-atomic left every assertion green.
# What actually distinguishes them is what a CONCURRENT READER can see. A reader
# decides a block is cached with os.path.exists(), so the real path must not
# appear until the content is complete.
seen = {}
p3 = os.path.join(d, "atomic.bin")


def watch_row(tidx, slot, nbytes):
    # called once per chunk, i.e. while the write is in progress
    seen.setdefault("mid_write_exists", os.path.exists(p3))
    return rows[(tidx, slot)][:nbytes]


diskio.write_chunks(p3, diskio.file_plan(REFS, 0, 0), watch_row)
check("*** the real path does NOT exist mid-write, so exists() can never see "
      "a partial ***", seen["mid_write_exists"] is False)
check("...and does once the write completes", os.path.exists(p3))
eq("...at full size", os.path.getsize(p3), 8192)


# A failure midway must leave NOTHING at the real path. A short file there would
# satisfy os.path.exists() -- which is how a reader decides a block is cached --
# and restore as truncated KV.
def boom(tidx, slot, nbytes):
    if tidx == 1:
        raise OSError("disk full")
    return rows[(tidx, slot)][:nbytes]


p2 = os.path.join(d, "partial.bin")
failed = False
try:
    diskio.write_chunks(p2, diskio.file_plan(REFS, 0, 0), boom)
except OSError:
    failed = True
check("a write that fails midway propagates", failed)
check("*** and leaves NO file at the real path, so exists() cannot lie ***",
      not os.path.exists(p2))
check("...and no .part either", not os.path.exists(p2 + ".part"))

# A staging row of the wrong length is refused rather than written short.
short = False
try:
    diskio.write_chunks(os.path.join(d, "short.bin"),
                        diskio.file_plan(REFS, 0, 0),
                        lambda t, s, n: b"x" * (n - 1))
except ValueError:
    short = True
check("a staging row of the wrong length is REFUSED", short)
check("...with no file left", not os.path.exists(os.path.join(d, "short.bin")))



# ---------------------------------------------------------------------------
# (b6) THE CHUNKING STATE MACHINE -- added after the box killed the engine.
#
# The first deploy died on the first request: transfer_async returned False
# because a store job (~190 blocks in group 0 for a 48K prompt) exceeded the
# 32-block staging pool, and the connector's `assert success` is fatal.
# `_unsubmitted_store_jobs` is a DEFERRAL queue, not a retry queue -- I inferred
# backpressure from its name, and the (b4) test above asserted that false
# premise in as many words.
#
# So the handler now queues a job and pumps it over rounds. That state machine
# is the new risk, and it is exercised here against fakes by constructing the
# handler WITHOUT __init__ -- the real transfer_async, _pump and get_finished
# run, which is where a bug would be.
# ---------------------------------------------------------------------------
print()
print("== (b6) chunking: a job larger than the pool must still be accepted ==")

from kvmgr.diskio import DiskStoreHandler  # noqa: E402

wrk = sys.modules["vllm.v1.kv_offload.worker.worker"]


class FakeInner:
    """Stands in for SingleDirectionOffloadingHandler: records and completes."""

    def __init__(self):
        self.subs = []
        self._done = []
        self.fail_next = False

    def transfer_async(self, job_id, spec):
        gpu, cpu = spec
        self.subs.append((job_id, list(gpu.block_ids), list(gpu.group_sizes),
                          list(cpu.block_ids)))
        self._done.append(wrk.TransferResult(job_id=job_id,
                                            success=not self.fail_next))
        return True

    def get_finished(self):
        out, self._done = self._done, []
        return out


class SyncWriters:
    """A ThreadPoolExecutor stand-in that runs inline, so the test is ordered."""

    class _F:
        def __init__(self, fn):
            try:
                self._v, self._e = fn(), None
            except BaseException as e:      # noqa: BLE001
                self._v, self._e = None, e

        def done(self):
            return True

        def result(self):
            if self._e:
                raise self._e
            return self._v

    def submit(self, fn, *a):
        return SyncWriters._F(lambda: fn(*a))

    def shutdown(self, wait=True):
        pass


def _dsrc_take():
    """The two take_upto call sites, so a reserve creeping back in is caught.

    Source-text rather than behaviour because a reserve of 0 behaves exactly
    like no reserve, so no fixture could tell them apart -- and the thing worth
    pinning is that the SPLIT made the argument meaningless, not that it
    happens to be zero today.
    """
    import os as _os
    src = open(_os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                             "diskio.py")).read()
    out = []
    for i in range(len(src)):
        j = src.find("self._pool.take_upto(", i)
        if j < 0:
            break
        out.append(src[j:src.index(")", j) + 1])
        i = j + 1
    return " ".join(dict.fromkeys(out))


def mkhandler(pool_size, nrefs=REFS):
    h = DiskStoreHandler.__new__(DiskStoreHandler)
    h._spec = type("S", (), {"disk_medium": "DSPARK_DISK"})()
    h._mapper = type("M", (), {"get_file_name": staticmethod(
        lambda k: os.path.join(WDIR, base.get_offload_block_hash(k).hex()[:20]
                               + f"_g{base.get_offload_group_idx(k)}.bin"))})()
    h._refs = nrefs
    h._sizes = diskio.group_file_sizes(nrefs)
    h._inner = FakeInner()
    h._pool = diskio.SlotPool(pool_size)
    h._writers = SyncWriters()
    h._jobs, h._inner_to_job, h._writing = {}, {}, []
    h._next_inner, h._saturated = 1, False
    # *** THE FIXTURE MUST CARRY EVERY COUNTER THE REAL CONSTRUCTOR SETS. ***
    # Hand-building `stats` means a new counter breaks the suite with a
    # KeyError rather than being covered -- which is a loud failure and so
    # tolerable, but it is the second time today a hand-built fixture needed
    # updating for a field the real object gained. Mirrored from
    # DiskStoreHandler.__init__; the fence counters are the ones that measure
    # whether a synchronous flush is what costs the ~2x prefill slowdown.
    h.stats = {"submitted": 0, "refused_pairing": 0, "refused_bounds": 0,
               "files": 0, "bytes": 0,
               "errors": 0, "wait_calls": 0, "wait_ours": 0,
               "wait_blocked_ms": 0, "wait_max_pending": 0, "wait_rounds": 0,
               # Added when the real constructor gained them, which the
               # fixture-parity assertion below is here to force.
               "skipped_before_copy": 0, "delete_errors": 0, "deleted": 0,
               "saturations": 0, "capped": 0}
    h._read_row = lambda t, s, n: b"\xab" * n
    # *** THE FIXTURE MUST DECLARE ITS TRANSFER BOUNDS. *** `_bounds()` derives
    # them from the real tensors, which this shell does not have -- so without
    # this it reports (0, 0) and the bounds guard correctly refuses every
    # transfer, which is a fixture that cannot express the behaviour under test.
    # 22,756 is the real GPU row count from the live geometry snapshot, and
    # staging rows equal the pool capacity by construction.
    h._bounds_cache = (22756, pool_size)
    return h


WDIR = tempfile.mkdtemp(prefix="kvchunk-")
K, gidx = base.make_offload_key, base.get_offload_group_idx


def gkeys(group, n, salt=0):
    return [K(bytes([group + 1, salt, i // 256, i % 256]) * 8, group)
            for i in range(n)]


# 100 blocks in group 0 against a pool of 8: the shape that killed the engine.
#
# *** THE PER-ROUND BUDGET MADE THIS SCENARIO UNREACHABLE, which is why the
# knobs are raised for it. *** With store_inflight_max=1 the pump returns after
# one round is outstanding, so it never reaches an empty pool and
# `saturations` -- the one existing signal for "disk writes are falling
# behind" -- could not fire at all. This block is about the POOL being the
# limiter, so the budget is lifted out of the way here and tested on its own
# below. Had the test simply been relaxed to match the new behaviour, a real
# signal would have gone quiet with every assertion green.
F.write(store_inflight_max=64, store_batch_max=4096, store_ops_max=0)
flags.reset_for_tests()
h = mkhandler(8)
ks = gkeys(0, 100)
gpu = base.GPULoadStoreSpec(list(range(1000, 1100)), group_sizes=[100, 0, 0, 0, 0],
                            block_indices=[0] * 5)
check("*** a 100-block job against an 8-block pool is ACCEPTED, not refused ***",
      h.transfer_async(77, (gpu, diskspec.DiskLoadStoreSpec(ks))) is True)
eq("...and only a pool's worth is submitted at first", len(h._inner.subs), 1)
# *** THE FLOOR NO LONGER CHANGES THIS NUMBER, and that is the staging SPLIT.
# The store and load used to share one pool, which is what made the same restore
# take 574.9s with stores live against 3.3s with them paused -- and, far worse,
# is what faulted the worker with Xid 13 five times, because the load reads that
# buffer under CU_MEMCPY_SRC_ACCESS_ORDER_ANY whose precondition is that no
# concurrent GPU stream writes the source. The load has its own allocation now,
# so each pool has ONE user and a store round takes the whole live size.
# `staging_plan` still computes the floors (it is a pure function with its own
# tests) but the take path no longer passes them; flags.py marks them
# superseded.
_sz, _wf, _rf = diskio.staging_plan(8, 0, 0, 0)
eq("staging_plan still derives a floor for an 8-slot pool", _rf, 2)
check("...but the take path does NOT use it -- one user per pool now",
      "reserve=" not in _dsrc_take())
eq("...so a store round takes the WHOLE live size", len(h._inner.subs[0][1]),
   _sz)
check("...with the job still queued", h._jobs[77]["batches"] != [])
check("...and the saturation note said ONCE", h._saturated is True)

# ---------------------------------------------------------------------------
# *** THE PER-ROUND BUDGET, where the BUDGET is the limiter rather than the
# pool. *** A big pool and a small cap: one scheduler step must carry the cap,
# not the pool, because what stalled the stream was a step that submitted ~256
# blocks x 62 refs at once (test-meter.sh section 1: one 1.052s stall, max gap
# 1.137s against a 0.102s baseline).
F.write(store_inflight_max=1, store_batch_max=5, store_ops_max=0)
flags.reset_for_tests()
_hb = mkhandler(256)
_kb = gkeys(0, 100, salt=9)
_gb = base.GPULoadStoreSpec(list(range(2000, 2100)), group_sizes=[100, 0, 0, 0, 0],
                            block_indices=[0] * 5)
check("a 100-block job is accepted against a 256-slot pool",
      _hb.transfer_async(91, (_gb, diskspec.DiskLoadStoreSpec(_kb))) is True)
eq("*** ONE round is submitted, not eight ***", len(_hb._inner.subs), 1)
eq("...and it carries the BATCH CAP, not the pool's 256", len(_hb._inner.subs[0][1]), 5)
eq("...and the shrink is COUNTED, so a binding cap is visible",
   _hb.stats["capped"], 1)
check("...and the job is still queued, to be pumped over later steps",
      _hb._jobs[91]["batches"] != [])
# the inflight gate is what bounds the STEP: pumping again while the round is
# open must add nothing
_before = len(_hb._inner.subs)
_hb._pump()
eq("a second pump while a round is open submits NOTHING (the step is bounded)",
   len(_hb._inner.subs), _before)
# and the ops budget must be able to bind on its own, below the block cap
F.write(store_inflight_max=1, store_batch_max=4096, store_ops_max=3)
flags.reset_for_tests()
_ho = mkhandler(256)
_ko = gkeys(0, 100, salt=11)
_go = base.GPULoadStoreSpec(list(range(3000, 3100)), group_sizes=[100, 0, 0, 0, 0],
                            block_indices=[0] * 5)
_ho.transfer_async(92, (_go, diskspec.DiskLoadStoreSpec(_ko)))
_refs0 = len(_ho._refs[0])
_want = max(1, min(100, 3 // max(1, _refs0)))
eq("the OPS budget binds below the block cap, in descriptors not blocks "
   "(group 0 has %d refs/block, so 3 descriptors buys %d)" % (_refs0, _want),
   len(_ho._inner.subs[0][1]), _want)
# restore the shipped defaults for anything after this block
F.write(store_inflight_max=1, store_batch_max=32, store_ops_max=2048)
flags.reset_for_tests()

rounds, done = 0, []
while not done and rounds < 100:
    done = h.get_finished()
    rounds += 1
eq("the job completes after pumping", len(done), 1)
eq("...reporting OUR job id, not an inner one", done[0].job_id, 77)
check("...as a success", done[0].success is True)
# *** NO FLOOR IN THE ARITHMETIC ANY MORE. *** With the load on its own pinned
# allocation each pool has exactly one user, so the store takes the WHOLE live
# size per round and the count is ceil(100 / size), not ceil(100 / (size -
# read_floor)). Derived from the knob rather than written as a constant, so it
# still fails if the pool stops handing out what it has.
import math as _math  # noqa: E402
eq("...over ceil(100 / size) rounds, the floor no longer applying",
   len(h._inner.subs), _math.ceil(100 / _sz))
check("*** and every block still moves exactly once -- chunking must not "
      "drop or duplicate work ***",
      sorted(b for sub in h._inner.subs for b in sub[1]) == list(range(1000, 1100)))
eq("...covering every block exactly once",
   sorted(b for s in h._inner.subs for b in s[1]), list(range(1000, 1100)))
eq("...and every key got a file", h.stats["files"], 100)
eq("all staging slots are returned", h._pool.available, 8)
eq("no job state is left behind", len(h._jobs), 0)
check("the transfer_type label names our medium",
      done[0].transfer_type == ("GPU", "DSPARK_DISK"))

print()
print("== (b7) batching is PER GROUP, because the inner walks group_sizes ==")
h2 = mkhandler(4)
mixed = gkeys(0, 3) + gkeys(2, 2) + gkeys(4, 1)
gpu2 = base.GPULoadStoreSpec([1, 2, 3, 4, 5, 6], group_sizes=[3, 0, 2, 0, 1],
                             block_indices=[0] * 5)
h2.transfer_async(9, (gpu2, diskspec.DiskLoadStoreSpec(mixed)))
done2 = []
for _ in range(30):
    done2 += h2.get_finished()
eq("the job completes", [d.job_id for d in done2], [9])
for jid, bids, gs, slots in h2._inner.subs:
    nz = [i for i, v in enumerate(gs) if v]
    eq(f"round {jid} touches exactly one group", len(nz), 1)
    eq(f"...and its group_sizes matches its block count", gs[nz[0]], len(bids))
eq("...and the files are named by their own group",
   sorted({os.path.basename(p).split('_g')[1][0]
           for p in os.listdir(WDIR) if p.endswith(".bin")}), ["0", "2", "4"])

# *** AND EACH FILE MUST BE ITS OWN GROUP'S SIZE. *** The name comes from the
# KEY, so a handler that used the wrong group for the BYTE PLAN would write
# group 0's layout under a group-2 name -- right name, wrong content, which is
# the exact silent corruption this suite exists for. Break-verification found
# this gap: passing the wrong group index to the write was NOT caught until
# this assertion existed.
want = {str(g): sz for g, sz in enumerate(diskio.group_file_sizes(REFS))}
got = {}
for f in os.listdir(WDIR):
    if not f.endswith(".bin"):
        continue
    got.setdefault(f.split("_g")[1][0], set()).add(
        os.path.getsize(os.path.join(WDIR, f)))
for g in sorted(got):
    eq(f"*** group {g}'s files are group {g}'s size, not another group's ***",
       got[g], {want[g]})

print()
print("== (b8) a failed round fails the JOB, and frees everything ==")
h3 = mkhandler(4)
h3._inner.fail_next = True
h3.transfer_async(5, (base.GPULoadStoreSpec([1, 2], group_sizes=[2, 0, 0, 0, 0],
                                            block_indices=[0] * 5),
                      diskspec.DiskLoadStoreSpec(gkeys(0, 2, salt=9))))
d3 = []
for _ in range(20):
    d3 += h3.get_finished()
eq("the job is reported", [x.job_id for x in d3], [5])
check("*** as a FAILURE, not silently dropped ***", d3[0].success is False)
eq("...and the slots came back", h3._pool.available, 4)

print()
print("== (b9) the pairing refusal is STILL fatal, on purpose ==")
h4 = mkhandler(8)
# one key too few: the off-by-one the scheduler's block_id==0 skip produces
check("a desynced job is refused",
      h4.transfer_async(1, (base.GPULoadStoreSpec([1, 2, 3], group_sizes=[3, 0, 0, 0, 0],
                                                  block_indices=[0] * 5),
                            diskspec.DiskLoadStoreSpec(gkeys(0, 2, salt=3)))) is False)
eq("...counted", h4.stats["refused_pairing"], 1)
eq("...and nothing was submitted", len(h4._inner.subs), 0)
eq("...nor any slot consumed", h4._pool.available, 8)



print()
print("== (b10) the geometry snapshot -- the thing that ends the restart tax ==")

# The question it exists for: our per-group sizes summed to 4.29x the per-block
# bytes the engine's own KV allocation implies, and arithmetic cannot say whether
# that is shared tensors or differing row counts. These two fixtures are the two
# answers, so the snapshot is asserted to DISTINGUISH them rather than merely to
# serialise.
class FakeT:
    def __init__(self, rows, page):
        self.tensor = type("X", (), {"shape": (rows, page)})()
        self.page_size_bytes = page


class FakeKV:
    def __init__(self, tensors, refs):
        self.tensors = tensors
        self.group_data_refs = refs


fake_spec = type("S", (), {"rank": 1, "world_size": 2, "block_size_factor": 1,
                           "whole_page_bytes": 2103552,
                           "page_size_per_worker": 1051776,
                           "gpu_block_size": (256, 64, 64, 4, 8),
                           "hash_block_size": 4})()

# (a) groups SHARE tensor 2 -- the redundancy answer
shared_kv = FakeKV([FakeT(100, 4096), FakeT(100, 4096), FakeT(100, 1024)], REFS)
gs = diskio.geometry(shared_kv, fake_spec)
eq("the snapshot carries the rank, so the two nodes are distinguishable",
   gs["rank"], 1)
eq("...and the derived page we write", gs["page_size_per_worker"], 1051776)
eq("...and the live block sizes", gs["gpu_block_size"], [256, 64, 64, 4, 8])
check("*** shared_tensors NAMES the groups that share -- the 4x answer ***",
      gs["shared_tensors"] == {"2": [1, 2]}, repr(gs["shared_tensors"]))
eq("...and per-group bytes are still reported per group",
   [g["bytes_per_block"] for g in gs["groups"]], [8192, 1024, 1024, 64, 128])
eq("...while the DISTINCT tensor pages sum to less than the group sum",
   gs["sum_distinct_tensor_page_bytes"] < gs["sum_group_bytes_per_block"], True)
eq("rows are reported, which is the OTHER candidate explanation",
   [t["rows"] for t in gs["tensors"]], [100, 100, 100])

# (b) groups DISJOINT -- then differing row counts must be the explanation
disjoint = [[REFS[0][0]], [REFS[1][0]], [REFS[3][0]]]
dj = FakeKV([FakeT(100, 4096), FakeT(50, 1024), FakeT(25, 64)], disjoint)
gd = diskio.geometry(dj, fake_spec)
eq("*** with disjoint groups, shared_tensors is EMPTY ***",
   gd["shared_tensors"], {})
eq("...and the differing row counts are visible instead",
   [t["rows"] for t in gd["tensors"]], [100, 50, 25])

# It is written atomically and never breaks a boot.
sd = tempfile.mkdtemp(prefix="kvgeo-")
pth = diskio.write_geometry(shared_kv, fake_spec, os.path.join(sd, "sub"))
check("the snapshot is written, creating its directory", pth and os.path.exists(pth))
check("...named by rank, so rank 0 and rank 1 cannot overwrite each other",
      os.path.basename(pth) == "geometry-r1.json")
check("...and left no .part", not os.path.exists(pth + ".part"))
import json  # noqa: E402
eq("...and is valid JSON holding the answer",
   json.load(open(pth))["shared_tensors"], {"2": [1, 2]})
check("*** a snapshot failure returns None rather than raising -- it must "
      "never be able to break a boot ***",
      diskio.write_geometry(object(), fake_spec, sd) is None)

print()
print("== (b11) read_chunks: the size check BEFORE any byte is read ==")

# Why this ordering is the safety property: upstream does `assert
# transfer_result.success` under "we currently do not support job failures", so
# a load has two honest outcomes on a bad file -- kill the engine, or hand back
# silently wrong KV. Checking length FIRST means a wrong file never reaches a
# GPU tensor and the caller can refuse loudly.
rdir = tempfile.mkdtemp(prefix="kvread-")
src = {(0, 0): b"A" * 4096, (1, 0): b"B" * 4096}
good = os.path.join(rdir, "good.bin")
diskio.write_chunks(good, diskio.file_plan(REFS, 0, 0),
                    lambda t, s_, n: src[(t, 0)][:n])

back = {}
n = diskio.read_chunks(good, diskio.file_plan(REFS, 0, 0),
                       lambda t, sl, b: back.__setitem__((t, sl), bytes(b)))
eq("a good file reads back its full length", n, 8192)
eq("...into the right staging rows, in ref order",
   sorted(back), [(0, 0), (1, 0)])
eq("*** and the bytes ROUND TRIP exactly ***",
   (hashlib.sha256(back[(0, 0)]).hexdigest()[:12],
    hashlib.sha256(back[(1, 0)]).hexdigest()[:12]),
   (hashlib.sha256(b"A" * 4096).hexdigest()[:12],
    hashlib.sha256(b"B" * 4096).hexdigest()[:12]))

# a SHORT file is refused, and nothing is written to staging
short = os.path.join(rdir, "short.bin")
open(short, "wb").write(b"A" * 4095)
touched = []
refused = False
try:
    diskio.read_chunks(short, diskio.file_plan(REFS, 0, 0),
                       lambda t, sl, b: touched.append(t))
except ValueError as e:
    refused = "REFUSING" in str(e)
check("*** a SHORT file is refused ***", refused)
eq("...and NOTHING reached staging, because the check precedes the read",
   touched, [])

# a file of the WRONG GROUP's size is refused too -- same mechanism, and this
# is the case a per-group byte plan makes detectable at all.
wrong = os.path.join(rdir, "wrong.bin")
open(wrong, "wb").write(b"C" * diskio.plan_nbytes(diskio.file_plan(REFS, 1, 0)))
wrefused = False
try:
    diskio.read_chunks(wrong, diskio.file_plan(REFS, 0, 0), lambda *a: None)
except ValueError:
    wrefused = True
check("a file sized for ANOTHER group is refused", wrefused)

# a missing file names the recovery, because the manifest is the only thing
# that could have claimed it.
missing = False
try:
    diskio.read_chunks(os.path.join(rdir, "nope.bin"),
                       diskio.file_plan(REFS, 0, 0), lambda *a: None)
except FileNotFoundError as e:
    missing = "presence.json" in str(e)
check("a missing file raises and NAMES the recovery (delete presence.json)",
      missing)

print()
print("== (b12) the presence manifest survives a restart, and is PRUNED ==")

mdir = tempfile.mkdtemp(prefix="kvman-")
mpath = os.path.join(mdir, "presence.json")
k_a = K(b"\xa1" * 32, 0)
k_b = K(b"\xb2" * 32, 0)


class FakeMapper:
    """Maps a key to a path under mdir, so presence can be made true or false."""

    @staticmethod
    def get_file_name(k):
        return os.path.join(mdir, base.get_offload_block_hash(k).hex()[:8] + ".bin")


ctx2 = base.ReqContext(req_id="r")
m = diskspec.DiskPresenceManager(FakeMapper, num_workers=2, manifest_path=mpath)
# make both files real, then record them
for k in (k_a, k_b):
    open(FakeMapper.get_file_name(k), "wb").write(b"x")
m.complete_store([k_a, k_b], ctx2)
m.flush_manifest(force=True)
check("the manifest file is written", os.path.exists(mpath))
check("...and left no .part", not os.path.exists(mpath + ".part"))

# A FRESH manager -- i.e. a restarted engine -- recovers the claim.
F.write(disk_restore=1)
flags.reset_for_tests()
m2 = diskspec.DiskPresenceManager(FakeMapper, num_workers=2, manifest_path=mpath)
eq("*** a restarted engine RECOVERS presence from disk -- the whole prize ***",
   m2._stats["manifest_loaded"], 2)
eq("...and lookup now hits for a key stored before the restart",
   m2.lookup(k_a, ctx2), True)

# ...and an entry whose own file has gone is PRUNED, not trusted.
os.unlink(FakeMapper.get_file_name(k_b))
m3 = diskspec.DiskPresenceManager(FakeMapper, num_workers=2, manifest_path=mpath)
eq("*** an entry whose file is gone is PRUNED ***", m3._stats["manifest_pruned"], 1)
eq("...the survivor is kept", m3._stats["manifest_loaded"], 1)
eq("...and lookup refuses the pruned key", m3.lookup(k_b, ctx2), False)
eq("...while still hitting the live one", m3.lookup(k_a, ctx2), True)

# a corrupt manifest must cost recompute, never correctness
open(mpath, "w").write("{not json")
m4 = diskspec.DiskPresenceManager(FakeMapper, num_workers=2, manifest_path=mpath)
eq("*** an unreadable manifest yields an EMPTY index, not a wrong one ***",
   len(m4._present), 0)
eq("...so lookup simply misses", m4.lookup(k_a, ctx2), False)
F.write(disk_restore=0)
flags.reset_for_tests()


print("\n== (b12a) disk_store_paused: the OFF SWITCH, and the trap it avoids ==")

# *** A MECHANISM GETS A HOT OFF SWITCH BEFORE IT GETS A FEATURE. *** Learned
# the expensive way 2026-10-08: the store path could only be disabled by a
# SIX-MINUTE restart, so the mitigation was unaffordable and a suspect path
# stayed running on an engine a real user depended on.
from kvmgr import flags as _f  # noqa: E402

mdirP = tempfile.mkdtemp(prefix="kvpause-")


class FMP:
    @staticmethod
    def get_file_name(k):
        return os.path.join(mdirP, base.get_offload_block_hash(k).hex()[:8] + ".bin")


ctxP = base.ReqContext(req_id="rp")
kp = [K(bytes([71, 0, 0, i]) * 8, 0) for i in range(4)]
for k in kp:
    open(FMP.get_file_name(k), "wb").write(b"x")

_f.reset_for_tests()
mgrP = diskspec.DiskPresenceManager(FMP, num_workers=2,
                                    manifest_path=os.path.join(mdirP, "p.json"))

# running: a store is prepared
out = mgrP.prepare_store(kp, ctxP)
check("with the switch at its default the store path RUNS", out is not None)
eq("...offering every key", len(out.keys_to_store), 4)

# paused: prepare_store declines, so NO JOB IS CREATED AT ALL
F.write(kvmgr_enabled=1, disk_store_paused=1)
# *** NOT None, AND THE DIFFERENCE IS A FOREVER-RETRY LOOP. *** This asserted
# `None` and that was the bug: scheduler.py treats None as "I cannot" and does
# NOT advance the stored index, so the same blocks are attempted every step
# indefinitely -- measured 2026-10-08 as 3,309 warnings at ~20/second. An empty
# output means "nothing new here" and lets the request progress. So the
# property to assert is "no keys to store", never the sentinel.
_paused_out = mgrP.prepare_store(kp, ctxP)
check("*** paused yields an EMPTY output, not None ***",
      _paused_out is not None and list(_paused_out.keys_to_store) == [],
      repr(_paused_out)[:80])
check("...so the scheduler can advance instead of retrying forever",
      _paused_out is not None,
      "None leaves the stored index put and the blocks are re-attempted")
eq("...and it is COUNTED, so a quiet pause is visible on /kv",
   mgrP.snapshot()["paused"], 4)
check("...and the snapshot says it is paused",
      mgrP.snapshot()["store_paused"] is True)

# *** THE TRAP: a pause must NOT mark keys present. ***
# The obvious placement is the handler, which cannot report False (the
# connector's `assert success` makes that fatal), so it would report SUCCESS
# with zero bytes -- firing complete_store, marking the keys PRESENT with
# nothing on disk, so a later lookup claims a hit and the restore dies on a
# missing file. A pause that poisons the index is worse than no pause.
eq("*** a paused store claims NOTHING present ***",
   mgrP.snapshot()["present"], 0)
F.write(kvmgr_enabled=1, disk_store_paused=0, disk_restore=1)
eq("...so a lookup after a pause correctly MISSES", mgrP.lookup(kp[0], ctxP), False)

# and it is HOT in both directions, which is the whole point
F.write(kvmgr_enabled=1, disk_store_paused=1)
_off = mgrP.prepare_store(kp, ctxP)
check("hot off", _off is not None and list(_off.keys_to_store) == [])
F.write(kvmgr_enabled=1, disk_store_paused=0)
check("*** hot ON again, with no restart ***", mgrP.prepare_store(kp, ctxP) is not None)

# the default must be TODAY'S BEHAVIOUR, not a safety position: an empty flags
# file has to behave exactly as the engine does now.
_f.reset_for_tests()
F.write(kvmgr_enabled=1)
eq("an empty-ish flags file leaves the store path RUNNING",
   bool(_f.peek("disk_store_paused")), False)
# And the OTHER nothing-to-store case: every key already on disk. Same
# requirement -- the scheduler must be able to advance.
F.write(kvmgr_enabled=1, disk_store_paused=0)
mgrP.complete_store(kp, ctxP)
_allpresent = mgrP.prepare_store(kp, ctxP)
check("*** every key already present also yields an EMPTY output, not None ***",
      _allpresent is not None and list(_allpresent.keys_to_store) == [],
      repr(_allpresent)[:80])
eq("...and the keys really are held", mgrP.snapshot()["present"], 4)
_out2 = mgrP.prepare_store([K(bytes([72, 0, 0, 9]) * 8, 0)], ctxP)
check("...while a NEW key still produces a real store job",
      _out2 is not None and len(_out2.keys_to_store) == 1)

# the shape the scheduler needs: an empty output must still carry a usable
# store_spec and evicted_keys, or the caller's own field access raises.
for name, out in (("paused", _paused_out), ("all-present", _allpresent)):
    check(f"the {name} output carries a store_spec", out.store_spec is not None)
    eq(f"...and an empty evicted_keys for {name}", list(out.evicted_keys), [])


print("\n== (b12b) copy_into: the one line that killed the first real restore ==")

# *** REPRODUCED WITH NO TORCH, which is the point. *** The live failure was
#     ValueError: memoryview assignment: lvalue and rvalue have different
#     structures
# from `self._staging[t][slot][:len(buf)].numpy().data[:] = buf`. The staging
# tensor is int8, so its view has format 'b' (signed char), while a buffer read
# off a file is `bytes`, format 'B'. memoryview assignment requires format AND
# shape to agree and does NOT coerce.
#
# It was invisible until the box because the STORE side only ever READS that
# same view: `_read_row` hands the memoryview to write_chunks, and reading needs
# no format agreement at all. Two directions that look symmetric and are not,
# with the asymmetry in one subscript. An `array('b')` reproduces it exactly.
import array as _array

_signed = _array.array("b", [0] * 16)          # format 'b', like int8 staging
_payload = bytes(range(16))                     # format 'B', like a file read

try:
    memoryview(_signed)[:] = _payload
    check("*** the bare assignment IS the live bug (it did not raise here) ***",
          False, "if this passes, the premise changed")
except ValueError as e:
    check("*** the bare assignment raises exactly the live error ***",
          "different structures" in str(e), str(e)[:70])

n = diskio.copy_into(memoryview(_signed), _payload)
eq("copy_into copies it anyway, by casting both sides", n, 16)
eq("...and the BYTES are right, not merely accepted",
   bytes(memoryview(_signed).cast("B")), _payload)

# The reverse pairing, and the already-matching one, must both work: the helper
# is called with whatever the tensor's dtype gives, which is not ours to choose.
_dst_u = bytearray(8)
eq("a 'B' destination from a 'B' source still works",
   diskio.copy_into(memoryview(_dst_u), bytes(range(8))), 8)
eq("...correctly", bytes(_dst_u), bytes(range(8)))

_wide = _array.array("i", [0] * 4)             # 4-byte items: 16 bytes total
eq("a MULTI-BYTE itemsize destination casts too",
   diskio.copy_into(memoryview(_wide), bytes(range(16))), 16)
eq("...and lands byte for byte",
   bytes(memoryview(_wide).cast("B")), bytes(range(16)))

# A length disagreement must name BOTH sides and the recovery. memoryview's own
# message names neither, which is what made the live failure take a log dive.
try:
    diskio.copy_into(bytearray(10), bytes(4))
    check("a length mismatch is refused", False, "it did not raise")
except ValueError as e:
    msg = str(e)
    check("*** a length mismatch names both sizes ***",
          "10" in msg and "4" in msg, msg[:90])
    check("...and the recovery, so it is actionable from the log alone",
          "presence.json" in msg)

print("\n== (b12c) transfer_time: without it the path is UNOBSERVABLE ==")

# *** `kv_connector_offloading/worker.py:274` gates record_transfer on ALL
# THREE of transfer_time (truthily), transfer_size and transfer_type. *** So
# omitting the time means `vllm:kv_offload_total_bytes_total` has NO SERIES AT
# ALL -- measured 2026-10-08, with the store path demonstrably writing 1,051
# files a run and `transfer_type="..."` matching nothing in /metrics.
#
# Not cosmetic: the ABSENCE of a restore series is precisely what proved
# `prepare_load` had never been called back when the tiering tier was in the
# path. A handler that omits the time makes its own path unprovable, and this
# one did. My own Phase 0 note said so in as many words.
ht = mkhandler(64)
kt = gkeys(2, 4, salt=55)
gput = base.GPULoadStoreSpec(list(range(4000, 4004)),
                             group_sizes=[0, 0, 4, 0, 0], block_indices=[0] * 5)
ht.transfer_async(1234, (gput, diskspec.DiskLoadStoreSpec(kt)))
res = []
for _ in range(50):
    res.extend(ht.get_finished())
    if res:
        break
eq("the job reports finished", [r.job_id for r in res], [1234])
r0 = res[0]
check("*** transfer_time is SET, or the connector drops the metric ***",
      r0.transfer_time is not None, repr(r0.transfer_time))
check("...and TRUTHY, because the gate is `if transfer_result.transfer_time`",
      bool(r0.transfer_time), repr(r0.transfer_time))
check("...with transfer_size beside it", r0.transfer_size is not None)
check("...and transfer_type naming OUR medium, so the series is findable",
      r0.transfer_type == ("GPU", diskspec.DISK_MEDIUM), repr(r0.transfer_type))

# A transfer too fast to measure must still be observable: 0.0 is dropped by
# the same truthiness gate that drops None, so the floor is what keeps a fast
# restore from being invisible.
eq("an unrecorded start is None, not 0.0", diskio._elapsed(None), None)
check("a zero-length interval floors above zero rather than reading 0.0",
      diskio._elapsed(diskio._now()) > 0,
      repr(diskio._elapsed(diskio._now())))

# And the lint: the bare memoryview assignment must not come back. It is one
# subscript, it looks right, and it only fails in the load direction.
_dpath = os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr", "diskio.py")
_dsrc = open(_dpath).read()


def code_only(path):
    """The source with comments and string literals BLANKED IN PLACE.

    *** THIS REPO HAS LEARNED THE FIRST HALF TWICE ALREADY. *** A lint that
    reads raw source matches the paragraph EXPLAINING the trap, so the prose
    naming the bad idiom is itself a failure -- the `.sysdiag` CSS lint and the
    CHARTS per-second lint both shipped with that bug. `copy_into`'s docstring
    quotes the bare assignment verbatim, on purpose.

    *** AND THE OBVIOUS FIX MADE THE LINT VACUOUS, which is the second half and
    cost a break-verification to find. *** Joining the surviving tokens with a
    separator turns `.data[:] =` into `. data [ : ] =`, so the pattern can
    NEVER match and the lint passes however broken the module is. Verified by
    reintroducing the bug: only the other assertion fired.

    So the regions are blanked IN PLACE, by position, leaving every other byte
    and all the original spacing exactly as written. A lint that cannot fail
    is worse than no lint, because it reads as coverage.
    """
    import tokenize

    with open(path, "rb") as f:
        lines = f.read().decode().splitlines(keepends=True)
    with open(path, "rb") as f:
        toks = list(tokenize.tokenize(f.readline))
    for tok in toks:
        if tok.type not in (tokenize.COMMENT, tokenize.STRING):
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        for r in range(r1, r2 + 1):
            line = lines[r - 1]
            a = c1 if r == r1 else 0
            b = c2 if r == r2 else len(line.rstrip("\n"))
            keep_nl = "\n" if line.endswith("\n") and b >= len(line.rstrip("\n")) else ""
            lines[r - 1] = line[:a] + " " * (b - a) + line[b:] if not keep_nl \
                else line[:a] + " " * (b - a) + "\n"
    return "".join(lines)


_dcode = code_only(_dpath)
check("the stripper removed the docstring that quotes the idiom",
      ".data[:] =" in _dsrc and "memoryview assignment" not in _dcode,
      "otherwise this lint fails for the wrong reason")
# And the lint must be able to FAIL, which the joined-token version could not.
# Asserted directly rather than trusted: a canary with the bad idiom inside a
# docstring AND in code must be caught by the code half only.
_canary = os.path.join(WDIR, "canary.py")
open(_canary, "w").write(
    'x = 1\n'
    'def f():\n'
    '    """doc mentioning .data[:] = buf, which must be ignored"""\n'
    '    return x\n')
check("the stripper ignores the idiom in a DOCSTRING",
      ".data[:] =" not in code_only(_canary))
open(_canary, "w").write('def f(v, buf):\n    v.numpy().data[:] = buf\n')
check("*** ...and CATCHES it in code, so the lint can actually fail ***",
      ".data[:] =" in code_only(_canary),
      "a lint that cannot fail reads as coverage")
check("*** no bare `.numpy().data[:] =` survives in the CODE ***",
      ".data[:] =" not in _dcode,
      "that idiom is the live ValueError")
check("...and _write_row goes through copy_into",
      "copy_into(self._staging[" in _dsrc)

print("\n== (b12d) the disk index's counters must be REACHABLE from /kv ==")

# *** THE FIRST WORKING RESTORE HIT 2,048 OF 11,764 TOKENS AND NOTHING ON THE
# BOX COULD SAY WHY. *** `DiskPresenceManager.snapshot()` existed, with the
# `stored` / `lookup` / `hit` counters that are the whole diagnosis, and no
# endpoint served it -- /kv's disk section came back `{}`. An instrument that
# exists and is unreachable is the same defect as a mechanism that runs and
# reports nothing, which is this project's signature failure.
from kvmgr import stats as _st  # noqa: E402

_saved_live = diskspec._LIVE_MANAGER
try:
    diskspec._LIVE_MANAGER = None
    eq("*** no disk spec in force reports None, NOT {} ***",
       diskspec.live_manager_snapshot(), None)
    eq("...and the endpoint carries that None through",
       _st._disk_snapshot(), None)

    mdir3 = tempfile.mkdtemp(prefix="kvlive-")

    class FM3:
        @staticmethod
        def get_file_name(k):
            return os.path.join(mdir3,
                                base.get_offload_block_hash(k).hex()[:8] + ".bin")

    live = diskspec.DiskPresenceManager(
        FM3, num_workers=2, manifest_path=os.path.join(mdir3, "presence.json"))
    diskspec._LIVE_MANAGER = live
    snap = diskspec.live_manager_snapshot()
    check("a live manager reports a dict", isinstance(snap, dict))
    for field in ("lookup", "hit", "stored", "present", "restore_enabled"):
        check(f"...carrying `{field}`, which is what answers the shortfall",
              field in snap, sorted(snap))
    eq("a fresh index holds nothing, and says so as a ZERO not an absence",
       snap["present"], 0)

    # The counters must MOVE, or the field is decoration.
    kk = []
    for i in range(3):
        k = K(bytes([91, 0, 0, i]) * 8, 0)
        open(FM3.get_file_name(k), "wb").write(b"x")
        kk.append(k)
    live.complete_store(kk, base.ReqContext(req_id="rq"))
    eq("...`present` follows a store",
       diskspec.live_manager_snapshot()["present"], 3)
    live.lookup(kk[0], base.ReqContext(req_id="rq"))
    s2 = diskspec.live_manager_snapshot()
    check("...and `lookup` counts, so a miss is distinguishable from no call",
          s2["lookup"] >= 1)

    # And it must never be the thing that breaks the endpoint.
    class _Boom:
        def snapshot(self):
            raise RuntimeError("boom")

    diskspec._LIVE_MANAGER = _Boom()
    eq("*** a raising snapshot degrades, it does not take /kv down ***",
       diskspec.live_manager_snapshot(), {"error": "snapshot raised"})
    # And the stash itself: get_manager needs a whole spec to construct, so the
    # wiring is asserted on the SOURCE. Without this line the counters above are
    # all reachable in a test and permanently None on the box -- a guard that
    # passes offline and fails live is the shape the fixture lesson is about.
    _sp = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                            "diskspec.py")).read()
    check("*** get_manager PUBLISHES the instance, or /kv stays None on the box ***",
          "_LIVE_MANAGER = self._manager" in _sp
          and "global _LIVE_MANAGER" in _sp,
          "the global must be declared AND assigned")
finally:
    diskspec._LIVE_MANAGER = _saved_live

print("\n== (b13) the manifest must flush on TIME too, or a short session is lost ==")

# *** THE LIVE SHAPE, 2026-10-08. *** A run stored 1,052 new .bin files -- about
# 210 keys -- and presence.json kept the PREVIOUS day's mtime. 210 is under
# MANIFEST_FLUSH_KEYS (512), and the only other force=True was reset_cache,
# which CLEARS the index first. So the keys existed in memory only, the files
# sat on disk, and a restart could not name them: restore-after-restart broken
# by construction, with nothing saying so. Three flush triggers now -- enough
# keys, enough time, or shutdown.
mdir2 = tempfile.mkdtemp(prefix="kvman2-")
mpath2 = os.path.join(mdir2, "presence.json")


class FakeMapper2:
    @staticmethod
    def get_file_name(k):
        return os.path.join(mdir2, base.get_offload_block_hash(k).hex()[:8] + ".bin")


def mkkeys(n, g=0, salt=77):
    out = []
    for i in range(n):
        k = K(bytes([salt, g, i // 256, i % 256]) * 8, g)
        open(FakeMapper2.get_file_name(k), "wb").write(b"x")
        out.append(k)
    return out


ctx3 = base.ReqContext(req_id="rt")
mt = diskspec.DiskPresenceManager(FakeMapper2, num_workers=2,
                                  manifest_path=mpath2)
mt.complete_store(mkkeys(3), ctx3)          # far below 512
check("*** a handful of keys does NOT flush immediately -- that is the design,"
      " not the bug ***", not os.path.exists(mpath2),
      "the bug was that NOTHING ELSE ever did")

# Now make it stale rather than numerous. The floor is monotonic, so the test
# moves the recorded time rather than sleeping 30 seconds.
mt._last_flush -= diskspec.DiskPresenceManager.MANIFEST_FLUSH_S + 1
mt.flush_manifest()
# *** THE WRITE IS ASYNCHRONOUS NOW, so "I triggered it" and "it is on disk"
# are different statements and the test has to say which it means. *** The
# flush moved off the engine's thread because serialising the live 771,775-key
# index costs 0.779s measured (hex+sorted 0.227s, json.dump 0.552s) and that was
# the ~1.1s mid-stream stall test-meter.sh section 1 failed on. `flush_wait`
# exists so this is a deadline rather than a sleep-and-hope.
check("the manifest write is joinable, not a race the test has to sleep on",
      mt.flush_wait(30.0) is True)
check("*** ...but ENOUGH TIME does, with the same three keys ***",
      os.path.exists(mpath2))
eq("...writing all of them", len(json.loads(open(mpath2).read())["keys"]), 3)
eq("...and counted as a write", mt._stats["manifest_writes"], 1)

# A stale clock with NOTHING new must not rewrite the file: a periodic write
# that says nothing is pure cost, and this is a real filesystem.
before_w = mt._stats["manifest_writes"]
mt._last_flush -= diskspec.DiskPresenceManager.MANIFEST_FLUSH_S + 1
mt.flush_manifest()
eq("a stale clock with no new keys writes NOTHING",
   mt._stats["manifest_writes"], before_w)

# The count trigger must still work on its own, with a fresh clock.
mt2 = diskspec.DiskPresenceManager(FakeMapper2, num_workers=2,
                                   manifest_path=mpath2)
w0 = mt2._stats["manifest_writes"]
mt2.complete_store(mkkeys(diskspec.DiskPresenceManager.MANIFEST_FLUSH_KEYS + 1,
                          g=1, salt=78), ctx3)
# the counter is incremented by the WRITER, which now runs on its own thread,
# so join before reading it -- see flush_wait
mt2.flush_wait(30.0)
check("*** enough KEYS still flushes on its own, clock untouched ***",
      mt2._stats["manifest_writes"] > w0)

# ---------------------------------------------------------------------------
# *** THE FLUSH MUST NOT RUN ON THE CALLER'S THREAD, and that is the whole
# *** point of the change -- so it is asserted directly. *** Serialising the
# live 771,775-key index costs 0.779s measured on spark-45ea (hex+sorted
# 0.227s, json.dump 0.552s), which is the ~1.1s mid-stream stall
# scripts/test-meter.sh section 1 failed on (max gap 1.137s vs a 0.102s
# baseline). Every other manifest assertion here would stay green if somebody
# moved the write back inline, because they all check the FILE and an inline
# write produces the same file -- just a second later, on the engine's thread.
import time as _t_async
mt_a = diskspec.DiskPresenceManager(FakeMapper2, num_workers=2,
                                    manifest_path=mpath2 + ".async")
mt_a.complete_store(mkkeys(3, g=1, salt=91), ctx3)
_real_write = mt_a._write_manifest
_slow = {"n": 0}


def _slow_write(snap):
    # takes the snapshot the real writer now takes -- a no-arg stand-in raised
    # TypeError inside the thread, where nothing was watching, and the only
    # symptom was "writer started 0 time(s)"
    _slow["n"] += 1
    _t_async.sleep(1.5)
    _real_write(snap)


mt_a._write_manifest = _slow_write
mt_a._last_flush -= diskspec.DiskPresenceManager.MANIFEST_FLUSH_S + 1
_t0 = _t_async.monotonic()
mt_a.flush_manifest()
_took = _t_async.monotonic() - _t0
check("*** flush_manifest RETURNS while the write is still going ***",
      _took < 0.5,
      "took %.3fs against a 1.5s write -- if this is ~1.5s the flush is back "
      "on the caller's thread and every stream pays for it" % _took)
# ...and a second trigger while it is in flight is SKIPPED, not queued
mt_a._last_flush -= diskspec.DiskPresenceManager.MANIFEST_FLUSH_S + 1
mt_a._unflushed = 1
mt_a.flush_manifest()
eq("...and a trigger while one is in flight is skipped, and COUNTED",
   mt_a._stats["manifest_skipped_busy"], 1)
check("...and the skipped one did NOT start a second writer",
      _slow["n"] == 1, "writer started %d time(s)" % _slow["n"])
check("...and it lands once joined", mt_a.flush_wait(30.0) is True)
check("...leaving the file on disk", os.path.exists(mpath2 + ".async"))
eq("...counted as exactly one write", mt_a._stats["manifest_writes"], 1)

# *** AND IT MUST STILL WRITE WHILE THE INDEX IS MUTATING, which the first
# *** version did not -- it wrote NOTHING. *** That version iterated the live
# `_present` on the writer thread and retried on RuntimeError, reasoning that a
# missed key is picked up next time. Measured on the box with a live store:
#     12:38:41 kvmgr: manifest flush gave up after 3 attempts against a
#              mutating index; the next trigger will try again
#     12:39:53 (again)
# A live store mutates continuously, so the iteration never completed and the
# manifest was never written -- strictly worse than the 0.78s stall it
# replaced, because that cost a second of stream and this costs
# restore-after-restart. The fix is a `list()` snapshot on the caller's thread,
# atomic under the GIL, so there is no race to survive.
mt_m = diskspec.DiskPresenceManager(FakeMapper2, num_workers=2,
                                    manifest_path=mpath2 + ".mutating")
mt_m.complete_store(mkkeys(4, g=1, salt=123), ctx3)
_mreal = mt_m._write_manifest
_mseen = {}


def _mutating_write(snap):
    # mutate the index WHILE the writer runs -- the shape that broke it
    _mseen["len_at_write"] = len(snap)
    for _k in mkkeys(6, g=1, salt=250):
        mt_m._present[bytes(_k)] = (1, 0)
    _mreal(snap)


mt_m._write_manifest = _mutating_write
mt_m._last_flush -= diskspec.DiskPresenceManager.MANIFEST_FLUSH_S + 1
mt_m._unflushed = 1
mt_m.flush_manifest()
mt_m.flush_wait(30.0)
check("*** the manifest IS written while the index mutates under it ***",
      os.path.exists(mpath2 + ".mutating"),
      "the retry-on-RuntimeError version wrote nothing at all here")
eq("...and it is counted as a write", mt_m._stats["manifest_writes"], 1)
_got = json.loads(open(mpath2 + ".mutating").read())["keys"]
eq("...holding exactly the SNAPSHOT's keys, not a torn half-mutated view",
   len(_got), _mseen.get("len_at_write"))
check("...and the keys added mid-write are simply in the NEXT flush, not lost",
      len(mt_m._present) > len(_got))

# And shutdown, which is what makes the loss window ZERO on a clean restart.
mt3 = diskspec.DiskPresenceManager(FakeMapper2, num_workers=2,
                                   manifest_path=mpath2)
n_before = len(json.loads(open(mpath2).read())["keys"])
mt3.complete_store(mkkeys(2, g=2, salt=79), ctx3)
eq("...two more keys, still unflushed", mt3._unflushed, 2)
mt3.shutdown()
n_after = len(json.loads(open(mpath2).read())["keys"])
check("*** shutdown() flushes, so a CLEAN restart keeps what it stored ***",
      n_after == n_before + 2, f"{n_before} -> {n_after}")

# reset_cache still clears and persists the clearance -- it must not be
# confused with the new triggers, which only ever ADD.
mt3.complete_store(mkkeys(1, g=3, salt=80), ctx3)
mt3.reset_cache()
eq("reset_cache still clears the manifest",
   len(json.loads(open(mpath2).read())["keys"]), 0)

print("\n== (b13) wait(): ONLY our own jobs, or the engine blocks for minutes ==")

# *** THE SHAPE THAT WEDGED THE WORKER LIVE, 2026-10-07. *** The worker stopped
# consuming shm broadcasts for seven minutes with zero files written, because
# `OffloadingWorker.wait` fans the whole id set out to EVERY registered handler
# (worker.py:172) and our wait() did `remaining = set(job_ids)`. An id belonging
# to the load handler, or one whose completion we already reported and popped,
# can never appear in a future get_finished() -- so the loop spun its full
# deadline with the engine stalled behind it. Upstream skips an unknown id in
# one line (`cpu/gpu_worker.py:373`): `if event is not None`.
import re as _re
import time as _time

hw = mkhandler(8)
t0 = _time.monotonic()
hw.wait({4242})                      # never ours: the load handler's
el = _time.monotonic() - t0
check("*** a FOREIGN job id returns immediately, it does not spin ***",
      el < 1.0, f"{el:.3f}s")

# The second half of the live shape: an id we DID own and have already
# reported. get_finished() pops it, so it can never be seen again either.
hr = mkhandler(64)
krep = gkeys(1, 4, salt=9)
gpur = base.GPULoadStoreSpec(list(range(2000, 2004)),
                             group_sizes=[0, 4, 0, 0, 0], block_indices=[0] * 5)
check("a small job submits in one round",
      hr.transfer_async(500, (gpur, diskspec.DiskLoadStoreSpec(krep))) is True)
seen = []
for _ in range(50):
    seen.extend(hr.get_finished())
    if seen:
        break
eq("...and is reported finished once", [r.job_id for r in seen], [500])
check("...after which it is no longer in _jobs", 500 not in hr._jobs)
t0 = _time.monotonic()
hr.wait({500})                       # already done and popped
el = _time.monotonic() - t0
check("*** an ALREADY-REPORTED id returns immediately too ***",
      el < 1.0, f"{el:.3f}s")

# A mixed set must still wait for the one that IS ours and pending.
hm = mkhandler(8)
km = gkeys(0, 40, salt=31)
gpum = base.GPULoadStoreSpec(list(range(3000, 3040)),
                             group_sizes=[40, 0, 0, 0, 0], block_indices=[0] * 5)
hm.transfer_async(900, (gpum, diskspec.DiskLoadStoreSpec(km)))
check("a 40-block job against an 8-block pool is still queued",
      hm._jobs[900]["batches"] != [])
t0 = _time.monotonic()
hm.wait({900, 4242, 777})            # one ours-and-pending, two foreign
el = _time.monotonic() - t0
check("*** a MIXED set still waits for the one that is ours ***",
      900 not in hm._jobs, "it completed rather than being skipped")
check("...and it finished by pumping, not by timing out", el < 30.0, f"{el:.3f}s")
eq("...writing every block", hm.stats["files"], 40)

# An empty intersection must not even start the clock.
he = mkhandler(8)
t0 = _time.monotonic()
he.wait(set())
check("an empty set is a no-op", (_time.monotonic() - t0) < 0.5)

# The LOAD handler has the identical bug and the identical fix. Asserted on the
# SOURCE because constructing one needs torch; what must never drift is that
# BOTH filter to their own `_jobs` before looping.
_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                         "diskio.py")).read()
_waits = _re.findall(r"def wait\(self, job_ids: set\) -> None:(.*?)(?=\n    def )",
                     _src, _re.S)
eq("both handlers define wait()", len(_waits), 2)
for _i, _body in enumerate(_waits):
    check(f"wait() #{_i + 1} filters to ids it actually holds",
          _re.search(r"\{\s*j\s+for\s+j\s+in\s+job_ids\s+if\s+j\s+in\s+self\._jobs\s*\}",
                     _body) is not None,
          "a bare set(job_ids) is the hang")
    check(f"wait() #{_i + 1} returns early on an empty intersection",
          _re.search(r"if not (remaining|rem):\s*\n\s*return", _body) is not None)
    check(f"wait() #{_i + 1} is bounded well under the engine's patience",
          "deadline = time.monotonic() + 120.0" in _body)
    check(f"wait() #{_i + 1} says WHICH jobs timed out",
          "sorted(" in _body and "timed out" in _body)

print("\n== (b14) the fence counters, and the fixture that must keep up ==")

# The fixture hand-builds `stats`, so a counter the real constructor gains
# breaks the suite with a KeyError. Loud, but an assertion naming the missing
# key is better than a traceback -- so the two are compared directly.
import re as _re14

_src14 = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                           "diskio.py")).read()
_i14 = _src14.index("self.stats = {")
_j14 = _src14.index("}", _src14.index('"saturations"', _i14))
_real_keys = set(_re14.findall(r'"(\w+)":', _src14[_i14:_j14]))
_fix_keys = set(mkhandler(4).stats)
eq("*** the fixture carries every counter the real constructor sets ***",
   sorted(_real_keys - _fix_keys), [])

# and the fence counters must actually MOVE, or they measure nothing.
# The per-round budget is lifted here for the same reason as the (b2) block:
# with store_inflight_max=1 the pump returns before it can drain the pool, so
# `saturations` -- the signal for "disk writes are falling behind" -- cannot
# fire at all. This block is about the POOL being the limiter; the budget has
# its own block above.
F.write(store_inflight_max=64, store_batch_max=4096, store_ops_max=0)
flags.reset_for_tests()
hf = mkhandler(8)
kf = gkeys(0, 40, salt=66)
gf = base.GPULoadStoreSpec(list(range(7000, 7040)),
                           group_sizes=[40, 0, 0, 0, 0], block_indices=[0] * 5)
hf.transfer_async(4242, (gf, diskspec.DiskLoadStoreSpec(kf)))
check("a 40-block job against an 8-block pool is queued in rounds",
      hf._jobs[4242]["batches"] != [])
eq("...and the saturation is counted", hf.stats["saturations"], 1)
hf.wait({4242})
eq("*** wait() counts the fence firing ***", hf.stats["wait_calls"], 1)
eq("...and how many of OUR jobs it blocked on", hf.stats["wait_ours"], 1)
F.write(store_inflight_max=1, store_batch_max=32, store_ops_max=2048)
flags.reset_for_tests()
check("...and how long it blocked, which is the 2x question",
      hf.stats["wait_blocked_ms"] >= 0, hf.stats["wait_blocked_ms"])
check("...and how many pump rounds it took", hf.stats["wait_rounds"] >= 1,
      hf.stats["wait_rounds"])
check("...and the job really completed", 4242 not in hf._jobs)

hf2 = mkhandler(8)
hf2.wait({9999})
eq("a foreign id counts the CALL but blocks on nothing",
   (hf2.stats["wait_calls"], hf2.stats["wait_ours"]), (1, 0))

print("\n== (b15) CONCURRENT writes of the SAME key -- the crash, reproduced ==")

# *** THIS IS THE BUG THAT TOOK THE ENGINE DOWN THREE TIMES. *** write_chunks
# used a FIXED `<path>.part`, four writer threads run, and the same block hash
# can be in two IN-FLIGHT jobs at once -- `_write_batch`'s exists() check only
# skips files already COMPLETED. The live traceback:
#     FileNotFoundError: [Errno 2] ... in write_chunks: os.replace(tmp, path)
# raised in a writer thread and re-raised by fut.result() on the worker's
# critical path. The other outcome is worse: two writers interleaving into one
# temp file produce a file of exactly the right LENGTH holding bytes from two
# blocks, which passes read_chunks' size check and restores as wrong KV.
import threading as _th14

WDIR15 = tempfile.mkdtemp(prefix="kvrace-")
# one group, one 4096-byte page -- the shape REFS already uses
REFS15 = [[Ref(0, 4096)]]


def _writer(path, byte, out, barrier):
    def read_row(t, s_, n):
        return bytes([byte]) * n
    try:
        barrier.wait(timeout=5)
        out.append(("ok", diskio.write_chunks(
            path, diskio.file_plan(REFS15, 0, 0), read_row)))
    except Exception as e:
        out.append(("err", f"{type(e).__name__}: {e}"))


def race(n=6):
    """n threads writing the SAME path at once, each with its own byte."""
    path = os.path.join(WDIR15, f"race{next(_ctr)}.bin")
    out, barrier = [], _th14.Barrier(n)
    ts = [_th14.Thread(target=_writer, args=(path, 0xA0 + i, out, barrier))
          for i in range(n)]
    for t in ts:
        t.start()
    for t in ts:
        t.join(timeout=10)
    return path, out


import itertools as _it14
_ctr = _it14.count()

path15, out15 = race(6)
_errs = [o for o in out15 if o[0] == "err"]
eq("*** six concurrent writers of one key: NO exceptions ***", _errs, [])
eq("...all six reported a full write", len([o for o in out15 if o[0] == "ok"]), 6)
check("the file exists and is exactly one page", os.path.getsize(path15) == 4096,
      os.path.getsize(path15))

# *** THE CORRECTNESS HALF: the surviving file must be ONE writer's bytes,
# never a blend. A blend is the same length and so passes every size check.
_data = open(path15, "rb").read()
eq("*** and it is ONE writer's bytes, not a blend of several ***",
   len(set(_data)), 1,)
check("...and that byte is one a writer actually used",
      _data[0] in range(0xA0, 0xA6), hex(_data[0]))

# no temp files left behind, or they accumulate forever
_leftover = [f for f in os.listdir(WDIR15) if ".part" in f]
eq("no .part files survive a clean race", _leftover, [])

# And the lint: a FIXED temp suffix is the bug, so assert the name varies.
_p1 = os.path.join(WDIR15, "u1.bin")
_seen = set()
_orig_replace = os.replace


def _spy(src, dst):
    _seen.add(os.path.basename(src))
    return _orig_replace(src, dst)


os.replace = _spy
try:
    for _ in range(4):
        diskio.write_chunks(_p1, diskio.file_plan(REFS15, 0, 0),
                            lambda t, s_, n: b"\x11" * n)
finally:
    os.replace = _orig_replace
check("*** the temp name is UNIQUE per write, which is the whole fix ***",
      len(_seen) == 4, sorted(_seen))
check("...and carries pid and thread, so two PROCESSES cannot collide either",
      all(str(os.getpid()) in n for n in _seen), sorted(_seen)[:1])

print()
# -- (b16) the stall watchdog's verdict ------------------------------------
#
# This is the instrument that was MISSING when a worker died on 2026-10-08: the
# engine complained for ten minutes about a worker that was alive and had a
# walkable stack, and nobody was there to send SIGUSR1. The verdict is a pure
# function so the part that decides "stalled" is testable with no threads, no
# torch and no GPU -- the same split as file_plan and check_pairing.
#
# The property that matters most is the one in the middle: a LEGITIMATELY SLOW
# flush must never be called stalled. That is why the verdict keys on a
# progress fingerprint and not on elapsed time; a 700K flush takes rounds of a
# 32-block pool and would trip any pure timer.
wv = diskio.watchdog_verdict

eq("nothing outstanding is idle, whatever the clock says",
   wv(0, ("a",), ("a",), 1000.0, 9999.0)[0], "idle")
eq("...and idle CLEARS any stall in progress",
   wv(0, ("a",), ("a",), 1000.0, 9999.0)[1], None)
eq("a moving counter is progress, however long the job has run",
   wv(5, ("b",), ("a",), None, 500.0)[0], "progress")
eq("...and progress clears the stall clock too",
   wv(5, ("b",), ("a",), 100.0, 500.0)[1], None)
eq("outstanding and unmoved STARTS the clock rather than crying stall",
   wv(5, ("a",), ("a",), None, 500.0)[0], "waiting")
eq("...stamping the clock at now",
   wv(5, ("a",), ("a",), None, 500.0)[1], 500.0)
eq("still waiting one second short of the deadline",
   wv(5, ("a",), ("a",), 500.0, 500.0 + diskio._WATCHDOG_STALL_S - 1.0)[0],
   "waiting")
eq("STALLED exactly at the deadline",
   wv(5, ("a",), ("a",), 500.0, 500.0 + diskio._WATCHDOG_STALL_S)[0], "stalled")
eq("...and it reports the stall's START, not now, so the log can say how long",
   wv(5, ("a",), ("a",), 500.0, 9999.0)[1], 500.0)

# *** THE FALSE-POSITIVE GUARD, STATED AS A SCENARIO. *** A slow flush that
# advances by one file every look must stay out of "stalled" forever, even past
# many multiples of the deadline. If this ever fails, the watchdog will dump
# stacks in the middle of the exact workload the disk tier exists for.
_since, _t, _verdicts = None, 0.0, set()
for _i in range(50):
    _t += 5.0
    _v, _since = wv(9, ("files", _i), ("files", _i - 1), _since, _t)
    _verdicts.add(_v)
eq("a slow-but-advancing flush is NEVER stalled, even over 250s",
   sorted(_verdicts), ["progress"])

# And the mirror: genuinely frozen counters must reach stalled and stay there.
_since, _t, _hit = None, 0.0, False
for _i in range(50):
    _t += 5.0
    _v, _since = wv(9, ("frozen",), ("frozen",), _since, _t)
    _hit = _hit or _v == "stalled"
check("...while frozen counters DO reach stalled", _hit)

# *** AND THAT ASSERTION WAS VACUOUS WHEN FIRST WRITTEN. *** It read
# `is False or is True`, which is true of every bool that will ever exist. The
# property worth asserting is IDEMPOTENCE: _start_watchdog must create exactly
# one thread however many handlers are constructed, because a second watchdog
# means two stack dumps for one stall and a thread leaked per reload. Tested by
# counting Thread() constructions rather than by starting real threads, so the
# harness stays free of a daemon that outlives it.
import threading as _th  # noqa: E402

_made = []
_real_thread = _th.Thread


class _CountingThread:
    def __init__(self, *a, **kw):
        _made.append(kw.get("name"))

    def start(self):
        pass


_saved_flag = diskio._watchdog_started
try:
    _th.Thread = _CountingThread
    diskio._watchdog_started = False
    diskio._start_watchdog()
    diskio._start_watchdog()
    diskio._start_watchdog()
finally:
    _th.Thread = _real_thread
    diskio._watchdog_started = _saved_flag

eq("three _start_watchdog calls arm exactly ONE watchdog", len(_made), 1)
eq("...and it is named, so it is identifiable in a stack dump",
   _made, ["kvmgr-watchdog"])

# -- (b17) the worker's UNSHIPPED completion queue -------------------------
#
# The scheduler holds 123 of 133 jobs open with pending_count=2 on 116 -- so
# neither rank's completion arrived -- while our handlers report jobs_open=0.
# Upstream marks each completion on the worker and `build_connector_worker_meta`
# ships them AND RESETS the queue; if that stops being called, the queue climbs
# forever and the scheduler never learns. This exposes its size, which is the
# measurement that confirms or kills that hypothesis.
del diskio._LIVE_CONN_WORKER[:]
_s = diskio.conn_worker_snapshot()
eq("*** uncaptured is an ABSENCE, not an empty queue ***",
   _s.get("available"), False)
check("...and unshipped_completions is absent, not 0",
      "unshipped_completions" not in _s)


class _Meta:
    def __init__(self, n):
        self.completed_jobs = {i: 1 for i in range(n)}


class OffloadingConnectorWorker:          # the name is what is matched
    def __init__(self, n):
        self._connector_worker_meta = _Meta(n)
        self._unsubmitted_store_jobs = [1, 2, 3]
        self._load_jobs = {7: "r1"}

    def call_handler(self, fn):
        return fn()


_cw = OffloadingConnectorWorker(41)
diskio._LIVE_CONN_WORKER.append(_cw)
_s = diskio.conn_worker_snapshot()
eq("the unshipped queue is reported", _s.get("unshipped_completions"), 41)
eq("so are the deferred store jobs", _s.get("unsubmitted_store_jobs"), 3)
eq("and open load jobs", _s.get("load_jobs_open"), 1)

# capture off the call stack, the same route as the scheduler handle
del diskio._LIVE_CONN_WORKER[:]
_cw2 = OffloadingConnectorWorker(5)
_cw2.call_handler(lambda: diskio.capture_conn_worker_from_caller())
eq("*** captured from the call stack ***",
   diskio._LIVE_CONN_WORKER and diskio._LIVE_CONN_WORKER[0] is _cw2, True)
_cw3 = OffloadingConnectorWorker(9)
_cw3.call_handler(lambda: diskio.capture_conn_worker_from_caller())
eq("...and the handle is STABLE, not replaced by a later caller",
   diskio._LIVE_CONN_WORKER[0] is _cw2, True)


class _Hostile:
    @property
    def _connector_worker_meta(self):
        raise RuntimeError("upstream moved it")


del diskio._LIVE_CONN_WORKER[:]
diskio._LIVE_CONN_WORKER.append(_Hostile())
_s = diskio.conn_worker_snapshot()
eq("an upstream change is an absence, never an exception",
   _s.get("available"), False)
check("...and still never a zero", "unshipped_completions" not in _s)
del diskio._LIVE_CONN_WORKER[:]

# -- (b18) A COMPLETION OBSERVED BY wait() MUST STILL BE REPORTED ----------
#
# *** THE ROOT CAUSE OF THE WHOLE PHASE, AND IT WAS MINE. *** wait() looped
# `for r in self.get_finished(): remaining.discard(r.job_id)` -- and
# get_finished POPS the job, so a result can never be produced twice. Upstream's
# OffloadingConnectorWorker.get_finished() is the ONLY caller of
# mark_completed(), so every completion the fence happened to observe was EATEN.
# Measured live: 5 store jobs, 1 report delivered; 7% of a 41K prompt's keys
# indexed against ~90% of a 3K one, because the fence waits far more under load.
# Downstream of that one line: the +13,312-tokens-per-pass convergence,
# jobs_open 0 on the worker against 125 open on the scheduler, 9,300 blocks
# fenced forever, a request deferred until NCCL's watchdog killed the worker.
#
# The property is simple to state and was impossible to see: wait() may OBSERVE
# completions, and only get_finished() may CONSUME them.


class _FakeHandler:
    """The smallest thing with the real collect/report/wait split."""

    def __init__(self):
        self._jobs = {1: {}, 2: {}}
        self.stats = {"wait_calls": 0, "wait_ours": 0, "wait_rounds": 0,
                      "wait_blocked_ms": 0, "wait_max_pending": 0}

    # the same accessor shape as the real handlers
    _ready_buf = diskio.DiskStoreHandler._ready_buf

    def _collect(self):
        # one job finishes per call
        for jid in list(self._jobs):
            del self._jobs[jid]
            self._ready_buf().append(
                type("R", (), {"job_id": jid, "success": True})())
            break

    get_finished = diskio.DiskStoreHandler.get_finished
    wait = diskio.DiskStoreHandler.wait


_h = _FakeHandler()
_h.wait({1, 2})
eq("wait() observes both jobs and returns", len(_h._jobs), 0)
_reported = [r.job_id for r in _h.get_finished()]
eq("*** and BOTH completions are still reported afterwards ***",
   sorted(_reported), [1, 2])
eq("...and a second report is empty, so nothing is double-counted",
   _h.get_finished(), [])

# the structural guard: get_finished must be the ONLY thing that clears the
# buffer, because a drain anywhere else is unreportable and invisible.
_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                         "diskio.py")).read()
_drains = [ln.strip() for ln in _src.splitlines()
           if "self._ready = []" in ln and not ln.strip().startswith("#")]
eq("exactly one drain site per handler, and no more", len(_drains), 2)
_calls_gf_in_wait = [ln.strip() for ln in
                     _src[_src.index("def wait("):].splitlines()[:40]
                     if "self.get_finished()" in ln]
eq("*** wait() must never call get_finished ***", _calls_gf_in_wait, [])

# -- (b19) the staging pool: a HOT size and a floor per direction ----------
#
# Two things the 574.9s-against-3.3s result asked for. The store and load
# handlers SHARE one pool, and the pump advances once per SCHEDULER STEP, so
# fewer slots per round means fewer blocks per step: the cold pass had the whole
# pool for writes and ran at 177 MB/s, the restore pass shared it and managed
# 17.7 MB/s on the same disk.
#
# The ALLOCATION cannot be hot -- the pinned staging is 63 tensors held by both
# handlers and two inner handlers -- so it is a boot ceiling and the LIVE SIZE
# moves within it. A slot is an index into rows that already exist, so raising
# and lowering it allocates nothing.
sp = diskio.SlotPool(16)
eq("capacity is the allocation", sp.capacity, 16)
eq("...and the live size starts at it", sp.size, 16)
eq("shrinking is instant", sp.set_limit(4), 4)
eq("...and only the live size moves, never the allocation",
   (sp.size, sp.capacity), (4, 16))
eq("*** a shrunk pool hands out only slots BELOW the limit ***",
   all(x < 4 for x in sp.take_upto(99)), True)
eq("...and exactly that many", len(sp._free), 12)
eq("available reports ELIGIBLE free slots, not every free row",
   sp.snapshot()["available"], 0)
eq("...while rows_free shows the allocation that is idle",
   sp.snapshot()["rows_free"], 12)
check("*** reporting the ineligible rows as available would read as headroom "
      "that is not there ***",
      sp.snapshot()["available"] != sp.snapshot()["rows_free"])
eq("growing back within the allocation is instant too", sp.set_limit(16), 16)
eq("...and is clamped to the allocation, never above", sp.set_limit(999), 16)
eq("...and never to zero, which would wedge both directions", sp.set_limit(0), 1)

# shrinking must not claw back slots that are IN USE
sp2 = diskio.SlotPool(8)
held = sp2.take_upto(8)
eq("all eight taken", len(held), 8)
sp2.set_limit(2)
sp2.give(held)
eq("*** a shrink drains rather than snatching: returned slots above the new "
   "limit are simply not re-offered ***", len(sp2.take_upto(99)), 2)

# the floors
sp3 = diskio.SlotPool(10)
eq("a reserve leaves that many free", len(sp3.take_upto(99, reserve=3)), 7)
eq("...and the other direction can then take them",
   len(sp3.take_upto(99, reserve=0)), 3)
sp4 = diskio.SlotPool(10)
eq("all-or-nothing honours the reserve too", sp4.take(8, reserve=3), None)
eq("...while a fitting request succeeds", len(sp4.take(7, reserve=3) or []), 7)

# *** THE DEADLOCK THE TWO FLOORS COULD BUILD. *** If each direction reserved
# half, both would see zero available and both would stall for ever -- a hang
# assembled out of two anti-starvation measures. Each floor is clamped so the
# other side can always take one.
for _cap in (1, 2, 3, 4, 8, 32, 256):
    _sz, _wf, _rf = diskio.staging_plan(_cap, 0, 0, 0)
    check(f"cap {_cap}: a write can always take at least one slot",
          _sz - _rf >= 1)
    check(f"cap {_cap}: a read can always take at least one slot",
          _sz - _wf >= 1)
_sz, _wf, _rf = diskio.staging_plan(32, 0, 99, 99)
check("*** absurd floors cannot starve either direction ***",
      _sz - _wf >= 1 and _sz - _rf >= 1)

eq("0 means the whole allocation, not a silent constant",
   diskio.staging_plan(256, 0, 0, 0)[0], 256)
eq("...and the floors scale with the size rather than staying at 8",
   diskio.staging_plan(256, 0, 0, 0)[1:], (64, 64))
eq("an explicit size overrides the allocation",
   diskio.staging_plan(256, 64, 0, 0)[0], 64)
eq("...and the derived floors follow the LIVE size, not the allocation",
   diskio.staging_plan(256, 64, 0, 0)[1:], (16, 16))
eq("explicit floors are honoured", diskio.staging_plan(32, 0, 5, 7)[1:], (5, 7))

# and the knob reader must never stop a transfer
eq("an unreadable knob falls back to the documented default, it does not raise",
   diskio.staging_knobs(64)[0], 64)

# *** THE RESIZE MUST BE DRIVEN BY THE PER-STEP EDGE, NOT THE TAKE PATH. ***
# It was in the take path first, so a resize only happened when a transfer round
# actually ran -- and setting the knob on an engine with nothing to store
# changed nothing. Measured live: 256 -> 32 -> 128 applied, while a third change
# to 256 did not, because by then every key was present and no round ran. That
# is the documented "a hot knob read on an IDLE engine shows the OLD derived
# figure" in a new costume, and the structural guard is that `_collect` (called
# once per step via get_finished) is what reconciles it.
_dsrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskio.py")).read()
_collect_body = _dsrc[_dsrc.index("def _collect(self) -> None:"):]
_collect_body = _collect_body[:_collect_body.index("\n    def ")]
check("*** _collect reconciles the pool, so the knob lands on the next STEP ***",
      "self.reconcile_pool()" in _collect_body)
# Re-aimed after the staging SPLIT removed `reserve=_rf)` from the take path:
# each pool now has one user, so a cross-direction reserve would shrink that
# user for nothing. The property under test is unchanged -- the take path must
# not resize -- so it anchors on the take call itself rather than on the
# vanished reserve argument. An index() on absent text raises, which is how
# this was caught rather than silently passing.
# Anchored on the CALL, not on one spelling of its argument: the load's take is
# now `take_upto(min(len(bids), _cap))` because ~256 blocks in one batch faults
# the worker, and a lint pinned to the old literal raised ValueError rather than
# failing with a name -- which is the abort-instead-of-fail trap again, this time
# in a lint I wrote.
_take = "self._pool.take_upto("
_ti = [i for i in range(len(_dsrc)) if _dsrc.startswith(_take, i)]
eq("both take sites are present to anchor on", len(_ti), 2)
for _i in _ti:
    check("...and the take path no longer resizes (the idle-engine bug)",
          "set_limit" not in _dsrc[max(0, _i - 400):_i])

# and reconcile_pool must never raise into the step loop
class _P:
    capacity = 8
    size = 8
    def set_limit(self, k):
        raise RuntimeError("boom")


class _H:
    _pool = _P()
    reconcile_pool = diskio.DiskStoreHandler.reconcile_pool


_ok = True
try:
    flags.set_override("staging_slots", 4)
    _H().reconcile_pool()
except Exception:
    _ok = False
finally:
    flags.clear_override("staging_slots")
check("*** a failing resize cannot raise into the scheduler step ***", _ok)

# -- (b20) the bounds guard: turn an Xid into a named refusal ---------------
#
# 2026-10-08, one second before the engine took a CUDA launch failure:
#   NVRM: Xid (PCI:000f:01:00): 13, Graphics Exception: ... Data deaddead
# Xid 13 is in practice an illegal memory access FROM A KERNEL, and this box
# logged zero Xid lines across the previous two boots (three weeks) and six on
# the boot where the disk offload went live -- the only new thing doing bulk
# GPU<->pinned-host DMA.
#
# check_pairing already guards WHICH key goes with which block but explicitly
# does not bound the indices. A block id past the GPU tensor's row count is
# exactly how a copy kernel reads off the end, and the kernel does not
# validate: it faults asynchronously, with the traceback landing somewhere
# unrelated. So bound them while they are still Python.
cb = diskio.check_bounds
eq("a well-formed round passes", cb([1, 2, 3], [0, 1, 2], 100, 8, [3], "x"), None)
check("*** a block id AT the row count is refused -- off-by-one is the whole "
      "class of bug ***",
      cb([99, 100], [0, 1], 100, 8, [2], "x") is not None)
check("a negative block id is refused", cb([-1], [0], 100, 8, [1], "x") is not None)
check("a slot past the pinned buffer is refused",
      cb([1], [8], 100, 8, [1], "x") is not None)
check("a length mismatch is refused (the pairing is positional)",
      cb([1, 2], [0], 100, 8, [2], "x") is not None)
check("group_sizes that do not sum to the block count is refused",
      cb([1, 2, 3], [0, 1, 2], 100, 8, [2], "x") is not None)
for _args, _want in (
    (([100], [0], 100, 8, [1], "GPU->staging"), "Xid 13"),
    (([1], [9], 100, 8, [1], "staging->GPU"), "pinned buffer"),
):
    _why = cb(*_args)
    check(f"...and the reason NAMES the hazard ({_want})", _want in (_why or ""))
check("...and names the direction, because the two are different code paths",
      "GPU->staging" in (cb([100], [0], 100, 8, [1], "GPU->staging") or ""))
eq("an empty round is not an error", cb([], [], 100, 8, [0], "x"), None)

# *** AND THE GUARD ITSELF CRASHED THE ENGINE, 2026-10-08. *** `_bounds()`
# returned 0 rows for a REBOUND load handler -- its old __init__ never set
# `_gpu_tensors`, which only the new one does -- and 0 made every legitimate
# block id "outside [0, 0)":
#
#   REFUSING a restore round -- staging->GPU: GPU block id 21586 is
#   outside [0, 0)
#
# It refused a valid restore, failed the job, and took the worker's exception
# path with it. ABSENT IS NOT ZERO, inside the guard written to prevent a
# crash. So an unknown bound is None and SKIPS the range check -- the safe
# direction -- while the shape checks, which need no bounds, still run.
# Called through a wrapper: with the None-skip removed these raise TypeError,
# and an escaping exception aborts the file instead of reporting a named
# failure. A test that fails for the wrong reason is as unhelpful as one that
# passes for it -- the same correction this suite needed for the arrival
# counter.
def _cb_safe(*a):
    try:
        return cb(*a)
    except Exception as exc:
        return "RAISED: %r" % (exc,)

eq("*** an unknown GPU bound SKIPS the range check, it does not refuse ***",
   _cb_safe([21586], [0], None, 8, [1], "staging->GPU"), None)
eq("...same for an unknown staging bound",
   _cb_safe([5], [999], 100, None, [1], "x"), None)
_shape1 = _cb_safe([1, 2], [0], None, None, [2], "x")
check("*** but a SHAPE error is still caught with bounds unknown -- those "
      "checks need no bounds at all ***",
      _shape1 is not None and not str(_shape1).startswith("RAISED"))
_shape2 = _cb_safe([1, 2, 3], [0, 1, 2], None, None, [2], "x")
check("...including group_sizes that do not sum",
      _shape2 is not None and not str(_shape2).startswith("RAISED"))

# and _bounds must never report 0, because 0 is what did the damage
class _NoTensors:
    _staging = []
    _bounds = diskio.DiskStoreHandler._bounds


_nb = _NoTensors()._bounds()
check("*** _bounds reports None, NEVER 0, when it cannot tell ***",
      _nb[0] is not 0 and _nb[1] is not 0 and 0 not in _nb)


class _Tiny:
    class _T:
        shape = (7,)
    _staging = [_T()]
    _gpu_tensors = [_T()]
    _bounds = diskio.DiskStoreHandler._bounds


eq("...and a real row count is reported as itself", _Tiny()._bounds(), (7, 7))

# *** BOTH HANDLERS MUST REPORT THEIR BOUNDS, and only the store one did. ***
# The load handler's snapshot omitted the fields, so it showed gpu_rows=None --
# and "the key is absent" and "the bound is unknown" are different facts that
# want opposite responses. The load path is where the 0-bound crash happened,
# so it is the one that most needs to be readable. Asserted against the SOURCE,
# because a snapshot needs a live handler to call.
_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                         "diskio.py")).read()
eq("both handler snapshots report gpu_rows", _src.count("gpu_rows=self._bounds()[0]"), 2)
eq("...and staging_rows", _src.count("staging_rows=self._bounds()[1]"), 2)
eq("...and the ledger", _src.count("ledger=self._ledger_snapshot()"), 2)
check("...and an unusable answer is NOT cached, so a later call can succeed",
      getattr(_NoTensors(), "_bounds_cache", None) is None)

# *** AND IT MUST REFUSE RATHER THAN SUBMIT. *** A guard that logs and then
# hands the kernel the same indices is worse than none: it would read as
# diagnosed while still faulting.
hb = mkhandler(8)
hb._bounds_cache = (50, 8)          # tiny GPU, so real ids are out of range
ksb = gkeys(0, 4)
gpub = base.GPULoadStoreSpec([1000, 1001, 1002, 1003],
                             group_sizes=[4, 0, 0, 0, 0], block_indices=[0] * 5)
_before = len(hb._inner.subs)
hb.transfer_async(99, (gpub, diskspec.DiskLoadStoreSpec(ksb)))
eq("*** nothing was submitted to the copy kernel ***",
   len(hb._inner.subs), _before)
eq("...the refusal is counted", hb.stats["refused_bounds"], 1)
check("...and the job is marked failed rather than left pending for ever",
      hb._jobs[99]["failed"] is True)
eq("...and every staging slot was returned, so the pool does not leak",
   hb._pool.snapshot()["available"], 8)

# the ledger: shape only, never data, and absent is not a range of zero
hl = mkhandler(8)
eq("*** an untouched ledger is an ABSENCE, not an index range of 0 ***",
   hl._ledger_snapshot()["transfers"], 0)
check("...and says so", "ABSENCE" in hl._ledger_snapshot().get("why", ""))
hl._ledger_note("store", 2, [5, 9, 7], [1, 0, 2])
hl._ledger_note("load", 0, [40], [3])
_ls = hl._ledger_snapshot()
eq("the ledger counts transfers", _ls["transfers"], 2)
eq("...and keeps the RANGE of block ids handed to the kernel",
   _ls["block_id_range"], [5, 40])
eq("...and of staging slots", _ls["slot_range"], [0, 3])
eq("...and the recent ring records shape, not data",
   _ls["recent"][-1], ("load", 0, 1, 40, 40, 3, 3))
check("the ring is bounded, so a 180K prefill cannot grow it without limit",
      diskio._LEDGER_RING <= 64)
for _i in range(diskio._LEDGER_RING * 3):
    hl._ledger_note("store", 0, [_i], [0])
check("...and stays bounded in practice",
      len(hl._ledger["recent"]) <= diskio._LEDGER_RING)
_ok = True
try:
    hl._ledger_note("store", 0, None, None)   # hostile input
except Exception:
    _ok = False
check("*** the ledger can never raise into the transfer path ***", _ok)

# -- (b21) EVERYTHING HANDED TO A FRAMEWORK TO SERIALISE MUST BE A BUILTIN ---
#
# *** THIS KILLED THE ENGINE'S OUTPUT THREAD AND LOOKED LIKE A MEMORY WEDGE
# *** FOR AN HOUR. *** `page_size_bytes` comes from the engine's own tensor
# metadata, so it can be a numpy integer. It flows:
#
#   Chunk.nbytes -> plan_nbytes -> st["bytes"] -> TransferResult.transfer_size
#     -> KVConnectorStats -> the engine core's OUTPUT SOCKET (msgpack)
#
# and msgpack refuses it:
#
#   TypeError: Object of type <class 'numpy.int64'> is not serializable
#     vllm/v1/engine/core.py:1561 in process_output_sockets
#
# The consequence is the nastiest shape available: that THREAD dies, so results
# never reach the API server. Requests complete, the core goes idle, clients
# hang for ever, and the main loop logs NOTHING -- which is why an hour went on
# memory and concurrency theories. The fixture could not express it, because its
# refs used plain ints.
class _NpLike(int):
    """Stands in for numpy.int64: an int subclass that is NOT `int` exactly,
    which is the distinction msgpack's enc_hook actually makes."""
    __module__ = "numpy"
    def __repr__(self):
        return "numpy.int64(%d)" % int(self)


_npref = type("R", (), {})()
_npref.tensor_idx = _NpLike(0)
_npref.page_size_bytes = _NpLike(4096)
_npplan = diskio.file_plan([[_npref]], 0, slot=_NpLike(3))
check("*** file_plan coerces a numpy-typed page size to a builtin int ***",
      type(_npplan[0].nbytes) is int)
check("...and the tensor index", type(_npplan[0].tensor_idx) is int)
check("...and the slot", type(_npplan[0].slot) is int)
check("*** so plan_nbytes is a builtin, which is what gets serialised ***",
      type(diskio.plan_nbytes(_npplan)) is int)
check("...and group_file_sizes too",
      all(type(x) is int for x in diskio.group_file_sizes([[_npref]])))

# the type distinction is real: an int SUBCLASS is not caught by `== int`
check("...and the test's own stand-in would have slipped past a loose check",
      isinstance(_NpLike(1), int) and type(_NpLike(1)) is not int)

# *** THE SUMMARY AND THE EXIT LIVE AT THE END OF THE FILE. ***
# They used to sit here, in the middle, and a block appended after them printed
# its own FAIL lines while the suite still exited 0 -- so the new tests could
# not fail the suite at all. That is "a lint that cannot fail reads as
# coverage", created by appending in the wrong place, and it was found only by
# mutating the code the new tests covered and seeing exit=0 with FAIL on screen.
# Anything added below is therefore accounted for; anything added below the
# summary would not be.


# ===========================================================================
# IF THE FILE EXISTS, DO NOT COPY IT TO STAGING -- and run every guard first
# ===========================================================================
# `_write_batch` has always skipped an existing path, but it runs in `_collect`
# AFTER the GPU->staging copy has completed, so an already-stored block cost a
# full 63-tensor GPU->pinned copy that was then discarded. The project's own
# record carries the shape of that waste: a boot where the store path "wrote
# nothing, because every block it was asked to store was already on disk",
# reported as `stored: 92` against ZERO bytes written.
#
# It is also the Xid 13 hazard, not merely waste: the LOAD reads this same
# shared pinned pool under CU_MEMCPY_SRC_ACCESS_ORDER_ANY, whose precondition
# is that no concurrent GPU stream writes the source, and these copies are
# exactly such a write -- commonest during a restore, which is when the load is
# reading the pool.
_ff = __import__("kvmgr.flags", fromlist=["peek"])


def _mkjob(pool, keys, bids, group=0, bounds=None):
    hh = mkhandler(pool)
    if bounds is not None:
        hh._bounds_cache = bounds
    gs = [0] * 5
    gs[group] = len(bids)
    sp = base.GPULoadStoreSpec(list(bids), group_sizes=gs, block_indices=[0] * 5)
    hh.transfer_async(1234, (sp, diskspec.DiskLoadStoreSpec(keys)))
    return hh


def _touch(h, keys):
    for k in keys:
        p = h._mapper.get_file_name(k)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "wb").write(b"x")


# gkeys with a fresh salt gives keys whose files do NOT exist yet.
_fresh = gkeys(0, 4, salt=201)
_h1 = mkhandler(8)
for _k in _fresh:                      # make sure they really are absent
    _p = _h1._mapper.get_file_name(_k)
    if os.path.exists(_p):
        os.unlink(_p)
_h1 = _mkjob(8, _fresh, range(10, 14))
eq("a key whose file is ABSENT is still copied", len(_h1._inner.subs), 1)
eq("...and nothing is counted as skipped",
   _h1.stats.get("skipped_before_copy", 0), 0)

# Now the same keys with their files present.
_h2 = mkhandler(8)
_touch(_h2, _fresh)
_h2 = _mkjob(8, _fresh, range(10, 14))
eq("*** a key whose file EXISTS is NOT copied to staging ***",
   len(_h2._inner.subs), 0)
eq("...and the skip is COUNTED (it was a bare `continue` with no counter)",
   _h2.stats.get("skipped_before_copy", 0), 4)
eq("...and its staging slots are RETURNED, not leaked",
   _h2._pool.available, 8)
check("...and the job does not hang pending for ever -- it COMPLETES",
      1234 not in _h2._jobs or _h2._jobs[1234]["pending"] == 0)

# A MIXED batch: present and absent together, pairing preserved.
_mix_absent = gkeys(0, 2, salt=202)
_h3 = mkhandler(8)
for _k in _mix_absent:
    _p = _h3._mapper.get_file_name(_k)
    if os.path.exists(_p):
        os.unlink(_p)
_mixed = [_fresh[0], _mix_absent[0], _fresh[1], _mix_absent[1]]
_h3 = _mkjob(8, _mixed, [20, 21, 22, 23])
eq("a MIXED batch submits one round", len(_h3._inner.subs), 1)
eq("...skipping exactly the present ones",
   _h3.stats.get("skipped_before_copy", 0), 2)
# The pairing is positional and load-bearing, so the surviving block ids must be
# the ones belonging to the surviving keys -- 21 and 23, never 20 and 21.
# FakeInner records (job_id, block_ids, group_sizes, slot_ids), so the block
# ids are element 1 -- NOT a spec object. My first version guarded this behind
# `hasattr(spec, "block_ids")` and the assertion never ran at all: it printed
# nothing and the suite still said PASS, which is a skipped check masquerading
# as coverage. Unconditional now, and it is the assertion that matters most
# here, because the key->block pairing is positional and a desync writes one
# block's KV under another block's name with nothing crashing.
_sub = _h3._inner.subs[0]
eq("*** and the SURVIVING block ids are the absent keys' own ***",
   list(_sub[1]), [21, 23])
eq("...and group_sizes was RECOUNTED for the filtered batch, not left at 4",
   list(_sub[2]), [2, 0, 0, 0, 0])
eq("...and exactly two slots were handed to the copy",
   len(_sub[3]), 2)

# *** THE REGRESSION DISCRIMINATOR. *** My first version skipped present keys
# BEFORE check_bounds, so a present key was never bounds-checked. The existing
# assertion "nothing was submitted to the copy kernel" still PASSED -- vacuously
# -- and only a second assertion counting the refusal exposed it. The bounds
# guard exists to DETECT corruption, not merely to prevent a submit, so it must
# run on every key regardless of whether its bytes are already on disk.
_hb2 = mkhandler(8)
_touch(_hb2, _fresh)
_hb2 = _mkjob(8, _fresh, [1000, 1001, 1002, 1003], bounds=(50, 8))
eq("*** bounds are STILL refused for a key whose file exists ***",
   _hb2.stats["refused_bounds"], 1)
eq("...and nothing was submitted", len(_hb2._inner.subs), 0)
eq("...and it is a REFUSAL, not a silent skip",
   _hb2.stats.get("skipped_before_copy", 0), 0)

# And the knob turns it off, restoring the old (wasteful) behaviour exactly.
F.write(disk_skip_present_before_copy=0)
flags.reset_for_tests()
_h4 = mkhandler(8)
_touch(_h4, _fresh)
_h4 = _mkjob(8, _fresh, range(30, 34))
eq("knob OFF: the present key is copied again, as before",
   len(_h4._inner.subs), 1)
F.write(disk_skip_present_before_copy=1)
flags.reset_for_tests()



# ===========================================================================
# THE LOAD'S STAGING MUST BE ITS OWN -- the Xid 13 root cause, structurally
# ===========================================================================
# Sharing one pinned buffer faulted the worker five times. Upstream's load path
# sets `is_src_access_order_any = not gpu_to_cpu`, and its own comment gives the
# precondition: CPU->GPU "reads from host pinned memory, which is never written
# by a concurrent GPU stream". Our store writes that buffer FROM A GPU STREAM,
# so the precondition was false for us -- structurally, not as a race.
#
# Pinned by SOURCE TEXT, and the reason is worth stating rather than hiding: the
# harness stubs torch and `mkhandler` builds handlers with `__new__`, bypassing
# `__init__` entirely, so no offline fixture can observe the allocation at all.
# A lint is the only thing here that can fail if somebody restores the sharing,
# and a revert is exactly the mistake to guard against, because sharing is the
# cheaper-looking option and its comment argued for it persuasively for weeks.
_lsrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskio.py")).read()
_load_init = _lsrc[_lsrc.index("class DiskLoadHandler"):]
_load_init = _load_init[:_load_init.index("\n    def _write_row")]

check("*** the load does NOT take the store's staging ***",
      "self._staging = store._staging" not in _load_init)
check("*** nor the store's slot pool ***",
      "self._pool = store._pool" not in _load_init)
check("...it allocates its own pinned rows", "torch.empty(" in _load_init)
check("...and its own pool", "self._pool = SlotPool(" in _load_init)
check("...with pin_memory requested, like the store's",
      "pin_memory=pin" in _load_init)
check("...using the SAME import path the store uses, not a guessed one "
      "(torch_utils would be an ImportError at boot)",
      "from vllm.utils.platform_utils import is_pin_memory_available"
      in _load_init)
check("...and the log line says WHY, so an operator sees the cost and the "
      "reason together", "SEPARATE from the store's" in _load_init)

# The live size must be reconcilable, or the split would hand the load a pool
# whose knob silently did nothing -- this project's most repeated defect.
check("*** the load has its OWN reconcile_pool ***",
      "def reconcile_pool" in _lsrc[_lsrc.index("class DiskLoadHandler"):])
check("...driven from its _collect, the once-per-step edge",
      "self.reconcile_pool()" in _lsrc[_lsrc.index("class DiskLoadHandler"):])
check("...reading its own knob, not the store's",
      'peek("staging_load_slots")' in _lsrc)
check("...and staging_load_slots is a declared knob, not an invented name",
      "staging_load_slots" in flags.DEFAULTS)



# ===========================================================================
# EVERY stats COUNTER MUST BE INITIALISED IN THE CLASS THAT INCREMENTS IT
# ===========================================================================
# `skipped_before_copy` was incremented in DiskStoreHandler and initialised in
# DiskLoadHandler. Two consequences, and the second is the worse one: the store
# reported `ABSENT`, which reads as "no such counter" when the truth was "zero
# skips"; and the load reported a permanent measured **0** for an event it
# cannot have, which looks like evidence. The handlers' stats dicts are nearly
# identical, so a `count(...) == 1` check over the whole FILE passed -- exactly
# one matched, and it was the wrong one.
#
# So this audits per CLASS and enumerates from the declaration rather than from
# a hand-written list, which is this project's stated rule for exactly this
# shape of defect.
import ast as _ast  # noqa: E402

_src = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                         "diskio.py")).read()
_tree = _ast.parse(_src)


def _stats_keys(cls_node):
    """(initialised, incremented) key sets for one class."""
    init, used = set(), set()
    for n in _ast.walk(cls_node):
        # self.stats = {...}
        if isinstance(n, _ast.Assign) and isinstance(n.value, _ast.Dict):
            for t in n.targets:
                if (isinstance(t, _ast.Attribute) and t.attr == "stats"
                        and isinstance(t.value, _ast.Name) and t.value.id == "self"):
                    for k in n.value.keys:
                        if isinstance(k, _ast.Constant) and isinstance(k.value, str):
                            init.add(k.value)
        # self.stats["x"] ... anywhere it is written or read-modified
        if (isinstance(n, _ast.Subscript) and isinstance(n.value, _ast.Attribute)
                and n.value.attr == "stats"
                and isinstance(n.value.value, _ast.Name)
                and n.value.value.id == "self"
                and isinstance(n.slice, _ast.Constant)
                and isinstance(n.slice.value, str)):
            used.add(n.slice.value)
    return init, used


_classes = {n.name: n for n in _tree.body if isinstance(n, _ast.ClassDef)}
check("both handler classes were located",
      {"DiskStoreHandler", "DiskLoadHandler"} <= set(_classes))
for _name in ("DiskStoreHandler", "DiskLoadHandler"):
    _init, _used = _stats_keys(_classes[_name])
    check("%s initialises its stats dict at all" % _name, bool(_init))
    _missing = sorted(_used - _init)
    eq("*** %s: every counter it touches is initialised ***" % _name, _missing, [])
    # And the mirror: a key initialised but never touched by this class is a
    # measured zero for something that cannot happen -- how the load handler
    # came to report skipped_before_copy=0 forever.
    _dead = sorted(_init - _used)
    eq("...and none is initialised that it never touches (a zero that cannot "
       "move reads as evidence) [%s]" % _name, _dead, [])



# ===========================================================================
# THE LOAD BATCH IS BOUNDED -- ~256 blocks in one transfer_async kills a worker
# ===========================================================================
# Measured 2026-10-10 by walking the hot staging_load_slots knob up against a
# 327-key restore: 96, 128, 160, 192, 224 all clean at 1-2s; 256 faulted on the
# first attempt with Xid 13 and a dead engine. Five clean arms then a hard
# failure at exactly 256 reads as a 256-entry limit in upstream's batch copy,
# not gradual exhaustion.
#
# The configuration that worked before this was staging_load_blocks=64 -- an
# arbitrary quarter of the store's pool that happened to sit under the limit.
# That is not a safety property, so the cap is explicit and independent now.
_bsrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskio.py")).read()
_ns_cap = {}
exec(compile(_bsrc[_bsrc.index("def _load_batch_max"):
                   _bsrc.index("def staging_knobs")], "diskio.py", "exec"), _ns_cap)

check("the cap default is BELOW the measured-clean ceiling of 224, not at it",
      flags.DEFAULTS["load_batch_max"] == 192)
# TWO SEPARATE 192s: the declared default and the function's own fallback for an
# unreadable knob. They can drift apart silently, and then a box with a broken
# flags file would run a different cap from one with a good file -- so assert
# they agree rather than trusting two literals to stay equal.
F.write(load_batch_max=0)
flags.reset_for_tests()
eq("*** the in-code fallback equals the declared default ***",
   _ns_cap["_load_batch_max"](), flags.DEFAULTS["load_batch_max"])
F.write(load_batch_max=192)
flags.reset_for_tests()
check("...and it is a declared knob so it is hot",
      "load_batch_max" in flags.DEFAULTS)

_bload = _bsrc[_bsrc.index("class DiskLoadHandler"):]
check("*** the LOAD take is capped, not just pool-limited ***",
      "take_upto(min(len(bids), _cap))" in _bload)
# Re-aimed when the cap gained a SECOND bound in a different unit. It pinned
# the literal `_cap = _load_batch_max()`, which the op budget replaced -- so it
# failed on a change that strengthened the thing it was guarding. The property
# is that the cap is composed of knob READERS and contains no magic number.
check("...and the cap comes from knob readers, not a literal",
      "_cap = min(_load_batch_max()," in _bload
      and "load_ops_budget(self._refs, g," in _bload)
import re as _re6
_capline = _bload[_bload.index("_cap = min("):]
_capline = _capline[:_capline.index("\n                slots")]
check("...and no bare integer is baked into the cap expression",
      not _re6.search(r"\b\d{2,}\b", _capline))
check("...and the STORE is NOT capped by it (its direction never faulted, and "
      "a shared cap would be a coincidence not a reason)",
      "_load_batch_max()" not in _bsrc[_bsrc.index("class DiskStoreHandler"):
                                       _bsrc.index("class DiskLoadHandler")])
# A bad or zero knob must fall back, never mean "no blocks ever".
_ns = _ns_cap
F.write(load_batch_max=0)
flags.reset_for_tests()
eq("a ZERO knob falls back to the default, it does not forbid every block",
   _ns["_load_batch_max"](), 192)
F.write(load_batch_max=-5)
flags.reset_for_tests()
eq("...and a negative one too", _ns["_load_batch_max"](), 192)
F.write(load_batch_max=64)
flags.reset_for_tests()
eq("...and a real value is honoured", _ns["_load_batch_max"](), 64)
F.write(load_batch_max=192)
flags.reset_for_tests()

# The load's pool must be OBSERVABLE, or a size knob cannot be verified -- the
# bisection printed "size=? cap=?" on every arm because only the store's
# snapshot carried `staging`.
check("*** the LOAD snapshot exposes its staging pool ***",
      "staging=self._pool.snapshot()" in _bload)
eq("...and both handlers now do", _bsrc.count("staging=self._pool.snapshot()"), 2)



# ===========================================================================
# THE FENCE: a stalled job dies ATTRIBUTABLY instead of hanging to an NCCL abort
# ===========================================================================
# *** AND THE DIAGNOSIS THIS REPLACES WAS MINE AND IT WAS WRONG. *** I had
# recorded "the chunked store pump deadlocks against the fence, because it is
# driven BY THE STEP IT IS BLOCKING -- the job needs a step and the step needs
# the job". That is refuted by upstream's own code:
#
#   gpu_model_runner.py:  if not num_scheduled_tokens:
#                             if not has_kv_transfer_group(): return EMPTY...
#                             return self.kv_connector_no_forward(...)
#   kv_connector_no_forward:  "# KV send/recv even if no work to do."
#   EngineCore.step():        gates only on has_requests(), then ALWAYS calls
#                             execute_model
#
# So with a deferred request present the scheduler returns zero tokens and the
# connector is driven ANYWAY, every loop iteration -- our _collect and _pump
# included. The pump is not starved of steps.
#
# What actually happened (measured 2026-10-10): Xid 32 at 14:58:49, the stalled
# job's t0 at 14:58:47. The GPU broke two seconds before the job started, so its
# completion event can never fire, and the pump ran thousands of times against a
# copy that will never finish. Then NCCL's 600s watchdog aborted the worker in
# C++ at 15:09:49 -- eleven minutes of `verdict=stalled` and no Python traceback.
#
# So the fix is NOT a recovery and must not be described as one. Upstream's
# `assert transfer_result.success` has no failure path, so reporting failure
# kills the worker. The choice is between an undiagnosable death in eleven
# minutes and an attributed death now.
check("stall_fail_after_s is a declared knob, so the deadline is hot",
      "stall_fail_after_s" in flags.DEFAULTS)
check("...and the default is INSIDE NCCL's 600s budget, which is the whole point",
      0 < flags.DEFAULTS["stall_fail_after_s"] < 600)

_fsrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskio.py")).read()
_fns = {}
exec(compile(_fsrc[_fsrc.index("def stall_action"):
                   _fsrc.index("# Set by the WATCHDOG THREAD")],
             "diskio.py", "exec"), _fns)
_sa = _fns["stall_action"]
eq("a job that is PROGRESSING is never failed", _sa("progress", 9999, 240), "none")
eq("...nor one merely WAITING", _sa("waiting", 9999, 240), "none")
eq("...nor an IDLE path", _sa("idle", 9999, 240), "none")
eq("a stall SHORT of the deadline is left alone", _sa("stalled", 239, 240), "none")
eq("*** a stall PAST the deadline is failed ***", _sa("stalled", 240, 240), "fail")
eq("...and well past", _sa("stalled", 10_000, 240), "fail")
eq("deadline 0 means HANG FOR EVER, the old behaviour, switchable with no "
   "restart", _sa("stalled", 10_000, 0), "none")
eq("...and a negative deadline is treated as off, not as instant death",
   _sa("stalled", 10_000, -5), "none")

# The watchdog must NOT mutate `_jobs` itself: `_collect` owns them on the
# worker's thread, and two writers in that one structure is a data race in the
# place that can least afford one.
_wd = _fsrc[_fsrc.index("def _watchdog_loop"):]
_wd = _wd[:_wd.index("\ndef ")] if "\ndef " in _wd else _wd
# COMMENTS STRIPPED FIRST. `_jobs` appears in the watchdog body only inside the
# comment explaining that it must not touch them -- so a raw substring check
# fails on its own documentation, which is this file's "a lint read the
# stylesheet with comments left in" trap arriving a third time. And checking for
# one SPELLING is not enough either: the mutation that beat the first version
# wrote `st.update(failed=True)` rather than `st["failed"] = True`. The
# structural property is that the watchdog never NAMES the job dicts at all --
# it reads progress through _handler_progress() and nothing else.
_wd_code = "\n".join(l.split("#", 1)[0] for l in _wd.splitlines())
check("*** the watchdog raises a REQUEST and never names _jobs in CODE ***",
      "_STALL_FAIL_REQUEST" in _wd_code and "_jobs" not in _wd_code)
check("...nor reaches the handler list directly to mutate it",
      "_LIVE_HANDLERS" not in _wd_code)
check("...and the acting helper is called from _collect, which owns them",
      _fsrc.count("honour_stall_fail(self)") == 2)
check("...once per handler, store and load",
      "def honour_stall_fail" in _fsrc)

# And it must never raise into the transfer path.
class _Boom:
    @property
    def _jobs(self):
        raise RuntimeError("boom")
_hsf = {}
exec(compile(_fsrc[_fsrc.index("def honour_stall_fail"):
                   _fsrc.index("def _handler_progress")], "diskio.py", "exec"),
     {"_STALL_FAIL_REQUEST": 1.0, **_hsf}, _hsf)
eq("*** a handler that raises while being marked returns 0, it does not "
   "propagate into the transfer path ***", _hsf["honour_stall_fail"](_Boom()), 0)



# ===========================================================================
# BOUND THE ROUNDS IN FLIGHT -- the per-round cap was not enough
# ===========================================================================
# load_batch_max=192 made a 327-key restore clean 3/3 where an uncapped 256
# pool faulted 2/2. At 861 keys (the 700K scale test, 2026-10-10) it faulted
# anyway WITH the per-round cap respected throughout -- Xid 13, worker dead one
# second later, zero stall lines. So the size of any one batch is not what
# moved between clean and dead; the number in flight TOGETHER is. A chunked
# prefill issues many prepare_load calls, _pump loops over every job, each
# takes slots until the pool empties, and upstream chains them all on one
# stream with an event each and bounds nothing.
#
# This is the THIRD time this investigation has changed two things at once and
# attributed the result to the wrong one (shared-vs-size, allocation-vs-limit,
# batch-size-vs-rounds). The knob exists so the hypothesis can be tested rather
# than assumed: 1 makes the path fully serial.
check("load_inflight_max is a declared knob, so it is hot",
      "load_inflight_max" in flags.DEFAULTS)
check("...and the default is SMALL -- a less parallel restore beats a dead "
      "worker", 0 < flags.DEFAULTS["load_inflight_max"] <= 8)

_csrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskio.py")).read()
_cload = _csrc[_csrc.index("class DiskLoadHandler"):]
check("*** the LOAD pump bounds rounds in flight before taking slots ***",
      "if len(self._inner_to_job) >= _load_inflight_max():" in _cload)
check("...and the STORE pump does NOT -- its direction never faulted this way, "
      "and a shared bound would be a coincidence not a reason",
      "_load_inflight_max()" not in _csrc[_csrc.index("class DiskStoreHandler"):
                                          _csrc.index("class DiskLoadHandler")])
check("*** the count and its cap are OBSERVABLE, or the knob cannot be "
      "verified ***",
      "inflight=len(self._inner_to_job)," in _cload
      and "inflight_max=_load_inflight_max()," in _cload)

# the reader: 0 means unbounded-was-the-bug, so it must fall back, and a bad
# value must not mean "never submit anything".
_cns = {}
exec(compile(_csrc[_csrc.index("def _load_inflight_max"):
                   _csrc.index("def _load_batch_max")], "diskio.py", "exec"), _cns)
_lim = _cns["_load_inflight_max"]
F.write(load_inflight_max=0)
flags.reset_for_tests()
eq("a ZERO knob falls back to the default, it does not mean unbounded again",
   _lim(), flags.DEFAULTS["load_inflight_max"])
F.write(load_inflight_max=-3)
flags.reset_for_tests()
eq("...and a negative one too, it does not mean 'never submit'",
   _lim(), flags.DEFAULTS["load_inflight_max"])
F.write(load_inflight_max=1)
flags.reset_for_tests()
eq("...and 1 is honoured, which is the fully-serial test arm", _lim(), 1)
F.write(load_inflight_max=4)
flags.reset_for_tests()

# Returning from _pump when the bound is hit must not be a REFUSAL: the work has
# to resume. That is only safe because _pump is re-entered every step, and
# kv_connector_no_forward means those steps keep coming with nothing scheduled.
check("*** hitting the bound RETURNS rather than failing the job -- the work "
      "resumes on the next step ***",
      "if len(self._inner_to_job) >= _load_inflight_max():\n                    return"
      in _cload)



# ===========================================================================
# THE REAL LIMIT IS COPY OPS, NOT BLOCKS -- read out of upstream, not guessed
# ===========================================================================
# `_custom_ops.swap_blocks_batch` "submit[s] all copies in a single driver
# call" (cuMemcpyBatchAsync on CUDA 12.8+), and `cpu/gpu_worker.py` builds it as
#
#     num_copy_ops += group_size * len(group_data_refs)
#
# So one submit contains BLOCKS x REFS-FOR-THAT-GROUP descriptors. load_batch_max
# caps blocks, which means a DIFFERENT descriptor count per group: 192 blocks is
# 384 ops for a 2-ref group and 5,760 for a 30-ref one. Group 0 has the most
# layers and the largest block size, so it reaches any descriptor limit first,
# and a single cap in blocks cannot express that. My cap bounded the wrong unit.
_osrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
                          "diskio.py")).read()
_ons = {}
exec(compile(_osrc[_osrc.index("def load_ops_for"):
                   _osrc.index("def _load_inflight_max")], "diskio.py", "exec"), _ons)
_ops_for, _budget = _ons["load_ops_for"], _ons["load_ops_budget"]

# refs shaped like the engine's: group 0 has many layers, the SWA groups few.
_R = [[0]*30, [0]*8, [0]*8, [0]*2, [0]*4]
eq("*** ops = blocks x refs for THAT group, group 0 ***", _ops_for(_R, 0, 192), 5760)
eq("...and the same 192 blocks is far fewer ops in a 2-ref group",
   _ops_for(_R, 3, 192), 384)
eq("...which is exactly why one cap in BLOCKS cannot fit five groups",
   _ops_for(_R, 0, 192) // _ops_for(_R, 3, 192), 15)
eq("unknown geometry caps at nblocks rather than becoming unbounded",
   _ops_for([], 9, 50), 50)

eq("the budget converts ops back to a block count, group 0",
   _budget(_R, 0, 192, 2048), 68)
eq("...and lets a low-ref group take far more of the same budget",
   _budget(_R, 3, 192, 2048), 192)
eq("a budget of 0 means unbounded, the behaviour that faulted",
   _budget(_R, 0, 192, 0), 192)
eq("*** it NEVER returns 0 for a positive want -- a restore that can never "
   "submit is worse than one that submits one block ***",
   _budget(_R, 0, 192, 1), 1)
eq("...and 0 blocks wanted is still 0, not 1", _budget(_R, 0, 0, 2048), 0)
eq("unknown geometry falls back to one ref per block, so it still bounds",
   _budget([], 9, 192, 64), 64)

_oload = _osrc[_osrc.index("class DiskLoadHandler"):]
check("*** the LOAD pump applies BOTH bounds, in different units ***",
      "min(_load_batch_max()," in _oload and "load_ops_budget(self._refs, g," in _oload)
check("...and refs_per_group is EXPOSED, or the op arithmetic checks against "
      "nothing -- the engine log truncates the fingerprint that carries it",
      "refs_per_group=[len(r) for r in (self._refs or [])]," in _oload)
check("...with the live cap beside it", "ops_max=_load_ops_max()," in _oload)
check("the inflight bound is applied ONCE, not duplicated by the edit",
      _oload.count("if len(self._inner_to_job) >= _load_inflight_max():") == 1)


# ---------------------------------------------------------------------------
# *** EVERY IMPORT-FAILED LITERAL MUST AGREE WITH ITS DECLARATION. ***
# Each of the three knob readers falls back to flags.DEFAULTS, with a bare
# literal behind it for the one case where flags cannot be imported at all.
# That literal is the second copy of the number this fix exists to remove, so
# it is only safe while it matches. It did NOT match once: DEFAULTS was lowered
# to load_inflight_max=1 after 4 was measured to kill the engine (Xid 32 x2,
# 2026-10-10) and the reader still said 4, so a zero or unparseable knob
# reinstated the fatal value through the safety path.
#
# ENUMERATED FROM THE SOURCE, not from a hand-written list -- this project's own
# rule, because a hand-written table of 6 of 24 knobs is how the last one hid.
import re as _re_lit
_litsrc = open(_DISKIO).read() if "_DISKIO" in dir() else _csrc
_found = {}
for _m in _re_lit.finditer(
        r'int\(_f\.DEFAULTS\["(\w+)"\]\)\s*or\s*(\d+)', _litsrc):
    _found[_m.group(1)] = int(_m.group(2))
check("every knob reader derives its default from flags.DEFAULTS",
      len(_found) >= 3, f"found {sorted(_found)}")
for _n, _lit in sorted(_found.items()):
    eq(f"...and {_n}'s import-failed literal agrees with the declaration",
       _lit, flags.DEFAULTS[_n])
# and the trailing `except Exception: return <literal>` of each reader too --
# a different line, the same hazard
for _m in _re_lit.finditer(
        r'int\(_f\.DEFAULTS\["(\w+)"\]\)\s*or\s*(\d+)\s*\n\s*except Exception:\s*\n\s*return (\d+)',
        _litsrc):
    eq(f"...and {_m.group(1)}'s except-path literal matches it too",
       int(_m.group(3)), flags.DEFAULTS[_m.group(1)])

print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
