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
               "saturations": 0}
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
h = mkhandler(8)
ks = gkeys(0, 100)
gpu = base.GPULoadStoreSpec(list(range(1000, 1100)), group_sizes=[100, 0, 0, 0, 0],
                            block_indices=[0] * 5)
check("*** a 100-block job against an 8-block pool is ACCEPTED, not refused ***",
      h.transfer_async(77, (gpu, diskspec.DiskLoadStoreSpec(ks))) is True)
eq("...and only a pool's worth is submitted at first", len(h._inner.subs), 1)
# *** THE FLOOR CHANGES THIS NUMBER ON PURPOSE, so the expectation is derived
# rather than hardcoded. *** The store and load handlers share one pool, and
# that sharing made the same restore take 574.9s with stores live against 3.3s
# with them paused. A quarter of the pool is now reserved for the other
# direction, so a store round gets `size - read_floor`, not `size`. Writing the
# arithmetic out keeps the test honest about WHY the number moved.
_sz, _wf, _rf = diskio.staging_plan(8, 0, 0, 0)
eq("the derived read floor for an 8-slot pool", _rf, 2)
eq("...so a store round takes size - read_floor", len(h._inner.subs[0][1]),
   _sz - _rf)
check("...with the job still queued", h._jobs[77]["batches"] != [])
check("...and the saturation note said ONCE", h._saturated is True)

rounds, done = 0, []
while not done and rounds < 100:
    done = h.get_finished()
    rounds += 1
eq("the job completes after pumping", len(done), 1)
eq("...reporting OUR job id, not an inner one", done[0].job_id, 77)
check("...as a success", done[0].success is True)
# Derived too, for the same reason: with a floor the per-round batch is
# size - read_floor, so the round count is ceil(100 / (size - floor)).
import math as _math  # noqa: E402
eq("...over ceil(100 / (size - read_floor)) rounds", len(h._inner.subs),
   _math.ceil(100 / (_sz - _rf)))
check("*** and every block still moves exactly once -- the floor must not "
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
check("*** enough KEYS still flushes on its own, clock untouched ***",
      mt2._stats["manifest_writes"] > w0)

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

# and the fence counters must actually MOVE, or they measure nothing
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
check("...and the take path no longer resizes (that was the idle-engine bug)",
      "set_limit" not in _dsrc[_dsrc.index("reserve=_rf)") - 400:
                               _dsrc.index("reserve=_rf)")])

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

print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
