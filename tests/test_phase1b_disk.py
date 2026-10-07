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
from kvmgr import diskio, diskspec  # noqa: E402


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
    h.stats = {"submitted": 0, "refused_pairing": 0, "files": 0, "bytes": 0,
               "errors": 0}
    h._read_row = lambda t, s, n: b"\xab" * n
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
eq("...of exactly the pool size", len(h._inner.subs[0][1]), 8)
check("...with the job still queued", h._jobs[77]["batches"] != [])
check("...and the saturation note said ONCE", h._saturated is True)

rounds, done = 0, []
while not done and rounds < 100:
    done = h.get_finished()
    rounds += 1
eq("the job completes after pumping", len(done), 1)
eq("...reporting OUR job id, not an inner one", done[0].job_id, 77)
check("...as a success", done[0].success is True)
eq("...over ceil(100/8) rounds", len(h._inner.subs), 13)
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
print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
