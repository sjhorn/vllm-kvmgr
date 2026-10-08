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
    h.stats = {"submitted": 0, "refused_pairing": 0, "files": 0, "bytes": 0,
               "errors": 0, "wait_calls": 0, "wait_ours": 0,
               "wait_blocked_ms": 0, "wait_max_pending": 0, "wait_rounds": 0,
               "saturations": 0}
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

print()
print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
