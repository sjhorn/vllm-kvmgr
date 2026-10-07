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
from kvmgr import diskio  # noqa: E402


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
check("*** over-asking returns None rather than raising -- the connector "
      "retries ***", pool.take(2) is None)
eq("...and nothing was consumed by the refusal", pool.available, 1)
pool.give(s1)
eq("returning restores them", pool.available, 4)
eq("high water is remembered", pool.snapshot()["high_water"], 3)
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

print()
print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
