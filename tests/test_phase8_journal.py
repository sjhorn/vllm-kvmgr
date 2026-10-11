#!/usr/bin/env python3.12
"""Phase 8: the presence journal's pure half.

The manifest is O(index) to persist O(changes), and on the live box that
saturates: manifest_writes 109 against manifest_skipped_busy 462 under a real
store load. A refused flush is a window in which a key is on disk with nothing
able to NAME it, which is how 155 GiB of unreachable orphans accumulated. So
these tests are about DURABILITY, not speed.

The property to defend above all others is that replay is IDEMPOTENT, because
that is the only reason compaction can be crash-safe: the snapshot is written
first and the journal truncated second, so a crash in between replays records
the snapshot already holds.
"""
from __future__ import annotations

import os
import sys

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


harness.load_kv_offload()
from kvmgr import diskspec as D  # noqa: E402

K1 = bytes.fromhex("aa" * 32)
K2 = bytes.fromhex("bb" * 32)
K3 = bytes.fromhex("cc" * 32)

# -- encode ----------------------------------------------------------------
line = D.journal_encode(D.JOURNAL_ADD, [(K1, 0), (K2, 7)])
check("an add record ends in exactly one newline", line.endswith("\n") and line.count("\n") == 1)
check("an add record embeds no newline of its own", "\n" not in line[:-1])
eq("an add round-trips to its keys and positions",
   D.journal_decode(line), (D.JOURNAL_ADD, [(K1, 0), (K2, 7)]))
eq("a remove round-trips", D.journal_decode(D.journal_encode(D.JOURNAL_REMOVE, [K3])),
   (D.JOURNAL_REMOVE, [K3]))
eq("a clear round-trips", D.journal_decode(D.journal_encode(D.JOURNAL_CLEAR, [])),
   (D.JOURNAL_CLEAR, []))
try:
    D.journal_encode("zzz", [])
    check("encode refuses an unknown kind", False, "no raise")
except ValueError:
    check("encode refuses an unknown kind", True)

# -- decode: every way a line can be untrustworthy -------------------------
for name, bad in [
    ("an empty line", ""),
    ("whitespace only", "   \n"),
    ("not json at all", "{this is not json"),
    ("a truncated json object", '{"a":[["aa",0'),
    ("a json array rather than an object", "[1,2,3]"),
    ("a two-key object (ambiguous record)", '{"a":[],"r":[]}'),
    ("an unknown kind", '{"zzz":[]}'),
    ("odd-length hex", '{"r":["abc"]}'),
    ("non-hex characters", '{"r":["zz"]}'),
    ("an add whose pos is not an int", '{"a":[["aa","later"]]}'),
    ("an add missing its pos", '{"a":[["aa"]]}'),
]:
    eq(f"decode returns None for {name}", D.journal_decode(bad), None)

# -- replay ----------------------------------------------------------------
base = [(K1, 3)]
ents, st = D.journal_replay(base, [])
eq("an empty journal leaves the snapshot alone", ents, [(K1, 3)])
eq("...and reports no records", st["records"], 0)

ents, st = D.journal_replay(base, [D.journal_encode(D.JOURNAL_ADD, [(K2, 5)])])
eq("an add appends after the snapshot", ents, [(K1, 3), (K2, 5)])
eq("...and is counted", st["adds"], 1)

# Mirrors _remember's early return, which is half of the idempotence property.
ents, st = D.journal_replay(base, [D.journal_encode(D.JOURNAL_ADD, [(K1, 99)])])
eq("re-adding a present key does not change its position", ents, [(K1, 3)])
eq("...and is not counted as an add", st["adds"], 0)

ents, st = D.journal_replay([(K1, 3), (K2, 5)],
                            [D.journal_encode(D.JOURNAL_REMOVE, [K1])])
eq("a remove drops the key", ents, [(K2, 5)])
eq("...and is counted", st["removes"], 1)

# Mirrors _forget's pop default -- the other half of idempotence.
ents, st = D.journal_replay(base, [D.journal_encode(D.JOURNAL_REMOVE, [K3])])
eq("removing an absent key is a no-op", ents, [(K1, 3)])
eq("...and is not counted as a remove", st["removes"], 0)

ents, st = D.journal_replay([(K1, 3), (K2, 5)], [D.journal_encode(D.JOURNAL_CLEAR, [])])
eq("a clear wipes everything before it", ents, [])
eq("...and is counted", st["clears"], 1)

ents, _ = D.journal_replay([], [D.journal_encode(D.JOURNAL_ADD, [(K3, 1)]),
                                D.journal_encode(D.JOURNAL_ADD, [(K1, 2)]),
                                D.journal_encode(D.JOURNAL_ADD, [(K2, 3)])])
eq("replay preserves append order", [k for k, _ in ents], [K3, K1, K2])

# -- the torn tail: the shape an unclean kill actually leaves ---------------
good = D.journal_encode(D.JOURNAL_ADD, [(K1, 1)])
torn = D.journal_encode(D.JOURNAL_ADD, [(K2, 2)])[:-6]      # cut mid-record
after = D.journal_encode(D.JOURNAL_ADD, [(K3, 3)])
ents, st = D.journal_replay([], [good, torn, after])
eq("records before the tear are applied", [k for k, _ in ents], [K1])
eq("the tear is reported", st["torn"], 1)
check("*** NOTHING PAST THE TEAR IS APPLIED ***", K3 not in dict(ents),
      "a skipped tear would apply changes from the far side of a gap")

ents, st = D.journal_replay([], [good, after])
eq("an intact journal reports no tear", st["torn"], 0)
eq("...and applies every record", len(ents), 2)

# -- IDEMPOTENCE: the compaction crash-safety property ---------------------
# Compaction writes the snapshot, THEN truncates the journal. A crash between
# the two replays records the snapshot already contains, so replay must be a
# fixed point. This is the single assertion that makes that ordering safe.
j = [D.journal_encode(D.JOURNAL_ADD, [(K1, 1), (K2, 2)]),
     D.journal_encode(D.JOURNAL_REMOVE, [K1]),
     D.journal_encode(D.JOURNAL_ADD, [(K3, 3)])]
once, _ = D.journal_replay([], j)
twice, _ = D.journal_replay(once, j)
eq("replaying a journal over its own result changes nothing", twice, once)
thrice, _ = D.journal_replay(twice, j)
eq("...and is still a fixed point a third time", thrice, once)

# *** AND THAT FIXED-POINT PAIR IS A WEAK GUARD -- MEASURED, NOT ASSUMED. ***
# Break the add's early return and the two assertions above still PASS, because
# every surviving entry's position came from the same journal, so overwriting
# it writes the value already there. It was the headline assertion here until a
# mutation showed it could not fail. Kept as a regression guard against an
# implementation that stops using a mapping, and labelled for what it is.
#
# THE REAL DISCRIMINATOR IS A STALE POSITION, and it is the case compaction
# actually produces: the new snapshot records each key's position, while the
# not-yet-truncated journal still carries an OLDER add for the same key. A
# non-idempotent replay reinstates the stale one -- and tail-first eviction
# ORDERS on position, so a silently reinstated value degrades the victim rule
# rather than erroring, which is this repo's worst failure shape.
ents, st = D.journal_replay([(K1, 42)], [D.journal_encode(D.JOURNAL_ADD, [(K1, 1)])])
eq("*** A STALE ADD DOES NOT REINSTATE AN OLDER POSITION ***", dict(ents)[K1], 42)
eq("...and the stale add is not counted", st["adds"], 0)

# The direction that is NOT safe, asserted so the ordering is never flipped:
# a remove replayed over a snapshot that predates its add must not resurrect.
eq("a remove is not undone by replaying the add that preceded it",
   dict(D.journal_replay(once, j)[0]).get(K1), None)

# -- compaction_due --------------------------------------------------------
check("the record bound trips it", D.compaction_due(1000, 0, 1000, 0))
check("just under the record bound does not", not D.compaction_due(999, 0, 1000, 0))
check("the byte bound trips it", D.compaction_due(0, 5_000_000, 0, 5_000_000))
check("just under the byte bound does not", not D.compaction_due(0, 4_999_999, 0, 5_000_000))
check("either bound alone is enough", D.compaction_due(0, 9_999, 1000, 9_999))
check("neither bound set means never", not D.compaction_due(10**9, 10**9, 0, 0))
check("a ZEROED bound disables that half rather than firing always",
      not D.compaction_due(10**9, 0, 0, 5_000_000))
check("a NEGATIVE bound disables that half too",
      not D.compaction_due(10**9, 0, -1, 0))


# ==========================================================================
# THE I/O SHELL, against real files in a real temp directory.
#
# The pure functions above cannot see an ordering mistake, and the ordering IS
# the crash-safety contract: snapshot, then rotate, then (on success) discard.
# These exercise it through the manager.
# ==========================================================================
import json  # noqa: E402
import shutil  # noqa: E402
import tempfile  # noqa: E402


class FakeMapper:
    """Only `get_file_name` is used -- by the load-time filesystem prune."""

    def __init__(self, root):
        self.root = root

    def get_file_name(self, key):
        return os.path.join(self.root, bytes(key).hex() + ".bin")


class Ctx:
    def __init__(self, rid="r1"):
        self.req_id = rid


def mkmgr(tmp, world=2):
    m = D.DiskPresenceManager(FakeMapper(tmp), num_workers=world,
                              manifest_path=os.path.join(tmp, "presence.json"))
    return m


def touchfile(tmp, key):
    with open(os.path.join(tmp, bytes(key).hex() + ".bin"), "wb") as f:
        f.write(b"x")


FJ = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv8-flags-"), "flags.json"))
from kvmgr import flags as KF  # noqa: E402

print()
print("== (8f) the knobs are declared, inert, and the bounds fail safe ==")
for n, want in [("journal_enabled", False), ("journal_fsync", False),
                ("journal_max_records", 20000), ("journal_max_bytes", 33554432)]:
    check(f"{n} is declared", n in KF.DEFAULTS, f"DEFAULTS has {n}")
    eq(f"...default {n}", KF.DEFAULTS[n], want)
check("*** journal_enabled defaults OFF, so an empty flags file is a no-op ***",
      KF.DEFAULTS["journal_enabled"] is False)

print()
print("== (8g) the append edges, and what lands on disk ==")
tmp = tempfile.mkdtemp(prefix="kv8-on-")
FJ.write(journal_enabled=1)
KF.reset_for_tests()
m = mkmgr(tmp)
ctx = Ctx()
m.complete_store([K1, K2], ctx)
jpath = os.path.join(tmp, "presence.json.journal")
check("complete_store wrote a journal file", os.path.exists(jpath))
# *** READ VIA A HELPER THAT TOLERATES ABSENCE. *** The first version did
# `open(jpath).readlines()` directly, so deleting the append from
# complete_store made this ABORT on FileNotFoundError after correctly failing
# the assertion above -- and an aborted suite prints no summary, so the
# mutation read as uncaught. A mutation that aborts is weaker than one that
# fails: the discriminator has to survive the thing it discriminates.
def jlines(path):
    try:
        return open(path).readlines()
    except Exception:
        return []


def first(path):
    recs = [D.journal_decode(l) for l in jlines(path)]
    return recs[0] if recs and recs[0] else (None, [])


recs = [D.journal_decode(l) for l in jlines(jpath)]
eq("one record for the batch", len(recs), 1)
eq("...an ADD", first(jpath)[0], D.JOURNAL_ADD)
eq("...naming both keys", sorted(k for k, _ in first(jpath)[1]), sorted([K1, K2]))
check("*** THE RECORD IS ON DISK BEFORE ANY SNAPSHOT EXISTS ***",
      not os.path.exists(os.path.join(tmp, "presence.json")),
      "this is the orphan window closing: a key is nameable immediately")

print()
print("== (8h) with the journal OFF nothing is appended ==")
tmp2 = tempfile.mkdtemp(prefix="kv8-off-")
FJ.write(journal_enabled=0)
KF.reset_for_tests()
m2 = mkmgr(tmp2)
m2.complete_store([K1], ctx)
check("no journal file is created", not os.path.exists(
    os.path.join(tmp2, "presence.json.journal")))

print()
print("== (8i) a restart is reconstructed from snapshot + journal ==")
# The whole point, end to end: store, DO NOT snapshot, build a fresh manager
# over the same directory, and see the keys come back.
tmp3 = tempfile.mkdtemp(prefix="kv8-restart-")
FJ.write(journal_enabled=1)
KF.reset_for_tests()
m3 = mkmgr(tmp3)
for k in (K1, K2, K3):
    touchfile(tmp3, k)          # the prune is real, so the files must exist
m3.complete_store([K1, K2, K3], ctx)
check("still no snapshot", not os.path.exists(os.path.join(tmp3, "presence.json")))
m4 = mkmgr(tmp3)
m4.lookup(K1, ctx)              # _ensure_index() loads on first use
eq("*** ALL THREE KEYS SURVIVED WITH NO SNAPSHOT AT ALL ***",
   len(m4._present), 3)
eq("...reported as replayed adds", m4._stats.get("journal_adds"), 3)
eq("...and the snapshot contributed nothing", m4._stats.get("manifest_loaded"), 3)

print()
print("== (8j) the filesystem prune still applies to replayed keys ==")
# A replayed key whose file has gone must be dropped: upstream ASSERTS on a
# failed load, so a stale claim is an engine death, not a miss.
os.unlink(os.path.join(tmp3, bytes(K2).hex() + ".bin"))
m5 = mkmgr(tmp3)
m5.lookup(K1, ctx)
eq("*** A REPLAYED KEY WITH NO FILE IS PRUNED ***", len(m5._present), 2)
check("...and it is the right one", bytes(K2) not in m5._present)
eq("...counted as pruned", m5._stats.get("manifest_pruned"), 1)

print()
print("== (8k) eviction journals its removes ==")
tmp4 = tempfile.mkdtemp(prefix="kv8-evict-")
m6 = mkmgr(tmp4)
for k in (K1, K2):
    touchfile(tmp4, k)
m6.complete_store([K1, K2], ctx)
m6._journal_append(D.JOURNAL_REMOVE, [K1])     # the edge _enforce_budget uses
m7 = mkmgr(tmp4)
m7.lookup(K1, ctx)
eq("a journalled remove is honoured on reload", len(m7._present), 1)
check("...and it removed the right key", bytes(K1) not in m7._present)

print()
print("== (8l) compaction: snapshot, rotate, THEN discard ==")
tmp5 = tempfile.mkdtemp(prefix="kv8-compact-")
m8 = mkmgr(tmp5)
for k in (K1, K2):
    touchfile(tmp5, k)
m8.complete_store([K1, K2], ctx)
st = m8._jstate()
check("the journal has records before compaction", st["records"] >= 1)
m8.flush_manifest(force=True)
m8.flush_wait(timeout=10.0)
check("the snapshot now exists", os.path.exists(os.path.join(tmp5, "presence.json")))
check("*** compacting IS DISCARDED once the snapshot is durable ***",
      not os.path.exists(os.path.join(tmp5, "presence.json.compacting")))
eq("the live journal's counters are reset", m8._jstate()["records"], 0)
m9 = mkmgr(tmp5)
m9.lookup(K1, ctx)
eq("the index reloads from the snapshot alone", len(m9._present), 2)
eq("...with no journal records to replay", m9._stats.get("journal_records"), 0)

print()
print("== (8m) a crash DURING compaction loses nothing ==")
# Snapshot written, compacting NOT discarded -- the window the ordering exists
# to survive. Replay must absorb the duplicate records.
tmp6 = tempfile.mkdtemp(prefix="kv8-crash-")
m10 = mkmgr(tmp6)
for k in (K1, K2, K3):
    touchfile(tmp6, k)
m10.complete_store([K1, K2], ctx)
snap = list(m10._present.items())
m10._journal_rotate()                       # journal -> compacting
m10._write_manifest(snap)                   # ...which also discards compacting
# Re-create the crash: the snapshot is there and compacting is back.
shutil.copy(os.path.join(tmp6, "presence.json"), os.path.join(tmp6, "presence.json.bak"))
with open(os.path.join(tmp6, "presence.json.compacting"), "w") as f:
    f.write(D.journal_encode(D.JOURNAL_ADD, [(K1, 0), (K2, 0)]))
with open(os.path.join(tmp6, "presence.json.journal"), "w") as f:
    f.write(D.journal_encode(D.JOURNAL_ADD, [(K3, 0)]))
m11 = mkmgr(tmp6)
m11.lookup(K1, ctx)
eq("*** A CRASH MID-COMPACTION STILL YIELDS EVERY KEY, ONCE ***",
   len(m11._present), 3)
check("...no key is duplicated", len(set(m11._present)) == len(m11._present))

print()
print("== (8n) an append must never raise into complete_store ==")
tmp7 = tempfile.mkdtemp(prefix="kv8-raise-")
m12 = mkmgr(tmp7)
touchfile(tmp7, K1)
# Make the journal path unopenable by putting a DIRECTORY where the file goes.
os.makedirs(os.path.join(tmp7, "presence.json.journal"), exist_ok=True)
raised = False
try:
    m12.complete_store([K1], ctx)
except Exception as exc:
    raised = True
    print("   raised:", type(exc).__name__, exc)
check("*** complete_store SURVIVES AN UNWRITABLE JOURNAL ***", not raised,
      "an exception here is an engine outage, not a lost record")
eq("...the key is still in the in-memory index", len(m12._present), 1)
check("...and the failure is counted", m12._stats.get("journal_errors", 0) >= 1)

print()
print("== (8o) a FAILED snapshot must NOT discard compacting ==")
# This is the crash-safety contract stated as a test rather than a comment.
# `_write_manifest` never raises -- it logs and counts -- so the only way to
# tell a successful write from a failed one is whether the records it was
# supposed to absorb are still there afterwards. Discard on failure and those
# keys are gone from every future boot: files on disk that nothing can name.
tmp8 = tempfile.mkdtemp(prefix="kv8-failwrite-")
m13 = mkmgr(tmp8)
touchfile(tmp8, K1)
m13.complete_store([K1], ctx)
snap8 = list(m13._present.items())
m13._journal_rotate()
check("compacting exists after the rotate",
      os.path.exists(os.path.join(tmp8, "presence.json.compacting")))
# *** THE FAILURE HAS TO LAND ON os.replace, NOT ON THE .part OPEN. ***
# A directory at the .part path fails BEFORE the discard would run either way,
# so moving the discard earlier was invisible -- the mutation read as uncaught
# when really the test never reached the mutated line. A directory at the
# MANIFEST path lets the .part be written and json.dump succeed, and fails
# exactly at the rename, which is the only point the ordering is about.
os.makedirs(os.path.join(tmp8, "presence.json"), exist_ok=True)
before_writes = m13._stats.get("manifest_writes", 0)
m13._write_manifest(snap8)          # logs, counts nothing, must not raise
eq("the write did not succeed", m13._stats.get("manifest_writes", 0), before_writes)
check("*** COMPACTING SURVIVES A FAILED SNAPSHOT WRITE ***",
      os.path.exists(os.path.join(tmp8, "presence.json.compacting")),
      "discarding here would orphan every key the snapshot failed to absorb")
# ...and the keys are still recoverable, which is the point of surviving.
shutil.rmtree(os.path.join(tmp8, "presence.json"))
try:
    os.unlink(os.path.join(tmp8, "presence.json.part"))
except Exception:
    pass
m14 = mkmgr(tmp8)
m14.lookup(K1, ctx)
eq("...so the key still reloads", len(m14._present), 1)

print()
print("== (8p) turning the journal OFF must not discard what it already holds ==")
# *** REPLAY DELIBERATELY DOES NOT CONSULT journal_enabled, and no test covered
# that until a mutation added the check and nothing failed. *** The knob gates
# WRITING. Records already on disk are NEWER than the snapshot, so honouring
# the knob on the read path would silently drop real presence the moment
# somebody turned the feature off -- files on disk that nothing can name, which
# is the exact orphan this mechanism exists to prevent, caused by its own off
# switch. The recovery from a journal you distrust is to DELETE it.
tmp9 = tempfile.mkdtemp(prefix="kv8-onoff-")
FJ.write(journal_enabled=1)
KF.reset_for_tests()
m15 = mkmgr(tmp9)
for k in (K1, K2):
    touchfile(tmp9, k)
m15.complete_store([K1, K2], ctx)
check("records were written while it was on",
      os.path.exists(os.path.join(tmp9, "presence.json.journal")))
FJ.write(journal_enabled=0)             # somebody turns the feature off
KF.reset_for_tests()
m16 = mkmgr(tmp9)
m16.lookup(K1, ctx)
eq("*** THE RECORDS ARE STILL REPLAYED WITH THE KNOB OFF ***",
   len(m16._present), 2)
eq("...and counted", m16._stats.get("journal_adds"), 2)

FJ.write(journal_enabled=0)
KF.reset_for_tests()

print()
print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
