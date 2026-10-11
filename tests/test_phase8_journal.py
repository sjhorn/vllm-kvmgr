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

print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
