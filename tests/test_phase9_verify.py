#!/usr/bin/env python3.12
"""Phase 9: cross-rank presence verification and the free-space floor.

*** BOTH MECHANISMS EXIST BECAUSE UPSTREAM CANNOT EXPRESS A FAILED TRANSFER.
*** `offloading/worker.py:273` is a bare `assert transfer_result.success` under
the comment "we currently do not support job failures". So a file missing on
ONE node, a truncated file, or a full disk is not a miss -- it is a dead engine
core. "Handle the error gracefully" is not on the menu: a hazard here is closed
only by making that assert UNREACHABLE.

Three paths ran from an ordinary disk condition to a dead engine:

  1. the presence index is verified on ONE node (`_load_manifest`'s prune is
     rank-local, and its own docstring admits it);
  2. the walk checks existence, not SIZE -- in a syscall that already returns
     the size;
  3. a full disk had no preflight anywhere; there was no statvfs in the plugin.

Everything decided here is PURE or drivable with no GPU, no torch and no RPC,
for the reason everything dangerous in this plugin is: it is the half that can
be wrong in a way nothing crashes on.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the PACKAGE root
sys.path.insert(0, os.path.join(REPO, "src"))

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
from kvmgr import diskspec as D  # noqa: E402
from kvmgr import flags as KF  # noqa: E402

K = base.make_offload_key
F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv9-"), "flags.json"))


class FakeMapper:
    """Key -> path under `root`, so presence can be made true, false or WRONG."""

    def __init__(self, root):
        self.root = root

    def get_file_name(self, key):
        return os.path.join(self.root, bytes(key).hex() + ".bin")


class Ctx:
    def __init__(self, rid="r1"):
        self.req_id = rid


def mkmgr(tmp, world=2):
    return D.DiskPresenceManager(FakeMapper(tmp),
                                 num_workers=world,
                                 manifest_path=os.path.join(tmp, "presence.json"))


def touch(tmp, key, nbytes=1):
    with open(os.path.join(tmp, bytes(key).hex() + ".bin"), "wb") as fh:
        fh.write(b"x" * nbytes)


def first(seq, default=b""):
    """`seq[0]` or a sentinel. *** A MUTATION THAT ABORTS IS WEAKER THAN ONE
    THAT FAILS: *** `[0]` of an emptied list detects the bug while saying
    nothing about it, and an aborted suite prints no summary -- so the mutation
    reads as uncaught. The discriminator has to survive the thing it
    discriminates.
    """
    try:
        return seq[0]
    except Exception:
        return default


def jtext(tmpdir):
    """The journal's text, tolerating its ABSENCE -- which is the mutation."""
    try:
        return open(os.path.join(tmpdir, "presence.json.journal")).read()
    except Exception:
        return ""


def vouch(rank, missing=(), wrong=(), checked=0, free=None):
    return {"rank": rank, "missing": list(missing), "wrong_size": list(wrong),
            "checked": checked, "free_bytes": free}


def restored(tmp, keys, world=2):
    """A manager whose keys came from the MANIFEST -- i.e. unverified.

    This is the state the whole phase is about, and it has to be built the way
    a boot builds it (write a manifest, construct a fresh manager) rather than
    by poking `_present`: the point is that `_load_manifest`'s prune checked
    ONE node, so a poked index would be testing a state no boot produces.
    """
    with open(os.path.join(tmp, "presence.json"), "w") as fh:
        json.dump({"keys": [[bytes(k).hex(), i] for i, k in enumerate(keys)]}, fh)
    return mkmgr(tmp, world)


K1, K2, K3 = K(b"\x11" * 32, 0), K(b"\x22" * 32, 0), K(b"\x33" * 32, 0)

print("== (9a) free_band: three bands, and hysteresis that cannot flap ==")
GB = 10 ** 9
eq("well clear of the floor is ok",
   D.free_band(100 * GB, 64 * GB, 16 * GB, 8 * GB), D.BAND_OK)
eq("below the floor is degraded",
   D.free_band(40 * GB, 64 * GB, 16 * GB, 8 * GB), D.BAND_DEGRADED)
eq("at or below hard is frozen",
   D.free_band(16 * GB, 64 * GB, 16 * GB, 8 * GB), D.BAND_FROZEN)
eq("a full disk is frozen, not unknown",
   D.free_band(0, 64 * GB, 16 * GB, 8 * GB), D.BAND_FROZEN)
# *** HYSTERESIS IS NOT POLISH: without it a tier sitting at the floor flaps on
# every store round, and the band EDGE is what gets logged -- so a flapping
# tier is also a log flood, which is how a reader learns to ignore the one
# transition that matters.
eq("*** recovering to just above the floor does NOT leave degraded ***",
   D.free_band(65 * GB, 64 * GB, 16 * GB, 8 * GB, D.BAND_DEGRADED),
   D.BAND_DEGRADED)
eq("...it takes the floor PLUS the hysteresis",
   D.free_band(73 * GB, 64 * GB, 16 * GB, 8 * GB, D.BAND_DEGRADED), D.BAND_OK)
eq("...and a frozen tier cannot jump straight to ok",
   D.free_band(70 * GB, 64 * GB, 16 * GB, 8 * GB, D.BAND_FROZEN),
   D.BAND_DEGRADED)
eq("entering degraded does not need the hysteresis (pressure is immediate)",
   D.free_band(63 * GB, 64 * GB, 16 * GB, 8 * GB, D.BAND_OK), D.BAND_DEGRADED)

print()
print("== (9b) verify_verdict: ANY rank's miss is the key's miss ==")
v = D.verify_verdict([vouch(0, checked=2), vouch(1, checked=2)], 2)
eq("two clean ranks leave nothing bad", v["bad"], [])
eq("...and no fault", v["fault"], None)
v = D.verify_verdict([vouch(0, missing=[bytes(K1).hex()], checked=2),
                      vouch(1, checked=2)], 2)
eq("*** ONE rank missing a file condemns the key ***", v["bad"], [bytes(K1)])
v = D.verify_verdict([vouch(0, checked=2),
                      vouch(1, wrong=[bytes(K2).hex()], checked=2)], 2)
eq("*** a WRONG SIZE is as bad as absent ***", v["bad"], [bytes(K2)])
v = D.verify_verdict([vouch(0, missing=[bytes(K1).hex()]),
                      vouch(1, missing=[bytes(K1).hex()])], 2)
eq("the same key condemned by both ranks is listed once", v["bad"], [bytes(K1)])
# *** A SHORT REPLY LIST IS A FAULT, NOT A SMALLER CHECK. *** Treating it as
# "the ranks that answered all said yes" is the manufactured zero this repo
# keeps paying for, and it would vouch for a node on the strength of not having
# asked it.
v = D.verify_verdict([vouch(0, checked=2)], 2)
check("*** one reply for two ranks is a FAULT ***", bool(v["fault"]),
      str(v["fault"]))
eq("...and it condemns nothing, because it proved nothing", v["bad"], [])
v = D.verify_verdict([vouch(0), {"rank": 1, "error": "no live disk handler"}], 2)
check("a rank reporting an error is a fault", bool(v["fault"]), str(v["fault"]))
v = D.verify_verdict([vouch(0), "not a dict"], 2)
check("a rank replying with the wrong TYPE is a fault", bool(v["fault"]))

print()
print("== (9c) the defer: upstream's own None, and what it costs ==")
tmp = tempfile.mkdtemp(prefix="kv9-defer-")
F.write(disk_restore=1, disk_verify_enabled=1)
KF.reset_for_tests()
for k in (K1, K2):
    touch(tmp, k)
m = restored(tmp, [K1, K2])
eq("the manifest loaded both keys", m._stats["manifest_loaded"], 2)
eq("*** a present but UNVERIFIED key DEFERS -- None, not True ***",
   m.lookup(K1, Ctx()), None)
eq("...and `hit` is NOT counted for a defer", m._stats["hit"], 0)
eq("...the defer is counted as what it is", m._stats.get("verify_deferred"), 1)
eq("...and the key is queued for the pump", list(m._wanted_map()), [bytes(K1)])
eq("a key that is simply ABSENT still answers False at once",
   m.lookup(K3, Ctx()), False)
eq("...and absent keys are never queued", list(m._wanted_map()), [bytes(K1)])

print()
print("== (9d) the pump's verdict, applied: the SAME key then answers ==")
m.verify_apply([bytes(K1)], [vouch(0, checked=1), vouch(1, checked=1)], 2)
eq("*** and THEN the key hits ***", m.lookup(K1, Ctx()), True)
eq("...now `hit` is counted", m._stats["hit"], 1)
eq("...and the queue is drained", list(m._wanted_map()), [])
eq("the verified count is visible", m._verify_snapshot()["verified"], 1)
eq("...as is what has not been checked yet",
   m._verify_snapshot()["unverified"], 1)

print()
print("== (9e) a rank reporting a miss FORGETS the key and journals it ==")
tmp = tempfile.mkdtemp(prefix="kv9-heal-")
F.write(disk_restore=1, disk_verify_enabled=1, journal_enabled=1)
KF.reset_for_tests()
for k in (K1, K2):
    touch(tmp, k)
m = restored(tmp, [K1, K2])
m.lookup(K1, Ctx())
m.verify_apply([bytes(K1), bytes(K2)],
               [vouch(0, checked=2), vouch(1, missing=[bytes(K1).hex()], checked=2)], 2)
check("*** the condemned key is OUT of the index ***", bytes(K1) not in m._present)
check("...and the clean one is still in it", bytes(K2) in m._present)
eq("...the drop is counted", m._stats.get("verify_forgot"), 1)
eq("...and lookup now simply misses rather than claiming",
   m.lookup(K1, Ctx()), False)
# *** JOURNALLED, or a boot resurrects a key the cluster has just said it
# cannot serve. Same ordering argument as eviction: a remove recorded for a
# file that survives costs one re-store; a delete with no record is an orphan.
_j = jtext(tmp)
check("*** the removal is JOURNALLED, so a boot cannot resurrect it ***",
      bytes(K1).hex() in _j and '"r"' in _j,
      "no journal file at all" if not _j else "")

print()
print("== (9f) THE SELF-HEAL: the next store re-indexes it, verified ==")
# This is the property the whole design is for: a file lost on ONE node is
# repaired by the next store instead of killing the engine.
out = m.prepare_store([K1], Ctx())
eq("the forgotten key is offered again", [bytes(k) for k in out.keys_to_store],
   [bytes(K1)])
m.complete_store([K1], Ctx())
check("...it is back in the index", bytes(K1) in m._present)
eq("*** and born VERIFIED, because complete_store IS the all-ranks edge ***",
   m.lookup(K1, Ctx()), True)

print()
print("== (9g) a defer EXPIRES to False, so nothing waits for ever ==")
tmp = tempfile.mkdtemp(prefix="kv9-exp-")
F.write(disk_restore=1, disk_verify_enabled=1, disk_verify_defer_s=0.0001)
KF.reset_for_tests()
touch(tmp, K1)
m = restored(tmp, [K1])
eq("the first ask defers", m.lookup(K1, Ctx()), None)
import time as _t
_t.sleep(0.01)
eq("*** and once the budget is spent it answers MISS, not a claim ***",
   m.lookup(K1, Ctx()), False)
eq("...counted", m._stats.get("verify_defer_expired"), 1)

print()
print("== (9h) a rank that cannot answer REFUSES, and the fault is queryable ==")
tmp = tempfile.mkdtemp(prefix="kv9-fault-")
F.write(disk_restore=1, disk_verify_enabled=1)
KF.reset_for_tests()
D._VERIFY_FAULT = None
touch(tmp, K1)
m = restored(tmp, [K1])
m.verify_fault("rank 1 did not answer (test)")
check("*** a fault is recorded ***", D._VERIFY_FAULT is not None)
eq("...presence REFUSES while it stands -- never 'probably fine'",
   m.lookup(K1, Ctx()), False)
eq("...and the refusal is counted", m._stats.get("verify_refused_fault"), 1)
_snap = m._verify_snapshot()
check("*** the fault is in the snapshot, so a script can FAIL on it ***",
      isinstance(_snap.get("fault"), dict) and "reason" in _snap["fault"])
m.verify_fault("a second, different reason")
eq("...a second fault does not overwrite the first (the first is the cause)",
   D._VERIFY_FAULT["reason"], "rank 1 did not answer (test)")
eq("...but it is counted", D._VERIFY_FAULT["seen"], 2)
D._VERIFY_FAULT = None
eq("and `fault` is explicitly null when there is none -- never absent",
   m._verify_snapshot()["fault"], None)

print()
print("== (9i) verification OFF is exactly today's behaviour ==")
F.write(disk_restore=1, disk_verify_enabled=0)
KF.reset_for_tests()
eq("*** with the knob off a present key hits at once, no defer ***",
   m.lookup(K1, Ctx()), True)
F.write(disk_restore=1, disk_verify_enabled=1)
KF.reset_for_tests()

print()
print("== (9j) the RPC timeout can never become `None` or zero ==")
# *** timeout=None IS A WAIT WITH NO END. *** collective_rpc does
# `deadline = None if timeout is None`, which becomes `mq.dequeue(timeout=None)`
# -- so a rank whose filesystem is in D-state, exactly what this defends
# against, would block the engine core for ever with NO collective pending and
# therefore no NCCL watchdog to end it. A hung engine with no traceback is
# worse than the crash it replaces.
def code_only(path, blank_strings=True):
    """The source with COMMENTS AND STRINGS BLANKED IN PLACE, by position.

    *** THIS LINT FAILED ON ITS FIRST RUN FOR EXACTLY THE REASON THIS REPO
    *** ALREADY RECORDS. *** The comment in `pump` that explains the hazard --
    "NEVER None, which collective_rpc turns into dequeue(timeout=None)" --
    contains the very string the lint searches for, so the paragraph warning
    about the trap WAS the failure. A stylesheet lint learnt this once and a
    tokenize lint learnt the sibling lesson: blank the regions IN PLACE, by
    position, because joining the surviving tokens turns `.data[:] =` into
    `. data [ : ] =` and the pattern can then never match at all.
    """
    import io
    import tokenize
    src = open(path).read()
    lines = src.splitlines(keepends=True)
    offs, acc = [], 0
    for ln in lines:
        offs.append(acc)
        acc += len(ln)
    out = list(src)
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except Exception:
        return src
    kinds = ((tokenize.COMMENT, tokenize.STRING) if blank_strings
             else (tokenize.COMMENT,))
    for tok in toks:
        if tok.type not in kinds:
            continue
        (r1, c1), (r2, c2) = tok.start, tok.end
        a, b = offs[r1 - 1] + c1, offs[r2 - 1] + c2
        for i in range(a, min(b, len(out))):
            if out[i] != "\n":
                out[i] = " "
    return "".join(out)


_dspath = os.path.join(REPO, "src", "kvmgr", "diskspec.py")
_src = open(_dspath).read()
_code = code_only(_dspath)
# *** AND A SECOND VIEW, WITH STRINGS KEPT. *** A check about a dict LOOKUP --
# `flags.DEFAULTS["disk_verify_rpc_timeout_s"]` -- is a check about a string
# literal, so the view that blanks strings cannot see it. One blanker with a
# flag, rather than two subtly different copies that drift.
_code_s = code_only(_dspath, blank_strings=False)
# The canary: the helper must IGNORE the idiom in a comment and CATCH it in
# code. A lint that cannot fail reads as coverage.
check("the comment-blanker really did blank something",
      "timeout=None" in _src and "timeout=None" not in _code)
check("*** no collective_rpc in the plugin passes timeout=None ***",
      "timeout=None" not in _code)
check("...the pump passes a timeout at all",
      "collective_rpc(" in _code and "timeout=timeout" in _code)
# And the fallback literal must agree with the declaration -- the
# load_inflight_max lesson, where a reader's own fallback reinstated the fatal
# value through the safety path.
check("*** a zeroed knob falls back to the DECLARATION, not to a literal ***",
      'flags.DEFAULTS["disk_verify_rpc_timeout_s"]' in _code_s)
check("...and that check can see string literals at all (canary)",
      '"disk_verify_rpc_timeout_s"' in _code_s
      and "disk_verify_rpc_timeout_s" not in _code)
check("disk_verify_rpc_timeout_s is declared non-zero",
      float(KF.DEFAULTS["disk_verify_rpc_timeout_s"]) > 0)

print()
print("== (9k) the floor's three bands, through prepare_store ==")
tmp = tempfile.mkdtemp(prefix="kv9-floor-")
F.write(disk_restore=1, disk_verify_enabled=0, disk_budget_gb=0,
        disk_free_floor_gb=64, disk_free_hard_gb=16,
        disk_free_hysteresis_gb=8, disk_free_evict_step_gb=2)
KF.reset_for_tests()
KEYS = [K(bytes([i]) * 32, 0) for i in range(1, 9)]
for k in KEYS:
    touch(tmp, k)
m = mkmgr(tmp)
m.complete_store(KEYS[:6], Ctx())


def band_at(mgr, free):
    """Pin the free-space reading without a filesystem."""
    st = mgr._free_state()
    st["at"] = D._now() + 10 ** 6   # defeat the throttle: never re-stat
    st["local"] = free
    st["ranks"] = None
    return mgr._free_pressure()[0]


eq("plenty free -> ok", band_at(m, 200 * GB), D.BAND_OK)
eq("under the floor -> degraded", band_at(m, 40 * GB), D.BAND_DEGRADED)
eq("under hard -> frozen", band_at(m, 5 * GB), D.BAND_FROZEN)

print()
print("== (9l) DEGRADED is net-negative by one step, with NO byte budget ==")
# *** AND THIS IS THE ORDER-OF-OPERATIONS TRAP. *** `budget_bytes` returns 0 for
# "unbounded" and `_enforce_budget` used to return [] on that immediately, so
# evaluating the floor after it would have left a disk_budget_gb=0 deployment
# with no floor AT ALL.
eq("the budget really is unset here", D.budget_bytes(KF.peek("disk_budget_gb")), 0)
band_at(m, 200 * GB)
eq("*** with free space OK and no budget, nothing is evicted ***",
   m._enforce_budget([KEYS[6]]), [])
band_at(m, 40 * GB)
_ev = m._enforce_budget([KEYS[6]])
check("*** DEGRADED evicts even though disk_budget_gb is 0 ***", len(_ev) > 0,
      f"{len(_ev)} victim(s)")
eq("...and the round is counted", m._stats.get("free_evict_rounds"), 1)

print()
print("== (9m) FROZEN still admits ONE key, so the deletes can ride out ==")
# *** A REFUSAL WOULD MAKE THE FREEZE PERMANENT. *** scheduler.py:745 does
# `if not store_output.keys_to_store: advance_stored_idx(...); continue`, which
# drops the store spec -- and the unlinks ride that spec by design.
tmp = tempfile.mkdtemp(prefix="kv9-frozen-")
m2 = mkmgr(tmp)
for k in KEYS:
    touch(tmp, k)
m2.complete_store(KEYS[:4], Ctx())
band_at(m2, 5 * GB)
out = m2.prepare_store(KEYS[4:8], Ctx())
eq("*** exactly one key is admitted, not zero ***", len(out.keys_to_store), 1)
check("...and a store spec exists to carry the unlinks",
      out.store_spec is not None)
eq("...the refusal of the rest is counted", m2._stats.get("free_refusals"), 3)
eq("...and the frozen round is counted", m2._stats.get("free_frozen_rounds"), 1)

print()
print("== (9n) the hard stop returns _nothing_to_store(), NEVER None ==")
# *** None MEANS "I CANNOT" AND RE-OFFERS THE SAME BLOCKS EVERY STEP FOR EVER
# *** -- measured here at 3,309 warnings at ~20/s. An empty output means
# "nothing new here" and lets the request progress.
band_at(m2, 100 * 2 ** 20)      # 100 MB: below FREE_ABSOLUTE_MIN_BYTES
out = m2.prepare_store(KEYS[4:8], Ctx())
check("*** the hard stop is not None ***", out is not None,
      "None means 'I cannot' and re-offers the same blocks every step for ever")
# Guarded, so the None mutation FAILS these two instead of aborting on them.
eq("...it is an EMPTY store output",
   list(out.keys_to_store) if out is not None else "<None>", [])
check("...which is exactly what _nothing_to_store() returns",
      out is not None and type(out) is type(D._nothing_to_store()))
check("the absolute minimum is in CODE, not a knob",
      isinstance(D.FREE_ABSOLUTE_MIN_BYTES, int)
      and D.FREE_ABSOLUTE_MIN_BYTES > 0)

print()
print("== (9o) an unreadable statvfs reads as OK, and SAYS so ==")
# *** THE OPPOSITE OF THE DROP GUARD'S CHOICE, AND FOR THE OPPOSITE REASON. ***
# There "I cannot tell" risks a dead worker, so it refuses. Here it risks only a
# cold cache, so refusing would turn a diagnostic failure into a feature outage.
m3 = mkmgr(tempfile.mkdtemp(prefix="kv9-vfs-"))
st = m3._free_state()
st["at"] = D._now() + 10 ** 6
st["local"] = None
st["ranks"] = None
eq("unknown free space does not freeze the tier", m3._free_pressure()[0], D.BAND_OK)
eq("*** and it is published as null, never as a zero ***",
   m3._free_snapshot()["bytes"], None)
m4 = mkmgr(tempfile.mkdtemp(prefix="kv9-vfs2-"))
m4._manifest = ""
eq("a manager with no manifest path cannot stat, and says None",
   m4._statvfs_free(), None)

print()
print("== (9p) the binding constraint is the MINIMUM across nodes ==")
# Each rank writes its own slice to its OWN node's disk, so a head-only reading
# is the wrong number on exactly the day it matters -- and the worker is
# currently the fuller of the two.
m5 = mkmgr(tempfile.mkdtemp(prefix="kv9-min-"))
st = m5._free_state()
st["at"] = D._now() + 10 ** 6
st["local"] = 200 * GB
m5.note_worker_free([300 * GB, 40 * GB])
eq("the worker minimum is recorded", m5._free_state()["ranks"], 40 * GB)
eq("*** a healthy head does not excuse a full worker ***",
   m5._free_pressure()[0], D.BAND_DEGRADED)
eq("...and the snapshot shows both", m5._free_snapshot()["worker_min_bytes"],
   40 * GB)
m5.note_worker_free([None, -1])
eq("unusable rank readings leave the worker minimum UNKNOWN, not zero",
   m5._free_state()["ranks"], None)

print()
print("== (9q) reset_cache clears the vouching too ==")
tmp = tempfile.mkdtemp(prefix="kv9-reset-")
m6 = mkmgr(tmp)
for k in (K1, K2):
    touch(tmp, k)
m6.complete_store([K1, K2], Ctx())
eq("stored keys are vouched for", len(m6._verified_set()), 2)
m6.reset_cache()
eq("*** a reset leaks no verdicts across it ***", len(m6._verified_set()), 0)
eq("...and no pending asks either", len(m6._wanted_map()), 0)

print()
print("== (9r) _forget drops the vouching, so a re-store cannot inherit it ==")
m7 = mkmgr(tempfile.mkdtemp(prefix="kv9-forget-"))
m7.complete_store([K1], Ctx())
check("vouched", bytes(K1) in m7._verified_set())
m7._forget(bytes(K1))
check("*** forgotten keys are unvouched -- a later file is a DIFFERENT file ***",
      bytes(K1) not in m7._verified_set())

print()
print("== (9s) the chunk is bounded, and demand beats background ==")
tmp = tempfile.mkdtemp(prefix="kv9-chunk-")
F.write(disk_restore=1, disk_verify_enabled=1, disk_verify_chunk=3,
        disk_verify_bg_per_s=0)
KF.reset_for_tests()
MANY = [K(bytes([i]) * 32, 0) for i in range(1, 20)]
for k in MANY:
    touch(tmp, k)
m8 = restored(tmp, MANY)
for k in MANY:
    m8.lookup(k, Ctx())
eq("every unverified ask is queued", len(m8._wanted_map()), len(MANY))
eq("*** the chunk bounds the RPC message ***", len(m8._verify_chunk()), 3)
eq("...oldest first, so the request closest to giving up is served first",
   first(m8._verify_chunk()), bytes(MANY[0]))
F.write(disk_restore=1, disk_verify_enabled=1, disk_verify_chunk=100,
        disk_verify_bg_per_s=0)
KF.reset_for_tests()
eq("with background off, the chunk is demand only",
   len(m8._verify_chunk()), len(MANY))

print()
print("== (9t) background re-verification never defers a lookup ==")
tmp = tempfile.mkdtemp(prefix="kv9-bg-")
F.write(disk_restore=1, disk_verify_enabled=1, disk_verify_chunk=100,
        disk_verify_bg_per_s=1000000)
KF.reset_for_tests()
for k in (K1, K2):
    touch(tmp, k)
m9 = mkmgr(tmp)
m9.complete_store([K1, K2], Ctx())        # born verified
_st = m9._verify_state()
_st["bg_at"] = D._now() - 1.0             # a second of credit
_chunk = m9._verify_chunk()
check("*** the background sweep picks up already-verified keys ***",
      len(_chunk) >= 1, f"{len(_chunk)} key(s)")
eq("*** and it does NOT enqueue them, so they stay claimable ***",
   len(m9._wanted_map()), 0)
eq("...a lookup during re-verification still hits", m9.lookup(K1, Ctx()), True)
check("...and the background work is counted",
      (m9._stats.get("verify_bg_checked") or 0) >= 1)
F.write(disk_restore=1, disk_verify_enabled=1, disk_verify_bg_per_s=0)
KF.reset_for_tests()
eq("a zero rate disables the background sweep entirely",
   m9._verify_chunk(), [])

print()
print("== (9u) the deferral queue is bounded in CODE ==")
check("VERIFY_WANTED_MAX is a code constant, not a knob",
      isinstance(D.DiskPresenceManager.VERIFY_WANTED_MAX, int))
check("...and it is not in the knob table",
      "verify_wanted_max" not in KF.DEFAULTS
      and "disk_verify_wanted_max" not in KF.DEFAULTS)

print()
print("== (9v) every new knob is declared, published and inert-safe ==")
for n in ("disk_verify_enabled", "disk_verify_chunk",
          "disk_boot_verify_chunk", "disk_verify_defer_s",
          "disk_verify_rpc_timeout_s", "disk_verify_bg_per_s",
          "disk_free_floor_gb", "disk_free_hard_gb",
          "disk_free_hysteresis_gb", "disk_free_evict_step_gb",
          "disk_free_check_s", "disk_free_rpc_s"):
    check(f"{n} is declared", n in KF.DEFAULTS)
# *** disk_verify_enabled SHIPPED DEFAULTING ON AND THE BOX SAID NO. ***
# Measured 2026-10-12: with it on, the 700K restore that costs 11s warm
# returned ZERO tokens off disk. The mechanism is right and the TRIGGER is
# wrong -- `_sliding_window_lookup` keeps scanning past a DEFERRED key, so one
# deferring pass walks a ~175,000-key sliding-window group and the request's
# verification job never finishes inside its budget. This assertion is pinned
# the other way round now, so flipping the default back cannot happen by
# accident: it needs the boot-time all-ranks pass first.
# *** AND IT IS ON AGAIN, because the TRIGGER MOVED TO BOOT. *** The lazy
# version broke the 700K restore; `_boot_verify` asks every rank before the API
# opens, which is cheaper than the rank-local walk it replaces (~200s against
# 365s) and makes the deferral structurally impossible.
check("*** disk_verify_enabled defaults ON, now the check is at BOOT ***",
      KF.DEFAULTS["disk_verify_enabled"] is True)
check("*** and the deferral budget defaults to 0, meaning NEVER DEFER ***",
      float(KF.DEFAULTS["disk_verify_defer_s"]) == 0.0)
# The FLOOR, by contrast, remains the documented safety-bound exception and is
# verified live: band `ok` at 2.42 TB free, inert, with both nodes' free space
# published. Same reasoning as load_inflight_max=1 -- the only state in which a
# non-zero floor changes behaviour is one where today's behaviour is ENOSPC and
# a dead engine.
check("*** the free-space floor defaults NON-ZERO, as the safety exception ***",
      float(KF.DEFAULTS["disk_free_floor_gb"]) > 0
      and float(KF.DEFAULTS["disk_free_hard_gb"]) > 0)
check("...and the floor is above the hard stop, not below it",
      float(KF.DEFAULTS["disk_free_floor_gb"])
      > float(KF.DEFAULTS["disk_free_hard_gb"]))
check("...which is in turn above the absolute code minimum",
      float(KF.DEFAULTS["disk_free_hard_gb"]) * 1e9
      > D.FREE_ABSOLUTE_MIN_BYTES)

print()
print("== (9w) the worker RPC is declared, and applied locally as a NO-OP ==")
_csrc = open(os.path.join(REPO, "src", "kvmgr",
                          "control.py")).read()
check("kvmgr_verify_presence is installed on Worker",
      "Worker.kvmgr_verify_presence = kvmgr_verify_presence" in _csrc)
# *** _apply_locally IGNORES UNKNOWN NAMES SILENTLY, so a reader would assume
# it ran in the core. The branch is explicit and commented for that reason.
check("*** and _apply_locally names it explicitly rather than ignoring it ***",
      'elif method == "kvmgr_verify_presence":' in _csrc)
check("the RPC compares SIZE, not just existence",
      "group_file_sizes" in _csrc and "st_size" in _csrc)
check("*** and unlinks a wrong-sized file, which exists() would preserve ***",
      "unlinked" in _csrc)
check("a rank with no handler reports an ERROR rather than an empty pass",
      "no live disk handler on this rank" in _csrc)
check("the pump rides the existing step wrap rather than a new installer",
      "_verify_pump(self)" in _csrc and "_install_verify" not in _csrc)

print()
print("== (9x) THE BOOT CHECK: every rank, before the API opens ==")
# *** THIS IS THE FIX FOR THE ONE THING THE BOX REFUTED. *** Verifying from
# `lookup` meant deferring, and `_sliding_window_lookup` keeps scanning past a
# deferred key -- so one pass walked a ~175,000-key SWA group and the 700K
# restore returned ZERO tokens. Asking every rank at BOOT removes the deferral
# structurally, and it is CHEAPER than the rank-local walk it replaces (~200s
# against 365s for 922,992 keys, because the ranks stat in parallel).


class FakeCore:
    """Just enough EngineCore: `collective_rpc` and a `model_executor`.

    The capture matches on CAPABILITY rather than a class name, so this is a
    faithful stand-in -- and that is deliberate, because `EngineCoreProc`
    subclasses `EngineCore` and a name list is a thing that goes stale.
    """

    def __init__(self, missing=(), wrong=(), ranks=2, raise_after=None):
        self.model_executor = object()
        self._missing = {bytes(k).hex() for k in missing}
        self._wrong = {bytes(k).hex() for k in wrong}
        self._ranks = ranks
        self._raise_after = raise_after
        self.calls = 0
        self.seen = []

    def build(self, tmp):
        """Construct the manager FROM A BOUND METHOD, so `self` is on the stack.

        *** A LAMBDA DOES NOT WORK HERE, AND THE FIRST VERSION USED ONE. ***
        The capture walks caller frames for a `self` with `collective_rpc` and
        a `model_executor`; a lambda assigned to an attribute is not a bound
        method, so it puts no `self` in its frame and the capture correctly
        found nothing -- which made twelve assertions fail for the wrong
        reason. That the capture refused rather than finding something else is
        the reassuring half.
        """
        return mkmgr(tmp)

    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        self.calls += 1
        if self._raise_after is not None and self.calls > self._raise_after:
            raise RuntimeError("rank 1 stopped answering (test)")
        hexes = list((kwargs or {}).get("keys") or ())
        self.seen.append(len(hexes))
        assert timeout is not None, "the boot RPC must never pass timeout=None"
        out = []
        for r in range(self._ranks):
            out.append({"rank": r,
                        "missing": [h for h in hexes if h in self._missing],
                        "wrong_size": [h for h in hexes if h in self._wrong],
                        "checked": len(hexes), "free_bytes": 10 ** 12})
        return out


def boot_with(tmp, keys, core, **fl):
    """Build a manager whose _load_manifest runs with `core` on the stack."""
    with open(os.path.join(tmp, "presence.json"), "w") as fh:
        json.dump({"keys": [[bytes(k).hex(), i] for i, k in enumerate(keys)]}, fh)
    F.write(disk_restore=1, disk_verify_enabled=1, **fl)
    KF.reset_for_tests()
    # The real capture walks caller frames for a `self` with collective_rpc and
    # a model_executor. A method on `core` puts exactly that on the stack, so
    # the capture is exercised for real rather than monkeypatched away -- which
    # is what makes the "no engine core" case below a genuine discriminator.
    return core.build(tmp)


tmp = tempfile.mkdtemp(prefix="kv9-boot-")
BK = [K(bytes([i]) * 32, 0) for i in range(1, 7)]
for k in BK:
    touch(tmp, k)
core = FakeCore()
m = boot_with(tmp, BK, core)
eq("*** the boot asked the ranks, not just this node ***", core.calls >= 1, True)
eq("...and kept every key they vouched for", m._stats["manifest_loaded"], len(BK))
eq("...pruning none", m._stats["manifest_pruned"], 0)
check("*** every survivor is BORN VERIFIED, so no lookup can defer ***",
      all(bytes(k) in m._verified_set() for k in BK))
eq("...and a lookup answers at once, with no deferral",
   m.lookup(BK[0], Ctx()), True)
eq("...nothing was queued", len(m._wanted_map()), 0)
check("the boot names HOW it checked, and it says all-ranks",
      (getattr(m, "_boot_how", "") or "").startswith("all-ranks"),
      getattr(m, "_boot_how", ""))
check("...and it is published on /kv, not only logged",
      (m._verify_snapshot().get("boot_check") or "").startswith("all-ranks"))

print()
print("== (9y) a file missing on ONE rank is pruned AT BOOT ==")
tmp = tempfile.mkdtemp(prefix="kv9-boot2-")
for k in BK:
    touch(tmp, k)          # present on THIS node -- the old walk would keep it
core = FakeCore(missing=[BK[2]])
m = boot_with(tmp, BK, core)
eq("*** the key whose file is gone on the OTHER node is PRUNED ***",
   m._stats["manifest_pruned"], 1)
check("...even though it exists on this one, which the old walk could not see",
      os.path.exists(os.path.join(tmp, bytes(BK[2]).hex() + ".bin")))
check("...so it cannot be claimed", bytes(BK[2]) not in m._present)
eq("...and lookup misses it rather than loading into an assert",
   m.lookup(BK[2], Ctx()), False)
eq("...the drop is counted", m._stats.get("boot_verify_dropped"), 1)

print()
print("== (9z) a WRONG SIZE is pruned at boot too ==")
tmp = tempfile.mkdtemp(prefix="kv9-boot3-")
for k in BK:
    touch(tmp, k)
core = FakeCore(wrong=[BK[1]])
m = boot_with(tmp, BK, core)
eq("*** a truncated or foreign-sized file is as bad as absent ***",
   m._stats["manifest_pruned"], 1)
check("...and it is the wrong-sized one", bytes(BK[1]) not in m._present)

print()
print("== (9aa) the FALLBACK is the old walk, and it is NAMED ==")
# *** NEVER "TRUST THE INDEX". *** If the ranks cannot be asked, the honest
# degradation is the weaker check, and the log and /kv must say which happened
# -- the old log line claimed "can only be checked against this one" as though
# it were the only possibility, which is false once the ranks are asked.
tmp = tempfile.mkdtemp(prefix="kv9-boot4-")
for k in BK[:4]:
    touch(tmp, k)          # BK[4], BK[5] have no file on this node
with open(os.path.join(tmp, "presence.json"), "w") as fh:
    json.dump({"keys": [[bytes(k).hex(), i] for i, k in enumerate(BK)]}, fh)
F.write(disk_restore=1, disk_verify_enabled=1)
KF.reset_for_tests()
m = mkmgr(tmp)             # NO FakeCore on the stack -> capture must fail
eq("*** with no engine core reachable it still prunes, rank-locally ***",
   m._stats["manifest_loaded"], 4)
eq("...dropping the two whose own file is gone", m._stats["manifest_pruned"], 2)
check("*** and it SAYS it was head-only rather than implying a guarantee ***",
      (getattr(m, "_boot_how", "") or "").startswith("head-only"),
      getattr(m, "_boot_how", ""))
check("...so nothing is marked verified on that path",
      len(m._verified_set()) == 0)

print()
print("== (9ab) a mid-way failure keeps the ALL-RANKS verdicts already earned ==")
tmp = tempfile.mkdtemp(prefix="kv9-boot5-")
MK = [K(bytes([i]) * 32, 0) for i in range(1, 13)]
for k in MK:
    touch(tmp, k)
core = FakeCore(raise_after=1)
m = boot_with(tmp, MK, core, disk_boot_verify_chunk=4)
eq("every key survives (the files are all here)",
   m._stats["manifest_loaded"], len(MK))
check("*** the first chunk's keys keep their all-ranks verdict ***",
      len(m._verified_set()) == 4, f"{len(m._verified_set())} verified")
check("...and the rest are NOT claimed as verified",
      not all(bytes(k) in m._verified_set() for k in MK))
check("...the fallback is recorded", m._stats.get("boot_verify_fell_back") == 1)
check("*** and `how` names both halves rather than rounding to one ***",
      "all-ranks for 4" in (getattr(m, "_boot_how", "") or "")
      and "head-only" in (getattr(m, "_boot_how", "") or ""),
      getattr(m, "_boot_how", ""))

print()
print("== (9ab2) A RANK THAT DOES NOT ANSWER MUST NOT READ AS A RANK SAYING YES ==")
# *** THIS HOLE WAS FOUND BY BREAK-VERIFICATION, NOT BY WRITING THE TEST. ***
# A mutation that ignored `verdict["fault"]` in the boot loop was NOT CAUGHT:
# with a short reply list the boot would have kept every key on the strength of
# having failed to ask one of the ranks, which is the manufactured zero this
# repo keeps paying for -- and at boot it would have been printed as
# "all-ranks", a guarantee that had not been made.
tmp = tempfile.mkdtemp(prefix="kv9-boot-fault-")
for k in BK[:4]:
    touch(tmp, k)          # all four exist HERE
with open(os.path.join(tmp, "presence.json"), "w") as fh:
    json.dump({"keys": [[bytes(k).hex(), i] for i, k in enumerate(BK)]}, fh)
F.write(disk_restore=1, disk_verify_enabled=1)
KF.reset_for_tests()
core = FakeCore(ranks=1)   # one reply for a two-rank cluster
m = core.build(tmp)
check("*** a short reply list makes the boot FALL BACK, not accept ***",
      m._stats.get("boot_verify_fell_back") == 1,
      "fell_back=%r" % (m._stats.get("boot_verify_fell_back"),))
check("...and `how` does NOT claim all-ranks",
      "head-only" in (getattr(m, "_boot_how", "") or ""),
      getattr(m, "_boot_how", ""))
check("...nothing is marked verified on the strength of the short reply",
      len(m._verified_set()) == 0, f"{len(m._verified_set())} verified")
eq("...and the rank-local walk still did its job",
   m._stats["manifest_loaded"], 4)
eq("...pruning the two with no file here", m._stats["manifest_pruned"], 2)

print()
print("== (9ac) the boot RPC is chunked and never passes timeout=None ==")
tmp = tempfile.mkdtemp(prefix="kv9-boot6-")
CK = [K(bytes([i]) * 32, 0) for i in range(1, 11)]
for k in CK:
    touch(tmp, k)
core = FakeCore()
m = boot_with(tmp, CK, core, disk_boot_verify_chunk=3, disk_verify_chunk=99)
eq("*** the key list is chunked, so no single message is large ***",
   core.seen, [3, 3, 3, 1])
# *** AND IT IS THE BOOT'S OWN KNOB, NOT THE RUNTIME PUMP'S. *** A chunk is a
# blocking RPC and therefore a step stall: at 4,000 keys that is ~1,054ms warm,
# already past test-meter section 1's 1.0s gate, while at boot nothing is being
# served and 40,000 is measurably faster (3,794 -> 7,998 keys/s, A/B/A). One
# value cannot be right for both, so the test pins WHICH knob the boot reads --
# setting disk_verify_chunk to 99 above would have produced chunks of 99 if the
# boot read the wrong one.
eq("...the runtime knob did NOT drive the boot", 99 in core.seen, False)
eq("...and the RPC count is recorded", m._stats.get("boot_verify_rpcs"), 4)
# FakeCore.collective_rpc asserts timeout is not None, so reaching here at all
# is the proof -- a boot RPC with timeout=None is a wait with no end, and at
# boot there is no NCCL watchdog to break it.
check("*** the boot RPC carried a timeout (FakeCore asserts it) ***", True)
# *** THE RUNTIME RPC RATE IS ZERO BY DEFAULT, AND THAT IS PINNED HERE. ***
# 2026-10-12 17:37:35: Xid 32 twice on the head and an NCCL kill, on the first
# boot with runtime verification RPCs enabled through a 700K restore. NOT
# proven -- the previous boot ran the same sweep through the same restore with
# Xid unchanged -- but this repo's own comment in `_enforce_budget` says "a
# collective_rpc from inside the scheduler step is precisely the
# blocking-the-critical-path mistake that has already cost this project an NCCL
# kill", and I overrode it. The BOOT check does all the safety work before any
# transfer exists, so OFF costs nothing. These two knobs are the only things
# that issue a step RPC while serving, so the assertion is on both.
eq("*** the background sweep is OFF by default (no step RPCs while serving) ***",
   float(KF.DEFAULTS["disk_verify_bg_per_s"]), 0.0)
eq("*** and so is the worker free-space keep-alive, for the same reason ***",
   float(KF.DEFAULTS["disk_free_rpc_s"]), 0.0)
check("*** so the pump issues NOTHING on a default config with a clean index ***",
      float(KF.DEFAULTS["disk_verify_bg_per_s"]) == 0.0
      and float(KF.DEFAULTS["disk_free_rpc_s"]) == 0.0
      and float(KF.DEFAULTS["disk_verify_defer_s"]) == 0.0)
check("*** the boot chunk is 10x the runtime one, which is the measurement ***",
      int(KF.DEFAULTS["disk_boot_verify_chunk"])
      >= 10 * int(KF.DEFAULTS["disk_verify_chunk"]))
check("BOOT_VERIFY_TIMEOUT_S is generous and a CODE constant",
      D.BOOT_VERIFY_TIMEOUT_S >= 120.0
      and "disk_boot_verify_timeout_s" not in KF.DEFAULTS)

print()
print("== (9ad) the knob still turns the whole thing off ==")
tmp = tempfile.mkdtemp(prefix="kv9-boot7-")
for k in BK[:3]:
    touch(tmp, k)
with open(os.path.join(tmp, "presence.json"), "w") as fh:
    json.dump({"keys": [[bytes(k).hex(), i] for i, k in enumerate(BK[:3])]}, fh)
F.write(disk_restore=1, disk_verify_enabled=0)
KF.reset_for_tests()
core = FakeCore()
m = core.build(tmp)
eq("with the knob off the ranks are NOT asked", core.calls, 0)
check("...and `how` says why", "disk_verify_enabled is off"
      in (getattr(m, "_boot_how", "") or ""), getattr(m, "_boot_how", ""))
eq("...the index still loads", m._stats["manifest_loaded"], 3)

print()
print("== (9ae) POST /kv/verify: a cross-thread handoff with ONE writer ==")
# The route was structurally dead on :8888 -- the router is built in the API
# server and the index is in the engine core, so every call 409'd. It lives on
# the :9199 listener now, which IS the engine core, and that puts the request
# on a DAEMON THREAD. `_wanted` and `_verified` are written by `lookup` and
# `pump` on the scheduler's thread, so the listener must not touch them.
tmp = tempfile.mkdtemp(prefix="kv9-req-")
RK = [K(bytes([i]) * 32, 0) for i in range(10, 20)]
for k in RK:
    touch(tmp, k)
F.write(disk_restore=1, disk_verify_enabled=1, disk_verify_chunk=3,
        disk_verify_defer_s=0.0, disk_verify_bg_per_s=0.0, disk_free_rpc_s=0.0)
KF.reset_for_tests()
D._VERIFY_REQUEST = None
D._VERIFY_LAST = None
D._VERIFY_FAULT = None
m = restored(tmp, RK)
eq("a restored index starts unverified", len(m._verified_set()), 0)

r1 = D.request_verify(requeue_all=True)
eq("*** the listener touches NO manager state -- nothing is queued yet ***",
   len(m._wanted_map()), 0)
check("...it only parks the request", D._VERIFY_REQUEST is not None)
eq("...and reports what it accepted", (r1["all"], r1["clear_fault"]),
   (True, False))
check("...with a seq", isinstance(r1.get("seq"), int))
r2 = D.request_verify(clear_fault=True)
check("*** a seq is unique, so a caller cannot match somebody else's receipt",
      r2["seq"] != r1["seq"], f"{r1['seq']} vs {r2['seq']}")

# The pump is the single writer, and it must take the request even on a round
# with no keys and no RPC due -- otherwise an idle-ish engine never applies one.
D._VERIFY_REQUEST = None
D._VERIFY_LAST = None
D._VERIFY_FAULT = {"reason": "rank 1 did not answer (test)", "at": 0.0}
core = FakeCore()
D.request_verify(clear_fault=True)
m.pump(core)
check("*** the pump applies a request on a round that does nothing else ***",
      D._VERIFY_FAULT is None)
eq("...and issued NO rpc to do it", core.calls, 0)
check("...recording what it cleared, so the 202 is checkable",
      (D._VERIFY_LAST or {}).get("fault_was", {}).get("reason", "")
      == "rank 1 did not answer (test)", D._VERIFY_LAST)
check("...and the request is consumed", D._VERIFY_REQUEST is None)

# Taken ONCE. A request re-applied every step would re-queue the whole index
# forever, which is a permanent stall rather than an operator action.
# *** THE SEQ IS NOT A DISCRIMINATOR: re-applying the SAME request writes the
# *** same seq, so comparing it passes however broken the consume is. The
# receipt's `applied_at` is what moves, which is the mutation-caught version.
applied_was = (D._VERIFY_LAST or {}).get("applied_at")
m.pump(core)
eq("*** a consumed request is NOT re-applied on the next step ***",
   (D._VERIFY_LAST or {}).get("applied_at"), applied_was)

print()
print("== (9af) `all` re-queues, is capped, and does NOT unverify ==")
D._VERIFY_LAST = None
# *** RK[0] IS INSIDE THE CHUNK, so a `_verified.clear()` is undone by the
# *** same pump call and the obvious assertion cannot see it -- measured, as a
# surviving mutation. RK[9] is past the chunk, so if the re-queue ever clears
# the vouching, this is what notices.
m._verified_set().add(bytes(RK[0]))
m._verified_set().add(bytes(RK[9]))
D.request_verify(requeue_all=True)
m.pump(core)
last = D._VERIFY_LAST or {}
eq("every indexed key is queued", last.get("requeued"), len(RK))
eq("...and `capped` says it was not truncated", last.get("capped"), False)
check("*** `_verified` is NOT cleared -- a re-check must not be a cache "
      "outage ***", bytes(RK[9]) in m._verified_set(),
      "RK[9] is beyond the chunk, so only the re-queue could have dropped it")
eq("...and the key inside the chunk is vouched for too",
   bytes(RK[0]) in m._verified_set(), True)
eq("...so an already-vouched-for key still answers at once",
   m.lookup(RK[0], Ctx()), True)
# RK[1] is inside the chunk the pump just verified, so the discriminator has
# to be a key the chunk did not reach -- otherwise this passes for the wrong
# reason and would keep passing with the re-queue removed.
# RK[8]: past the chunk AND not one of the two vouched for above, so it is
# the only key here whose answer is decided by the verification alone.
check("...while a key the chunk has not reached yet does NOT claim presence",
      m.lookup(RK[8], Ctx()) in (None, False))
check("...and the pump then actually checked a chunk", core.calls >= 1)

# *** THE FILES MUST EXIST, or `_load_manifest` prunes every key and this
# *** tests an EMPTY index -- which reports `requeued: 0` and looks like a
# broken re-queue. The fixture has to be able to express the failure.
tmp2 = tempfile.mkdtemp(prefix="kv9-cap-")
for k in RK:
    touch(tmp2, k)
m2 = restored(tmp2, RK)
eq("the capped fixture really holds the index", len(m2._present), len(RK))
m2.VERIFY_WANTED_MAX = 4
D._VERIFY_LAST = None
D.request_verify(requeue_all=True)
m2.pump(FakeCore())
last = D._VERIFY_LAST or {}
# *** READ THE ENQUEUE, NOT THE QUEUE: the pump drains a chunk in the same
# *** call that applies the request, so the queue alone is already short by
# `disk_verify_chunk`. Queued-plus-verified is what the cap actually bounds.
eq("the re-queue is bounded by VERIFY_WANTED_MAX",
   len(m2._wanted_map()) + len(m2._verified_set()), 4)
eq("...and the pump drained part of it in the same round",
   len(m2._wanted_map()) < 4, True)
check("...and SAYS it was capped, rather than looking complete",
      last.get("capped") is True, last)

print()
print("== (9ag) a request that raises is DROPPED, not retried forever ==")
# The hot-path trap this repo records twice: a one-shot that raises becomes a
# retry on every step. The flag is taken before the work for that reason.
class _Boom:
    def __getattr__(self, n):
        raise RuntimeError("wanted map unavailable (test)")

tmp3 = tempfile.mkdtemp(prefix="kv9-boom-")
for k in RK:
    touch(tmp3, k)
m3 = restored(tmp3, RK)
eq("the raising fixture really holds the index", len(m3._present), len(RK))
m3._wanted = _Boom()  # the real attribute `_wanted_map` returns
D._VERIFY_REQUEST = None
D._VERIFY_LAST = None
D.request_verify(requeue_all=True)
m3.pump(FakeCore())
check("the failure is recorded rather than swallowed",
      (D._VERIFY_LAST or {}).get("ok") is False, D._VERIFY_LAST)
check("*** and the request is GONE, so it cannot raise every step ***",
      D._VERIFY_REQUEST is None)

print()
print("== (9ah) the snapshot makes a 202 falsifiable ==")
snap = m._verify_snapshot()
for key in ("last_request", "pending_request", "fault"):
    check(f"`{key}` is present even when there is nothing to say -- absent "
          f"must not read as fine", key in snap)
D._VERIFY_REQUEST = None
D._VERIFY_LAST = None
snap = m._verify_snapshot()
eq("...null rather than missing, with nothing applied",
   (snap["last_request"], snap["pending_request"]), (None, None))
D.request_verify(requeue_all=True)
check("...and a parked request is visible as pending, so an IDLE engine's "
      "unapplied request can be told from a lost one",
      m._verify_snapshot()["pending_request"] is not None)
m.pump(FakeCore())

print()
print("== (9ai) the one-writer discipline, enforced structurally ==")
# A lint, because the property is "only these two functions assign it" and a
# test of behaviour cannot see a third writer somebody adds later.
DS = code_only(D.__file__)
import re as _re
writers = []
for mo in _re.finditer(r"^\s*_VERIFY_REQUEST\s*=", DS, _re.M):
    fn = None
    for fmo in _re.finditer(r"^(\s*)def (\w+)", DS[:mo.start()], _re.M):
        fn = fmo.group(2)
    writers.append(fn)
eq("*** `_VERIFY_REQUEST` is assigned in exactly two places ***",
   len(writers), 2)
eq("...the listener's hand-off and the scheduler's take, and nothing else",
   sorted(set(writers)), ["_apply_verify_request", "request_verify"])
check("the lint can see assignments at all (canary)",
      "_VERIFY_REQUEST" in DS)

SV = code_only(os.path.join(os.path.dirname(D.__file__), "serve.py"))
post = SV.split("def do_POST", 1)
eq("the listener has a POST handler", len(post), 2)
body = post[1].split("\ndef ", 1)[0]
for forbidden in ("_wanted_map", "_verified_set", "_present", "_forget",
                  "_journal", "pump("):
    check(f"*** the listener thread does not touch `{forbidden}` ***",
          forbidden not in body)
check("...it hands over through request_verify and nothing else",
      "request_verify" in body)
check("...and reads the manager only to refuse when it is absent",
      "_LIVE_MANAGER" in body)
# A strings-KEPT view, because the header name is a string literal and the
# default view blanks those -- the same two-view split the dict-lookup checks
# above need. Asserting on the blanked view would pass vacuously.
SV_STR = code_only(os.path.join(os.path.dirname(D.__file__), "serve.py"),
                   blank_strings=False)
body_str = SV_STR.split("def do_POST", 1)[1].split("\ndef ", 1)[0]
check("the body is bounded, so a lever cannot make the core allocate",
      "> self.MAX_BODY" in body and "Content-Length" in body_str,
      "the bare name survives in the 413 payload, so the TEST must be the "
      "comparison -- a surviving mutation taught this")
check("...and over-long bodies are refused rather than truncated",
      "413" in body_str)
check("the POST lint is looking at real code (canary)", "do_POST" in SV)

print()
print("== (9aj) the dead :8888 route is GONE, and named where it went ==")
CT = os.path.join(os.path.dirname(D.__file__), "control.py")
# *** STRINGS KEPT, DELIBERATELY. *** The route path is a string literal, so
# the default blanked view cannot see it and this lint passed vacuously --
# found by re-adding the route and watching the suite stay green. Comments are
# still blanked, which is what lets the explanatory comment below mention the
# path without becoming the failure.
ct_code = code_only(CT, blank_strings=False)
check("*** no router.post for /kv/verify on the API server's app ***",
      '"/kv/verify"' not in ct_code,
      "the :8888 route answers 409 by construction -- the index is elsewhere")
check("...the lint can see sibling routes, so it is looking at real code",
      '"/kv/knob"' in ct_code)
ct_raw = open(CT).read()
check("...but the reason survives as a comment, rather than vanishing",
      "/kv/verify" in ct_raw and "9199" in ct_raw)
# The TABLE itself, not a string search of whichever file holds it (it moved
# to kvmgr.diag.reload when hot reload became an opt-in diagnostic).
import importlib as _il9  # noqa: E402

_pres9 = {(m, a) for m, a, _ in _il9.import_module("kvmgr.diag.reload")._PRESERVE}
for g in ("_VERIFY_REQUEST", "_VERIFY_LAST", "_VERIFY_SEQ"):
    check(f"`{g}` is in _PRESERVE -- a reload between the hand-off and the "
          f"take would lose a 202'd request", ("kvmgr.diskspec", g) in _pres9)

print()
print(f"{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)
