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


_dspath = os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr", "diskspec.py")
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
for n in ("disk_verify_enabled", "disk_verify_chunk", "disk_verify_defer_s",
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
check("*** disk_verify_enabled defaults OFF -- the trigger is not ready ***",
      KF.DEFAULTS["disk_verify_enabled"] is False)
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
_csrc = open(os.path.join(REPO, "patches", "kvmgr_plugin", "kvmgr",
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
print(f"{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)
