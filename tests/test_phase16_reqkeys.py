"""Phase 16: GET /kv/keys -- one request's disk files (kvmgr.diag.reqkeys, ask6).

For the v0.4.6 recompute gate, which has to corrupt one file of a restored
prefix on the worker only. Read-only, bounded, and an unknown request says so.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
from kvmgr import hooks  # noqa: E402
from kvmgr.diag import reqkeys as RK  # noqa: E402

K = base.make_offload_key
FileMapper = sys.modules["vllm.v1.kv_offload.file_mapper"].FileMapper
F = harness.Flags(os.path.join(tempfile.mkdtemp(prefix="kv16-"), "flags.json"))

root = tempfile.mkdtemp(prefix="kv16-root-")
mapper0 = FileMapper(root_dir=root, model_name="m", tokens_per_hash=4,
                     blocks_per_file=1, tp_size=2, pp_size=1, pcp_size=1,
                     dcp_size=1, rank=0, dtype="nvfp4_ds_mla")


def gkeys(group, n, salt):
    return [K(bytes([salt, group, i]) * 10 + b"\x00\x00", group) for i in range(n)]


G0, G1 = gkeys(0, 4, 1), gkeys(1, 6, 1)
Cfg = types.SimpleNamespace
osched = types.SimpleNamespace(
    config=Cfg(kv_group_configs=[Cfg(group_idx=0, offloaded_block_size=256),
                                 Cfg(group_idx=1, offloaded_block_size=64)]),
    _req_status={})


def state(rid, ntok, groups):
    return types.SimpleNamespace(
        req=types.SimpleNamespace(request_id=rid, num_tokens=ntok),
        group_states=tuple(types.SimpleNamespace(offload_keys=list(k),
                                                 num_hit_blocks=h)
                           for k, h in groups))


osched._req_status["canary"] = state("canary", 1000, [(G0, 1), (G1, 0)])
D._LIVE_OFFLOAD_SCHED[:] = [osched]

F.write(disk_restore=1, disk_verify_enabled=0)
KF.reset_for_tests()
mgr = D.DiskPresenceManager(mapper0, num_workers=2,
                            manifest_path=os.path.join(root, "presence.json"))
mgr._remember(G0[1], 0)
mgr._remember(G0[2], 0)
mgr._verified_set().add(bytes(G0[1]))
D._LIVE_MANAGER = mgr

print("== (16a) a live request: every group, in prompt order ==")
code, body = RK.http_get("/kv/keys", {"req": ["canary"]})
eq("200", code, 200)
eq("source live", body["source"], "live")
eq("both groups", [g["group"] for g in body["groups"]], [0, 1])
g0 = body["groups"][0]
eq("group 0: keys in prompt order, as hex",
   [r["key"] for r in g0["keys"]], [bytes(k).hex() for k in G0])
eq("...with the block index and its token span",
   [(r["i"], r["tokens"]) for r in g0["keys"]][:2], [(0, [0, 256]), (1, [256, 512])])
eq("...and how many blocks the request hit at start", g0["num_hit_blocks"], 1)
eq("*** in_index and verified are the index's own answer ***",
   [(r["in_index"], r["verified"]) for r in g0["keys"]],
   [(False, False), (True, True), (True, False), (False, False)])
_p = g0["keys"][2]["path"]
eq("*** the path on rank 0 is upstream FileMapper's own ***",
   _p["0"], mapper0.get_file_name(G0[2]))
_m1 = FileMapper(root_dir=root, model_name="m", tokens_per_hash=4,
                 blocks_per_file=1, tp_size=2, pp_size=1, pcp_size=1,
                 dcp_size=1, rank=1, dtype="nvfp4_ds_mla")
eq("*** ...and rank 1's is what rank 1's own FileMapper says ***",
   _p["1"], _m1.get_file_name(G0[2]))
check("...the two differ only in the rank directory",
      _p["0"].replace("_r0/", "_r1/") == _p["1"], _p)
eq("the manager's own mapper is left at rank 0", mgr._mapper.rank, 0)
check("the answer serialises as JSON", json.dumps(body) is not None)

print("\n== (16b) filters ==")
_, b = RK.http_get("/kv/keys", {"req": ["canary"], "group": ["1"], "limit": ["2"]})
eq("group filter", [g["group"] for g in b["groups"]], [1])
eq("limit", len(b["groups"][0]["keys"]), 2)
eq("...and says it truncated, with the real count",
   (b["groups"][0]["truncated"], b["groups"][0]["num_keys"]), (True, 6))
eq("bad integers are a 400", RK.http_get("/kv/keys", {"req": ["canary"],
                                                      "group": ["x"]})[0], 400)

print("\n== (16c) a finished request is kept in the ring; unknown is a 404 ==")
RK.capture(osched, "canary")
del osched._req_status["canary"]
code, b = RK.http_get("/kv/keys", {"req": ["canary"]})
eq("*** after it finishes: still answered, from the ring ***",
   (code, b["source"]), (200, "recent"))
eq("...with the same keys", [r["key"] for r in b["groups"][0]["keys"]],
   [bytes(k).hex() for k in G0])
code, b = RK.http_get("/kv/keys", {"req": ["nope"]})
eq("*** an unknown request is a 404, never an empty 200 ***", code, 404)
check("...that names what IS known", "canary" in b["known"], b)
code, b = RK.http_get("/kv/keys", {})
check("no req: the list of known requests, with token counts",
      code == 200 and b["requests"][0]["req"] == "canary"
      and b["requests"][0]["num_tokens"] == 1000, b)
eq("any other path is not ours", RK.http_get("/kv/other", {}), None)

print("\n== (16d) bounded ==")
RK._RING.clear()
_old = (RK.RING_REQUESTS, RK.RING_KEYS)
RK.RING_REQUESTS, RK.RING_KEYS = 3, 25
for n in range(5):
    rid = "r%d" % n
    osched._req_status[rid] = state(rid, 10, [(gkeys(0, 4, 10 + n), 0),
                                              (gkeys(1, 4, 10 + n), 0)])
    RK.capture(osched, rid)
    del osched._req_status[rid]
eq("*** at most RING_REQUESTS kept, oldest out first ***", list(RK._RING),
   ["r2", "r3", "r4"])
RK.capture(osched, "gone")
eq("capturing a request with no state does nothing", len(RK._RING), 3)
osched._req_status["big"] = state("big", 10, [(gkeys(0, 20, 50), 0),
                                              (gkeys(1, 4, 50), 0)])
RK.capture(osched, "big")
eq("*** and at most RING_KEYS keys between them ***", list(RK._RING), ["big"])
RK.RING_REQUESTS, RK.RING_KEYS = _old
del osched._req_status["big"]

print("\n== (16e) no manager: nulls with a reason, never False ==")
D._LIVE_MANAGER = None
osched._req_status["x"] = state("x", 10, [(G0[:1], 0), ([], 0)])
_, b = RK.http_get("/kv/keys", {"req": ["x"]})
r0 = b["groups"][0]["keys"][0]
eq("in_index / verified / path are None", (r0["in_index"], r0["verified"],
                                           r0["path"]), (None, None, None))
check("...and the reply says why", "no disk manager" in b.get("why", ""), b)
D._LIVE_MANAGER = mgr

print("\n== (16f) the capture wrap runs before upstream drops the request ==")
_osm = types.ModuleType("vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler")


class OffloadingConnectorScheduler:
    def __init__(self):
        self.config = osched.config
        self._req_status = {}

    def request_finished(self, request):
        self._req_status.pop(request.request_id, None)    # what upstream does
        return False, None


class Sched031(OffloadingConnectorScheduler):
    """0.31's shape: the final key is appended and the state is kept."""

    def request_finished(self, request):
        st = self._req_status[request.request_id]
        st.group_states[0].offload_keys.append(FINAL)
        return False, None


FINAL = K(b"\x7f" * 30 + b"\x00\x00", 0)
_osm.OffloadingConnectorScheduler = OffloadingConnectorScheduler
_pkg = "vllm.distributed.kv_transfer.kv_connector.v1.offloading"
for _n in ("vllm.distributed", "vllm.distributed.kv_transfer",
           "vllm.distributed.kv_transfer.kv_connector",
           "vllm.distributed.kv_transfer.kv_connector.v1", _pkg):
    sys.modules.setdefault(_n, types.ModuleType(_n))
sys.modules[_pkg].scheduler = _osm
sys.modules[_pkg + ".scheduler"] = _osm
RK._install_capture()
RK._install_capture()
s = OffloadingConnectorScheduler()
s._req_status["fin"] = state("fin", 77, [(G0, 0), (G1, 0)])
eq("upstream's return value is passed through",
   s.request_finished(types.SimpleNamespace(request_id="fin")), (False, None))
check("*** the finished request's keys are in the ring ***", "fin" in RK._RING)
eq("...and upstream still dropped its state", s._req_status, {})
_cls031 = Sched031          # its own, unwrapped request_finished
_osm.OffloadingConnectorScheduler = _cls031
RK._install_capture()
s31 = _cls031()
s31._req_status["fin31"] = state("fin31", 99, [(list(G0), 0), (G1, 0)])
s31.request_finished(types.SimpleNamespace(request_id="fin31"))
check("*** 0.31 shape: the ring holds the FINAL key appended inside "
      "request_finished ***",
      RK._RING["fin31"]["groups"][0]["keys"][-1] == FINAL,
      RK._RING["fin31"]["groups"][0]["keys"][-1:])

_up = open(os.path.join(REPO, "tests", "upstream", "vllm",
                        "kv_connector_offloading", "scheduler.py")).read()
_rf = _up[_up.index("    def request_finished("):]
_rf = _rf[:_rf.index("\n    def ", 10)]
check("UPSTREAM'S SHAPE (0.31): request_finished keeps _req_status alive and "
      "appends the final key first, which is why the capture also runs AFTER it",
      "req_status.update_offload_keys()" in _rf
      and "del self._req_status[request.request_id]" not in _rf)
check("...and the per-group key list is RequestGroupState.offload_keys",
      "offload_keys: list[OffloadKey]" in _up)

_e31 = RK.entry_of(
    types.SimpleNamespace(config=Cfg(kv_group_configs=[
        Cfg(group_idx=0, tokens_per_chunk=256)])),
    types.SimpleNamespace(req=types.SimpleNamespace(request_id="c", num_tokens=5),
                          group_states=(types.SimpleNamespace(
                              offload_keys=list(G0), num_hit_chunks=2),)))
eq("*** 0.31 field names: tokens_per_chunk and num_hit_chunks are read ***",
   (_e31["groups"][0]["block_tokens"], _e31["groups"][0]["num_hit_blocks"]),
   (256, 2))
_eNo = RK.entry_of(types.SimpleNamespace(config=Cfg(kv_group_configs=[Cfg()])),
                   types.SimpleNamespace(req=types.SimpleNamespace(
                       request_id="d", num_tokens=5), group_states=(
                       types.SimpleNamespace(offload_keys=[]),)))
eq("...and neither present is None, not 0",
   (_eNo["groups"][0]["block_tokens"], _eNo["groups"][0]["num_hit_blocks"]),
   (None, None))

print("\n== (16g) the route, and the ledger's keys ==")
hooks.register("http_get", RK.http_get)
from kvmgr import serve  # noqa: E402
_ssrc = open(os.path.join(REPO, "src", "kvmgr", "serve.py")).read()
check("serve.py routes unknown GETs through the http_get hook",
      'hooks.first("http_get", path,' in _ssrc)
from kvmgr.diag import ledger as LG  # noqa: E402
_h = types.SimpleNamespace(_submit_keys=G0[:2])
F.write(diag_ledger_keys=0)
KF.reset_for_tests()
eq("diag_ledger_keys off: no keys field", LG._keys_field(_h), {})
F.write(diag_ledger_keys=1)
KF.reset_for_tests()
eq("*** on: the round's keys as hex ***", LG._keys_field(_h),
   {"keys": [bytes(k).hex() for k in G0[:2]]})
eq("...and a handler that named none says so, not []",
   LG._keys_field(types.SimpleNamespace())["keys"], None)
_dsrc = open(os.path.join(REPO, "src", "kvmgr", "diskio.py")).read()
check("the store names its round's keys before the ledger line",
      _dsrc.index("self._submit_keys = _sk")
      < _dsrc.index('self._ledger_submit("store"'))
check("...and the load does too, from the read's future",
      _dsrc.index('self._submit_keys = getattr(fut, "_kvmgr_keys", None)')
      < _dsrc.index('self._ledger_submit("load"'))
F.write()
KF.reset_for_tests()

print()
print(f"{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print(f"  FAILED: {f}")
sys.exit(1 if FAIL else 0)
