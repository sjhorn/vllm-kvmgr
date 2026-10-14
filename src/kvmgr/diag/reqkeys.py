"""kvmgr.diag.reqkeys -- which disk files belong to ONE request (ask6).

For the v0.4.6 recompute gate (vllm-proxy `docs/kv/gates-v0.4.6.md`, Gate B),
which corrupts one file of a restored prefix on the worker only. Nothing else
maps a request to its keys:

  * the store path writes only keys NOT already in the index
    (`prepare_store` filters them), so re-priming a prompt whose blocks are on
    disk writes no store line at all -- the ledger cannot be the source;
  * recomputing block hashes outside the engine would be a second
    implementation of PYTHONHASHSEED-dependent hashing.

The source is upstream's own per-request state in the engine core:
`OffloadingConnectorScheduler._req_status[req].group_states[g].offload_keys`,
the full ordered key list per KV group, in prompt order. While the request is
alive it is read live; when it finishes, `request_finished` (wrapped here; it
runs before upstream drops the state) copies the lists into a bounded ring.

`GET /kv/keys` on the engine-core stats port (kvmgr_stats_port):

  /kv/keys                        the recent and live requests: id, tokens, keys
  /kv/keys?req=<id>[&group=G][&limit=N]
      per group, each key in prompt order: block index `i`, token span,
      key hex, its file on EVERY rank (upstream FileMapper.get_file_name with
      the rank swapped -- not a reimplementation; container paths, under
      /kvoffload), and whether the index holds and has verified it.
      An unknown or evicted request is a 404 that says so and lists what is
      known -- never an empty answer.

Read-only: no KV bytes, no prompt text, nothing written. Bounded: at most
RING_REQUESTS finished requests and RING_KEYS keys between them, oldest out
first. The capture is a list copy of key references, on the scheduler thread,
once per finished request.
"""

from __future__ import annotations

import copy
import threading
import time
from collections import OrderedDict

from vllm.logger import init_logger

from .. import hooks

logger = init_logger("vllm." + __name__)

RING_REQUESTS = 32
RING_KEYS = 400_000
LIMIT_DEFAULT = 5000

_RING: "OrderedDict[str, dict]" = OrderedDict()
_LOCK = threading.Lock()
_STATS = {"installed": False, "captured": 0, "evicted": 0, "capture_errors": 0}


def _osched():
    from .. import diskspec
    return diskspec._LIVE_OFFLOAD_SCHED[-1] if diskspec._LIVE_OFFLOAD_SCHED else None


def entry_of(osched, req_status) -> dict:
    """The keys of one request, per group. Plain types, copied lists."""
    req = req_status.req
    cfgs = list(getattr(getattr(osched, "config", None), "kv_group_configs", ()) or ())
    groups = []
    for gi, gs in enumerate(req_status.group_states):
        cfg = cfgs[gi] if gi < len(cfgs) else None
        bt = getattr(cfg, "offloaded_block_size", None)
        groups.append({
            "group": int(getattr(cfg, "group_idx", gi)),
            "block_tokens": int(bt) if bt is not None else None,
            "num_hit_blocks": int(getattr(gs, "num_hit_blocks", 0) or 0),
            "keys": list(gs.offload_keys),
        })
    return {"req": str(req.request_id), "num_tokens": int(req.num_tokens),
            "at": time.time(), "groups": groups}


def _nkeys(e) -> int:
    return sum(len(g["keys"]) for g in e["groups"])


def capture(osched, req_id) -> None:
    """Copy a finishing request's keys into the ring. Never raises."""
    try:
        st = getattr(osched, "_req_status", {}).get(req_id)
        if st is None:
            return
        e = entry_of(osched, st)
        with _LOCK:
            _RING.pop(e["req"], None)
            _RING[e["req"]] = e
            _STATS["captured"] += 1
            total = sum(_nkeys(x) for x in _RING.values())
            while _RING and (len(_RING) > RING_REQUESTS or total > RING_KEYS):
                _, old = _RING.popitem(last=False)
                total -= _nkeys(old)
                _STATS["evicted"] += 1
    except Exception:
        _STATS["capture_errors"] += 1


def find(req_id: str):
    """(entry, source) -- 'live' from the scheduler, 'recent' from the ring,
    or (None, None)."""
    o = _osched()
    if o is not None:
        try:
            st = getattr(o, "_req_status", {}).get(req_id)
            if st is not None:
                return entry_of(o, st), "live"
        except Exception:
            pass
    with _LOCK:
        e = _RING.get(req_id)
    return (e, "recent") if e is not None else (None, None)


def known() -> list:
    out = []
    o = _osched()
    if o is not None:
        try:
            for rid, st in list(getattr(o, "_req_status", {}).items())[:64]:
                out.append({"req": str(rid), "source": "live",
                            "num_tokens": int(st.req.num_tokens)})
        except Exception:
            pass
    with _LOCK:
        for rid, e in reversed(_RING.items()):
            out.append({"req": rid, "source": "recent",
                        "num_tokens": e["num_tokens"], "keys": _nkeys(e),
                        "finished_at": e["at"]})
    return out


def render(e: dict, source: str, mgr=None, group=None,
           limit: int = LIMIT_DEFAULT) -> dict:
    """The answer for one request. `mgr` is the scheduler-side presence
    manager (index, verified set, mapper, rank count); absent -> nulls with a
    reason, never False."""
    why = None
    mappers = []
    present = verified = None
    if mgr is None:
        why = "no disk manager in this process"
    else:
        try:
            base = mgr._mapper
            for r in range(int(mgr._num_workers)):
                m = copy.copy(base)
                m.rank = r
                mappers.append(m)
        except Exception as exc:
            why = ("per-rank paths unreadable: %r" % (exc,))[:200]
            mappers = []
        try:
            present = mgr._present
            verified = mgr._verified_set()
        except Exception as exc:
            why = ("index unreadable: %r" % (exc,))[:200]
            present = verified = None
    groups = []
    for g in e["groups"]:
        if group is not None and g["group"] != group:
            continue
        bt = g["block_tokens"]
        rows = []
        for i, k in enumerate(g["keys"][:max(0, int(limit))]):
            kb = bytes(k)
            row = {"i": i, "key": kb.hex(),
                   "tokens": [i * bt, (i + 1) * bt] if bt else None,
                   "in_index": (kb in present) if present is not None else None,
                   "verified": (kb in verified) if verified is not None else None}
            if mappers:
                try:
                    row["path"] = {str(r): m.get_file_name(kb)
                                   for r, m in enumerate(mappers)}
                except Exception as exc:
                    row["path"] = None
                    row["path_why"] = repr(exc)[:160]
            else:
                row["path"] = None
            rows.append(row)
        groups.append({"group": g["group"], "block_tokens": bt,
                       "num_hit_blocks": g["num_hit_blocks"],
                       "num_keys": len(g["keys"]),
                       "truncated": len(g["keys"]) > len(rows),
                       "keys": rows})
    out = {"req": e["req"], "source": source, "num_tokens": e["num_tokens"],
           "groups": groups,
           "paths_are": "container paths (the host's ~/kvoffload is /kvoffload)"}
    if why:
        out["why"] = why
    return out


def http_get(path: str, query: dict):
    """(code, body) for /kv/keys, or None for any other path."""
    if path != "/kv/keys":
        return None
    req = (query.get("req") or [None])[0]
    if not req:
        return 200, {"requests": known(), "ring_requests": RING_REQUESTS,
                     "ring_keys": RING_KEYS, "stats": dict(_STATS)}
    try:
        group = query.get("group", [None])[0]
        group = int(group) if group not in (None, "") else None
        limit = int(query.get("limit", [LIMIT_DEFAULT])[0])
    except ValueError:
        return 400, {"error": "group and limit must be integers"}
    e, source = find(req)
    if e is None:
        return 404, {"error": "unknown request: not live, and not among the "
                              "%d most recently finished" % RING_REQUESTS,
                     "req": req, "known": [k["req"] for k in known()][:64]}
    from .. import diskspec
    return 200, render(e, source, diskspec._LIVE_MANAGER, group, limit)


def _install_capture() -> None:
    try:
        from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
            scheduler as _osm,
        )
    except Exception:
        logger.info("kvmgr: reqkeys: no OffloadingConnectorScheduler; "
                    "/kv/keys will answer for live requests only")
        return
    cls = _osm.OffloadingConnectorScheduler
    orig = cls.request_finished
    if getattr(orig, "_kvmgr_wrapped", False):
        return

    def request_finished(self, request, *a, **kw):
        capture(self, request.request_id)     # before upstream drops it
        return orig(self, request, *a, **kw)

    request_finished._kvmgr_wrapped = True
    cls.request_finished = request_finished
    _STATS["installed"] = True


def install() -> None:
    hooks.register("http_get", http_get)
    _install_capture()
