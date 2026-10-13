"""kvmgr.diag.geometry -- write the KV tensor geometry to a JSON file.

A cross-process channel with no IPC: the worker knows the geometry, the engine
core and the operator did not, and every geometry question used to cost a
four-minute boot. Moved verbatim from kvmgr.diskio on 2026-10-13; OFF by
default (diag_geometry_snapshot) -- the geometry it was built to find is
measured and written down, and the handler logs its own sizes.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger

from .. import hooks

logger = init_logger("vllm." + __name__)


def geometry(kv_caches: Any, spec: Any) -> dict:
    """Everything about the live KV geometry, as plain JSON-able data.

    *** WHY THIS EXISTS: THE WORKERS HOLD THE ANSWERS AND HAVE NO CHANNEL. ***
    `kv_caches` lives in the WORKER processes while our HTTP endpoint lives in
    the engine core, so every question about the real geometry has cost a
    4-minute boot to observe. A file in the already-bind-mounted flags directory
    is a cross-process channel that needs no IPC and no new listener, and it is
    written once at construction, so it costs nothing per request.

    The specific question it was built for: our per-group sizes sum to 4,108,352
    B against ~957,537 B per block per rank (KV_CACHE_MEMORY_BYTES /
    num_gpu_blocks). That sum CANNOT be the per-block total -- it would need
    93.5 GB against the 21.8 GB allocated -- so either several groups reference
    the same tensor, or the groups do not all have num_gpu_blocks rows.
    Arithmetic cannot tell those apart; `tensors` and `shared_tensors` below can.
    """
    tensors = []
    for i, t in enumerate(kv_caches.tensors):
        shape = tuple(getattr(t.tensor, "shape", ()) or ())
        tensors.append({
            "idx": i,
            "page_size_bytes": t.page_size_bytes,
            "shape": [int(x) for x in shape],
            # rows is the number of BLOCKS this tensor holds, which is the half
            # of the question arithmetic cannot reach.
            "rows": int(shape[0]) if shape else None,
        })
    groups = []
    owners: dict[int, list[int]] = {}
    for g, refs in enumerate(kv_caches.group_data_refs):
        idxs = [r.tensor_idx for r in refs]
        for ti in idxs:
            owners.setdefault(ti, []).append(g)
        groups.append({
            "group": g,
            "nrefs": len(refs),
            "tensor_idxs": idxs,
            "distinct_tensor_idxs": sorted(set(idxs)),
            "bytes_per_block": sum(r.page_size_bytes for r in refs),
        })
    shared = {str(ti): sorted(set(gs)) for ti, gs in owners.items()
              if len(set(gs)) > 1}
    distinct_bytes = sum(t["page_size_bytes"] for t in tensors)
    return {
        "rank": getattr(spec, "rank", None),
        "world_size": getattr(spec, "world_size", None),
        "block_size_factor": getattr(spec, "block_size_factor", None),
        "whole_page_bytes": getattr(spec, "whole_page_bytes", None),
        "page_size_per_worker": getattr(spec, "page_size_per_worker", None),
        "gpu_block_size": list(getattr(spec, "gpu_block_size", ()) or ()),
        "hash_block_size": getattr(spec, "hash_block_size", None),
        "num_tensors": len(tensors),
        "sum_distinct_tensor_page_bytes": distinct_bytes,
        "sum_group_bytes_per_block": sum(g["bytes_per_block"] for g in groups),
        # *** THE ANSWER. *** Non-empty means groups share tensors, so summing
        # every group's refs double-counts and the 4x is redundancy. Empty means
        # they are disjoint and the per-group rows differ instead.
        "shared_tensors": shared,
        "groups": groups,
        "tensors": tensors,
    }


def write_geometry(kv_caches: Any, spec: Any, out_dir: str) -> str | None:
    """Best-effort: a snapshot must never be able to break a boot."""
    import json

    try:
        g = geometry(kv_caches, spec)
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, f"geometry-r{g.get('rank')}.json")
        tmp = path + ".part"
        with open(tmp, "w") as f:
            json.dump(g, f, indent=1, sort_keys=True)
        os.replace(tmp, path)
        logger.info(
            "kvmgr: geometry snapshot -> %s (%d tensors, groups share tensors: "
            "%s, sum of distinct tensor pages %d B, sum of per-group bytes %d B)",
            path, g["num_tensors"], bool(g["shared_tensors"]),
            g["sum_distinct_tensor_page_bytes"], g["sum_group_bytes_per_block"],
        )
        return path
    except Exception:
        logger.exception("kvmgr: geometry snapshot failed (continuing)")
        return None


def _on_handler_built(kv_caches: Any, spec: Any) -> None:
    write_geometry(kv_caches, spec,
                   os.environ.get("KVMGR_SNAPSHOT_DIR", spec.root_dir))


def install() -> None:
    hooks.register("disk_handler_built", _on_handler_built)
