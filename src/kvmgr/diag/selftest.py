"""kvmgr.diag.selftest -- replay known-bad load destinations at boot, on an idle GPU.

ask3 C (2026-10-14). Twice the same disk->GPU load failed on BOTH GPUs (Xid 32
at 10:43, a hang at 12:36), with identical destination block ids on both ranks.
If the trigger is in the copy's parameters, it should reproduce with nothing
else running. Before the engine serves, every GPU block is free, so this:

  1. fills load-staging rows with a seeded pattern (only group 0's tensors);
  2. issues ONE group-0 load copy per set, through the load handler's own
     inner handler -- upstream's `transfer_async` and `swap_blocks_batch` --
     at the set's real size, into exactly its block ids;
  3. waits by polling the copy's end event, COPY_TIMEOUT_S at most -- never
     an unbounded sync;
  4. reads the blocks back through the store handler's inner handler into
     poisoned store-staging rows, and compares each ref's bytes;
  5. records per set: completed / timed_out / faulted / skipped, the time,
     and the comparison.

Controls run FIRST: the same sizes into contiguous low blocks (1..n; block 0 is
vLLM's null block and is never touched). If a control fails, the path is
broken, not the ids. A timeout or fault stops the run (the stream can no
longer be trusted); the whole run is bounded by TOTAL_BUDGET_S. The verdict is
rewritten after every step to `<ledger dir>/selftest-r<rank>.json`, so a wedge
mid-run still leaves the set it wedged on, marked `running`.

The 16-block set from the 12:36 hang is built in. Others come from
`<ledger dir>/selftest-sets.json`: `{"sets": [{"name", "bids", "sha"?}]}`,
`sha` being the 12-hex id the vllm-proxy session quotes (sha1 of the ledger's
`bids` ranges); a mismatch skips the set rather than replaying the wrong ids.

WRITES INTO FREE KV BLOCKS before serving. Off by default
(`diag_copy_selftest`, boot-time knob); its first run is a deliberate boot.
"""

from __future__ import annotations

import json
import os
import time

from vllm.logger import init_logger

from .. import hooks

logger = init_logger("vllm." + __name__)

COPY_TIMEOUT_S = 10.0
TOTAL_BUDGET_S = 60.0
SETS_FILE = "selftest-sets.json"
INNER_BASE = 9_000_000          # disjoint from the store's and the load's ids

# job 5490, the 12:36:18 UTC hang on both ranks: group 0, 16 blocks, 992 ops.
BUILTIN_SETS = (
    {"name": "job5490-hang", "sha": "cadfed0fc6ce",
     "bids": [4892, 16178, 16524, 18563, 18924, 19257, 19329, 19376, 19861,
              20196, 20805, 20964, 20967, 21069, 21440, 21696]},
)


class TorchOps:
    """The tensor work, behind a seam so the run logic tests offline."""

    def fill_pattern(self, t, rows: int, seed: int) -> None:
        import torch
        g = torch.Generator().manual_seed(seed)
        t[:rows].copy_(torch.randint(-128, 128, (rows, t.shape[1]),
                                     dtype=torch.int8, generator=g))

    def poison(self, t, rows: int) -> None:
        t[:rows].fill_(0x5A)

    def equal(self, a, ai: int, b, bi: int, nbytes: int) -> bool:
        import torch
        return bool(torch.equal(a[ai, :nbytes], b[bi, :nbytes]))


def make_specs(bids, ngroups: int, slots):
    from vllm.v1.kv_offload.base import GPULoadStoreSpec
    from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
    n = len(bids)
    gpu = GPULoadStoreSpec(list(bids), group_sizes=[n] + [0] * (ngroups - 1),
                           block_indices=[0] * ngroups)
    return gpu, CPULoadStoreSpec(list(slots))


def wait_event(inner, iid: int, timeout: float, clock=time.monotonic,
               sleep=time.sleep) -> tuple:
    """(state, ms, why) by polling the copy's end event. Never syncs."""
    t0 = clock()
    while True:
        try:
            ev = inner._transfer_events.get(iid)
            if ev is None:
                return "faulted", (clock() - t0) * 1e3, "no end event recorded"
            if ev.query():
                return "completed", (clock() - t0) * 1e3, None
        except Exception as exc:
            return "faulted", (clock() - t0) * 1e3, ("query raised: %r" % (exc,))[:200]
        if clock() - t0 >= timeout:
            return "timed_out", (clock() - t0) * 1e3, (
                "end event not reached after %.0fs" % timeout)
        sleep(0.001)


def sha_of(bids) -> str:
    from .ledger import bids_sha
    return bids_sha(set(int(b) for b in bids))


def plan(sets) -> list:
    """Controls first (one per distinct size, blocks 1..n), then the sets."""
    sizes = []
    for s in sets:
        if len(s["bids"]) not in sizes:
            sizes.append(len(s["bids"]))
    controls = [{"name": "control-low-%d" % n, "bids": list(range(1, n + 1)),
                 "control": True} for n in sizes]
    return controls + [dict(s) for s in sets]


def load_sets(dirp: str) -> tuple:
    """(sets, why) -- the built-ins plus the operator's file, if any."""
    sets = [dict(s) for s in BUILTIN_SETS]
    path = os.path.join(dirp, SETS_FILE)
    if not os.path.exists(path):
        return sets, "no %s (built-in sets only)" % path
    try:
        with open(path) as fh:
            extra = json.load(fh).get("sets") or []
        for s in extra:
            sets.append({"name": str(s["name"]),
                         "bids": [int(b) for b in s["bids"]],
                         **({"sha": str(s["sha"])} if s.get("sha") else {})})
        return sets, None
    except Exception as exc:
        return sets, ("%s unreadable, built-ins only: %r" % (path, exc))[:240]


def run(h, store, ops=None, clock=time.monotonic, sleep=time.sleep,
        dirp=None, sets=None) -> dict:
    """The self-test. Returns the verdict it wrote. Never raises."""
    ops = ops or TorchOps()
    t_start = clock()
    from . import ledger as _led
    rank = getattr(getattr(h, "_spec", None), "rank", None)
    dirp = dirp or _led.ledger_dir(h)
    os.makedirs(dirp, exist_ok=True)
    path = os.path.join(dirp, "selftest-r%s.json" % (rank if rank is not None else "x"))
    sets_why = None
    if sets is None:
        sets, sets_why = load_sets(dirp)
    verdict = {"rank": rank, "pid": os.getpid(), "boot": _led.boot_id(),
               "started_at": time.time(), "state": "running",
               "copy_timeout_s": COPY_TIMEOUT_S, "budget_s": TOTAL_BUDGET_S,
               "sets_why": sets_why, "results": []}

    def write():
        try:
            tmp = path + ".tmp"
            with open(tmp, "w") as fh:
                json.dump(verdict, fh, indent=1)
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("kvmgr: selftest verdict write failed: %r", exc)

    try:
        refs = h._refs[0]
        ngroups = len(h._refs)
        tensors = sorted({int(r.tensor_idx) for r in refs})
        sizes = {}
        for r in refs:   # one compare per (tensor, size) is enough
            sizes[int(r.tensor_idx)] = max(sizes.get(int(r.tensor_idx), 0),
                                           int(r.page_size_bytes))
        gpu_rows = min(int(t.shape[0]) for t in h._gpu_tensors)
        load_rows = min(int(h._staging[t].shape[0]) for t in tensors)
        store_rows = min(int(store._staging[t].shape[0]) for t in tensors)
        verdict.update(gpu_rows=gpu_rows, load_staging_rows=load_rows,
                       store_staging_rows=store_rows, group0_refs=len(refs),
                       group0_tensors=len(tensors))
    except Exception as exc:
        verdict.update(state="error", why=("geometry unreadable: %r" % (exc,))[:240])
        write()
        return verdict

    iid = INNER_BASE
    stop = None
    for k, s in enumerate(plan(sets)):
        bids, n = s["bids"], len(s["bids"])
        res = {"name": s["name"], "n": n, "ops": n * len(refs),
               "control": bool(s.get("control")), "bids_sha": sha_of(bids)}
        verdict["results"].append(res)
        if stop:
            res.update(result="skipped", why=stop)
            continue
        if s.get("sha") and s["sha"] != res["bids_sha"]:
            res.update(result="skipped", why="sha mismatch: expected %s, these "
                       "ids hash to %s" % (s["sha"], res["bids_sha"]))
            continue
        if any(b <= 0 or b >= gpu_rows for b in bids):
            res.update(result="skipped", why="ids outside 1..%d" % (gpu_rows - 1))
            continue
        if n > load_rows:
            res.update(result="skipped", why="load staging has %d rows, the set "
                       "needs %d in ONE copy" % (load_rows, n))
            continue
        if clock() - t_start >= TOTAL_BUDGET_S:
            stop = "budget of %.0fs spent" % TOTAL_BUDGET_S
            res.update(result="skipped", why=stop)
            continue
        try:
            for t in tensors:
                ops.fill_pattern(h._staging[t], n, seed=1000 * k + t)
            gpu, cpu = make_specs(bids, ngroups, range(n))
            res["result"] = "running"
            write()
            iid += 1
            if not h._inner.transfer_async(iid, (cpu, gpu)):
                res.update(result="faulted", why="transfer_async returned False")
                stop = "a copy failed to submit"
                write()
                continue
            state, ms, why = wait_event(h._inner, iid, COPY_TIMEOUT_S, clock, sleep)
            res.update(result=state, copy_ms=round(ms, 3))
            if why:
                res["why"] = why
            if state != "completed":
                stop = "stopped after %s %s" % (s["name"], state)
                write()
                continue
            h._inner.get_finished()      # drain ours; nothing else is queued
            # read back, in chunks the store's staging can hold
            bad, rb_ms = [], 0.0
            for c0 in range(0, n, store_rows):
                chunk = bids[c0:c0 + store_rows]
                for t in tensors:
                    ops.poison(store._staging[t], len(chunk))
                gpu2, cpu2 = make_specs(chunk, ngroups, range(len(chunk)))
                iid += 1
                if not store._inner.transfer_async(iid, (gpu2, cpu2)):
                    raise RuntimeError("readback submit returned False")
                st2, ms2, why2 = wait_event(store._inner, iid, COPY_TIMEOUT_S,
                                            clock, sleep)
                rb_ms += ms2
                if st2 != "completed":
                    res.update(readback=st2, readback_why=why2)
                    stop = "readback %s" % st2
                    break
                store._inner.get_finished()
                for i, b in enumerate(chunk):
                    for t in tensors:
                        if not ops.equal(h._staging[t], c0 + i, store._staging[t],
                                         i, sizes[t]):
                            bad.append([int(b), t])
            res["readback_ms"] = round(rb_ms, 3)
            if "readback" not in res:
                res.update(readback="completed", match=not bad,
                           mismatches=len(bad), mismatch_examples=bad[:32])
        except Exception as exc:
            res.update(result="faulted", why=("raised: %r" % (exc,))[:240])
            stop = "stopped after %s raised" % s["name"]
        write()

    bad_sets = [r["name"] for r in verdict["results"]
                if r.get("result") not in ("completed", "skipped")
                or r.get("match") is False]
    verdict.update(state="done", elapsed_s=round(clock() - t_start, 3),
                   failed=bad_sets)
    write()
    (logger.warning if bad_sets else logger.info)(
        "kvmgr: COPY SELFTEST %s -- %s. Verdict: %s",
        "FOUND A FAILURE" if bad_sets else "all sets completed and matched",
        ", ".join("%s=%s%s" % (r["name"], r.get("result"),
                               "" if r.get("match", True) else "/MISMATCH")
                  for r in verdict["results"]), path)
    return verdict


def on_built(h, store) -> None:
    try:
        run(h, store)
    except Exception:
        logger.exception("kvmgr: copy selftest failed to run (boot continues)")


def install() -> None:
    hooks.register("disk_load_handler_built", on_built)
