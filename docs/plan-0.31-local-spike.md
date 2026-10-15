# Plan: kvmgr on vLLM 0.31, run end to end on this Mac (2026-10-15)

Branch `spike/vllm-0.31-local`, worktree `../vllm-kvmgr-031-local`, cut from
`main` at v0.4.8 (`66d3b55`; rebased 2026-10-15). Local only: nothing is pushed to `head` or
`worker`, and nothing to `origin` unless asked. The earlier offline spike
(`spike/vllm-0.31`, `620c3a8`) is the reference, not the base. It predates
v0.4.2–v0.4.8 (load_wait_compute, the parallel walk, loadfail,
checksums, /kv/keys), and the port has to carry those.

## What was checked before writing this (2026-10-15)

- **Latest vLLM is still v0.31.0** (PyPI). Wheels are Linux only. On macOS the
  docs say "build from source", which gives the CPU backend (FP32/FP16), and
  "there are no pre-built Apple silicon CPU wheels" or images.
- **This machine:** Apple M5 Pro, 48 GB, macOS 27.0.1, Xcode with Apple Clang
  21, Python 3.12, uv. The Docker CLI is installed but no daemon is running.
- **Every built-in offload path in 0.31 is CUDA-only.** The OffloadingConnector's
  CPU handler (`kv_offload/cpu/gpu_worker.py`: `torch.cuda.Stream`,
  `swap_blocks_batch`, pinned memory), `simple_kv_offload` (worker, copy and
  disk backends all create `torch.cuda.Stream`s), and tiering (which reuses
  that handler). So on a Mac no built-in tier runs at all, and **neither does
  our spec as it stands**, since it wraps the same `SingleDirectionOffloadingHandler`.
- **0.31 still asserts on a failed load** (`offloading/worker.py:288`) and still
  unpacks one KV group in `_update_requests_with_invalid_blocks`
  (`scheduler.py:3204`). `kvmgr.loadfail` stays necessary on 0.31.
- **0.31's load copy already waits on compute** (`gpu_worker.py:690-696`), so
  `load_wait_compute` / `_arm_compute_wait` become redundant there. Keep the
  knob and verify it, rather than delete it in the port.
- macOS Python has no `os.setxattr`, so the checksum tags read as absent here.
  Testing checksums needs Linux (Docker later, or the cluster).

## What a local run can and cannot tell us

It **can** prove the integration that the offline harness only imitates:
- the spec loads through 0.31's factory into a real engine;
- the scheduler-side manager works end to end: lookup, prepare/complete store
  and load, eviction, journal, and boot verification across a real restart;
- the restore actually gives back the same answer (an exact-echo canary);
- the v0.4.6/0.4.7 features behave the same on 0.31: recompute on failure
  (including the multi-group fix), the parallel walk, and `/kv/keys`.

It **cannot** tell us about:
- the CUDA copy path or stream ordering (the Xid class of problem);
- DeepSeek V4 / MLA / NVFP4, or the DSpark patches;
- two nodes. CPU TP=2 on one host might exercise the `_r1` paths; that's
  optional (phase 5).

Those stay with a cluster validation boot, which this spike sizes but doesn't do.

## Phases (time-boxed; stop at the first wall worth reporting)

**0. Re-port onto this branch (offline, ~half a day).**
- Re-vendor `tests/upstream/` from v0.31.0. The old spike has the file list
  and MANIFEST.
- Re-apply `620c3a8`'s port onto v0.4.7. It touched `diskspec.py` (the spec
  constructor, `get_worker()` and `DiskOffloadingWorker`, `LookupResult`,
  `on_new_request`, `Medium` events), `diskio.py` (the new handler signature,
  `TransferResult` from `base`), `register.py` (GPU manager gated off) and
  the harness.
- Then port the v0.4.2+ additions:
  - `loadfail`'s scheduler and connector hooks against 0.31's shapes, the
    phase15 pins included;
  - `reqkeys` against 0.31's `request_finished` / `RequestGroupState`;
  - `_arm_compute_wait` against 0.31's `transfer_async(job, src, dst)`.
- Gate: `MODES=pristine` green, except the GPU-manager suites, which are
  expected red as before. DSpark mode doesn't apply on this branch, because
  its patches target 0.21.

**1. Build vLLM 0.31.0 on macOS (~1–2 h, mostly compile).**
- Steps: `uv venv --python 3.12` in a scratch dir, clone the v0.31.0 tag,
  `uv pip install -r requirements/cpu.txt`, `uv pip install -e .`
  (`VLLM_TARGET_DEVICE=cpu` is automatic).
- Smoke test: serve `Qwen/Qwen3-0.6B` (FP16, one KV group, ungated), one
  completion, `--enable-prefix-caching`.
- Wall to report if it fails: the build or the CPU backend on macOS 27.

**2. A host-copy handler (the one new piece of code, ~half a day).**
- When the platform isn't CUDA-like, `DiskStoreHandler` and `DiskLoadHandler`
  use a small handler with the same interface: index copies between the KV
  tensors and staging rows with `torch`, done synchronously, with completion
  reported on the next `get_finished`.
- The CUDA path stays byte-for-byte as it is.
- First unknown: the CPU backend's KV tensor layout as the connector's
  `CanonicalKVCaches` presents it, and whether our `file_plan` refs map onto
  it. This is also where we learn whether `OffloadingConnectorWorker` itself
  runs on CPU. It's the most likely wall.
- This is something only our own spec can do: the built-in tiers would need
  upstream changes to run here.

**3. End to end, one group (Qwen3-0.6B).**
- Setup: `KV_TRANSFER_CONFIG` names `DSparkDiskOffloadingSpec`, the tier root
  is `~/kvlocal`, a local flags file, and `kvmgr_stats_port` is set.
- The cycle, with `/kv/keys` showing the files and `/kv/workers` the counters:
  1. Prime a ~4K-token exact-echo canary (the same design as vllm-proxy
     `ask1.md`).
  2. Confirm the files are written.
  3. Restart.
  4. Check the boot walk line.
  5. Send the canary again. Pass means the echo is exact AND the
     external-prefix hit counter moved.
- Then rehearse Gate B: corrupt one file, check it recomputes, the engine
  stays up, the key leaves the index and the next store rewrites it.
- Then: `offload_prompt_only=True` (are decode blocks stored?) and the
  `prefix_cacheable_group_ids` behaviour.

**4. Multi-group (`google/gemma-3-270m-it`; gated on HF, needs a token).**
- Sliding-window plus global layers give two KV groups. That exercises:
  per-group files, upstream's SWA tail pruning on the store side, the loadfail
  multi-group path on a real scheduler, and `/kv/keys` across groups.
- If Gemma is unavailable, pick another small hybrid-attention model, not a
  single-group one.

**5. Optional: TP=2 on one host (CPU, gloo).** If 0.31's CPU backend allows it,
this exercises rank-dependent paths (`_r0`/`_r1`), the cross-rank boot verify,
and the multi-rank invalid-block union that offline testing couldn't prove.

**6. Report.**
- What ran, the walls hit, and the lines changed per area.
- A revised effort estimate for the real port: the DSpark patches, DSv4 on
  0.31, and a validation boot.
- Update `docs/plan-2026-10-14.md` item 4 on `main`.

## Recommendation: keep our own spec, don't move to the built-in tiers

Unchanged from the first spike, with two new reasons from today's checks.

1. **Topology.** Upstream's CPU offload and tiering share one host region per
   host, with `rank = device_index % world_size`. With one GPU per node across
   two nodes, both ranks compute rank 0 (inferred from source, untested). Ours
   is one directory per rank by construction.
2. **Persistence.** `simple_kv_offload`'s disk backend keeps its index in
   memory, so nothing survives a restart. The tiering fs tier writes from the
   scheduler's view of one filesystem. Ours has a journalled index plus a boot
   verify across every rank, and surviving restarts is the reason the tier
   exists.
3. **New: platform.** All built-in paths are CUDA-only (checked above). Owning
   the worker is what lets us add a host-copy handler for this Mac. The same
   seam would serve any non-CUDA target later.
4. **New: safety work upstream still lacks on 0.31.**
   - Failed load → recompute: upstream asserts, and its scheduler unpacks one
     group.
   - Our own guards: pairing/bounds checks, separate load/store staging (the
     Xid 13 lesson), op and in-flight caps, the ledger and copy probe,
     checksums, `/kv/keys`, budget/LRU eviction, the free-space floor.
   - None of these exist in the built-in tiers, and most can only be added by
     owning the handler.
5. **What we get from upstream anyway:** the scheduler-side connector (SWA
   tail pruning, `offload_prompt_only`, job fencing) is upstream's either way,
   because our spec rides it. Upstream's compute wait on loads means we can
   retire our own arm on 0.31.
6. **Cost:** a port per vLLM bump. The first spike measured 1–2 days for the
   tier itself. The real cost of a move is re-deriving the DSpark patches
   (1–3 weeks), which applies whichever tier we use.

When to revisit: if upstream ships a per-rank, multi-node, persistent disk tier
whose index survives restarts, and drops the success assert. At that point,
compare its feature list against item 4 above.

## Not in this spike

- The GPU-manager half: dropped on 0.31 unless item 3 of the main plan says
  protection still pays.
- The DSpark patches.
- Anything on the cluster.
