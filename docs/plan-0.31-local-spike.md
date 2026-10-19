# Plan: kvmgr on vLLM 0.31, run end to end in a Linux container on this Mac (2026-10-15)

Branch `spike/vllm-0.31-local`, worktree `../vllm-kvmgr-031-local`, cut from
`main` at v0.4.8 (`66d3b55`; rebased 2026-10-15). Local only: nothing is pushed to `head` or
`worker`, and nothing to `origin` unless asked. The earlier offline spike
(`spike/vllm-0.31`, `620c3a8`) is the reference, not the base. It predates
v0.4.2–v0.4.8 (load_wait_compute, the parallel walk, loadfail,
checksums, /kv/keys), and the port has to carry those.

## What was checked before writing this (2026-10-15)

- **Latest vLLM is still v0.31.0** (PyPI). The release also ships CPU wheels
  for Linux arm64/x86_64 and (despite the docs) macOS arm64, and Docker Hub
  has `vllm/vllm-openai-cpu:v0.31.0-arm64`.
- **Decision (2026-10-15): a Linux arm64 container under Apple's `container`
  tool**, not a native Mac build. It needs no build, it's the same OS as the
  cluster, it keeps user xattrs (so the checksums can be tested), and it's the
  CI-supported CPU path. Apple `container` 1.5.0 (Homebrew) runs each
  container in its own lightweight VM on macOS 26+. Fallbacks, with the same
  image and mounts: OrbStack or Colima; the native macOS wheel as a last
  resort.
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
- macOS Python has no `os.setxattr`. Inside the Linux container it works,
  but only on a container-local volume: keep the tier root off the macOS
  bind mount.
- **0.31's copy handler is reached through `current_platform`**
  (`current_platform.Stream()`, `.current_stream()`), not `torch.cuda`
  directly, so the CPU backend may get further than "CUDA-only" suggests. The
  copy op itself (`swap_blocks_batch`) is still the question. Phase 2 finds
  out.

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

**0. Re-port onto this branch: DONE (`63d9304`, 2026-10-15).**
- `620c3a8` replayed onto v0.4.8 with one conflict (the load's
  `TransferResult`).
- Then the v0.4.2+ work against 0.31:
  - selftest is on the new `transfer_async(job, src, dst)`;
  - reqkeys captures before AND after `request_finished`, because 0.31 keeps
    the state and appends the final key there, and reads the per-chunk field
    names;
  - loadfail is unchanged: 0.31 still asserts, and still unpacks one group;
  - `_arm_compute_wait` is now a second, harmless wait.
- Pristine gate: the disk, journal, verify, ledger, copyprobe, loadfail and
  reqkeys suites are green.
- phase1/2/3/4/6/10/11 stop at `harness.load()` on the GPU-manager core,
  which has no hook on 0.31. phase6 (reload audit) and phase11 (diag install)
  are collateral and worth decoupling from the GPU manager later.
- **Known gaps carried forward:**
  - `diag/copyprobe.describe_copy` still mirrors 0.21's copy arithmetic;
  - `diag/tiering` wraps upstream internals that 0.31 made private.

  Both are diagnostics and both are off the critical path.

**1. Run vLLM 0.31.0 in an Apple `container` (~1 h).**
- Install: `brew install container && container system start` (the user's
  step; it adds a system service).
- Image: `vllm/vllm-openai-cpu:v0.31.0-arm64`.
- One scripted `container run`, kept in `scripts/local031.sh` on this branch:
  - `--memory 16G --cpus 8`, port 8000 for the API and the
    `kvmgr_stats_port` published;
  - mounts: this worktree's `src/kvmgr` and its generated dist-info (the
    same shape as DSpark's compose), the flags file, and the HF cache
    read-only;
  - the tier root on a container volume, not the macOS bind mount, so
    xattrs work.
- Smoke test: serve `Qwen/Qwen3-0.6B` (one KV group, ungated) with
  `--enable-prefix-caching`, one completion, and the `kvmgr: registered`
  lines in the log.
- Walls to report: the image doesn't start under `container`, or the plugin
  doesn't load from the mount.

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
  is on the container volume, a local flags file, and `kvmgr_stats_port` is
  set. Turn `disk_checksum_write` on from the first prime, so the
  restore-time check can run here.
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

## Results so far (2026-10-15)

Apple `container` 1.5.0, `vllm/vllm-openai-cpu:v0.31.0-arm64` (vLLM 0.31.0,
torch 2.13 CPU, Python 3.12), Qwen3-0.6B (one KV group, 128-token blocks,
14.7 MB a block over 28 layers), 8 CPUs / 16 GB, `scripts/local031.sh`.

| step | result |
|---|---|
| plugin load in 0.31 | entry point, loadfail (both hooks), reqkeys, disk spec registered; GPU manager skipped as designed |
| first wall | `/dev/shm` too small for vLLM's engine queue: `--shm-size 4G` |
| second wall | 0.31's copy handler asserts CUDA/XPU tensors: `HostCopyHandler` (plain torch) on non-CUDA |
| third gap | the stats port (`/kv`, `/kv/keys`) only started from the GPU manager: now also from the disk manager |
| prime, 4,850-token canary | 19.19 s; 37 files (543 MB), 37 CRC32 xattrs, journal and ledger written |
| restart + restore | journal replayed 37; BOOT WALK 37/37; **37 blocks restored, 4,736 external-hit tokens, 1.87 s, identical reply** |
| checksum verify on | 37/37 matched at restore; +0.04 s |
| Gate B rehearsal (flip 1 byte in block 18, POLICY=recompute) | checksum caught it; file unlinked; job's 37 blocks reported invalid; vLLM "Recovered from KV load failure"; 37 keys left the index (journalled); engine up; identical reply in 19.06 s (a full recompute) |
| self-heal | **on the NEXT request with that prefix**: 36 stores skipped (file present), 1 rewritten, with the original CRC; index back to 37/38 |
| **phase 4**: Gemma 3 270M (`unsloth/gemma-3-270m-it`, the ungated mirror) | 0.31 makes **six** KV groups of 3 layers (five sliding-window, one full); 128-token blocks |
| prime, 5,014 tokens | 4.0 s; **59 files**: the full group stores all 39 blocks, each sliding group only its last 4 (window 512): upstream's tail pruning, through our spec |
| restart + restore | 59/59 blocks restored, 59/59 CRC ok, **0.28 s against 4.0 s**, identical reply |
| **fourth wall**: Gate B on six groups | the checksum caught it, but **the engine core died**: 0.31's `_handle_invalid_blocks` raises on more than one group before our multi-group fix is reached ("report failed requests via `failed_recving` instead") |
| fix (`b43a455`) | on 0.31 a failed load is reported by REQUEST (`failed_recving`), never by block; `_handle_failed_recving` restarts it from token 0 and is where the keys leave the index |
| Gate B rerun, group 2 block 36 | checksum caught; engine up; identical reply after a 3.93 s recompute; all 59 keys left the index; next request restored the index; reboot restores 59/59 with CRC ok |
| **phase 5**: TP=2 on one host (Qwen3-0.6B, `CPUBIND="0-3\|4-7"`) | both ranks write their half page (7.3 of 14.7 MB): 37 files each in `_r0` and `_r1`; boot walk across both ranks |
| restart + restore, TP=2 | 37 blocks per rank, CRC ok, 1.77 s against 16.6 s, identical reply (and identical to TP=1's) |
| Gate B, **rank 1 only** (block 18) | rank 1's checksum failed while rank 0's 18 reads were fine; the failure reached the scheduler through upstream's cross-rank aggregation; recompute 16.8 s, engine up, identical; heal on the next request; reboot restores 37/37 per rank |

Notes for the real port and for the vllm-proxy gates:
- The rewrite is not in the same request: the connector marks loaded blocks
  as stored for that request, so the heal comes from the next request with
  the prefix. Gate B should send the canary twice after the corruption. The
  0.21 connector does the same (`next_stored_block_idx = num_blocks` on a
  load), so this is not 0.31-specific.
- `loadfail`'s snapshot is per process: `/kv/workers` shows the worker's
  side (`noted_jobs`, `drained_blocks`); the scheduler side's `forgot_keys`
  shows as the manager's `load_failed_forgot`, not here.
- Our compute wait on a CPU backend reports "could not arm" once (no stream
  pool), which is correct there.

Notes from phases 4 and 5:
- **The 0.31 port needs request-level failure reporting** (above). A
  block-level report is fatal on any multi-group model there, and DeepSeek V4
  has five groups. The 0.21 path (invalid blocks, our multi-group scheduler
  fix) stays: 0.21 has no such raise and no `get_transfer_results`.
- On 0.31 a failure restarts the whole request (upstream's
  `_handle_failed_recving`), single group or not. That is the same as our 0.21
  multi-group semantics, and the cost of a corrupt file is one cold prefill.
- The cross-rank case works through upstream's `KVOutputAggregator`. A request
  is failed when any rank reports it, and only once every rank has finished
  it. This is the one the cluster's Gate B has to prove on 0.21.
- **The presence index is not per model.** `presence.json` sits at the format
  root, and the index is keyed by block hash. Changing the model (or TP, which
  changes the fingerprint directory) makes the boot walk drop the other
  directory's keys, and those files are then invisible to the index. The same
  is true on `main`. It's harmless with one model, but worth a look at what
  budget/LRU eviction does with the old directory's files after a fingerprint
  change. Not measured here.
- vLLM's CPU backend needs threads bound by hand for TP=2 on one NUMA node
  (`VLLM_CPU_OMP_THREADS_BIND`, the script's `CPUBIND`).

All phases are done. Report below.

## Report (2026-10-15)

**What ran:** kvmgr's disk tier, on vLLM 0.31.0, end to end in a real engine.
- one group (Qwen3-0.6B), six groups (Gemma 3 270M) and TP=2;
- prime, restart, journal replay, boot walk and restore, with checksums;
- Gate B (a flipped byte) on each, including a failure on rank 1 only.

Every restore gave the identical reply. Every corruption became a recompute
with the engine up, and the tier healed itself.

**Walls, in order:**
1. `/dev/shm` size, a container flag.
2. 0.31's copy handler is CUDA/XPU only: `HostCopyHandler`.
3. The stats port started only from the GPU manager.
4. 0.31 raises on block-level load failures with more than one KV group.

Only 4 matters for the cluster port.

**Lines changed in `src/`, from v0.4.9:** +393/−186 over six files.
- **The API port (phase 0, `373ebe6`):** `diskspec.py` and `register.py`, the
  0.31 field names in `diag/reqkeys.py`, and the new `transfer_async`
  signature in `diag/selftest.py`.
- **Phases 1–5:** +210/−20 in three files:
  - `diskio.py`: the host copy, and the job id on a failure;
  - `diskspec.py`: starting the stats port;
  - `loadfail.py`: request-level reporting, +90.

**What this spike cannot say:**
- the CUDA copy path and stream ordering (the Xid class);
- DeepSeek V4 / MLA / NVFP4 on 0.31, and the DSpark patches;
- two nodes, and throughput.

**Revised effort for the real port:**
- disk tier: 1–2 days to bring this branch to a mergeable state, plus a
  validation boot. Lower than the first estimate, because the integration
  questions are answered;
- GPU manager: drop it (item 3 of the main plan decides);
- DSpark patches: 1–3 weeks, unchanged and still the real cost.

The recommendation below is unchanged: phase 4's wall is one more piece of
safety the built-in tiers would not have given us.

## On a GB10 (spark-45ea, 2026-10-15)

This was a one-off run on the head node while both engines were down under
`~/.engine-hold`, with the user's go-ahead. Nothing of the cluster's was
touched: a container named `kvmgr031`, everything under `~/kvmgr031-test`
(deleted afterwards), ports 18000/18899 on 127.0.0.1, and 25% of the GPU.
The scripts are `scripts/cuda031.sh` and `scripts/cycle031.sh`.

(This section was written on 2026-10-16. Its first version was never
committed: a failed command earlier in the same shell line stopped it.)

**Two walls, neither kvmgr's** (a bare vLLM stalled the same way):
1. **The 64K-page kernel hangs vLLM's default lazy safetensors loader** at
   the first weight copy, as DSpark's compose documents. The fix is
   `--safetensors-load-strategy eager`.
   - *Corrected 2026-10-16:* this, not missing sm_121 kernels, is also what
     stalled the stock `vllm/vllm-openai:v0.31.0`. That image was abandoned
     before eager loading was found, and was never re-tested.
   - sm_120 code runs on sm_121, and v0.31.0 pins an SM120-capable DeepGEMM.
2. **The run used the node's `eugr/spark-vllm:latest`,** built for 12.1a with
   CUDA 13.0.2.
   - Its vLLM is a `main` commit (`0872ddf4b`), not v0.31.0: compared with the
     tag, it has 345 commits v0.31.0 lacks, and v0.31.0 has 27 it lacks.
   - They differ in DSV4's `attention.py` and `compressor.py`, and in
     `block_pool.py` and `scheduler.py`.
   - Its sources had every 0.31 shape this branch relies on.

The container also needed `--ulimit memlock=-1`.

**Results** (Qwen3-0.6B, 16-token blocks, 1.8 MB a page, a 4,850-token canary
of 303 blocks):

| step | result |
|---|---|
| prime | 0.39 s; stores flush in the background |
| reboot + restore | boot walk 256/256; 256 blocks restored, 256/256 CRC ok, identical |
| Gate B, block 18 | checksum caught it; request-level failure; **engine up**, recompute, identical; no Xid or illegal access |
| next request | index whole again, 303/303 |
| reboot + restore | 303/303 restored, CRC ok, identical |
| compute wait | `compute_waits 5`, `compute_wait_unarmed 0`: armed on every copy round |

This proves the CUDA copy path (0.31's `SingleDirectionOffloadingHandler`
under our handlers), the checksums and the request-level recompute on a real
GPU. It doesn't prove speed: the model is too small for a restore to beat a
0.39 s prefill. It also doesn't cover DeepSeek V4, two nodes, or the Xid
reproducer.

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
