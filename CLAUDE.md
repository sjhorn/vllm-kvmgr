# vllm-kvmgr

A vLLM general plugin (`kvmgr = kvmgr:register`): a hot-configurable KV cache
manager for the MLA group, plus a disk prefix cache that survives restarts.
`README.md` is the real documentation. This file covers how to maintain it.

## Where it lives, and who does what

- **Developed in its own Claude session, from this repo** (since 2026-10-14).
  The cluster is operated from the `~/dev/python/vllm-proxy` session (the
  restarts, deploys and measurements in its `CLAUDE.md`), and the recipe is the
  DSpark repo. This file is meant to stand alone, but **DSpark's `CLAUDE.md`
  governs the engine** and wins any disagreement about restarting it.
- Releases are tags (`v0.1.0` … `v0.4.11` for vLLM 0.21, `v0.5.0` for 0.31,
  `v0.6.0` for 0.32), each naming the vLLM build it is proven on
  (`pyproject.toml` states the window).
- **`main` is the vLLM 0.32 line (0.6.x), the one the cluster runs** (since
  2026-10-19; production since 2026-10-18 12:05Z). The older lines are
  retired: their checkouts, images and disk tiers were deleted from the nodes
  on 2026-10-19. Their branches stay for history only: `vllm-0.31` (0.5.x,
  tag `v0.5.0`) and `release/0.4.x` (0.4.x, vLLM 0.21). Never deploy them.
- **On the nodes** (`head`/`worker` = `/home/shorn/vllm-kvmgr` on spark-45ea and
  spark-da08, `receive.denyCurrentBranch=updateInstead`):
  - `~/vllm-kvmgr-032` is what serves: a worktree on branch `vllm-0.32`. The
    DSpark compose mounts its `src/kvmgr` and its dist-info.
  - `~/vllm-kvmgr` checks out `main` and is mounted by nothing. It is the
    git home of that worktree, so **never delete it**.
  - Keep `vllm-0.32` equal to `main`. **Deploy:** tests green → tag →
    `git push head main main:vllm-0.32 && git push worker main main:vllm-0.32`
    (`updateInstead` updates both checkouts). Push only as part of a deploy the
    user asked for. DSpark's start script refuses to start unless both nodes
    are at the same commit.
  - **A push does not reach a running engine**: it takes effect at the next
    restart, and the next restart may be a crash recovery nobody planned.
    **So never push code to the nodes you would not want booted.**
- **It is the only copy.** The DSpark recipe repo
  (`~/dev/spark/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`) no
  longer holds one: its compose mounts `~/vllm-kvmgr-032` on each node.
- **A version bump renames the dist-info** (`deploy/vllm_kvmgr-<ver>.dist-info`),
  and DSpark's `docker-compose.dspark-032.yml` names it in one mount line. Bump,
  regenerate, then change that line in DSpark and push BOTH repos together. A
  compose naming a dist-info that is not there has no entry point, so kvmgr
  silently does not load. `test_phase12_packaging.py` in dspark mode fails
  until the two agree.
- Restarting: **ask before every restart**, stating goal, change, cost (a
  10–22 min boot, every session on the cluster interrupted) and why it cannot
  wait.
- **Stop and start on 0.32:** on spark-45ea, `touch ~/.engine-hold; cd
  ~/dspark032 && ./serve-032.sh down` (stop both) or `./serve-032.sh`
  (restart both); `rm ~/.engine-hold` once it answers. **Never** the
  `stop-`/`start-deepseek-v4-flash-dspark.sh` scripts in either checkout: they
  drive the retired 0.21 compose, so `stop` removes nothing and still prints
  "stopped" (2026-10-19). `sparkstats.service` (a user unit) holds
  `/dev/nvidia-uvm` open, so stop it too before swapping the UVM module.
- **`engine-supervisor` restarts BOTH nodes by itself** (a user unit on
  spark-45ea, since 2026-10-14). It acts when either engine container exits
  abnormally (any code except 0/143, OOMKilled, or dead), and after a reboot.
  **So any experiment that may kill a container (fault hunts, `kvkillmid.sh`,
  a SIGKILL test) needs `touch ~/.engine-hold` on spark-45ea first and `rm`
  after**, or the supervisor will "recover" the experiment, and it counts
  toward its 2-restarts-in-6h limit. It also holds while a DSpark
  `local-ops/README.md` actuator runs. Read `local-ops/README.md`'s CLASS
  column before running any `kv*` tool. Several look like checks and are not.

## Layout

- `src/kvmgr/` is the core. `diag/` holds the opt-in diagnostics, reached only
  through `hooks.py`; **core must never import from `diag`** (a test enforces
  this). `compat/dspark.py` bridges the DSpark fork's patch6 and does nothing
  without it.
- **Patches since vLLM 0.31** (on top of the list below):
  - `loadfail.py` wraps OffloadingConnector `get_transfer_results` and
    Scheduler `_handle_failed_recving` (the request-level path).
  - `copyapi.py` wraps the load handler's `_swap_blocks_batch` instance
    attribute.

  phase15 (15d2) and phase17 pin them.
- **Tests on this branch:**
  - DSpark mode is skipped (DSpark's patches are 0.21 files).
  - phase1, 2, 3 and 10 skip with a reason: no GPU manager.
  - Every skip says so; none passes silently.
- **Patches on vLLM classes, which a port must re-check:** `loadfail.py`
  (Scheduler `_update_requests_with_invalid_blocks`, OffloadingConnector
  `get_block_ids_with_load_errors`), `diskio._arm_compute_wait` (upstream's
  `_stream_pool`), `register.py` (OffloadingConnectorScheduler `__init__`,
  KVCacheCoordinator `__init__`), `diag/reqkeys.py` (OffloadingConnectorScheduler
  `request_finished`), `copyapi.py` (the `ops` global of the copy handler's
  module). Tests pin the upstream code each of the first, second, fourth and
  last relies on (phase15, phase1b C7, phase16, phase17); the `__init__` wraps
  only need the classes to exist.
- `tests/upstream/` holds pristine vLLM sources the tests load (see its
  MANIFEST). Re-vendor it from the target build when porting.
- `deploy/` holds the generated dist-info the cluster bind-mounts. Regenerate it
  with `deploy/gen-dist-info.py` whenever modules are added or removed;
  `--check` verifies it.

## Testing

```sh
KVMGR_DSPARK_REPO=~/dev/spark/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark \
  bash tests/run-all.sh; echo $?
```

- **Without `KVMGR_DSPARK_REPO` every dspark-mode suite FAILS**, which reads like
  a broken tree and is only a missing variable. `MODES=pristine` runs one mode.

- Runs every suite in two modes: pristine vLLM, and with the DSpark patches.
  No GPU; vLLM leaf modules, numpy and torch are stubbed.
- **Read the exit status**, not the last line. Skipped sections are reported
  as skipped and are not passes.
- Use `python3.12`. Set `KVMGR_BUILD_PY` to a Python with `build` and
  `setuptools` to include the wheel build and install check.
- When you add a guard or test, break the code once to see it fail.

## Do not change without a measurement on the cluster

- **Frozen names:** `DSparkDiskOffloadingSpec`, `DSPARK_DISK` and the
  `dspark-kv-v1` root. They are part of the on-disk path, so renaming them
  orphans the existing tier. `KvmgrDiskOffloadingSpec` is the generic alias.
- **The on-disk layout.** A pinned hash of the layout functions fails offline
  when it changes. Never re-pin to silence it. `DISK_FORMAT_VERSION` names
  the root (`dspark-kv-v1`), so bumping it also orphans the existing tier.
- **`PYTHONHASHSEED=20261007`** in the deployment, the root of the block-hash
  chain. Changing it turns every file on disk into a silent miss.
- **`load_wait_compute=1`.** 0 is the Xid 32 (see below). Not a tunable.
- **`disk_load_failure_recompute`, `disk_checksum_write`/`_verify` stay off**
  until the vllm-proxy corruption gate has run them (plan item 6). Until a
  failed restore is proven survivable, the boot walk stays mandatory: it is the
  only thing standing between a missing file and a dead engine.
- **`load_ops_max=2048`, `load_inflight_max=1`** as code defaults. Inflight 4
  caused GPU faults (Xid 32), measured before `load_wait_compute` existed. **But until v0.4.0 the knob did not bound the
  GPU stage**: `_pump` checked it before a file read and `_collect` sent every
  finished read to the GPU unchecked. So "inflight 1 is safe" and "inflight 4
  faulted" were both measured without the bound, with up to 7 loads on the GPU
  at once. v0.4.0 bounds staging->GPU copies by it, and the read stage by it
  plus `load_read_ahead` (default 1, hot, not yet measured).

## Xid 32: the cause, and the rule it leaves (2026-10-14)

**Cause, demonstrated:** a disk->GPU load copy raced the forward pass. vLLM
0.21's `SingleDirectionOffloadingHandler` makes only STORES
`wait_stream(compute)`. With `--async-scheduling`, a load could run while the
in-flight step was still using its destination blocks. The result was Xid 32
(several `intr` words) or, once, a hang with no Xid. Every instance since
10-09 fell on a disk-tier load.

| `load_wait_compute` | loads | incidents |
|---|---|---|
| 1 (v0.4.2: organic traffic plus `kvrace.py run --arm on`) | 3,270+ | 0 |
| 0 (`kvrace.py run --arm off`, otherwise identical) | 36 | Xid 32 on the worker |

The faulting load was group 0, 16 blocks, `wait_compute=False`, alone on the
GPU, `in_step=0`. The boot self-test (`diag_copy_selftest`) had already shown
the same destination copies are byte-exact on an idle GPU, so the copy's
parameters were never the trigger. vLLM 0.31 does the same wait for loads
(`cpu/gpu_worker.py:690-696`).

**Fix:** `load_wait_compute=1` (v0.4.2, the code default):
`DiskLoadHandler._arm_compute_wait` makes the stream upstream will pop wait on
the current stream before each load copy.

- **Non-negotiable: `load_wait_compute` must stay 1 on a serving engine.**
  0 exists only for DSpark `local-ops/kvrace.py run --arm off`, the
  reproducer. While it is 0 the engine log says so as a WARNING (on the
  flip, then every 5 min, also from the watchdog with no load running), and
  `/kv/workers` counts `compute_wait_off`.
- **Earlier conclusions to re-read in this light.** "`load_inflight_max=4`
  killed the engine" and the op-cap and burst hypotheses were all measured
  without the wait, so they may be this race rather than limits of their own.
  The ledger (`/kvoffload/diag/ledger-r<rank>.jsonl`, `open_entries()` after
  a fault, `wait_compute` on every load line), the copy probe and the
  supervisor's copies to `~/kvdiag/supervisor/<ts>/` stay; they are how the
  next fault gets read.
- **No load copy is submitted off the step thread.** v0.4.4's poll thread
  (`load_poll_ms`) did, armed on the compute stream the last step captured,
  and the cluster faulted with Xid 32 after 27 loads, every one `via=poll`
  (ask7, 2026-10-14; evidence `~/kvdiag/wedge-20261014T171752Z` on both
  nodes). Why that wait was not enough is not known; no event a between-steps
  copy could wait on is known to be right. v0.4.8 removed the thread, and
  `load_poll_ms > 0` now only warns. Tests (phase1b C8) fail if the load
  handler starts a thread or submits a copy outside the step's `_collect`.
- **Class B is `cuMemcpyBatchAsync` itself (ask9, 2026-10-15).** This was
  reproduced with only ctypes and libcuda (DSpark `local-ops/kvbatchrepro.py`).
  `load_copy_api` (v0.4.10, default `batch`) can split a load's copy
  (`chunk`) or issue one `cuMemcpyAsync` per descriptor (`loop`). It works
  through a proxy for `ops` in the copy handler's module. The proxy acts only
  inside `DiskLoadHandler`'s own call (a thread-local), so stores and every
  other caller are untouched. `kvmgr.copyapi` must stay in `_PRESERVE`, or a
  reload silently puts every load back on `batch`.
- **`loop` is the fix, validated 2026-10-16** (vllm-proxy, DSpark `c119e21`):
  1,503 kvrace restores of the faulting shape with 0 Xid, then 15 h of
  production with 0 Xid, at no measurable restore cost. It is the boot
  setting through DSpark's `local-ops/flags.reference.json`. Without that
  key, any restart (a supervisor recovery included) comes back on `batch`.
  `chunk` is not safe: production faulted once on a 992-descriptor batch.
  The code default stays `batch`, as ask9 asked; never ship a flags file
  without the key.
- **Every change to the load path is gated on the reproducer:** `kvrace.py
  run --arm on --loads 1000` must stay clean, and a 700K restore must stay
  about 10 s idle. It is an actuator; the vllm-proxy session runs it, with the
  user's consent.

The write-up is vllm-proxy `docs/kv/history/2026-10-14-xid-outage.md`.

## Working agreement

- Work on `main`, and commit each finished, green unit of work.
- No pushes without asking. `origin` is the public GitHub repo
  (`github.com/sjhorn/vllm-kvmgr`, CI runs pristine mode there). `head` and
  `worker` are the two nodes, and a push there is a deploy (above).
- History was rewritten on 2026-10-14 (author email only, trees identical),
  so commit hashes cited before that date in other repos' docs are stale.
- Never commit secrets, keys or `data/`.
- **Where things get written:** rules about this package go here or in
  `README.md`. Measurements, deploy records and dated narrative go in
  `~/dev/python/vllm-proxy/docs/kv/` (`overview.md` is the current picture,
  `history/` the narrative). Engine and recipe rules go in DSpark's
  `CLAUDE.md`. A change to the cluster's state (a deploy, a knob default, a
  restart) should be recorded in the vllm-proxy docs, since that is what the
  operating session reads.
