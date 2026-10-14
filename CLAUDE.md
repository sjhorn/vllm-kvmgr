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
- Releases are tags (`v0.1.0` … `v0.4.0`), each
  naming the vLLM build it is proven on (`pyproject.toml` states the window).
- **It is the only copy.** The DSpark recipe repo
  (`~/dev/spark/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`) no
  longer holds one: its compose mounts `../vllm-kvmgr` on each node.
- Deploy: tests green → tag → `git push head main && git push worker main`
  (remotes `head`/`worker` = `/home/shorn/vllm-kvmgr` on spark-45ea and
  spark-da08, `receive.denyCurrentBranch=updateInstead`). Push only as part of
  a deploy the user asked for. DSpark's start script refuses to start unless
  both nodes are at the same commit. **A push does not reach a running engine**:
  it takes effect at the next restart, and the next restart may be a crash
  recovery nobody planned. **So never push code to the nodes you would not
  want booted.**
- **A version bump renames the dist-info** (`deploy/vllm_kvmgr-<ver>.dist-info`),
  and DSpark's `docker-compose.dspark.yml` names it in one mount line. Bump,
  regenerate, then change that line in DSpark and push BOTH repos together. A
  compose naming a dist-info that is not there has no entry point, so kvmgr
  silently does not load. `test_phase12_packaging.py` in dspark mode fails
  until the two agree.
- Restarting: **ask before every restart**, stating goal, change, cost (a
  10–22 min boot, every session on the cluster interrupted) and why it cannot
  wait.
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
- **`load_ops_max=2048`, `load_inflight_max=1`** as code defaults. Inflight 4
  caused GPU faults (Xid 32). **But until v0.4.0 the knob did not bound the
  GPU stage**: `_pump` checked it before a file read and `_collect` sent every
  finished read to the GPU unchecked. So "inflight 1 is safe" and "inflight 4
  faulted" were both measured without the bound, with up to 7 loads on the GPU
  at once. v0.4.0 bounds staging->GPU copies by it, and the read stage by it
  plus `load_read_ahead` (default 1, hot, not yet measured).

## The open question this repo is carrying (2026-10-14)

**Xid 32 recurs and is not proven ours.** Every instance since 10-09 coincided
with disk-tier transfers. The latest (2026-10-14 06:21:48) came 16.5h into a
boot, with a load submitted in the fault's second and 50 stores outstanding.
Leading hypothesis, unproven: a load writes a GPU block another transfer is
using. v0.3.0's write-ahead ledger exists to settle it:

- `/kvoffload/diag/ledger-r<rank>.jsonl` on each node (host:
  `~/kvoffload/diag/`), one line per GPU transfer BEFORE submit plus a `done`.
  After a fault, the open entries for the dead `pid` are what the GPU had.
  `kvmgr.diag.ledger.open_entries()` does the matching. An idle engine always
  leaves its last submit open until the next step, so read open entries
  against the Xid's timestamp (`journalctl -k`), not as a bare count.
- `LEDGER OVERLAP` warnings in the engine log, and `overlaps` on `GET
  /kv/workers` (`:8888`). It was 0 after the first 700K restore. A non-zero
  count is the first evidence for the hypothesis. It is not proof, because an
  entry's span is submit-to-OBSERVED completion.
- **The 2026-10-14 10:12:17 Xid 32 (head)** had 0 overlaps, which is evidence
  AGAINST the hypothesis, and three load copies open, all submitted within
  3 ms: two at the op cap (33 blocks × 62 refs = 2,046 descriptors) and one
  small one, with `load_inflight_max=1`. That burst is what v0.4.0 fixes. Since
  v0.4.0 each submit line carries `gpu_loads`, `gpu_stores` and `ops`,
  `/kv/workers` carries `max_gpu_loads`/`max_gpu_stores` (the deploy check is
  `max_gpu_loads == 1`), and a load submitted at or over the bound logs
  `LOAD BURST` once.
- The supervisor copies both nodes' ledger tails, logs and kernel logs to
  `~/kvdiag/supervisor/<ts>/` on the head before it restarts anything.

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
