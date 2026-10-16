# Plan: from the 0.31 spike to a 0.31 cluster (2026-10-16)

The spike (`docs/plan-0.31-local-spike.md`) showed kvmgr's disk tier working on
vLLM 0.31, on CPU and on a GB10. This plan takes the remaining steps in this
order:
1. ask9 onto the branch;
2. the disk tier mergeable;
3. DSpark on 0.31;
4. the validation on the cluster.

Nothing here changes the running cluster until step 4, and step 4 waits for
the user.

**Standing rules for this plan:**
- **No VM or container left running while unused.**
  - `scripts/local031.sh down` also stops the Apple `container` service (its
    VM and apiserver), and `up` starts it again. `status` says which.
  - On the nodes, every test container is `docker rm`'d when a run ends.
  - Test files live under one directory that is deleted afterwards (root-owned
    files go through a throwaway container).
- **The cluster's DSpark engine is never touched from here.** That means no
  stop, no restart, no push to `head`/`worker`, no edit of `~/kvoffload`.
  - GPU time on a node needs the engines down under `~/.engine-hold`, and the
    user's go-ahead each time.
  - Deploys and boots of the real engine belong to the vllm-proxy session.
- **DSpark's `CLAUDE.md` governs the engine and the recipe.** DSpark work
  happens on a branch in that repo, never on its `main` until step 4 is green.

## Step 1. ask9 onto the branch: DONE (`6567d4f`, 2026-10-16)

Rebased onto v0.4.10, and `load_copy_api` ported to 0.31's
resolved-at-init copy. Offline tests pass, plus three break-verifies. The
local six-group cycle (restore, Gate B, heal, reboot) re-ran with identical
replies. The GB10 check of `chunk` and `loop` waits for a free window: both
nodes are busy, so all testing is local until then.

The original step:

- Rebase `spike/vllm-0.31-local` onto `main` v0.4.10, and resolve the conflicts
  (likely in `diskio.py`, `flags.py`, `diag/ledger.py`, `diag/reload.py`).
- **Port `load_copy_api` to 0.31.**
  - 0.31's `SingleDirectionOffloadingHandler` resolves its copy routine once,
    at init (`self._swap_blocks_batch`, via `_select_swap_blocks_fn`), so the
    module-global proxy of v0.4.10 never runs there.
  - On 0.31, `copyapi.install_on` wraps that instance attribute of OUR load
    handler's inner handler only, with the same thread-local dispatch.
  - When 0.31 picked its Triton kernel (every page under 28 KiB), there is no
    batch call to split. `copy_api_why` says so, and the mode is reported as
    `batch`.
- Re-vendor nothing (the branch already carries 0.31.0).
  - phase17 gains a 0.31 section that pins `_select_swap_blocks_fn` and the
    instance attribute, and drives a fake handler built the 0.31 way.
  - Break-verify.
- **Verification:**
  - The offline suites, green in pristine mode (the seven GPU-manager suites
    stay red; see step 2).
  - The Mac container cannot exercise this: its copy is `HostCopyHandler`. A
    GB10 check of `chunk` and `loop` restores is optional, and needs a GPU
    window.

## Step 2. The disk tier mergeable (1–2 days, local)

**Decided 2026-10-16 (user): (a), its own release line.** Branch
`vllm-0.31`, versions `0.5.x`, rebased on `main` for each `main` release,
merged when the cluster moves. The options were:
- **(a) A release line on its own branch** (`vllm-0.31`, versions `0.5.x`),
  rebased on `main` per release and merged when the cluster moves.
  Recommended: `main` stays exactly what the cluster runs, and nothing on
  0.21 can regress.
- (b) One codebase for both, gated on the API present (as
  `loadfail`'s request-level path already is). Fewer branches, but every
  module carries two shapes, and 0.21 can regress through 0.31 code.

**Then, either way:**
- **The GPU-manager half.** Main plan item 3 decides whether prompt-block
  protection still pays. Until then it is skipped on 0.31 with the reason
  logged (as now). The seven red suites (phase1, 2, 3, 4, 6, 10, 11) become "skipped on 0.31, with a reason",
  never silent passes.
- **The diagnostics still written against 0.21:**
  - `diag/copyprobe.describe_copy`, the copy arithmetic;
  - `diag/tiering`, which wraps internals 0.31 made private;
  - `diag/ledger.step_hits`, which reads model runner V1's
    `input_batch.block_table` (0.31 defaults to V2, so it reports
    `in_step=None`).

  Fix them or turn them off on 0.31, each with a test.
- **The presence index is not per model** (found in the spike; true on
  `main` too). `presence.json` sits at the format root. Check what eviction
  does with an old fingerprint directory's files, and fix it on `main` if
  they are never reclaimed.
- **Stats port from the disk tier** (the spike's change): bring to `main`.
- **README/CLAUDE.md:** the vLLM window per line, the 0.31 patch list (with
  `failed_recving`), and the GB10 notes:
  - the 12.1a image;
  - `--safetensors-load-strategy eager`;
  - `--ulimit memlock=-1`.
- **Exit criterion:** both modes green on 0.21; pristine green on 0.31 with
  every skip explained; the CPU container cycle (one group, six groups, TP=2)
  re-run and identical.

## Step 3. DSpark on 0.31 (1–3 weeks; the real cost)

DSpark repo, branch `vllm-0.31`. What has to move:

| layer | today (0.21) | count |
|---|---|---|
| base image | `ghcr.io/bjk110/vllm-spark:unholy-fusion-prod-ready` (third-party, 0.21) | 1 |
| image overlays (`recipe/overlay`) | whole-file copies: DSv4 model and `sm120.py`, DSpark drafter and proposer, B12X MoE and MLA-sparse kernels, `gpu_model_runner.py`, `scheduler.py`, `envs.py`, registries, tokenizers | 21 files, ~26K lines |
| NVFP4 stages | `recipe/nvfp4/Dockerfile.stage-{a,b,c}` (`nvfp4_ds_mla`) | 3 |
| runtime mounts | patch5 detokenizer, patch6 `single_type_kv_cache_manager`, patch7 block-pool dedupe, patch8 `kv_cache_utils`, MNT-10 scheduler, two Anthropic fixes | 7 |
| plugins | prefill collector, safetensors E8M0, kvmgr | 3 |

**Prior work to start from:**
- `OFFICIAL_MAIN_PORT_PLAN.md` and `UPSTREAM_V024_STATUS.md` (2026-07): a
  Python-only overlay on a v0.24-era `main`. It passed smoke checks
  (`nvfp4_ds_mla` resolution, the 584-byte KV layout, the B12X mapping) but
  never got a full boot.
- Their upgrade lane (262K/fp8, then 1M/NVFP4, then 2/4/6 concurrency) is
  this step's boot ladder.
- The DSpark PR (vllm#46995) is merged upstream, so 0.31 carries DSpark
  natively.

**Stages:**
1. **Inventory (local, about 1 day).** For each of the 31 pieces: is it in
   0.31 already (drop), changed upstream (re-derive), or ours alone (carry)?
   Diff each overlay against its 0.21 original to isolate the delta, then
   look for that delta in 0.31.
   - The deliverable is a table in DSpark's repo, and the real effort
     estimate replaces "1–3 weeks".
2. **Base image (decision).** We need 0.31 built for sm_121 (12.1a). Options:
   - the node's `eugr/spark-vllm` (a 0.30.1rc1 dev build, a day before
     0.31.0; worked for the spike);
   - a newer build from the same source;
   - our own build of v0.31.0, which means hours of compile on a node.

   The image must match what upstream calls 0.31.0, or the kvmgr window is
   wrong.
3. **Re-derive the overlays (local, the bulk of it).**
   - Model and kernels first, since they gate any boot. Then the scheduler
     and KV manager patches (patch6, 7, 8 and MNT-10), each re-justified on
     0.31: 0.31's KV manager may already fix what patch6 fixes.
   - Offline checks where DSpark has them (`parity/`,
     `scripts/verify-overlay-sources.sh`).
4. **Build the image on a node.**
   - It needs CPU and RAM. While the engine runs, it holds most of the 128 GB
     unified memory, so the build may need a window. Ask each time.
   - The image is a new tag; the parked 0.21 image stays for rollback.
5. **Boot ladder, one node first if the model fits, then both** (engine down
   each time; user go-ahead):
   - 262K/fp8, then 1M/NVFP4;
   - a direct generation check;
   - concurrency 2, 4 and 6;
   - the garble checks (`AGENT_GARBLE_FIX.md`).

## Step 4. Validation on the cluster (the vllm-proxy session runs it)

With DSpark-0.31 booting and kvmgr's 0.31 line:
- **Restores:** a 700K-token restore at about 10 s idle; prefix hits through
  restarts; the boot walk.
- **Faults:**
  - the Xid reproducer, `kvrace.py run --arm on --loads 1000`, under each
    `load_copy_api` that passed on 0.21;
  - Gate B (corruption, recompute, heal) on both ranks.
- **Speed:** throughput and decode against the 0.21 baseline.
- **Promotion:** only then is the cluster moved to 0.31, with a rollback
  boot rehearsed.

## Decisions needed from the user

1. ~~Step 2: how 0.31 support lives~~: decided, its own branch.
2. Step 3.2: the base image source.
3. Steps 3.4 and 3.5: when GPU windows are available. Each one takes the
   cluster down.

Steps 1, 2 and 3.1 need none of these, so they can start now.
