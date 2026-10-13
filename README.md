# vllm-kvmgr — a hot-configurable KV cache manager and a cross-node disk prefix cache for vLLM

One `vllm.general_plugins` entry point, two halves:

1. **A KV cache manager for the full-attention MLA group**, registered through
   vLLM's public `spec_manager_map`. It decides which prompt blocks stay
   protected in GPU memory and which go first, and every policy knob moves on
   a running engine: the protection budget, how it is split across KV groups,
   and the eviction order.
2. **A disk-backed prefix cache**, `DSparkDiskOffloadingSpec` (generic alias
   `KvmgrDiskOffloadingSpec`), registered through vLLM's public
   `OffloadingSpecFactory`. Each tensor-parallel rank writes its **own** KV
   slice GPU → a small pinned staging pool → its own node's disk, and reads it
   back the same way. A prefix survives an engine restart.

**What it is for.** On the cluster it was built for, a 2,000-token delta on a
warm 770K-token prefix costs 0.9s, and the same prefix arrived at cold costs
430–795s. Measured with this package: a 700,445-token prompt costs **598s cold
and 11s after a full restart of both nodes (54x)**, with every lookup a hit.
A conversational prefix comes back at 98.7% after the GPU cache is dropped.

> **Where this lives.** This repository is the source of truth for kvmgr, with
> its full history carried over from the DSpark recipe repo, where it was
> developed. Releases are git tags (`v0.2.0` and on), each naming the vLLM
> build it is proven on. **If you are reading this inside the DSpark recipe
> at `packages/vllm-kvmgr/`, you are reading a vendored copy** — a squashed
> `git subtree` of one tag here, kept there because the cluster deploys by
> pushing that repo to its nodes. Do not edit the copy: change it here, tag,
> and `git subtree pull --prefix=packages/vllm-kvmgr <this repo> <tag> --squash`
> from the DSpark root.

## Requirements

- **vLLM `0.21.1rc1.dev339+g1967a5627bc3`.** That is the build it is proven on.
  `pyproject.toml` declares `vllm>=0.21.1rc1.dev0,<0.22`, but kvmgr wraps
  private internals (listed below), so a version inside that window is not a
  promise. To port it, re-vendor `tests/upstream/` from the target build and
  run the tests: an upstream change surfaces as a named failure, not as "the
  plugin did not load".
- **No patch to vLLM.** The DSpark recipe's overlays are optional; see "On a
  vLLM without the DSpark fork" for what differs.
- **`PYTHONHASHSEED` set to a fixed value, and never changed.** vLLM's block
  hash is a Merkle chain rooted at `NONE_HASH`, which is `os.urandom(32)` when
  the seed is unset. Every boot then computes different hashes, and the files
  a previous boot wrote become a permanent **miss**, not an error. Changing the
  value later orphans everything already written, just as silently.
- `--enable-prefix-caching`, and a writable directory on every node.

## Install

Into a vLLM image:

    pip install --no-deps vllm_kvmgr-0.2.0-py3-none-any.whl

The DSpark recipe instead bind-mounts `src/kvmgr` and
`deploy/vllm_kvmgr-0.2.0.dist-info` into site-packages. The dist-info is
**generated** (`python3 deploy/gen-dist-info.py`; `--check` fails when it is
stale) because the entry point vLLM looks for lives in it.

Registering is inert. The manager's policy is off until `kvmgr_enabled` is
set, and the disk tier is off until `KV_TRANSFER_CONFIG` names the spec:

    KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
      "kv_connector_extra_config":{"spec_name":"KvmgrDiskOffloadingSpec",
        "staging_blocks":256,"staging_load_blocks":256,"write_threads":4,
        "root_dir":"/kvoffload"}}'

## Configuration

Knobs are read from a JSON flags file, `VLLM_KV_FLAGS_FILE` (default
`/etc/vllm-kv-flags/flags.json`), which is re-read at most once a second. The
precedence is: **in-memory override** (set over HTTP), then `VLLM_<NAME>` env,
then the file, then the default.

**The authoritative list is `DEFAULTS` in `src/kvmgr/flags.py`** (50 knobs),
and `GET /kv` publishes every one with its effective value and **the source
that won**. A hand-written table here once listed 6 of 24, so there is none.
The defaults are inert by design: an empty file behaves exactly as upstream.
The safety bounds are the exception, because inert-by-default is right for
them:

- **`load_ops_max=2048`, `load_inflight_max=1`.** Four restores in flight
  killed the engine with a GPU fault (Xid 32) while the batch size was pinned
  at 2048. One costs nothing: 1 and 2 restore 700K in the same 12s.
- **`stall_fail_after_s=240`.** A store job that cannot complete fails with
  our line naming it, before NCCL's 600s watchdog aborts the worker in C++
  with no Python traceback.

The configuration the recipe actually runs is in the DSpark repo's
`local-ops/flags.reference.json`.

## Endpoints

| where | route | |
|---|---|---|
| stats listener, `127.0.0.1:<kvmgr_stats_port>` (0 binds nothing), engine-core process | `GET /kv` | the disk index, counters, policy, every knob with its source |
| | `GET /metrics` | Prometheus text, labelled by group, cause and victim rule |
| | `GET /flags` | effective knob values |
| | `POST /kv/verify` | re-verify keys on all ranks; answers **202** and applies at the next scheduler step |
| API server, only with `kvmgr_control_plane=1` | `GET /kv`, `GET /kv/workers` | per-rank handler state and the transfer ledger |
| | `POST /kv/knob`, `POST /kv/knob/clear` | set or clear an override in the API server, the engine core **and every worker**, with each rank's answer |
| | `POST /kv/drop_gpu_cache` | drop the GPU prefix cache, keeping the disk index; refused unless `gpu_cache_drop_allowed` |
| | `POST /kv/reload` | **absent** unless the hot-reload diagnostic is installed |

**The disk index lives in the engine core.** `GET /kv` on the API server
therefore answers `disk: null`, because it is the wrong process. Read the stats
listener. **The control plane shares the engine's own port**, so anything that
proxies that port exposes these routes to its clients.

## How the disk tier works

- **Store.** The scheduler decides what to offload. Our handler pairs blocks
  with keys **positionally** and checks the pairing on every transfer,
  refusing rather than writing another block's KV under a key. A job of any
  size is pumped over rounds through the staging pool. Each file is written to
  `.part` and renamed, because a file's existence **is** the presence check.
- **Presence.** A key becomes present only when every rank reports its slice
  written, which is the framework's own all-ranks aggregation. The index is
  persisted as an append-only **journal** with a snapshot as its compaction,
  and a SIGKILL, including one mid-compaction, loses nothing.
- **Boot.** The index is replayed, then every key is checked on **every rank**
  (file present, right size) before the API listens. This walk dominates boot
  time and grows with the index. It cannot be skipped: upstream's
  `offloading/worker.py` asserts on a failed load, so a vanished file would
  otherwise kill the engine.
- **Restore.** `lookup` answers from the index, and reuse is a contiguous run
  from token 0, exactly as on the GPU.
- **Growth.** A byte budget per node with LRU eviction **tail-first**.
  Evicting a prefix's head writes off everything behind it, so the tail goes
  first. A key with a load in flight holds a counted eviction hold. A
  free-space floor freezes stores before a disk fills. Files missing on any
  rank are condemned before a load can reach them (self-heal).
- **Fingerprint.** Files live under
  `<root>/dspark-kv-v1/<fingerprint>_r<rank>/`. The fingerprint hashes the
  model's content identity (`config.json` plus the safetensors
  `filename → blob` map, deliberately not the HF revision), the parallel
  sizes, cache dtype, every group's block size and layers, and our format
  version. A changed configuration misses by construction. An accidental
  layout change is caught **offline** by a pinned hash of the layout functions.

**Why not upstream's tiering spec.** `TieringOffloadingSpec` routes every tier
through one shared-memory CPU region per host. With one tensor-parallel rank
per node, rank 1's half reaches a different host's `/dev/shm` and goes nowhere:
every file it wrote had its second half zero, and nothing refused. The details
and four other findings are in the DSpark repo's `docs/UPSTREAM-FINDINGS.md`.

### This README was once confidently wrong about the disk tier

It used to carry a section headed **"The disk tier is not buildable here"**.
The section argued, correctly, that a KV cache manager holds no KV bytes, and
then concluded that a disk tier was impossible. The bytes are reachable from
the worker side, and the tier has since written and restored hundreds of
gigabytes. The section is named here rather than quietly deleted, because a
document that was wrong about its own central question is worth knowing about
when deciding how far to trust the rest.

## The manager's policies

- **`kvmgr_enabled` is read per call.** The manager class is chosen once at
  boot, so we are always constructed, and "off" means calling `super()`:
  literally upstream's code path, not our policy with default settings.
- **`protect_alloc=by_need`** splits the protection budget by need, not
  equally. A sliding-window group can only ever use its window, so it gets
  that and full attention gets the rest. On the reference geometry that is
  8,192 blocks (2,097,152 tokens) for group 0 against 3,643 under `equal`,
  2.25x. Read naively as "the same token extent per group" it would be a 21x
  regression, which is why the windows are load-bearing.
- **`evict_order`** is `fifo` (upstream), `lru`, or `lru_tail_first`. Measured
  live with four 12K-token prefixes: `lru` kept 87% of the surviving blocks
  reusable and `lru_tail_first` kept 100%.
- **Eviction is counted by cause** (`cap`, `pool`, `reset`) and by victim rule.
  A `cap` eviction on an idle pool is a policy bug; a `pool` eviction is
  capacity.

## Diagnostics (`kvmgr.diag`, opt-in)

| module | knob | default |
|---|---|---|
| hang dump: `faulthandler` on fatal signals plus `kill -USR1` for a live process | `diag_hang_dump` | **on** |
| transfer ledger: index ranges per transfer, logged on a fault | `diag_transfer_ledger` | **on** |
| numpy-scalar producer walk (the `.item()` **coercion** itself is core) | `diag_serializer_walk` | off |
| KV geometry snapshot file | `diag_geometry_snapshot` | off |
| shims for upstream's tiering spec (O_DIRECT, hybrid block sizes, verify) | `diag_tiering_shims` | off |
| hot reload of kvmgr in a running engine, and `/kv/reload` | `diag_hot_reload` | off |

These are installed at plugin load, so they are boot-time knobs. No functional
module imports `kvmgr.diag`; a test lint enforces that, and another proves that
with every knob off no diag module is imported at all.

## What it touches in vLLM

Public: `spec_manager_map`, `OffloadingSpecFactory.register_spec`, the
`vllm.general_plugins` entry point. Private, and the reason for the version
window:

- wraps `KVCacheCoordinator.__init__` (to read the real group geometry),
  `OffloadingConnectorScheduler.__init__`, `EngineCore.collective_rpc`,
  `EngineCore.step` (a one-shot boot step) and the API server's router hook;
- replaces `MsgpackEncoder.enc_hook` to coerce numpy scalars. Without it, one
  numpy scalar kills the engine core's **output thread**, and every request
  hangs with nothing logged;
- adds `kvmgr_*` methods to `vllm.v1.worker.gpu_worker.Worker` for
  `collective_rpc`;
- reads offload-scheduler and connector-worker privates, and finds a few of
  them by walking caller frames.

## On a vLLM without the DSpark fork

The DSpark recipe's `patch6` carries three mechanisms that are not kvmgr: SWA
recycle, free-queue prepend and static-head pin. When patch6 is present,
`kvmgr.compat.dspark` bridges to it. Its flag reader shares kvmgr's cache, an
override reaches its knobs, its knobs are declared, and its pending-blocks
hook gets kvmgr's provider. Without patch6:

- those knobs do not exist, and `/kv` does not list them;
- **releasing a protected block does not consult in-flight KV transfers**,
  because that hook is patch6's;
- with our policy off, the cap is upstream's own legacy cap, which knows no
  fraction;
- if the group geometry cannot be read, the manager uses upstream's cap rather
  than guessing a group count;
- registration relies on the entry point alone (patch6 also offers a lazy
  in-module hook).

## What this does not guarantee

- **A failed transfer is fatal upstream**, not a miss: `offloading/worker.py`
  is a bare `assert transfer_result.success`. We make the reachable cases
  impossible (boot verification, eviction holds, self-heal), but an `EIO` or
  `ENOSPC` mid-transfer will still take the engine down.
- **If the model directory cannot be read, the fingerprint degrades to the
  model name.** It is published as `model_identity: unknown`.
- **The space budget is per node and bounds the index, not the disk.** Files
  the index cannot name are invisible to it; the DSpark repo's
  `local-ops/kvsweep.sh` removes them.
- **GPU faults on the first large restore after a boot** (Xid 32) have been
  seen twice, with every index in range. They are not proven to be ours.

## Tests

    bash tests/run-all.sh

Sixteen suites, no GPU, no cluster, a few seconds, run against vLLM's own
sources with numpy and torch stubbed. They run twice: on pristine vLLM, and
with the DSpark patches. See `tests/README.md`.
