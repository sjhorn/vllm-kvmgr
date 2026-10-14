# vllm-kvmgr

A vLLM general plugin (`kvmgr = kvmgr:register`): a hot-configurable KV cache
manager for the MLA group, plus a disk prefix cache that survives restarts.
`README.md` is the real documentation. This file covers how to maintain it.

## Where it lives

- Releases are tags (`v0.2.0` …), each
  naming the vLLM build it is proven on (`pyproject.toml` states the window).
- **It is the only copy.** The DSpark recipe repo
  (`~/dev/spark/DeepSeek-v4-Flash-0731-DSpark-1M-NVFP4-KV-2x-DGX-Spark`) no
  longer holds one: its compose mounts `../vllm-kvmgr` on each node.
- Deploy: tests green → tag → `git push head main && git push worker main`
  (both nodes, `/home/shorn/vllm-kvmgr`). DSpark's start script refuses to
  start unless both nodes are at the same commit. A push does not reach a
  running engine, so it takes effect at the next restart. Deploying and
  restarting otherwise follow **DSpark's `CLAUDE.md`**, including asking before
  every restart.

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
KVMGR_DSPARK_REPO=<DSpark checkout> bash tests/run-all.sh; echo $?
```

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
  caused GPU faults (Xid 32).

## Working agreement

- Work on `main`, and commit each finished, green unit of work.
- No pushes or remotes without asking. There is no remote yet.
- Never commit secrets, keys or `data/`.
- Measurements and history live in `~/dev/python/vllm-proxy/docs/kv/`.
