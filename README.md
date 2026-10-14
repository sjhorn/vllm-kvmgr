# vllm-kvmgr

A vLLM plugin that does two jobs:

1. **It decides which prompt blocks stay in GPU memory.** You can change how
   it decides while the engine is running.
2. **It saves prompt blocks to disk.** The saved blocks survive a restart, so
   a long prompt you sent before does not have to be computed again.

```
                         +------------------------------+
   your prompt  ------>  |            vLLM              |
                         |                              |
                         |  +------------------------+  |
                         |  | 1. KV cache manager    |  |  which blocks stay
                         |  |    (in GPU memory)     |  |  on the GPU
                         |  +------------------------+  |
                         |  +------------------------+  |
                         |  | 2. disk prefix cache   |--+--> /kvoffload on
                         |  |    (on each node)      |  |     each node's disk
                         |  +------------------------+  |
                         +------------------------------+
```

## Why use it

Some words first:

- A **prompt** is turned into **KV cache** blocks on the GPU. Making them is
  the slow part of reading a long prompt.
- A **prefix** is the start of a prompt. If two prompts start the same way,
  the second one can reuse the first one's blocks.

Without this plugin, those blocks are lost when the GPU runs out of room or
the engine restarts. Then a long prompt has to be read again from the start.

On the cluster this was built for:

| case | time |
|---|---|
| a 700,445-token prompt, nothing cached | 598 s |
| the same prompt after restarting both nodes | **11 s** (54x faster) |

After the GPU cache is dropped, 98.7% of a chat's earlier prompt comes back
from disk.

## What you need

- **vLLM `0.21.1rc1.dev339+g1967a5627bc3`.** This is the version it is tested
  on. `pyproject.toml` allows `>=0.21.1rc1.dev0,<0.22`, but the plugin uses
  some private parts of vLLM, so other versions in that range may break. To
  move to a new vLLM, copy that version's files into `tests/upstream/` and run
  the tests. A change in vLLM shows up as a test failure with a name.
- **No changes to vLLM itself.** The DSpark recipe adds some patches, but you
  do not need them. See [Without the DSpark patches](#without-the-dspark-patches).
- **`PYTHONHASHSEED` set to a fixed number, and never changed.** vLLM names
  each block with a hash. If the seed is not set, every start makes different
  names, and the files on disk are never found again. There is no error; you
  just get misses. Changing the number later has the same effect.
- `--enable-prefix-caching`, and a folder you can write to on every node.

## Install

Into a vLLM image:

```sh
pip install --no-deps vllm_kvmgr-0.4.6-py3-none-any.whl
```

The DSpark recipe does not pip install. It mounts `src/kvmgr` and
`deploy/vllm_kvmgr-0.4.6.dist-info` into site-packages instead. vLLM finds the
plugin through the entry point in that dist-info folder. The folder is made by
a script:

```sh
python3 deploy/gen-dist-info.py          # write it
python3 deploy/gen-dist-info.py --check  # fail if it is out of date
```

Installing the plugin changes nothing on its own:

- The GPU manager does nothing until you set `kvmgr_enabled`.
- The disk cache does nothing until `KV_TRANSFER_CONFIG` names it:

```sh
KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"spec_name":"KvmgrDiskOffloadingSpec",
    "staging_blocks":256,"staging_load_blocks":256,"write_threads":4,
    "root_dir":"/kvoffload"}}'
```

## Settings (knobs)

Settings live in a JSON file. Set the path with `VLLM_KV_FLAGS_FILE`. The
default is `/etc/vllm-kv-flags/flags.json`. The file is read again at most
once a second, so most changes take effect without a restart.

When the same knob is set in more than one place, the first one in this list
wins:

```
  1. an override set over HTTP      (POST /kv/knob)
  2. an env var                     (VLLM_<NAME>, for example VLLM_KVMGR_ENABLED)
  3. the flags file
  4. the default in the code
```

The full list is `DEFAULTS` in `src/kvmgr/flags.py` (52 knobs). `GET /kv`
shows every knob, its value, and which of the four places it came from. There
is no copy of the list here, because a copy goes out of date.

An empty flags file behaves the same as plain vLLM. A few safety limits are
the exception. They are on by default because turning them off has crashed
the engine:

- **`load_ops_max=2048` and `load_inflight_max=1`.** These limit how much
  data is copied from disk back to the GPU at once.
  - With 4 copies at once, the engine crashed with a GPU fault (Xid 32).
  - With 1 or 2, a 700K-token prompt came back in the same 12 s.
  - **Before v0.4.0, `load_inflight_max` did not really work.** Finished disk
    reads went to the GPU with no check, up to 7 at once. So all the numbers
    above were measured without the limit. Since v0.4.0 the limit is real.
  - `load_read_ahead` (default 1) lets the next block group read from disk
    while the current one copies to the GPU.
- **`load_wait_compute=1`. Must stay 1 on a serving engine.** A copy from
  disk back to the GPU first waits for work already queued on the compute
  stream, as vLLM 0.31 does. vLLM 0.21 waits only when saving, and with async
  scheduling a reload could then write blocks the running step was still
  using. **That was the cause of the Xid 32 faults** (2026-10-14: 3,270+ loads
  clean with it at 1; Xid 32 within 36 loads at 0). 0 exists only for the
  reproducer (DSpark `local-ops/kvrace.py --arm off`). While it is 0 the log
  repeats a WARNING every 5 minutes and `/kv/workers` counts
  `compute_wait_off`. Each load's ledger line says whether the wait was in
  force (`wait_compute`).
- **`load_poll_ms=0` (off).** Normally a finished GPU copy is noticed once per
  scheduler step, so under heavy prefill a restore moves at most
  `load_inflight_max` rounds per step (an 858K restore took 166 s busy against
  11 s idle). Above 0, a thread checks every that many ms and starts the next
  round between steps. That round still waits on the compute stream the last
  step used, so the Xid 32 fix holds. Not yet measured on the cluster.
  `/kv/workers` shows what a step moves: `steps_loading`, `blocks_to_gpu`
  (and `blocks_to_gpu_poll`, the poll thread's share), `blocks_step_max`,
  `blocks_last_step`, `blocks_per_loading_step`, `poll_ms`, `poll_thread`.
  Compare settings by the change in `blocks_to_gpu / steps_loading` over one
  restore. Can be changed while running.
- **`disk_load_failure_recompute=0` (off).** In stock vLLM a failed restore
  (missing file, wrong size, bad checksum) stops the engine: the offloading
  connector asserts. With this on, the restore is reported to vLLM as invalid
  blocks instead, the request is rolled back, and its keys leave the index
  (and a provably bad file is deleted, so the next save rewrites it). vLLM
  then follows `kv_load_failure_policy` in `KV_TRANSFER_CONFIG`: `recompute`
  reschedules the request, `fail` (vLLM's default) ends it with an error.
  kvmgr also fixes vLLM 0.21's rollback for models with several KV groups,
  which would otherwise crash (borrowed from sparkcache). Not yet tested on the
  cluster. `/kv/workers`: `failed_recompute`, and `loadfail` per worker.
- **`disk_checksum_write=0`, `disk_checksum_verify=0` (off).** A CRC32 of each
  KV file, stored in the file's `user.kvmgr.crc32` extended attribute, so the
  file itself does not change. Write tags new files; verify checks tagged
  files on restore. Files written before, or on a filesystem without extended
  attributes, read as unchecked (`checksum_absent`). A mismatch fails the
  restore, which stops the engine unless `disk_load_failure_recompute` is on.
  `/kv/workers`: `checksum_set`, `checksum_set_errors`, `checksum_ok`,
  `checksum_absent`, `checksum_bad`, `bad_unlinked`.
- **`stall_fail_after_s=240`.** If a save to disk gets stuck, it fails after
  240 s with a log line that says which job it was. Without this, NCCL kills
  the worker after 600 s and leaves no Python trace.

The settings the DSpark recipe runs with are in the DSpark repo, in
`local-ops/flags.reference.json`.

## HTTP endpoints

vLLM runs in more than one process. Each endpoint lives in one of them:

```
  +------------------------------+        +-------------------------------+
  | API server (vLLM's own port) |        | engine core                   |
  |                              |  ----> |   stats listener              |
  | only with                    |        |   127.0.0.1:<kvmgr_stats_port>|
  |   kvmgr_control_plane=1      |        |                               |
  |                              |        |   holds the disk index        |
  +------------------------------+        +-------------------------------+
                 |
                 v
  +-------------+   +-------------+
  | worker r0   |   | worker r1   |   one per GPU; they move the bytes
  +-------------+   +-------------+
```

**Stats listener** (in the engine core; `kvmgr_stats_port=0` turns it off):

| route | what it does |
|---|---|
| `GET /kv` | the disk index, counters, settings, and every knob with where it came from |
| `GET /metrics` | Prometheus metrics |
| `GET /flags` | the current knob values |
| `POST /kv/verify` | check saved blocks on every node again. Returns **202** and runs at the next scheduler step |

**API server** (only with `kvmgr_control_plane=1`):

| route | what it does |
|---|---|
| `GET /kv`, `GET /kv/workers` | the state of each worker, and the transfer ledger |
| `POST /kv/knob`, `POST /kv/knob/clear` | set or clear an override in every process, and show each one's answer |
| `POST /kv/drop_gpu_cache` | empty the GPU prefix cache but keep the disk. Refused unless `gpu_cache_drop_allowed` is on |
| `POST /kv/reload` | only exists when the hot-reload diagnostic is on |

Two things to watch for:

- **The disk index is in the engine core.** `GET /kv` on the API server shows
  `disk: null`, because that process does not have it. Ask the stats listener.
- **The control plane uses vLLM's own port.** If you put that port behind a
  proxy, your users can reach these routes too.

## How the disk cache works

Each GPU (each "rank") holds part of every block. Each rank saves its own part
to its own node's disk. The data goes through a small pinned buffer in CPU
memory, called **staging**.

```
  node A (rank 0)                           node B (rank 1)

  GPU memory                                GPU memory
     |   ^                                     |   ^
save |   | restore                        save |   | restore
     v   |                                     v   |
  staging (pinned CPU memory)               staging (pinned CPU memory)
     |   ^                                     |   ^
     v   |                                     v   |
  /kvoffload/dspark-kv-v1/<fp>_r0/          /kvoffload/dspark-kv-v1/<fp>_r1/
```

A block counts as saved only when **both** ranks have written their part.

### Saving

1. vLLM's scheduler picks which blocks to save.
2. The plugin matches each block with its name (key), and checks the match on
   every transfer. If the match is wrong, it refuses to write.
3. Big jobs are split into rounds, so they fit in the staging buffer.
4. Each file is written as `.part` and then renamed. A file that exists is
   always a whole file.

### Restoring

```
  disk file --(read thread)--> staging --(GPU copy)--> GPU memory

  load_inflight_max   = how many GPU copies may run at once  (default 1)
  load_read_ahead     = how many more rounds may read from disk while
                        they wait                            (default 1)
  load_wait_compute   = the GPU copy waits for queued compute work first
                        (default 1; 0 is the Xid 32, never on a serving engine)
  load_poll_ms        = check for finished copies this often between steps
                                                       (default 0 = off)
```

`lookup` checks the index. A prompt reuses blocks in one unbroken run from
the start, the same as vLLM does on the GPU.

### Keeping track of what is saved

- The index of saved blocks is kept in a journal (an append-only log) plus a
  snapshot. A crash, even in the middle of writing the snapshot, loses
  nothing.
- **At startup**, every saved block is checked on every node (does the file
  exist, is it the right size) before vLLM starts taking requests. This is
  the slowest part of startup, and it grows with the cache. It cannot be
  skipped: in vLLM, a failed restore stops the engine, so a missing file
  would crash it.
  - The engine sends the index to every node in chunks
    (`disk_boot_verify_chunk`, default 40,000 keys). Each node checks a
    chunk's files on `disk_verify_threads` threads (default 16, hot; 1 is the
    old one-at-a-time walk). The answer is the same either way: a key is
    dropped if any node is missing its file, has it at the wrong size, or
    could not check it.
  - The walk ends with one log line, `BOOT WALK done`: keys, wall time, RPCs
    and chunk size, the slowest chunk, and keys/s and threads per node. A
    chunk taking over a quarter of the 300 s RPC timeout logs a WARNING.

### Running out of space

- Each node has a size budget. When it is full, the least recently used
  prompts are removed, **last blocks first**. A prompt is only useful from its
  start, so removing the first block would waste all the blocks after it.
- A block that is being restored right now cannot be removed.
- If the disk gets too full, saving stops before the disk fills up.
- If a file is missing on any node, that block is dropped from the index
  before anything tries to restore it.

### The folder name (fingerprint)

Files go in `<root>/dspark-kv-v1/<fingerprint>_r<rank>/`. The fingerprint is
a hash of:

- the model files (`config.json` and the safetensors files),
- the parallel sizes and the cache data type,
- each group's block size and layers,
- the plugin's file format version.

If any of these change, the new setup uses a new folder and simply misses.
Old files are never read by the wrong setup. A test also fails if someone
changes the file layout by accident.

### Why not vLLM's own tiering spec

vLLM's `TieringOffloadingSpec` sends every tier through one shared-memory area
per machine. With one rank per node, rank 1's half goes to the wrong machine's
`/dev/shm`. Every file it wrote had a second half full of zeros, and nothing
caught it. The DSpark repo's `docs/UPSTREAM-FINDINGS.md` has the details and
four other findings.

## How the GPU manager decides

- **`kvmgr_enabled` is checked on every call.** When it is off, the plugin
  runs vLLM's own code, not its own code with default settings.
- **`protect_alloc=by_need`** splits the protected space by what each group
  can use. A sliding-window group never needs more than its window, so it
  gets that, and full attention gets the rest. On the reference model, full
  attention gets 8,192 blocks (2,097,152 tokens) instead of 3,643 with
  `equal`, which is 2.25x more.
- **`evict_order`** can be `fifo` (vLLM's default), `lru`, or
  `lru_tail_first`. In a test with four 12K-token prompts, `lru` kept 87% of
  the kept blocks usable and `lru_tail_first` kept 100%.
- **Every removal is counted by reason** (`cap`, `pool`, `reset`). A `cap`
  removal when the pool is not full is a bug. A `pool` removal just means the
  GPU is full.

## Diagnostics (`kvmgr.diag`)

Optional tools for finding problems. Each one has a knob:

| tool | knob | default |
|---|---|---|
| hang dump: a stack trace on a crash, or on `kill -USR1` | `diag_hang_dump` | **on** |
| transfer ledger: which blocks each transfer used, logged on a fault | `diag_transfer_ledger` | **on** |
| copy probe: is a stuck copy finished on the GPU, and its addresses | `diag_copy_probe` | **on** |
| boot self-test: replay given load copies on an idle GPU | `diag_copy_selftest` | off |
| the ledger file (below), and the `LEDGER OVERLAP` check. Can be changed while running | `diag_ledger_file` | **on** |
| find where numpy numbers leak into vLLM's messages | `diag_serializer_walk` | off |
| write a file with the KV cache layout | `diag_geometry_snapshot` | off |
| fixes for vLLM's tiering spec | `diag_tiering_shims` | off |
| reload the plugin's code in a running engine | `diag_hot_reload` | off |

These are set up when the plugin loads, so changing them needs a restart.
`diag_ledger_file` is the one exception.

The main code never imports `kvmgr.diag`. A test checks this. Another test
checks that with every diag knob off, no diag code is loaded at all.

### Reading the ledger after a GPU fault

Each worker writes `<tier parent>/diag/ledger-r<rank>.jsonl`. It gets a
`submit` line **before** each GPU transfer starts, and a `done` line when the
plugin sees it finish.

```
  {"ev":"submit","kind":"load","inner":1000051,"boot":"1791974370-2e92673a",...}
  {"ev":"done",  "kind":"load","inner":1000051,"boot":"1791974370-2e92673a",...}
```

After a crash:

1. Find the `boot` value of the process that died. (Use `boot`, not `pid`.
   Inside a container, the worker gets the same pid every time.)
2. The `submit` lines from that boot with no `done` are what the GPU was
   working on. `kvmgr.diag.ledger.open_entries()` finds them for you.
3. Compare their `ts` times with the Xid time in `journalctl -k`.

Useful fields on a `submit` line:

| field | meaning |
|---|---|
| `gpu_loads`, `gpu_stores` | how many transfers were already on the GPU |
| `ops` | how many copy steps this one transfer has |
| `overlap` | another transfer was using some of the same GPU blocks |
| `in_step` | (loads) how many of the target blocks the current batch is using. `null` means it could not be read, which is not the same as 0 |
| `req`, `req_in_step` | (loads) the request this load is for, and whether it is in the current batch |
| `wait_compute` | (loads) whether the copy waited for queued compute work (`load_wait_compute`). `null` means it could not be set up |
| `via` | (loads) `step`, or `poll` when the poll thread submitted it (`load_poll_ms`). A `poll` line has `in_step: null`: the block table cannot be read from that thread |
| `dst_span`, `dst_contig_runs` | (loads) highest minus lowest target block id, and how many unbroken runs they form |
| `bids_sha` | (loads) a short id for the exact block set (sha1 of the `bids` ranges, 12 hex), to match the same copy across ranks |
| `tensor_idx_set_hash`, `ref_tensors`, `ref_dup_tensors` | (loads) which KV tensors the group's layers use, and how many layers share a tensor with another |
| `compute_stream`, `copy_stream` | (loads) the CUDA stream handles at submit. `null` with a `_why` if unreadable |

A `done` line is written when the plugin *notices* the transfer finished,
which can be up to one scheduler step late. So `overlap` means "the plugin
counted these blocks in two transfers at the same time". That is needed for
the suspected bug, but it does not prove it.

### When a copy gets stuck

`diag_copy_probe` (on by default) watches every copy that is still open. It
runs on the watchdog thread, which keeps going when the worker's main loop is
blocked, and asks the GPU through the copy's own end event (a check that never
blocks). If a copy is open for more than 30 s, or the watchdog says the store
path has stalled, it:

- logs `COPY PROBE ... copy_done_on_gpu=...` once. `False` means the GPU has
  not finished the copy. `True` means the GPU finished it but the worker never
  noticed. `null` means the GPU could not be asked, and gives the reason;
- writes an `ev:"stuck"` line to the ledger;
- writes `<tier parent>/diag/copydump-r<rank>-<kind>-<inner>.json`. This has
  every copy step with its source and target address and size, and checks
  that each target is inside its tensor and its memory allocation, the address
  alignment, and whether two targets in the one copy overlap.

A copy that fails to start is dumped the same way. `/kv/workers` shows the
counts under `copy_probe`.

### Boot self-test

`diag_copy_selftest` (off by default, read when the plugin loads) replays
load copies before the engine serves anything, while every GPU block is free.
It fills staging with a known pattern, copies it into the given block ids with
vLLM's own copy code, reads the blocks back and compares them. Each copy waits
at most 10 s (by checking, never by blocking), and the whole test at most
60 s. A control set (blocks 1..n) runs first, then each recorded set. A copy
that times out or faults stops the test.

The result goes to `<tier parent>/diag/selftest-r<rank>.json` after every
step, so a test that hangs still shows which copy it was on. The 16-block set
from the 2026-10-14 12:36 hang is built in. To add others, write
`<tier parent>/diag/selftest-sets.json`:

```json
{"sets": [{"name": "job902", "bids": [1988, 1990, ...], "sha": "7513f576f389"}]}
```

If `sha` is given and the ids do not match it, the set is skipped. This test
writes into GPU KV blocks, so turn it on only for a planned boot.

## What it changes inside vLLM

Public hooks: `spec_manager_map`, `OffloadingSpecFactory.register_spec`, and
the `vllm.general_plugins` entry point.

Private parts (this is why the vLLM version matters):

- wraps `KVCacheCoordinator.__init__`, `OffloadingConnectorScheduler.__init__`,
  `EngineCore.collective_rpc`, `EngineCore.step` (once, at startup) and the
  API server's router hook;
- replaces `MsgpackEncoder.enc_hook` to turn numpy numbers into plain Python
  numbers. Without this, one numpy number stops the engine's output thread,
  and every request hangs with nothing in the log;
- adds `kvmgr_*` methods to `vllm.v1.worker.gpu_worker.Worker`, for
  `collective_rpc`;
- reads private fields of the offload scheduler and connector worker, and
  finds some of them by looking up the call stack.

## Without the DSpark patches

The DSpark recipe's `patch6` adds three features that are not part of this
plugin: SWA recycle, free-queue prepend and static-head pin. When `patch6` is
there, `kvmgr.compat.dspark` connects the two: they share settings, and
overrides reach `patch6`'s knobs too.

Without `patch6`:

- those knobs do not exist, and `/kv` does not list them;
- **freeing a protected block does not check for KV transfers in progress**,
  because that check lives in `patch6`;
- with the plugin's GPU policy off, the limit is vLLM's own;
- if the group layout cannot be read, the manager uses vLLM's own limit
  instead of guessing;
- the plugin loads only through the entry point.

## Known limits

- **In vLLM, a failed restore stops the engine.** It is a plain
  `assert transfer_result.success` in `offloading/worker.py`. The plugin
  avoids the cases it can predict (startup checks, removal holds, dropping
  missing files). A disk error (`EIO`) or a full disk (`ENOSPC`) in the
  middle of a transfer will still stop the engine.
- **If the model folder cannot be read, the fingerprint falls back to the
  model name.** `/kv` then shows `model_identity: unknown`.
- **The size budget counts only files in the index.** Files the index does not
  know about are not counted. The DSpark repo's `local-ops/kvsweep.sh` removes
  them.
- **GPU faults (Xid 32) have happened while blocks were moving to or from
  disk.** We have not proven what causes them. The ledger above is how we are
  looking.

## Tests

```sh
bash tests/run-all.sh; echo $?
```

- 17 test files. No GPU and no cluster needed. Takes a few seconds.
- They run against vLLM's own source files, with numpy and torch faked.
- Everything runs twice: once on plain vLLM, and once with the DSpark patches.
  The DSpark run needs `KVMGR_DSPARK_REPO` set to a DSpark recipe checkout.
  Without it, every DSpark test fails. To run only the plain-vLLM tests, use
  `MODES=pristine bash tests/run-all.sh`.
- Check the exit code, not the last line.

See `tests/README.md` for more.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`. `tests/upstream/` holds
unchanged vLLM source files under the same license.
