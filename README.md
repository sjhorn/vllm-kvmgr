# kvmgr — a KV cache manager and a disk-backed prefix cache for vLLM

Two things, in one `vllm.general_plugins` entry point:

1. **A KV cache manager** for group 0 (full MLA), replacing upstream's via the
   public `spec_manager_map`, with every policy knob movable on a running
   engine — protection budget, allocation split, eviction order.
2. **A disk-backed prefix cache** (`DSparkDiskOffloadingSpec`), registered
   through the public `OffloadingSpecFactory`, which writes KV **GPU → disk
   worker-side** and reads it back, so a prefix survives an engine restart.

**The second is separable from the first.** `diskspec.py` and `diskio.py` import
only the standard library, `vllm.logger`, the two public `vllm.v1.kv_offload`
modules and this package's own `flags` — no overlay, no patched upstream file,
no dependency on the manager half. If you want the disk tier on a stock vLLM,
that is the subset to take.

**What it is for, in one number.** On this cluster a 2,000-token delta on a warm
770K prefix costs **0.9 s**; the same 770K arrived at cold costs **686 s**. An
engine restart or a head eviction is the whole difference, which is why
disk-backed restore is worth real risk.

## If you are reading this to take something

- **The disk tier only**: `kvmgr/{diskspec,diskio,flags}.py` plus a register
  shim that calls `OffloadingSpecFactory.register_spec`. No upstream file is
  patched.
- **What we found wrong in upstream `kv_offload`** while building it, with
  verified line numbers and what each cost: `docs/UPSTREAM-FINDINGS.md`. Five
  items, one of which is silent data corruption on a cross-node tensor
  parallel.
- **The offline harness**: `local-ops/kvharness/`, 1,257 assertions, no GPU and
  no cluster, loading the real upstream sources with numpy and torch stubbed.
  Start here; it is the fastest way to see what the contract is.

## Requirements

- **vLLM with `v1/kv_offload`** — developed against
  `0.21.1rc1.dev339+g1967a5627bc3`. The contract it depends on is pinned by
  `local-ops/kvharness/test_phase0_disk.py` against the real upstream sources,
  so an upstream bump that moves it fails as a named assertion rather than at
  boot as "the plugin did not load".
- **`PYTHONHASHSEED` set to a fixed value, and never changed.** This is not
  optional and it is the single easiest way to get nothing but misses. vLLM's
  block hash is a Merkle chain rooted at `NONE_HASH`, which is
  `os.urandom(32)` when `PYTHONHASHSEED` is unset — so every block hash chains
  off 32 random bytes chosen fresh at each process start, and the files a
  previous boot wrote are keyed by hashes this boot **cannot compute**. The fs
  lookup is `os.path.exists`, so that is a **miss, not an error**, for ever,
  with the data sitting right there. Measured before the fix: 54,070 queries,
  0 hits, 27 GB on disk. **Changing the value later orphans every file already
  written**, silently, as a miss.
- **`--enable-prefix-caching`**, and a writable directory per node for the tier.
- Selection is explicit: nothing happens unless `KV_TRANSFER_CONFIG` names the
  spec.

		KV_TRANSFER_CONFIG='{"kv_connector":"OffloadingConnector",
		  "kv_role":"kv_both",
		  "kv_connector_extra_config":{
		    "spec_name":"DSparkDiskOffloadingSpec",
		    "staging_blocks":256,"write_threads":4,"root_dir":"/kvoffload"}}'

  Registering the name is **inert** — the committed default is "no offload at
  all", so a restart without this reverts to known-good rather than to a
  half-built path.

## Why it exists

A restart of this engine costs a **672–795 s** cold prefill (measured
2026-10-06: 725K tokens in 795 s, 767K in 672 s). Re-tuning a policy knob must
never cost that. Today the single most valuable knob —
`VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION` — is read straight off `os.environ`
(`patch6:619`), so it is restart-only.

## Delivery

A `vllm.general_plugins` entry point, exactly as
`patches/safetensors_e8m0_plugin` is. **Not** a fourth whole-file overlay:
patch6 is 1,809 lines of upstream that must be re-cut on every image bump, and
this stack already spent a month pinned to an August engine for that reason. A
subclass registered into `spec_manager_map` is a delta rather than a copy.

Two mount lines in `docker-compose.dspark.yml`, beside the other plugins. **No
image rebuild** — Tier-4 rebuilds trigger only on `recipe/overlay/` moving ahead
of the image.

	git push head local/main && git push worker local/main
	./stop-deepseek-v4-flash-dspark.sh && ./start-deepseek-v4-flash-dspark.sh

**A PUSH IS NOT A RESTART, AND `start-…sh` ALONE IS NOT A RECREATE.** Measured
2026-10-07: running the start script against an already-running engine printed
`Container … Running` and exited 0 — because the container *spec* had not
changed, so compose had nothing to do. A bind-mounted `.py` edit is invisible to
compose, and the live processes keep the modules they imported at boot. The
second deploy of the day looked successful and ran the previous code.

So an engine-side Python change needs `stop` then `start` (or
`up -d --force-recreate`). The tell is the **container id**: if it is the same
id as before, nothing was reloaded.

## The kill switch is read PER CALL, and that is not a detail

The plan says our manager "falls through to upstream's class when false". **It
cannot.** The manager class is chosen once, by `get_manager_for_kv_cache_spec`,
while the engine is starting — so a class swap is only achievable at boot, which
is the restart the requirement forbids.

So we are **always** registered and **always** constructed, and `kvmgr_enabled`
is read inside each override. False means `super()`. Flipping it off is
therefore literally upstream's code path, not "our policy with today's
defaults", and the test asserts that by its observable signature: with our
policy off the cap comes from `os.environ` and the flags file is ignored.

## The knobs

All in `/var/tmp/vllm-kv-flags/flags.json` on the host (`/etc/vllm-kv-flags`
inside the container), re-stat'ed at most once a second. **Defaults are chosen
so an empty file behaves exactly as today.**

**THE AUTHORITATIVE LIST IS NOT IN THIS FILE, and that is deliberate.** A
hand-written table here listed six knobs while twenty-four were declared, and
said the disk ones "cannot" be built while the tier was writing — the same way
the published policy block listed seven of twenty-four. Enumerate from the
declaration instead:

- `DEFAULTS` in `kvmgr/flags.py` is the source of truth.
- `GET /kv` on the stats port publishes **every** declared knob with its
  **effective value and the source that won** — override (control plane) beats
  env beats the flags file beats the default. A knob reading `default` while
  you believe you set it is almost always an env pin, or a flags file on the
  other node.
- A test audits both directions: every knob read anywhere must be declared, and
  every declared knob must be published. It was written after finding
  `head_pin_partial` read by the manager and declared nowhere — a knob that
  governed something and that no reader could discover.

The handful whose values change behaviour most:

| flag | default | effect |
|---|---|---|
| `kvmgr_enabled` | 0 | our policy vs upstream's, re-read per call |
| `protected_fraction` | 0.30 | the protection budget |
| `protect_alloc` | `equal` | `equal` \| `by_need` — the 4.2x retention fix |
| `evict_order` | `fifo` | `fifo` \| `lru` \| `lru_tail_first` |
| `disk_restore` | 0 | read back from disk on a miss |
| `disk_store_paused` | 0 | the write path's hot off switch |
| `disk_budget_gb` | 0 | per-node byte budget; 0 means unbounded |
| `gpu_cache_epoch` | 0 | bump to drop the GPU prefix cache, keeping the disk index |
| `pin_static_head` | 0 | reserve KV for a static prompt head — **built, never enabled** |

Defaults are chosen so **an empty flags file behaves exactly as upstream**:
every new mechanism off, every pre-existing one at its prior value. A value that
is recognised but refused falls back to the default and **says so** in the log —
shipping a half-built policy as though it were live is worse than refusing it.

**A mechanism gets a hot off switch before it gets a feature.** That rule cost a
user-visible outage: the disk store path had no way to be disabled without a
six-minute restart, so the only mitigation available was one the box could not
afford. A knob whose OFF state needs a boot is not a knob.

	bash local-ops/kv-flags.sh                       # read both nodes
	bash local-ops/kv-flags.sh kvmgr_enabled 1

## The disk tier, which this README used to say was impossible

**This section previously read "The disk tier is not buildable here, and the
knobs say so", and every word of its conclusion was wrong.** It argued that a
KV cache manager holds no KV bytes — true, `KVCacheBlock` is metadata and the
tensors live in the workers — and concluded that a disk tier was therefore
unbuildable, blaming vllm#45663. The layer argument was right and the
conclusion did not follow: the bytes are reachable from the **worker** side,
which is where `kv_offload` already operates, and the tier has since written
tens of thousands of files and read them back.

It is left here, named, rather than quietly deleted. A README that was
confidently wrong about its own central question is worth knowing about if you
are deciding how much to trust the rest of it.

### What is live and measured

| | |
|---|---|
| GPU → disk writes, both ranks | **live**, byte-verified round trip |
| disk → GPU restore | **live and correct** — same prompt, byte-identical output |
| restore at scale | **3.3 s against a 122.1 s cold pass** (166,438-token prefix, 100% restored) |
| a prefix surviving an engine restart | **live** — 458,679 keys restored, 0 pruned |
| bounded growth | **live** — per-node byte budget with an LRU, 105,339 keys evicted in one session |
| the path fingerprint | **live** — format version plus a content identity for the model |

### Why not upstream's tiering spec

`TieringOffloadingSpec` routes every tier through a **CPU primary tier**, and
`tiering/spec.py` builds that region in **shared memory**, sized
`page × world_size`, with the secondary tiers created scheduler-side in one
process. On a single host every rank maps the same `/dev/shm` and that works.

Our topology is `world_size=2, local_world_size=1` — **one rank per node** — so
rank 1 maps a *different* kernel's `/dev/shm` and nothing on that node has a
secondary tier to cascade to. Measured: every file upstream's tier wrote was
`page × world_size` bytes with the **second half entirely zero**, six out of six
sampled. The disk held half a cache, and restoring it would have loaded zeros
into half the KV silently.

Upstream will not "fix" this because it is their stated design assumption —
*"the framework consolidates all shards into a single shared host memory
region"* — and `kv_offload` carries **no guard at all**: `world_size` only
*sizes* the region and never asserts the workers are local, so a cross-node
tensor parallel is silently wrong rather than refused. See
`docs/UPSTREAM-FINDINGS.md`.

So this spec does the transfer **worker-side**, GPU → a small pinned staging
buffer → the local disk, each rank writing its own slice to its own node. No CPU
tier, which also avoids paying for RAM out of the KV pool on unified memory.

### The knobs are live now, not refused

The three `BLOCKED` reasons quoted in the old section are gone. `disk_enabled`
remains blocked and is reported as **governing nothing**, because what actually
turns the store path on is `KV_TRANSFER_CONFIG` naming this spec — reporting a
flag there printed a confident `false` about a tier that was demonstrably
writing. Report the fact, not the ask.

## What this does NOT guarantee

The failure mode to design against is not a crash. **KV bytes are only valid
for the exact engine and writer that produced them, and vLLM's block hash is a
Merkle chain over token ids and the previous hash — it does not include the
engine config.** So a restore across a changed configuration returns plausible
tensors that decode to nonsense and nothing crashes. That is strictly worse than
refusing to start. Everything below is about that.

**Covered:**

- **Config changes** land in a different directory. `FileMapper`'s fingerprint
  is a sha256 over the model name, hash block size, tp/pp/pcp/dcp sizes, cache
  dtype and every KV group's `block_size` and `layer_names`, so a changed
  config simply misses — refuse-by-construction rather than best-effort.
- **Our own on-disk format**, via `DISK_FORMAT_VERSION`, which is inside the
  hash and not merely in a directory name.
- **An accidental format change** — somebody editing the writer and forgetting
  to bump the version — is caught **offline**, by a pinned hash of the
  layout-determining functions with comments stripped. A runtime hash would
  catch it too and would orphan the whole tier the first time anyone fixed a
  typo.
- **The model's content**: `config.json` plus the `(filename → blob)` mapping
  of every safetensors entry. Note this is deliberately **not** the HF
  revision: on this box the served model has two snapshots sharing all 49
  blob targets and a byte-identical config, so a revision-keyed fingerprint
  would have orphaned 400 GB for a documentation-only bump.
- **All-ranks presence.** The framework completes a transfer only when every
  rank confirms, so a key in the index is backed on every node. An independent
  disk walk later found all 434,074 keys present on both ranks.

**Not covered, and you should know before trusting it:**

- **If the model directory cannot be read, the identity degrades to the model
  NAME.** That is the pre-fingerprint behaviour, so it is never a regression,
  but it means a re-quantised checkpoint at the same id would silently share
  the cache. It is logged and published as `model_identity: unknown` rather
  than hidden — if you see that, you have name-level protection only.
- **Our patch-series version is not in the fingerprint**, deliberately: it
  changes on every commit and would orphan the tier daily, while only the
  *layout* affects byte compatibility. The offline guard covers the layout; a
  change that alters bytes without touching those functions would not be
  caught.
- **A failed transfer is fatal, not a miss.** Upstream's
  `offloading/worker.py` is a bare `assert transfer_result.success` under the
  comment *"we currently do not support job failures"*, so a disk read or
  write error raises in the worker's output path rather than degrading. We make
  the reachable cases impossible by construction instead — notably by never
  evicting a key with a load in flight — but an `EIO` or an `ENOSPC` will take
  the engine down.
- **Dropping the GPU prefix cache while transfers are in flight corrupted a
  worker.** Freeing a block that a store is still reading produced `Xid 32` and
  killed rank 1 with an asynchronous `CUDA error: unspecified launch failure`.
  The drop refuses while any job is outstanding now; the general lesson is that
  "idle" by *request* count is not idle by *transfer*, because a store job
  outlives its request.
- **Two GPU faults (`Xid 13`) remain unexplained**, from earlier work and
  without a cache drop involved. Our indices are acquitted —
  `refused_bounds=0` across ~19,000 transfers — but acquitted is not explained.
- **Space is bounded per node, not cluster-wide**, because the two nodes have
  separate disks and a cluster number could not be enforced on either. And the
  budget bounds the **index**: files the index does not know about are
  invisible to it, which once left 155 GiB of unreachable files in the live
  directory. `local-ops/kvsweep.sh` is the sweep.
- **Eviction can strand a prefix.** Reuse is a contiguous run from token 0, so
  evicting a prefix's *head* writes off every block behind it — measured as
  `lookup=1, hit=0` with 5,030 blocks of that prefix present on disk. The GPU
  side solved this with `evict_order=lru_tail_first`; the disk tier has not
  learned it yet.

## Registration is belt-and-braces, on purpose

The entry point is the right mechanism. What could not be verified without the
cluster is whether `vllm.general_plugins` loads in the **engine core** process —
the one that builds managers — or only in the workers. If only in the workers,
the entry point registers where there is no scheduler and **Phase 1 silently
does nothing**.

So `patch6` also carries **MNT-11**: a four-line lazy hook in
`get_manager_for_kv_cache_spec` that imports `kvmgr` and registers on the first
manager built. That is in the module which *owns* `spec_manager_map`, so it is
in the right process by construction. Lazy rather than at import, because
`kvmgr` imports `_flag` and the manager classes back out of `patch6` — an
import-time hook is a circular import against a half-executed module.
`register()` is idempotent, so the two routes cannot conflict. A missing plugin
is a silent `ImportError` (stock behaviour); any other failure is loud.

**The box check that settles it**: `kvmgr: registered` must appear in the engine
log. If it appears twice, both routes fired and that is fine.

## *** `protected_fraction` MUST COME OUT OF `.env.dspark` ***

`_flag()` checks `VLLM_<NAME>` **first**, so any knob also set in the
environment cannot be moved from the file — you would edit it, see nothing
change, and get **no indication why**.
`VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION=0.75` is set in `.env.dspark` on **both
nodes today**, so until it is removed `protected_fraction` is hot in name only.

Two things make that visible rather than puzzling: `flags.audit_env()` reports
every env-pinned knob and the manager logs it **once at startup**, warning for
the ones that must not be pinned. Note `.env.dspark` is gitignored, so this is
an on-box edit on both nodes, never a push.

The legacy name is honoured deliberately — renaming it to `VLLM_PROTECTED_FRACTION`
would silently drop the 0.75 that is live.

## `_num_managers` is not trustworthy, so we do not use it

Both of patch6's cap functions divide by
`SingleTypeKVCacheManager._num_managers`, a **process-global class counter**
incremented per manager built and never decremented. Anything that constructs a
second set of managers halves every cap in the running engine, silently.

`register()` wraps `KVCacheCoordinator.__init__` — the one place holding the
real `KVCacheConfig` — and stamps the group count, the per-group block sizes and
the per-group sliding windows onto our class. The counter stays as a fallback
and logs when it is used.

## The windows are load-bearing, and the obvious reading of "by need" is a 21x regression

A windowed group **cannot hold** more than its window: it only ever matches the
contiguous run ending at the boundary, so it needs `cdiv(window-1, bs)+1` blocks
**however long the prompt gets**. patch6 already knows this
(`_head_pin_candidates`, MNT-9) and the parity test asserts our formula agrees
with its code.

Read instead as "give every group the same token extent", the budget splits in
inverse proportion to block size — and group 3's 4-token blocks need 64x as many
as group 0's for the same extent. Measured: **44,412 tokens per group, against
group 0's 932,608 today.** It would look like a principled fix and cost 95% of
the retention.

Window-aware, at fraction 0.75 on the live pool:

	24,288 blocks x 0.75        = 18,216 blocks to divide
	groups 1-4, windowed        =  3 + 3 + 3 + 17 = 26 blocks  (0.1% of the pool)
	group 0, full MLA           = 18,190 -> clamped by the LEGACY cap to 8,192
	                            = 2,097,152 tokens  (2.25x today, 2x the 1M ceiling)

**CONFIRMED ON THE LIVE ENGINE 2026-10-07**, flipped with no restart:

	kvmgr: protected cap for KV group 0 = 8192 blocks = 2097152 tokens
	       (pool 24288, groups 5, fraction 0.75, alloc by_need, legacy 8192)
	kvmgr: protected cap for KV group 0 = 3643 blocks =  932608 tokens
	       (pool 24288, groups 5, fraction 0.75, alloc equal, legacy 8192) [was 8192 blocks]

The engine's own geometry line reads `windows (None, 128, 128, 8, 128)` --
**group 3's window is 8, not 128**, which the offline fixture had wrong until
this deploy printed the truth. It does not move the headline, because group 0 is
clamped by the legacy cap either way; it changes the windowed groups' cost from
56 blocks to 26.

## The two policies (Phase 3)

Two independent knobs, so they can be A/B'd on one engine without a restart.

### `protect_alloc=by_need` — which group gets the budget

	equal    (today)  group 0 -> 3,643 blocks =   932,608 tokens
	by_need           group 0 -> 8,192 blocks = 2,097,152 tokens   (2.25x)

The four windowed groups take **56 blocks between them** — 0.2% of the pool —
and full attention takes the remainder. Group 0 then hits the **legacy** cap
(`2 x max_model_len / block_size`), which is why this is 2.25x and not the
4.2x the plan expected: past that point raising the fraction buys group 0
nothing at all.

### `evict_order` — which prefix, and which end

Two dimensions, and conflating them is why the measured loss was total:

| | victim prefix | end eaten | the half-evicted prefix |
|---|---|---|---|
| `fifo` (today) | the longest-**held** | head | reuses **nothing** |
| `lru` | the most **abandoned** | head | reuses **nothing** |
| `lru_tail_first` | the most **abandoned** | **tail** | still **hits** |

`lru` fixes the prefix and **not** the end, which is why tail-first is its own
arm rather than a refinement. Measured in the harness — three 20-block
prefixes, the oldest still active, cap cut to 30:

	fifo             A gone          B  10/20 head LOST   C 20/20
	lru              A 10/20 head LOST   B gone           C 20/20
	lru_tail_first   A 10/20 HEAD KEPT   B gone           C 20/20

Same residency, different **value**: because reuse is a contiguous match from
token 0, the 30 surviving blocks are worth 20 under `lru` and 30 under
`lru_tail_first`. On the real case that is the difference between a ~518 s
recompute of a 740K prefix and ~35 s for the 40K tail.

**CONFIRMED LIVE 2026-10-07.** All three rules fired on the real engine, each
logging its own victim rule, with `cause=cap` and the surviving extent in
tokens:

	victims={'fifo_oldest_protected': 492,
	         'lru_least_recently_reused': 105,
	         'lru_tail_first': 105}

Driven by squeezing `protected_fraction` to 0.004 (cap 18 blocks) against ~108
protected blocks across six distinct prefixes — which is only testable at all
because the fraction became hot the same day. Both eviction causes separated
correctly (`cap` on the trim, `reset` on the flag-change reconcile; `pool`
stayed 0 on an idle box) and `vllm:num_preemptions_total` stayed 0.

### The gentle squeeze, 2026-10-07: head-first stranding MEASURED LIVE

The `fifo` arm is the clean result, and it is the first live measurement of the
damage the whole policy exists to avoid. Four distinct 12K-token prefixes, 46
blocks each, then the cap squeezed from 3,413 to 114 blocks:

	before   lineages=4  surviving=184  REUSABLE=184  (100%)
	after    lineages=4  surviving=114  REUSABLE= 93  ( 82%)
	           blocks= 21  head_intact=FALSE  reusable=  0   <- STRANDED
	           blocks= 46  head_intact=True   reusable= 46
	           blocks= 46  head_intact=True   reusable= 46
	           blocks=  1  head_intact=True   reusable=  1

**21 of 114 surviving blocks — 18% — are worth nothing.** They are resident,
they cost pool, and the cache cannot use one token of them, because the head of
their prefix went first. That is the mechanism, on the real engine.

### ...AND THE OTHER TWO ARMS DID NOT GET A FAIR TEST

Recorded because the numbers look like a win and are not evidence of one. Both
`lru` and `lru_tail_first` came out at **100% reusable** — but with
**`lineages=1`**. The squeeze removed whole prefixes rather than half-evicting
one, so there was no partial victim to inspect and nothing to discriminate.

Two design faults, both mine:

- **State accumulated between arms.** Each arm set `protected_fraction=0.75`
  (cap 3,413), which trims nothing, so the previous arm's blocks survived:
  `protected` went 184 → 902 → 920 across the three arms. The squeeze target was
  computed as 5/8 of that inflated total, which is far more than one prefix.
- **The prefixes merged.** Distinct seeds but identical filler, so later prompts
  shared prefixes and collapsed into one 604-block lineage. A 46-block prefix
  and a 604-block one do not squeeze comparably.

**What a fair run needs:** reset the protected set between arms (squeeze to
near-zero, then back up), filler that genuinely differs per prefix so lineages
cannot merge, and a target chosen to land **mid-lineage** — total minus half of
one prefix — rather than as a fraction of the total.

So: **head-first stranding is confirmed live; tail-first beating `lru` is still
offline-only.** `lru` and `lru_tail_first` are *selected* correctly live (the
victim counters prove that) and their outcome difference is not yet measured on
the box.

**WHAT THIS DOES *NOT* SHOW, and it is the interesting half.** It proves the
rules fire, are selected live, and are attributed correctly. It does **not**
prove the differential OUTCOME — which prefix survived, and whether it kept its
head. The squeeze to 18 blocks destroys nearly everything, so there is no
interesting partial survivor to inspect; the per-decision lines read
`1 lineage(s) held` by the end because only one was left. **The head-vs-tail
difference is still verified offline only**, in `test_phase3.py`, where the
scenario is built to leave exactly one half-evicted prefix. Showing it live
needs a cap that bites *gently* — a few hundred blocks below the protected
total, not two orders of magnitude.

A prefix is identified by its **head block's hash**, the convention MNT-9
already uses for head pins — the one handle that is stable across turns and
distinct between two lineages that share no prefix. Tracking hangs off the fact
that `MLAAttentionManager.cache_blocks` passes a prefix *in order*, so
`blocks[0]` is the head; nothing else in the manager knows which end is which.
A shared head is attributed to the first lineage that protected it, and is then
refreshed whenever **either** conversation protects — so a shared head is harder
to evict, which is the right bias.

`fifo` remains literally patch6's own release: the lineage structures are kept
*beside* its set and deque rather than replacing them, so the two cannot drift.

## Observability (Phase 2)

Eviction used to be **inferred**, after the fact, from `engine_cached_tokens` on
the proxy side. It is observed now, per group:

| | |
|---|---|
| lookups | hit / miss / **partial**, in calls, blocks and tokens |
| protection | calls and blocks, counted as a **delta** of the set |
| eviction | by **cause** — `cap` \| `pool` \| `reset` — and by **victim rule** |
| state | the cap in blocks *and* tokens, and what is protected now |

**Three lookup outcomes, not two.** A partial hit is a prefix that survived in
*part* — exactly what tail-first eviction is meant to produce — so collapsing it
into "hit" would make Phase 3 unmeasurable.

**Eviction by cause is the point.** During the measured 373,833-token loss KV
usage was **22–33%**, the pool two-thirds *empty*, while the cap was exceeded by
43%. A `cap` eviction on an idle pool is a policy bug; a `pool` eviction is
capacity. One counter for both would hide the distinction the whole plan turns
on.

**Protection is counted as a delta, not as the offer.** patch6 skips a block
that is null, unhashed or already protected, and on an appending turn almost
every block is already protected — so counting `len(blocks)` overstates by an
order of magnitude.

### *** A STAT LOGGER CANNOT SEE THESE NUMBERS ***

The plan says to export "via the existing `vllm.stat_logger_plugins` entry point
(as `prefill_collector.py` already does — no fork)". It cannot work. Stat
loggers run on AsyncLLM's output-handler loop — **the API server process**,
which `prefill_collector`'s own header states — while the KV managers run in the
**engine core** process behind the IPC that carries `EngineCoreOutputs`. A stat
logger reading these counters would report **zeros forever**: a number the page
knows how to draw and the wire can never fill, which is the prefix gauge's
failure for the third time.

So the export is a loopback JSON + Prometheus endpoint served **from the process
that holds the counters**, started from the manager's constructor rather than
from `register()` — so it is in the right process by construction. Same shape as
`/prefill`, which sparkstats already proxies and turns into series.

	kvmgr_stats_port 9107     # 0, the default, BINDS NOTHING

	GET /kv       the snapshot, with the policy in force beside the counters
	GET /metrics  prometheus text, group/cause/rule labels
	GET /flags    the EFFECTIVE values -- which reading the file cannot answer,
	              because env wins over it, silently

Not on `:8888`: the proxy's Caddy catch-all forwards every path to the vLLM
pool, so a route there is reachable by every mTLS and LAN client.

`kvmgr_log_level 1` logs each eviction with its cause and victim rule. Off by
default — sixty evictions would be sixty lines on the scheduler loop.

### Did the flag take?

	bash local-ops/kvmgr-flag.sh                        # EFFECTIVE values, both nodes
	bash local-ops/kvmgr-flag.sh kvmgr_enabled 1        # writes, then WAITS for the log line

It merges rather than overwrites (the file carries the MNT-6b/8/9 flags too, and
clobbering `swa_recycle_cached` restores a bug already paid for), writes
atomically, **refuses** when the container env pins the knob, and fails if no
confirmation line arrives. `kv-flags.sh` answers "what did I write"; this
answers "what is the engine using", and those differ in two silent ways.

## Tests

	cd local-ops/kvharness
	for f in test_phase*.py; do python3.12 "$f" || echo "FAIL $f"; done

**1,257 assertions across ten files, no GPU, no cluster, a few seconds.** Use
`python3.12`: 3.9 cannot parse the overlays' evaluated `X | None` annotations.

They load the **real** upstream sources — patch6/7/8, the pristine coordinator,
and `kv_offload/{base,worker,file_mapper,factory}.py` — with `numpy` and `torch`
**stubbed rather than installed**, because the whole value of this harness is
running on a laptop with nothing installed. The stub header states exactly what
is and is not modelled, so no test can quietly assert on the stub instead of on
vLLM. See `local-ops/kvharness/README.md`.

**Count the assertions by EXIT CODE, not by the last line.** A `tail -1` sweep
reported a file as failing when it had passed — the final PASS message carried a
multi-line log dump, so the last line was log text rather than the summary. The
loop above tests the command.

**Every guard is verified by breaking it, and that is the only reason most of
this is trustworthy.** Roughly a dozen of the guards written for this work
passed *vacuously* on the first attempt and were found only by mutation —
including an anti-stacking check that could not fail (stacked wrappers reach the
inner function exactly once too), a "never clobber" check the kernel was
catching rather than the guard (POSIX `rename` refuses a non-empty target), a
lint that read its own comment as code, and an eviction assertion that was true
of an empty victim list. **A guard needs a partner that fails when nothing
happened at all.**

Earlier breaks, each caught and each naming its own cause: a knob captured at import, the kill switch not reaching
patch6, reconcile doing nothing, the cap not logging on change, `by_need`
reverting to the naive reading, an unbuilt policy value shipping silently, the
patch6 hook removed, protection counted as the offer, cap and pool evictions
conflated, a partial hit collapsed into a hit, an absent cap exported as a zero
gauge, and the stats port binding when set to 0.

**One of those breaks was not caught at first, and the fix is the lesson.** The
"port 0 binds nothing" check asserted `_server is None` — and with the guard
broken an earlier case had already taken the fallback port, so the re-bind
raised "address already in use", was swallowed, and left `_server` None. The
test went green about a server that had started. It records whether a socket was
**attempted** now. A test that passes for the wrong reason is as unhelpful as one
that fails for it.

## Still a box check

The pool **sizer**. `num_gpu_blocks` is an input to the harness (24,288 is
passed in), so the boot line — `GPU KV cache size` 3,604,786 tokens /
`num_gpu_blocks` 24,288 at `KV_CACHE_MEMORY_BYTES=23400318587` — must be
confirmed unchanged on the box. Any drift means the sizer disagrees with us.

Also owed at the box: the MNT-6a cap line prints on **first use**, not at boot,
so a deploy script that greps for it straight after the API answers finds
nothing — and under `set -e` that kills the run.
