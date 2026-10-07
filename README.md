# kvmgr — a KV cache manager of our own

Our manager for **KV group 0** (full MLA, 256-token blocks), with every policy
knob movable on a running engine. Phase 1 of the plan; see
`docs/local/KVMGR-PHASE0-20261007.md` for what the reading settled.

**Phase 1 changes no behaviour.** `kvmgr_enabled` defaults false, and when false
every override calls `super()` — patch6, unmodified. With it true and the
defaults (`protect_alloc=equal`, `evict_order=fifo`) the cap agrees with patch6
block for block, asserted at five fractions plus the negative-fraction contract.

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
	# then restart; a push is not a restart.

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

| flag | default | effect | built |
|---|---|---|---|
| `kvmgr_enabled` | 0 | our policy vs patch6's, per call | yes |
| `protected_fraction` | 0.30 | the cap, moved off `os.environ` | yes |
| `protect_alloc` | `equal` | `equal` \| `by_need` | Phase 3 |
| `evict_order` | `fifo` | `fifo` \| `lru` \| `lru_tail_first` | Phase 3 |
| `disk_enabled` / `disk_budget_gb` / `disk_restore` | off | the disk tier | Phase 4/5 |
| `kvmgr_log_level` | 0 | per-decision eviction logging | Phase 2 |

A value that is **recognised but not yet built** falls back to the default and
**says so** in the log. Shipping a half-built policy as though it were live is
worse than refusing it.

	bash local-ops/kv-flags.sh                       # read both nodes
	bash local-ops/kv-flags.sh kvmgr_enabled 1

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
	groups 1-4, windowed        =  3 + 3 + 33 + 17 = 56 blocks  (0.2% of the pool)
	group 0, full MLA           = 18,160 -> clamped by the LEGACY cap to 8,192
	                            = 2,097,152 tokens  (2.25x today, 2x the 1M ceiling)

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

	python3.12 local-ops/kvharness/test_phase0.py    #  59 — the plan's premises
	python3.12 local-ops/kvharness/test_phase1.py    # 118 — parity + hot config
	python3.12 local-ops/kvharness/test_phase2.py    #  67 — counters and exporters

No GPU, no cluster, ~2 s. Both load the **real** patch6/7/8 and the pristine
coordinator; see `local-ops/kvharness/README.md`.

Every guard is verified by breaking it — twelve breaks, each caught and each
naming its own cause: a knob captured at import, the kill switch not reaching
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
