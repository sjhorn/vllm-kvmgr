# kvharness — manager logic under test, with no GPU and no cluster

	python3.12 local-ops/kvharness/test_phase0.py     # 59 assertions, ~1s

Phase 0 of the own-KV-manager work. Findings and the reasoning are in
`docs/local/KVMGR-PHASE0-20261007.md`; this is how to run it.

## What it is

`vllm_stubs.py` installs fake `vllm.*` leaf modules into `sys.modules`;
`harness.py` then loads the **real** `patches/patch8` → `patch7` → `patch6`
under their canonical module names. The code under test is the code that runs on
the box — the cap arithmetic, the FIFO release, `_flag()`'s TTL. Only the leaves
are faked, and none of them decides a block's fate.

## Rules

- **`python3.12`.** The patches use `X | None` in evaluated annotations with no
  `from __future__ import annotations`; the system 3.9 fails at import. 3.12 is
  what the container runs.
- **Never reorder the loads.** patch8 must be registered as
  `vllm.v1.core.kv_cache_utils` before patch7 imports it, patch7 as
  `vllm.v1.core.block_pool` before patch6. Out of order you get the stub,
  silently. `load()` asserts object identity afterwards; keep that.
- **Reset `_num_managers` between cases.** It is a class counter, incremented per
  manager built and never decremented, and both cap functions divide by it — so
  without `harness.reset_manager_count()` the tests halve each other's caps.
- **Use `harness.Flags`, not `sleep`.** It invalidates patch6's 1 s flag cache
  directly. One sleep per assertion is how a suite stops being run.

## What it cannot see

Tensors, CUDA, and the pool **sizer**. `num_gpu_blocks` is an input here, so the
boot line (`GPU KV cache size`, `num_gpu_blocks`) is still a box check.

## Relation to kv-mech-tests.py

`local-ops/kv-mech-tests.py` stages the same patched files *into the container*
and runs them there. Stronger when the cluster is up; unavailable when it is
down, which is what this exists for. Complements, not replacements.
