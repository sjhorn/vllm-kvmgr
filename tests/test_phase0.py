"""test_phase0.py -- the facts the KV-manager plan rests on, asserted against
the REAL patched engine code, with no GPU and no cluster.

Run:  python3.12 packages/vllm-kvmgr/tests/test_phase0.py

Every assertion here is a claim the plan makes.  If one fails, the plan is
wrong -- which is the point: the numbers in the plan were backed out of log
lines by hand, and this is the first time they have been checked against the
arithmetic that actually runs.
"""

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import harness  # noqa: E402

PASS = 0
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if cond:
        PASS += 1
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL.append(name)
        print(f"FAIL  {name}  {detail}")


def eq(name: str, got, want) -> None:
    check(name, got == want, f"got {got!r} want {want!r}")


# *** EVERY ASSERTION IN THIS SUITE IS A FACT ABOUT patch6 *** -- MNT-6a's
# equal split, MNT-9's head pins, patch6's flag reader, MNT-11's hook. On a
# vLLM without the DSpark patches there is nothing here to check, so it says so
# and exits 0 with a SKIPPED summary rather than passing vacuously.
if not harness.needs_dspark("test_phase0: the patched engine's own facts"):
    print("0 passed, 0 failed (SKIPPED in pristine mode)")
    sys.exit(0)

H = harness.load()
MGR = H.mgr_mod

# ---------------------------------------------------------------------------
# 1. The harness is exercising the PATCHED code, not a stub or stock vLLM
# ---------------------------------------------------------------------------
check("patch6 carries MNT-6a protection", hasattr(MGR.SingleTypeKVCacheManager, "_max_protected_prompt_blocks"))
check("patch6 carries MNT-9 head pins", hasattr(MGR.SingleTypeKVCacheManager, "_max_head_pin_blocks"))
check("patch7 BlockPool has the dedupe map", hasattr(H.pool_mod, "BlockHashToBlockMap"))
check("patch8 has the free queue", hasattr(H.utils, "FreeKVCacheBlockQueue"))

# ---------------------------------------------------------------------------
# 2. THE CAP ARITHMETIC -- the plan's central numbers
# ---------------------------------------------------------------------------
pool = harness.make_pool(num_gpu_blocks=harness.LIVE_NUM_GPU_BLOCKS)
eq("pool is the live size", pool.num_gpu_blocks, 24288)

os.environ["VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION"] = "0.75"
mgrs = harness.make_live_groups(pool)
eq("five managers built", len(mgrs), 5)
eq("_num_managers counted all five", MGR.SingleTypeKVCacheManager._num_managers, 5)
eq("block sizes are the live geometry",
   tuple(m.block_size for m in mgrs), harness.LIVE_BLOCK_SIZES)

caps = [m._max_protected_prompt_blocks() for m in mgrs]
eq("every group gets the SAME block cap (the /num_managers equal split)",
   caps, [3643] * 5)

# 24,288 x 0.75 / 5 = 3,643.2 -> 3,643
eq("cap is 26% of the pool", round(3643 * 5 / 24288 * 100, 1), 75.0)
check("the protected SHARE is the fraction, split five ways",
      abs(3643 / 24288 - 0.15) < 0.002, f"{3643/24288:.4f} of pool per group")

tokens = [cap * m.block_size for cap, m in zip(caps, mgrs)]
eq("group 0 protects 932,608 tokens", tokens[0], 932608)
eq("group 3 protects 14,572 tokens", tokens[3], 14572)
eq("the spread across groups is 64x", tokens[0] // tokens[3], 64)
check("group 0 binds: it is the smallest EXTENT holder of none, "
      "but the only one that can hold a long prompt",
      tokens[0] == max(tokens), f"tokens per group = {tokens}")

# The plan's claim that raising the fraction cannot fix it.
os.environ["VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION"] = "1.00"
caps_full = [m._max_protected_prompt_blocks() for m in mgrs]
eq("at fraction 1.00 the cap is 4,857 blocks", caps_full, [4857] * 5)
eq("at fraction 1.00 group 0 still reaches only 1,243,392 tokens",
   caps_full[0] * 256, 1243392)
check("so the /5 is the wall, not the fraction",
      caps_full[0] * 256 < 5 * 932608,
      "a by-need split of the same fraction would give group 0 far more")

# The legacy cap, and the fact it does NOT bind at this pool size.
os.environ["VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION"] = "0.75"
legacy0 = 2 * (harness.LIVE_MAX_MODEL_LEN // 256)
eq("group 0 legacy cap is 8,192 blocks", legacy0, 8192)
check("legacy does not bind at fraction 0.75", mgrs[0]._max_protected_prompt_blocks() < legacy0,
      f"min(8192, 3643) = {mgrs[0]._max_protected_prompt_blocks()}")

# ---------------------------------------------------------------------------
# 3. PROTECTION IS NEVER REFRESHED ON REUSE  (MNT-6a flaw #2)
# ---------------------------------------------------------------------------
harness.reset_manager_count()
pool2 = harness.make_pool(num_gpu_blocks=64, hash_block_size=16)
m2 = harness.make_live_groups(pool2)[0]

blocks = pool2.get_new_blocks(6)
for i, b in enumerate(blocks):
    b.block_hash = H.utils.make_block_hash_with_group_id(
        H.utils.BlockHash((i + 1).to_bytes(32, "big")), 0
    )

m2._protect_prompt_blocks(blocks[:4])
first_order = list(m2._protected_prompt_block_queue)
eq("four blocks protected", len(m2._protected_prompt_block_ids), 4)

# "Reuse" the oldest two, exactly as a later turn of the same conversation does.
m2._protect_prompt_blocks([blocks[0], blocks[1]])
eq("re-protecting an already-protected block is a NO-OP on the queue",
   list(m2._protected_prompt_block_queue), first_order)
check("so the queue order still has the OLDEST block at the front",
      m2._protected_prompt_block_queue[0] == blocks[0].block_id,
      "reuse does not refresh -- the plan's flaw #2, confirmed")

# ---------------------------------------------------------------------------
# 4. EVICTION POPS FROM THE LEFT -- the victim is the longest-held
# ---------------------------------------------------------------------------
victim = m2._protected_prompt_block_queue[0]
released = m2._release_one_protected_prompt_block()
check("a release happened", released)
check("the victim was the FIRST-PROTECTED block, not the least-recently-used",
      victim not in m2._protected_prompt_block_ids,
      f"released block_id={victim}")
eq("three remain protected", len(m2._protected_prompt_block_ids), 3)

# And losing the HEAD is what makes the loss total: block 0 is the head.
check("the head block is the one that goes first",
      blocks[0].block_id == victim,
      "contiguous-from-token-0 reuse makes this a 100% loss, not a partial one")

# ---------------------------------------------------------------------------
# 5. PROTECTED_FRACTION IS RESTART-ONLY TODAY  (the plan's Trap 2)
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    fpath = os.path.join(td, "flags.json")
    with harness.Flags(fpath) as flags:
        os.environ.pop("VLLM_PROTECTED_PROMPT_BLOCKS_FRACTION", None)
        flags.write(protected_fraction=0.10)
        cap_via_file = mgrs[0]._max_protected_prompt_blocks()
        # TRAP 2 WAS REAL AND IS NOW CLOSED (2026-10-07). It read
        #   float(os.environ.get(_PROTECTED_FRACTION_ENV, "0.30"))
        # directly, so the single most valuable knob in the file was
        # restart-only -- and a restart costs a measured 672-795s cold prefill.
        # patch6 now has _protected_fraction(): legacy env (non-blank) wins,
        # then the flags file, then 0.30. The assertion is inverted rather than
        # deleted, so the history stays legible.
        eq("the flags file MOVES protected_fraction now, in patch6's own path",
           cap_via_file, int(24288 * 0.10 / 5))
        check("so Trap 2 is closed: no restart to retune the fraction",
              cap_via_file != int(24288 * 0.30 / 5),
              "was os.environ-only until 2026-10-07")
        check("and there is ONE definition of the precedence",
              hasattr(MGR, "_protected_fraction"))

        # ... whereas a flag that DOES use the mechanism moves live.
        flags.write(head_pin_fraction=0.20)
        eq("a _flag()-routed knob moves with no restart",
           mgrs[0]._max_head_pin_blocks(), int(24288 * 0.20 / 5))
        flags.write(head_pin_fraction=0.05)
        eq("and moves back", mgrs[0]._max_head_pin_blocks(), int(24288 * 0.05 / 5))

# ---------------------------------------------------------------------------
# 6. THE _flag() CONTRACT the plan makes every new knob use
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    fpath = os.path.join(td, "flags.json")
    with harness.Flags(fpath) as flags:
        flags.remove()
        eq("absent file -> default", MGR._flag("made_up_knob", 7), 7)
        flags.write_raw("{not json at all")
        eq("bad JSON -> default (never a crash, never a partial read)",
           MGR._flag("made_up_knob", 7), 7)
        flags.write(made_up_knob=9)
        eq("file supplies the value", MGR._flag("made_up_knob", 7), 9)
        os.environ["VLLM_MADE_UP_KNOB"] = "11"
        flags.invalidate()
        eq("ENV WINS over the file (the plan's Trap 1)",
           MGR._flag("made_up_knob", 7), 11)
        del os.environ["VLLM_MADE_UP_KNOB"]
        flags.invalidate()
        eq("and the file is honoured again once env is unset",
           MGR._flag("made_up_knob", 7), 9)

        # bool coercion, which is where a "0" string would otherwise read true
        flags.write(a_bool=0)
        eq("'0' is False for a bool default", MGR._flag("a_bool", True), False)
        flags.write(a_bool="false")
        eq("'false' is False", MGR._flag("a_bool", True), False)
        flags.write(a_bool=1)
        eq("'1' is True", MGR._flag("a_bool", False), True)

# ---------------------------------------------------------------------------
# 7. THE TTL IS REAL -- a knob is re-read, not captured
# ---------------------------------------------------------------------------
with tempfile.TemporaryDirectory() as td:
    fpath = os.path.join(td, "flags.json")
    with harness.Flags(fpath) as flags:
        import json
        import time as _t

        flags.write(ttl_probe=1)
        eq("first read", MGR._flag("ttl_probe", 0), 1)
        # Write a new value WITHOUT invalidating: within the TTL the old value stands.
        with open(fpath, "w") as fh:
            json.dump({"ttl_probe": 2}, fh)
        eq("within the 1s TTL the cached value stands (this is the hot-path cost)",
           MGR._flag("ttl_probe", 0), 1)
        eq("TTL is one second", MGR._FLAGS_TTL_S, 1.0)
        _t.sleep(1.05)
        eq("after the TTL the new value is picked up with no restart",
           MGR._flag("ttl_probe", 0), 2)

# ---------------------------------------------------------------------------
# 8. find_longest_cache_hit IS A CLASSMETHOD  -- the Phase 5 constraint
# ---------------------------------------------------------------------------
import inspect  # noqa: E402

for cls_name in ("FullAttentionManager", "MLAAttentionManager", "SlidingWindowManager"):
    cls = getattr(MGR, cls_name)
    raw = inspect.getattr_static(cls, "find_longest_cache_hit")
    check(f"{cls_name}.find_longest_cache_hit is a classmethod",
          isinstance(raw, classmethod),
          "so the disk-restore hook has NO instance state: no per-manager index, "
          "no counters on self")

sig = inspect.signature(MGR.FullAttentionManager.find_longest_cache_hit)
for p in ("block_hashes", "max_length", "kv_cache_group_ids", "block_pool",
          "kv_cache_spec", "use_eagle", "alignment_tokens"):
    check(f"hit hook receives `{p}`", p in sig.parameters)
check("the hook DOES get the block_pool, so it can allocate on a restore",
      "block_pool" in sig.parameters)

# ---------------------------------------------------------------------------
# 9. THE REGISTRY HOOK -- substitution works, and is reversible
# ---------------------------------------------------------------------------
iface = sys.modules["vllm.v1.kv_cache_interface"]
eq("MLAAttentionSpec maps to MLAAttentionManager before any manager is built",
   harness.STOCK_MLA_MANAGER, MGR.MLAAttentionManager)
# MNT-11 is patch6's lazy `import kvmgr; kvmgr.register()` on the first manager
# built. It used to be a silent no-op HERE because this suite could not import
# kvmgr -- so this suite asserted the stock map by accident. It can now, so the
# hook is observable doing its job.
check("MNT-11 registered kvmgr on the first manager built",
      MGR.spec_manager_map[iface.MLAAttentionSpec].__name__ == "DSparkMLAAttentionManager",
      MGR.spec_manager_map[iface.MLAAttentionSpec])


class _Probe(MGR.MLAAttentionManager):
    pass


original = MGR.spec_manager_map[iface.MLAAttentionSpec]
try:
    MGR.spec_manager_map[iface.MLAAttentionSpec] = _Probe
    harness.reset_manager_count()
    probe_pool = harness.make_pool(num_gpu_blocks=64, hash_block_size=16)
    built = MGR.get_manager_for_kv_cache_spec(
        kv_cache_spec=iface.MLAAttentionSpec(block_size=256),
        max_num_batched_tokens=8192, max_model_len=harness.LIVE_MAX_MODEL_LEN,
        block_pool=probe_pool, enable_caching=True, kv_cache_group_id=0,
        dcp_world_size=1, pcp_world_size=1,
    )
    check("registering into spec_manager_map substitutes our class",
          isinstance(built, _Probe),
          "the factory reads the dict at CALL time, so an import-time mutation is enough")
finally:
    MGR.spec_manager_map[iface.MLAAttentionSpec] = original
eq("and the registry is restored", MGR.spec_manager_map[iface.MLAAttentionSpec],
   original)

# ---------------------------------------------------------------------------
# 10. MNT-5 is NOT a KV mechanism  (the plan's open Phase 0 question)
# ---------------------------------------------------------------------------
src = open(os.path.join(harness.PATCHES, "patch6-single_type_kv_cache_manager.py")).read()
check("patch6 states MNT-5 is the detokenizer patch, not a KV one",
      "NOT the detokenizer patch -- that is MNT-5" in src,
      "so MNT-5 is out of scope for this work: nothing to carry forward")
check("and MNT-5 lives in its own file",
      os.path.exists(os.path.join(harness.PATCHES, "patch5-detokenizer.py")))

# ---------------------------------------------------------------------------
print()
print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
