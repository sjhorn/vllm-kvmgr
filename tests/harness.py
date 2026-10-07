"""harness.py -- load the REAL patch8/patch7/patch6 with no GPU and no cluster.

    import harness
    H = harness.load()            # H.utils, H.pool_mod, H.mgr_mod
    pool = harness.make_pool(num_gpu_blocks=24288)
    mgrs = harness.make_live_groups(pool)   # the five real groups, 256/64/64/4/8

The load order matters and is the whole trick: patch8 must be in `sys.modules`
under `vllm.v1.core.kv_cache_utils` BEFORE patch7 imports that name, and patch7
under `vllm.v1.core.block_pool` before patch6 imports it.  Load them out of
order and you get the stub, silently, and every assertion below is about nothing.
`load()` asserts the identity afterwards rather than trusting the order.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from dataclasses import dataclass

import vllm_stubs

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PATCHES = os.path.join(REPO, "patches")

# The five live KV groups, in group order, as tokens per block.  Backed out of
# the engine's own "legacy cap" lines; `--block-size 256` is group 0's only.
LIVE_BLOCK_SIZES = (256, 64, 64, 4, 8)

# The live pool, for cap arithmetic that must match the box.
LIVE_NUM_GPU_BLOCKS = 24288
LIVE_MAX_MODEL_LEN = 1_048_576


def _load_file(path: str, module_name: str) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec so a self-referential import resolves to this object.
    sys.modules[module_name] = mod
    parent, _, leaf = module_name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], leaf, mod)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Loaded:
    utils: types.ModuleType
    pool_mod: types.ModuleType
    mgr_mod: types.ModuleType


_loaded: Loaded | None = None


def load(force: bool = False) -> Loaded:
    global _loaded
    if _loaded is not None and not force:
        return _loaded

    vllm_stubs.install()

    utils = _load_file(
        os.path.join(PATCHES, "patch8-kv_cache_utils.py"),
        "vllm.v1.core.kv_cache_utils",
    )
    pool_mod = _load_file(
        os.path.join(PATCHES, "patch7-block_pool-dedupe.py"),
        "vllm.v1.core.block_pool",
    )
    mgr_mod = _load_file(
        os.path.join(PATCHES, "patch6-single_type_kv_cache_manager.py"),
        "vllm.v1.core.single_type_kv_cache_manager",
    )

    # The thing that silently goes wrong: a stub winning over the real file.
    assert pool_mod.KVCacheBlock is utils.KVCacheBlock, (
        "patch7 bound a different KVCacheBlock than patch8 defines -- load order"
    )
    assert mgr_mod.BlockPool is pool_mod.BlockPool, (
        "patch6 bound a different BlockPool than patch7 defines -- load order"
    )
    assert hasattr(mgr_mod, "_flag"), "patch6 has no _flag(): not the patched file"
    assert hasattr(mgr_mod, "spec_manager_map"), "patch6 has no spec_manager_map"

    _loaded = Loaded(utils, pool_mod, mgr_mod)
    return _loaded


def make_pool(num_gpu_blocks: int = LIVE_NUM_GPU_BLOCKS, enable_caching: bool = True,
              hash_block_size: int = 256):
    H = load()
    return H.pool_mod.BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=enable_caching,
        hash_block_size=hash_block_size,
    )


def live_specs() -> list:
    """The five groups' specs, with the real block sizes.

    Group 0 is full MLA (the one whose cap binds); the other four are the
    windowed/compressor groups.  Only `block_size` matters to the cap
    arithmetic, which is what this harness is for.
    """
    H = load()
    iface = sys.modules["vllm.v1.kv_cache_interface"]
    specs = [iface.MLAAttentionSpec(block_size=LIVE_BLOCK_SIZES[0])]
    for bs in LIVE_BLOCK_SIZES[1:]:
        specs.append(iface.SlidingWindowMLASpec(block_size=bs, sliding_window=128))
    assert len(specs) == 5
    del H
    return specs


def reset_manager_count() -> None:
    """`_num_managers` is a CLASS counter incremented in __init__ and never
    decremented, so a second set of managers built in one process halves every
    cap.  Tests must reset it or they measure each other."""
    H = load()
    H.mgr_mod.SingleTypeKVCacheManager._num_managers = 0


def make_live_groups(pool=None, max_model_len: int = LIVE_MAX_MODEL_LEN,
                     reset: bool = True) -> list:
    """Build the five managers through the engine's own factory."""
    H = load()
    if reset:
        reset_manager_count()
    if pool is None:
        pool = make_pool()
    mgrs = []
    for i, spec in enumerate(live_specs()):
        mgrs.append(
            H.mgr_mod.get_manager_for_kv_cache_spec(
                kv_cache_spec=spec,
                max_num_batched_tokens=8192,
                max_model_len=max_model_len,
                block_pool=pool,
                enable_caching=True,
                kv_cache_group_id=i,
                dcp_world_size=1,
                pcp_world_size=1,
            )
        )
    return mgrs


# ---------------------------------------------------------------------------
# flags-file control, so a test can flip a knob the way an operator does
# ---------------------------------------------------------------------------
class Flags:
    """Point patch6's `_flag()` at a temp file and drive it.

    `_flag()` caches for `_FLAGS_TTL_S` (1.0s) against `time.monotonic()`, so a
    test that writes the file and reads immediately gets the OLD value.  This
    invalidates the cache explicitly rather than sleeping -- a one-second sleep
    per assertion is how a suite stops being run.
    """

    def __init__(self, path: str):
        self.path = path
        H = load()
        self.mod = H.mgr_mod
        self.mod._FLAGS_PATH = path

    def write(self, **kw) -> None:
        import json

        with open(self.path, "w") as fh:
            json.dump(kw, fh)
        self.invalidate()

    def remove(self) -> None:
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass
        self.invalidate()

    def write_raw(self, text: str) -> None:
        with open(self.path, "w") as fh:
            fh.write(text)
        self.invalidate()

    def invalidate(self) -> None:
        self.mod._flags_checked_at = -1.0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


def log() -> list:
    return vllm_stubs.LOG


def clear_log() -> None:
    vllm_stubs.LOG.clear()


def log_text() -> str:
    return "\n".join(r.text() for r in vllm_stubs.LOG)
