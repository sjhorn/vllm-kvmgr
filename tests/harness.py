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

# kvmgr itself, from the package's src/ -- never an installed copy.
_SRC_FOR_IMPORT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
if _SRC_FOR_IMPORT not in sys.path:
    sys.path.insert(0, _SRC_FOR_IMPORT)

# *** EVERY PATH A SUITE OPENS COMES FROM HERE. *** The suites used to build
# their own from `REPO`, five different spellings of it, and one of them read
# sources through a helper that SKIPPED a missing file -- so the move into
# packages/ would have left a knob audit auditing an empty string, green.
PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # packages/vllm-kvmgr
SRC = os.path.join(PKG, "src")
KVMGR_DIR = os.path.join(SRC, "kvmgr")
# The vLLM sources the tests load, vendored from the image kvmgr is proven on.
UPSTREAM = os.path.join(PKG, "tests", "upstream")
PRISTINE = UPSTREAM  # the old name, kept because the suites use it

# The DSpark fork's patch overlays, needed only for --with-dspark-patches.
# Located by env, else by this package's place inside the DSpark repo.
DSPARK_REPO = os.environ.get("KVMGR_DSPARK_REPO") or os.path.dirname(
    os.path.dirname(PKG))
PATCHES = os.path.join(DSPARK_REPO, "patches")
REPO = DSPARK_REPO  # the old name, for the few checks that read the compose

# WHICH vLLM THE SUITES RUN AGAINST, and run-all.sh runs both:
#   pristine (default)      tests/upstream/ alone -- kvmgr on a vLLM with NO
#                           patch of ours, which is the package's claim
#   --with-dspark-patches   the DSpark fork's patch6/7/8 over the same sources,
#                           which is what the box runs
# Read from argv OR the environment, so `python3.12 test_x.py
# --with-dspark-patches` and run-all.sh's exported variable both work.
DSPARK = ("--with-dspark-patches" in sys.argv
          or os.environ.get("KVMGR_WITH_DSPARK_PATCHES") == "1")
if DSPARK and not os.path.isfile(os.path.join(
        PATCHES, "patch6-single_type_kv_cache_manager.py")):
    raise SystemExit(
        f"--with-dspark-patches: no patch6 under {PATCHES}. Set "
        "KVMGR_DSPARK_REPO to the DSpark recipe checkout.")
MODE = "dspark" if DSPARK else "pristine"


def needs_dspark(what: str) -> bool:
    """True in DSpark mode. In pristine mode, print that `what` was NOT run.

    A section skipped silently reads as a section that passed; this names it,
    so the absence is visible in the suite's own output."""
    if not DSPARK:
        print(f"SKIP  [pristine mode] {what} -- needs --with-dspark-patches")
        SKIPPED.append(what)
    return DSPARK


SKIPPED: list[str] = []


def kvmgr_source(name: str) -> str:
    """The text of one kvmgr module, by bare name ("diskio") or relative path.

    Raises rather than returning "" -- a lint fed an empty string passes."""
    rel = name if name.endswith(".py") else name + ".py"
    with open(os.path.join(KVMGR_DIR, rel)) as fh:
        return fh.read()

# The five live KV groups, in group order, as tokens per block.  Backed out of
# the engine's own "legacy cap" lines; `--block-size 256` is group 0's only.
LIVE_BLOCK_SIZES = (256, 64, 64, 4, 8)

# Group 0 is full MLA, so its need is unbounded (None). The other four are
# windowed, and a windowed group needs its window and no more however long the
# prompt gets.
#
# READ OFF THE LIVE ENGINE 2026-10-07, not inferred: our own geometry line
# prints `windows (None, 128, 128, 8, 128)`. Note GROUP 3's WINDOW IS 8, not
# 128 -- this fixture said 128 for all four until the first deploy printed the
# truth. It does not move the headline (group 0 is clamped by the legacy cap
# either way) but it changes the windowed groups' cost from 56 blocks to 26,
# and a fixture that disagrees with the box is how an offline suite goes quietly
# wrong.
LIVE_SLIDING_WINDOWS = (None, 128, 128, 8, 128)

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
    coord_mod: types.ModuleType


_loaded: Loaded | None = None
STOCK_MLA_MANAGER = None


def load_kv_offload() -> types.ModuleType:
    """Load the REAL `vllm.v1.kv_offload.*` from patches/pristine/ and return it.

    *** WHY PRISTINE AND NOT A COPY IN THIS DIRECTORY. *** Phase 1 implements an
    `OffloadingSpec` against this contract, and the single most expensive way to
    get that wrong is to write it against a contract I only skimmed, or against a
    stale copy that has since moved. These are the image's own files, pulled by
    the method `patches/pristine/MANIFEST.txt` prescribes, so the assertions in
    test_phase0_disk.py are against the code that will actually run.

    Safe to read from site-packages for THESE files specifically, which the
    MANIFEST otherwise forbids: `docker inspect` confirms no mount touches
    `kv_offload`, so there is no patched file to read by mistake. Verified
    2026-10-07 against the 20 mounts the container actually has.

    `load()` must run first: base.py imports `resolve_kv_cache_block_sizes` from
    `vllm.v1.core.kv_cache_utils`, which is PATCH 8 -- so the block-size
    resolution these tests see is our patched one, which is the point.
    """
    load()  # patch8 must be in sys.modules under its canonical name first
    # *** IDEMPOTENT. *** A second load re-executes base.py and mints NEW class
    # objects, so a spec built against the first copy is no longer an instance
    # of the second's LoadStoreSpec -- every isinstance check then fails for a
    # reason that has nothing to do with the code under test.
    if "vllm.v1.kv_offload.base" in sys.modules and getattr(
            sys.modules["vllm.v1.kv_offload.base"], "__file__", None):
        return sys.modules["vllm.v1.kv_offload.base"]

    kvo = os.path.join(PRISTINE, "vllm", "v1", "kv_offload")
    for name in ("vllm.v1.kv_offload", "vllm.v1.kv_offload.worker",
                 "vllm.v1.kv_offload.tiering"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, mod)

    _load_file(os.path.join(kvo, "base.py"), "vllm.v1.kv_offload.base")
    _load_file(os.path.join(kvo, "worker", "worker.py"),
               "vllm.v1.kv_offload.worker.worker")
    _load_file(os.path.join(kvo, "factory.py"), "vllm.v1.kv_offload.factory")
    # cpu/common.py holds CPULoadStoreSpec, which the disk handler uses as the
    # inner (staging) destination -- so the chunking tests need it.
    for name in ("vllm.v1.kv_offload.cpu",):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            sys.modules[name] = mod
            parent, _, leaf = name.rpartition(".")
            setattr(sys.modules[parent], leaf, mod)
    _load_file(os.path.join(kvo, "cpu", "common.py"),
               "vllm.v1.kv_offload.cpu.common")
    _load_file(os.path.join(kvo, "file_mapper.py"),
               "vllm.v1.kv_offload.file_mapper")
    return sys.modules["vllm.v1.kv_offload.base"]


def load(force: bool = False) -> Loaded:
    global _loaded
    if _loaded is not None and not force:
        return _loaded

    vllm_stubs.install()

    core = os.path.join(UPSTREAM, "vllm", "v1", "core")
    if DSPARK:
        sources = (os.path.join(PATCHES, "patch8-kv_cache_utils.py"),
                   os.path.join(PATCHES, "patch7-block_pool-dedupe.py"),
                   os.path.join(PATCHES, "patch6-single_type_kv_cache_manager.py"))
    else:
        sources = (os.path.join(core, "kv_cache_utils.py"),
                   os.path.join(core, "block_pool.py"),
                   os.path.join(core, "single_type_kv_cache_manager.py"))
    utils = _load_file(sources[0], "vllm.v1.core.kv_cache_utils")
    pool_mod = _load_file(sources[1], "vllm.v1.core.block_pool")
    mgr_mod = _load_file(sources[2], "vllm.v1.core.single_type_kv_cache_manager")
    # The coordinator is NOT one of our overlays -- it is the pristine capture,
    # loaded so the layer ABOVE the managers is under test too: the group
    # geometry hook wraps its __init__, and the fixed-point hit loop that makes
    # a group-0-only disk restore pointless lives here.
    coord_mod = _load_file(
        os.path.join(UPSTREAM, "vllm", "v1", "core", "kv_cache_coordinator.py"),
        "vllm.v1.core.kv_cache_coordinator",
    )

    # The thing that silently goes wrong: a stub winning over the real file.
    assert pool_mod.KVCacheBlock is utils.KVCacheBlock, (
        "patch7 bound a different KVCacheBlock than patch8 defines -- load order"
    )
    assert mgr_mod.BlockPool is pool_mod.BlockPool, (
        "patch6 bound a different BlockPool than patch7 defines -- load order"
    )
    if DSPARK:
        assert hasattr(mgr_mod, "_flag"), "patch6 has no _flag(): not the patched file"
    else:
        # *** THE CLAIM UNDER TEST: nothing of patch6 is reachable. ***
        for _sym in ("_flag", "_flags", "_protected_fraction", "_FLAGS_PATH",
                     "set_pending_block_ids_provider"):
            assert not hasattr(mgr_mod, _sym), (
                f"pristine mode loaded a manager module carrying patch6's {_sym}")
    assert hasattr(mgr_mod, "spec_manager_map"), "patch6 has no spec_manager_map"

    assert coord_mod.BlockPool is pool_mod.BlockPool, (
        "coordinator bound a different BlockPool -- load order"
    )
    assert (
        coord_mod.get_manager_for_kv_cache_spec is mgr_mod.get_manager_for_kv_cache_spec
    ), "coordinator bound a different manager factory -- load order"

    # The registry entry BEFORE any manager is built. In DSpark mode patch6's
    # MNT-11 hook imports kvmgr and registers it on the first manager built, so
    # by the time a suite looks, the map may already hold our class.
    global STOCK_MLA_MANAGER
    STOCK_MLA_MANAGER = mgr_mod.spec_manager_map[
        sys.modules["vllm.v1.kv_cache_interface"].MLAAttentionSpec]
    _loaded = Loaded(utils, pool_mod, mgr_mod, coord_mod)
    if DSPARK:
        # What register() does in production: bridge patch6 to kvmgr's flags.
        # kv_offload first, because the bridge wires patch6's pending-blocks
        # hook to kvmgr.diskspec, which imports it.
        load_kv_offload()
        from kvmgr.compat import dspark as _bridge

        assert _bridge.install(), "DSPARK mode but the bridge found no patch6"
    return _loaded


def invalidate_flags() -> None:
    """Make the next knob read re-open the flags file, in every reader."""
    from kvmgr import flags as _kf

    _kf.invalidate_file_cache()
    mod = sys.modules.get("vllm.v1.core.single_type_kv_cache_manager")
    if mod is not None and hasattr(mod, "_flags_checked_at"):
        mod._flags_checked_at = -1.0


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
    """Point the flags-file reader at a temp file and drive it.

    The reader is kvmgr's (``kvmgr.flags.file_flags``); in DSpark mode patch6's
    ``_flags`` IS that reader via the bridge, and patch6's own path is pointed
    at the file too so an unbridged read agrees.

    The reader caches for 1.0s against ``time.monotonic()``, so a test that
    writes the file and reads immediately gets the OLD value.  This
    invalidates the cache explicitly rather than sleeping -- a one-second sleep
    per assertion is how a suite stops being run.
    """

    def __init__(self, path: str):
        self.path = path
        H = load()
        self.mod = H.mgr_mod
        from kvmgr import flags as _kf

        _kf.FLAGS_PATH = path
        if hasattr(self.mod, "_FLAGS_PATH"):
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
        invalidate_flags()

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
