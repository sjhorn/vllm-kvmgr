"""vllm_stubs.py -- enough of `vllm` to import patch6/7/8 ON A LAPTOP.

WHY THIS EXISTS, and why it is not the same thing as local-ops/kv-mech-tests.py.

kv-mech-tests.py stages the patched files INTO THE CONTAINER and runs them there,
so it needs a reachable node and a running image.  That is the stronger test when
the cluster is up and it is worthless when the cluster is down -- which is the
state this harness was written in (2026-10-07: LAN down, SSH to both Sparks
timing out, the menubar bridge the only route to anything).

So this installs fake `vllm.*` leaf modules into `sys.modules` and then loads the
REAL patches/patch8, patch7 and patch6 under their canonical module names.  The
code under test is the actual patched code -- the cap arithmetic, the FIFO
release, the `_flag()` TTL -- not a reimplementation of it.  A reimplementation
would be a second copy of the thing being tested, which is how two tables
describing one wire end up disagreeing (this repo's own recurring failure).

WHAT IS FAKED is only what the three files touch and none of it is policy:
dtype sizes, a logger, the kv-event dataclasses, the metrics collector, and
`Request`.  Everything that decides a block's fate is real.

WHAT THIS CANNOT TELL YOU: anything about tensors, CUDA, or the pool SIZER.
`num_gpu_blocks` is an input here, so boot-line parity is still a box check.
"""

import sys
import types
from dataclasses import dataclass, field
from typing import Any


def _mod(name: str) -> types.ModuleType:
    """Create (or fetch) a module and register it, parents included."""
    if name in sys.modules:
        return sys.modules[name]
    m = types.ModuleType(name)
    sys.modules[name] = m
    if "." in name:
        parent_name, _, leaf = name.rpartition(".")
        parent = _mod(parent_name)
        setattr(parent, leaf, m)
    return m


# --------------------------------------------------------------------------
# vllm.logger -- a real logger, because the patches log decisions we assert on
# --------------------------------------------------------------------------
class _Rec:
    """A captured log call, kept structured so tests assert on args not prose."""

    def __init__(self, level: str, msg: str, args: tuple):
        self.level, self.msg, self.args = level, msg, args

    def text(self) -> str:
        try:
            return self.msg % self.args if self.args else self.msg
        except Exception:
            return self.msg + " " + repr(self.args)

    def __repr__(self) -> str:
        return f"<{self.level} {self.text()}>"


LOG: list[_Rec] = []


class _Logger:
    def __init__(self, name: str):
        self.name = name

    def _emit(self, level, msg, *args, **kw):
        LOG.append(_Rec(level, str(msg), args))

    def debug(self, m, *a, **k):
        self._emit("debug", m, *a, **k)

    def info(self, m, *a, **k):
        self._emit("info", m, *a, **k)

    def warning(self, m, *a, **k):
        self._emit("warning", m, *a, **k)

    warn = warning

    def error(self, m, *a, **k):
        self._emit("error", m, *a, **k)

    def exception(self, m, *a, **k):
        self._emit("error", m, *a, **k)


def install() -> None:
    """Install every stub. Idempotent."""
    lg = _mod("vllm.logger")
    lg.init_logger = _Logger

    # ---- vllm.envs / vllm.config -------------------------------------
    envs = _mod("vllm.envs")
    envs.VLLM_V1_KV_CACHE_HASH_ALGO = "sha256"

    # patch8 reads attributes off envs; be permissive rather than guess the set.
    def _envs_getattr(name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return None

    envs.__getattr__ = _envs_getattr  # type: ignore[attr-defined]

    cfg = _mod("vllm.config")

    class VllmConfig:  # only ever a type annotation in these files
        pass

    cfg.VllmConfig = VllmConfig

    # ---- vllm.utils.* -------------------------------------------------
    hashing = _mod("vllm.utils.hashing")

    def sha256_cbor(x):
        import hashlib
        import pickle

        return int.from_bytes(
            hashlib.sha256(pickle.dumps(x, protocol=4)).digest()[:8], "big"
        )

    hashing.sha256_cbor = sha256_cbor
    hashing.xxhash_cbor = sha256_cbor

    math_utils = _mod("vllm.utils.math_utils")

    def cdiv(a, b):
        return -(-a // b)

    def round_up(a, b):
        return cdiv(a, b) * b

    math_utils.cdiv = cdiv
    math_utils.round_up = round_up

    mem_utils = _mod("vllm.utils.mem_utils")
    mem_utils.format_gib = lambda n: f"{n / (1 << 30):.2f}GiB"

    torch_utils = _mod("vllm.utils.torch_utils")
    # Only ever used to size a KV entry; the real one asks torch.
    torch_utils.get_dtype_size = lambda dtype: 1

    # ---- vllm.distributed.kv_events ----------------------------------
    ev = _mod("vllm.distributed.kv_events")
    ev.MEDIUM_GPU = "gpu"

    class KVCacheEvent:
        pass

    @dataclass
    class BlockStored(KVCacheEvent):
        block_hashes: list = field(default_factory=list)
        parent_block_hash: Any = None
        token_ids: list = field(default_factory=list)
        block_size: int = 0
        lora_id: Any = None
        medium: Any = None

    @dataclass
    class BlockRemoved(KVCacheEvent):
        block_hashes: list = field(default_factory=list)
        medium: Any = None

    @dataclass
    class AllBlocksCleared(KVCacheEvent):
        medium: Any = None

    ev.KVCacheEvent = KVCacheEvent
    ev.BlockStored = BlockStored
    ev.BlockRemoved = BlockRemoved
    ev.AllBlocksCleared = AllBlocksCleared

    # ---- vllm.v1.core.kv_cache_metrics -------------------------------
    met = _mod("vllm.v1.core.kv_cache_metrics")

    class KVCacheMetricsCollector:
        """Records the same calls the real one does, so tests can read them."""

        def __init__(self):
            self.calls: list[tuple] = []

        def __getattr__(self, name):
            def _rec(*a, **k):
                self.calls.append((name, a, k))

            return _rec

    met.KVCacheMetricsCollector = KVCacheMetricsCollector

    # ---- vllm.v1.utils -----------------------------------------------
    v1u = _mod("vllm.v1.utils")
    v1u.tensor_data = lambda t: b""

    # ---- vllm.v1.request ---------------------------------------------
    req = _mod("vllm.v1.request")

    class Request:
        """The attribute surface patch6/7/8 actually read off a request."""

        def __init__(
            self,
            request_id: str,
            prompt_token_ids: list[int] | None = None,
            num_prompt_tokens: int | None = None,
            block_hashes: Any = None,
            num_tokens: int | None = None,
            **extra,
        ):
            self.request_id = request_id
            self.prompt_token_ids = list(prompt_token_ids or [])
            self.num_prompt_tokens = (
                num_prompt_tokens
                if num_prompt_tokens is not None
                else len(self.prompt_token_ids)
            )
            self.block_hashes = block_hashes if block_hashes is not None else []
            self._num_tokens = num_tokens
            self.mm_features = []
            self.lora_request = None
            self.sampling_params = None
            self.all_token_ids = self.prompt_token_ids
            for k, v in extra.items():
                setattr(self, k, v)

        @property
        def num_tokens(self) -> int:
            if self._num_tokens is not None:
                return self._num_tokens
            return len(self.all_token_ids)

    req.Request = Request

    # ---- vllm.v1.kv_cache_interface ----------------------------------
    _install_kv_cache_interface()


def _install_kv_cache_interface() -> None:
    """The KV cache specs.

    These carry the per-group geometry the cap arithmetic reads -- `block_size`
    above all -- so they are dataclasses with the real field names rather than
    permissive mocks.  The five live groups are 256/64/64/4/8 tokens per block
    (backed out of the engine's own `legacy cap` lines, 2026-10-07).
    """
    iface = _mod("vllm.v1.kv_cache_interface")

    @dataclass
    class KVCacheSpec:
        block_size: int = 256

        @property
        def type_id(self) -> str:
            return f"{type(self).__name__}_{self.block_size}"

        def max_memory_usage_bytes(self, vllm_config=None) -> int:
            return self.page_size_bytes

        @property
        def page_size_bytes(self) -> int:
            return self.block_size

    @dataclass
    class AttentionSpec(KVCacheSpec):
        num_kv_heads: int = 1
        head_size: int = 576
        dtype: Any = "fp8"
        use_mla: bool = False

    @dataclass
    class FullAttentionSpec(AttentionSpec):
        sliding_window: int | None = None
        attention_chunk_size: int | None = None

    @dataclass
    class TQFullAttentionSpec(FullAttentionSpec):
        pass

    @dataclass
    class SinkFullAttentionSpec(FullAttentionSpec):
        sink_len: int | None = None

    @dataclass
    class MLAAttentionSpec(FullAttentionSpec):
        # The three triggers MLAAttentionManager._should_protect_prompt_blocks
        # reads.  Defaults are DSv4's, which is the engine we run.
        use_mla: bool = True
        model_version: str = "deepseek_v4"
        cache_dtype_str: str = "fp8_ds_mla"
        compress_ratio: int = 2

    @dataclass
    class SlidingWindowSpec(AttentionSpec):
        sliding_window: int = 128

        def max_admission_blocks_per_request(
            self, max_num_batched_tokens: int, max_model_len: int
        ) -> int:
            import math

            return math.ceil(self.sliding_window / self.block_size) + math.ceil(
                max_num_batched_tokens / self.block_size
            )

    @dataclass
    class SlidingWindowMLASpec(SlidingWindowSpec):
        use_mla: bool = True
        model_version: str = "deepseek_v4"
        cache_dtype_str: str = "fp8_ds_mla"
        compress_ratio: int = 2

    @dataclass
    class ChunkedLocalAttentionSpec(AttentionSpec):
        attention_chunk_size: int = 128

        def max_admission_blocks_per_request(
            self, max_num_batched_tokens: int, max_model_len: int
        ) -> int:
            import math

            return math.ceil(self.attention_chunk_size / self.block_size) + math.ceil(
                max_num_batched_tokens / self.block_size
            )

    @dataclass
    class CrossAttentionSpec(AttentionSpec):
        pass

    @dataclass
    class HiddenStateCacheSpec(KVCacheSpec):
        pass

    @dataclass
    class MambaSpec(KVCacheSpec):
        shapes: tuple = ()
        dtypes: tuple = ()
        page_size_padded: int | None = None
        mamba_type: str = "mamba2"
        num_speculative_blocks: int = 0

    @dataclass
    class KVCacheTensor:
        size: int = 0
        shared_by: list = field(default_factory=list)

    @dataclass
    class KVCacheGroupSpec:
        layer_names: list = field(default_factory=list)
        kv_cache_spec: Any = None
        is_eagle_group: bool = False

    @dataclass
    class KVCacheConfig:
        num_blocks: int = 0
        kv_cache_tensors: list = field(default_factory=list)
        kv_cache_groups: list = field(default_factory=list)

    @dataclass
    class UniformTypeKVCacheSpecs(KVCacheSpec):
        kv_cache_specs: dict = field(default_factory=dict)

        @classmethod
        def is_uniform_type(cls, specs) -> bool:
            return False

        @classmethod
        def from_specs(cls, specs):
            return None

    for name, obj in list(locals().items()):
        if name.startswith("_"):
            continue
        setattr(iface, name, obj)


# ---------------------------------------------------------------------------
# numpy and torch, stubbed rather than installed.
#
# WHY NOT JUST pip install: this harness's whole value is that it runs on a
# laptop with NOTHING installed -- that is what makes it the thing you reach for
# before a 4-minute boot. torch would have to be stubbed regardless (it is a
# multi-GB dependency for one annotation), so stubbing numpy beside it keeps the
# harness dependency-free rather than half-and-half.
#
# *** WHAT THESE DO AND DO NOT MODEL. *** kv_offload/base.py needs exactly two
# things: np.array(ids, dtype=np.int64) in BlockIDsLoadStoreSpec.__init__, and
# torch.Tensor as an evaluated dataclass annotation. Both are modelled. NOTHING
# ELSE IS, so a test must not assert on numpy broadcasting, dtype coercion or
# tensor behaviour -- it would be asserting on this file instead of on vLLM. The
# contract tests are written to read .block_ids as a sequence and no more.
# ---------------------------------------------------------------------------


def _install_numpy_stub() -> None:
    if "numpy" in sys.modules:
        return
    np = types.ModuleType("numpy")

    class _NDArray(list):
        """Enough of an ndarray for our one call site: a sequence with a dtype."""

        def __init__(self, data, dtype=None):
            super().__init__(data)
            self.dtype = dtype

        def tolist(self):
            return list(self)

        def __repr__(self):
            return f"array({list(self)})"

    np.ndarray = _NDArray
    np.int64 = "int64"
    np.int32 = "int32"
    np.array = lambda data, dtype=None: _NDArray(data, dtype)
    sys.modules["numpy"] = np


def _install_torch_stub() -> None:
    if "torch" in sys.modules:
        return
    torch = types.ModuleType("torch")

    class _Tensor:
        pass

    torch.Tensor = _Tensor
    # tiering/spec.py's create_handlers reads the worker's own rank from here,
    # and OUR spec will have to do the same. Stubbed to 0 so a test can assert
    # the per-rank path shape without a GPU.
    accel = types.SimpleNamespace(current_device_index=lambda: 0)
    torch.accelerator = accel
    torch.int64 = "torch.int64"
    sys.modules["torch"] = torch


_install_numpy_stub()
_install_torch_stub()
