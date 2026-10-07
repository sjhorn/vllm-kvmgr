"""test_phase1d_disk.py -- the on-disk FINGERPRINT, and the fork that killed restore.

    python3.12 local-ops/kvharness/test_phase1d_disk.py

*** THE BUG THIS FILE EXISTS FOR. *** Measured live 2026-10-07: the worker wrote
943 files per rank under `..._06be92914bd2_r0` while the engine core's manifest
prune looked under `..._500ceb39bfc9` and discarded all 552 keys as "the file is
gone" -- with the files sitting right there on disk. Restore could never fire,
and nothing errored: a fingerprint fork is a silent miss by construction, which
is the worst shape a bug can have here.

The two `fields` dicts differed in EXACTLY ONE key, `hash_block_size`: 256 in the
worker against 4 in the engine core. `FileMapper.from_offloading_spec` reads
`vllm_config.cache_config.block_size` (file_mapper.py:87), and on a HYBRID layout
the engine core rewrites that attribute to the resolved GCD of the five group
block sizes while the worker keeps the CLI's `--block-size 256`. Upstream's
helper is correct for a uniform model, which is why nothing upstream trips on it.

So every assertion here is about one property: **the base path must not depend on
`cache_config.block_size`.** That is the only field known to differ between the
two processes, and it is the one the helper used.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "patches", "kvmgr_plugin"))

import harness  # noqa: E402

PASS = 0
FAIL: list[str] = []


def check(name, cond, detail=""):
    global PASS
    if cond:
        PASS += 1
        print(f"PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAIL.append(name)
        print(f"FAIL  {name}  {detail}")


def eq(name, got, want):
    check(name, got == want, f"got {got!r} want {want!r}")


harness.load_kv_offload()
from kvmgr import diskspec  # noqa: E402

FileMapper = sys.modules["vllm.v1.kv_offload.file_mapper"].FileMapper

# The five groups this engine really has, from the boot line's legacy caps.
GROUPS = [(256, "g0"), (64, "g1"), (64, "g2"), (4, "g3"), (8, "g4")]


class _Cfg:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def fake_self(cache_block_size: int, *, hash_block_size: int = 4, rank: int = 0):
    """A stand-in for a constructed spec, differing ONLY in the forked field.

    `cache_block_size` is what `cache_config.block_size` reads in this process:
    256 in a worker, 4 in the engine core. A correct mapper ignores it.
    """
    vllm_config = _Cfg(
        model_config=_Cfg(model="deepseek-ai/DeepSeek-V4-Flash-Vision-Exp"),
        cache_config=_Cfg(block_size=cache_block_size, cache_dtype="nvfp4_ds_mla"),
        parallel_config=_Cfg(
            tensor_parallel_size=2, pipeline_parallel_size=1,
            prefill_context_parallel_size=1, decode_context_parallel_size=1,
            rank=rank, world_size=2,
        ),
    )
    kv_cache_config = _Cfg(
        kv_cache_groups=[
            _Cfg(kv_cache_spec=_Cfg(block_size=bs), layer_names=[name])
            for bs, name in GROUPS
        ],
    )
    return _Cfg(
        vllm_config=vllm_config,
        kv_cache_config=kv_cache_config,
        hash_block_size=hash_block_size,
        block_size_factor=1,
        root_dir="/kvoffload/" + diskspec.DISK_FORMAT_ROOT,
        _mapper=None,
    )


def mapper_for(**kw):
    """Call the real `file_mapper` property function against a fake self."""
    fn = diskspec.DSparkDiskOffloadingSpec.file_mapper
    return fn(fake_self(**kw))


print("\n--- the fork itself: two processes, one path ---")

worker = mapper_for(cache_block_size=256)     # what the worker sees
engine = mapper_for(cache_block_size=4)       # what the engine core sees

eq("THE FIX: worker and engine-core agree on the base path",
   worker.base_path, engine.base_path)
eq("...and on every field", worker.fields, engine.fields)
check("the live fork is reproduced by the HELPER we stopped using",
      True, "see the next block")

# Prove the bug is real rather than asserted -- build both sides the way
# `from_offloading_spec` does and show they diverge. If this ever stops
# diverging, upstream fixed it and the override can go.
def helper_fields(cache_block_size):
    s = fake_self(cache_block_size=cache_block_size)
    return FileMapper(
        root_dir=s.root_dir,
        model_name=s.vllm_config.model_config.model,
        hash_block_size=s.vllm_config.cache_config.block_size,   # <- the bug
        gpu_blocks_per_file=1,
        tp_size=2, pp_size=1, pcp_size=1, dcp_size=1, rank=0,
        dtype="nvfp4_ds_mla",
        kv_cache_groups=[{"block_size": bs, "layer_names": [n]} for bs, n in GROUPS],
    )


h_worker, h_engine = helper_fields(256), helper_fields(4)
check("upstream's own derivation DOES fork on this layout",
      h_worker.base_path != h_engine.base_path,
      f"{os.path.basename(h_worker.base_path)} != {os.path.basename(h_engine.base_path)}")
eq("...and it forks in exactly one field, as measured",
   [k for k in h_worker.fields if h_worker.fields[k] != h_engine.fields[k]],
   ["hash_block_size"])

print("\n--- the fingerprint still fingerprints ---")

eq("hash_block_size is the RESOLVED value, not the CLI's",
   worker.fields["hash_block_size"], 4)
check("the five real block sizes are in the hash",
      [g["block_size"] for g in worker.fields["kv_cache_groups"]]
      == [bs for bs, _ in GROUPS])

for field, alt in (("hash_block_size", 8),):
    other = mapper_for(cache_block_size=256, hash_block_size=alt)
    check(f"a changed {field} still changes the path",
          other.base_path != worker.base_path,
          f"{alt} vs 4")

# A different model, quantisation or TP size must NOT inherit the same bytes.
for label, mut in (
    ("model", lambda s: setattr(s.vllm_config.model_config, "model", "other/model")),
    ("dtype", lambda s: setattr(s.vllm_config.cache_config, "cache_dtype", "fp8")),
    ("tp_size", lambda s: setattr(s.vllm_config.parallel_config,
                                  "tensor_parallel_size", 4)),
    ("group block size", lambda s: setattr(
        s.kv_cache_config.kv_cache_groups[0].kv_cache_spec, "block_size", 128)),
):
    s = fake_self(cache_block_size=256)
    mut(s)
    got = diskspec.DSparkDiskOffloadingSpec.file_mapper(s)
    check(f"a changed {label} refuses the old bytes", got.base_path != worker.base_path)

print("\n--- rank lives OUTSIDE the hash, which is what shares the index ---")

r1 = mapper_for(cache_block_size=4, rank=1)
eq("both ranks share one base", r1.base_path, worker.base_path)
check("...and separate only in the file name",
      r1.rank == 1 and worker.rank == 0)

print("\n--- upstream drift: a new field must fail LOUDLY, not be dropped ---")

# We construct FileMapper directly, so a field upstream adds would silently
# default. The contract to re-check on an engine bump is this key set.
probe = FileMapper(
    root_dir="/x", model_name="m", hash_block_size=4, gpu_blocks_per_file=1,
    tp_size=1, pp_size=1, pcp_size=1, dcp_size=1, rank=0, dtype="d",
    kv_cache_groups=[],
)
eq("our fields cover upstream's whole set",
   sorted(worker.fields), sorted(probe.fields))

print("\n--- `disk_store_live` is a FACT, not a flag that governs nothing ---")

# Measured 2026-10-07: /kv reported `disk_enabled: false` while the worker was
# writing. The flag is BLOCKED and toggles nothing; what selects the spec is
# KV_TRANSFER_CONFIG. So the field must follow the config, not the flag.
from kvmgr import stats as _stats  # noqa: E402

_saved = os.environ.get("KV_TRANSFER_CONFIG")
try:
    for label, val, want in (
        ("unset", None, False),
        ("empty", "", False),
        ("another spec", '{"kv_connector_extra_config":'
                         '{"spec_name":"TieringOffloadingSpec"}}', False),
        ("ours", '{"kv_connector":"OffloadingConnector","kv_connector_extra_config":'
                 '{"spec_name":"' + diskspec.DSPARK_SPEC_NAME + '"}}', True),
        # It must never be the thing that raises: an observability field that
        # throws takes the whole endpoint with it.
        ("unparseable, naming ours", "{not json " + diskspec.DSPARK_SPEC_NAME, True),
        ("unparseable, naming nothing", "{not json at all", False),
    ):
        if val is None:
            os.environ.pop("KV_TRANSFER_CONFIG", None)
        else:
            os.environ["KV_TRANSFER_CONFIG"] = val
        eq(f"disk_store_live with KV_TRANSFER_CONFIG {label}",
           _stats._disk_spec_is_live(), want)
finally:
    os.environ.pop("KV_TRANSFER_CONFIG", None)
    if _saved is not None:
        os.environ["KV_TRANSFER_CONFIG"] = _saved

eq("the registered name and the config key are ONE constant",
   diskspec.DSPARK_SPEC_NAME, "DSparkDiskOffloadingSpec")

print("\n--- the hang dump: no ptrace, and it must NOT kill the process ---")

# The store path has wedged the worker twice with no stack obtainable: the
# container has no SYS_PTRACE, so `gdb -p` answers "Operation not permitted"
# and there is no py-spy. faulthandler needs neither. Run in a CHILD process,
# because the assertion is that the process SURVIVES its own signal -- getting
# that wrong in-process would take the test runner down with it.
import subprocess  # noqa: E402

_PLUG = os.path.join(REPO, "patches", "kvmgr_plugin")
_STUB = (
    "import sys, types, logging\n"
    "_v = types.ModuleType('vllm'); _l = types.ModuleType('vllm.logger')\n"
    "_l.init_logger = lambda name: logging.getLogger(name)\n"
    "_v.logger = _l\n"
    "sys.modules['vllm'] = _v; sys.modules['vllm.logger'] = _l\n"
    "sys.path.insert(0, %r)\n" % (_PLUG,)
)

_child = _STUB + r"""
import os, signal, sys
import importlib
R = importlib.import_module("kvmgr.register")
R._install_hang_dump()

def deep(n):
    if n:
        return deep(n - 1)
    os.kill(os.getpid(), signal.SIGUSR1)
    return 1

deep(4)
print("SURVIVED", flush=True)
"""

proc = subprocess.run([sys.executable, "-c", _child], capture_output=True,
                      text=True, timeout=60)
check("the process survives its own dump signal",
      proc.returncode == 0 and "SURVIVED" in proc.stdout,
      f"rc={proc.returncode} out={proc.stdout.strip()[:60]!r}")
check("...and every thread's Python stack reaches stderr",
      "Current thread" in proc.stderr or "Thread 0x" in proc.stderr,
      proc.stderr.strip().splitlines()[:1])
check("...naming the frames, so a deadlock in OUR code is identifiable",
      "deep" in proc.stderr, "the recursion marker is in the dump")

# Arming must never be the thing that breaks a boot.
_child_bad = _STUB + r"""
import signal
del signal.SIGUSR1           # a platform without it
import importlib
R = importlib.import_module("kvmgr.register")
R._install_hang_dump()
print("BOOTED", flush=True)
"""
proc2 = subprocess.run([sys.executable, "-c", _child_bad], capture_output=True,
                       text=True, timeout=60)
check("no SIGUSR1 on the platform is a no-op, not a failed boot",
      proc2.returncode == 0 and "BOOTED" in proc2.stdout,
      f"rc={proc2.returncode}")

# And the half that makes chaining safe rather than fatal: a REAL handler must
# still be chained to, or a future vLLM that takes SIGUSR1 silently loses it.
_child_chain = _STUB + r"""
import os, signal, sys, importlib
seen = []
signal.signal(signal.SIGUSR1, lambda *a: seen.append(1))
R = importlib.import_module("kvmgr.register")
R._install_hang_dump()
os.kill(os.getpid(), signal.SIGUSR1)
print("CHAINED" if seen else "LOST", flush=True)
"""
proc3 = subprocess.run([sys.executable, "-c", _child_chain], capture_output=True,
                       text=True, timeout=60)
check("an existing handler is still chained to, not stolen",
      "CHAINED" in proc3.stdout, f"rc={proc3.returncode} {proc3.stdout.strip()!r}")
check("...and that case dumps the stack too",
      "Current thread" in proc3.stderr or "Thread 0x" in proc3.stderr)

print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
