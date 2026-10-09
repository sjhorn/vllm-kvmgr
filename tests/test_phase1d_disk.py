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

print("== the serializer shim: a numpy scalar must not kill the output thread ==")
# *** THIS PRESENTED AS A HANG, TWICE. *** vLLM's enc_hook handles np.ndarray
# but NOT numpy scalars, so one numpy int in EngineCoreOutputs raises inside
# process_output_sockets -- killing that THREAD, not the engine. Results then
# never reach the API server: requests complete, the core goes idle on an empty
# input queue, clients hang for ever, and nothing logs an error. I read that as
# a memory ceiling for an evening.
import types as _types

_calls = []


class _FakeEncoder:
    def enc_hook(self, obj):
        _calls.append(obj)
        raise TypeError("Object of type %s is not serializable" % type(obj))


class _FakeGeneric:
    """Stands in for numpy.generic: has .item() and is caught by isinstance."""

    def __init__(self, v):
        self._v = v

    def item(self):
        return self._v

    def __repr__(self):
        return "numpy.int64(%r)" % self._v


_fake_np = _types.ModuleType("numpy")
_fake_np.generic = _FakeGeneric
_fake_su = _types.ModuleType("vllm.v1.serial_utils")
_fake_su.MsgpackEncoder = _FakeEncoder

_saved = {k: sys.modules.get(k) for k in ("numpy", "vllm.v1.serial_utils")}
sys.modules["numpy"] = _fake_np
sys.modules["vllm.v1.serial_utils"] = _fake_su
try:
    import importlib
    reg = importlib.import_module("kvmgr.register")
    reg._install_serializer_shim()
    check("the shim installs",
          getattr(_FakeEncoder.enc_hook, "_kvmgr_wrapped", False))
    reg._install_serializer_shim()
    check("...and is idempotent",
          getattr(_FakeEncoder.enc_hook, "_kvmgr_wrapped", False))

    enc = _FakeEncoder()

    # Wrapped: under mutation these RAISE, and an escaping exception aborts the
    # file instead of reporting a named failure. Third time this suite has
    # needed that correction -- any assertion whose subject can raise under
    # mutation has to catch, or break-verification reads a crash as a pass.
    def _hook_safe(v):
        try:
            return enc.enc_hook(v)
        except Exception as exc:
            return "RAISED: %r" % (exc,)

    eq("*** a numpy scalar is COERCED, not raised ***",
       _hook_safe(_FakeGeneric(22755)), 22755)
    eq("...losslessly", _hook_safe(_FakeGeneric(1039680)), 1039680)
    eq("...and upstream's hook was never reached for it", _calls, [])

    # `except TypeError` was too narrow: an over-wide shim calls .item() on a
    # plain object and raises AttributeError, which escaped and aborted the
    # file. Catch everything and DISTINGUISH, so "upstream raised correctly"
    # and "the shim grabbed something it should not have" are separate results.
    _what = None
    try:
        enc.enc_hook(object())
    except Exception as exc:
        _what = type(exc).__name__
    check("*** a non-numpy object still reaches upstream's hook, so the shim "
          "narrows nothing ***",
          _what == "TypeError" and len(_calls) == 1)
    check("...and the shim did NOT try to coerce it", _what != "AttributeError")
finally:
    for _k, _v in _saved.items():
        if _v is not None:
            sys.modules[_k] = _v
        else:
            sys.modules.pop(_k, None)

print("== the producer walk: name the FIELD, not just the value ==")
# Two runs logged "value=1" then "value=39" -- a varying small count, which
# narrows the field without naming it. msgspec hands enc_hook the scalar alone,
# so the only way to name the producer is to walk the object being encoded.
import importlib as _il2
_reg2 = _il2.import_module("kvmgr.register")
_walk = _reg2._numpy_paths


class _NpInt(int):
    __module__ = "numpy"


# the real numpy is not installed in the harness, so the walk's isinstance
# against np.generic cannot match -- drive it with a stub module instead
import sys as _s2
import types as _t2
_np_stub = _t2.ModuleType("numpy")


class _Gen:
    def __init__(self, v):
        self.v = v
    def __repr__(self):
        return repr(self.v)


_np_stub.generic = _Gen
_np_stub.ndarray = type("ndarray", (), {})
_saved_np = _s2.modules.get("numpy")
_s2.modules["numpy"] = _np_stub
try:
    class _Stats:
        __slots__ = ("hits", "queries")
        def __init__(self):
            self.hits = _Gen(39)
            self.queries = 7

    class _Out:
        def __init__(self):
            self.scheduler_stats = _Stats()
            self.things = [1, {"deep": _Gen(1)}]

    _paths = _walk(_Out())
    check("*** the walk names a __slots__ attribute path ***",
          any("outputs.scheduler_stats.hits" in p for p in _paths), _paths)
    check("...and one nested in a list and a dict",
          any("things[1]['deep']" in p for p in _paths), _paths)
    check("...carrying the VALUE too, so it ties to the hook's line",
          any("39" in p for p in _paths), _paths)
    eq("a clean object yields no paths", _walk(_Stats.__new__(_Stats)) if False
       else _walk(type("Clean", (), {"a": 1})()), [])

    # bounded: a cycle must not spin, and a deep chain must stop
    class _Node:
        def __init__(self):
            self.next = None
            self.val = _Gen(5)

    _a, _b = _Node(), _Node()
    _a.next, _b.next = _b, _a
    _cyc = _walk(_a)
    # *** THE DEPTH CAP IS WHAT STOPS A CYCLE, NOT THE VISITED SET. ***
    # I claimed the opposite in the walker's own docstring, and
    # break-verification refuted it: removing the visited set left this green,
    # because depth terminates the walk either way. So assert the real
    # properties -- termination, and a bound on findings -- and test the
    # visited set for what it actually does, which is avoid re-walking a
    # SHARED subgraph.
    check("*** a CYCLE terminates (the DEPTH cap does this) ***",
          isinstance(_cyc, list))
    check("...and findings are capped", len(_walk(_a, limit=1)) <= 1)
    check("...and a shallow depth cap genuinely truncates",
          len(_walk(_a, max_depth=0)) == 0)

    # A shared CONTAINER, not a shared scalar: `walk` returns on a scalar
    # before touching the visited set, so a shared scalar is deliberately
    # reported once PER PATH -- which is more useful for diagnosis, since both
    # routes to it are named. My first version of this assertion got that
    # backwards and failed with the visited set present.
    _shared_box = {"v": _Gen(11)}

    class _Diamond:
        def __init__(self):
            self.left = _shared_box
            self.right = _shared_box

    eq("*** a SHARED CONTAINER is walked once, not once per path -- that is "
       "what the visited set is for ***", len(_walk(_Diamond())), 1)
finally:
    if _saved_np is not None:
        _s2.modules["numpy"] = _saved_np
    else:
        _s2.modules.pop("numpy", None)

print("== the proactive scan's WINDOW: it must cover the life, not the boot ==")
# *** THE FIRST VERSION OF THIS SCAN COULD NOT SEE THE EVENT IT EXISTED FOR. ***
# Its budget was "the first 25 encodes of the process" -- which are the BOOT's
# encodes, handshake and startup traffic carrying no generated output. The
# numpy scalar only ever appeared under load. So the scan ran, cost its walks,
# found nothing, and read as coverage. No test covered the window, which is
# exactly why the dead window shipped; this is that test.
import types as _t3
import sys as _s3
import importlib as _il3

_np3 = _t3.ModuleType("numpy")


class _G3:
    def __init__(self, v):
        self.v = v

    def item(self):
        return self.v

    def __repr__(self):
        return "numpy.int64(%r)" % self.v


_np3.generic = _G3
_np3.ndarray = type("ndarray", (), {})

_warned = []


class _Enc3:
    def enc_hook(self, obj):
        raise TypeError("not serializable")

    def encode_into(self, obj, buf, *a, **kw):
        return b""


_su3 = _t3.ModuleType("vllm.v1.serial_utils")
_su3.MsgpackEncoder = _Enc3

_sv3 = {k: _s3.modules.get(k) for k in ("numpy", "vllm.v1.serial_utils")}
_s3.modules["numpy"] = _np3
_s3.modules["vllm.v1.serial_utils"] = _su3
try:
    reg3 = _il3.import_module("kvmgr.register")

    class _Clock:
        def __init__(self):
            self.t = 1000.0

        def monotonic(self):
            return self.t

    _clk = _Clock()
    _real_time, reg3._time = reg3._time, _clk
    _real_log, reg3.logger = reg3.logger, _t3.SimpleNamespace(
        warning=lambda *a, **k: _warned.append(a[0] % tuple(a[1:])),
        info=lambda *a, **k: None,
        exception=lambda *a, **k: _warned.append("EXCEPTION"))
    try:
        reg3._install_serializer_shim()
        enc3 = _Enc3()
        buf = bytearray()

        # *** THE WINDOW TEST: a clean payload for far more encodes than the
        # *** old budget of 25, and only THEN the scalar. *** Under the old
        # code every one of these would have burned the budget on boot
        # traffic and the scalar would never be scanned.
        class _Clean:
            def __init__(self):
                self.a = 1

        class _Dirty:
            def __init__(self):
                self.late_field = _G3(39)

        for n in range(60):
            _clk.t += 6.0           # past the spacing, so each one scans
            enc3.encode_into(_Clean(), buf)
        eq("60 clean encodes past the old 25-encode budget find nothing",
           _warned, [])

        _clk.t += 6.0
        enc3.encode_into(_Dirty(), buf)
        check("*** a scalar on encode #61 is STILL named -- the old window "
              "would have been spent on boot traffic ***",
              any("late_field" in w and "39" in w for w in _warned), _warned)
        check("...and says it found it by scanning, before any coercion",
              any("before any coercion" in w for w in _warned), _warned)

        _found = len(_warned)
        _clk.t += 6.0
        enc3.encode_into(_Dirty(), buf)
        eq("...and it stops once found, so one path is reported not thousands",
           len(_warned), _found)
    finally:
        reg3._time, reg3.logger = _real_time, _real_log

    # THE SPACING, verified the same way: without it a busy output thread pays
    # a walk per encode. Drive many encodes inside one spacing interval.
    _warned2 = []
    _scans = []
    _real_walk = reg3._numpy_paths
    _real_time2, reg3._time = reg3._time, _clk
    try:
        reg3._numpy_paths = lambda o, **k: (_scans.append(1), [])[1]
        # A fresh shim needs a class with its OWN methods. *** SUBCLASSING
        # _Enc3 INHERITED `_kvmgr_wrapped`, so the install returned early and
        # this drove the PREVIOUS shim -- whose root already had found=True, so
        # the assertion saw 0 walks and read as a spacing that never fires. ***
        class _Enc4:
            def enc_hook(self, obj):
                raise TypeError("not serializable")

            def encode_into(self, obj, buf, *a, **kw):
                return b""
        _su3.MsgpackEncoder = _Enc4
        reg3._install_serializer_shim()
        enc4 = _Enc4()
        _clk.t += 100.0
        for n in range(40):
            _clk.t += 0.1           # 4.0s total: INSIDE one 5s interval
            enc4.encode_into(_Clean(), buf)
        check("*** 40 encodes inside one spacing interval cost ONE walk, not "
              "40 -- this runs in the engine's output thread ***",
              len(_scans) == 1, len(_scans))
    finally:
        reg3._numpy_paths = _real_walk
        reg3._time = _real_time2
finally:
    for _k, _v in _sv3.items():
        if _v is not None:
            _s3.modules[_k] = _v
        else:
            _s3.modules.pop(_k, None)


print(f"\n{PASS} passed, {len(FAIL)} failed")
for f in FAIL:
    print("  FAILED:", f)
sys.exit(1 if FAIL else 0)
