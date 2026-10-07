"""test_phase0_disk.py -- PIN THE CONTRACT our own disk OffloadingSpec will
implement, before a line of it is written.

    python3.12 local-ops/kvharness/test_phase0_disk.py

WHY THIS FILE EXISTS. Upstream's tiering offload cannot capture half of this
cluster's KV: it consolidates every rank's shard into ONE SHARED HOST memory
region (their docs say so in as many words), which is single-host TP by
construction, while ours is one rank per node. Measured 2026-10-07: every .bin
on disk is page*world_size bytes with the second half EXACTLY ZERO -- rank 1's
slice, never written, because the secondary tiers live scheduler-side and
/dev/shm does not cross machines.

So we write our own spec. These assertions are against the REAL
vllm/v1/kv_offload/{base,worker/worker}.py, loaded from patches/pristine/, for
one reason: the most expensive way to get Phase 1 wrong is to write it against a
contract I only skimmed. Every `pin` below is a thing Phase 1 depends on, so an
upstream bump that moves it fails HERE -- on a laptop, in a second -- rather than
at boot, four minutes in, with a cold prefill to pay for.

WHAT THIS CANNOT TELL YOU: nothing here touches a tensor, a GPU or a real file
transfer. It pins the SHAPE of the contract and the arithmetic. The gate that
proves the premise is Phase 1's, on the box: both nodes must grow a `ds4f`
directory and the files must carry no zero tail.
"""

from __future__ import annotations

import inspect
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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


base = harness.load_kv_offload()
wrk = sys.modules["vllm.v1.kv_offload.worker.worker"]
fmap = sys.modules["vllm.v1.kv_offload.file_mapper"]
fact = sys.modules["vllm.v1.kv_offload.factory"]

LIVE_BLOCK_SIZES = harness.LIVE_BLOCK_SIZES          # (256, 64, 64, 4, 8)
WORLD_SIZE = 2                                        # one rank per node
WHOLE_PAGE = 2_103_552                                # measured on disk

print("== (0a) the abstract surface Phase 1 must implement ==")

# If an upstream bump adds an abstract method, our spec stops being
# instantiable -- and it would fail at BOOT, in the engine core, where the only
# symptom this project reliably gets is "the plugin did not load".
eq("OffloadingManager has exactly three abstract methods",
   sorted(base.OffloadingManager.__abstractmethods__),
   ["lookup", "prepare_load", "prepare_store"])
eq("OffloadingSpec has exactly two",
   sorted(base.OffloadingSpec.__abstractmethods__),
   ["get_handlers", "get_manager"])
eq("OffloadingHandler has exactly three",
   sorted(wrk.OffloadingHandler.__abstractmethods__),
   ["get_finished", "transfer_async", "wait"])
check("touch/complete_*/take_events/reset_cache are CONCRETE, so we inherit them",
      all(m not in base.OffloadingManager.__abstractmethods__
          for m in ("touch", "complete_load", "complete_store",
                    "take_events", "reset_cache", "shutdown")))

print()
print("== (0b) MEDIUM is the routing key, and ours must be novel ==")

# worker.py: "transfers are forwarded to workers by (src_medium, dst_medium)",
# and register_handler ASSERTS on a duplicate pair. So picking a medium string
# already in use is not a subtle collision -- it is a boot-time assert.
check("LoadStoreSpec.medium is abstract", "medium" in base.LoadStoreSpec.__abstractmethods__)
check("...and is a staticmethod",
      isinstance(inspect.getattr_static(base.LoadStoreSpec, "medium"), staticmethod))
eq("GPU's medium string", base.GPULoadStoreSpec.medium(), "GPU")

OUR_MEDIUM = "DSPARK_DISK"

# CHECK THE COLLISION DIRECTLY, rather than leaving it to blow up downstream.
# With a used string the duplicate-pair assert below does fire -- but as a bare
# traceback with no name, which is this repo's "fails for the wrong reason" trap.
# Scanned out of the pristine tree so a medium added upstream is picked up.
_PRISTINE_KVO = os.path.join(REPO, "patches", "pristine", "vllm", "v1", "kv_offload")
UPSTREAM_MEDIA = set()
for _dir, _, _files in os.walk(_PRISTINE_KVO):
    for _f in _files:
        if not _f.endswith(".py"):
            continue
        _src = open(os.path.join(_dir, _f)).read()
        for _m in re.finditer(r'def medium\(\)[^\n]*\n\s*return\s+"([^"]+)"', _src):
            UPSTREAM_MEDIA.add(_m.group(1))
check("found the upstream medium strings to avoid", bool(UPSTREAM_MEDIA),
      repr(sorted(UPSTREAM_MEDIA)))
eq("...and it is the WHOLE set -- the scan is only as complete as the tree, so "
   "this fails if a pristine file is missing or upstream adds a medium",
   sorted(UPSTREAM_MEDIA), ["CPU", "GPU"])
check("*** our medium collides with none of them ***",
      OUR_MEDIUM not in UPSTREAM_MEDIA, f"{OUR_MEDIUM} vs {sorted(UPSTREAM_MEDIA)}")


class DiskSpecProbe(base.LoadStoreSpec):
    """Minimal stand-in for Phase 1's DiskLoadStoreSpec: just the medium."""

    @staticmethod
    def medium() -> str:
        return OUR_MEDIUM


w = wrk.OffloadingWorker()


class H(wrk.OffloadingHandler):
    def __init__(self):
        self.calls = []

    def transfer_async(self, job_id, spec):
        self.calls.append((job_id, spec[0].medium(), spec[1].medium()))
        return True

    def get_finished(self):
        return []

    def wait(self, job_ids):
        return None


h_store, h_load = H(), H()
w.register_handler(base.GPULoadStoreSpec, DiskSpecProbe, h_store)
w.register_handler(DiskSpecProbe, base.GPULoadStoreSpec, h_load)
eq("both directions register under their medium PAIR",
   sorted(w.transfer_type_to_handler),
   [("DSPARK_DISK", "GPU"), ("GPU", "DSPARK_DISK")])

dup = False
try:
    w.register_handler(base.GPULoadStoreSpec, DiskSpecProbe, H())
except AssertionError:
    dup = True
check("*** a duplicate medium pair ASSERTS -- so our medium must be novel ***", dup)

print()
print("== (0c) routing actually reaches the right handler ==")

gpu = base.GPULoadStoreSpec(block_ids=[7, 8, 9], group_sizes=[3], block_indices=[0])
ok = w.transfer_async(1, (gpu, DiskSpecProbe()))
check("a GPU->DISK transfer is submitted", ok)
eq("and it reached the STORE handler", h_store.calls, [(1, "GPU", "DSPARK_DISK")])
eq("and not the load handler", h_load.calls, [])
w.transfer_async(2, (DiskSpecProbe(), gpu))
eq("the reverse reaches the LOAD handler", h_load.calls, [(2, "DSPARK_DISK", "GPU")])

print()
print("== (0d) TransferResult is where the Prometheus labels come from ==")

# THIS IS NOT COSMETIC, and today proved it. The absence of a
# `transfer_type="CPU_to_GPU"` series is what told us prepare_load had never
# been called once -- a stronger statement than "the lookup missed". A handler
# that omits these fields makes the restore path UNOBSERVABLE.
res = wrk.TransferResult(job_id=1, success=True, transfer_size=123,
                         transfer_time=0.5, transfer_type=("DSPARK_DISK", "GPU"))
eq("TransferResult carries size", res.transfer_size, 123)
eq("...time", res.transfer_time, 0.5)
eq("...and the medium pair that becomes the metric label", res.transfer_type,
   ("DSPARK_DISK", "GPU"))
check("size/time/type all default to None, so forgetting them is silent",
      wrk.TransferResult(job_id=2, success=True).transfer_size is None)

print()
print("== (0e) GPULoadStoreSpec's group contract -- the 5-group layout ==")

# "blocks are expected to be ordered by the group index" and
# sum(group_sizes) == len(block_ids). Our handler has to split one flat
# block_ids array into five groups, so these two invariants ARE the parser.
g = base.GPULoadStoreSpec(block_ids=[1, 2, 3, 4, 5], group_sizes=[2, 1, 1, 1, 0],
                          block_indices=[0, 0, 0, 0, 0])
eq("group_sizes is kept as given", list(g.group_sizes), [2, 1, 1, 1, 0])
eq("block_ids survives as a sequence", list(g.block_ids), [1, 2, 3, 4, 5])
eq("one block_index per group", len(g.block_indices), len(g.group_sizes))

for name, ids, sizes, idx in [
    ("sum(group_sizes) != len(block_ids) is refused", [1, 2], [3], [0]),
    ("len(block_indices) != len(group_sizes) is refused", [1, 2], [1, 1], [0]),
]:
    bad = False
    try:
        base.GPULoadStoreSpec(block_ids=ids, group_sizes=sizes, block_indices=idx)
    except AssertionError:
        bad = True
    check(name, bad)

print()
print("== (0f) PrepareStoreOutput: the scheduler's answer shape ==")

pso = base.PrepareStoreOutput(keys_to_store=[], store_spec=DiskSpecProbe(),
                              evicted_keys=[])
check("keys_to_store / store_spec / evicted_keys",
      hasattr(pso, "keys_to_store") and hasattr(pso, "store_spec")
      and hasattr(pso, "evicted_keys"))
ev = base.OffloadingEvent(keys=[], medium=OUR_MEDIUM, removed=False)
eq("OffloadingEvent carries OUR medium verbatim", ev.medium, OUR_MEDIUM)
check("...and distinguishes stored from removed", ev.removed is False)

print()
print("== (0g) OUR SPEC NEEDS NO HYBRID SHIM, which is the find of this phase ==")

# `offload_allow_hybrid` exists only because TieringOffloadingSpec adds
# `assert len(self.gpu_block_size) == 1`. The BASE asserts something weaker that
# our five groups already satisfy, so a spec of our own inherits no such limit.
tier_src = open(os.path.join(REPO, "patches", "pristine", "vllm", "v1",
                             "kv_offload", "tiering", "spec.py")).read()
base_src = inspect.getsource(base.OffloadingSpec.__init__)
check("TieringOffloadingSpec is the one asserting a single block size",
      "assert len(self.gpu_block_size) == 1" in tier_src)
check("*** and the BASE does NOT ***",
      "len(self.gpu_block_size) == 1" not in base_src)
check("the base asserts only divisibility by hash_block_size",
      "% self.hash_block_size == 0" in base_src)
HASH_BLOCK_SIZE = 4   # the GCD of our five, resolved by patch8
check("...which all five of our block sizes satisfy",
      all(b % HASH_BLOCK_SIZE == 0 for b in LIVE_BLOCK_SIZES),
      f"{LIVE_BLOCK_SIZES} % {HASH_BLOCK_SIZE}")

# And block_size_factor must stay 1: setting it needs ONE block size, which we
# do not have. Factor 1 is also the only case the hybrid relaxation was ever
# safe for, so this is the same constraint seen from the other side.
check("block_size_factor defaults to 1", "self.block_size_factor: int = 1" in base_src)
check("...and setting it REQUIRES a single block size, so we must not",
      "len(gpu_block_sizes) == 1" in base_src)

print()
print("== (0h) the page arithmetic -- the number Phase 1's gate checks ==")

# Upstream sizes its region page*world_size and writes the WHOLE page to one
# file, which is why ours are half empty. Our handler writes only this rank's
# slice, so our files must be HALF the size.
eq("today's file size, measured on disk", WHOLE_PAGE, 2_103_552)
eq("*** per-worker slice, which is what OUR files must be ***",
   WHOLE_PAGE // WORLD_SIZE, 1_051_776)
check("tiering sizes its region by world_size (the half-empty cause)",
      "* world_size" in tier_src)

print()
print("== (0i) rank is in the PATH, deliberately not in the fingerprint ==")

# This is what makes per-rank files possible at all: _r0 and _r1 are siblings
# under ONE fingerprint, each rank restoring its own half.
fm_src = inspect.getsource(fmap.FileMapper)
check("get_file_name appends _r<rank>", "_r" in fm_src and "rank" in fm_src)
check("...and the config file is shared BECAUSE rank is outside the hash",
      "rank lives outside the hash" in fm_src)
for field in ("model_name", "hash_block_size", "tp_size", "dtype", "kv_cache_groups"):
    check(f"fingerprint covers {field}", f'"{field}"' in fm_src)
check("fingerprint does NOT cover rank",
      '"rank":' not in fm_src)
check("*** nor any format version of OURS -- Phase 4 adds one ***",
      "format_version" not in fm_src and "dspark" not in fm_src.lower())

print()
print("== (0j) the factory is a public registry, so no upstream patch ==")

check("register_spec exists", hasattr(fact.OffloadingSpecFactory, "register_spec"))
fact.OffloadingSpecFactory.register_spec(
    "DSparkDiskOffloadingSpec", "kvmgr.diskspec", "DSparkDiskOffloadingSpec")
clash = False
try:
    fact.OffloadingSpecFactory.register_spec(
        "DSparkDiskOffloadingSpec", "kvmgr.diskspec", "DSparkDiskOffloadingSpec")
except ValueError:
    clash = True
check("our name registers, and re-registering is refused", clash)
check("the two upstream specs are still there",
      "CPUOffloadingSpec" in open(os.path.join(
          REPO, "patches", "pristine", "vllm", "v1", "kv_offload",
          "factory.py")).read())

print()
print("== (0k) what the CONNECTOR already gives us -- the control channel ==")

conn = open(os.path.join(REPO, "patches", "pristine", "vllm",
                         "distributed_kv_connector_offloading_connector.py")).read()
for sym, why in [
    ("build_connector_meta", "the scheduler's per-step decisions"),
    ("OffloadingConnectorMetadata", "...carried to EVERY worker process"),
    ("start_load_kv", "the worker-side load entry"),
    ("wait_for_save", "the worker-side store barrier"),
    ("get_finished", "completion, aggregated ACROSS workers -- Phase 2's handshake"),
]:
    check(f"connector has {sym} ({why})", sym in conn)

print()
print("== (0l) *** UPSTREAM DOES NOT SUPPORT TRANSFER FAILURE *** ==")

# THIS IS THE FACT THAT SHAPES THE WHOLE READ PATH, and it is upstream's own
# words: "we currently do not support job failures", followed by
# `assert transfer_result.success`. So a handler reporting failure KILLS THE
# ENGINE CORE. That is the safe direction -- a dead engine beats wrong tokens --
# but it means a load has only two honest answers on a bad file: kill the
# engine, or return KV that is silently wrong. There is no third option, so the
# entire safety burden falls on lookup() being RIGHT.
#
# Pinned because an upstream bump that adds real failure handling would change
# the design space, and because somebody reading our handler might otherwise
# "soften" a refusal without realising what it costs.
wkr = open(os.path.join(REPO, "patches", "pristine", "vllm",
                        "kv_connector_offloading", "worker.py")).read()
check("the worker ASSERTS on a failed transfer",
      "assert transfer_result.success" in wkr)
check("...and says so in its own comment",
      "we currently do not support job failures" in wkr)
check("...and marks the job completed REGARDLESS of our stats fields",
      "mark_completed(job_id)" in wkr)

sch = open(os.path.join(REPO, "patches", "pristine", "vllm",
                        "kv_connector_offloading", "scheduler.py")).read()
_flat = "".join(sch.split())
check("and the scheduler has NO success/failure branch at all -- it only "
      "counts down pending_count and then calls complete_load",
      "job_status.pending_count-=count" in _flat
      and "self.manager.complete_load(" in sch)
check("...so a load that 'completed' is BELIEVED, which is why a missing file "
      "must be caught before the load is ever scheduled",
      "is_store" in sch)

print()
print(f"{PASS} passed, {len(FAIL)} failed")
if FAIL:
    for f in FAIL:
        print("  FAILED:", f)
    sys.exit(1)
