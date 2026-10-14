"""test_phase0_disk.py -- PIN THE CONTRACT our own disk OffloadingSpec will
implement, before a line of it is written.

    python3.12 tests/test_phase0_disk.py

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
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the PACKAGE root

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
fmap = sys.modules["vllm.v1.kv_offload.file_mapper"]
fact = sys.modules["vllm.v1.kv_offload.factory"]

LIVE_BLOCK_SIZES = harness.LIVE_BLOCK_SIZES          # (256, 64, 64, 4, 8)
WORLD_SIZE = 2                                        # one rank per node
WHOLE_PAGE = 2_103_552                                # measured on disk

print("== (0a) the abstract surface (vLLM 0.31) ==")

# If an upstream bump adds an abstract method, our spec stops being
# instantiable -- and it would fail at BOOT, in the engine core.
eq("OffloadingManager has exactly four abstract methods (on_new_request is new)",
   sorted(base.OffloadingManager.__abstractmethods__),
   ["lookup", "on_new_request", "prepare_load", "prepare_store"])
eq("OffloadingSpec has exactly two (get_worker replaced get_handlers)",
   sorted(base.OffloadingSpec.__abstractmethods__),
   ["get_manager", "get_worker"])
eq("OffloadingWorker has exactly four",
   sorted(base.OffloadingWorker.__abstractmethods__),
   ["get_finished", "submit_load", "submit_store", "wait"])
check("touch/complete_*/take_events/reset_cache are CONCRETE, so we inherit them",
      all(m not in base.OffloadingManager.__abstractmethods__
          for m in ("touch", "complete_load", "complete_store",
                    "take_events", "reset_cache", "shutdown")))
eq("lookup answers a LookupResult, not a bool",
   sorted(m.name for m in base.LookupResult), ["HIT", "HIT_PENDING", "MISS", "RETRY"])

print()
print("== (0b) no medium routing: one worker per spec, direction explicit ==")
check("LoadStoreSpec has no medium()", not hasattr(base.LoadStoreSpec, "medium"))
eq("OffloadingEvent.medium is an enum", sorted(m.name for m in base.Medium),
   ["CPU", "STORAGE"])
OUR_MEDIUM = base.Medium.STORAGE


class DiskSpecProbe(base.LoadStoreSpec):
    pass


print()
print("== (0d) TransferResult: no transfer_type on 0.31 ==")
res = base.TransferResult(job_id=1, success=True, transfer_size=123,
                          transfer_time=0.5)
eq("TransferResult carries size", res.transfer_size, 123)
check("...and no transfer_type (load/store is decided by _load_jobs)",
      not hasattr(res, "transfer_type"))
check("size and time default to None",
      base.TransferResult(job_id=2, success=True).transfer_size is None)
check("block ids are int32 now", base.GPULoadStoreSpec(
    [1], group_sizes=[1], block_indices=[0]).block_ids.dtype == "int32")

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
tier_src = open(os.path.join(REPO, "tests", "upstream", "vllm", "v1",
                             "kv_offload", "tiering", "spec.py")).read()
base_src = inspect.getsource(base.OffloadingSpec.__init__)
# vLLM 0.31: neither spec asserts a single block size any more, and the
# divisibility check moved out of the spec into build_offloading_config.
check("0.31: TieringOffloadingSpec no longer asserts a single block size",
      "len(self.gpu_block_size) == 1" not in tier_src)
check("*** and the BASE does NOT ***",
      "len(self.gpu_block_size) == 1" not in base_src)
_occ = open(os.path.join(REPO, "tests", "upstream", "vllm",
                         "kv_connector_offloading", "config.py")).read()
check("divisibility by tokens_per_hash is asserted in build_offloading_config",
      "group.tokens_per_block % tokens_per_hash == 0" in _occ)
HASH_BLOCK_SIZE = 4   # the GCD of our five, resolved by patch8
check("...which all five of our block sizes satisfy",
      all(b % HASH_BLOCK_SIZE == 0 for b in LIVE_BLOCK_SIZES),
      f"{LIVE_BLOCK_SIZES} % {HASH_BLOCK_SIZE}")

# And block_size_factor must stay 1: setting it needs ONE block size, which we
# do not have. Factor 1 is also the only case the hybrid relaxation was ever
# safe for, so this is the same constraint seen from the other side.
check("blocks_per_chunk defaults to 1", "blocks_per_chunk = 1" in _occ)
check("...and setting it via block_size REQUIRES a single block size, so we must not",
      "len(unique_tokens_per_block) == 1" in _occ)

print()
print("== (0h) the page arithmetic -- the number Phase 1's gate checks ==")

# Upstream sizes its region page*world_size and writes the WHOLE page to one
# file, which is why ours are half empty. Our handler writes only this rank's
# slice, so our files must be HALF the size.
eq("today's file size, measured on disk", WHOLE_PAGE, 2_103_552)
eq("*** per-worker slice, which is what OUR files must be ***",
   WHOLE_PAGE // WORLD_SIZE, 1_051_776)
_sor = open(os.path.join(REPO, "tests", "upstream", "vllm", "v1", "kv_offload",
                         "cpu", "shared_offload_region.py")).read()
check("0.31 tiering: ONE /dev/shm region per host, rows of the WHOLE page",
      '/dev/shm/vllm_offload_{engine_id}.mmap' in _sor
      and "self._row_stride = kv_bytes_per_chunk" in _sor)
check("...each worker writes at rank * cpu_page_size in it",
      "self._worker_offset = rank * cpu_page_size" in _sor)
check("...and the worker's rank is device_index % world_size, so one GPU per "
      "node makes BOTH nodes rank 0",
      "current_device_index() % world_size" in tier_src)

print()
print("== (0i) rank is in the PATH, deliberately not in the fingerprint ==")

# This is what makes per-rank files possible at all: _r0 and _r1 are siblings
# under ONE fingerprint, each rank restoring its own half.
fm_src = inspect.getsource(fmap.FileMapper)
check("get_file_name appends _r<rank>", "_r" in fm_src and "rank" in fm_src)
check("...and the config file is shared BECAUSE rank is outside the hash",
      "rank lives outside the hash" in fm_src)
for field in ("model_name", "tokens_per_hash", "tp_size", "dtype", "kv_cache_groups"):
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
          REPO, "tests", "upstream", "vllm", "v1", "kv_offload",
          "factory.py")).read())

print()
print("== (0k) what the CONNECTOR already gives us -- the control channel ==")

conn = open(os.path.join(REPO, "tests", "upstream", "vllm",
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
wkr = open(os.path.join(REPO, "tests", "upstream", "vllm",
                        "kv_connector_offloading", "worker.py")).read()
check("the worker ASSERTS on a failed transfer",
      "assert transfer_result.success" in wkr)
check("...and says so in its own comment",
      "we currently do not support job failures" in wkr)
check("...and marks the job completed REGARDLESS of our stats fields",
      "mark_completed(job_id)" in wkr)

sch = open(os.path.join(REPO, "tests", "upstream", "vllm",
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
