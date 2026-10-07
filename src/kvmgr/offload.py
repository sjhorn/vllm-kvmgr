"""kvmgr.offload -- let a HYBRID KV layout reach the disk tier.

THE ONE LINE THAT BLOCKS IT, measured on the box 2026-10-07 after the engine got
three steps further than the 2026-09-14 probe ever did:

    vllm/v1/kv_offload/tiering/spec.py:113
        assert len(self.gpu_block_size) == 1        # ours is (256,64,64,4,8)

`OffloadingConnector` and `TieringOffloadingSpec` both construct, and the WORKER
side initialises successfully. Only the scheduler side raises, here.

*** WHY RELAXING IT IS DEFENSIBLE. ***  `gpu_block_size` is read in exactly four
places, and three of them are in `OffloadingSpec.__init__`:

    base.py:364   the hash-divisibility loop -- PASSES for us, because
                  hash_block_size is the GCD (4) and every group divides by it
    base.py:378   the "all groups must share a block size" assert, which fires
                  ONLY when `block_size` is set in kv_connector_extra_config,
                  i.e. only when COARSENING is requested
    base.py:387   block_size_factor = offloaded // gpu, same branch
    spec_t.py:113 <- the ONLY read outside __init__, and it is the assert

So `base.py` already encodes the real rule CONDITIONALLY: uniformity is required
only in order to coarsen. The tiering spec asserts it unconditionally, which is
strictly stronger than its own base class demands.

And at `block_size_factor == 1` -- the default, and what we run, because we do
not set `block_size` -- uniformity is irrelevant to the transfer:

  * the page check is PER TENSOR PAIR, inside a loop:
        for gpu_tensor, cpu_tensor in zip(gpu_tensors, cpu_tensors):
            assert cpu_page_size == gpu_page_size * block_size_factor
    so each group's own tensors are checked against each other rather than
    against one global size (cpu/gpu_worker.py:145-154)
  * `compute_sub_block_ptrs` takes an explicit fast path at factor 1 --
    "Fast path: 1:1 mapping, no sub-block expansion needed" -- so the scalar
    never enters the pointer arithmetic at all
  * the transfer loop is ALREADY per-group, over
    `zip(group_sizes, block_indices, kv_cache_groups_data_refs)`, and its own
    comment reads "If we have multiple KV cache groups (when using HMA with
    hybrid models)"
  * the on-disk layout is ALREADY per-group: `_g<group_idx>` in the path, and
    `file_mapper` carries each group's own `block_size` and `layer_names`

UPSTREAM WENT FURTHER DOWN THIS ROAD, which is the strongest argument that the
direction is right rather than a hack against the design: `gpu_block_size` and
`block_size_factor` are GONE from upstream main's kv_offload entirely, replaced
by per-group `group_data_refs` with its own `_validate_canonical_refs`.

*** WHAT NONE OF THAT PROVES. ***  Everything above is a CODE READ. The gate is
OUTPUT EQUIVALENCE -- same prompt, greedy, same seed, offload off vs on, and the
generated tokens must match, including after a round trip through disk. "It
booted" would not tell us the per-group slicing is right, and a wrong slice
yields fluent text from corrupted KV, which is the F8_E8M0 failure class and
strictly worse than refusing to start.

OFF BY DEFAULT (`offload_allow_hybrid`). Guarded on `block_size_factor == 1`,
scoped to the single `get_manager()` call, restored in a `finally`, and every
failure falls through to upstream behaviour.
"""

from __future__ import annotations

from vllm.logger import init_logger

from . import flags

logger = init_logger("vllm." + __name__)


def install() -> bool:
    """Wrap `TieringOffloadingSpec.get_manager`. Idempotent and total.

    Returns True if the wrapper is in place (now or already), False if the
    offload machinery is not importable here -- which is the normal case in any
    process that does not build an offloading spec.
    """
    try:
        from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
    except Exception:
        # Not an error: kv_offload is absent, or not importable in this
        # process. Nothing to relax, and nothing worth logging about.
        return False

    if getattr(TieringOffloadingSpec.get_manager, "_kvmgr_hybrid", False):
        return True

    original = TieringOffloadingSpec.get_manager

    def get_manager(self):
        # *** THE DECISION IS GUARDED; THE CALL IS NOT. ***
        # The first version wrapped the whole body in `except Exception` and
        # called `original(self)` from the handler. A DOWNSTREAM failure --
        # anything the real get_manager raises -- was then caught, logged as
        # "the shim failed", and get_manager was called a SECOND time. That
        # double-calls an allocator and buries the real error under whatever
        # the retry raises (in the test: an AssertionError, because the retry
        # saw the already-restored 5-tuple). Caught by test_phase4.
        #
        # So: work out whether to relax inside the guard, then call `original`
        # exactly once, outside it, on every path.
        relax = False
        sizes: tuple = ()
        try:
            if flags.raw("offload_allow_hybrid"):
                sizes = tuple(getattr(self, "gpu_block_size", ()) or ())
                factor = getattr(self, "block_size_factor", 1)
                if len(set(sizes)) > 1:
                    if factor != 1:
                        # Coarsening genuinely DOES need one block size -- that
                        # is base.py's own rule and we are not overriding it.
                        logger.warning(
                            "kvmgr: NOT relaxing the hybrid-offload assert, "
                            "because block_size_factor=%d (you set `block_size` "
                            "in kv_connector_extra_config). Coarsening several "
                            "GPU blocks into one offloaded block really does "
                            "require a single block size. Drop `block_size` to "
                            "use the disk tier on a hybrid layout.",
                            factor,
                        )
                    else:
                        logger.warning(
                            "kvmgr: relaxing TieringOffloadingSpec's "
                            "single-block-size assert for a HYBRID layout -- "
                            "block sizes %s, block_size_factor=1, so the "
                            "transfer is per-tensor and the scalar never "
                            "enters the pointer arithmetic. OUTPUT EQUIVALENCE "
                            "IS THE GATE, not a successful boot.",
                            sizes,
                        )
                        relax = True
        except Exception:
            logger.exception(
                "kvmgr: hybrid-offload decision failed; leaving upstream "
                "behaviour unchanged"
            )
            relax = False

        if not relax:
            return original(self)

        self.gpu_block_size = (sizes[0],)
        try:
            return original(self)
        finally:
            self.gpu_block_size = sizes

    get_manager._kvmgr_hybrid = True  # type: ignore[attr-defined]
    TieringOffloadingSpec.get_manager = get_manager  # type: ignore[method-assign]
    logger.info(
        "kvmgr: hybrid-offload shim installed (opt-in, off unless "
        "offload_allow_hybrid is set)"
    )
    return True
