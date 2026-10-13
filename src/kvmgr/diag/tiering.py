"""kvmgr.diag.tiering -- let a HYBRID KV layout reach upstream's tiering tier.

*** A DIAGNOSTIC NOW, OFF BY DEFAULT (``diag_tiering_shims``), moved from
kvmgr.offload on 2026-10-13. *** Every shim here patches upstream's
``TieringOffloadingSpec`` and its fs tier; kvmgr's own disk tier
(``DSparkDiskOffloadingSpec``) uses none of them, because upstream's tiering
cannot capture rank 1's half on a one-rank-per-node cluster. They are kept for
anyone running that spec, and so the offload_* knobs still mean something
when it is installed. The original header follows.

Original title: let a HYBRID KV layout reach the disk tier.

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

from .. import flags
from .. import stats

logger = init_logger("vllm." + __name__)


_real_odirect: int | None = None


def install_io() -> bool:
    """Drop `O_DIRECT` from the fs tier's block writes.

    WHY NOTHING REACHED DISK, measured 2026-10-07: `tiering/fs/io.py` opens
    every block with O_DIRECT, which requires the length and offset to be
    multiples of the device logical block size. Ours are not --

        per-block CPU page : 2,103,552 bytes
        mod  512 : 256      (nvme0n1 LOG-SEC is 512)
        mod 4096 : 2,304

    -- so every write was rejected before a byte landed:

        ERROR [thread_pool.py:150] FileSystemTierManagerPython:
              job 208 block I/O failed: [Errno 22] Invalid argument

    13,519 of them in one boot, against 1 file on disk. The 17,222 directories
    exist because `mkdir` succeeds; only the writes fail.

    *** WHY DROP IT RATHER THAN PAD THE WRITES. *** Padding to a 512 boundary
    keeps O_DIRECT's benefits, and it means the writer must record the TRUE
    length and the reader must honour it. Get that wrong and the reader returns
    padding as KV -- it CORRUPTS rather than failing. Dropping the flag cannot:
    the write either succeeds whole or raises. And it is not an invention --
    `io.py` itself resolves O_DIRECT to 0 wherever the platform lacks it
    (`getattr(os, "O_DIRECT", 0)`, for macOS), so the no-O_DIRECT path is a
    configuration upstream already supports.

    *** THE COST, WHICH IS REAL ON THIS BOX. *** Without O_DIRECT the writes go
    through the page cache, and host RAM is the scarce resource here
    (MemAvailable ~1.7 GB with the 1.07 GB /dev/shm tier live). Page cache is
    reclaimable, so the kernel gives it back under pressure rather than OOMing
    -- but it can drive swap, and swap is earlyoom's real gate on these nodes.
    WATCH MemAvailable AND SwapFree while this is on. The proper fix if it bites
    is posix_fadvise(POSIX_FADV_DONTNEED) after each write, which is a change to
    io.py's own logic rather than one integer, and is deliberately not done here.

    Hot: the flag is re-read per block I/O, so it can be flipped on a running
    engine like every other knob.
    """
    global _real_odirect
    try:
        from vllm.v1.kv_offload.tiering.fs import io as fsio
    except Exception:
        return False

    if getattr(fsio, "_kvmgr_odirect_wrapped", False):
        return True

    if _real_odirect is None:
        _real_odirect = int(getattr(fsio, "O_DIRECT", 0) or 0)

    def _sync() -> None:
        # O_DIRECT is a module-level int that io.py ORs into every os.open, so
        # setting it here is exactly "use O_DIRECT or do not". Both I/O threads
        # would write the same value, so the race is benign.
        want = _real_odirect if flags.raw("offload_odirect") else 0
        if getattr(fsio, "O_DIRECT", None) != want:
            fsio.O_DIRECT = want

    def _wrap(fn):
        def inner(*args, **kwargs):
            try:
                _sync()
            except Exception:
                pass  # never fail a block transfer over a flag read
            return fn(*args, **kwargs)

        inner.__name__ = getattr(fn, "__name__", "wrapped")
        return inner

    for name in ("store_block", "load_block"):
        fn = getattr(fsio, name, None)
        if fn is not None:
            setattr(fsio, name, _wrap(fn))

    fsio._kvmgr_odirect_wrapped = True  # type: ignore[attr-defined]
    _sync()
    logger.info(
        "kvmgr: fs-tier O_DIRECT shim installed (platform O_DIRECT=%s; "
        "offload_odirect=%s -> effective O_DIRECT=%s). Dropping it is what lets "
        "a %d-byte block write at all; watch MemAvailable and SwapFree, because "
        "the writes then go through the page cache.",
        _real_odirect, flags.raw("offload_odirect"),
        getattr(fsio, "O_DIRECT", "?"), 2103552,
    )
    return True


def install_verify() -> bool:
    """Round-trip every block written to disk and compare it byte for byte.

    *** WHY THIS AND NOT AN OUTPUT-EQUIVALENCE TEST. *** The output gate was
    worthless: measured 2026-10-07, four runs of the same prompt on ONE
    unchanged configuration gave two different answers (eq1 at 156 and 157
    tokens, differing even in capitalisation), because
    `SpeculativeConfig(num_spec_tokens=5)` plus `--async-scheduling` and chunked
    prefill move the float reduction order and flip a near-tie argmax. A SHA of
    this engine's output fingerprints nothing, so "the hashes matched" was luck.

    A byte compare of the data path has no such problem: it is exact, it needs
    no determinism, and it tests precisely what the O_DIRECT change touched.

    *** WHAT IT PROVES: *** that what `load_block` reads back equals what
    `store_block` was handed -- so the write, the file, and the read agree. That
    is the disk round trip, which is the thing we changed.

    *** WHAT IT CANNOT PROVE, and this must not be glossed: *** that the bytes
    handed in were the RIGHT bytes. It compares against the same slice the
    writer received, so a wrong offset or a wrong length upstream of here would
    round-trip perfectly and still be wrong. The open question it cannot touch:
    every file on disk is 2,103,552 bytes for ALL FIVE groups, and five of those
    exceed the CPU page computed for one block index -- either the page
    arithmetic is off by the world_size factor or the write length is uniform
    where it should be per-group. Answering that needs the GPU-side bytes, which
    only the worker can read.

    Reassurance that is NOT proof: the same block hash stored under different
    groups has different content on disk (distinct md5s), so per-group selection
    is picking different data rather than duplicating one buffer.

    DOUBLES THE I/O of every store, so it is off by default and is a test mode,
    not something to leave on.
    """
    try:
        from vllm.v1.kv_offload.tiering.fs import io as fsio
    except Exception:
        return False

    if getattr(fsio, "_kvmgr_verify_wrapped", False):
        return True

    orig_store = fsio.store_block

    def store_block(dest_path, buffer, offset, block_size, *a, **kw):
        result = orig_store(dest_path, buffer, offset, block_size, *a, **kw)
        try:
            if not flags.raw("offload_verify"):
                return result
            back = bytearray(block_size)
            # Through load_block, not a raw read: the READER is part of what we
            # are testing, and it carries its own O_DIRECT handling.
            fsio.load_block(dest_path, memoryview(back), 0, block_size, *a, **kw)
            src = bytes(buffer.cast("B")[offset : offset + block_size])
            if bytes(back) == src:
                stats.record_verify(True)
            else:
                stats.record_verify(False)
                first = next(
                    (i for i, (x, y) in enumerate(zip(back, src)) if x != y), -1
                )
                logger.error(
                    "kvmgr: *** DISK ROUND TRIP MISMATCH *** %s: %d bytes, "
                    "first differing byte at %d. The KV written to disk is not "
                    "the KV that was handed to the writer.",
                    dest_path, block_size, first,
                )
        except Exception:
            # Verification must never break a transfer it is only observing.
            logger.exception("kvmgr: round-trip verify failed to run")
        return result

    fsio.store_block = store_block
    fsio._kvmgr_verify_wrapped = True  # type: ignore[attr-defined]
    logger.info(
        "kvmgr: disk round-trip verifier installed (off unless offload_verify "
        "is set; it doubles store I/O)"
    )
    return True


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
    # The fs tier's O_DIRECT is a separate defect with a separate knob, so the
    # two can be A/B'd independently.
    install_io()
    install_verify()
    return True
