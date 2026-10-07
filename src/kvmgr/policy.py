"""kvmgr.policy -- the cap arithmetic, as pure functions.

Lifted out of the manager on purpose.  This is the arithmetic that decides how
much of a conversation survives, it is wrong in ways no smoke test can see, and
both sides of the decision look like a working engine on screen.  A pure
function with a test per case is the only way to pin it -- the same reason
`specVerdict` and `shareAxis` are lifted in the dashboard.

THE NUMBERS THIS REPRODUCES, measured on the live engine 2026-10-07:

    num_gpu_blocks = 24,288      five groups, block sizes 256/64/64/4/8
    fraction 0.75, mode "equal"  -> 3,643 blocks EVERY group
                                 -> 932,608 tokens in group 0
                                 ->  14,572 tokens in group 3   (a 64x spread)

`equal` is today's behaviour, byte for byte, and is the default.
"""

from __future__ import annotations

from vllm.utils.math_utils import cdiv

# Deliberately NOT a copy of patch6's constant: this is the same formula
# (2 * max_model_len / block_size) restated where the arithmetic lives, and the
# parity test asserts the two agree.
def legacy_cap(max_model_len: int | None, block_size: int) -> int | None:
    """Upstream-of-this-fork's bound: two model-lengths' worth of blocks.

    NOT dead weight.  It is what BINDS under `by_need` -- at max_model_len 1M
    and block_size 256 it is 8,192 blocks = 2,097,152 tokens, which is less than
    a by-need share of fraction 0.75 would give group 0.  So removing the
    /num_groups division does not buy the ~4.2x it looks like it should; it buys
    2.25x, and this is the term that says so.
    """
    if max_model_len is None:
        return None
    return 2 * cdiv(max(1, max_model_len), block_size)


def pool_share_equal(num_gpu_blocks: int, fraction: float, num_groups: int) -> int:
    """Today's rule: an equal BLOCK count per group.

    The flaw is that a block is not a unit of context.  The five groups' blocks
    hold 256/64/64/4/8 tokens, so an equal block count protects 932,608 tokens
    in group 0 and 14,572 in group 3 -- group 0 binds, and the other four hold
    blocks they cannot fill.
    """
    return int(num_gpu_blocks * fraction / max(1, num_groups))


def window_need_blocks(sliding_window: int | None, block_size: int) -> int | None:
    """How many blocks THIS group must hold for a hit at a given boundary.

    `None` means unbounded -- a full-attention group needs the whole prefix, so
    its need grows with the conversation.  A windowed group does NOT: it only
    ever matches the contiguous run ending at the boundary, so it needs its
    window and nothing more, however long the prompt gets.

    The formula is patch6's own (`_head_pin_candidates`, MNT-9), restated here
    because this is where the budget is divided.  The parity test asserts the
    two agree; they must not drift.
    """
    if sliding_window is None:
        return None
    return cdiv(max(1, sliding_window - 1), block_size) + 1


def pool_share_by_need(
    num_gpu_blocks: int,
    fraction: float,
    block_sizes: tuple[int, ...],
    group_index: int,
    sliding_windows: tuple[int | None, ...] = (),
) -> int:
    """Give each group what it can actually USE, and the rest to full attention.

    *** THE OBVIOUS READING OF "BY NEED" IS A 21x REGRESSION. ***  Read as "give
    every group the same token extent", the budget splits in inverse proportion
    to block size -- and because group 3's blocks hold 4 tokens it needs 64x as
    many of them as group 0 for the same extent.  Measured: that lands every
    group on 44,412 tokens, against group 0's 932,608 TODAY.  It would look like
    a principled fix and it would cost 95% of the retention.

    The error is treating a windowed group's need as proportional to context.
    It is not: a 128-token window over 4-token blocks is 33 blocks, forever.  So
    the four windowed groups want 56 blocks BETWEEN THEM, and full attention
    takes the remainder:

        fraction 0.75, pool 24,288  ->  18,216 blocks to divide
        groups 1-4 (windowed)       ->       3 + 3 + 33 + 17 = 56 blocks
        group 0 (full MLA)          ->  18,160 blocks, clamped by the legacy
                                        cap to 8,192 = 2,097,152 tokens

    which is 2.25x today, and 2x the 1M context ceiling.
    """
    if not block_sizes:
        return 0
    total_blocks = int(num_gpu_blocks * fraction)
    if not sliding_windows:
        sliding_windows = (None,) * len(block_sizes)

    needs = [
        window_need_blocks(w, bs)
        for w, bs in zip(sliding_windows, block_sizes)
    ]
    # A group whose need is bounded gets exactly it -- never more, because a
    # block past the window is one no lookup can use, and never less, because
    # one block short of the window is a zero hit for that group and the
    # coordinator takes the minimum across groups.
    if needs[group_index] is not None:
        return min(needs[group_index], total_blocks)

    bounded = sum(n for n in needs if n is not None)
    unbounded = sum(1 for n in needs if n is None)
    remainder = max(0, total_blocks - bounded)
    return remainder // max(1, unbounded)


def protected_cap(
    *,
    num_gpu_blocks: int,
    fraction: float,
    num_groups: int,
    block_size: int,
    max_model_len: int | None,
    mode: str = "equal",
    block_sizes: tuple[int, ...] = (),
    group_index: int = 0,
    sliding_windows: tuple[int | None, ...] = (),
) -> int | None:
    """The cap in BLOCKS for one group, or None for "no cap".

    A negative fraction means "disable our bound and keep only the legacy one",
    which is patch6's own contract and is preserved exactly.
    """
    legacy = legacy_cap(max_model_len, block_size)
    if fraction < 0:
        return legacy
    if mode == "by_need" and block_sizes:
        share = pool_share_by_need(
            num_gpu_blocks, fraction, block_sizes, group_index, sliding_windows
        )
    else:
        share = pool_share_equal(num_gpu_blocks, fraction, num_groups)
    return share if legacy is None else min(legacy, share)


def cap_tokens(cap_blocks: int | None, block_size: int) -> int | None:
    """The cap as CONTEXT EXTENT, which is the figure an operator can act on.

    A cap in blocks is not comparable across groups and is the reason the
    original equal split looked reasonable for months."""
    if cap_blocks is None:
        return None
    return cap_blocks * block_size
