"""Pack a rank's Engram IDs into a fixed communication slot, entirely on GPU."""

import torch
import triton
import triton.language as tl


@triton.jit
def _pack_ids(src, dst, N: tl.constexpr, SLOT: tl.constexpr,
              RANK: tl.constexpr, WORLD: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    local = x - RANK * SLOT
    owns = (local >= 0) & (local < SLOT)
    value = tl.load(src + local, mask=owns & (local < N), other=-1)
    # Only the owner writes its slot. -1 marks unused rows and is not looked up.
    tl.store(dst + x, tl.where(owns, value, 0), mask=x < WORLD * SLOT)


def pack_engram_ids(ids: torch.Tensor, output: torch.Tensor, *, rank: int) -> None:
    slot = output.numel() // output.shape[0]
    if ids.numel() > slot:
        raise ValueError("Engram IDs exceed the configured decode communication slot")
    _pack_ids[(triton.cdiv(output.numel(), 256),)](
        ids, output, ids.numel(), slot, rank, output.shape[0], 256
    )
