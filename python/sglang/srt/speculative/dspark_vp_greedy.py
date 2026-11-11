"""Experimental DSPARK VP8 reductions: communicate one winner per vocab shard.

Enable with SGLANG_DSPARK_VP_GREEDY_TOP1=1, DraftVP8/TargetVP8, static verification
and DP-local attention. Requests must use unmodified greedy sampling; logprobs,
grammars and penalties are rejected before scheduling. The default is disabled.
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
import triton
import triton.language as tl


@lru_cache(maxsize=1)
def enabled() -> bool:
    return os.environ.get("SGLANG_DSPARK_VP_GREEDY_TOP1", "0") == "1"


def markov_has_full_vocab(head) -> bool:
    # The normal DP-local head still configures a TP1 LocalVocabGather.
    # Its metadata is present, but its source weight covers the entire vocab.
    shard = getattr(head, "_tp_shard", None)
    return shard is None or (
        shard.tp_size == 1
        and shard.org_vocab_start == 0
        and shard.org_vocab_end == head.vocab_size
    )


def validate_request(params, request) -> None:
    """Reject unsupported requests at admission, before any DP collective."""
    if not enabled():
        return
    if params.top_k != 1:
        raise ValueError(
            "DSPARK VP top-1 mode requires greedy sampling (temperature=0)."
        )
    for name, default in (
        ("frequency_penalty", 0),
        ("presence_penalty", 0),
        ("repetition_penalty", 1),
        ("min_new_tokens", 0),
    ):
        if getattr(params, name, default) != default:
            raise ValueError("DSPARK VP top-1 mode does not support " + name)
    for name in (
        "logit_bias",
        "json_schema",
        "regex",
        "ebnf",
        "structural_tag",
        "custom_params",
        "watermark",
    ):
        if getattr(params, name, None):
            raise ValueError("DSPARK VP top-1 mode does not support " + name)
    for name in (
        "return_logprob",
        "return_sampling_mask",
        "return_flat_raw_top_logprobs",
        "custom_logit_processor",
        "token_ids_logprob",
    ):
        if getattr(request, name, None):
            raise ValueError("DSPARK VP top-1 mode does not support " + name)
    if (getattr(params, "beam_width", None) or 1) != 1:
        raise ValueError("DSPARK VP top-1 mode does not support beam search.")


@triton.jit
def _finish_local(
    partial, output, parts: tl.constexpr, start: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    p = tl.arange(0, BLOCK)
    values = tl.load(partial + row * parts * 2 + p * 2, p < parts, other=-float("inf"))
    bits = tl.load(partial + row * parts * 2 + p * 2 + 1, p < parts, other=0.0).to(
        tl.int32, bitcast=True
    )
    ids = bits & 0x7FFFFFFF
    valid = (p < parts) & (ids != 0x7FFFFFFF)
    nan = valid & (values != values)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, values, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (values == maximum))
    token = tl.min(tl.where(wins, ids, 0x7FFFFFFF), 0)
    score = tl.where(has_nan, float("nan"), maximum)
    tl.store(output + row * 2, score)
    tl.store(output + row * 2 + 1, (token - start).to(tl.float32))


@triton.jit
def _merge(
    gathered,
    scores_out,
    ids_out,
    rows: tl.constexpr,
    world: tl.constexpr,
    width: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    rank = tl.arange(0, BLOCK)
    values = tl.load(
        gathered + (rank * rows + row) * 2, rank < world, other=-float("inf")
    )
    local = tl.load(gathered + (rank * rows + row) * 2 + 1, rank < world, other=0.0).to(
        tl.int32
    )
    ids = rank * width + local
    valid = (rank < world) & (local >= 0) & (local < width)
    nan = valid & (values != values)
    has_nan = tl.sum(nan.to(tl.int32), 0) > 0
    maximum = tl.max(tl.where(valid & ~nan, values, -float("inf")), 0)
    wins = tl.where(has_nan, nan, valid & (values == maximum))
    token = tl.min(tl.where(wins, ids, 0x7FFFFFFF), 0)
    tl.store(scores_out + row, tl.where(has_nan, float("nan"), maximum))
    tl.store(ids_out + row, token.to(tl.int64))


def local_candidates(base_logits, bias, *, vocab_start, out):
    # Reuse this branch's exact FP32 BuildStepLocal reduction, including its
    # NaN and lowest-token-ID tie handling, then reduce to one pair per shard.
    from sglang.kernels.ops.speculative.dspark.sharded_greedy import (
        _sharded_greedy_partial_kernel,
    )

    rows, width = base_logits.shape
    assert width > 0 and out.shape == (rows, 2) and out.dtype == torch.float32
    assert base_logits.stride(1) == 1
    if bias is not None:
        assert bias.shape == base_logits.shape and bias.stride(1) == 1
    if not rows:
        return
    block = 4096
    parts = triton.cdiv(width, block)
    partial = torch.empty(
        (rows, parts, 2), device=base_logits.device, dtype=torch.float32
    )
    _sharded_greedy_partial_kernel[(rows, parts)](
        bias if bias is not None else base_logits,
        base_logits,
        partial,
        bias.stride(0) if bias is not None else 0,
        base_logits.stride(0),
        width,
        vocab_start,
        parts,
        block,
        HAS_BIAS=bias is not None,
        num_warps=4,
    )
    _finish_local[(rows,)](
        partial, out, parts, vocab_start, triton.next_power_of_2(parts), num_warps=4
    )


def merge_candidates(gathered, scores, tokens, *, vp_size, shard_size):
    rows = scores.numel()
    assert tokens.numel() == rows and gathered.shape == (vp_size * rows, 2)
    if not rows:
        return
    _merge[(rows,)](
        gathered,
        scores,
        tokens,
        rows,
        vp_size,
        shard_size,
        triton.next_power_of_2(vp_size),
        num_warps=4,
    )


def int8_bias_slice(latent, weight, scales, *, output_dtype):
    """Keep SERVICE_OPT's activation quantizer and ordered scale arithmetic."""
    from sglang.kernels.ops.gemm.dsv41_int8 import load_native
    from sglang.kernels.ops.quantization import int8_kernel as quant

    x = latent.contiguous()
    m, k = x.shape
    n = weight.shape[0]
    q = torch.empty((m, 1, k), device=x.device, dtype=torch.int8)
    xs = torch.empty((m, 1, 1), device=x.device, dtype=torch.float32)
    raw = torch.empty((m, 1, n), device=x.device, dtype=torch.int32)
    y = torch.empty((m, 1, n), device=x.device, dtype=output_dtype)
    block = triton.next_power_of_2(k)
    quant._per_token_quant_int8[(m,)](
        x,
        q,
        xs,
        None,
        stride_x=k,
        stride_xq=k,
        N=k,
        CAL_SUM=False,
        BLOCK=block,
        IS_HIP=quant._is_hip,
        num_warps=min(max(block // 256, 1), 8),
        num_stages=1,
    )
    ops = load_native()
    ops.gemm_out(q, weight, raw, 1, 0, False, 0)
    ops.scale_out(raw, xs, scales, y, 1, False)
    return y[:, 0]


def draft_block(model, raw_hidden, anchor_tokens, *, gamma, padded_rows=None):
    """Apply Markov bias on the shard before selecting every greedy token."""
    vp = model.draft_lm_head_vp
    head = model.markov_head
    if not markov_has_full_vocab(head):
        raise ValueError("Greedy VP requires the replicated Markov head.")
    x = (
        model.stages[-1].norm(model.collapse_hc_head(raw_hidden))
        if raw_hidden.shape[0]
        else vp.local_hidden[:0]
    )
    partial, rows, padded = vp.project_sharded(x, padded_rows=padded_rows)
    bs = anchor_tokens.numel()
    assert rows == bs * gamma and padded % gamma == 0
    if not padded:
        return anchor_tokens.new_empty((0, gamma))
    padded_bs = padded // gamma
    local_anchor = vp.greedy_anchor_local[:padded_bs]
    local_anchor[:bs].copy_(anchor_tokens)
    local_anchor[bs:].zero_()
    global_bs = padded_bs * vp.vp_size
    prev = vp.greedy_anchor_global[:global_bs]
    vp._all_gather(prev, local_anchor)
    base = partial.view(global_bs, gamma, vp.shard_size)
    result = []
    start = vp.vp_rank * vp.shard_size
    for step in range(gamma):
        bias = head.step_bias_vocab_slice(prev, start, start + vp.shard_size)
        _, prev = vp.top1_from_shard(base[:, step], bias=bias, all_rows=True)
        # The next reduction reuses its token buffer. Keep this step's tokens.
        result.append(
            prev[vp.vp_rank * padded_bs : vp.vp_rank * padded_bs + bs].clone()
        )
    return torch.stack(result, dim=1)
