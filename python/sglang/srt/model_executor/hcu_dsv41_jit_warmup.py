"""Keep the known HCU V4.1 cold compiles before KV sizing and server Ready.

This primes compile entry points, not a distributed model forward. It neither
allocates serving KV slots nor inserts synthetic prefixes into HiCache/L3.
Other dtypes, backends and business shapes can still need their own warmup.
"""

from __future__ import annotations

import logging
import time

import torch

logger = logging.getLogger(__name__)


def _timed(label, callback):
    logger.info("HCU V4.1 startup JIT warmup begin: %s", label)
    start = time.perf_counter()
    callback()
    # Compilation/loading is synchronous; kernel launches also must finish
    # before the caller samples available memory for KV sizing.
    torch.cuda.synchronize()
    logger.info(
        "HCU V4.1 startup JIT warmup done: %s (%.3fs)",
        label,
        time.perf_counter() - start,
    )


def _warm_vision_attention(vision):
    from sglang.srt.layers.attention.vision import (
        VisionAttention,
        VisionAttentionMetadata,
    )

    seen = set()
    for attn in vision.modules():
        if not isinstance(attn, VisionAttention):
            continue
        # Do not compile a fallback which the loaded model does not use.
        if attn.qkv_backend_name != "triton_attn":
            logger.info(
                "HCU V4.1 startup JIT warmup: vision backend %s is not Triton",
                attn.qkv_backend_name,
            )
            continue
        weight = attn.qkv_proj.weight
        heads = attn.num_attention_heads_per_partition
        kv_heads = attn.num_attention_kv_heads_per_partition
        dim = attn.head_size
        signature = (
            weight.device,
            weight.dtype,
            heads,
            kv_heads,
            dim,
            attn.pass_strided_qkv,
            attn.softmax_scale,
        )
        if signature in seen:
            continue
        seen.add(signature)
        # Match the loaded tower's per-rank dimensions, dtype and QKV strides.
        # V4.1's custom RoPE opts out of strided QKV; all three are contiguous.
        for batch, length in ((1, 197), (8, 512)):
            qkv = torch.ones(
                (batch * length, heads + 2 * kv_heads, dim),
                device=weight.device,
                dtype=weight.dtype,
            )
            q, k, v = qkv.split((heads, kv_heads, kv_heads), dim=1)
            if not attn.pass_strided_qkv:
                q, k, v = (x.contiguous() for x in (q, k, v))
            metadata = VisionAttentionMetadata(
                cu_seqlens=(
                    torch.arange(batch + 1, device=weight.device, dtype=torch.int32)
                    * length
                ),
                seq_lens=torch.full(
                    (batch,), length, device=weight.device, dtype=torch.int32
                ),
                max_seqlen=length,
            )
            _timed(
                f"vision triton dtype={weight.dtype} heads={heads}/{kv_heads} "
                f"dim={dim} batch={batch} length={length} strides={q.stride()}",
                lambda: attn.qkv_backend(
                    q,
                    k,
                    v,
                    metadata.cu_seqlens,
                    batch,
                    length,
                    softmax_scale=attn.softmax_scale,
                    forward_metadata=metadata,
                ),
            )


@torch.inference_mode()
def prewarm_hcu_dsv41_kernels(model):
    """Called by the model's post-load hook on each local scheduler rank.

    The enclosing loader synchronizes and empties the allocator cache after this
    returns, before measuring KV capacity. Errors deliberately propagate: Ready
    must not hide a failed mandatory warmup. No model weights are read or changed.
    """
    from sglang.kernels.ops.attention.dsv4.attn import _jit_fused_store_module
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import select_dsv4_kv_layout
    from sglang.srt.runtime_context import get_exec, get_memory, get_schedule

    language_model = getattr(model, "model", None)
    vision = getattr(model, "vision", None)
    if language_model is not None:
        weight = language_model.embed_tokens.weight
    elif vision is not None:
        weight = vision.patch_embed.proj.weight
    else:
        return

    with torch.cuda.device(weight.device):
        if language_model is not None:
            page_size = get_schedule().page_size
            if not isinstance(page_size, int) or page_size <= 0:
                raise ValueError("Startup JIT warmup requires a resolved page_size")
            layout, _ = select_dsv4_kv_layout()
            if layout is KVLayout.V4 and get_exec().kernel.dsv4_attn_backend != "trtllm":
                # The observed HCU V4 HIP writers use 128/256-token physical
                # pages. These are page sizes, not input lengths or batch sizes.
                pages = sorted({128, 256, page_size})
                for page in pages:
                    _timed(
                        f"flashmla store dtype={weight.dtype} index=int64 page={page} V4",
                        lambda p=page: _jit_fused_store_module(
                            "flashmla", weight.dtype, torch.int64, p, KVLayout.V4
                        ),
                    )
            memory = get_memory()
            if not memory.disable_radix_cache and (
                memory.enable_unified_cache_external_linker
                or memory.enable_hierarchical_cache
            ):
                from sglang.srt.mem_cache.cpp_utils.native_hash import get_native_hash

                _timed(
                    f"hicache_hash_cpp two complete pages ({page_size} tokens/page)",
                    lambda: get_native_hash(
                        list(range(2 * page_size)), None, page_size=page_size
                    ),
                )
        if vision is not None:
            _warm_vision_attention(vision)
