"""Compare the effective KV sequence and real Attention before/after packing."""

import itertools
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU required")


def make_case(lengths, extends, cp, rank, ratio):
    from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
        SparsePrefillChunkCache,
    )

    starts = [0] + list(itertools.accumulate(extends))
    real = [0] * len(lengths)
    positions = []
    for global_row in range(rank, sum(extends), cp):
        request = next(
            i for i in range(len(lengths)) if starts[i] <= global_row < starts[i + 1]
        )
        real[request] += 1
        positions.append(
            lengths[request] - extends[request] + global_row - starts[request]
        )
    physical = list(real)
    padded = (sum(extends) + cp - 1) // cp - len(positions)
    physical[-1] += padded
    positions += [-1] * padded
    q = len(positions)
    page = 256 // ratio
    compressed_lengths = [max(n // ratio, 1) for n in lengths]
    page_counts = [(n + page - 1) // page for n in compressed_lengths]
    table = torch.full(
        (len(lengths), max(page_counts)), -1, dtype=torch.int32, device="cuda"
    )
    cursor = 1
    for request, count in enumerate(page_counts):
        table[request, :count] = torch.arange(
            cursor, cursor + count, dtype=torch.int32, device="cuda"
        ).flip(0)
        cursor += count
    pages = torch.repeat_interleave(table, torch.tensor(physical, device="cuda"), dim=0)
    window = [min(n, e + 127) for n, e in zip(lengths, extends)]
    offsets = [0] + list(itertools.accumulate(window))
    tensor = lambda value: torch.tensor(value, dtype=torch.int32, device="cuda")
    cache = SparsePrefillChunkCache(
        num_reqs=len(lengths),
        num_qo_tokens=q,
        max_seq_len=max(lengths),
        swa_window_size=128,
        swa_page_size=256,
        seq_lens=tensor(lengths),
        query_start_loc=tensor([0] + list(itertools.accumulate(physical))),
        query_pos=tensor(positions),
        swa_token_ids=torch.arange(offsets[-1], dtype=torch.int32, device="cuda"),
        swa_first_pos=tensor([n - w for n, w in zip(lengths, window)]),
        swa_gather_lens=tensor(window),
        swa_offsets=tensor(offsets),
        real_query_seq_lens=tensor(real),
        compact_seq_lens_cpu=tuple(lengths),
    )
    topk = 512
    causal = ((cache.query_pos.long() + 1) // ratio).clamp_min(1)
    raw = (
        (
            torch.arange(q, device="cuda")[:, None] * 23
            + torch.arange(topk, device="cuda")[None, :] * 179
        )
        % causal[:, None]
    ).int()
    raw[:, 1::4] = raw[:, ::4].clone()
    raw[cache.query_pos < 0] = -1
    metadata = SimpleNamespace(page_table=pages, sparse_raw_indices=lambda _: raw)
    ids, indices, lens = cache.layer_inputs(ratio, metadata, page)
    return cache, ids, indices, lens, page, cursor


def make_v4_pool(pages, page, seed):
    stride = ((page * 584 + 575) // 576) * 576
    pool = torch.zeros((pages, stride), dtype=torch.uint8, device="cuda")
    rows = torch.arange(pages * page, device="cuda").reshape(pages, page)
    data = pool[:, : page * 576].view(pages, page, 576)
    data[:, :, :448].copy_(((rows + seed) % 60 + 16).byte()[:, :, None])
    data[:, :, 448:].view(torch.bfloat16).copy_(
        ((rows + seed).float() % 97 / 64).bfloat16()[:, :, None]
    )
    pool[:, page * 576 : page * 584].fill_(127)
    return pool


@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("cp,rank", [(1, 0)] + [(8, rank) for rank in range(8)])
@pytest.mark.parametrize(
    "lengths,extends", [([65537, 4097, 1025], [33, 7, 1]), ([1], [1])]
)
def test_effective_kv_and_native_attention(ratio, cp, rank, lengths, extends):
    from flash_mla.flash_mla_interface import flash_mla_sparse_fwd
    from sglang.kernels.ops.attention.dsv4.dequant_k_cache import (
        dequantize_k_cache_paged,
    )

    cache, ids, indices, lens, page, pages = make_case(
        lengths, extends, cp, rank, ratio
    )
    packed, remapped, count = cache.pack_compressed_topk(ids, indices, lens, 512)
    old_ids = torch.cat((ids, cache.swa_token_ids + 2**24))
    new_ids = torch.cat((packed, cache.swa_token_ids + 2**24))
    valid = torch.arange(indices.shape[1], device="cuda")[None, :] < lens[:, None]
    assert torch.equal(old_ids[indices[valid].long()], new_ids[remapped[valid].long()])
    assert count.untyped_storage().nbytes() == 4

    pool = make_v4_pool(pages, page, 17)
    swa_pool = make_v4_pool((cache.swa_token_ids.numel() + 255) // 256, 256, 29)
    q = (
        torch.randn((cache.num_qo_tokens, 64, 512), dtype=torch.bfloat16, device="cuda")
        * 0.1
    )
    sink = torch.zeros(64, device="cuda")
    outputs = []
    for selected, mapping, valid_count in [
        (ids, indices, None),
        (packed, remapped, count),
    ]:
        n = selected.numel()
        kv = torch.full(
            (n + cache.swa_token_ids.numel(), 1, 512),
            19,
            dtype=torch.bfloat16,
            device="cuda",
        )
        dequantize_k_cache_paged(
            pool, selected, page, out=kv[:n], valid_count=valid_count
        )
        dequantize_k_cache_paged(swa_pool, cache.swa_token_ids, 256, out=kv[n:])
        if valid_count is not None:
            assert bool((kv[int(count) : n] == 19).all())
        output, _, _ = flash_mla_sparse_fwd(
            q=q,
            kv=kv,
            indices=mapping.unsqueeze(1),
            sm_scale=512**-0.5,
            d_v=512,
            attn_sink=sink,
            topk_length=lens,
        )
        outputs.append(output)
    real_queries = cache.query_pos >= 0
    assert torch.equal(outputs[0][real_queries], outputs[1][real_queries])


@pytest.mark.parametrize("layout_name", ["v4", "v41", "v41_fp4"])
@pytest.mark.parametrize("page", [1, 7, 128, 256])
def test_counted_triton_layouts(monkeypatch, layout_name, page):
    import sglang.kernels.ops.attention.dsv4.dequant_k_cache as dequant
    from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

    monkeypatch.setattr(dequant, "_hcu_dequantize_k_cache_paged", None)
    layout = KVLayout(layout_name)
    pages = (37 + page - 1) // page
    if layout is KVLayout.V4:
        pool = make_v4_pool(pages, page, 17)
    else:
        pool = torch.zeros(
            (pages, layout.page_bytes(page)), dtype=torch.uint8, device="cuda"
        )
        pool[:, : page * layout.data_bytes].fill_(0x24 if layout.is_fp4 else 0x38)
        pool[:, page * layout.data_bytes : page * layout.bytes_per_token].fill_(
            0x38 if layout.is_fp4 else 127
        )
    ids = torch.arange(37, dtype=torch.int32, device="cuda")
    reference = dequant.dequantize_k_cache_paged(pool, ids, page, layout=layout)
    ids[17:] = -(2**30)
    output = torch.full_like(reference, 19)
    count = torch.tensor([17], dtype=torch.int32, device="cuda")
    dequant.dequantize_k_cache_paged(
        pool, ids, page, out=output, layout=layout, valid_count=count
    )
    assert torch.equal(reference[:17].view(torch.uint8), output[:17].view(torch.uint8))
    assert bool((output[17:] == 19).all())
