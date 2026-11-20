"""RequestWindow TopK dequantization, CP padding and physical-page remapping."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import (
    SparsePrefillChunkCache,
    SparsePrefillWorkspace,
)
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.srt.utils import is_hcu
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=60, suite="stage-b-test-1-hcu-small")


@unittest.skipUnless(is_hcu(), "HCU GPU regression")
class TestRequestWindowSelectedDequant(unittest.TestCase):
    @staticmethod
    def _kv(page_size, pages, shift):
        """Independently pack unit-scale FP8 and BF16 RoPE values."""
        rows = page_size * pages
        values = torch.arange(rows * 448, device="cuda").reshape(rows, 448)
        values = ((values + shift) % 29 - 14).float() / 16
        fp8 = values.to(backend_module.fp8_dtype)
        rope = torch.arange(rows * 64, device="cuda").reshape(rows, 64)
        rope = (((rope + shift) % 31 - 15).float() / 16).bfloat16()
        cache = torch.zeros((pages, 1, page_size * 584), dtype=torch.uint8, device="cuda")
        data = cache[:, 0, :page_size * 576].view(pages, page_size, 576)
        data[:, :, :448] = fp8.view(torch.uint8).reshape(pages, page_size, 448)
        data[:, :, 448:] = rope.view(torch.uint8).reshape(pages, page_size, 128)
        cache[:, 0, page_size * 576:].fill_(127)
        return cache, torch.cat((fp8.bfloat16(), rope), dim=-1)

    def _case(self, prefixes, extends, topk, ratio, rank, reset, enabled):
        from lightop import kvcache
        from sglang.kernels.ops.attention.dsv4.kv_layout import KVLayout

        page_size, pages, window = 8, 64, 128
        seqs = [p + e for p, e in zip(prefixes, extends)]
        starts = [sum(extends[:r]) for r in range(len(extends))]
        total = sum(extends)
        padded = (total + 7) // 8 * 8
        global_req = torch.tensor([r for r, n in enumerate(extends) for _ in range(n)], device="cuda")
        global_pos = torch.tensor([p + i for p, n in zip(prefixes, extends) for i in range(n)], device="cuda")
        floors = torch.tensor([p if reset else 0 for p, n in zip(prefixes, extends) for _ in range(n)], device="cuda")
        layout = window_layout(global_req, global_pos, window=window, capacity=256, floor=floors, num_groups=len(extends))
        indices = torch.full((padded, window), -1, dtype=torch.int32, device="cuda")
        lengths = torch.zeros(padded, dtype=torch.int32, device="cuda")
        positions = torch.full((padded,), -1, dtype=torch.int32, device="cuda")
        indices[:total], lengths[:total], positions[:total] = layout.indices, layout.lengths, global_pos.int()
        local_rows = list(range(rank, padded, 8))
        real_lens = [sum(start <= i < start + n for i in local_rows) for start, n in zip(starts, extends)]
        physical_lens = list(real_lens)
        physical_lens[-1] += sum(i >= total for i in local_rows)
        qn = len(local_rows)
        cache = SparsePrefillChunkCache.build(
            seq_lens=torch.tensor(seqs, dtype=torch.int32, device="cuda"),
            extend_seq_lens=torch.tensor(extends, dtype=torch.int32, device="cuda"),
            query_lens=torch.tensor(physical_lens, dtype=torch.int32, device="cuda"),
            real_query_seq_lens=torch.tensor(real_lens, dtype=torch.int32, device="cuda"),
            query_pos=positions[rank::8].clone(),
            req_pool_indices=torch.arange(len(seqs), dtype=torch.int32, device="cuda"),
            req_to_token=torch.empty((0, 0), dtype=torch.int32, device="cuda"),
            full_to_swa=torch.empty(0, dtype=torch.int64, device="cuda"),
            swa_window_size=window, swa_page_size=page_size, num_qo_tokens=qn,
            max_seq_len=max(seqs), total_swa=layout.size,
            request_window_layout=layout, request_window_indices=indices[rank::8],
            request_window_lengths=lengths[rank::8], seq_lens_cpu=seqs,
            compact_seq_lens_cpu=tuple(seqs),
        )
        blocks = (max(seqs) + page_size - 1) // page_size
        # Deliberately share physical pages between different logical requests.
        table = ((torch.arange(blocks, device="cuda") * 7 + 3) % pages).int()
        page_table = table.repeat(qn, 1)
        raw = torch.arange(topk, device="cuda", dtype=torch.int32).repeat(qn, 1)
        raw[:, 1::5] = -1
        if topk > 3:
            raw[:, 3] = 0  # Preserve repeated logical references.
        core = SimpleNamespace(request_window_layout=layout, page_table=page_table, sparse_raw_indices=lambda _: raw)
        swa_cache, swa_ref = self._kv(page_size, pages, 5)
        extra_cache, extra_ref = self._kv(page_size, pages, 11)
        remaps = []

        def remap(layer, ids, *, family):
            remaps.append(family)
            return torch.where(ids >= 0, (ids + page_size) % (pages * page_size), ids).int()

        pool = SimpleNamespace(
            get_swa_key_page_size=lambda: page_size,
            get_extra_key_page_size=lambda _: page_size,
            get_swa_key_buffer_radix=lambda _: swa_cache,
            get_extra_key_buffer=lambda _: extra_cache,
            get_swa_key_layout=lambda: KVLayout.V4,
            get_extra_key_layout=lambda _: KVLayout.V4,
            remap_flat_token_ids_for_read=remap,
        )
        backend = backend_module.DeepseekV4AttnBackend.__new__(backend_module.DeepseekV4AttnBackend)
        backend.is_dsv41 = True
        backend.forward_metadata = SimpleNamespace(sparse_prefill_cache=cache)
        backend.sparse_prefill_workspace = SparsePrefillWorkspace(torch.device("cuda"))
        backend.softmax_scale, backend.head_dim_v = 512 ** -0.5, 512
        q = torch.zeros((qn, 1, 64, 512), dtype=torch.bfloat16, device="cuda")
        captured = []

        def inspect_attention(**kw):
            idx, lens = kw['indices'][:, 0].cpu(), kw['topk_length'].cpu()
            expected_rows, actual_rows = [], []
            for local, global_row in enumerate(local_rows):
                expected = []
                if global_row < total:
                    req = next(r for r, start in enumerate(starts) if start <= global_row < start + extends[r])
                    nc = min((int(global_pos[global_row]) + 1) // ratio, topk)
                    for logical in raw[local, :nc].cpu().tolist():
                        if logical >= 0:
                            physical = int(table[logical // page_size]) * page_size + logical % page_size
                            expected.append(extra_ref[(physical + page_size) % (pages * page_size)])
                    for slot in layout.indices[global_row, :int(layout.lengths[global_row])].cpu().tolist():
                        expected.append(swa_ref[(slot + page_size) % (pages * page_size)])
                valid = idx[local, :int(lens[local])]
                valid = valid[valid >= 0].cuda().long()
                actual = kw['kv'][valid, 0]
                expected = torch.stack(expected) if expected else extra_ref[:0]
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                expected_rows.append(expected)
                actual_rows.append(actual)
            captured.append(dict(workspace_rows=kw['kv'].shape[0], storage_bytes=kw['kv'].untyped_storage().nbytes()))
            # The oracle above verifies the exact ordered KV values consumed by
            # Attention. The model-level run separately exercises real kernels.
            return q[:, 0], None, None

        try:
            import flash_mla.flash_mla_interface as mla
        except ImportError:
            import flash_mla as mla
        with (
            patch.object(backend_module, '_use_dsv41_compact_sparse_prefill_workspace', return_value=True),
            patch.object(backend_module, 'is_cp_cache_layer_split_deepseek_v4_pool', return_value=True),
            patch.object(backend_module.envs.SGLANG_DSV41_PREFILL_TOPK_KV_WORKSPACE, 'get', return_value=enabled),
            patch.object(mla, 'flash_mla_sparse_fwd', side_effect=inspect_attention),
            patch.object(kvcache, 'dsv4_dequantize_sparse_topk_out', wraps=kvcache.dsv4_dequantize_sparse_topk_out) as selected,
        ):
            backend._forward_prefill_sparse(q=q, layer_id=0, compress_ratio=ratio, forward_batch=object(), token_to_kv_pool=pool, core_attn_metadata=core, attn_sink=torch.zeros(64, device="cuda"))
        n = sum(max(s // ratio, 1) for s in seqs)
        self.assertEqual(selected.call_count, int(enabled and n <= qn * topk))
        self.assertEqual(remaps, ['swa', 'extra'])
        self.assertEqual(captured[0]['workspace_rows'], layout.size + (min(n, qn * topk) if enabled else n))
        return captured[0]

    def test_selected_and_packed_preserve_global_swa_and_cp_padding(self):
        for prefixes, extends, topk in [([0, 300, 17], [17, 3, 9], 512), ([4096, 128], [1, 2], 16)]:
            for reset in (False, True):
                for ratio in (1, 2):
                    for rank in range(8):
                        with self.subTest(prefixes=prefixes, reset=reset, ratio=ratio, rank=rank):
                            self._case(prefixes, extends, topk, ratio, rank, reset, True)

    def test_switch_disabled_retains_full_dequantization(self):
        self._case([0, 300, 17], [17, 3, 9], 512, 1, 0, False, False)
        self._case([4096, 128], [1, 2], 16, 2, 7, True, False)


if __name__ == '__main__':
    unittest.main()
