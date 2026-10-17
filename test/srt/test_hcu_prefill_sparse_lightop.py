import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils._python_dispatch import TorchDispatchMode

REPO_ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = (
    REPO_ROOT / "python/sglang/srt/layers/attention/dsv4/prefill_sparse_lightop.py"
)
SPEC = importlib.util.spec_from_file_location("prefill_sparse_under_test", ADAPTER_PATH)
sparse = importlib.util.module_from_spec(SPEC)
ENV_SPEC = importlib.util.spec_from_file_location(
    "sparse_environ_under_test", REPO_ROOT / "python/sglang/srt/environ.py"
)
environ = importlib.util.module_from_spec(ENV_SPEC)
ENV_SPEC.loader.exec_module(environ)
# Load the checkout's real registry without importing SGLang's serving runtime.
with patch.dict(sys.modules, {"sglang.srt.environ": environ}):
    SPEC.loader.exec_module(sparse)


class NoScalarDeviceRead(TorchDispatchMode):
    """Reject implicit item(), including scalar Tensor indexing in C++."""

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func == torch.ops.aten._local_scalar_dense.default:
            raise AssertionError("sparse request mapping extracted a tensor scalar")
        return func(*args, **(kwargs or {}))


class TestHcuPrefillSparseLightop(unittest.TestCase):
    def test_sparse_switch_defaults_on_and_can_be_disabled(self):
        flag = environ.envs.SGLANG_HCU_PREFILL_PAGED_SPARSE
        self.assertIsInstance(flag, environ.EnvBool)
        self.assertIs(sparse.envs, environ.envs)
        with patch.dict(os.environ, {}, clear=True):
            self.assertTrue(sparse.prefill_paged_sparse_enabled())
        for value, expected in [
            ("0", False),
            ("1", True),
            ("false", False),
            ("true", True),
        ]:
            with (
                self.subTest(value=value),
                patch.dict(
                    os.environ, {"SGLANG_HCU_PREFILL_PAGED_SPARSE": value}, clear=True
                ),
            ):
                self.assertEqual(sparse.prefill_paged_sparse_enabled(), expected)
        with flag.override(False):
            self.assertFalse(sparse.prefill_paged_sparse_enabled())

    def test_compact_lengths_include_partial_block(self):
        lengths = torch.tensor([0, 1, 3, 8, 9, 11, 32, 33], dtype=torch.int32)
        expected = torch.tensor([0, 1, 3, 8, 9, 11, 32, 25], dtype=torch.int32)
        torch.testing.assert_close(sparse.compact_valid_lens(lengths, 4), expected)

    def test_mask_conversion_retains_partial_block_and_padding(self):
        mask = torch.zeros((2, 11), dtype=torch.bool)
        mask[0, 2] = True
        mask[1, 10] = True
        result = sparse.candidate_blocks_from_mask(mask, 4)
        invalid = sparse.INVALID_BLOCK
        expected = torch.tensor(
            [[0, invalid, invalid, invalid], [1, invalid, invalid, invalid]],
            dtype=torch.int32,
        )
        torch.testing.assert_close(result, expected)

    def test_whole_block_scorer_and_valid_prefix_before_topk(self):
        q = torch.zeros((2, 1, 4, 64), dtype=torch.int8)
        sf = torch.zeros((2, 1, 4), dtype=torch.int32)
        weights = torch.ones((2, 4), dtype=torch.float32)
        invalid = sparse.INVALID_BLOCK
        blocks = torch.tensor([[0, 1, invalid, invalid]] * 2, dtype=torch.int32)
        lengths = torch.tensor([3, 11], dtype=torch.int32)
        table = torch.zeros((2, 1), dtype=torch.int32)
        kv = torch.zeros((1, 64, 1, 68), dtype=torch.uint8)
        schedule_calls, scorer_calls, topk_lengths = [], [], []

        def schedule(lengths, pages, pair_ids, page_size, selected, dtype, block, **kw):
            schedule_calls.append((lengths.clone(), pair_ids.clone()))
            self.assertEqual(page_size, 64)
            self.assertEqual(block, 8)
            return torch.zeros((16,), dtype=torch.uint8)

        def scorer(*args, **kwargs):
            self.assertEqual(len(args), 6)
            self.assertEqual(kwargs, {"num_sms": 1, "is_prefill": True})
            self.assertEqual(args[2].dtype, torch.float32)
            scorer_calls.append(args[0][0].shape[0])
            # Tail values are deliberately larger than every valid score.
            scores = torch.arange(32, dtype=torch.float32).repeat(
                args[0][0].shape[0], 1
            )
            return scores

        def topk(scores, valid_lens, *, out_offsets, out_indices):
            topk_lengths.extend(valid_lens.tolist())
            allowed = torch.arange(scores.shape[1])[None, :] < valid_lens[:, None]
            ranked = scores.masked_fill(~allowed, -torch.inf)
            out_indices.copy_(ranked.topk(out_indices.shape[1], dim=-1).indices)

        cache = {}
        operands = dict(
            q_fp4=q,
            q_sf=sf,
            k_cache=kv,
            weights=weights,
            blocks=blocks,
            paged_table=table,
            context_lens=lengths,
            index_topk=2,
            budget_bytes=1,
            num_sms=1,
            schedule_fn=schedule,
            logits_fn=scorer,
            topk_fn=topk,
            schedule_cache=cache,
        )
        expected = torch.tensor([[1, 2], [9, 10]], dtype=torch.int32)
        torch.testing.assert_close(
            sparse.select_prefill_sparse_rows(**operands), expected
        )
        self.assertEqual(len(schedule_calls), 2)
        self.assertEqual(scorer_calls, [1, 1])
        self.assertEqual(topk_lengths, [3, 11])
        self.assertTrue(all(ids.tolist() == [0] for _, ids in schedule_calls))
        # Reuse only within the same caller-owned forward cache.
        torch.testing.assert_close(
            sparse.select_prefill_sparse_rows(**operands), expected
        )
        self.assertEqual(len(schedule_calls), 2)

    def test_request_page_table_keeps_scalar_request_id_on_device(self):
        table = torch.arange(4 * 256, dtype=torch.int32).reshape(4, 256) * 8
        for ratio in (1, 2):
            for length in (0, 1, 65):
                columns = torch.arange((length + 63) // 64) * 64 * ratio
                expected = (table[2, columns].long() // ratio // 64).int()
                for request in (torch.tensor(2), torch.tensor([2])):
                    with self.subTest(ratio=ratio, length=length, shape=request.shape):
                        with NoScalarDeviceRead():
                            actual = sparse.request_page_table(
                                table, request, length, ratio, 64
                            )
                        torch.testing.assert_close(actual, expected)

    def test_consumer_request_mapping_never_extracts_tensor_scalar(self):
        # Two nonempty ragged requests with an empty request between them;
        # their pool IDs differ from their batch positions. Exercise both the
        # first consumer (page-table build) and cached subsequent consumer.
        rows, heads, ratio = 5, 4, 2
        table = torch.arange(4 * 256, dtype=torch.int32).reshape(4, 256) * ratio
        request_ids = torch.tensor([2, 1, 0], dtype=torch.int32)
        selected_parts = [
            torch.tensor([[0, 7], [8, -1], [24, 31]], dtype=torch.int32),
            torch.tensor([[1, 15], [16, 23]], dtype=torch.int32),
        ]
        selected = torch.cat(selected_parts)
        expected_pages = torch.full_like(selected, -1)
        for begin, end, pool_id in ((0, 3, 2), (3, 5, 0)):
            indices = selected[begin:end]
            expected_pages[begin:end] = torch.where(
                indices >= 0,
                table[pool_id, indices.clamp_min(0).long() * ratio] // ratio,
                -1,
            )
        page_out = torch.empty((rows, 2), dtype=torch.int32)
        raw_out = torch.empty_like(page_out)
        metadata = SimpleNamespace(
            request_masks=[
                torch.zeros((3, 32), dtype=torch.bool),
                torch.zeros((0, 0), dtype=torch.bool),
                torch.zeros((2, 24), dtype=torch.bool),
            ],
            request_block_ids=[
                torch.arange(4, dtype=torch.int32).repeat(3, 1),
                torch.empty((0, 4), dtype=torch.int32),
                torch.arange(4, dtype=torch.int32).repeat(2, 1),
            ],
        )
        pool = SimpleNamespace(
            get_index_k_page_size=lambda _: 64,
            get_index_k_with_scale_buffer=lambda _: torch.zeros(
                (4, 64 * 68), dtype=torch.uint8
            ),
        )
        indexer = SimpleNamespace(
            uses_candidates=True,
            is_candidate_source=False,
            candidate_block_size=8,
            index_head_dim=128,
            n_local_heads=heads,
            candidate_topk_blocks=4,
            index_topk=2,
            rope_head_dim=0,
            head_weights=lambda _: torch.ones((rows, heads), dtype=torch.bfloat16),
            wq_b=lambda _: (
                torch.zeros((rows, heads * 128), dtype=torch.bfloat16),
                None,
            ),
        )
        arguments = dict(
            pool=pool,
            indexer=indexer,
            layer_id=24,
            x=torch.zeros((rows, 1)),
            q_lora=torch.zeros((rows, 1)),
            freqs_cis=torch.ones((64, 1), dtype=torch.complex64),
            positions=torch.tensor([15, 31, 63, 31, 47]),
            req_to_token=table,
            request_pool_indices=request_ids,
            lc_per_request=[32, 0, 24],
            rows_per_request=[3, 0, 2],
            candidate_metadata=metadata,
            compress_ratio=ratio,
            page_indices=page_out,
            raw_indices=raw_out,
        )
        packed = (
            torch.zeros((rows, 1, heads, 64), dtype=torch.int8),
            torch.zeros((rows, 1, heads), dtype=torch.int32),
        )
        with (
            patch.object(sparse, "_lightop_functions", return_value=(None, None, None)),
            patch.object(sparse, "pack_prefill_query", return_value=packed),
        ):
            for consumer in range(2):
                with patch.object(
                    sparse, "select_prefill_sparse_rows", side_effect=selected_parts
                ):
                    with NoScalarDeviceRead():
                        self.assertTrue(
                            sparse.consume_prefill_candidate_sparse(**arguments)
                        )
                torch.testing.assert_close(page_out, expected_pages)
                torch.testing.assert_close(raw_out, selected)


if __name__ == "__main__":
    unittest.main()
