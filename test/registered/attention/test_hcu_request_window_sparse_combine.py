"""GPU regression for explicit RequestWindow indices, including CP padding."""
import unittest

import torch

from sglang.srt.layers.attention.dsv4.sparse_prefill_utils import combine_topk_swa_indices
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.srt.utils import is_hcu
from sglang.test.ci.ci_register import register_hcu_ci

register_hcu_ci(est_time=30, suite="stage-b-test-1-hcu-small")


@unittest.skipUnless(is_hcu(), "HCU GPU regression")
class TestHcuRequestWindowSparseCombine(unittest.TestCase):
    def test_explicit_global_swa_indices_survive_cp_sharding(self):
        device = "cuda:0"
        window = 128
        cases = [([0, 300, 17], [17, 3, 9]), ([0, 128], [1, 2])]
        checks = 0
        for prefixes, extends in cases:
            nr = len(extends)
            starts = [sum(extends[:r]) for r in range(nr)]
            total = sum(extends)
            padded = ((total + 7) // 8) * 8
            global_pos = torch.tensor([p + q for p, n in zip(prefixes, extends) for q in range(n)])
            global_req = torch.tensor([r for r, n in enumerate(extends) for _ in range(n)])
            for reset in [False, True]:
                floors = torch.tensor([p if reset else 0 for p, n in zip(prefixes, extends) for _ in range(n)])
                layout = window_layout(global_req, global_pos, window=window, capacity=256, floor=floors, num_groups=nr)
                positions = torch.full((padded,), -1, dtype=torch.int32)
                positions[:total] = global_pos.to(torch.int32)
                indices = torch.full((padded, window), -1, dtype=torch.int32)
                indices[:total] = layout.indices
                lengths = torch.zeros(padded, dtype=torch.int32)
                lengths[:total] = layout.lengths
                for rank in range(8):
                    rows = list(range(rank, padded, 8))
                    counts = [sum(start <= q < start + n for q in rows) for start, n in zip(starts, extends)]
                    counts[-1] += sum(q >= total for q in rows)
                    query_start = torch.tensor([0] + [sum(counts[:r + 1]) for r in range(nr)], dtype=torch.int32)
                    n = len(rows)
                    # Noncontiguous CP slices are intentional.
                    swa = indices.cuda()[rank::8]
                    swa_len = lengths.cuda()[rank::8]
                    for ratio, topk in [(1, 0), (1, 16), (2, 16), (4, 16), (128, 4)]:
                        with self.subTest(prefixes=prefixes, reset=reset, rank=rank, ratio=ratio, topk=topk):
                            top = torch.arange(max(topk, 1), dtype=torch.int32).repeat(n, 1)
                            if topk > 2:
                                top[:, 1] = -1
                            cb = torch.arange(nr, dtype=torch.int32) * 10000
                            sb = torch.full((nr,), 100000, dtype=torch.int32)
                            expected = torch.full((n, ((topk + window + 127) // 128) * 128), -1, dtype=torch.int32)
                            expected_len = torch.zeros(n, dtype=torch.int32)
                            for local, global_row in enumerate(rows):
                                if global_row >= total:
                                    continue
                                req = next(r for r, start in enumerate(starts) if start <= global_row < start + extends[r])
                                pos = int(positions[global_row])
                                nc = min((pos + 1) // ratio, topk)
                                ns = int(lengths[global_row])
                                expected[local, :nc] = torch.where(top[local, :nc] >= 0, top[local, :nc] + cb[req], -1)
                                expected[local, nc:nc + ns] = indices[global_row, :ns] + sb[req]
                                expected_len[local] = nc + ns
                            actual, actual_len = combine_topk_swa_indices(
                                topk_indices=top.cuda(), query_start_loc=query_start.cuda(), query_pos=positions[rank::8].cuda(),
                                seq_lens=torch.tensor([p + e for p, e in zip(prefixes, extends)], dtype=torch.int32, device=device),
                                gather_lens=torch.zeros(nr, dtype=torch.int32, device=device), compressed_base=cb.cuda(), swa_base=sb.cuda(),
                                window_size=window, compress_ratio=ratio, topk=topk, swa_indices=swa, swa_lengths=swa_len,
                            )
                            torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)
                            torch.testing.assert_close(actual_len.cpu(), expected_len, rtol=0, atol=0)
                            checks += 1
        self.assertEqual(checks, 160)
        print(f"GPU explicit-SWA combine checks passed: {checks}", flush=True)


if __name__ == "__main__":
    unittest.main()
