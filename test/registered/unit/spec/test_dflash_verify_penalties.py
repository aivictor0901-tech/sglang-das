"""Verify-time penalties must preserve the ordinary sampler's decisions."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.sampling.penaltylib.orchestrator import BatchedPenalizerOrchestrator
from sglang.srt.sampling.penaltylib.repetition_penalty import (
    BatchedRepetitionPenalizer,
)
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo
from sglang.srt.speculative.dflash_utils import apply_dflash_verify_logits_adjustments
from sglang.srt.speculative.dspark_components.dspark_verify import (
    verify_logits_adjustments_are_noop,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def _sampling_info(batch_size, **kwargs):
    return SamplingBatchInfo(
        temperatures=torch.ones(batch_size, 1),
        top_ps=torch.ones(batch_size),
        top_ks=torch.ones(batch_size, dtype=torch.int32),
        min_ps=torch.zeros(batch_size),
        is_all_greedy=True,
        is_any_greedy=True,
        need_top_p_sampling=False,
        need_top_k_sampling=False,
        need_min_p_sampling=False,
        vocab_size=3,
        device="cpu",
        **kwargs,
    )


class _Batch:
    device = "cpu"

    def __init__(self, penalties):
        self.reqs = [
            SimpleNamespace(sampling_params=SimpleNamespace(repetition_penalty=p))
            for p in penalties
        ]


class _GrammarMask:
    def apply(self, logits):
        logits[:, 1] = float("-inf")


class TestVerifyPenalties(CustomTestCase):
    def test_forward_snapshot_repetition_changes_verify_winner(self):
        """Overlap snapshots drop the live penalizer; repetition must still work."""
        batch = _Batch([2.0])
        penalizer = BatchedPenalizerOrchestrator(
            vocab_size=3, batch=batch, penalizers={BatchedRepetitionPenalizer}
        )
        penalizer.cumulate_output_tokens(torch.tensor([0]))
        info = _sampling_info(1, penalizer_orchestrator=penalizer).copy_for_forward()
        logits = torch.tensor([[4.0, -1.0, 3.0]]).repeat(3, 1)

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits, sampling_info=info, draft_token_num=3
        )

        torch.testing.assert_close(
            logits, torch.tensor([[2.0, -1.0, 3.0]]).repeat(3, 1)
        )
        self.assertEqual(logits.argmax(-1).tolist(), [2, 2, 2])

    def test_snapshot_transforms_match_sampler_for_each_verify_position(self):
        """Additive/scaling penalties, masks and biases must keep sampler order."""
        for dtype in (torch.float32, torch.bfloat16):
            for grammar in (None, _GrammarMask()):
                with self.subTest(dtype=dtype, grammar=grammar is not None):
                    info = _sampling_info(
                        2,
                        acc_additive_penalties=torch.tensor(
                            [[-0.5, -1.0, 0.0], [0.0, -0.25, -1.5]]
                        ),
                        acc_scaling_penalties=torch.tensor(
                            [[2.0, 2.0, 1.0], [1.0, 2.0, 1.5]]
                        ),
                        logit_bias=torch.tensor([[0.2, 0.0, -0.3], [-0.1, 0.7, 0.0]]),
                        grammar_mask=grammar,
                    )
                    logits = torch.tensor(
                        [[[4, -2, 1], [3, -1, 2]], [[2, -4, 3], [1, -3, 4]]],
                        dtype=dtype,
                    )
                    expected = logits.clone()
                    for position in range(2):
                        info.apply_logits_bias(expected[:, position, :])

                    apply_dflash_verify_logits_adjustments(
                        next_token_logits=logits.view(4, 3),
                        sampling_info=info,
                        draft_token_num=2,
                    )

                    torch.testing.assert_close(logits, expected)

    def test_live_repetition_applies_to_real_logits(self):
        """Applying a multiplicative penalty to zeros silently erased its effect."""
        batch = _Batch([2.0, 1.5])
        penalizer = BatchedPenalizerOrchestrator(
            vocab_size=3, batch=batch, penalizers={BatchedRepetitionPenalizer}
        )
        penalizer.cumulate_output_tokens(torch.tensor([0, 2]))
        info = _sampling_info(2, penalizer_orchestrator=penalizer)
        logits = torch.tensor(
            [[[4.0, -2.0, 3.0], [-4.0, 1.0, 2.0]], [[2.0, -1.0, 3.0], [1.0, 3.0, -6.0]]]
        )
        expected = logits.clone()
        for position in range(2):
            info.apply_logits_bias(expected[:, position, :])

        apply_dflash_verify_logits_adjustments(
            next_token_logits=logits.view(4, 3), sampling_info=info, draft_token_num=2
        )

        torch.testing.assert_close(logits, expected)

    def test_dspark_cannot_skip_snapshot_penalties(self):
        """Graph-folded acceptance must not bypass the corrected penalty helper."""
        self.assertTrue(verify_logits_adjustments_are_noop(_sampling_info(1)))
        for field in ("acc_additive_penalties", "acc_scaling_penalties"):
            with self.subTest(field=field):
                info = _sampling_info(1, **{field: torch.full((1, 3), 2.0)})
                self.assertFalse(verify_logits_adjustments_are_noop(info))


if __name__ == "__main__":
    unittest.main()
