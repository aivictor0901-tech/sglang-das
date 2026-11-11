import dataclasses
import os
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.kernels.ops.speculative.dspark.dspark_accept import AcceptGreedy
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.sampler import Sampler
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative import dspark_vp_greedy as greedy
from sglang.srt.speculative.dspark_components.dspark_draft import (
    DraftBlockProposer,
    DraftForwardResult,
)
from sglang.srt.speculative.dspark_components.dspark_draft_sampler import (
    DsparkDraftSampler,
    _resolve_folded_sampling,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def params(**kwargs):
    result = SamplingParams(**kwargs)
    result.normalize(None)
    result.verify(129280)
    return result


class TestGreedyVP(unittest.TestCase):
    def setUp(self):
        self.environment = mock.patch.dict(os.environ, SGLANG_DSPARK_VP_GREEDY_TOP1="1")
        self.environment.start()
        greedy.enabled.cache_clear()

    def tearDown(self):
        self.environment.stop()
        greedy.enabled.cache_clear()

    def test_default_off_and_disabled_request_behavior(self):
        with mock.patch.dict(os.environ):
            os.environ.pop("SGLANG_DSPARK_VP_GREEDY_TOP1", None)
            greedy.enabled.cache_clear()
            self.assertFalse(greedy.enabled())
            greedy.validate_request(
                params(temperature=0.6, frequency_penalty=0.5), SimpleNamespace()
            )

    def test_request_admission(self):
        request = SimpleNamespace()
        greedy.validate_request(params(temperature=0), request)
        greedy.validate_request(params(temperature=0.6, top_k=1), request)
        for kwargs in [
            dict(temperature=0.6),
            dict(temperature=0, frequency_penalty=0.5),
            dict(temperature=0, presence_penalty=0.5),
            dict(temperature=0, repetition_penalty=1.05),
            dict(temperature=0, min_new_tokens=8),
            dict(temperature=0, logit_bias={"9": 1}),
            dict(temperature=0, json_schema='{"type":"object"}'),
            dict(temperature=0, custom_params={"thinking_budget": 5}),
        ]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                greedy.validate_request(params(**kwargs), request)
        for field in [
            "return_logprob",
            "return_sampling_mask",
            "return_flat_raw_top_logprobs",
            "custom_logit_processor",
            "token_ids_logprob",
        ]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                greedy.validate_request(
                    params(temperature=0), SimpleNamespace(**{field: True})
                )

    def test_markov_layout(self):
        self.assertTrue(greedy.markov_has_full_vocab(SimpleNamespace(_tp_shard=None)))
        for geometry, expected in [
            ((1, 0, 129280), True),
            ((8, 0, 16160), False),
            ((1, 1, 129280), False),
            ((1, 0, 16160), False),
        ]:
            head = SimpleNamespace(
                vocab_size=129280,
                _tp_shard=SimpleNamespace(
                    tp_size=geometry[0],
                    org_vocab_start=geometry[1],
                    org_vocab_end=geometry[2],
                ),
            )
            with self.subTest(geometry=geometry):
                self.assertEqual(greedy.markov_has_full_vocab(head), expected)

    def test_accept_preserves_cutoffs_and_bonus(self):
        for seed in range(12):
            torch.manual_seed(seed)
            logits = torch.randn(48, 32)
            winners = logits.argmax(-1)
            candidates = torch.randint(0, 32, (8, 6))
            candidates[:, 1:] = winners.view(8, 6)[:, :5]
            if seed % 2:
                candidates[::2, 2] += 1
            kwargs = dict(
                candidates=candidates,
                verify_num_draft_tokens=6,
                cutoff_verify_lens=torch.arange(1, 9).clamp_max(6),
            )
            reference = AcceptGreedy.execute(target_logits=logits, **kwargs)
            compact = AcceptGreedy.execute(
                target_logits=logits[:, :1], target_token_ids=winners, **kwargs
            )
            with self.subTest(seed=seed):
                for expected, actual in zip(reference, compact):
                    self.assertTrue(torch.equal(expected, actual))

    def test_sampler_returns_owned_winners(self):
        output = LogitsProcessorOutput(
            next_token_logits=torch.randn(8, 1), target_top1_token_ids=torch.arange(8)
        )
        info = SimpleNamespace(
            is_all_greedy=True,
            has_custom_logit_processor=False,
            acc_linear_penalties=None,
            penalizer_orchestrator=None,
            grammar_mask=None,
            logit_bias=None,
        )
        result = Sampler.forward(None, output, info, False, [], [], torch.arange(8))
        self.assertTrue(torch.equal(result, output.target_top1_token_ids))
        self.assertNotEqual(result.data_ptr(), output.target_top1_token_ids.data_ptr())
        sliced = dataclasses.replace(
            output, target_top1_token_ids=output.target_top1_token_ids[:3]
        )
        self.assertEqual(sliced.target_top1_token_ids.numel(), 3)
        for field, value in [("is_all_greedy", False), ("logit_bias", torch.ones(1))]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                Sampler.forward(
                    None,
                    output,
                    SimpleNamespace(**{**vars(info), field: value}),
                    False,
                    [],
                    [],
                    torch.arange(8),
                )

    def test_greedy_graph_does_not_allocate_sampling_buffers(self):
        self.assertFalse(
            _resolve_folded_sampling(
                model=SimpleNamespace(),
                gamma=5,
                max_bs=8,
                device="cpu",
                tp_rank=0,
                available_memory_gb=100,
            )
        )

    def test_draft_graph_hook_uses_current_confidence_interface(self):
        hidden = torch.randn(10, 4)
        tokens = torch.arange(10).view(2, 5)
        confidence = mock.Mock(return_value=torch.ones(2, 5))
        model = SimpleNamespace(
            sample_from_anchor=True,
            markov_head=SimpleNamespace(),
            draft_lm_head_vp=object(),
            collapse_hc_head=lambda x: x,
        )
        sampler = DsparkDraftSampler(
            model=model,
            gamma=5,
            max_bs=2,
            device="cpu",
            tp_sync=None,
            confidence_fn=confidence,
            folded_sampling=False,
        )
        with mock.patch.object(greedy, "draft_block", return_value=tokens) as reduce:
            sampler(hidden, torch.arange(10))
        self.assertTrue(torch.equal(sampler.out.view(2, 5), tokens))
        self.assertTrue(torch.equal(sampler.confidence_out, torch.ones(2, 5)))
        self.assertEqual(reduce.call_args.kwargs["padded_rows"], 10)
        self.assertTrue(
            torch.equal(confidence.call_args.kwargs["draft_tokens"], tokens)
        )

    def test_eager_proposal_preserves_hidden_and_confidence_tap(self):
        hidden = torch.randn(10, 4)
        hidden_3d = hidden.view(2, 5, 4)
        block_ids = torch.arange(10).view(2, 5)
        tokens = block_ids + 1
        model = SimpleNamespace(sample_from_anchor=True, collapse_hc_head=lambda x: x)
        with mock.patch(
            "sglang.srt.speculative.dspark_components.dspark_draft.enable_num_token_non_padded",
            return_value=False,
        ):
            proposer = DraftBlockProposer(
                draft_model=model,
                draft_model_runner=SimpleNamespace(device="cpu"),
                gamma=5,
                mask_token_id=0,
                draft_block_spec_info=None,
                tp_sync=None,
            )
        forward = DraftForwardResult(
            draft_block_ids=block_ids,
            raw_hidden=hidden,
            draft_hidden_3d=hidden_3d,
            can_run_graph=False,
        )
        with (
            mock.patch.object(proposer, "_run_forward", return_value=forward),
            mock.patch.object(
                proposer, "_base_logits_context", return_value=nullcontext()
            ),
            mock.patch.object(greedy, "draft_block", return_value=tokens),
        ):
            result = proposer.propose(
                batch=None,
                draft_input=None,
                verify_window=None,
                bs=2,
                device="cpu",
                target_model=SimpleNamespace(get_input_embeddings=lambda: None),
                sampling_info=None,
            )
        self.assertTrue(torch.equal(result.draft_block.draft_tokens, tokens))
        self.assertIs(result.draft_hidden, hidden_3d)
        self.assertIs(result.confidence_tap, hidden)
        self.assertIsNone(result.draft_block.corrected_logits)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA or ROCm device required")
class TestGreedyVPKernels(unittest.TestCase):
    def test_reduction_matches_full_vocab_argmax(self):
        torch.manual_seed(897)
        for rows, width in [(0, 37), (1, 37), (8, 16160), (48, 16160)]:
            for case in ["random", "ties", "negative_inf", "nan", "bias_flip"]:
                logits = torch.randn(
                    8, rows, width, device="cuda", dtype=torch.bfloat16
                )
                bias = (
                    torch.zeros_like(logits, dtype=torch.float32)
                    if case == "bias_flip"
                    else None
                )
                if case == "ties":
                    logits.fill_(-1)
                    logits[:, :, 0] = 7
                elif case == "negative_inf":
                    logits.fill_(-torch.inf)
                elif case == "nan":
                    logits.fill_(0)
                    logits[0, :, 20] = torch.nan
                    logits[7, :, 0] = torch.nan
                elif case == "bias_flip":
                    logits.fill_(0)
                    logits[0, :, 0] = 1
                    bias[7, :, -1] = 2
                local = torch.empty((8, rows, 2), device="cuda")
                for rank in range(8):
                    greedy.local_candidates(
                        logits[rank],
                        None if bias is None else bias[rank],
                        vocab_start=rank * width,
                        out=local[rank],
                    )
                scores = torch.empty(rows, device="cuda")
                winners = torch.empty(rows, device="cuda", dtype=torch.int64)
                greedy.merge_candidates(
                    local.view(8 * rows, 2),
                    scores,
                    winners,
                    vp_size=8,
                    shard_size=width,
                )
                full = logits.float() if bias is None else logits.float() + bias
                full = full.permute(1, 0, 2).reshape(rows, 8 * width)
                expected = full.argmax(-1)
                expected_scores = full.gather(1, expected[:, None])[:, 0]
                with self.subTest(rows=rows, case=case):
                    self.assertTrue(torch.equal(winners, expected))
                    self.assertTrue(
                        torch.all(
                            (scores == expected_scores)
                            | (scores.isnan() & expected_scores.isnan())
                        )
                    )


if __name__ == "__main__":
    unittest.main()
