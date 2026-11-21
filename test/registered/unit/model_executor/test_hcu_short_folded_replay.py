"""Bounded HCU CP8 folding: cached stores, padding, counts and DSpark tails."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.srt.model_executor import encoder_swa_replay as replay_module
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.model_executor.runner.eager_runner import EagerRunner
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class TestHcuShortFoldedReplay(unittest.TestCase):
    def make(self):
        req = NS(rid="short-hit", full_untruncated_fill_ids=list(range(2486)))
        mapping = torch.arange(4 * 4096).reshape(4, 4096)
        batch = NS(
            forward_mode=ForwardMode.EXTEND,
            reqs=[req],
            encoder_swa_reset=[True],
            prefix_lens=[2304],
            extend_lens=[182],
            extend_num_tokens=182,
            req_pool_indices=torch.tensor([3]),
            req_pool_indices_cpu=[3],
            input_ids=torch.arange(2304, 2486),
            prefill_input_ids_cpu=torch.arange(2304, 2486),
            out_cache_loc=mapping[3, 2304:2486],
            extend_logprob_start_lens=[182],
            engram_history=None,
            global_num_tokens=[182],
            global_num_tokens_for_logprob=[1],
            global_cp_num_tokens=None,
            return_logprob=False,
            spec_info=None,
            has_grammar=False,
            multimodal_inputs=[None],
            input_embeds=None,
        )
        window = NS(reset=Mock())
        runner = NS(
            device="cpu",
            token_to_kv_pool=NS(request_window=window),
            attn_backend=NS(is_dsv41=True),
            prefill_cuda_graph_runner=None,
            req_to_token_pool=NS(req_to_token=mapping),
            model=NS(model=NS(engram_hasher=None)),
            forward=Mock(),
            spec_algorithm=NS(is_dspark=Mock(return_value=True)),
        )
        parallel = NS(
            attn_cp_size=8,
            attn_cp_rank=0,
            attn_dp_size=1,
            pp_size=1,
            cp_strategy="interleave",
            attn_cp_group=NS(cpu_group=object()),
        )
        return NS(model_runner=runner), batch, parallel

    def fold(
        self,
        worker,
        batch,
        parallel,
        *,
        disagree=False,
        cp_v2=False,
        mode="prefill",
        cp_pool=True,
    ):
        def frames(outputs, frame, group):
            for x in outputs:
                x.copy_(frame)
            if disagree:
                outputs[-1][0] ^= 1

        def objects(outputs, plan, group):
            outputs[:] = [plan] * len(outputs)
            outputs[-1] = (("different-request", True, 2304),)

        with ExitStack() as stack:
            for path, value in [
                ("sglang.srt.utils.is_hcu", True),
                ("sglang.srt.runtime_context.get_parallel", parallel),
                ("sglang.srt.runtime_context.get_disagg", NS(disaggregation_mode=mode)),
                ("sglang.srt.layers.cp.utils.enable_cp_v2", cp_v2),
                (
                    "sglang.srt.mem_cache.cp_cache_layer_split.deepseek_v4_helpers.is_cp_cache_layer_split_deepseek_v4_pool",
                    cp_pool,
                ),
            ]:
                stack.enter_context(patch(path, return_value=value))
            vote = stack.enter_context(
                patch("torch.distributed.all_gather", side_effect=frames)
            )
            object_vote = stack.enter_context(
                patch("torch.distributed.all_gather_object", side_effect=objects)
            )
            folded = replay_module.try_fold_hcu_encoder_swa_replay(worker, batch)
            if folded is None:
                vote.assert_not_called()
                object_vote.assert_not_called()
            return folded

    def test_exact_2486_hit_preserves_scheduled_counts_and_cached_locations(self):
        worker, batch, parallel = self.make()
        folded = self.fold(worker, batch, parallel)
        self.assertEqual(folded.batch.prefix_lens, [2176])
        self.assertEqual(folded.batch.extend_lens, [310])
        self.assertEqual(folded.batch.input_ids.tolist(), list(range(2176, 2486)))
        self.assertEqual(folded.keep_rows.tolist(), list(range(128, 310)))
        self.assertEqual(folded.scheduled_row.tolist(), [-1] * 128 + list(range(182)))
        self.assertEqual(folded.row_floor.tolist(), [2176] * 310)
        self.assertEqual(folded.batch.global_num_tokens, [310])
        self.assertEqual(folded.batch.global_num_tokens_for_logprob, [1])
        self.assertEqual(batch.global_num_tokens, [182])
        self.assertEqual(batch.prefix_lens, [2304])
        self.assertEqual(batch.extend_num_tokens, 182)
        torch.testing.assert_close(
            folded.batch.out_cache_loc[128:], batch.out_cache_loc, rtol=0, atol=0
        )
        worker.model_runner.token_to_kv_pool.request_window.reset.assert_called_once()
        worker.model_runner.forward.assert_not_called()

    def test_unsupported_domains_fall_back_before_reset_or_forward(self):
        cases = [
            ("short-new", lambda w, b, p: setattr(b, "extend_lens", [127])),
            ("large-new", lambda w, b, p: setattr(b, "extend_lens", [4097])),
            ("short-prefix", lambda w, b, p: setattr(b, "prefix_lens", [64])),
            ("odd-prefix", lambda w, b, p: setattr(b, "prefix_lens", [2305])),
            ("DP", lambda w, b, p: setattr(p, "attn_dp_size", 2)),
            ("PP", lambda w, b, p: setattr(p, "pp_size", 2)),
            ("CP", lambda w, b, p: setattr(p, "attn_cp_size", 4)),
            ("strategy", lambda w, b, p: setattr(p, "cp_strategy", "zigzag")),
            (
                "graph",
                lambda w, b, p: setattr(
                    w.model_runner, "prefill_cuda_graph_runner", object()
                ),
            ),
            ("logprob", lambda w, b, p: setattr(b, "return_logprob", True)),
            ("spec", lambda w, b, p: setattr(b, "spec_info", object())),
            ("image", lambda w, b, p: setattr(b, "multimodal_inputs", [object()])),
            ("embedding", lambda w, b, p: setattr(b, "input_embeds", object())),
            ("grammar", lambda w, b, p: setattr(b, "has_grammar", True)),
            ("sync-count", lambda w, b, p: setattr(b, "global_num_tokens", [183])),
        ]
        for name, change in cases:
            with self.subTest(name=name):
                worker, batch, parallel = self.make()
                change(worker, batch, parallel)
                self.assertIsNone(self.fold(worker, batch, parallel))
                worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()
                worker.model_runner.forward.assert_not_called()
        for kwargs in [dict(cp_v2=True), dict(mode="decode"), dict(cp_pool=False)]:
            worker, batch, parallel = self.make()
            self.assertIsNone(self.fold(worker, batch, parallel, **kwargs))

    def test_disabled_prefill_graph_uses_eager_runner_and_can_fold(self):
        worker, batch, parallel = self.make()
        worker.model_runner.prefill_cuda_graph_runner = EagerRunner.__new__(EagerRunner)
        folded = self.fold(worker, batch, parallel)
        self.assertIsNotNone(folded)
        self.assertEqual(folded.batch.extend_num_tokens, 310)

    def test_other_speculative_algorithms_keep_separate_replay(self):
        worker, batch, parallel = self.make()
        worker.model_runner.spec_algorithm.is_dspark.return_value = False
        self.assertIsNone(self.fold(worker, batch, parallel))
        worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()
        worker.model_runner.forward.assert_not_called()

    def test_cached_images_fold_only_when_all_spans_precede_replay(self):
        for offsets, supported in [
            ([(32, 95), (2112, 2175)], True),
            ([(32, 95), (2112, 2176)], False),
            ([(2304, 2340)], False),
            (None, False),
        ]:
            with self.subTest(offsets=offsets):
                worker, batch, parallel = self.make()
                item = NS(modality=Modality.IMAGE, offsets=offsets)
                mm_input = NS(mm_items=[item])
                batch.multimodal_inputs = [mm_input]
                folded = self.fold(worker, batch, parallel)
                if supported:
                    self.assertIs(folded.batch.multimodal_inputs[0], mm_input)
                    self.assertEqual(folded.batch.prefix_lens, [2176])
                else:
                    self.assertIsNone(folded)
                    worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()

    def test_consensus_failure_precedes_reset(self):
        worker, batch, parallel = self.make()
        with self.assertRaisesRegex(RuntimeError, "plan differs across CP ranks"):
            self.fold(worker, batch, parallel, disagree=True)
        worker.model_runner.token_to_kv_pool.request_window.reset.assert_not_called()

    def test_cold_path_and_general_hcu_guard_stay_separate(self):
        worker, batch, parallel = self.make()
        batch.prefix_lens = [0]
        self.assertIsNone(self.fold(worker, batch, parallel))
        backend = backend_module.DeepseekV4AttnBackend.__new__(
            backend_module.DeepseekV4AttnBackend
        )
        with patch.object(backend_module, "_is_hcu", True):
            self.assertFalse(backend.folds_encoder_swa_replay)

    def test_cp_owner_padding_and_ring_commit_match_separate_replay(self):
        worker, batch, parallel = self.make()
        folded = self.fold(worker, batch, parallel)
        for rank in range(8):
            expected = [i for i in range(182) if i % 8 == rank]
            actual = [i - 128 for i in range(128, 310) if i % 8 == rank]
            self.assertEqual(actual, expected)
        self.assertEqual((310 + 7) // 8 * 8, 312)
        for capacity in [128, 256]:

            def commit(tags, start, end, floor):
                positions = torch.arange(start, end)
                layout = window_layout(
                    torch.zeros(end - start, dtype=torch.int64),
                    positions,
                    capacity=capacity,
                    floor=torch.full_like(positions, floor),
                    num_groups=1,
                )
                mask = layout.commit_mask
                tags[positions[mask] % capacity] = positions[mask]

            separate = torch.full((capacity,), -1, dtype=torch.int64)
            combined = separate.clone()
            commit(separate, 2176, 2304, 2176)
            commit(separate, 2304, 2486, 0)
            commit(combined, 2176, 2486, 2176)
            torch.testing.assert_close(combined, separate, rtol=0, atol=0)

    def test_dspark_tail_maps_only_new_rows_back_to_scheduled_extend(self):
        worker, batch, parallel = self.make()
        folded = self.fold(worker, batch, parallel)
        hidden = torch.arange(128 * 3).reshape(128, 3)
        output = NS(
            hidden_states=hidden.clone(),
            hidden_states_token_indices=torch.arange(182, 310),
        )
        replay_module.drop_folded_rows(
            logits_output=output, folded=folded, tail_lens=[128]
        )
        self.assertEqual(
            output.hidden_states_token_indices.tolist(), list(range(54, 182))
        )
        torch.testing.assert_close(output.hidden_states, hidden, rtol=0, atol=0)

    def test_deferred_engram_history_uses_fold_start_instead_of_cached_end(self):
        worker, batch, parallel = self.make()
        worker.model_runner.model.model.engram_hasher = NS(
            max_ngram_size=4, history_row=lambda ids: ids
        )
        folded = self.fold(worker, batch, parallel)
        self.assertEqual(folded.batch.engram_history.tolist(), [[2173, 2174, 2175]])
        self.assertIsNone(batch.engram_history)

    def compressor_case(self, *, tail_rows=None, keep_rows=182):
        backend = backend_module.DeepseekV4AttnBackend.__new__(
            backend_module.DeepseekV4AttnBackend
        )
        positions = torch.arange(2176, 2486)
        rows = 310 if tail_rows is None else tail_rows
        x = (
            torch.arange((312 if tail_rows is None else rows) * 3)
            .reshape(-1, 3)
            .float()
        )
        tail = (
            None
            if tail_rows is None
            else NS(
                local_lens_cpu=[rows // 8],
                req_global=torch.zeros(rows, dtype=torch.int64),
                pos_global=positions[-rows:],
            )
        )
        backend.forward_metadata = NS(late_layer_tail=tail)
        backend._low_ratio_compress_torch = Mock()
        batch = NS(
            attn_cp_metadata=object(),
            extend_seq_lens_cpu=[310],
            positions=positions,
            encoder_swa_compress_rows=torch.arange(310 - keep_rows, 310),
        )
        with (
            patch.object(backend_module, "_is_hcu", True),
            patch.object(
                backend_module,
                "get_parallel",
                return_value=NS(attn_cp_rank=0, attn_cp_size=8),
            ),
            patch.object(
                backend_module,
                "token_req_indices",
                return_value=torch.zeros(310, dtype=torch.int64),
            ),
            patch.object(
                backend_module, "cp_materialize_global_token_order", return_value=x
            ),
            patch("torch.cuda.current_stream", return_value=None),
        ):
            backend._forward_low_ratio_sources_cp(
                layer=NS(compressor=object(), indexer=None),
                x=x[:39],
                q_lora=None,
                positions=positions[:39],
                forward_batch=batch,
                run_compressor=True,
                run_indexer=False,
            )
        return x, backend._low_ratio_compress_torch.call_args

    def test_compressor_never_rewrites_replay_or_padding_rows(self):
        x, call = self.compressor_case()
        torch.testing.assert_close(call.args[1], x[128:310], rtol=0, atol=0)
        self.assertEqual(call.args[3].tolist(), list(range(2304, 2486)))
        self.assertEqual(call.args[1].shape[0], 182)
        self.assertEqual(call.kwargs["out_loc_rows"].tolist(), list(range(128, 310)))

    def test_compressor_filters_physical_output_slots_and_padding_with_values(self):
        for ratio in [1, 2]:
            with self.subTest(ratio=ratio):
                backend = backend_module.DeepseekV4AttnBackend.__new__(
                    backend_module.DeepseekV4AttnBackend
                )
                absolute = torch.arange(2176, 2486)
                physical = absolute * 3 + 7
                raw = physical.clone()
                raw[:128] = 0
                backend.forward_metadata = NS(
                    core_metadata=NS(
                        raw_out_loc=raw, c1_out_loc=physical, c2_out_loc=physical
                    )
                )
                backend._low_ratio_pair_partners = Mock(
                    side_effect=lambda **kw: (kw["kv"], kw["score"])
                )
                backend._low_ratio_write_group = Mock()
                x = torch.arange(182 * 3).reshape(182, 3).float()
                layer = NS(
                    layer_id=0,
                    compress_ratio=ratio,
                    compressor=NS(
                        project=lambda v: (v, v), pool_pairs=lambda v, s: v[:, 0]
                    ),
                )
                backend._low_ratio_compress_torch(
                    layer,
                    x,
                    torch.zeros(182, dtype=torch.int64),
                    torch.arange(2304, 2486),
                    out_loc_rows=torch.arange(128, 310),
                )
                call = backend._low_ratio_write_group.call_args
                self.assertEqual(
                    call.args[2].tolist(), [p * 3 + 7 for p in range(2304, 2486)]
                )
                if ratio == 2:
                    self.assertFalse(
                        backend._low_ratio_pair_partners.call_args.kwargs["pad"].any()
                    )

    def test_decoder_tail_is_all_new_and_rejects_replay_overlap(self):
        x, call = self.compressor_case(tail_rows=128)
        torch.testing.assert_close(call.args[1], x, rtol=0, atol=0)
        with self.assertRaisesRegex(RuntimeError, "decoder tail contains replay rows"):
            self.compressor_case(tail_rows=128, keep_rows=64)


if __name__ == "__main__":
    unittest.main()
