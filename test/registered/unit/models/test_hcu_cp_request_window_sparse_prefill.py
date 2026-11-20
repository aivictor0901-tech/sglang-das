"""Read-side CP geometry for the GLOBAL request-window SWA workspace. CPU only."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention import deepseek_v4_backend as backend_module
from sglang.srt.layers.attention.dsv4 import sparse_prefill_utils
from sglang.srt.mem_cache import dsv41_request_window as window_module
from sglang.srt.mem_cache.cp_cache_layer_split import deepseek_v4_pool, staging
from sglang.srt.mem_cache.cp_cache_layer_split.staging import (
    request_window_page_capacity,
)
from sglang.srt.mem_cache.dsv41_request_window import window_layout
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def combine_cpu(**kw):
    """Independent explicit-SWA reference for the c0 index-combine operation."""
    n = kw['query_pos'].numel()
    width = ((kw['window_size'] + 63) // 64) * 64
    indices = torch.full((n, width), -1, dtype=torch.int32)
    lengths = torch.zeros(n, dtype=torch.int32)
    assert kw['topk'] == 0
    for q in range(n):
        if kw['query_pos'][q] < 0:
            continue
        valid = kw['swa_indices'][q]
        valid = valid[valid >= 0]
        indices[q, :valid.numel()] = valid
        lengths[q] = valid.numel()
    return indices, lengths


class TestHcuCpRequestWindowSparsePrefill(unittest.TestCase):
    def test_first_swa_stage_covers_dense_cold_window_before_sparse_cache_exists(self):
        page_size = 128
        layout = window_layout(
            torch.tensor([0, 1, 1]), torch.tensor([0, 0, 1]), num_groups=2
        )
        full_rows = torch.arange(layout.size, dtype=torch.int32)
        reference = getattr(
            staging.remap_indices_to_staging,
            "__wrapped__",
            staging.remap_indices_to_staging,
        )
        pool = deepseek_v4_pool.CpCacheLayerSplitDeepSeekV4TokenToKVPool
        state = SimpleNamespace(
            request_window=SimpleNamespace(layout=layout),
            _swa_read_pool=lambda: SimpleNamespace(page_size=page_size),
            _pool_num_pages=lambda kv_pool: 4,
            _batch_active_pages={"swa": deepseek_v4_pool._BatchActivePages()},
            _active_pages_for_indices=Mock(return_value=torch.tensor([2, -1, -1])),
            _clear_extra_read_state=Mock(),
            _get_staging_buffer=Mock(),
            _get_layer_owner_rank=lambda layer_id: 1,
            _compact_broadcast_for_read=Mock(),
            wait_layer_transfer=Mock(),
            cp_rank=0,
        )
        with (
            patch.object(staging, "_is_hcu", False),
            patch.object(deepseek_v4_pool, "remap_indices_to_staging", reference),
            patch.object(deepseek_v4_pool, "_USE_STATIC_CP_LAYER_SPLIT_PAGES", True),
        ):
            pool.prefetch_swa_layer(
                state, 5, layout.indices, read_indices=None, num_reqs=2
            )
            dense = pool.remap_flat_token_ids_for_read(
                state, 5, full_rows, family="swa"
            )
        # Both history pages are masked on this cold batch, but dequant still
        # reads their workspace rows. The first broadcast must include them.
        self.assertTrue((layout.indices[layout.indices >= 0] >= 2 * page_size).all())
        torch.testing.assert_close(dense, full_rows)
        torch.testing.assert_close(state._swa_remapped_indices, layout.indices)
        self.assertEqual(
            state._compact_broadcast_for_read.call_args.args[4].tolist(), [0, 1, 2]
        )
        state._active_pages_for_indices.assert_not_called()

    @staticmethod
    def _history_window():
        window = window_module.RequestWindow.__new__(window_module.RequestWindow)
        window.capacity, window.zero_row, window.page_size = 4, 8, 1
        window.prepared = None
        window.tags = torch.full((3, 9), -12345, dtype=torch.int64)
        window.tags[:, 1], window.tags[:, 6] = 1, 6
        window.state = SimpleNamespace(kv_buffer=[torch.ones(1) for _ in range(3)], kv_layout=object())
        window.workspace = SimpleNamespace(kv_buffer=[torch.zeros(1)], size=4)
        window.layout = SimpleNamespace(
            history_req=torch.tensor([0, 0, 1, 1]),
            history_pos=torch.tensor([1, -20, 6, -9]),
            history_valid=torch.tensor([True, False, True, False]),
            history_loc=torch.arange(4), size=4,
        )
        return window

    def test_prevalidated_history_is_consumed_and_rechecked_after_a_layer_write(self):
        w = self._history_window()
        with (
            patch.object(window_module, 'get_is_capture_mode', return_value=False),
            patch.object(window_module, '_capturing', return_value=False),
            patch.object(window_module, 'copy_packed_tokens') as copy,
            patch.object(torch, 'equal', wraps=torch.equal) as equal,
        ):
            w.prevalidate_history([0, 1])
            w.buffer(0)
            w.buffer(1)
            self.assertEqual(equal.call_count, 1)
            self.assertEqual(copy.call_count, 2)
            # A commit can overwrite this layer's old ring positions. A second
            # history copy must validate again, rather than reusing the first proof.
            w.tags[0, 1] = -1
            with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                w.buffer(0)
            self.assertEqual(equal.call_count, 2)
            self.assertEqual(copy.call_count, 2)

    def test_pending_tail_history_uses_its_own_layout_once(self):
        w = self._history_window()
        tail = SimpleNamespace(**vars(w.layout))
        tail.history_pos = torch.tensor([2, -20, 7, -9])
        w.tags[2, 2], w.tags[2, 7] = 2, 7
        with (
            patch.object(window_module, 'get_is_capture_mode', return_value=False),
            patch.object(window_module, '_capturing', return_value=False),
            patch.object(window_module, 'copy_packed_tokens') as copy,
            patch.object(torch, 'equal', wraps=torch.equal) as equal,
        ):
            w.prevalidate_history([0, 1])
            w.prevalidate_pending_history(tail, [2])
            w.buffer(0)
            w.buffer(1)
            self.assertEqual(equal.call_count, 2)
            w.activate(tail)
            w.buffer(2)
            self.assertEqual(equal.call_count, 2)
            self.assertEqual(copy.call_count, 3)
            # The proof cannot cover a second copy after this layer was written.
            w.tags[2, 2] = -1
            w.prepared = None
            with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                w.buffer(2)
            self.assertEqual(equal.call_count, 3)

    def test_pending_tail_proof_is_invalidated_by_intervening_changes(self):
        with (
            patch.object(window_module, 'get_is_capture_mode', return_value=False) as capture,
            patch.object(window_module, '_capturing', return_value=False),
            patch.object(window_module, 'copy_packed_tokens'),
            patch.object(window_module, 'commit_packed_rows'),
        ):
            for operation in ['reset', 'other_layout', 'dummy', 'capture', 'commit', 'new_batch']:
                with self.subTest(operation=operation):
                    w = self._history_window()
                    tail = SimpleNamespace(**vars(w.layout))
                    w.prevalidate_pending_history(tail, [2])
                    if operation == 'reset':
                        w.reset(torch.tensor([0]))
                    elif operation == 'other_layout':
                        w.activate(SimpleNamespace(**vars(w.layout)))
                    elif operation == 'dummy':
                        w.initialize_dummy_history()
                    elif operation == 'capture':
                        capture.return_value = True
                        w.buffer(0)
                        capture.return_value = False
                    elif operation == 'commit':
                        w.commit(2)
                    else:
                        w.prevalidate_history([])
                    w.tags[2, 1] = -1
                    w.activate(tail)
                    with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                        w.buffer(2)

    def test_pending_tail_missing_history_fails_before_forward(self):
        w = self._history_window()
        tail = SimpleNamespace(**vars(w.layout))
        with (
            patch.object(window_module, 'get_is_capture_mode', return_value=False),
            patch.object(window_module, '_capturing', return_value=False),
        ):
            w.tags[2, 1] = -1
            with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                w.prevalidate_pending_history(tail, [2])
            self.assertIsNone(w._pending_prevalidated_history)
            for layers in [[-1], [3]]:
                with self.assertRaises(IndexError):
                    w.prevalidate_pending_history(tail, layers)

    def test_legacy_cp_tail_is_prevalidated_immediately_after_rebuild(self):
        backend = backend_module.DeepseekV4AttnBackend.__new__(backend_module.DeepseekV4AttnBackend)
        w = self._history_window()
        backend.is_dsv41 = True
        backend.enable_decoder_swa_bounded_replay = True
        backend.model_runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(late_layer_start=5)))
        backend.token_to_kv_pool = SimpleNamespace(request_window=w, _swa_global_to_local={1: 0, 4: 1, 7: 2})
        old = SimpleNamespace(core_attn_metadata=SimpleNamespace(request_window_layout=SimpleNamespace(**vars(w.layout))))
        tail = SimpleNamespace(**vars(w.layout))
        rebuilt = SimpleNamespace(core_attn_metadata=SimpleNamespace(request_window_layout=tail))
        backend.tail_forward_metadata = old
        batch = SimpleNamespace(forward_mode=backend_module.ForwardMode.EXTEND)
        with (
            patch.object(backend_module, '_is_hcu', True),
            patch.object(backend_module, 'is_cp_cache_layer_split_deepseek_v4_pool', return_value=True),
            patch('sglang.srt.layers.moe.get_moe_a2a_backend', return_value=SimpleNamespace(is_none=lambda: False)),
            patch.object(backend, '_build_late_layer_tail_metadata', return_value=rebuilt) as rebuild,
            patch.object(window_module, 'get_is_capture_mode', return_value=False),
            patch.object(window_module, '_capturing', return_value=False),
            patch.object(window_module, 'copy_packed_tokens'),
            patch.object(torch, 'equal', wraps=torch.equal) as equal,
        ):
            backend.convert_late_layer_tail_to_hcu_cp(batch)
            rebuild.assert_called_once_with(batch, legacy_cp=True)
            self.assertIs(w._pending_prevalidated_history[0], tail)
            self.assertEqual(w._pending_prevalidated_history[1], {2})
            w.buffer(0)
            w.activate(tail)
            w.buffer(2)
            self.assertEqual(equal.call_count, 2)

    def test_prevalidation_only_covers_selected_layers_and_keeps_missing_history_error(self):
        w = self._history_window()
        w.tags[2, 1] = -1
        with (
            patch.object(window_module, 'get_is_capture_mode', return_value=False),
            patch.object(window_module, '_capturing', return_value=False),
            patch.object(window_module, 'copy_packed_tokens') as copy,
        ):
            w.prevalidate_history([0, 1])
            w.buffer(0)
            with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                w.buffer(2)
            self.assertEqual(copy.call_count, 1)
            with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                w.prevalidate_history([0, 1, 2])
            self.assertEqual(w._prevalidated_history_layers, set())
            for layers in [[-1], [3]]:
                with self.assertRaises(IndexError):
                    w.prevalidate_history(layers)

    def test_prevalidation_is_invalidated_by_reset_layout_dummy_history_and_capture(self):
        with (
            patch.object(window_module, '_capturing', return_value=False),
            patch.object(window_module, 'copy_packed_tokens'),
            patch.object(window_module, 'get_is_capture_mode', return_value=False) as capture,
        ):
            for operation in ['reset', 'activate', 'dummy', 'capture']:
                with self.subTest(operation=operation):
                    w = self._history_window()
                    w.prevalidate_history([0, 1])
                    if operation == 'reset':
                        w.reset(torch.tensor([0]))
                    elif operation == 'activate':
                        w.activate(SimpleNamespace(**vars(w.layout)))
                    elif operation == 'dummy':
                        w.initialize_dummy_history()
                    else:
                        capture.return_value = True
                        w.buffer(0)
                        capture.return_value = False
                    w.tags[1, 1] = -1
                    with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                        w.buffer(1)

    def test_cp_history_prevalidation_excludes_late_layers_and_other_backends(self):
        backend = backend_module.DeepseekV4AttnBackend.__new__(backend_module.DeepseekV4AttnBackend)
        window = SimpleNamespace(prevalidate_history=Mock())
        backend.token_to_kv_pool = SimpleNamespace(
            request_window=window, _swa_global_to_local={1: 0, 4: 1, 7: 2},
        )
        backend.is_dsv41 = True
        backend.enable_decoder_swa_bounded_replay = True
        backend.model_runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(late_layer_start=5)))
        batch = SimpleNamespace(forward_mode=SimpleNamespace(is_extend_without_speculative=lambda: True))
        with (
            patch.object(backend_module, '_is_hcu', True),
            patch.object(backend_module, 'is_cp_cache_layer_split_deepseek_v4_pool', return_value=True),
        ):
            backend._prevalidate_cp_request_window_history(batch)
            window.prevalidate_history.assert_called_once_with([0, 1])
            window.prevalidate_history.reset_mock()
            with patch.object(backend_module, '_is_hcu', False):
                backend._prevalidate_cp_request_window_history(batch)
            with patch.object(backend_module, 'is_cp_cache_layer_split_deepseek_v4_pool', return_value=False):
                backend._prevalidate_cp_request_window_history(batch)
            window.prevalidate_history.assert_not_called()
            backend.enable_decoder_swa_bounded_replay = False
            backend._prevalidate_cp_request_window_history(batch)
            window.prevalidate_history.assert_called_once_with([0, 1, 2])
        # Legacy HCU creates query-split metadata later, inside model.forward.
        # The sharded pool already requires history at this earlier planning hook.
        backend.enable_decoder_swa_bounded_replay = True
        window.prevalidate_history.reset_mock()
        real_batch = SimpleNamespace(
            forward_mode=backend_module.ForwardMode.EXTEND, attn_cp_metadata=None,
        )
        with (
            patch.object(backend_module, '_is_hcu', True),
            patch.object(backend_module, 'is_cp_cache_layer_split_deepseek_v4_pool', return_value=True),
            patch.dict(backend_module.dsa_use_prefill_cp.__globals__, {'is_dsa_enable_prefill_cp': lambda: True}),
        ):
            self.assertFalse(backend_module.dsa_use_prefill_cp(real_batch))
            backend._prevalidate_cp_request_window_history(real_batch)
        window.prevalidate_history.assert_called_once_with([0, 1])

    def test_history_prevalidation_runs_before_legacy_cp_metadata_is_created(self):
        w = self._history_window()
        backend = backend_module.DeepseekV4AttnBackend.__new__(backend_module.DeepseekV4AttnBackend)
        backend.token_to_kv_pool = SimpleNamespace(
            request_window=w, _swa_global_to_local={1: 0, 4: 1, 7: 2},
        )
        backend.is_dsv41, backend.mtp_enabled = True, False
        backend.enable_decoder_swa_bounded_replay = True
        backend.model_runner = SimpleNamespace(model=SimpleNamespace(model=SimpleNamespace(late_layer_start=5)))
        batch = SimpleNamespace(
            forward_mode=backend_module.ForwardMode.EXTEND,
            attn_cp_metadata=None, encoder_swa_replay=False,
        )
        metadata = SimpleNamespace(core_attn_metadata=SimpleNamespace(request_window_layout=w.layout))
        with (
            patch.object(backend_module, '_is_hcu', True),
            patch.object(backend_module, 'is_cp_cache_layer_split_deepseek_v4_pool', return_value=True),
            patch.object(backend_module, 'maybe_prepare_cp_cache_layer_split_forward'),
            patch.object(backend, '_build_forward_metadata', return_value=metadata),
            patch.object(backend, 'init_forward_metadata_in_graph'),
            patch.object(backend, '_skips_late_layers', return_value=False),
            patch.object(backend, '_build_late_layer_tail_metadata', return_value=None),
            patch.object(window_module, 'get_is_capture_mode', return_value=False),
            patch.object(window_module, '_capturing', return_value=False),
            patch.object(window_module, 'copy_packed_tokens'),
            patch.object(torch, 'equal', wraps=torch.equal) as equal,
        ):
            backend.init_forward_metadata(batch)
            self.assertIsNone(batch.attn_cp_metadata)
            self.assertEqual(w._prevalidated_history_layers, {0, 1})
            # The subsequent model CP split changes query metadata only, then
            # the early layers consume their prevalidated global history.
            batch.attn_cp_metadata = object()
            w.buffer(0)
            w.buffer(1)
            self.assertEqual(equal.call_count, 1)

    def test_history_guard_masks_invalid_positions_and_keeps_missing_history_error(self):
        for mask, missing in [([True, False, True, False], False),
                              ([False] * 4, True),
                              ([True, False, True, False], True)]:
            with self.subTest(mask=mask, missing=missing):
                window = window_module.RequestWindow.__new__(window_module.RequestWindow)
                window.capacity = 4
                window.zero_row = 8
                window.page_size = 1
                window.prepared = None
                window.tags = torch.full((1, 9), -12345, dtype=torch.int64)
                window.tags[0, 1] = 1
                window.tags[0, 6] = -1 if missing else 6
                window.state = SimpleNamespace(kv_buffer=[torch.ones(1)], kv_layout=object())
                window.workspace = SimpleNamespace(kv_buffer=[torch.zeros(1)])
                window.layout = SimpleNamespace(
                    history_req=torch.tensor([0, 0, 1, 1]),
                    history_pos=torch.tensor([1, -20, 6, -9]),
                    history_valid=torch.tensor(mask), history_loc=torch.arange(4),
                )
                with (
                    patch.object(window_module, 'get_is_capture_mode', return_value=False),
                    patch.object(window_module, '_capturing', return_value=False),
                    patch.object(window_module, 'copy_packed_tokens') as copy,
                ):
                    if missing and any(mask):
                        with self.assertRaisesRegex(RuntimeError, 'SWA history is missing'):
                            window.buffer(0)
                        copy.assert_not_called()
                        self.assertIsNone(window.prepared)
                    else:
                        self.assertIs(window.buffer(0), window.workspace.kv_buffer[0])
                        self.assertIs(window.buffer(0), window.workspace.kv_buffer[0])
                        copy.assert_called_once()

    def test_only_supported_request_window_route_is_enabled(self):
        choose = backend_module._request_window_allows_sparse_prefill
        batch = object()
        core = SimpleNamespace(request_window_layout=object())
        with (
            patch.object(backend_module, '_is_hcu', True),
            patch.object(backend_module, 'dsa_use_prefill_cp', autospec=True, return_value=True) as cp,
            patch.object(
                backend_module,
                'is_cp_cache_layer_split_deepseek_v4_pool',
                return_value=True,
            ),
        ):
            self.assertTrue(choose(SimpleNamespace(request_window=object()), batch, core))
            cp.assert_called_once_with(batch)
            self.assertTrue(choose(SimpleNamespace(request_window=None), batch, core))
            self.assertFalse(choose(
                SimpleNamespace(request_window=object()), batch,
                SimpleNamespace(request_window_layout=None),
            ))
            with patch.object(backend_module, 'dsa_use_prefill_cp', return_value=False):
                self.assertFalse(choose(SimpleNamespace(request_window=object()), batch, core))
            with patch.object(backend_module, '_is_hcu', False):
                self.assertFalse(choose(SimpleNamespace(request_window=object()), batch, core))
            with patch.object(
                backend_module, 'is_cp_cache_layer_split_deepseek_v4_pool', return_value=False
            ):
                self.assertFalse(choose(SimpleNamespace(request_window=object()), batch, core))

    def _check_geometry(self, prefixes, extends, *, reset):
        cp_size = 8
        window = 128
        ids = [3 + 4 * r for r in range(len(extends))]
        starts = [sum(extends[:r]) for r in range(len(extends))]
        global_req = torch.tensor([
            ids[r] for r, length in enumerate(extends) for _ in range(length)
        ])
        global_pos = torch.tensor([
            prefixes[r] + q for r, length in enumerate(extends) for q in range(length)
        ])
        floor = torch.tensor([
            prefixes[r] if reset else 0
            for r, length in enumerate(extends) for _ in range(length)
        ])
        layout = window_layout(
            global_req, global_pos, window=window, capacity=256,
            floor=floor, num_groups=len(extends),
        )
        total = sum(extends)
        padded = ((total + cp_size - 1) // cp_size) * cp_size
        padded_pos = torch.full((padded,), -1, dtype=torch.int32)
        padded_pos[:total] = global_pos.to(torch.int32)
        padded_indices = torch.full((padded, window), -1, dtype=torch.int32)
        padded_indices[:total] = layout.indices
        padded_lengths = torch.zeros(padded, dtype=torch.int32)
        padded_lengths[:total] = layout.lengths
        for rank in range(cp_size):
            with self.subTest(rank=rank, prefixes=prefixes, extends=extends, reset=reset):
                local_rows = list(range(rank, padded, cp_size))
                real_lens = [
                    sum(start <= q < start + length for q in local_rows)
                    for start, length in zip(starts, extends)
                ]
                physical_lens = list(real_lens)
                physical_lens[-1] += sum(q >= total for q in local_rows)
                local_pos = padded_pos[rank::cp_size]
                core = SimpleNamespace(
                    request_window_layout=layout,
                    seq_lens_casual=local_pos + 1,
                    swa_page_indices=padded_indices[rank::cp_size],
                    swa_topk_lengths=padded_lengths[rank::cp_size],
                )
                batch = SimpleNamespace(
                    seq_lens_cpu=torch.tensor([p + e for p, e in zip(prefixes, extends)]),
                    seq_lens=torch.tensor([p + e for p, e in zip(prefixes, extends)]),
                    extend_seq_lens=torch.tensor(extends, dtype=torch.int32),
                    extend_seq_lens_cpu=extends,
                    extend_start_loc=torch.tensor(starts, dtype=torch.int32),
                    req_pool_indices=torch.tensor(ids, dtype=torch.int32),
                )
                backend = backend_module.DeepseekV4AttnBackend.__new__(
                    backend_module.DeepseekV4AttnBackend
                )
                backend.forward_metadata = SimpleNamespace(late_layer_tail=None)
                backend.req_to_token = torch.empty((0, 0), dtype=torch.int32)
                backend.token_to_kv_pool = SimpleNamespace(
                    full_to_swa_index_mapping=torch.empty(0, dtype=torch.int32),
                    get_swa_key_page_size=lambda: 256,
                )
                backend.is_dsv41 = True
                with (
                    patch.object(backend_module, '_is_hcu', True),
                    patch.object(backend_module, 'dsa_use_prefill_cp', return_value=True),
                    patch.object(backend_module, 'get_parallel', return_value=SimpleNamespace(
                        attn_cp_size=cp_size, attn_cp_rank=rank,
                    )),
                    patch.object(backend_module, 'build_cp_sparse_query_metadata', return_value=(
                        torch.tensor(physical_lens, dtype=torch.int32),
                        torch.tensor(real_lens, dtype=torch.int32), local_pos,
                    )),
                    patch.object(sparse_prefill_utils, 'combine_topk_swa_indices', side_effect=combine_cpu),
                    patch.object(backend_module, '_use_dsv41_compact_sparse_prefill_workspace', return_value=True),
                ):
                    cache = backend._build_sparse_prefill_chunk_cache(
                        batch, core, num_qo_tokens=len(local_rows)
                    )
                torch.testing.assert_close(cache.swa_token_ids, torch.arange(layout.size, dtype=torch.int32))
                self.assertEqual(cache.compact_seq_lens_cpu, tuple(p + e for p, e in zip(prefixes, extends)))
                torch.testing.assert_close(cache.query_pos, local_pos)
                for local_q, global_q in enumerate(local_rows):
                    expected = []
                    if global_q < total:
                        r = next(r for r, start in enumerate(starts) if start <= global_q < start + extends[r])
                        pos = prefixes[r] + global_q - starts[r]
                        lower = max(0, pos - window + 1, prefixes[r] if reset else 0)
                        for key_pos in range(pos, lower - 1, -1):
                            if key_pos < prefixes[r]:
                                expected.append(r * window + key_pos - (prefixes[r] - window))
                            else:
                                expected.append(len(extends) * window + starts[r] + key_pos - prefixes[r])
                    length = cache.c0_combined_lens[local_q].item()
                    self.assertEqual(length, len(expected))
                    self.assertEqual(cache.c0_combined_indices[local_q, :length].tolist(), expected)
                    self.assertTrue((cache.c0_combined_indices[local_q, length:] == -1).all())
        for page_size in [16, 128, 256]:
            max_pages = (layout.size + page_size - 1) // page_size + 2
            capacity = request_window_page_capacity(layout.size, page_size, max_pages)
            mask = torch.zeros(max_pages, dtype=torch.int32)
            valid = layout.indices[layout.indices >= 0] // page_size
            mask[valid.long()] = 1
            dynamic = torch.nonzero(mask).flatten()
            selected = torch.nonzero_static(mask, size=capacity, fill_value=-1).flatten()
            self.assertEqual(selected[selected >= 0].tolist(), dynamic.tolist())

    def test_global_workspace_with_cp_local_queries_and_unequal_requests(self):
        for reset in [False, True]:
            self._check_geometry([0, 300, 17], [17, 3, 9], reset=reset)

    def test_cp_padding_and_ranks_without_real_queries(self):
        for reset in [False, True]:
            self._check_geometry([0, 128], [1, 2], reset=reset)

    def test_request_window_static_bound_rejects_invalid_geometry(self):
        self.assertEqual(request_window_page_capacity(257, 256, 4), 2)
        for args in [(-1, 256, 4), (128, 0, 4), (128, 256, 0), (1025, 256, 4)]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                request_window_page_capacity(*args)


if __name__ == '__main__':
    unittest.main()
