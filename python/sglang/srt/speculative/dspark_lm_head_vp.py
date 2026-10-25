"""Vocabulary-parallel DSPARK projection with full logits on each DP owner.

Unlike the EAGLE top-1 path, this retains the complete proposal distribution.
Hidden rows are gathered, each rank projects its vocabulary slice, and an
all-to-all returns only the requesting rank's rows. Markov correction and
sampling then run unchanged on the reconstructed logits.
"""

from __future__ import annotations

import logging

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


class DSparkLMHeadVocabParallel:
    def __init__(
        self,
        *,
        full_weight: torch.Tensor,
        vocab_size: int,
        vp_size: int,
        max_rows_per_rank: int,
        local_world_size: int,
        process_group=None,
    ) -> None:
        if vp_size <= 1 or max_rows_per_rank <= 0:
            raise ValueError(
                "DSPARK VP requires vp_size > 1 and positive row capacity."
            )
        if full_weight.ndim != 2 or full_weight.shape[0] != vocab_size:
            raise ValueError("DSPARK VP requires a full, unpadded 2-D LM-head weight.")
        if full_weight.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            raise ValueError(
                "DSPARK VP requires an unquantized floating-point LM-head."
            )
        if vocab_size % vp_size:
            raise ValueError("The vocabulary size must be divisible by DSPARK VP size.")
        if process_group is None:
            from sglang.srt.distributed.parallel_state import (
                create_custom_parallel_group,
            )
            from sglang.srt.runtime_context import get_parallel
            from sglang.srt.speculative.draft_lm_head_vp import (
                build_node_local_vp_groups,
            )

            tp_group = get_parallel().tp_group
            groups = build_node_local_vp_groups(
                tp_group.ranks, vp_size, local_world_size
            )
            rank = dist.get_rank()
            ranks = next(ranks for ranks in groups if rank in ranks)
            process_group = create_custom_parallel_group(
                ranks, backend=dist.get_backend(tp_group.device_group)
            )
        if process_group is None or dist.get_world_size(process_group) != vp_size:
            raise ValueError("DSPARK VP process group does not match vp_size.")

        self.process_group = process_group
        self.vp_size = vp_size
        self.vp_rank = dist.get_rank(process_group)
        self.max_rows_per_rank = max_rows_per_rank
        self.vocab_size = vocab_size
        self.hidden_size = full_weight.shape[1]
        self.shard_size = vocab_size // vp_size
        self.dtype = full_weight.dtype
        self.device = full_weight.device
        # A contiguous transpose also avoids the HCU strided-transpose GEMM
        # issue handled by the existing EAGLE VP implementation.
        self.weight_shard_t = full_weight.narrow(
            0, self.vp_rank * self.shard_size, self.shard_size
        ).T.contiguous()
        options = dict(device=self.device, dtype=self.dtype)
        total_rows = max_rows_per_rank * vp_size
        self.local_hidden = torch.empty(
            (max_rows_per_rank, self.hidden_size), **options
        )
        self.gathered_hidden = torch.empty((total_rows, self.hidden_size), **options)
        self.shard_logits = torch.empty((total_rows, self.shard_size), **options)
        self.received_logits = torch.empty_like(self.shard_logits)
        self.full_logits = torch.empty((max_rows_per_rank, vocab_size), **options)
        self.row_count = torch.empty((), dtype=torch.int64, device=self.device)

        # Warm both collective types before capture and memory accounting.
        warm_send = torch.zeros(vp_size, device=self.device, dtype=self.dtype)
        warm_recv = torch.empty_like(warm_send)
        dist.all_gather_into_tensor(warm_recv, warm_send[:1], group=self.process_group)
        dist.all_to_all_single(warm_recv, warm_send, group=self.process_group)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        if self.vp_rank == 0:
            logger.info(
                "DSPARK draft LM-head VP enabled: size=%d, vocab_shard=%d, "
                "max_rows=%d; full proposal logits retained for greedy and sampling.",
                vp_size,
                self.shard_size,
                max_rows_per_rank,
            )

    @torch.no_grad()
    def project_logits(
        self, hidden_states: torch.Tensor, *, padded_rows: int | None = None
    ) -> torch.Tensor:
        """Return local full-vocabulary logits, including for uneven/idle DP ranks.

        Graph callers supply their common capture bucket. Eager callers agree
        on the largest local row count. Every VP rank must call, including idle
        ranks with zero rows. Returned storage is reused on the next call.
        """
        if hidden_states.ndim != 2 or hidden_states.shape[1] != self.hidden_size:
            raise ValueError("DSPARK VP hidden states have an unexpected shape.")
        rows = hidden_states.shape[0]
        if hidden_states.device != self.device:
            raise ValueError("DSPARK VP hidden states changed device.")
        if padded_rows is None:
            if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "DSPARK VP capture requires a common padded row count."
                )
            self.row_count.fill_(rows)
            dist.all_reduce(
                self.row_count, op=dist.ReduceOp.MAX, group=self.process_group
            )
            padded_rows = int(self.row_count.item())
        if not rows <= padded_rows <= self.max_rows_per_rank:
            raise ValueError(
                f"DSPARK VP rows={rows}, padded={padded_rows} exceed "
                f"capacity={self.max_rows_per_rank}."
            )
        if padded_rows == 0:
            return self.full_logits[:0]
        local = self.local_hidden[:padded_rows]
        local[:rows].copy_(hidden_states)
        local[rows:].zero_()
        total = padded_rows * self.vp_size
        gathered = self.gathered_hidden[:total]
        partial = self.shard_logits[:total]
        received = self.received_logits[:total]
        dist.all_gather_into_tensor(gathered, local, group=self.process_group)
        torch.mm(gathered, self.weight_shard_t, out=partial)
        dist.all_to_all_single(received, partial, group=self.process_group)
        self.full_logits[:padded_rows].view(
            padded_rows, self.vp_size, self.shard_size
        ).copy_(
            received.view(self.vp_size, padded_rows, self.shard_size).permute(1, 0, 2)
        )
        return self.full_logits[:rows]
