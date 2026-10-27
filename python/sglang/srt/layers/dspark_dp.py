"""The decode-only DSpark/DeepEP path that does not exchange scheduler rows."""


def use_local_dspark_dp_batches() -> bool:
    # Read the scoped runtime configuration, rather than caching a draft scope.
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_model,
        get_parallel,
        get_spec,
    )
    from sglang.srt.speculative.ragged_verify import (
        RaggedVerifyMode,
        read_ragged_verify_mode,
    )
    from sglang.srt.utils.common import is_hcu

    if not envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get() or not is_hcu():
        return False
    p, spec, moe = get_parallel(), get_spec(), get_exec().moe
    algorithm = getattr(spec.speculative_algorithm, "name", spec.speculative_algorithm)
    return (
        algorithm == "DSPARK"
        and get_disagg().disaggregation_mode == "decode"
        and spec.speculative_draft_model_path == get_model().model_path
        and read_ragged_verify_mode() is RaggedVerifyMode.STATIC
        and p.attn_dp_enabled
        and p.attn_dp_size == p.tp_size
        and p.attn_tp_size == p.attn_cp_size == p.attn_dcp_size == p.pp_size == 1
        and p.num_dp_ranks == p.attn_dp_size > 1
        and moe.moe_a2a_backend == "deepep"
        and moe.deepep_mode == "low_latency"
        and spec.speculative_moe_a2a_backend in (None, "deepep")
        and p.enable_dp_lm_head
        and not get_exec().overlap.enable_two_batch_overlap
    )
