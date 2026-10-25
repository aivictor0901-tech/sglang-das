"""CPU contracts for optional CP output dispatch and bounded-replay selection."""

import ast
from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("cp,local", [(1, 3), (3, 7), (8, 3)])
@pytest.mark.parametrize("trailing", [(13,), (3, 17)])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("tail", [False, True])
def test_cp_output_dispatch_and_tail_order(monkeypatch, cp, local, trailing, enabled, tail):
    repo = Path(__file__).resolve().parents[5]
    source = repo / "python/sglang/srt/layers/utils/hcu_cp_utils.py"
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in {"_interleave_cp_output", "cp_all_gather_rerange_output"}]
    namespace = {"torch": torch, "use_symmetric_memory": lambda *a, **k: nullcontext(),
                 "get_parallel": lambda: SimpleNamespace(attn_cp_group=None),
                 "is_allocation_symmetric": lambda: False}
    exec(compile(tree, str(source), "exec"), namespace)
    dsa = ModuleType("sglang.srt.layers.attention.dsa.utils")
    dsa.is_dsa_prefill_cp_round_robin_split = lambda: True
    monkeypatch.setitem(sys.modules, dsa.__name__, dsa)
    native = ModuleType("lightop.attention.cp_output")
    calls = []

    def mutate(data, group_size):
        calls.append(tuple(data.shape))
        result = data.reshape(group_size, local, *trailing).transpose(0, 1).reshape_as(data).clone()
        data.copy_(result)
        return data

    native.interleave_cp_output_ = mutate
    monkeypatch.setitem(sys.modules, native.__name__, native)
    monkeypatch.setenv("SGLANG_DSV4_CP_OUTPUT_INPLACE", "1" if enabled else "0")
    elements = cp * local
    for dimension in trailing:
        elements *= dimension
    full = torch.arange(elements).reshape(cp * local, *trailing)
    namespace["attn_cp_all_gather_into_tensor"] = lambda output, value: output.copy_(full)
    indices = torch.tensor([cp * local - 1, 0, min(local, cp * local - 1)]) if tail else None
    batch = SimpleNamespace(attn_cp_metadata=SimpleNamespace(gather_index=indices))
    result = namespace["cp_all_gather_rerange_output"](full[:local], cp, batch, None)
    expected = full[indices] if tail else full.reshape(cp, local, *trailing).transpose(0, 1).reshape_as(full)
    assert torch.equal(result, expected)
    assert len(calls) == int(enabled and not tail)
