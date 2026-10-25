"""Entry-copy shape and padding contracts, without loading model weights."""

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


@pytest.fixture(scope="module")
def helpers():
    source = (
        Path(__file__).resolve().parents[5] / "python/sglang/srt/models/deepseek_v4.py"
    )
    tree = ast.parse(source.read_text())
    names = {"_can_defer_mhc_repeat_cp_split", "_repeat_mhc_input_on_cp_rank"}
    tree.body = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    scope = {"torch": torch, "_get_mhc_repeat_cp_ops": lambda: None}
    exec(
        compile(tree, str(source), "exec", flags=__future__.annotations.compiler_flag),
        scope,
    )
    return (
        scope["_can_defer_mhc_repeat_cp_split"],
        scope["_repeat_mhc_input_on_cp_rank"],
    )


@pytest.mark.parametrize(
    "rows,hidden,hc,cp",
    [(0, 11, 2, 3), (13, 7, 3, 5), (3, 17, 5, 8), (137, 513, 7, 1)],
)
def test_padded_and_ragged_physical_rows(helpers, rows, hidden, hc, cp):
    gate, repeat = helpers
    value = torch.arange(rows * hidden).reshape(rows, hidden)
    batch = SimpleNamespace(
        input_ids=torch.empty(rows),
        positions=torch.empty(rows),
        out_cache_loc=torch.empty(rows),
        extend_seq_lens_cpu=[max(0, rows - 3)],
        _original_num_tokens=max(0, rows - 3),
    )
    for rank in range(cp):
        assert gate(value, batch, hc, cp, rank)
        actual = repeat(value, hc, cp, rank)
        expected = value[rank::cp].unsqueeze(1).repeat(1, hc, 1)
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)
    batch.positions = torch.empty(rows + 1)
    assert not gate(value, batch, hc, cp, 0)
