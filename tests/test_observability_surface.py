"""The observability surface (RFC 007): per-surface hashing and the JSONL trace store.

The property that makes attribution work: flipping one surface moves exactly that
surface's sub-hash and the combined hash, and nothing else.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hgym.harness import Harness, harness_hash, surface_hashes
from hgym.mcp.types import MCPServerSpec
from hgym.trace import append_trace, load_rows, load_traces, record_for


def _harness(**overrides) -> Harness:
    base = dict(
        model="openai/gpt-5.4-nano",
        inference_params={"temperature": 0.7, "max_tokens": 256},
        system_template="You are playing wordle. {{ value }}",
        extra_specs=[],
        horizon=6,
    )
    base.update(overrides)
    return Harness(**base)


def test_hash_is_deterministic() -> None:
    assert harness_hash(_harness()) == harness_hash(_harness())
    assert surface_hashes(_harness()) == surface_hashes(_harness())


def test_inference_params_order_independent() -> None:
    a = _harness(inference_params={"temperature": 0.7, "max_tokens": 256})
    b = _harness(inference_params={"max_tokens": 256, "temperature": 0.7})
    assert harness_hash(a) == harness_hash(b)


def _assert_only_surface_changed(a: Harness, b: Harness, changed: str) -> None:
    ha, hb = surface_hashes(a), surface_hashes(b)
    assert ha[changed] != hb[changed], f"{changed} sub-hash should move"
    for surface in ha:
        if surface != changed:
            assert ha[surface] == hb[surface], f"{surface} must stay fixed"
    assert harness_hash(a) != harness_hash(b)


def test_changing_model_moves_only_inference() -> None:
    _assert_only_surface_changed(
        _harness(), _harness(model="openai/gpt-5.4"), "inference"
    )


def test_changing_template_moves_only_instruction() -> None:
    _assert_only_surface_changed(
        _harness(), _harness(system_template="Solve it. {{ value }}"), "instruction"
    )


def test_adding_extra_moves_only_tool() -> None:
    extra = MCPServerSpec(
        name="think", transport="stdio", command=["python", "-m", "think_mcp"]
    )
    _assert_only_surface_changed(_harness(), _harness(extra_specs=[extra]), "tool")


def test_changing_horizon_moves_only_control() -> None:
    _assert_only_surface_changed(_harness(), _harness(horizon=3), "control")


def test_none_template_hashes_stably() -> None:
    h = _harness(system_template=None)
    assert harness_hash(h) == harness_hash(replace(h))


def test_trace_store_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "traces" / "run.jsonl"
    h = _harness()
    append_trace(path, record_for(h, "wordle_v1", reward=1.0, metrics={"guesses": 3}))
    append_trace(path, record_for(h, "wordle_v1", reward=0.0, metrics={"guesses": 6}))

    rows = load_traces(path)
    assert len(rows) == 2
    assert rows[0]["harness_hash"] == harness_hash(h)
    assert rows[0]["surface_hashes"]["instruction"] == surface_hashes(h)["instruction"]
    assert rows[0]["reward"] == 1.0


def test_load_rows_flattens_for_dataframe(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    h = _harness()
    append_trace(path, record_for(h, "wordle_v1", reward=1.0, metrics={"guesses": 3}))

    (row,) = load_rows(path)
    assert row["harness_hash"] == harness_hash(h)
    assert row["surface_hash_inference"] == surface_hashes(h)["inference"]
    assert row["metric_guesses"] == 3
    assert row["reward"] == 1.0
