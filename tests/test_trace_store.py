"""The JSONL trace store (RFC 008 §4/§8): one row per step, feedback in wire form."""

from __future__ import annotations

from pathlib import Path

from hgym.trace import append_trace, load_traces, step_record
from hgym.types import EpisodeFeedback, InferenceFeedback


def test_step_record_serializes_feedback_to_wire_form() -> None:
    rec = step_record(
        session_id="s1",
        env_name="wordle_v1",
        task_id="7",
        step=3,
        tool="guess",
        feedback=[InferenceFeedback(name="green_count", value=2, step=3)],
    )
    assert rec.tool == "guess"
    assert rec.feedback == [
        {"name": "green_count", "value": 2, "level": "inference", "step": 3}
    ]
    assert rec.terminated is False


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.jsonl"  # parent dir created on write
    append_trace(
        path,
        step_record(session_id="s1", env_name="wordle_v1", task_id="7", step=1, tool="guess"),
    )
    append_trace(
        path,
        step_record(
            session_id="s1",
            env_name="wordle_v1",
            task_id="7",
            step=2,
            tool="terminate",
            feedback=[EpisodeFeedback(name="solved", value=1.0)],
            terminated=True,
        ),
    )

    rows = load_traces(path)
    assert len(rows) == 2
    assert rows[0]["tool"] == "guess" and rows[0]["terminated"] is False
    assert rows[1]["terminated"] is True
    assert rows[1]["feedback"][0] == {"name": "solved", "value": 1.0, "level": "episode"}
    # Every row carries the session/env/task identity for group-by attribution.
    assert {r["session_id"] for r in rows} == {"s1"}
