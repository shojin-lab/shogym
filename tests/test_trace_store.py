"""The JSONL trace store (RFC 008 §4/§8): one row per step, feedback in wire form."""

from __future__ import annotations

from pathlib import Path

import pytest

from hgym.trace import TraceRecord, append_trace, load_traces, step_record
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
    assert rec.feedback == [{"name": "green_count", "value": 2, "level": "inference", "step": 3}]
    assert rec.terminated is False


def test_append_and_load_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.jsonl"
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
    assert {r["session_id"] for r in rows} == {"s1"}


def test_non_finite_feedback_value_rejected_at_construction() -> None:
    # A NaN/Inf feedback value is caught by the wire validator (_load_item) when the
    # record is built, so it can never reach the file in the first place.
    with pytest.raises(ValueError, match="must be finite"):
        TraceRecord(
            session_id="s1",
            env_name="wordle_v1",
            task_id="7",
            step=1,
            feedback=[{"name": "reward", "value": float("nan"), "level": "episode"}],
        )


def test_inference_step_must_match_row_step() -> None:
    # One row per tool call: an inference item's step must equal the row's step, or the
    # dense signal is misattributed. Enforced in both step_record and direct construction.
    with pytest.raises(ValueError, match="does not match row step"):
        step_record(
            session_id="s1", env_name="e", task_id="7", step=1,
            feedback=[InferenceFeedback(name="dense", value=0.5, step=2)],
        )
    with pytest.raises(ValueError, match="does not match row step"):
        TraceRecord(
            session_id="s1", env_name="e", task_id="7", step=1,
            feedback=[{"name": "dense", "value": 0.5, "level": "inference", "step": 2}],
        )
    # Matching step is fine.
    rec = step_record(
        session_id="s1", env_name="e", task_id="7", step=3,
        feedback=[InferenceFeedback(name="dense", value=0.5, step=3)],
    )
    assert rec.feedback[0]["step"] == 3


def test_append_revalidates_after_mutation(tmp_path: Path) -> None:
    # __post_init__ only guards construction, but TraceRecord is a mutable dataclass;
    # append_trace must re-validate so a post-construction mutation can't be persisted.
    path = tmp_path / "run.jsonl"
    rec = step_record(session_id="s1", env_name="e", task_id="7", step=1, tool="guess")
    rec.terminated = "false"  # mutate a scalar field to an off-wire value
    with pytest.raises(ValueError, match="terminated must be a boolean"):
        append_trace(path, rec)
    rec2 = step_record(session_id="s1", env_name="e", task_id="7", step=2)
    rec2.feedback.append({"name": "r", "value": 1, "level": "typo"})  # mutate the list
    with pytest.raises(ValueError, match="unknown feedback level"):
        append_trace(path, rec2)
    assert not path.exists()


def test_trace_record_enforces_wire_invariant_on_construction() -> None:
    # TraceRecord is public; a directly-built record with any off-wire field must be
    # rejected at construction so it can never be persisted / break parse_meta.
    good = dict(session_id="s1", env_name="wordle_v1", task_id="7", step=1)

    def rec(**over):
        return TraceRecord(**{**good, **over})

    # wire-semantic fields
    with pytest.raises(ValueError, match="unknown feedback level"):
        rec(feedback=[{"name": "x", "value": 1, "level": "typo"}])
    with pytest.raises(ValueError, match="terminated must be a boolean"):
        rec(terminated="false")
    # identifier / step / container fields
    with pytest.raises(ValueError, match="session_id must be a string"):
        rec(session_id=123)
    with pytest.raises(ValueError, match="task_id must be a string or None"):
        rec(task_id=["not-an-id"])
    with pytest.raises(ValueError, match="step must be an int"):
        rec(step=True)  # bool is an int subclass
    # A fully well-formed direct construction is fine.
    ok = rec(feedback=[{"name": "x", "value": 1.0, "level": "episode"}])
    assert ok.terminated is False and ok.step == 1
