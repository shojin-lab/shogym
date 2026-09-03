"""The retired serving path's run directories, read back after the path that wrote them is gone.

A version one run directory is a record of something that happened, so it outlives the code
that produced it. Nothing here writes one: the fixtures are the bytes such a run left on disk,
written by hand, which is also the only way to write one now.

Whether such a directory can be picked up and continued as a protocol v2 generation is settled
in ``tests/test_protocol_v2_blobs.py``: it cannot, and the refusal happens before anything is
claimed.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shogym.serve.v1_runs import (
    DISPENSES_FILE,
    RESULTS_FILE,
    read_dispenses,
    read_results,
    reconcile,
)

DISPENSES = [
    {"seq": 1, "lease": "L1", "position": 0, "env": "wordle_v1", "task_idx": 0,
     "feedback_regime": "immediate", "extensions": {"tracker": {"run": "abc"}}},
    {"seq": 2, "lease": "L2", "position": 1, "env": "wordle_v1", "task_idx": 1,
     "feedback_regime": "immediate", "extensions": {"tracker": {"run": "def"}}},
]

RESULTS = [
    {"seq": 1, "lease": "L1", "position": 0, "env": "wordle_v1", "task_idx": 0,
     "closure": "sealed",
     "score": {"reward": 1.0, "success": True,
               "feedback": [{"name": "reward", "value": 1.0, "level": "episode"}]},
     "observed": [{"name": "reward", "value": 1.0, "level": "episode"}],
     "diagnostic": None, "extensions": {"tracker": {"dispensed": {"run": "abc"},
                                                    "sealed": {"ok": True}}},
     "feedback_regime": "immediate"},
]


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    """One run directory, as a run that crashed after its second dispense left it."""
    directory = tmp_path / "runs" / "wordle_v1-immediate-20260101T000000Z"
    directory.mkdir(parents=True)
    (directory / DISPENSES_FILE).write_text(
        "".join(json.dumps(row) + "\n" for row in DISPENSES), encoding="utf-8"
    )
    (directory / RESULTS_FILE).write_text(
        "".join(json.dumps(row) + "\n" for row in RESULTS), encoding="utf-8"
    )
    return directory


def test_the_rows_a_finished_run_wrote_read_back_whole(run_dir: Path) -> None:
    """Every field the writer put on a row is a field the reader gives back."""
    rows = read_results(run_dir)
    assert [row.lease for row in rows] == ["L1"]
    row = rows[0]
    assert row.closure == "sealed"
    assert row.score is not None and row.score.reward == 1.0 and row.score.success is True
    assert row.observed == [{"name": "reward", "value": 1.0, "level": "episode"}]
    assert row.extensions["tracker"]["sealed"] == {"ok": True}
    assert row.feedback_regime == "immediate"
    assert row.to_wire() == RESULTS[0]

    assert read_dispenses(run_dir) == DISPENSES


def test_a_dispense_with_no_result_reconciles_to_an_unscored_row(run_dir: Path) -> None:
    """The task that went out and never came back is countable and unaggregatable.

    It carries no score, because nothing earned one, and it keeps the regime the dispense
    recorded rather than defaulting to the one with no channel: a practice run's abandoned task
    may not read back as evaluation-grade.
    """
    abandoned = reconcile(run_dir)
    assert [row.lease for row in abandoned] == ["L2"]
    row = abandoned[0]
    assert row.closure == "broker_abort"
    assert row.score is None
    assert row.feedback_regime == "immediate"
    # The half of the span that happened, and no invented other half.
    assert row.extensions == {"tracker": {"dispensed": {"run": "def"}}}


def test_a_record_that_names_no_regime_reads_as_the_channel_that_was_never_opened(
    tmp_path: Path,
) -> None:
    """A row written before the regime was recorded came from a run that revealed nothing."""
    directory = tmp_path / "old"
    directory.mkdir()
    (directory / RESULTS_FILE).write_text(
        json.dumps(
            {"seq": 1, "lease": "L1", "position": 0, "env": "wordle_v1", "task_idx": 0,
             "closure": "sealed", "score": None}
        )
        + "\n",
        encoding="utf-8",
    )
    assert [row.feedback_regime for row in read_results(directory)] == ["never"]


def test_a_write_that_died_mid_record_costs_only_that_record(tmp_path: Path) -> None:
    """An unterminated tail is a write nobody was told about, so it reads as absent.

    Everything before it committed, and taking those rows down with the last one is the failure
    this rule exists against.
    """
    directory = tmp_path / "killed"
    directory.mkdir()
    (directory / DISPENSES_FILE).write_text(
        json.dumps(DISPENSES[0]) + "\n" + json.dumps(DISPENSES[1])[:20], encoding="utf-8"
    )
    assert read_dispenses(directory) == [DISPENSES[0]]


def test_a_committed_record_that_will_not_parse_is_raised_rather_than_skipped(
    tmp_path: Path,
) -> None:
    """A terminated line that is not a record is corruption of something somebody was told."""
    directory = tmp_path / "corrupt"
    directory.mkdir()
    (directory / DISPENSES_FILE).write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1 is not a JSON record"):
        read_dispenses(directory)


def test_a_directory_with_no_logs_is_an_empty_run_rather_than_an_error(tmp_path: Path) -> None:
    assert read_dispenses(tmp_path) == []
    assert read_results(tmp_path) == []
    assert reconcile(tmp_path) == []
