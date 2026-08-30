"""Read a version one run directory. Nothing here serves, resumes, or writes one.

The version one serving path is gone. What it left behind is not: a run directory holds
``dispenses.jsonl`` and ``results.jsonl``, one dispense record and one result row per task,
and those records are the whole record of the runs that produced them. So the reader outlives
the writer, and this module is the reader on its own.

It is deliberately a reader and only a reader. There is no writer here, no policy, no stream,
and no way to continue a directory: a version one row cannot say whether feedback reached the
model, so there is no state a version two generation could resume into, and
:mod:`shogym.serve.protocol_v2.rundir` refuses such a directory before anything is claimed.

``read_results`` returns the rows a run wrote. ``reconcile`` pairs them against the dispenses
and reports every task that went out and never came back, which is the only thing a reader has
to reconstruct rather than read.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union

#: The two logs a version one run directory holds.
DISPENSES_FILE = "dispenses.jsonl"
RESULTS_FILE = "results.jsonl"

#: The regime a record that names none was written under: every stream that could have written
#: one revealed nothing, so an absent member reads as the channel that was never opened.
NEVER_REGIME = "never"

Closure = Literal[
    "sealed",
    "aborted",
    "drained",
    "timeout",
    "finalize_error",
    "broker_abort",
]


@dataclass(frozen=True)
class Score:
    """An earned outcome, as the run recorded it.

    ``reward`` and ``success`` are the headline numbers when the env published them. A missing
    one stays ``None`` rather than becoming a zero; ``feedback`` is everything the env emitted,
    verbatim.
    """

    reward: Optional[float]
    success: Optional[bool]
    feedback: List[Dict[str, Any]]

    def to_wire(self) -> Dict[str, Any]:
        """Return this score as the JSON object a row holds."""
        return {
            "reward": self.reward,
            "success": self.success,
            "feedback": [dict(item) for item in self.feedback],
        }


@dataclass(frozen=True)
class ResultRow:
    """One dispensed task's outcome, exactly one per dispense.

    ``score`` is ``None`` unless the closure was earned and the env's headline was readable, so
    aggregating ``score`` can never average in an infrastructure failure. ``observed`` keeps
    every item the env emitted, in wire form, for audit even on an unscored row; it is evidence
    and never a score. ``extensions`` is namespaced provenance, never merged into the fields
    above.

    ``feedback_regime`` names the channel the task was assigned, which is the one thing a reader
    needs to tell an evaluation-grade row from a practice one without joining against anything.
    It is the assignment and never the exposure: the row was durable before any answer was
    composed from it, so it says which channel the task was served under and nothing about
    whether a value reached the caller.
    """

    seq: int
    lease: str
    position: int
    env: str
    task_idx: int
    closure: Closure
    score: Optional[Score]
    observed: List[Dict[str, Any]] = field(default_factory=list)
    diagnostic: Optional[str] = None
    extensions: Dict[str, Any] = field(default_factory=dict)
    feedback_regime: str = NEVER_REGIME

    def to_wire(self) -> Dict[str, Any]:
        """Return this row as the JSON object the log holds."""
        return {
            "seq": self.seq,
            "lease": self.lease,
            "position": self.position,
            "env": self.env,
            "task_idx": self.task_idx,
            "closure": self.closure,
            "score": self.score.to_wire() if self.score is not None else None,
            "observed": [dict(item) for item in self.observed],
            "diagnostic": self.diagnostic,
            "extensions": dict(self.extensions),
            "feedback_regime": self.feedback_regime,
        }

    @classmethod
    def from_wire(cls, row: Dict[str, Any]) -> "ResultRow":
        """Read one stored row back."""
        score = row.get("score")
        return cls(
            seq=int(row["seq"]),
            lease=str(row["lease"]),
            position=int(row["position"]),
            env=str(row["env"]),
            task_idx=int(row["task_idx"]),
            closure=row["closure"],
            score=(
                Score(
                    reward=score.get("reward"),
                    success=score.get("success"),
                    feedback=[dict(item) for item in (score.get("feedback") or [])],
                )
                if isinstance(score, dict)
                else None
            ),
            observed=[dict(item) for item in (row.get("observed") or [])],
            diagnostic=row.get("diagnostic"),
            extensions=dict(row.get("extensions") or {}),
            feedback_regime=recorded_regime(row),
        )


def recorded_regime(record: Mapping[str, Any]) -> str:
    """The feedback regime a stored dispense record or result row says it was written under.

    A record carrying no such member predates the policy, and every stream that could have
    written one revealed nothing, so it reads back as the regime with no channel rather than as
    unknown. Coerced with ``str`` because a stored value that is not text would otherwise decide
    a comparison by its own equality; no non-string renders as ``"never"``, so coercion cannot
    launder a wrong value into a right one.
    """
    return str(record.get("feedback_regime", NEVER_REGIME))


def read_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Every committed record in one of these logs.

    A record is committed once its terminating newline is on disk, and the writer only put the
    newline there once the record itself was. So an unterminated tail is a write that died
    before it returned: nothing was published on the strength of it, and reading it as absent is
    what lets a log survive the crash it exists to record.

    Everything else that will not parse is corruption of a record that did commit, and is raised
    naming the file and the line that holds it. The asymmetry is the point: recovery may skip a
    row nobody was ever told about, and may never quietly skip one somebody was.
    """
    path = Path(path)
    if not path.exists():
        return []
    committed, terminator, _uncommitted = path.read_bytes().rpartition(b"\n")
    records: List[Dict[str, Any]] = []
    for number, line in enumerate((committed + terminator).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except ValueError as exc:  # JSONDecodeError, or a line that is not even UTF-8
            raise ValueError(f"{path} line {number} is not a JSON record: {exc}") from exc
    return records


def read_dispenses(run_dir: Union[str, Path]) -> List[Dict[str, Any]]:
    """Every dispense record written under ``run_dir``."""
    return read_jsonl(Path(run_dir) / DISPENSES_FILE)


def read_results(run_dir: Union[str, Path]) -> List[ResultRow]:
    """Every recorded result row under ``run_dir``."""
    return [ResultRow.from_wire(row) for row in read_jsonl(Path(run_dir) / RESULTS_FILE)]


def reconcile(run_dir: Union[str, Path]) -> List[ResultRow]:
    """Pair dispense records with results and report the unmatched ones.

    A dispense with no result means the run died between handing the task out and sealing it.
    Each unmatched dispense becomes a ``broker_abort`` row with no score, so it can be counted
    but never averaged.

    Provenance survives with it. Whatever each extension observed before the task went out is in
    the dispense record, so the row carries ``extensions[namespace] = {"dispensed": ...}``, the
    half of the span that actually happened and nothing else. There is no ``sealed`` member and
    no ``error`` one, because no finalizer result was ever committed for this dispense and
    inventing either would put a value on the row that no extension produced.

    The regime is taken from the dispense, which is the only place it could have been recorded
    before the crash. Defaulting it instead would make every abandoned task of a practice run
    read back as evaluation-grade, which is the one direction this record may never round in.
    """
    run_dir = Path(run_dir)
    sealed = {row.lease for row in read_results(run_dir)}
    return [
        ResultRow(
            seq=int(record["seq"]),
            lease=str(record["lease"]),
            position=int(record["position"]),
            env=str(record["env"]),
            task_idx=int(record["task_idx"]),
            closure="broker_abort",
            score=None,
            diagnostic="dispensed but never sealed; the stream did not exit in an orderly way",
            extensions={
                namespace: {"dispensed": observed}
                for namespace, observed in dict(record.get("extensions") or {}).items()
            },
            feedback_regime=recorded_regime(record),
        )
        for record in read_dispenses(run_dir)
        if record["lease"] not in sealed
    ]


__all__ = [
    "DISPENSES_FILE",
    "NEVER_REGIME",
    "RESULTS_FILE",
    "Closure",
    "ResultRow",
    "Score",
    "read_dispenses",
    "read_jsonl",
    "read_results",
    "recorded_regime",
    "reconcile",
]
