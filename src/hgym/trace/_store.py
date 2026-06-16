"""The trace store (RFC 007, the observability surface): append-only JSONL where each
line is one rollout tagged with the harness hashes that produced it.

JSONL is the whole format — zero infra, one object per line, append-safe across
processes, and trivially loadable into a dataframe (``pd.DataFrame(load_rows(path))``).
Each record carries the combined ``harness_hash`` and the per-surface sub-hashes, so a
sweep can ``groupby`` any surface to attribute a reward delta to the surface that moved.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from hgym.harness import Harness, harness_hash, surface_hashes


@dataclass
class TraceRecord:
    """One rollout's observability row: which harness, on which env, to what reward."""

    harness_hash: str
    surface_hashes: Dict[str, str]
    env_name: str
    reward: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


def record_for(
    harness: Harness,
    env_name: str,
    *,
    reward: Optional[float] = None,
    metrics: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> TraceRecord:
    """Build a :class:`TraceRecord`, stamping the harness's combined + per-surface
    hashes so the row is attributable without re-deriving them later."""
    return TraceRecord(
        harness_hash=harness_hash(harness),
        surface_hashes=surface_hashes(harness),
        env_name=env_name,
        reward=reward,
        metrics=dict(metrics or {}),
        extra=dict(extra or {}),
    )


def append_trace(path: Union[str, Path], record: TraceRecord) -> None:
    """Append one record to the JSONL store at ``path`` (created if absent)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(asdict(record), sort_keys=True)
    with p.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_traces(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read the JSONL store into a list of dicts (blank lines skipped)."""
    p = Path(path)
    rows: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def flatten_record(row: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a record's nested hashes/metrics into top-level columns, so a list of
    these is a clean dataframe: ``surface_hash_inference``, ``metric_<k>``, ..."""
    flat: Dict[str, Any] = {
        "harness_hash": row.get("harness_hash"),
        "env_name": row.get("env_name"),
        "reward": row.get("reward"),
    }
    for surface, value in (row.get("surface_hashes") or {}).items():
        flat[f"surface_hash_{surface}"] = value
    for key, value in (row.get("metrics") or {}).items():
        flat[f"metric_{key}"] = value
    return flat


def load_rows(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """:func:`load_traces` with every record flattened — drop-in for a dataframe."""
    return [flatten_record(row) for row in load_traces(path)]
