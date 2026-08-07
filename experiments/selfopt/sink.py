"""``MetricSink``: a thin logging seam so nobody needs an account to run the study.

- :class:`LocalSink` — JSONL + console. The zero-setup default: no key, no network.
- :class:`WandbSink` — opt-in (``SELFOPT_WANDB=1`` + ``WANDB_API_KEY``). Streams the same
  records live and uploads workdir snapshots as artifacts. Never required; offline by default.

Both share one small interface:
  - ``log(record)``            — a structured event (train score, cheating flag, …) to JSONL.
  - ``metric(name, value, step)`` — a scalar on the learning / held-out curve.
  - ``artifact(path, name)``   — register a file/dir (a workdir snapshot, a trace) as an artifact.
  - ``close()``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

from . import config


class MetricSink:
    """No-op base / interface. Subclasses override; callers program to this type."""

    def log(self, record: Dict[str, Any]) -> None: ...
    def metric(self, name: str, value: float, step: Optional[int] = None) -> None: ...
    def artifact(self, path: Path, name: Optional[str] = None, kind: str = "artifact") -> None: ...
    def close(self) -> None: ...


class LocalSink(MetricSink):
    """Append every event to ``<run_dir>/metrics.jsonl`` and echo a one-line summary to the
    console. Artifacts are recorded by path (they already live on disk under the run dir), so
    there is nothing to upload and nothing to configure."""

    def __init__(self, run_dir: Path, echo: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self._echo = echo
        self._fh = self.path.open("a", encoding="utf-8")

    def _write(self, row: Dict[str, Any]) -> None:
        row = {"ts": time.time(), **row}
        self._fh.write(json.dumps(row) + "\n")
        self._fh.flush()

    def log(self, record: Dict[str, Any]) -> None:
        self._write({"kind": "event", **record})
        if self._echo:
            print(f"[selfopt] {json.dumps(record)}", file=sys.stderr)

    def metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self._write({"kind": "metric", "name": name, "value": value, "step": step})
        if self._echo:
            s = "" if step is None else f" @{step}"
            print(f"[selfopt] {name}{s} = {value}", file=sys.stderr)

    def artifact(self, path: Path, name: Optional[str] = None, kind: str = "artifact") -> None:
        self._write(
            {"kind": "artifact", "artifact_kind": kind, "name": name or Path(path).name,
             "path": str(path)}
        )

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


class WandbSink(MetricSink):
    """Mirror everything to Weights & Biases. Lazily imports ``wandb`` and inits a run only
    when actually constructed, so importing this module never pulls wandb in. Falls back to a
    hard error only if you asked for it (``SELFOPT_WANDB=1``) but wandb/key are missing — an
    honest failure beats a silent no-op."""

    def __init__(self, run_dir: Path, run_name: str, project: str = config.WANDB_PROJECT) -> None:
        import wandb  # lazy, optional

        self._wandb = wandb
        self._local = LocalSink(run_dir)  # always keep the offline record too
        self.run = wandb.init(project=project, name=run_name, dir=str(run_dir))

    def log(self, record: Dict[str, Any]) -> None:
        self._local.log(record)
        self.run.log({"event": record})

    def metric(self, name: str, value: float, step: Optional[int] = None) -> None:
        self._local.metric(name, value, step)
        self.run.log({name: value}, step=step)

    def artifact(self, path: Path, name: Optional[str] = None, kind: str = "artifact") -> None:
        self._local.artifact(path, name, kind)
        art = self._wandb.Artifact(name=(name or Path(path).name), type=kind)
        p = Path(path)
        art.add_dir(str(p)) if p.is_dir() else art.add_file(str(p))
        self.run.log_artifact(art)

    def close(self) -> None:
        self._local.close()
        try:
            self.run.finish()
        except Exception:
            pass


def make_sink(run_id: str, use_wandb: Optional[bool] = None) -> MetricSink:
    """Construct the configured sink. LocalSink unless W&B is explicitly enabled *and*
    importable *and* keyed — otherwise LocalSink with a warning (never a hard requirement)."""
    run_dir = config.run_dir(run_id)
    want_wandb = config.USE_WANDB if use_wandb is None else use_wandb
    if not want_wandb:
        return LocalSink(run_dir)
    import os

    if not os.environ.get("WANDB_API_KEY"):
        print("[selfopt] SELFOPT_WANDB set but WANDB_API_KEY absent — using LocalSink.",
              file=sys.stderr)
        return LocalSink(run_dir)
    try:
        return WandbSink(run_dir, run_name=run_id)
    except Exception as exc:  # wandb missing / init failed — degrade, don't crash the study
        print(f"[selfopt] wandb unavailable ({exc}) — using LocalSink.", file=sys.stderr)
        return LocalSink(run_dir)
