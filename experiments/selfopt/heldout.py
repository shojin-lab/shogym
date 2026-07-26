"""The authoritative held-out evaluation — the honest generalization curve.

At each checkpoint (start / mid / end) we run the **current self** over the held-out pool (the
private-set proxy that never appears in the train stream) and score it **authoritatively through
the seal**: the broker owns the :class:`ServedEpisode`, AutomationBench's ``finalize`` scores the
sealed ``WorldState`` and returns core-owned evidence, and the broker reads the number off that
evidence. The agent (the policy) only emits tool calls — it can neither reach nor forge the
verdict. Because AutomationBench's scoring is deterministic + offline, this is reproducible and
needs **no OpenAI key**.

Two policy paths, one scoring path:
  - in-process (this module): drive a :class:`Broker` (``split="heldout"``) with an in-proc
    :class:`Policy` (e.g. :class:`StubPolicy`) — keyless, deterministic, used by the smoke test.
  - real Claude Code (see :mod:`.arms`): the current self is mounted into a container and
    ``claude`` plays a held-out HTTP broker; the same broker seal-scores it. :func:`aggregate`
    reads that broker's results file the same way.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .broker import Broker, public_split
from .policy import Policy, StubPolicy
from .sink import MetricSink
from .snapshot import content_hash, copy_tree


def aggregate(results_path: Path, split: str = "heldout") -> Dict[str, Any]:
    """Mean reward + success-rate over the held-out result rows a broker recorded."""
    rows: List[Dict[str, Any]] = []
    if Path(results_path).exists():
        for line in Path(results_path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("split") == split:
                rows.append(r)
    n = len(rows)
    mean_reward = sum(r["reward"] for r in rows) / n if n else 0.0
    success_rate = sum(1 for r in rows if r["success"]) / n if n else 0.0
    return {"n": n, "mean_reward": mean_reward, "success_rate": success_rate, "rows": rows}


def _heldout_indices(size: Optional[int]) -> List[int]:
    """The held-out pool indices (a cheaper checkpoint caps to the first ``size``)."""
    pool = list(public_split().heldout)
    return pool[:size] if size else pool


async def _run_real_heldout(
    *,
    run_dir: Path,
    checkpoint: str,
    prov_dir: Path,
    size: Optional[int],
    self_src: Optional[Path],
    arm: str,
    sink: MetricSink,
) -> Dict[str, Any]:
    """Drive a REAL Claude Code pass over the held-out split, seal-scored by the broker.

    One **fresh process per held-out task** so each task is played by an *identical* copy of the
    checkpoint self — no in-context carryover between held-out tasks, and (crucially) no held-out
    learning: the treatment probe gets a **throwaway copy** of ``self_src`` as its workdir, so any
    write it makes is discarded and NEVER reaches the training self. The control probe gets a
    fresh empty workdir (no persistent self, no "Get Better") — the baseline.
    """
    # Imported lazily: heldout.py is the keyless in-process default; the real path pulls the
    # Claude Code runners only when actually asked to spend.
    from .arms import (
        build_control_command,
        build_heldout_command,
        run_claude_stream,
        write_broker_mcp_config,
    )

    indices = _heldout_indices(size)
    stream_dir = Path(run_dir) / "heldout" / checkpoint
    stream_dir.mkdir(parents=True, exist_ok=True)

    measured_self: Optional[str] = None
    if self_src is not None and Path(self_src).exists():
        measured_self = content_hash(Path(self_src))  # which self-version this checkpoint measured
        sink.log({"event": "heldout_self_version", "checkpoint": checkpoint, "arm": arm,
                  "self_hash": measured_self})

    for i, idx in enumerate(indices):
        with tempfile.TemporaryDirectory(prefix=f"selfopt-heldout-{arm}-") as tmp:
            work = Path(tmp)
            if self_src is not None:
                # Throwaway COPY of the self: plays with its skills/memory; writes stay here.
                copy_tree(Path(self_src), work)
                mcp = write_broker_mcp_config(work / ".mcp.json", split="heldout",
                                              prov_dir=prov_dir, indices=[idx])
                cmd = build_heldout_command(mcp)
            else:
                mcp = write_broker_mcp_config(work / ".mcp.json", split="heldout",
                                              prov_dir=prov_dir, indices=[idx])
                cmd = build_control_command(mcp)
            run_claude_stream(cmd, stream_dir / f"stream-{i:03d}.jsonl", cwd=work)

    agg = aggregate(prov_dir / "results.jsonl")
    agg["measured_self"] = measured_self
    return agg


async def evaluate_heldout(
    policy: Optional[Policy] = None,
    *,
    run_dir: Path,
    checkpoint: str,
    sink: MetricSink,
    size: Optional[int] = None,
    step: Optional[int] = None,
    real: bool = False,
    self_src: Optional[Path] = None,
    arm: str = "treatment",
) -> Dict[str, Any]:
    """Score the held-out pool at ``checkpoint``, seal-scored, and log the generalization curve.

    Two policy paths, one authoritative scoring path:
      - ``real=False`` (default): drive an in-process :class:`Broker` with ``policy`` (a
        :class:`StubPolicy` if none given) — keyless, deterministic, the cheap smoke path.
      - ``real=True``: a **real Claude Code** pass over the held-out split (see
        :func:`_run_real_heldout`). ``self_src`` is the self-version to probe — a directory the
        treatment arm's probe gets a throwaway copy of (its evolved skills/memory); pass ``None``
        for the control arm (fresh, no persistent self). Either way the broker seal-scores it.

    ``checkpoint`` labels the point on the curve (``"start"``/``"mid"``/``"end"``); ``size`` caps
    how many held-out tasks to score. Returns the aggregate.
    """
    prov_dir = Path(run_dir) / "heldout" / checkpoint
    if real:
        agg = await _run_real_heldout(run_dir=run_dir, checkpoint=checkpoint, prov_dir=prov_dir,
                                      size=size, self_src=self_src, arm=arm, sink=sink)
    else:
        broker = Broker("heldout", queue_size=size, prov_dir=prov_dir)
        policy = policy if policy is not None else StubPolicy()
        while True:
            framing = await broker.get_task()
            if framing.get("done"):
                break
            await policy.play(framing, broker)
        agg = aggregate(broker.results_path)
    sink.metric("heldout/mean_reward", agg["mean_reward"], step=step)
    sink.metric("heldout/success_rate", agg["success_rate"], step=step)
    sink.log({"event": "heldout_checkpoint", "checkpoint": checkpoint, "arm": arm, "real": real,
              "n": agg["n"], "mean_reward": agg["mean_reward"],
              "success_rate": agg["success_rate"], "measured_self": agg.get("measured_self")})
    if (prov_dir / "results.jsonl").exists():
        sink.artifact(prov_dir / "results.jsonl", name=f"heldout-{checkpoint}", kind="results")
    return agg
