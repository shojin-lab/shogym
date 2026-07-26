"""The two-arm study orchestrator (issue #57). **This is the maintainer's spend decision.**

It composes the spine into the full study:

    baseline held-out (default self)
        → TREATMENT: one Claude Code process over the train stream (persistent self, "Get Better")
              with held-out checkpoints (start / mid / end) on the evolving self
        → CONTROL: a fresh Claude Code process per train task (no persistence, no instruction)
        → honest report (held-out gain? treatment > control? explained by cheating/memorization?)

By default this only **prints the plan** — running it drives many real Claude Code tasks (real
money). Pass ``--go`` to actually run, and prefer the isolated two-container topology in
``sandbox/`` for the real study (the agent physically cannot reach the broker's memory or the
held-out answers there). The held-out scoring is deterministic + keyless either way.

Nothing here writes a secret; Claude credentials are read from the runtime environment.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import List

from . import config
from .arms import (
    build_control_command,
    build_treatment_command,
    run_claude_stream,
    write_broker_mcp_config,
)
from .broker import public_split
from .heldout import aggregate, evaluate_heldout
from .sink import make_sink
from .snapshot import content_hash
from .split import train_stream


def plan(train_size: int, heldout_size: int) -> dict:
    split = public_split()
    stream = train_stream(split, train_size)
    return {
        "env": config.ENV_NAME,
        "model": config.MODEL,
        "effort": config.EFFORT,
        "public_tasks": split.n,
        "split": {"train_pool": len(split.train), "heldout_pool": len(split.heldout),
                  "seed": split.seed, "heldout_frac": split.heldout_frac},
        "train_stream": len(stream),
        "heldout_eval": heldout_size,
        "arms": ["treatment (persistent self, Get Better, full tools incl web)",
                 "control (fresh context per task, no persistence, no instruction)"],
        "checkpoints": ["start", "mid", "end"],
        "authoritative_heldout": "seal-scored (RFC-009), deterministic, keyless",
        "sinks": ["LocalSink (default)"] + (["WandbSink"] if config.USE_WANDB else []),
    }


def _resolve_checkpoint_selves(prov_dir: Path, self_dir: Path) -> dict:
    """Map ``start`` / ``mid`` / ``end`` to a directory holding that self-version.

    The train run is ONE persistent process; the broker snapshotted the self at every train task
    boundary (``prov_dir/snapshots/<hash>/``). We read the train provenance to find which archived
    snapshot is the self *before the first* task (``start``) and *after ~half* the stream
    (``mid``); ``end`` is the live final self. Probing archived snapshots (never the live self)
    keeps the held-out pass a pure measurement — the training self is only ever read. Falls back
    to the live self_dir when a snapshot is unavailable (e.g. the train run scored nothing)."""
    results = prov_dir / "results.jsonl"
    rows: List[dict] = []
    if results.exists():
        rows = [json.loads(line) for line in results.read_text().splitlines() if line.strip()]
        rows = sorted((r for r in rows if r.get("split") == "train"), key=lambda r: r["seq"])
    snaps = prov_dir / "snapshots"

    def snap(h: object) -> Path | None:
        if not isinstance(h, str):
            return None
        d = snaps / h
        return d if d.exists() else None

    start = (snap(rows[0]["self_hash_before"]) if rows else None) or self_dir
    mid = (snap(rows[len(rows) // 2]["self_hash_before"]) if rows else None) or self_dir
    return {"start": start, "mid": mid, "end": self_dir}


async def run_treatment(run_id: str, stream: List[int], heldout_size: int,
                        real_heldout: bool) -> dict:
    """One persistent Claude Code process over the whole train stream ("Get Better"), then the
    held-out generalization curve at start / mid / end. Each checkpoint probes an *archived
    snapshot* of the self (start = the seed, mid = after ~half the stream, end = the final self),
    so the train stream stays ONE process and the held-out pass never mutates the training self.
    ``real_heldout`` runs a real Claude Code held-out pass; otherwise the keyless StubPolicy."""
    sink = make_sink(f"{run_id}-treatment")
    rd = config.run_dir(f"{run_id}-treatment")
    self_dir = rd / "self"
    self_dir.mkdir(parents=True, exist_ok=True)
    (self_dir / "CLAUDE.md").write_text("# self\n")

    prov = rd / "train"
    mcp = write_broker_mcp_config(rd / ".mcp.json", split="train", prov_dir=prov,
                                  self_dir=self_dir, queue_size=len(stream))
    cmd = build_treatment_command(mcp)
    code = run_claude_stream(cmd, rd / "stream.jsonl", cwd=self_dir)
    train = aggregate(prov / "results.jsonl", split="train")
    sink.log({"event": "treatment_train_done", "exit_code": code, "n": train["n"],
              "mean_reward": train["mean_reward"]})

    # Held-out generalization curve on the evolving self — a probe, never a training signal.
    selves = _resolve_checkpoint_selves(prov, self_dir)
    self_hash_pre_heldout = content_hash(self_dir)  # guard: held-out must not mutate the self
    steps = {"start": 0, "mid": len(stream) // 2, "end": len(stream)}
    curve: dict = {}
    for name in ("start", "mid", "end"):
        agg = await evaluate_heldout(run_dir=rd, checkpoint=name, sink=sink, size=heldout_size,
                                     step=steps[name], real=real_heldout,
                                     self_src=selves[name], arm="treatment")
        curve[name] = {"mean_reward": agg["mean_reward"], "n": agg["n"],
                       "measured_self": agg.get("measured_self")}
    # The subtle correctness point: held-out is a probe, not a training signal. The training self
    # must be byte-identical after the held-out pass (it was only ever read / copied from).
    assert content_hash(self_dir) == self_hash_pre_heldout, \
        "held-out mutated the training self — isolation broken"
    sink.log({"event": "treatment_heldout_curve", "self_unchanged_by_heldout": True,
              "curve": curve})
    sink.close()
    return {"exit_code": code, "train_n": train["n"], "heldout_curve": curve,
            "self_unchanged_by_heldout": True}


async def run_control(run_id: str, stream: List[int], heldout_size: int,
                      real_heldout: bool) -> dict:
    """A fresh Claude Code process per train task — no persistence, no instruction. Baseline,
    plus a held-out baseline: the DEFAULT harness (fresh, no persistent self, no "Get Better")
    over the same held-out tasks, seal-scored — what the treatment's held-out curve is measured
    against."""
    sink = make_sink(f"{run_id}-control")
    rd = config.run_dir(f"{run_id}-control")
    prov = rd / "train"
    for i, idx in enumerate(stream):
        with tempfile.TemporaryDirectory(prefix="selfopt-control-") as work:
            work = Path(work)
            mcp = write_broker_mcp_config(work / ".mcp.json", split="train", prov_dir=prov,
                                          indices=[idx])  # dispense exactly this train task
            cmd = build_control_command(mcp)
            run_claude_stream(cmd, rd / f"stream-{i:03d}.jsonl", cwd=work)  # fresh workdir, discarded
    train = aggregate(prov / "results.jsonl", split="train")
    sink.metric("control/train_mean_reward", train["mean_reward"], step=len(stream))
    # Held-out baseline. Control has no evolving self, so its held-out is checkpoint-invariant
    # (start == end by construction) — measure it once, at "end", for the treatment comparison.
    heldout = await evaluate_heldout(run_dir=rd, checkpoint="end", sink=sink, size=heldout_size,
                                     step=len(stream), real=real_heldout, self_src=None,
                                     arm="control")
    sink.close()
    return {"train_n": train["n"], "train_mean_reward": train["mean_reward"],
            "heldout_baseline": {"mean_reward": heldout["mean_reward"], "n": heldout["n"]}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-size", type=int, default=30)
    ap.add_argument("--heldout-size", type=int, default=20)
    ap.add_argument("--go", action="store_true",
                    help="ACTUALLY run (real Claude Code spend). Default: print the plan only.")
    ap.add_argument("--arm", choices=["treatment", "control", "both"], default="both")
    ap.add_argument("--stub-heldout", action="store_true",
                    help="use the keyless in-process StubPolicy for held-out (cheap dry-run) "
                         "instead of a real Claude Code held-out pass. Default --go = real.")
    args = ap.parse_args()

    real_heldout = not args.stub_heldout
    p = plan(args.train_size, args.heldout_size)
    p["heldout_policy"] = "real Claude Code" if real_heldout else "StubPolicy (keyless dry-run)"
    print(json.dumps({"plan": p}, indent=2))
    if not args.go:
        print("\n[dry-run] This prints the plan only. Re-run with --go to spend on the real "
              "study (many Claude Code tasks). Prefer sandbox/run_sandbox.sh for isolation.")
        return

    run_id = f"study-{int(time.time())}"
    stream = train_stream(public_split(), args.train_size)
    out: dict = {}
    if args.arm in ("treatment", "both"):
        out["treatment"] = asyncio.run(
            run_treatment(run_id, stream, args.heldout_size, real_heldout))
    if args.arm in ("control", "both"):
        out["control"] = asyncio.run(
            run_control(run_id, stream, args.heldout_size, real_heldout))
    print(json.dumps({"result": out}, indent=2))


if __name__ == "__main__":
    main()
