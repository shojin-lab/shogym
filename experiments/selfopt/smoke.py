"""Smoke test — prove the loop end-to-end, cheaply. Does NOT run the study.

Default (keyless, deterministic, no model spend, no Docker):
    uv run python -m experiments.selfopt.smoke

  1. broker splits the 600 public tasks into disjoint train / held-out (asserts no overlap);
  2. a train task is dispensed (no index/target leaks in the framing);
  3. a model-free StubPolicy plays it and the broker seal-scores it authoritatively;
  4. the workdir ("the self") is snapshotted at the task boundary (content hash + archive);
  5. the held-out set is scored authoritatively via the seal (keyless);
  6. LocalSink records the train score, held-out curve, cheating flags, and the snapshot artifact.

Optional real single Claude Code train task (spends a little; needs `claude` + auth):
    uv run python -m experiments.selfopt.smoke --real
It runs ONE real train task through the broker and captures the full stream-json trace. If the
CLI/auth is missing or the run fails, the smoke still passes on the mocked spine and says so.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import time
from pathlib import Path

from . import config
from .arms import build_treatment_command, run_claude_stream, write_broker_mcp_config
from .broker import Broker, public_split
from .heldout import aggregate, evaluate_heldout
from .policy import StubPolicy
from .sink import make_sink
from .snapshot import content_hash


def _seed_self(self_dir: Path) -> None:
    """Seed a minimal workdir so the snapshotter has something to hash/archive."""
    self_dir.mkdir(parents=True, exist_ok=True)
    (self_dir / "CLAUDE.md").write_text("# self\n\nNotes the agent could evolve.\n")


async def mocked_spine(run_id: str) -> dict:
    """The guaranteed-green core: broker split → dispense → stub play → seal score → snapshot →
    held-out seal eval → LocalSink. No model, no key, no Docker."""
    sink = make_sink(run_id)
    rd = config.run_dir(run_id)
    report: dict = {"run_id": run_id, "parts": {}}

    # (1) Split integrity.
    split = public_split()
    assert set(split.train).isdisjoint(split.heldout)
    report["parts"]["split"] = {
        "n": split.n, "train": len(split.train), "heldout": len(split.heldout),
        "disjoint": True,
    }
    sink.log({"event": "split", "n": split.n, "train": len(split.train),
              "heldout": len(split.heldout)})

    # (2)+(3)+(4) Dispense one train task, play it with the stub, seal-score it, snapshot the self.
    self_dir = rd / "self"
    _seed_self(self_dir)
    train_prov = rd / "train"
    broker = Broker("train", queue_size=1, prov_dir=train_prov, self_dir=self_dir)
    framing = await broker.get_task()
    # No index / target / handle leaks in what the agent sees.
    for forbidden in ("task_idx", "task_id", "idx", "handle", "target", "answer"):
        assert forbidden not in framing, f"framing leaked {forbidden!r}"
    tool_names = [t["name"] for t in framing["tools"]]
    stub = StubPolicy()
    await stub.play(framing, broker)
    # Drain so the last task finalizes + snapshots.
    assert (await broker.get_task()).get("done")
    train_rows = [
        json.loads(line)
        for line in (train_prov / "results.jsonl").read_text().splitlines()
        if line.strip()
    ]
    assert len(train_rows) == 1, train_rows
    row = train_rows[0]
    sink.metric("train/reward", row["reward"], step=0)
    sink.log({"event": "train_task", "reward": row["reward"], "success": row["success"],
              "self_hash_before": row["self_hash_before"], "self_hash_after": row["self_hash_after"]})
    # The snapshot archive exists and matches the recorded hash.
    snap_dir = train_prov / "snapshots" / row["self_hash_before"]
    assert snap_dir.exists(), "workdir snapshot was not archived"
    assert content_hash(self_dir) == row["self_hash_after"]
    sink.artifact(snap_dir, name="self-snapshot", kind="workdir")
    report["parts"]["train_dispense_and_score"] = {
        "tools": tool_names, "reward": row["reward"], "success": row["success"],
        "authoritative_feedback_keys": list(row["feedback"].keys()),
        "snapshot_archived": True,
    }

    # (5)+(6) Authoritative held-out eval on a couple of tasks, keyless, via the seal.
    agg = await evaluate_heldout(StubPolicy(), run_dir=rd, checkpoint="start", sink=sink,
                                 size=2, step=0)
    report["parts"]["heldout_authoritative"] = {
        "n": agg["n"], "mean_reward": agg["mean_reward"], "success_rate": agg["success_rate"],
        "keyless": True,
    }

    sink.close()
    report["parts"]["metric_sink"] = {"path": str(rd / "metrics.jsonl"),
                                      "exists": (rd / "metrics.jsonl").exists()}
    return report


def real_single_task(run_id: str) -> dict:
    """Attempt ONE real Claude Code train task through the broker, capturing the full stream-json
    trace. Honest about auth/CLI availability; never writes a secret."""
    rd = config.run_dir(run_id)
    out: dict = {"attempted": True}
    if shutil.which("claude") is None:
        return {**out, "ran": False, "reason": "no `claude` CLI on PATH"}
    if not (os.environ.get("CLAUDE_CODE_OAUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")):
        return {**out, "ran": False,
                "reason": "no CLAUDE_CODE_OAUTH_TOKEN / ANTHROPIC_API_KEY in env (runtime-only)"}

    work = rd / "real_treatment"
    work.mkdir(parents=True, exist_ok=True)
    _seed_self(work)
    prov = rd / "real_train"
    mcp = write_broker_mcp_config(work / ".mcp.json", split="train", prov_dir=prov,
                                  self_dir=work, queue_size=1)
    cmd = build_treatment_command(mcp)
    stream_path = rd / "real_stream.jsonl"
    t0 = time.time()
    code = run_claude_stream(cmd, stream_path, cwd=work)
    dur = time.time() - t0
    results = prov / "results.jsonl"
    scored = results.exists() and results.read_text().strip()
    agg = aggregate(results, split="train") if scored else {"n": 0}
    events = sum(1 for _ in stream_path.open()) if stream_path.exists() else 0
    return {
        **out, "ran": True, "exit_code": code, "seconds": round(dur, 1),
        "stream_events": events, "stream_path": str(stream_path),
        "train_scored": bool(scored), "train_aggregate": agg,
    }


async def _amain(real: bool) -> None:
    run_id = f"smoke-{int(time.time())}"
    report = await mocked_spine(run_id)
    report["real"] = real_single_task(run_id + "-real") if real else {"attempted": False}

    print("\n================ SMOKE REPORT ================")
    print(json.dumps(report, indent=2))
    print("=============================================")
    p = report["parts"]
    print("\nSummary:")
    print(f"  split: disjoint train={p['split']['train']} / heldout={p['split']['heldout']} "
          f"of {p['split']['n']}  [REAL]")
    print(f"  train dispense+seal-score: reward={p['train_dispense_and_score']['reward']} "
          f"feedback={p['train_dispense_and_score']['authoritative_feedback_keys']}  [REAL scoring, MOCK policy]")
    print(f"  workdir snapshot archived: {p['train_dispense_and_score']['snapshot_archived']}  [REAL]")
    print(f"  held-out seal eval: n={p['heldout_authoritative']['n']} "
          f"mean_reward={p['heldout_authoritative']['mean_reward']}  [REAL scoring, MOCK policy, keyless]")
    print(f"  metric sink jsonl: {p['metric_sink']['exists']}  [REAL]")
    if real:
        r = report["real"]
        if r.get("ran"):
            print(f"  REAL Claude Code 1-task: exit={r['exit_code']} events={r['stream_events']} "
                  f"scored={r['train_scored']} agg={r.get('train_aggregate')}  [REAL model run]")
        else:
            print(f"  REAL Claude Code 1-task: SKIPPED ({r['reason']})  [MOCK only]")
    print("\nNOTE: the StubPolicy is a model-free stand-in for Claude Code; it exercises the exact\n"
          "broker/seal/snapshot/sink plumbing but does not actually solve tasks (scores near 0).\n"
          "The scoring, split, snapshot, and sink are REAL. Full two-arm study is deferred.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true",
                    help="also attempt ONE real Claude Code train task (spends a little)")
    args = ap.parse_args()
    asyncio.run(_amain(args.real))


if __name__ == "__main__":
    main()
