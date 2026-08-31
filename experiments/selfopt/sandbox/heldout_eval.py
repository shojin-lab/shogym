"""Standalone, parallel held-out evaluator for an ALREADY-EXISTING treatment run.

The training study (``study.py``) grows a persistent self over a stream of tasks; the self's real
self-improvement surface is the memory/knowledge-base the CLI writes to ``~/.claude``
(``projects/<slug>/memory/*.md``), NOT the inert workdir ``self/``. This tool measures how much
that accumulated self generalizes, by replaying a checkpoint of it over the held-out split — WITHOUT
re-training. It reads an existing run's provenance, resolves each checkpoint's PAIR of (workdir
self, filtered memory home) at the same task boundary, mounts them into a fresh throwaway agent per
held-out task, and seal-scores the pass authoritatively through the broker.

It reuses the training machinery verbatim (``run_agent`` / ``start_broker`` / ``resolve_checkpoints``
/ ``build_filtered_home`` / ``_heldout_pass`` / ``aggregate``) — so the eval and the training arms
can never drift. It NEVER mutates the training self: the workdir self and the home are only copied
from (throwaway per-unit copies), and both the workdir-self hash and the memory-home hash are
asserted unchanged afterwards.

Held-out runs WEB+BASH-off (the held-out split is public — deny answer-lookup via web AND
curl-via-Bash; tasks are solved through the ``tasks`` MCP ``api_*`` tools, so denying Bash costs
nothing). The (checkpoint × task) units are fully independent and run through a bounded worker pool.

    # plan only (no spend):
    uv run python experiments/selfopt/sandbox/heldout_eval.py --run sandbox-1785221622
    # the real eval (spend) for begin+end over 40 held-out tasks, 4 in parallel:
    uv run python experiments/selfopt/sandbox/heldout_eval.py --run sandbox-1785221622 --go \
        --checkpoints start,end --n 40 --effort low --concurrency 4
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import List, Optional

# Run as a script (sys.path[0] is this dir); put the repo root on the path so the shared experiment
# code — the SAME prompts / split / snapshot / docker plumbing the training arms use — is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.selfopt import config  # noqa: E402
from experiments.selfopt.arms import HELDOUT_PROMPT, _HELDOUT_OFF  # noqa: E402
from experiments.selfopt.broker import public_split  # noqa: E402
from experiments.selfopt.sandbox.study import (  # noqa: E402
    HeldoutIncomplete,
    _heldout_pass,
    build_images,
    ensure_network,
    read_session_id,
    resolve_checkpoints,
)
from experiments.selfopt.snapshot import content_hash, home_skip  # noqa: E402

CHECKPOINTS = ("start", "mid", "end", "end-context", "end-context-nomem")

# ``end-context`` is the CONTEXT-LOADED probe: same final self as ``end``, but the agent boots by
# ``--resume``-ing the training session, so it carries the original IN-CONTEXT HISTORY as well as
# the memory. ``end`` (fresh context, memory only) measures the DURABLE artifact; the gap between
# them is how much of the learning lived in the conversation and would be lost on a restart.
_CONTEXT_CP = "end-context"
# Same resumed conversation, but the memory FILES withheld — completes the 2x2 (context x memory)
# and isolates what the on-disk knowledge base adds once the conversation is already loaded.
_CONTEXT_NOMEM_CP = "end-context-nomem"
_CONTEXT_CPS = (_CONTEXT_CP, _CONTEXT_NOMEM_CP)


def detect_server_name(stream_path: Path, default: str = "tasks") -> str:
    """The MCP server name THIS run was trained against, read off its trace (tools appear as
    ``mcp__<name>__get_task``). A resumed probe must be given the same name: an agent whose recorded
    context is full of ``mcp__curriculum__*`` reads a differently-named server as its tools having
    disconnected, and stops without playing the task."""
    if stream_path.exists():
        m = re.search(r"mcp__([a-z0-9_]+)__get_task",
                      stream_path.read_text(encoding="utf-8", errors="ignore")[:2_000_000])
        if m:
            return m.group(1)
    return default


def plan(run: str, arm: str, checkpoints: List[str], n: int, cps: dict) -> dict:
    return {
        "run": run, "arm": arm, "checkpoints": checkpoints, "n_heldout": n,
        "model": config.MODEL, "effort": config.EFFORT,
        "held_out_tools": "web OFF + Bash OFF (public split — deny lookup and curl-the-answer)",
        "measures": {cp: {"self": cps[cp]["self_hash"], "home": cps[cp]["home_hash"]}
                     for cp in checkpoints},
        "isolation": "throwaway copies only; training self/home asserted unchanged after",
    }


def run_eval(*, run: str, arm: str, checkpoints: List[str], n: int, concurrency: int,
             oauth: str, wandb_key: Optional[str], project: str) -> dict:
    """Resolve the run's checkpoints, run the held-out pass for each (bounded-parallel), roll up a
    per-checkpoint summary + the begin→end delta, and write it under ``<arm>/heldout_eval/``. The
    training self (workdir) AND memory home are asserted byte-identical before/after — this is a
    measurement, never a training signal.

    A checkpoint whose pass cannot account for every requested unit stops the eval: the scored units
    remain on disk (re-run the same command to retry only the failures) and the only thing written
    is ``summary.incomplete.json``, never a summary that could be mistaken for a finished arm."""
    root = config.RUNS_DIR / run
    rd = root / arm
    self_dir = rd / "self"
    self_home = rd / "self_home"
    prov = rd / "prov"
    if not self_dir.exists():
        raise SystemExit(f"BLOCKED: no training self at {self_dir} — is {run!r}/{arm} a real run?")

    cps = resolve_checkpoints(prov, self_dir, self_home)
    # The context arm replays the FINAL self, but resumed into the training conversation.
    cps[_CONTEXT_CP] = dict(cps["end"])
    cps[_CONTEXT_NOMEM_CP] = dict(cps["end"])
    session_id = read_session_id(root, rd / "stream.jsonl")
    server_name = detect_server_name(rd / "stream.jsonl")
    indices = list(public_split().heldout)[:n]
    out_root = rd / "heldout_eval"

    pre_self = content_hash(self_dir)
    pre_home = content_hash(self_home, skip=home_skip) if self_home.exists() else None

    # A distinct run-scope for THIS eval's containers so they never collide with a concurrently
    # running training study's brokers (which are scoped by the bare run id).
    scope = f"{run}-hoeval"
    summary: dict = {}
    incomplete: Optional[HeldoutIncomplete] = None
    for cp in checkpoints:
        info = cps[cp]
        how = ("CONTEXT-LOADED, memory withheld" if cp == _CONTEXT_NOMEM_CP else
               "CONTEXT-LOADED (--resume)" if cp == _CONTEXT_CP else "fresh context")
        print(f"==> held-out {cp}: probe self {info['self_hash']} + home {info['home_hash']} "
              f"[{how}] over {len(indices)} tasks (<= {concurrency} parallel)", flush=True)
        ctx = cp in _CONTEXT_CPS
        if ctx and not session_id:
            raise SystemExit(f"BLOCKED: {run!r} has no recoverable session id — "
                             f"cannot run the context-loaded probe.")
        try:
            agg = _heldout_pass(run_id=scope, arm=arm, cp=cp, indices=indices,
                                src=info["self"], home_src=info["home"],
                                stream_dir=out_root / cp, disallowed=_HELDOUT_OFF,
                                system=HELDOUT_PROMPT, oauth=oauth, wandb_key=wandb_key,
                                project=project, concurrency=concurrency,
                                resume=session_id if ctx else None, keep_context=ctx,
                                drop_memory=cp == _CONTEXT_NOMEM_CP, server_name=server_name)
        except HeldoutIncomplete as exc:
            # Stop here: an arm that cannot account for every requested unit gets NO summary.json.
            # The scored units stay on disk, so the same command re-run retries only the failures.
            incomplete = exc
            print(f"    {cp}: INCOMPLETE — {exc.scored}/{exc.requested} units scored", flush=True)
            break
        summary[cp] = {"n": agg["n"], "mean_reward": agg["mean_reward"],
                       "success_rate": agg["success_rate"],
                       "measured_self": info["self_hash"], "measured_home": info["home_hash"]}
        print(f"    {cp}: n={agg['n']} mean_reward={agg['mean_reward']:.4f}", flush=True)

    assert content_hash(self_dir) == pre_self, "eval mutated the training self — isolation broken"
    if pre_home is not None:
        assert content_hash(self_home, skip=home_skip) == pre_home, \
            "eval mutated the training memory home — isolation broken"

    out_root.mkdir(parents=True, exist_ok=True)
    marker = out_root / "summary.incomplete.json"
    if incomplete is not None:
        # The only artifact a partial arm leaves: an explicitly-incomplete record naming the dead
        # units. No summary.json, so nothing here can be read as a finished measurement.
        marker.write_text(json.dumps(
            {"run": run, "arm": arm, "n": n, "complete": False,
             "failed_checkpoint": incomplete.cp, "requested": incomplete.requested,
             "scored": incomplete.scored, "failures": incomplete.failures,
             "completed_checkpoints": summary}, indent=2))
        raise SystemExit(f"BLOCKED: {incomplete}\n  wrote {marker} — NO summary.json for a partial "
                         f"arm; re-run the same command to finish it.")
    marker.unlink(missing_ok=True)  # a previous partial attempt was completed by this pass

    result: dict = {"run": run, "arm": arm, "n": n, "checkpoints": summary,
                    "self_unchanged": True}
    if "start" in summary and "end" in summary:
        result["delta_start_to_end"] = summary["end"]["mean_reward"] - summary["start"]["mean_reward"]
    (out_root / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="existing run id (e.g. sandbox-1785221622)")
    ap.add_argument("--arm", default="treatment", help="which arm's self to probe")
    ap.add_argument("--n", type=int, default=40, help="held-out tasks (first N of the pool)")
    ap.add_argument("--checkpoints", default="start,end",
                    help="comma list, subset of start,mid,end,end-context "
                         "(end-context = resume the training session: memory AND in-context history)")
    ap.add_argument("--effort", default="low")
    ap.add_argument("--model", default="claude-opus-5")
    ap.add_argument("--concurrency", type=int, default=4, help="max parallel (checkpoint x task)")
    ap.add_argument("--go", action="store_true", help="ACTUALLY run (real spend). Else: plan.")
    ap.add_argument("--build", action="store_true", help="(re)build the broker + agent images")
    ap.add_argument("--wandb", action="store_true",
                    help="stream broker-side metrics live to W&B (needs WANDB_API_KEY in env)")
    ap.add_argument("--project", default=config.WANDB_PROJECT)
    args = ap.parse_args()

    # Effort/model are read off ``config`` at call time by ``run_agent``; override the module
    # attributes so the CLI flags take effect without touching the training arms.
    config.MODEL = args.model
    config.EFFORT = args.effort

    checkpoints = [c.strip() for c in args.checkpoints.split(",") if c.strip()]
    bad = [c for c in checkpoints if c not in CHECKPOINTS]
    if bad:
        raise SystemExit(f"BLOCKED: unknown checkpoints {bad} (want a subset of {list(CHECKPOINTS)})")

    root = config.RUNS_DIR / args.run
    rd = root / args.arm
    cps = resolve_checkpoints(rd / "prov", rd / "self", rd / "self_home")
    cps[_CONTEXT_CP] = dict(cps["end"])  # same final self, replayed inside the training session
    cps[_CONTEXT_NOMEM_CP] = dict(cps["end"])
    p = plan(args.run, args.arm, checkpoints, args.n, cps)
    print(json.dumps({"plan": p}, indent=2))
    if not args.go:
        print("\n[dry-run] plan only. Re-run with --go to spend on the held-out eval.")
        return

    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if not oauth:
        raise SystemExit("BLOCKED: no CLAUDE_CODE_OAUTH_TOKEN in env (runtime-only credential).")
    wandb_key = os.environ.get("WANDB_API_KEY") if args.wandb else None
    if args.wandb and not wandb_key:
        print("[heldout_eval] --wandb set but WANDB_API_KEY absent — broker falls back to LocalSink.",
              file=sys.stderr)

    ensure_network()
    if args.build:
        build_images()
    out = run_eval(run=args.run, arm=args.arm, checkpoints=checkpoints, n=args.n,
                   concurrency=args.concurrency, oauth=oauth, wandb_key=wandb_key,
                   project=args.project)
    print(json.dumps({"result": out}, indent=2))


if __name__ == "__main__":
    main()
