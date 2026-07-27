"""Smoke test for cell #2 (Codex) — prove the loop end-to-end through the Codex plumbing, cheaply.
Does NOT run the study.

Default (keyless, deterministic, no model spend, no Docker):
    uv run python -m experiments.selfopt.smoke_codex

  A. the shared broker/seal/snapshot/sink spine (reuses cell #1's ``mocked_spine`` — the exact
     same StubPolicy-driven core: split → dispense → seal-score → snapshot → held-out eval → sink);
  B. the Codex plumbing, keyless: the ``config.toml`` writer emits valid TOML carrying the
     curriculum as an HTTP MCP server + the ``web_search`` toggle + the model; the command builder
     emits a ``codex exec --json -s danger-full-access`` invocation; the run-env strips the billed
     ``OPENAI_API_KEY``.

Optional real single Codex train task (spends a little; needs `codex` + ChatGPT-subscription auth):
    uv run python -m experiments.selfopt.smoke_codex --real
It runs ONE real ``codex exec`` train task against a LOCAL (no-Docker) HTTP broker and captures the
full ``--json`` trace. No Docker is touched (safe alongside a live containerized run). If the CLI /
subscription auth / env provisioning is missing, the smoke still passes on the mocked spine and
says so — and it NEVER uses the billed ``OPENAI_API_KEY``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import tomllib
from pathlib import Path

from . import config
from .codex_arms import (
    CODEX_MODEL,
    MCP_PATH,
    TREATMENT_PROMPT,
    build_codex_command,
    codex_run_env,
    prepare_codex_home,
    run_codex_stream,
    write_codex_config,
)
from .smoke import mocked_spine


def codex_plumbing_checks(run_id: str) -> dict:
    """Keyless assertions on the Codex harness adapter (no CLI, no spend)."""
    rd = config.run_dir(run_id)
    cfg = write_codex_config(rd / "config.toml", broker_url="http://broker:9000/mcp/",
                             web_search=True, model=CODEX_MODEL, effort="low")
    parsed = tomllib.loads(cfg.read_text())
    assert parsed["mcp_servers"]["curriculum"]["url"] == "http://broker:9000/mcp/", parsed
    assert parsed["tools"]["web_search"] is True, parsed
    assert parsed["model"] == CODEX_MODEL, parsed
    # Reasoning summaries surfaced into the --json trace (the thinking-visibility lever).
    assert parsed["model_reasoning_summary"] == "detailed", parsed
    # web-off variant flips exactly the one lever
    off = tomllib.loads(write_codex_config(rd / "config_off.toml",
                                           broker_url="http://broker:9000/mcp/",
                                           web_search=False).read_text())
    assert off["tools"]["web_search"] is False

    cmd = build_codex_command(TREATMENT_PROMPT, cwd=rd)
    assert cmd[:4] == ["codex", "exec", "--json", "-s"], cmd
    assert "danger-full-access" in cmd and "--skip-git-repo-check" in cmd, cmd
    assert cmd[-1] == TREATMENT_PROMPT, cmd

    env = codex_run_env({"OPENAI_API_KEY": "sk-BILLED", "PATH": "/usr/bin"}, codex_home=rd)
    assert "OPENAI_API_KEY" not in env, "billed key must be stripped from the codex run env"
    assert env["CODEX_HOME"] == str(rd)
    return {"config_toml_valid": True, "http_mcp_curriculum": True,
            "web_toggle": {"on": True, "off": False}, "reasoning_summary": "detailed",
            "billed_key_stripped": True, "command": cmd[:4] + ["…", "<prompt>"]}


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port(port: int, tries: int = 60) -> bool:
    for _ in range(tries):
        try:
            socket.create_connection(("127.0.0.1", port), 1).close()
            time.sleep(4.0)  # let FastMCP settle past the open socket before the MCP handshake
            return True
        except OSError:
            time.sleep(1.0)
    return False


def real_single_task(run_id: str) -> dict:
    """ONE real ``codex exec`` train task against a LOCAL HTTP broker (no Docker), full trace
    captured. Honest about CLI/auth/env availability; subscription-only; never writes a secret to
    a tracked file."""
    out: dict = {"attempted": True}
    if shutil.which("codex") is None:
        return {**out, "ran": False, "reason": "no `codex` CLI on PATH"}
    auth = Path.home() / ".codex" / "auth.json"
    if not auth.exists():
        return {**out, "ran": False, "reason": "no ~/.codex/auth.json — run `codex login` (ChatGPT)"}

    rd = config.run_dir(run_id)
    work = rd / "real_treatment"
    work.mkdir(parents=True, exist_ok=True)
    (work / "AGENTS.md").write_text("# self\n")
    prov = rd / "real_train"
    port = _free_port()

    # Start a LOCAL broker (1 train task) over HTTP MCP — no Docker, so nothing can collide with a
    # live containerized run. Provisioning AutomationBench needs the cached upstream source.
    broker_env = {**os.environ, "SELFOPT_HTTP": "1", "SELFOPT_HOST": "127.0.0.1",
                  "SELFOPT_PORT": str(port), "SELFOPT_SPLIT": "train", "SELFOPT_QUEUE_SIZE": "1",
                  "SELFOPT_PROV_DIR": str(prov), "SELFOPT_SELF_DIR": str(work),
                  "SELFOPT_RUN_NAME": run_id}
    broker_env.pop("OPENAI_API_KEY", None)
    broker_log = (rd / "real_broker.log").open("w")
    broker = subprocess.Popen([sys.executable, "-m", "experiments.selfopt.broker"],
                              stdout=broker_log, stderr=subprocess.STDOUT, env=broker_env,
                              cwd=str(config.PKG_DIR.parent.parent))
    try:
        if not _wait_port(port):
            return {**out, "ran": False, "reason": "local broker never bound its port "
                    "(AutomationBench provisioning?) — see real_broker.log"}
        cfg_home = rd / "codexhome"
        prepare_codex_home(cfg_home, broker_url=f"http://127.0.0.1:{port}{MCP_PATH}",
                           web_search=True, effort="low")  # low effort keeps the smoke cheap
        cmd = build_codex_command(TREATMENT_PROMPT, cwd=work)
        stream_path = rd / "real_stream.jsonl"
        t0 = time.time()
        code = run_codex_stream(cmd, stream_path, codex_home=cfg_home, cwd=work)
        dur = time.time() - t0
    finally:
        broker.terminate()
        try:
            broker.wait(timeout=10)
        except subprocess.TimeoutExpired:
            broker.kill()
        broker_log.close()

    results = prov / "results.jsonl"
    scored = results.exists() and results.read_text().strip()
    events = sum(1 for _ in stream_path.open()) if stream_path.exists() else 0
    agg = None
    if scored:
        from .heldout import aggregate
        agg = aggregate(results, split="train")
    return {**out, "ran": True, "exit_code": code, "seconds": round(dur, 1),
            "stream_events": events, "stream_path": str(stream_path),
            "train_scored": bool(scored), "train_aggregate": agg,
            "subscription_only": True}


async def _amain(real: bool) -> None:
    run_id = f"smoke-codex-{int(time.time())}"
    report = await mocked_spine(run_id)                     # A. shared spine (StubPolicy)
    report["codex_plumbing"] = codex_plumbing_checks(run_id)  # B. Codex adapter, keyless
    report["real"] = real_single_task(run_id + "-real") if real else {"attempted": False}

    print("\n============= SMOKE REPORT (cell #2 / Codex) =============")
    print(json.dumps(report, indent=2))
    print("=========================================================")
    p = report["parts"]
    print("\nSummary:")
    print(f"  split: disjoint train={p['split']['train']} / heldout={p['split']['heldout']} "
          f"of {p['split']['n']}  [REAL]")
    print(f"  train dispense+seal-score: reward={p['train_dispense_and_score']['reward']} "
          f"feedback={p['train_dispense_and_score']['authoritative_feedback_keys']}  "
          f"[REAL scoring, MOCK policy]")
    print(f"  workdir snapshot archived: {p['train_dispense_and_score']['snapshot_archived']}  [REAL]")
    print(f"  held-out seal eval: n={p['heldout_authoritative']['n']} "
          f"mean_reward={p['heldout_authoritative']['mean_reward']}  [REAL scoring, MOCK, keyless]")
    c = report["codex_plumbing"]
    print(f"  codex plumbing: http-MCP curriculum={c['http_mcp_curriculum']} "
          f"web-toggle={c['web_toggle']} billed-key-stripped={c['billed_key_stripped']}  [REAL]")
    if real:
        r = report["real"]
        if r.get("ran"):
            print(f"  REAL Codex 1-task: exit={r['exit_code']} events={r['stream_events']} "
                  f"scored={r['train_scored']} agg={r.get('train_aggregate')}  [REAL model run]")
        else:
            print(f"  REAL Codex 1-task: SKIPPED ({r['reason']})  [MOCK only]")
    print("\nNOTE: the StubPolicy is a model-free stand-in for Codex; it exercises the exact\n"
          "broker/seal/snapshot/sink plumbing but does not solve tasks (scores near 0). The\n"
          "scoring, split, snapshot, sink, and the Codex config/command builders are REAL.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", action="store_true",
                    help="also attempt ONE real Codex train task via a local HTTP broker (spends)")
    args = ap.parse_args()
    asyncio.run(_amain(args.real))


if __name__ == "__main__":
    main()
