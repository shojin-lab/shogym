"""The full two-arm + held-out study, run UNDER the isolated two-container sandbox (issue #57).

This is the credible launch vehicle — NOT the local ``run_claude_stream`` path. Every Claude Code
pass runs in the **agent container** (full tools; NO mount of the broker's filesystem) and talks
to an **isolated broker container** (targets + train/held-out split + provenance on a broker-only
volume) over HTTP MCP. The current task's target and the held-out answers are physically
unreachable from the agent — integrity is the environment's job, not an allow-list's.

Phases (all against the isolated broker):
  - TREATMENT train : ONE persistent agent over the train stream ("Get Better"), full tools incl
                      web, self-dir RW. The broker snapshots the self at every task boundary.
  - TREATMENT held-out : at start / mid / end, a fresh agent per held-out task plays a THROWAWAY
                      COPY of that checkpoint's archived self — **web-off** (held-out answers are
                      online) — so held-out probes capability and never mutates the training self.
  - CONTROL train + held-out : a fresh agent per task, no persistent self, no "Get Better",
                      curriculum tools only — the baseline the treatment curve is measured against.

Broker-side metrics stream LIVE to Weights & Biases when ``WANDB_API_KEY`` is in the env
(``--wandb``); with no key it degrades to the offline LocalSink with no error. The agent
container never gets a wandb key or a broker mount — network egress to wandb.ai is the broker's
alone; volume isolation is untouched.

Creds are RUNTIME-only: supply ``CLAUDE_CODE_OAUTH_TOKEN`` (and optionally ``WANDB_API_KEY``) in
the shell env; they are passed via ``-e`` at run time and never written to a tracked file.

    uv run python experiments/selfopt/sandbox/study.py --plan          # dry-run: the phase plan
    uv run python experiments/selfopt/sandbox/study.py --go --build \
        --arm both --train-size 2 --heldout-size 2 --wandb            # the real run (spend)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import List, Optional

# Run as a script (sys.path[0] is this dir); put the repo root on the path so the shared
# experiment code — the SAME prompts / tool-policies / split / snapshot the local path uses — is
# importable. Reusing them is what guarantees the sandbox and local arms never drift.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.selfopt import config  # noqa: E402
from experiments.selfopt.arms import (  # noqa: E402
    CONTINUE_PROMPT,
    CONTROL_PROMPT,
    HELDOUT_PROMPT,
    KICKOFF,
    TREATMENT_PROMPT,
    _WEB_OFF,
)
from experiments.selfopt.broker import public_split  # noqa: E402
from experiments.selfopt.heldout import aggregate  # noqa: E402
from experiments.selfopt.snapshot import content_hash, copy_tree  # noqa: E402
from experiments.selfopt.split import train_stream  # noqa: E402

NET = "selfopt-net"
BROKER_IMG = "selfopt-ab-broker:latest"
AGENT_IMG = "selfopt-ab-agent:latest"
SANDBOX = Path(__file__).resolve().parent


# --- docker plumbing ------------------------------------------------------------------

def _run(cmd: List[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, **kw)


def _rm(*names: str) -> None:
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True)


def ensure_network() -> None:
    if subprocess.run(["docker", "network", "inspect", NET], capture_output=True).returncode != 0:
        _run(["docker", "network", "create", NET], capture_output=True)


def build_images() -> None:
    print("==> building images (broker + agent)", flush=True)
    _run(["docker", "build", "-q", "-f", str(SANDBOX / "broker.Dockerfile"),
          "-t", BROKER_IMG, str(_REPO_ROOT)], stdout=subprocess.DEVNULL)
    _run(["docker", "build", "-q", "-f", str(SANDBOX / "agent.Dockerfile"),
          "-t", AGENT_IMG, str(_REPO_ROOT)], stdout=subprocess.DEVNULL)


def start_broker(name: str, *, split: str, prov: Path, run_name: str,
                 self_ro: Optional[Path] = None, home_ro: Optional[Path] = None,
                 queue_size: Optional[int] = None,
                 indices: Optional[List[int]] = None, wandb_key: Optional[str] = None,
                 project: str, resume: bool = False) -> None:
    """Start an ISOLATED broker container and wait until its HTTP MCP port is listening.

    The broker holds the targets + the split + provenance on ``/provenance`` (a broker-ONLY
    volume). ``wandb_key`` (RUNTIME -e only) turns on live streaming; without it the broker's
    LocalSink writes to the same provenance volume — no crash, no network."""
    _rm(name)
    prov.mkdir(parents=True, exist_ok=True)
    args = ["docker", "run", "-d", "--name", name, "--network", NET,
            "-e", f"SELFOPT_SPLIT={split}", "-e", f"SELFOPT_RUN_NAME={run_name}",
            "-e", f"SELFOPT_SPLIT_SEED={config.SPLIT_SEED}",
            "-v", f"{prov}:/provenance"]
    if self_ro is not None:
        args += ["-v", f"{self_ro}:/self:ro"]
    if home_ro is not None:
        # The agent's native Claude Code home (~/.claude), mounted read-only so the broker can
        # snapshot its durable self-surface (memory/skills) at each task boundary — the OTHER half
        # of "the self", alongside /self.
        home_ro.mkdir(parents=True, exist_ok=True)
        args += ["-v", f"{home_ro}:/self_home:ro", "-e", "SELFOPT_HOME_DIR=/self_home"]
    if queue_size is not None:
        args += ["-e", f"SELFOPT_QUEUE_SIZE={queue_size}"]
    if indices is not None:
        args += ["-e", "SELFOPT_INDICES=" + ",".join(str(i) for i in indices)]
    if resume:
        # Rebuild the SAME seeded stream but dispense from the next UNscored task (the broker reads
        # the already-scored train rows off the persisted provenance volume and skips them).
        args += ["-e", "SELFOPT_RESUME=1"]
    if wandb_key:
        args += ["-e", "SELFOPT_WANDB=1", "-e", f"WANDB_API_KEY={wandb_key}",
                 "-e", f"SELFOPT_WANDB_PROJECT={project}"]
    args.append(BROKER_IMG)
    _run(args, stdout=subprocess.DEVNULL)
    _wait_listening(name)


def _wait_listening(name: str, tries: int = 40, settle: float = 6.0) -> None:
    probe = "import socket; socket.create_connection(('localhost', 9000), 1).close()"
    for _ in range(tries):
        if subprocess.run(["docker", "exec", name, "python", "-c", probe],
                          capture_output=True).returncode == 0:
            # The socket opens before FastMCP is ready to serve /mcp/; settle so the FIRST agent's
            # MCP handshake doesn't race the server (an unready broker => claude exits empty).
            time.sleep(settle)
            return
        time.sleep(1.0)
    logs = subprocess.run(["docker", "logs", "--tail", "30", name],
                          capture_output=True, text=True)
    raise RuntimeError(f"broker {name} never came up:\n{logs.stdout}\n{logs.stderr}")


def run_agent(*, work: Path, prompt: str, oauth: str,
              stream_path: Path, broker_name: str, home: Optional[Path] = None,
              session_id: Optional[str] = None, resume: Optional[str] = None,
              append: bool = False, disallowed: Optional[List[str]] = None,
              append_system_prompt: Optional[str] = None) -> int:
    """Run ONE agent-container Claude Code pass to completion, teeing its stream-json trace.

    ``work`` is the agent's /work (RW): the persistent self for treatment-train, a throwaway self
    copy for a held-out probe, or an empty dir for control. ``home`` (when given) is a host dir
    mounted at ``/root/.claude`` (the container's ``$HOME/.claude``) so the CLI's native
    memory/skills/settings PERSIST across the single persistent ``claude -p`` session and are
    snapshottable — the OTHER half of "the self". The broker's ``.mcp.json`` is written to a
    SEPARATE config dir mounted read-only at ``/cfg`` (never inside ``/work`` — so it never enters
    a snapshotted self, and there is no nested-file bind mount to fail on virtiofs). The agent gets
    NO mount of the broker's provenance/answer volume — that is the isolation.

    ``session_id`` pins the Claude Code session id on the INITIAL run (``--session-id``), so it is
    known up front and persisted before the agent starts. ``resume`` continues THAT session on a
    resumed run (``--resume <id>``): because the session transcript lives in the mounted
    ``~/.claude`` (``home``) under the same ``/work`` cwd, the whole conversation — the in-context
    learning from the tasks already played — carries across the separate container invocations.
    ``append`` tees onto the end of ``stream_path`` so a resume chain reads back as ONE trace."""
    work.mkdir(parents=True, exist_ok=True)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = stream_path.parent / f".cfg-{stream_path.stem}"  # sibling of work, outside /work
    cfg.mkdir(parents=True, exist_ok=True)
    (cfg / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {"curriculum": {"type": "http",
                                       "url": f"http://{broker_name}:9000/mcp/"}}}))
    # bypassPermissions: the container is the isolation boundary, so the full self-surface —
    # including writes to ~/.claude (the agent's own config, its self-improvement surface) — is
    # auto-approved; --forward-subagent-text captures Task subagents' text/thinking in the trace;
    # web-off arms enforce it with a --disallowedTools deny rule (which applies under bypass).
    claude = ["claude", "-p", prompt, "--model", config.MODEL, "--effort", config.EFFORT,
              "--mcp-config", "/cfg/.mcp.json", "--strict-mcp-config",
              "--permission-mode", "bypassPermissions", "--forward-subagent-text",
              "--output-format", "stream-json", "--verbose", "--include-partial-messages"]
    if append_system_prompt:
        claude += ["--append-system-prompt", append_system_prompt]
    if disallowed:
        claude += ["--disallowedTools", *disallowed]
    if resume is not None:
        claude += ["--resume", resume]
    elif session_id is not None:
        claude += ["--session-id", session_id]
    docker = ["docker", "run", "--rm", "--network", NET,
              "-e", f"CLAUDE_CODE_OAUTH_TOKEN={oauth}",
              "-v", f"{work}:/work:rw", "-v", f"{cfg}:/cfg:ro"]
    if home is not None:
        home.mkdir(parents=True, exist_ok=True)
        docker += ["-v", f"{home}:/root/.claude:rw"]
    docker += ["-w", "/work", AGENT_IMG] + claude
    err_path = stream_path.with_suffix(".err.txt")
    mode = "a" if append else "w"
    with stream_path.open(mode, encoding="utf-8") as out, err_path.open(mode) as err:
        return subprocess.run(docker, stdout=out, stderr=err).returncode


# --- self-version resolution (from the treatment train provenance) --------------------

def resolve_checkpoint_selves(prov: Path, self_dir: Path) -> dict:
    """start = the seed self (before task 1), mid = after ~half the stream, end = the final self —
    each an ARCHIVED snapshot dir the broker wrote to the provenance volume (falls back to the
    live self_dir if the train run scored nothing). Probing archived snapshots keeps held-out a
    pure measurement — the training self is only ever read."""
    results = prov / "results.jsonl"
    rows: List[dict] = []
    if results.exists():
        rows = [json.loads(x) for x in results.read_text().splitlines() if x.strip()]
        rows = sorted((r for r in rows if r.get("split") == "train"), key=lambda r: r["seq"])
    snaps = prov / "snapshots"

    def snap(h: object) -> Optional[Path]:
        d = snaps / h if isinstance(h, str) else None
        return d if (d and d.exists()) else None

    start = (snap(rows[0]["self_hash_before"]) if rows else None) or self_dir
    mid = (snap(rows[len(rows) // 2]["self_hash_before"]) if rows else None) or self_dir
    return {"start": start, "mid": mid, "end": self_dir}


# --- run metadata + session continuity ------------------------------------------------

def write_run_meta(root: Path, meta: dict) -> None:
    """Persist the knobs a resume must reconstruct (stream size, arm, model, …) to the run dir."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "run.json").write_text(json.dumps(meta, indent=2))


def read_run_meta(root: Path) -> dict:
    p = root / "run.json"
    if not p.exists():
        raise SystemExit(f"BLOCKED: no {p} — cannot resume a run without its saved metadata.")
    return json.loads(p.read_text())


def write_session(root: Path, session_id: str, origin: str) -> None:
    """Persist the Claude Code session id at launch, so an interrupted run can be resumed even if it
    is killed before the agent finishes (``origin`` records how we learned the id)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "session.json").write_text(
        json.dumps({"session_id": session_id, "origin": origin}, indent=2))


def read_session_id(root: Path, stream_path: Path) -> Optional[str]:
    """The session id to resume: prefer the id persisted at launch (session.json); fall back to the
    ``session_id`` on the init/system event of the captured stream-json trace (covers a run killed
    before session.json landed — the id is emitted on the very first event)."""
    p = root / "session.json"
    if p.exists():
        try:
            sid = json.loads(p.read_text()).get("session_id")
            if sid:
                return str(sid)
        except (json.JSONDecodeError, OSError):
            pass
    if stream_path.exists():
        for line in stream_path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            if ev.get("session_id"):
                return str(ev["session_id"])
    return None


def train_scored(prov: Path) -> int:
    """How many train tasks the broker has authoritatively sealed + scored so far (one provenance
    row per task). The broker owns the count, so the agent cannot fake resume progress."""
    return aggregate(prov / "results.jsonl", split="train")["n"]


# --- phases ---------------------------------------------------------------------------

def _heldout_pass(*, run_id: str, arm: str, cp: str, indices: List[int], src: Optional[Path],
                  prov: Path, stream_dir: Path, disallowed: Optional[List[str]], system: str,
                  oauth: str, wandb_key: Optional[str], project: str) -> dict:
    """One held-out pass over ``indices`` with per-task ISOLATION: a FRESH broker container
    (``indices=[idx]`` — it can dispense only that one task) plus a FRESH agent per task. Each
    treatment agent gets a throwaway COPY of ``src`` (the checkpoint self) as its workdir; control
    passes ``src=None`` (empty workdir). This mirrors the local held-out path exactly — no
    cross-task in-context carryover, and the training self is never handed to a held-out pass.
    Results accumulate in ``prov/results.jsonl``; each broker streams its seal-scored reward live
    to W&B under the checkpoint's run name."""
    for i, idx in enumerate(indices):
        bname = f"selfopt-broker-{arm}-ho-{cp}-{i:03d}"
        work = stream_dir / f"work-{i:03d}"
        try:
            start_broker(bname, split="heldout", prov=prov,
                         run_name=f"{run_id}-{arm}-heldout-{cp}", indices=[idx],
                         wandb_key=wandb_key, project=project)
            if src is not None:
                copy_tree(src, work)  # throwaway COPY — writes here never reach the training self
            run_agent(work=work, prompt=KICKOFF, append_system_prompt=system,
                      disallowed=disallowed, oauth=oauth,
                      stream_path=stream_dir / f"stream-{i:03d}.jsonl", broker_name=bname)
        finally:
            _rm(bname)
    return aggregate(prov / "results.jsonl")


def phase_treatment(run_id: str, stream: List[int], heldout_size: int, *, oauth: str,
                    wandb_key: Optional[str], project: str, root: Path,
                    session_id: str, resume: bool = False) -> dict:
    rd = root / "treatment"
    self_dir = rd / "self"
    # The agent's native Claude Code home (~/.claude): a persistent host dir mounted at the
    # container's $HOME/.claude so memory/skills the CLI writes SURVIVE across the single
    # persistent session and are snapshotted per task — the OTHER half of "the self". Under the
    # (gitignored) run dir, alongside self/. It ALSO holds the session transcript, which is what
    # makes ``--resume`` continue the same conversation.
    self_home = rd / "self_home"
    prov = rd / "prov"
    stream_path = rd / "stream.jsonl"
    if not resume:
        self_dir.mkdir(parents=True, exist_ok=True)
        (self_dir / "CLAUDE.md").write_text("# self\n")
        self_home.mkdir(parents=True, exist_ok=True)
    elif not self_dir.exists() or not self_home.exists():
        # RESUME reuses the persisted self + native home EXACTLY as they are (never recreates them —
        # that would wipe the accumulated self-surface). If they are gone, there is nothing to
        # resume against.
        raise SystemExit(f"BLOCKED: resume target {rd} is missing self/ or self_home/.")

    # -- train: ONE persistent agent over the whole stream, full tools incl web.
    bname = "selfopt-broker-train"
    session_mode: Optional[str] = None
    try:
        start_broker(bname, split="train", prov=prov, run_name=f"{run_id}-treatment-train",
                     self_ro=self_dir, home_ro=self_home, queue_size=len(stream),
                     wandb_key=wandb_key, project=project, resume=resume)
        if not resume:
            print("==> [treatment] train: persistent 'Get Better' agent over the stream",
                  flush=True)
            run_agent(work=self_dir, prompt=KICKOFF, append_system_prompt=TREATMENT_PROMPT,
                      oauth=oauth, stream_path=stream_path, broker_name=bname, home=self_home,
                      session_id=session_id)
            session_mode = "initial"
        else:
            before = train_scored(prov)
            sid = read_session_id(root, stream_path) or session_id
            print(f"==> [treatment] train RESUME: continue session {sid or '(none captured)'} from "
                  f"task {before + 1} (broker skipped {before} already-scored)", flush=True)
            if sid:
                rc = run_agent(work=self_dir, prompt=CONTINUE_PROMPT,
                               append_system_prompt=TREATMENT_PROMPT,
                               oauth=oauth, stream_path=stream_path, broker_name=bname,
                               home=self_home, resume=sid, append=True)
                after = train_scored(prov)
                if rc == 0 or after > before:
                    session_mode = "resumed-session"
                    print(f"==> [treatment] train RESUME: SAME session continued "
                          f"(scored {before} -> {after})", flush=True)
            if session_mode is None:
                # The session could not be revived (none captured, or the resume failed) — WARM
                # RESTART: a fresh ``claude -p`` on the SAME persisted self + native home (the
                # on-disk self is kept; only the in-context conversation is lost). The broker still
                # dispenses only the unscored tasks.
                print("==> [treatment] train RESUME: session not revivable — WARM RESTART "
                      "(fresh claude -p; self/ + self_home kept)", flush=True)
                run_agent(work=self_dir, prompt=KICKOFF, append_system_prompt=TREATMENT_PROMPT,
                          oauth=oauth, stream_path=stream_path, broker_name=bname, home=self_home,
                          append=True)
                session_mode = "warm-restart"
    finally:
        _rm(bname)
    train = aggregate(prov / "results.jsonl", split="train")

    # -- held-out curve at start / mid / end (web-off), each a throwaway copy of the checkpoint
    #    self. The training self is never handed to a held-out pass.
    selves = resolve_checkpoint_selves(prov, self_dir)
    pre = content_hash(self_dir)
    indices = list(public_split().heldout)[:heldout_size]
    curve: dict = {}
    for cp in ("start", "mid", "end"):
        src = selves[cp]
        print(f"==> [treatment] held-out {cp}: web-off probe of self {content_hash(src)}",
              flush=True)
        agg = _heldout_pass(run_id=run_id, arm="treatment", cp=cp, indices=indices, src=src,
                            prov=rd / "heldout" / cp / "prov", stream_dir=rd / "heldout" / cp,
                            disallowed=_WEB_OFF, system=HELDOUT_PROMPT, oauth=oauth,
                            wandb_key=wandb_key, project=project)
        curve[cp] = {"mean_reward": agg["mean_reward"], "n": agg["n"],
                     "measured_self": content_hash(src)}
    assert content_hash(self_dir) == pre, "held-out mutated the training self — isolation broken"
    return {"train": {"n": train["n"], "mean_reward": train["mean_reward"]},
            "heldout_curve": curve, "self_unchanged_by_heldout": True,
            "session_mode": session_mode}


def phase_control(run_id: str, stream: List[int], heldout_size: int, *, oauth: str,
                  wandb_key: Optional[str], project: str, root: Path) -> dict:
    rd = root / "control"
    prov = rd / "prov"
    # -- train: a FRESH agent per task (empty workdir, no persistence, no "Get Better").
    bname = "selfopt-broker-control-train"
    try:
        start_broker(bname, split="train", prov=prov, run_name=f"{run_id}-control-train",
                     queue_size=len(stream), wandb_key=wandb_key, project=project)
        print("==> [control] train: fresh agent per task, no self, no instruction", flush=True)
        for i in range(len(stream)):
            work = rd / "train" / f"work-{i:03d}"
            run_agent(work=work, prompt=KICKOFF, append_system_prompt=CONTROL_PROMPT,
                      disallowed=_WEB_OFF, oauth=oauth,
                      stream_path=rd / "train" / f"stream-{i:03d}.jsonl", broker_name=bname)
    finally:
        _rm(bname)
    train = aggregate(prov / "results.jsonl", split="train")

    # -- held-out baseline: the default harness (fresh, no self, curriculum-only ⇒ web-off).
    #    Control has no evolving self, so its held-out is checkpoint-invariant — measure it once.
    print("==> [control] held-out baseline: fresh default harness", flush=True)
    ho = _heldout_pass(run_id=run_id, arm="control", cp="end",
                       indices=list(public_split().heldout)[:heldout_size], src=None,
                       prov=rd / "heldout" / "prov", stream_dir=rd / "heldout",
                       disallowed=_WEB_OFF, system=CONTROL_PROMPT, oauth=oauth,
                       wandb_key=wandb_key, project=project)
    return {"train": {"n": train["n"], "mean_reward": train["mean_reward"]},
            "heldout_baseline": {"mean_reward": ho["mean_reward"], "n": ho["n"]}}


# --- entrypoint -----------------------------------------------------------------------

def plan(train_size: int, heldout_size: int, arm: str, wandb_on: bool) -> dict:
    split = public_split()
    stream = train_stream(split, train_size)
    return {
        "topology": "isolated two-container (agent has NO broker mount; broker owns the answers)",
        "env": config.ENV_NAME, "model": config.MODEL, "effort": config.EFFORT,
        "arms": arm, "train_stream": len(stream), "heldout_eval": heldout_size,
        "checkpoints": ["start", "mid", "end"],
        "held_out_web": "OFF (answers are online — measure capability, not lookup)",
        "treatment_train_web": "ON (cheating-there is a finding, by design)",
        "wandb": "LIVE (broker-side)" if wandb_on else "off → LocalSink (offline)",
        "authoritative_heldout": "seal-scored, deterministic, keyless",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-size", type=int, default=30)
    ap.add_argument("--heldout-size", type=int, default=20)
    ap.add_argument("--arm", choices=["treatment", "control", "both"], default="both")
    ap.add_argument("--go", action="store_true", help="ACTUALLY run (real spend). Else: plan.")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="resume an interrupted run: reuse its self/home + persisted session and "
                         "continue the seeded stream from the next unscored task (implies --go)")
    ap.add_argument("--build", action="store_true", help="(re)build the broker + agent images")
    ap.add_argument("--wandb", action="store_true",
                    help="stream broker-side metrics live to W&B (needs WANDB_API_KEY in env)")
    ap.add_argument("--project", default=config.WANDB_PROJECT)
    args = ap.parse_args()

    resuming = bool(args.resume)
    # A resume reconstructs the study knobs from the run's saved metadata (so the seeded stream is
    # rebuilt identically); a fresh run takes them from the CLI.
    if resuming:
        root = config.RUNS_DIR / args.resume
        meta = read_run_meta(root)
        run_id = args.resume
        train_size, heldout_size, arm = meta["train_size"], meta["heldout_size"], meta["arm"]
        project = meta.get("project", args.project)
    else:
        run_id = f"sandbox-{int(time.time())}"
        root = config.RUNS_DIR / run_id
        train_size, heldout_size, arm = args.train_size, args.heldout_size, args.arm
        project = args.project

    wandb_key = os.environ.get("WANDB_API_KEY") if args.wandb else None
    if args.wandb and not wandb_key:
        print("[study] --wandb set but WANDB_API_KEY absent — broker will fall back to LocalSink.",
              file=sys.stderr)

    p = plan(train_size, heldout_size, arm, bool(wandb_key))
    p["resume"] = args.resume if resuming else None
    print(json.dumps({"plan": p}, indent=2))
    if not (args.go or resuming):
        print("\n[dry-run] plan only. Re-run with --go to spend on the real sandbox study.")
        return

    oauth = os.environ.get("CLAUDE_CODE_OAUTH_TOKEN", "")
    if not oauth:
        raise SystemExit("BLOCKED: no CLAUDE_CODE_OAUTH_TOKEN in env (runtime-only credential).")

    ensure_network()
    if args.build:
        build_images()
    stream = train_stream(public_split(), train_size)

    if resuming:
        # The session id was fixed + persisted at the ORIGINAL launch; phase_treatment reads it back
        # (session.json, or the stream-json trace as a fallback) to continue the same conversation.
        session_id = read_session_id(root, root / "treatment" / "stream.jsonl") or ""
    else:
        # Pin the Claude Code session id UP FRONT (a fresh UUID) and persist it before the agent
        # starts — so even a run killed mid-task can be resumed into the SAME session.
        session_id = str(uuid.uuid4())
        write_run_meta(root, {"run_id": run_id, "train_size": train_size,
                              "heldout_size": heldout_size, "arm": arm, "model": config.MODEL,
                              "effort": config.EFFORT, "project": project})
        write_session(root, session_id, origin="assigned")

    out: dict = {"run_id": run_id, "runs_dir": str(root), "resumed": resuming}
    try:
        if arm in ("treatment", "both"):
            out["treatment"] = phase_treatment(run_id, stream, heldout_size, oauth=oauth,
                                                wandb_key=wandb_key, project=project, root=root,
                                                session_id=session_id, resume=resuming)
        if arm in ("control", "both"):
            out["control"] = phase_control(run_id, stream, heldout_size, oauth=oauth,
                                           wandb_key=wandb_key, project=project, root=root)
    finally:
        # Sweep every broker container this study may have started (train + per-task held-out).
        leftover = subprocess.run(["docker", "ps", "-aq", "--filter", "name=selfopt-broker"],
                                  capture_output=True, text=True).stdout.split()
        if leftover:
            subprocess.run(["docker", "rm", "-f", *leftover], capture_output=True)
    print(json.dumps({"result": out}, indent=2))


if __name__ == "__main__":
    main()
