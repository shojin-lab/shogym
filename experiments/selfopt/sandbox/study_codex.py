"""The full two-arm + held-out study for **cell #2 (Codex)**, under the isolated two-container
sandbox. The Codex analog of ``study.py``: same phases, same isolation, same authoritative
seal-scoring off an isolated broker — only the *agent harness* changes (Codex ``codex exec``
instead of Claude Code ``claude -p``). The broker / split / snapshot / held-out / sink modules
and the broker container image are **reused unchanged**; the harness-specific bits (command
builder, config, trace capture, auth, agent image) live behind :mod:`.codex_arms`.

Phases (identical in shape to cell #1, against the isolated broker):
  - TREATMENT train : ONE persistent ``codex exec`` over the train stream ("Get Better"), web ON
                      (cheating-there is a finding), self-dir RW. The broker snapshots the self at
                      every task boundary.
  - TREATMENT held-out : at start / mid / end, a fresh Codex per held-out task plays a THROWAWAY
                      COPY of that checkpoint's archived self — **web OFF** — so held-out probes
                      capability and never mutates the training self.
  - CONTROL train + held-out : a fresh Codex per task, no persistent self, no "Get Better",
                      web OFF — the baseline the treatment curve is measured against.

⚠️  ISOLATION FROM CELL #1: every container name here is prefixed ``selfopt-c2-`` and the network
is ``selfopt-c2-net`` — DISJOINT from cell #1's ``selfopt-broker*`` / ``selfopt-net``. The cleanup
sweep filters ``name=selfopt-c2-broker`` ONLY, so a concurrent cell-#1 run (e.g. the live 480) is
never touched.

AUTH is subscription-only + RUNTIME-only: the ChatGPT ``~/.codex/auth.json`` is copied into a
per-run isolated CODEX_HOME (gitignored ``runs/``, mode 600) and mounted; the billed
``OPENAI_API_KEY`` is never passed into the container. Verify with ``codex login status`` first.

    uv run python experiments/selfopt/sandbox/study_codex.py --plan          # dry-run: the plan
    uv run python experiments/selfopt/sandbox/study_codex.py --go --build \
        --arm both --train-size 2 --heldout-size 2 --wandb                    # the real run (spend)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import List, Optional

# Run as a script: put the repo root on the path so the shared experiment code — the SAME prompts /
# split / snapshot / broker the local path uses — is importable. Reuse is what stops drift.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from experiments.selfopt import config  # noqa: E402
from experiments.selfopt.codex_arms import (  # noqa: E402
    CODEX_EFFORT,
    CODEX_MODEL,
    CONTINUE_PROMPT,
    CONTROL_PROMPT,
    HELDOUT_PROMPT,
    MCP_PATH,
    STUDY_DIFFERENCES,
    TREATMENT_PROMPT,
    prepare_codex_home,
)
from experiments.selfopt.broker import public_split  # noqa: E402
from experiments.selfopt.heldout import aggregate  # noqa: E402
from experiments.selfopt.sink import MetricSink, make_sink  # noqa: E402
from experiments.selfopt.snapshot import content_hash, copy_tree, snapshot  # noqa: E402
from experiments.selfopt.split import train_stream  # noqa: E402

# Cell-#2 topology names — DISJOINT from cell #1 (never touches selfopt-net / selfopt-broker*).
NET = "selfopt-c2-net"
BROKER_IMG = "selfopt-ab-broker:latest"          # reused unchanged from cell #1
AGENT_IMG = "selfopt-ab-agent-codex:latest"      # the Codex agent image
BROKER_PREFIX = "selfopt-c2-broker"              # cleanup sweeps ONLY this prefix
SANDBOX = Path(__file__).resolve().parent
AUTH_SRC = Path.home() / ".codex" / "auth.json"


# --- the Codex home as the real self-surface ------------------------------------------
# The agent's real self-improvement channel is its persistent CODEX_HOME — the skills/rules/
# prompts it writes, its memory store, and config.toml — NOT the mounted self/ workdir stub. We
# snapshot that surface alongside self/. But CODEX_HOME ALSO holds the ChatGPT credential
# (auth.json, copied in at runtime), rotating session/oauth-lock state, sqlite session/state DBs,
# shell snapshots, an installation id, and caches — none of which may EVER be archived or
# exported. So we ALLOW-LIST the durable self-surface (never a deny-list: a new credential/state
# file the CLI starts writing then cannot silently leak in), drop CLI-bundled system skills and
# sqlite journals from it, and audit the staged copy before keeping it.
CODEX_SURFACE_FILES = {"config.toml", "AGENTS.md"}
CODEX_SURFACE_DIRS = ("skills", "rules", "memories", "prompts")  # str.startswith accepts a tuple
_SURFACE_SKIP_PARTS = {".system"}                 # CLI-bundled skills — not self-authored, bloat
_SURFACE_SKIP_SUFFIXES = ("-wal", "-shm", "-journal")  # sqlite journals — churn the hash, no signal
# Belt-and-suspenders over the allow-list: refuse to keep a snapshot if a credential/session-token
# file (auth.json / *.token / session.json / secret.key …) somehow landed in it.
_CRED_FILE_RE = re.compile(
    r"^(auth|tokens?|credentials?|secrets?|session)\.(json|jsonl|jwt|key|pem|env|txt)$", re.I)


def _stage_self_surface(cfg_home: Path, stage: Path) -> None:
    """Copy ONLY the durable self-surface out of CODEX_HOME into ``stage`` — skipping the ChatGPT
    credential, all session/oauth/state/cache files, the CLI's bundled ``.system`` skills, and
    sqlite journals. Everything not on the allow-list is left behind by construction."""
    for entry in sorted(cfg_home.iterdir()):
        if not (entry.name in CODEX_SURFACE_FILES or entry.name.startswith(CODEX_SURFACE_DIRS)):
            continue
        if entry.is_dir():
            for p in sorted(entry.rglob("*")):
                rel = p.relative_to(cfg_home)
                if (not p.is_file()
                        or any(part in _SURFACE_SKIP_PARTS for part in rel.parts)
                        or p.name.endswith(_SURFACE_SKIP_SUFFIXES)):
                    continue
                (stage / rel).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(p, stage / rel)
        elif not entry.name.endswith(_SURFACE_SKIP_SUFFIXES):
            shutil.copy2(entry, stage / entry.name)


def _audit_no_credentials(snap_dir: Path) -> None:
    """Fail loudly if a credential/session-token file made it into a codexhome snapshot — the
    archive + artifact must NEVER carry the ChatGPT auth (or any token)."""
    for p in snap_dir.rglob("*"):
        if p.is_file() and _CRED_FILE_RE.match(p.name):
            raise RuntimeError(
                f"credential-looking file {p} in codexhome snapshot — refusing to keep it")


def snapshot_codex_home(cfg_home: Path, dest_dir: Path, *, sink: MetricSink, seq: int,
                        label: Optional[str] = None) -> Optional[str]:
    """Snapshot the persistent Codex self-surface (skills/rules/prompts/memory + config.toml) from
    the isolated CODEX_HOME, EXCLUDING the ChatGPT credential and all session/state/cache files,
    then export it via the SAME ``sink.artifact`` path the self/ snapshots use. Returns the content
    hash (a self-edit there is detectable by a hash change), or None if the home is absent."""
    cfg_home = Path(cfg_home)
    if not cfg_home.exists():
        return None
    with tempfile.TemporaryDirectory() as tmp:
        stage = Path(tmp) / "codexhome"
        stage.mkdir()
        _stage_self_surface(cfg_home, stage)  # allow-list only — the credential never enters here
        digest = snapshot(stage, dest_dir, label=label)
    snap = Path(dest_dir) / digest
    _audit_no_credentials(snap)  # double-check the produced archive carries no credential/token
    sink.log({"event": "codexhome_snapshot", "seq": seq, "label": label,
              "codexhome_hash": digest})
    sink.artifact(snap, name=f"codexhome-{seq:03d}-{digest}", kind="workdir")
    return digest


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
    print("==> building images (broker [reused] + codex agent)", flush=True)
    _run(["docker", "build", "-q", "-f", str(SANDBOX / "broker.Dockerfile"),
          "-t", BROKER_IMG, str(_REPO_ROOT)], stdout=subprocess.DEVNULL)
    _run(["docker", "build", "-q", "-f", str(SANDBOX / "agent.codex.Dockerfile"),
          "-t", AGENT_IMG, str(_REPO_ROOT)], stdout=subprocess.DEVNULL)


def start_broker(name: str, *, split: str, prov: Path, run_name: str,
                 self_ro: Optional[Path] = None, queue_size: Optional[int] = None,
                 indices: Optional[List[int]] = None, wandb_key: Optional[str] = None,
                 project: str, resume: bool = False) -> None:
    """Start an ISOLATED broker container (cell #1's image, unchanged) and wait until its HTTP MCP
    port is listening. Holds targets + split + provenance on a broker-ONLY volume."""
    _rm(name)
    prov.mkdir(parents=True, exist_ok=True)
    args = ["docker", "run", "-d", "--name", name, "--network", NET,
            "-e", f"SELFOPT_SPLIT={split}", "-e", f"SELFOPT_RUN_NAME={run_name}",
            "-v", f"{prov}:/provenance"]
    if self_ro is not None:
        args += ["-v", f"{self_ro}:/self:ro"]
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
            time.sleep(settle)  # FastMCP settles after the socket opens; don't race the handshake
            return
        time.sleep(1.0)
    logs = subprocess.run(["docker", "logs", "--tail", "30", name],
                          capture_output=True, text=True)
    raise RuntimeError(f"broker {name} never came up:\n{logs.stdout}\n{logs.stderr}")


def run_agent(*, work: Path, cfg_home: Path, prompt: str, web_search: bool,
              stream_path: Path, broker_name: str, model: str, effort: str,
              resume_thread: Optional[str] = None, append: bool = False) -> int:
    """Run ONE Codex agent-container pass to completion, teeing its ``--json`` trace to
    ``stream_path``.

    ``work`` is the agent's /work (RW): the persistent self for treatment-train, a throwaway self
    copy for a held-out probe, or an empty dir for control. ``cfg_home`` is an isolated CODEX_HOME
    (config.toml + a runtime copy of auth.json) mounted at /codexhome — kept OUTSIDE /work so the
    subscription credential never enters a snapshotted self. The agent gets NO mount of the
    broker's provenance/answer volume — that is the isolation. The billed OPENAI_API_KEY is never
    passed in (subscription-only).

    ``resume_thread`` continues an EXISTING Codex session (``codex exec resume <id>``) instead of
    starting a fresh one — same mounted CODEX_HOME (where the session state lives) and same /work,
    so the conversation history AND the persistent self both carry across the separate container
    invocations. ``append`` tees onto the end of ``stream_path`` (and its ``.err.txt``) so a whole
    resume chain reads back as ONE continuous trace. ``resume`` has no ``-C``/``-s`` flags; the
    workdir comes from the container's ``-w /work`` and the sandbox from
    ``--dangerously-bypass-approvals-and-sandbox`` (the container IS the isolation boundary — the
    same no-double-sandboxing stance as the fresh pass's ``-s danger-full-access``)."""
    work.mkdir(parents=True, exist_ok=True)
    stream_path.parent.mkdir(parents=True, exist_ok=True)
    broker_url = f"http://{broker_name}:9000{MCP_PATH}"
    prepare_codex_home(cfg_home, broker_url=broker_url, web_search=web_search,
                       model=model, effort=effort, auth_src=AUTH_SRC)
    if resume_thread is not None:
        codex = ["codex", "exec", "resume", resume_thread, "--json",
                 "--dangerously-bypass-approvals-and-sandbox",
                 "--skip-git-repo-check", prompt]
    else:
        codex = ["codex", "exec", "--json", "-s", "danger-full-access",
                 "--skip-git-repo-check", "-C", "/work", prompt]
    docker = ["docker", "run", "--rm", "--network", NET,
              "-e", "CODEX_HOME=/codexhome",
              "-v", f"{work}:/work:rw", "-v", f"{cfg_home}:/codexhome:rw",
              "-w", "/work", AGENT_IMG] + codex
    err_path = stream_path.with_suffix(".err.txt")
    mode = "a" if append else "w"
    with stream_path.open(mode, encoding="utf-8") as out, err_path.open(mode) as err, \
            open(os.devnull) as devnull:
        return subprocess.run(docker, stdout=out, stderr=err, stdin=devnull).returncode


# --- treatment resume-loop helpers ----------------------------------------------------

def _first_thread_id(stream_path: Path) -> Optional[str]:
    """The Codex session id to resume: the ``thread_id`` on the first ``thread.started`` event of
    the run's trace. That id names the persistent session, so every resume continues it (full
    conversation history intact)."""
    if not stream_path.exists():
        return None
    for line in stream_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("type") == "thread.started" and ev.get("thread_id"):
            return str(ev["thread_id"])
    return None


def _train_scored(prov: Path) -> int:
    """How many train tasks the broker has authoritatively scored so far (one provenance row per
    sealed task). The orchestrator watches this — the broker owns the count, so the agent cannot
    fake progress."""
    return aggregate(prov / "results.jsonl", split="train")["n"]


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


def write_session(root: Path, thread_id: str, origin: str = "thread.started") -> None:
    """Persist the Codex session (``thread_id``) so an interrupted run can be resumed even if it is
    killed before the first pass finishes (``origin`` records how we learned the id)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "session.json").write_text(
        json.dumps({"thread_id": thread_id, "origin": origin}, indent=2))


def read_thread_id(root: Path, stream_path: Path) -> Optional[str]:
    """The Codex ``thread_id`` to resume: prefer the id persisted at launch (session.json); fall
    back to the ``thread.started`` event on the captured trace (covers a run killed before
    session.json landed — Codex emits that event at the very start of the pass)."""
    p = root / "session.json"
    if p.exists():
        try:
            tid = json.loads(p.read_text()).get("thread_id")
            if tid:
                return str(tid)
        except (json.JSONDecodeError, OSError):
            pass
    return _first_thread_id(stream_path)


# --- self-version resolution (from the treatment train provenance) --------------------

def resolve_checkpoint_selves(prov: Path, self_dir: Path) -> dict:
    """start / mid / end archived self snapshots from the train provenance (falls back to the live
    self_dir). Probing archived snapshots keeps held-out a pure measurement of a read-only self."""
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


# --- phases ---------------------------------------------------------------------------

def _heldout_pass(*, run_id: str, arm: str, cp: str, indices: List[int], src: Optional[Path],
                  prov: Path, stream_dir: Path, web_search: bool, prompt: str,
                  model: str, effort: str, wandb_key: Optional[str], project: str) -> dict:
    """One held-out pass over ``indices`` with per-task ISOLATION: a FRESH broker container
    (dispenses only that one task) plus a FRESH Codex per task, each treatment agent handed a
    throwaway COPY of the checkpoint self (control passes ``src=None``). Held-out is **web-off**."""
    for i, idx in enumerate(indices):
        bname = f"{BROKER_PREFIX}-{arm}-ho-{cp}-{i:03d}"
        work = stream_dir / f"work-{i:03d}"
        cfg_home = stream_dir / f"codexhome-{i:03d}"
        try:
            start_broker(bname, split="heldout", prov=prov,
                         run_name=f"{run_id}-{arm}-heldout-{cp}", indices=[idx],
                         wandb_key=wandb_key, project=project)
            if src is not None:
                copy_tree(src, work)  # throwaway COPY — writes here never reach the training self
            run_agent(work=work, cfg_home=cfg_home, prompt=prompt, web_search=web_search,
                      stream_path=stream_dir / f"stream-{i:03d}.jsonl", broker_name=bname,
                      model=model, effort=effort)
        finally:
            _rm(bname)
    return aggregate(prov / "results.jsonl")


def phase_treatment(run_id: str, stream: List[int], heldout_size: int, *,
                    wandb_key: Optional[str], project: str, root: Path,
                    model: str, effort: str, sink: MetricSink, resume: bool = False) -> dict:
    rd = root / "treatment"
    self_dir = rd / "self"
    prov = rd / "prov"
    # Codexhome (the real self-surface) snapshots — host-side, since the persistent CODEX_HOME is
    # mounted into the AGENT, not the broker (so the broker can't archive it the way it does self/).
    cx_snaps = rd / "codexhome_snapshots"
    bname = f"{BROKER_PREFIX}-train"
    cfg_home = rd / "codexhome"
    stream_path = rd / "stream.jsonl"
    if not resume:
        self_dir.mkdir(parents=True, exist_ok=True)
        (self_dir / "AGENTS.md").write_text("# self\n")  # Codex's persistent-workdir memory file
    elif not self_dir.exists() or not cfg_home.exists():
        # RESUME reuses the persisted self + isolated CODEX_HOME (which holds the session state
        # `codex exec resume` needs) EXACTLY as they are — never recreates them. If they are gone,
        # there is nothing to resume against.
        raise SystemExit(f"BLOCKED: resume target {rd} is missing self/ or codexhome/.")

    # -- train: ONE persistent Codex over the whole stream, web ON. Driven by an ORCHESTRATOR
    # resume loop: `codex exec` runs a single conversational turn (~a few tasks), then yields
    # control (it exits waiting for the user to say "continue") — unlike `claude -p`, which runs
    # to completion. So the orchestrator resumes the SAME session (`codex exec resume <thread>`,
    # same CODEX_HOME + /work mounts) until the broker's train provenance shows every task scored,
    # preserving the whole point of treatment: the in-context accumulation carries across resumes.
    # A session-continuous RESUME (--resume) is the same mechanism ACROSS processes: reconnect to
    # the persisted thread and keep going from the next unscored task.
    queue_size = len(stream)
    resumes = 0
    stall = 0
    STALL_LIMIT = 2           # K consecutive no-progress resumes ⇒ Codex won't continue; stop.
    resume_cap = 2 * queue_size  # hard safety cap on total resumes (never spin forever).
    session_mode: Optional[str] = None
    try:
        start_broker(bname, split="train", prov=prov, run_name=f"{run_id}-treatment-train",
                     self_ro=self_dir, queue_size=queue_size, wandb_key=wandb_key,
                     project=project, resume=resume)
        if not resume:
            print("==> [treatment] train: persistent 'Get Better' Codex over the stream",
                  flush=True)
            run_agent(work=self_dir, cfg_home=cfg_home, prompt=TREATMENT_PROMPT,
                      web_search=True, stream_path=stream_path, broker_name=bname,
                      model=model, effort=effort)
            cx_hash = snapshot_codex_home(cfg_home, cx_snaps, sink=sink, seq=0, label="train-0")
            thread_id = _first_thread_id(stream_path)
            if thread_id:
                write_session(root, thread_id)  # persist the session id for a cross-process resume
            scored = _train_scored(prov)
            session_mode = "initial"
            print(f"==> [treatment] train: first pass scored {scored}/{queue_size} "
                  f"(session {thread_id}, codexhome {cx_hash})", flush=True)
        else:
            before = _train_scored(prov)
            thread_id = read_thread_id(root, stream_path)
            print(f"==> [treatment] train RESUME: continue session {thread_id} from task "
                  f"{before + 1} (broker skipped {before} already-scored)", flush=True)
            if thread_id is not None:
                rc = run_agent(work=self_dir, cfg_home=cfg_home, prompt=CONTINUE_PROMPT,
                               web_search=True, stream_path=stream_path, broker_name=bname,
                               model=model, effort=effort, resume_thread=thread_id, append=True)
                after = _train_scored(prov)
                if rc == 0 or after > before:
                    session_mode = "resumed-session"
                    print(f"==> [treatment] train RESUME: SAME session continued "
                          f"(scored {before} -> {after})", flush=True)
            if session_mode is None:
                # The session could not be revived (gone, or the resume failed) — WARM RESTART: a
                # fresh `codex exec` on the SAME persisted self + CODEX_HOME (the on-disk self is
                # kept; only the in-context conversation is lost). The broker still dispenses only
                # the unscored tasks. Capture the NEW thread id so the resume loop below continues.
                print("==> [treatment] train RESUME: session not revivable — WARM RESTART "
                      "(fresh codex exec; self/ + codexhome kept)", flush=True)
                run_agent(work=self_dir, cfg_home=cfg_home, prompt=TREATMENT_PROMPT,
                          web_search=True, stream_path=stream_path, broker_name=bname,
                          model=model, effort=effort, append=True)
                thread_id = _first_thread_id(stream_path)
                session_mode = "warm-restart"
            snapshot_codex_home(cfg_home, cx_snaps, sink=sink, seq=0, label="resume-0")
            scored = _train_scored(prov)
        while scored < queue_size and resumes < resume_cap and stall < STALL_LIMIT:
            if thread_id is None:
                print("==> [treatment] train: no session id captured — cannot resume", flush=True)
                break
            resumes += 1
            run_agent(work=self_dir, cfg_home=cfg_home, prompt=CONTINUE_PROMPT,
                      web_search=True, stream_path=stream_path, broker_name=bname,
                      model=model, effort=effort, resume_thread=thread_id, append=True)
            snapshot_codex_home(cfg_home, cx_snaps, sink=sink, seq=resumes,
                                label=f"resume-{resumes}")
            now = _train_scored(prov)
            stall = stall + 1 if now == scored else 0
            print(f"==> [treatment] train: resume {resumes} scored {now}/{queue_size}"
                  f"{' (no progress)' if now == scored else ''}", flush=True)
            scored = now
        why = ("stream exhausted" if scored >= queue_size
               else f"stall guard ({STALL_LIMIT} no-progress resumes)" if stall >= STALL_LIMIT
               else f"resume cap ({resume_cap})" if resumes >= resume_cap
               else "no session id")
        print(f"==> [treatment] train done: {scored}/{queue_size} tasks scored, "
              f"{resumes} resume(s), stopped on {why}", flush=True)
    finally:
        _rm(bname)
    train = aggregate(prov / "results.jsonl", split="train")

    # -- held-out curve at start / mid / end (web-off), each a throwaway copy of the checkpoint.
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
                            web_search=False, prompt=HELDOUT_PROMPT, model=model, effort=effort,
                            wandb_key=wandb_key, project=project)
        curve[cp] = {"mean_reward": agg["mean_reward"], "n": agg["n"],
                     "measured_self": content_hash(src)}
    assert content_hash(self_dir) == pre, "held-out mutated the training self — isolation broken"
    return {"train": {"n": train["n"], "mean_reward": train["mean_reward"]},
            "heldout_curve": curve, "self_unchanged_by_heldout": True,
            "session_mode": session_mode}


def phase_control(run_id: str, stream: List[int], heldout_size: int, *,
                  wandb_key: Optional[str], project: str, root: Path,
                  model: str, effort: str) -> dict:
    rd = root / "control"
    prov = rd / "prov"
    # -- train: a FRESH Codex per task (empty workdir, no persistence, no "Get Better", web OFF).
    bname = f"{BROKER_PREFIX}-control-train"
    try:
        start_broker(bname, split="train", prov=prov, run_name=f"{run_id}-control-train",
                     queue_size=len(stream), wandb_key=wandb_key, project=project)
        print("==> [control] train: fresh Codex per task, no self, no instruction", flush=True)
        for i in range(len(stream)):
            work = rd / "train" / f"work-{i:03d}"
            run_agent(work=work, cfg_home=rd / "train" / f"codexhome-{i:03d}",
                      prompt=CONTROL_PROMPT, web_search=False,
                      stream_path=rd / "train" / f"stream-{i:03d}.jsonl", broker_name=bname,
                      model=model, effort=effort)
    finally:
        _rm(bname)
    train = aggregate(prov / "results.jsonl", split="train")

    # -- held-out baseline: the default harness (fresh, no self, web OFF). Checkpoint-invariant.
    print("==> [control] held-out baseline: fresh default harness", flush=True)
    ho = _heldout_pass(run_id=run_id, arm="control", cp="end",
                       indices=list(public_split().heldout)[:heldout_size], src=None,
                       prov=rd / "heldout" / "prov", stream_dir=rd / "heldout",
                       web_search=False, prompt=CONTROL_PROMPT, model=model, effort=effort,
                       wandb_key=wandb_key, project=project)
    return {"train": {"n": train["n"], "mean_reward": train["mean_reward"]},
            "heldout_baseline": {"mean_reward": ho["mean_reward"], "n": ho["n"]}}


# --- entrypoint -----------------------------------------------------------------------

def plan(train_size: int, heldout_size: int, arm: str, wandb_on: bool,
         model: str, effort: str) -> dict:
    split = public_split()
    stream = train_stream(split, train_size)
    return {
        "cell": "cell #2 — Codex on AutomationBench ('Get Better')",
        "harness": "codex exec (JSONL trace, danger-full-access sandbox, HTTP MCP curriculum)",
        "topology": "isolated two-container (agent has NO broker mount; broker owns the answers)",
        "network": NET, "broker_prefix": BROKER_PREFIX,
        "env": config.ENV_NAME, "model": model, "effort": effort,
        "arms": arm, "train_stream": len(stream), "heldout_eval": heldout_size,
        "checkpoints": ["start", "mid", "end"],
        "held_out_web": "OFF (answers are online — measure capability, not lookup)",
        "treatment_train_web": "ON (cheating-there is a finding, by design)",
        "wandb": "LIVE (broker-side)" if wandb_on else "off → LocalSink (offline)",
        "auth": "ChatGPT subscription only (auth.json); billed OPENAI_API_KEY never passed in",
        "differs_from_cell1": STUDY_DIFFERENCES,
    }


def _check_subscription_auth() -> None:
    """Gate a real run on subscription auth being active — NEVER fall back to the billed key."""
    if not AUTH_SRC.exists():
        raise SystemExit(f"BLOCKED: no {AUTH_SRC} — run `codex login` (ChatGPT subscription) first.")
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    out = subprocess.run(["codex", "login", "status"], capture_output=True, text=True, env=env)
    if "ChatGPT" not in (out.stdout + out.stderr):
        raise SystemExit("BLOCKED: `codex login status` does not show a ChatGPT subscription "
                         "login. Do NOT set OPENAI_API_KEY — run `codex login` first.\n"
                         f"{out.stdout}\n{out.stderr}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--train-size", type=int, default=30)
    ap.add_argument("--heldout-size", type=int, default=20)
    ap.add_argument("--arm", choices=["treatment", "control", "both"], default="both")
    ap.add_argument("--model", default=CODEX_MODEL)
    ap.add_argument("--effort", default=CODEX_EFFORT)
    ap.add_argument("--go", action="store_true", help="ACTUALLY run (real spend). Else: plan.")
    ap.add_argument("--resume", metavar="RUN_ID",
                    help="resume an interrupted run: reuse its self/CODEX_HOME + persisted session "
                         "and continue the seeded stream from the next unscored task (implies --go)")
    ap.add_argument("--build", action="store_true", help="(re)build the broker + codex agent images")
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
        model, effort, project = meta["model"], meta["effort"], meta.get("project", args.project)
    else:
        run_id = f"sandbox-codex-{int(time.time())}"
        root = config.RUNS_DIR / run_id
        train_size, heldout_size, arm = args.train_size, args.heldout_size, args.arm
        model, effort, project = args.model, args.effort, args.project

    wandb_key = os.environ.get("WANDB_API_KEY") if args.wandb else None
    if args.wandb and not wandb_key:
        print("[study] --wandb set but WANDB_API_KEY absent — broker will fall back to LocalSink.",
              file=sys.stderr)

    p = plan(train_size, heldout_size, arm, bool(wandb_key), model, effort)
    p["resume"] = args.resume if resuming else None
    print(json.dumps({"plan": p}, indent=2))
    if not (args.go or resuming):
        print("\n[dry-run] plan only. Re-run with --go to spend on the real sandbox study.")
        return

    _check_subscription_auth()
    ensure_network()
    if args.build:
        build_images()
    stream = train_stream(public_split(), train_size)
    if not resuming:
        write_run_meta(root, {"run_id": run_id, "train_size": train_size,
                              "heldout_size": heldout_size, "arm": arm, "model": model,
                              "effort": effort, "project": project})
    out: dict = {"run_id": run_id, "runs_dir": str(root), "resumed": resuming}
    # Host-side sink for the codexhome self-surface artifacts (the broker owns the self/ ones; it
    # can't reach the agent-mounted CODEX_HOME). WandbSink when keyed, else LocalSink (offline).
    sink = make_sink(run_id, use_wandb=bool(wandb_key))
    try:
        if arm in ("treatment", "both"):
            out["treatment"] = phase_treatment(run_id, stream, heldout_size,
                                                wandb_key=wandb_key, project=project,
                                                root=root, model=model, effort=effort,
                                                sink=sink, resume=resuming)
        if arm in ("control", "both"):
            out["control"] = phase_control(run_id, stream, heldout_size,
                                           wandb_key=wandb_key, project=project, root=root,
                                           model=model, effort=effort)
    finally:
        sink.close()
        # Sweep ONLY cell-#2 broker containers (prefix selfopt-c2-broker) — never cell #1's.
        leftover = subprocess.run(["docker", "ps", "-aq", "--filter", f"name={BROKER_PREFIX}"],
                                  capture_output=True, text=True).stdout.split()
        if leftover:
            subprocess.run(["docker", "rm", "-f", *leftover], capture_output=True)
    print(json.dumps({"result": out}, indent=2))


if __name__ == "__main__":
    main()
