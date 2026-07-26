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
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    _HELDOUT_OFF,
    _WEB_OFF,
)
from experiments.selfopt.broker import public_split  # noqa: E402
from experiments.selfopt.heldout import aggregate  # noqa: E402
from experiments.selfopt.sandbox.cuts import (  # noqa: E402
    Cut,
    CutSource,
    InvalidCut,
    home_sealed_for,
    materialize_cut,
    stream_audit,
    train_rows,
)
from experiments.selfopt.snapshot import (  # noqa: E402
    content_hash,
    copy_tree,
    home_skip,
    home_skip_with_context,
)  # noqa: E402
from experiments.selfopt.split import train_stream  # noqa: E402

NET = "selfopt-net"
BROKER_IMG = "selfopt-ab-broker:latest"
AGENT_IMG = "selfopt-ab-agent:latest"
SANDBOX = Path(__file__).resolve().parent

#: Whether the agent's MCP tools are deferred behind ``ToolSearch`` (``"true"``) or loaded up front
#: (``"false"``). PINNED, because it is not really a preference: the tool array is the FIRST thing
#: in a request, ahead of the system prompt and the conversation, so it is the front of the cached
#: prompt prefix. Two units whose tool arrays differ share no cached prefix at all, however
#: identical everything after it is.
#:
#: Left unset, the CLI decides per process, and the decision races the MCP server's connection
#: handshake: the same arm then sends two different tool arrays, i.e. two different ~900k-token
#: prefixes, and the first unit to send the minority one pays the full cache-creation price the
#: warm start already paid on the other. Observed at 2 units in 108 on a context arm, ~$8 each.
#: ``"true"`` is the branch these runs already take ~98% of the time, so pinning changes the tool
#: surface of almost no unit; what it removes is the coin flip.
TOOL_SEARCH = "true"


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
              append_system_prompt: Optional[str] = None,
              server_name: str = "tasks") -> int:
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
    # ``server_name`` keys the MCP server the agent sees (tools arrive as ``mcp__<name>__*``). It is
    # a parameter because a RESUMED probe must see the SAME server name its recorded context used —
    # an agent whose history is full of ``mcp__curriculum__*`` treats a differently-named server as
    # "my tools disconnected" and stops without playing the task.
    (cfg / ".mcp.json").write_text(json.dumps(
        {"mcpServers": {server_name: {"type": "http",
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
    # IS_SANDBOX=1: the container runs as root, where claude otherwise refuses bypassPermissions
    # ("cannot be used with root/sudo"); this flag is the sanctioned opt-in for isolated sandboxes.
    # ENABLE_TOOL_SEARCH: pin the tool array, which is the front of the cached prompt prefix every
    # unit of an arm shares (see TOOL_SEARCH). Unpinned it is decided per process, and one unit
    # deciding differently re-writes the whole prefix at full price.
    docker = ["docker", "run", "--rm", "--network", NET,
              "-e", f"CLAUDE_CODE_OAUTH_TOKEN={oauth}", "-e", "IS_SANDBOX=1",
              "-e", f"ENABLE_TOOL_SEARCH={TOOL_SEARCH}",
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

def resolve_checkpoints(prov: Path, self_dir: Path, home_dir: Path) -> dict:
    """Resolve, per checkpoint, the PAIR of (workdir self, native ~/.claude home) as they were at
    the SAME task boundary — the two halves of "the self" must be measured together.

    ``start`` = before task 1, ``mid`` = after ~half the stream: each pair is read off ONE
    ``results.jsonl`` row's ``self_hash_before`` + ``home_hash_before``, resolving each hash to the
    ARCHIVED snapshot dir the broker wrote to the provenance volume (the workdir self falls back to
    the live ``self_dir`` when its snapshot is absent). ``end`` = the LIVE final self_dir + live
    home_dir. A home hash with no archived snapshot — the empty seed before any memory was written —
    resolves to ``home=None`` (an empty home is mounted). Only READ here; the training self is never
    mutated."""
    results = prov / "results.jsonl"
    rows: List[dict] = []
    if results.exists():
        rows = [json.loads(x) for x in results.read_text().splitlines() if x.strip()]
        rows = sorted((r for r in rows if r.get("split") == "train"), key=lambda r: r["seq"])
    snaps = prov / "snapshots"

    def snap(h: object) -> Optional[Path]:
        d = snaps / h if isinstance(h, str) else None
        return d if (d and d.exists()) else None

    def at(row: dict) -> dict:
        # Pair the workdir self AND the home from the SAME boundary (this row's *_before hashes).
        home = snap(row.get("home_hash_before"))
        # An archived home dir that exists but is empty (the empty-seed hash) is equivalent to no
        # home — normalize it to None so callers uniformly mount an empty home.
        if home is not None and not any(home.rglob("*")):
            home = None
        return {"self": snap(row["self_hash_before"]) or self_dir, "home": home,
                "self_hash": row["self_hash_before"], "home_hash": row.get("home_hash_before")}

    out: dict = {}
    if rows:
        out["start"] = at(rows[0])
        out["mid"] = at(rows[len(rows) // 2])
    else:
        seed = {"self": self_dir, "home": None,
                "self_hash": content_hash(self_dir), "home_hash": None}
        out["start"], out["mid"] = dict(seed), dict(seed)
    live_home = home_dir if home_dir.exists() and any(home_dir.rglob("*")) else None
    out["end"] = {"self": self_dir, "home": live_home,
                  "self_hash": content_hash(self_dir),
                  "home_hash": content_hash(home_dir, skip=home_skip) if live_home else None}
    return out


def resolve_cut_checkpoint(prov: Path, self_dir: Path, cut: Cut) -> dict:
    """Resolve the (workdir self, memory home) pair for a transcript-cut arm — the SAME shape
    :func:`resolve_checkpoints` returns, so a cut arm drops into the held-out pass unchanged.

    Both halves come from the last task the agent had *completed* at the cut (its ``*_after``
    hashes), which is the nearest archived state to a mid-task compaction. Refuses rather than
    falls back if the paired memory home was never archived: an arm silently mounting an empty home
    is not the arm that was asked for.

    The memory home may deliberately come from a DIFFERENT task than the context — see
    :attr:`Cut.home_sealed`, which pins both members of a pre/post pair to one memory state so the
    contrast isolates context rather than bundling in whatever was written between them. The workdir
    self always tracks the cut's own task."""
    rows = _cut_row(prov, cut)
    snaps = prov / "snapshots"
    home_sealed = home_sealed_for(cut)
    home_row = rows if home_sealed == cut.expect_sealed else _cut_row(prov, cut, sealed=home_sealed)
    home_hash = home_row.get("home_hash_after")
    home = snaps / home_hash if isinstance(home_hash, str) else None
    if home is None or not home.exists() or not any(home.rglob("*")):
        raise InvalidCut(f"cut {cut.name!r} pairs memory home {home_hash!r} (after task "
                         f"{home_sealed}) but no archived snapshot holds it — the "
                         f"contemporaneous memory for this cut is unavailable.")
    self_hash = rows.get("self_hash_after")
    self_snap = snaps / self_hash if isinstance(self_hash, str) else None
    return {"self": self_snap if (self_snap and self_snap.exists()) else self_dir,
            "home": home, "self_hash": self_hash, "home_hash": home_hash}


def _cut_row(prov: Path, cut: Cut, sealed: Optional[int] = None) -> dict:
    """The provenance row of the last task completed at ``cut`` (its 1-indexed play order).

    ``sealed`` overrides which task to look up — used to resolve a pinned memory home from a
    different task than the context (see :attr:`Cut.home_sealed`)."""
    rows = train_rows(prov)
    want = cut.expect_sealed if sealed is None else sealed
    if not 1 <= want <= len(rows):
        raise InvalidCut(f"cut {cut.name!r} claims {want} sealed tasks but the run "
                         f"has {len(rows)} train rows")
    return rows[want - 1]


def build_filtered_home(src: Optional[Path], dst: Path, *, keep_context: bool = False,
                        drop_memory: bool = False, strip_thinking: bool = False,
                        cut: Optional[CutSource] = None) -> Path:
    """Materialize the DURABLE self-surface of a ``~/.claude`` source into a throwaway ``dst`` to be
    mounted read-write for a held-out probe.

    KEEPS the persistent memory/knowledge-base (``projects/*/memory/`` incl. ``MEMORY.md``),
    ``skills/``, ``CLAUDE.md`` and config (``settings*.json``); STRIPS the raw session transcript
    (``projects/*/*.jsonl`` — that is the training CONTEXT and must not leak into held-out), plus
    ``sessions/`` / ``backups/`` / ``telemetry/`` and machine-state files (``policy-limits.json``,
    ``remote-settings.json``, ``.last-cleanup``). The filter is :func:`snapshot.home_skip`, so the
    mounted home matches exactly what the broker archives. ``src=None`` (the empty seed) yields an
    empty home. ``src`` is only ever read.

    ``keep_context=True`` switches to :func:`snapshot.home_skip_with_context`, which ALSO keeps the
    conversation transcript — required by the context-loaded probe, whose agent ``--resume``s the
    training session so it boots with the original in-context history, not just the memory.
    ``drop_memory=True`` additionally withholds the memory FILES, isolating what the on-disk
    knowledge base adds over and above whatever the mounted surface already carries.

    ``cut`` writes a validated PREFIX of a training transcript into the home instead of (or on top
    of) whatever ``src`` carried, so the probe resumes the conversation as it stood part-way through
    the run rather than at its end. ``src`` is then the CONTEMPORANEOUS archived memory home for
    that point — the two are mounted together, and the cut is validated before anything is written
    (see :mod:`cuts`).
    ``strip_thinking=True`` drops every ``thinking`` block from the mounted transcript, so a
    resumed agent replays its own prior turns without the reasoning that produced them.
    """
    dst.mkdir(parents=True, exist_ok=True)
    if src is not None and Path(src).exists():
        base = home_skip_with_context if keep_context else home_skip
        skip = (lambda rel: base(rel) or "memory" in rel.parts) if drop_memory else base
        copy_tree(Path(src), dst, skip=skip)
    if cut is not None:
        materialize_cut(cut, dst)
    if strip_thinking:
        _strip_thinking(dst)
    return dst


def _strip_thinking(home: Path) -> dict:
    """Drop every ``thinking`` block from the transcripts mounted in ``home``, in place.

    The recorded thinking TEXT is already empty — the API returns none — so what this removes is
    the signature, which is 6-10% of a resumed conversation's tokens and is the only part of the
    model's reasoning the replay still carries. An arm that strips it and one that does not
    differ in whether the resumed agent has its own prior reasoning available, which is the
    contrast the arm exists to measure.

    Runs last, so it covers a transcript however it arrived: copied whole for a context arm, or
    written by :func:`materialize_cut` for a prefix arm. Every block here is a thinking-ONLY
    assistant turn — none shares a turn with a ``tool_use`` — so a dropped block never leaves a
    tool call unanswered."""
    blocks = files = 0
    for path in sorted(home.glob("projects/*/*.jsonl")):
        out, touched = [], False
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                out.append(line)
                continue
            content = (rec.get("message") or {}).get("content")
            if isinstance(content, list):
                kept = [b for b in content
                        if not (isinstance(b, dict) and b.get("type") == "thinking")]
                if len(kept) != len(content):
                    blocks += len(content) - len(kept)
                    touched = True
                    rec["message"]["content"] = kept
            out.append(json.dumps(rec, ensure_ascii=False))
        if touched:
            files += 1
            path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return {"thinking_blocks_dropped": blocks, "transcripts_rewritten": files}


def write_cut_manifest(cut: CutSource, stream_dir: Path, *, home_src: Optional[Path],
                       container: Optional[str] = None) -> dict:
    """Validate a cut and persist its MANIFEST — the audit trail that ties this arm's reported
    number back to an exact prefix of an exact file.

    Records what was cut (last retained record + uuid, record count, transcript sha256), what the
    model will see (tasks sealed at that point, context tokens), what memory rides along (the paired
    archived home + its content hash), and the container name the units will use — which is folded
    when it would exceed a DNS label, and folding is invisible in a log unless it is written down.
    Runs BEFORE any unit, into a throwaway dir: an invalid cut fails here, at zero cost."""
    stream_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        facts = materialize_cut(cut, Path(tmp))
    home_hash = content_hash(Path(home_src), skip=home_skip) if home_src is not None else None
    manifest = {**facts,
                "paired_home": str(home_src) if home_src is not None else None,
                "paired_home_hash": home_hash,
                "broker_container": container}
    (stream_dir / "cut_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


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

class HeldoutIncomplete(RuntimeError):
    """A held-out pass ended with units that hold NO valid sealed score.

    Raised only AFTER every unit has had its turn (so the scored units stay on disk and a re-run of
    the same command retries just the failures). Carries the per-unit reasons so the log says which
    unit died and why — a pass that cannot account for every requested task must never report a mean
    over the units that happened to survive."""

    def __init__(self, *, cp: str, requested: int, scored: int, failures: List[dict]) -> None:
        self.cp = cp
        self.requested = requested
        self.scored = scored
        self.failures = failures
        detail = "\n".join(
            f"  unit {f['unit']:03d} (task_idx {f['task_idx']}): {f['reason']}"
            + (f" | err: {f['stderr']}" if f.get("stderr") else "")
            for f in failures)
        super().__init__(
            f"held-out pass {cp!r} INCOMPLETE: {len(failures)} of {requested} units hold no valid "
            f"sealed score ({scored} scored) — refusing to average the survivors.\n{detail}\n"
            f"  the scored units are kept: re-run the SAME command to retry only the failed units.")


def _err_excerpt(path: Path, lines: int = 3, limit: int = 400) -> str:
    """The tail of a unit's agent stderr, where the real cause of a dead unit shows up (missing
    config / usage cap / MCP connect failure). Folded onto one line so it reads in a log."""
    if not path.exists():
        return ""
    try:
        tail = [ln.strip() for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines()
                if ln.strip()][-lines:]
    except OSError:
        return ""
    return " | ".join(tail)[-limit:]


def _unit_score(prov: Path, idx: int, split: str = "heldout") -> tuple[int, Optional[str]]:
    """Validate ONE unit's sealed provenance: exactly one scored row, and for the task this unit was
    asked to play (its broker is started with ``indices=[idx]``, so anything else means the row came
    from somewhere other than this unit's task). Returns ``(rows, reason-if-invalid)``; the broker
    owns the count, so this — not the agent's exit — is what "the unit happened" means."""
    rows = aggregate(prov / "results.jsonl", split=split)["rows"]
    if not rows:
        return 0, "no sealed score row"
    if len(rows) > 1:
        return len(rows), f"{len(rows)} sealed rows (expected exactly 1)"
    got = rows[0].get("task_idx")
    if got != idx:
        return 1, f"sealed row is for task_idx {got!r} (expected {idx})"
    return 1, None


def _aggregate_units(prov_dirs: List[Path], split: str = "heldout") -> dict:
    """Combine the per-unit result files of one held-out pass into a single aggregate. Each unit
    scores exactly one task into its OWN ``prov-<i>/results.jsonl`` (no shared-file contention
    between parallel units); this rolls them up by reusing :func:`aggregate` per unit."""
    n = 0
    reward_sum = 0.0
    success_sum = 0.0
    for pd in prov_dirs:
        agg = aggregate(pd / "results.jsonl", split=split)
        n += agg["n"]
        reward_sum += agg["mean_reward"] * agg["n"]
        success_sum += agg["success_rate"] * agg["n"]
    return {"n": n, "mean_reward": reward_sum / n if n else 0.0,
            "success_rate": success_sum / n if n else 0.0}


def _dns_safe(name: str, limit: int = 63) -> str:
    """Container names become DNS hostnames on the docker network, and a DNS label is capped at 63
    chars. A longer name does not error — it silently fails to RESOLVE, so the agent's MCP connect
    fails, it is handed ZERO tools, and it reports that its task server "disconnected". Keep a
    readable prefix and fold the overflow into a short stable hash."""
    if len(name) <= limit:
        return name
    return f"{name[:limit - 9]}-{hashlib.sha1(name.encode()).hexdigest()[:8]}"


def _heldout_unit(*, run_id: str, arm: str, cp: str, i: int, idx: int, src: Optional[Path],
                  home_src: Optional[Path], stream_dir: Path, disallowed: Optional[List[str]],
                  system: str, oauth: str, wandb_key: Optional[str], project: str,
                  resume: Optional[str] = None, keep_context: bool = False,
                  drop_memory: bool = False, strip_thinking: bool = False,
                  server_name: str = "tasks",
                  cut: Optional[CutSource] = None, prompt: str = KICKOFF) -> Optional[dict]:
    """One (checkpoint × task) held-out unit, fully self-contained so units run in parallel without
    colliding: its OWN broker container (``indices=[idx]`` — dispenses only that one task), its own
    throwaway workdir + filtered-home copy, its own per-unit provenance + stream file. Every name
    and path carries ``cp`` and ``i`` so two concurrent units never share a container, workdir,
    home, results file, or stream. The checkpoint self (``src``) and home (``home_src``) are only
    COPIED from — the training self/home are never mounted read-write.

    Returns ``None`` when the unit holds its one valid sealed score, else a failure record (unit,
    task_idx, agent exit code, reason, stderr excerpt) for the pass to refuse on."""
    prov = stream_dir / f"prov-{i:03d}"
    # Resume-friendly: if this unit already holds a sealed score, skip it — so a larger ``--n``
    # COMPLETES an earlier smaller pass (reusing its scored units) instead of re-running them. The
    # broker owns the count in ``prov-<i>/results.jsonl``; the position→task_idx mapping is stable
    # (``heldout[i]``), so unit ``i`` is the same task across passes. A pre-existing row that is NOT
    # this unit's one task is reported rather than re-run: re-running would only add a second row.
    scored, bad = _unit_score(prov, idx)
    if scored:
        # A sealed score is not enough for a cut arm: re-check the persisted stream, so a unit whose
        # context did not survive to the seal stays failed across re-runs instead of being counted
        # by the very skip that makes re-runs cheap. It stays failed and is NOT replayed: what
        # compacted it is the arm's own geometry (a ~1M-token prefix in a 1M window, this task's
        # tool traffic on top), so a replay buys the same verdict for another unit's worth of spend
        # — and, because the broker only ever APPENDS, it would leave the unit with two sealed rows
        # and a different complaint. The re-check instead re-stamps the row (idempotent), so an arm
        # run before the row carried its verdict acquires it on the next pass, without spending.
        bad = bad or _context_check(cut, stream_dir / f"stream-{i:03d}.jsonl", prov)
        return None if bad is None else {"unit": i, "task_idx": idx, "rc": None, "reason": bad,
                                         "stderr": _err_excerpt(stream_dir / f"stream-{i:03d}.err.txt")}
    bname = _dns_safe(f"selfopt-broker-{run_id}-{arm}-ho-{cp}-{i:03d}")
    work = stream_dir / f"work-{i:03d}"
    home = stream_dir / f"home-{i:03d}"
    try:
        start_broker(bname, split="heldout", prov=prov,
                     run_name=f"{run_id}-{arm}-heldout-{cp}", indices=[idx],
                     wandb_key=wandb_key, project=project)
        if src is not None:
            copy_tree(src, work)  # throwaway COPY — writes here never reach the training self
        # Mount the checkpoint's FILTERED memory home at the same project path the agent uses
        # (/work → slug -work), so MEMORY.md / memory recall loads; empty home for the seed.
        build_filtered_home(home_src, home, keep_context=keep_context, drop_memory=drop_memory,
                            strip_thinking=strip_thinking, cut=cut)
        # ``resume`` boots this probe with the training session's IN-CONTEXT HISTORY (its own
        # private copy of the transcript, in this unit's home) instead of a fresh context.
        rc = run_agent(work=work, prompt=prompt, append_system_prompt=system,
                       disallowed=disallowed, oauth=oauth, home=home, resume=resume,
                       server_name=server_name,
                       stream_path=stream_dir / f"stream-{i:03d}.jsonl", broker_name=bname)
    finally:
        _rm(bname)
    # A unit counts only if the broker sealed exactly one score for THIS task. The agent's exit code
    # cannot make a score exist, but a non-zero exit is the usual fingerprint of a unit that died
    # before playing (usage cap, missing config, no tools) — so it is reported alongside the reason.
    scored, bad = _unit_score(prov, idx)
    context = _context_check(cut, stream_dir / f"stream-{i:03d}.jsonl", prov)
    if bad is None and context is not None:
        # A unit whose mounted context did not survive to the seal is not a unit of THIS arm — the
        # sealed score is real, it just measures the other treatment. Fail it rather than average it.
        bad = context
    if bad is None:
        if rc != 0:
            print(f"[warn] held-out {cp} unit {i:03d} (task_idx {idx}): agent exited rc={rc} but "
                  f"its score is sealed — counting it", flush=True)
        return None
    return {"unit": i, "task_idx": idx, "rc": rc,
            "reason": f"agent exited rc={rc}, {bad}" if rc else bad,
            "stderr": _err_excerpt(stream_dir / f"stream-{i:03d}.err.txt")}


#: The verdict fields a failed context audit stamps onto the unit's own sealed row. They are
#: written ONLY on a row that failed — a clean row stays byte-for-byte what the broker wrote — so
#: the reader's idiom is ``row.get("context_ok", True)``: present and false == unusable for this
#: arm's contrast, absent == nothing objected to it.
CONTEXT_OK = "context_ok"
CONTEXT_REASON = "context_reason"


def flag_context(prov: Path, reason: str) -> int:
    """Stamp a failed context verdict onto the unit's sealed row(s), IN the file the broker wrote.

    The seal itself is untouched and stays on disk — the score is real, it just measures the other
    treatment. What is being recorded is precisely the part nothing in the row could otherwise
    reveal: the pass refuses to summarise such an arm, but a row lifted straight out of
    ``results.jsonl`` for a paired contrast is indistinguishable from a clean one, so it joins the
    analysis unnoticed (it did: 8 of them). Leaving the verdict in ``context_audit.json`` alone
    means a reader has to know to join against it; writing it here means a reader cannot reach a
    mixed row without also reading the field that condemns it.

    Additive, so existing readers keep working; one-way, since only the audit writes it; idempotent
    (a re-run re-derives the same verdict and rewrites nothing); and atomic, because this rewrites
    provenance the broker owns. Returns how many rows were newly stamped."""
    results = prov / "results.jsonl"
    if not results.exists():
        return 0
    out: List[str] = []
    stamped = 0
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:  # never drop a line this cannot parse
            out.append(line)
            continue
        if row.get(CONTEXT_OK) is False and row.get(CONTEXT_REASON) == reason:
            out.append(line)  # already stamped with this verdict — leave the bytes alone
            continue
        row[CONTEXT_OK] = False
        row[CONTEXT_REASON] = reason
        out.append(json.dumps(row))
        stamped += 1
    if stamped:
        tmp = results.with_suffix(".jsonl.tmp")
        tmp.write_text("".join(ln + "\n" for ln in out), encoding="utf-8")
        tmp.replace(results)
    return stamped


def audit_unit_context(cut: Optional[Cut], stream_path: Path, prov: Path) -> Optional[str]:
    """RAW STAYS RAW. A pre-compaction prefix boots ~900k tokens into a 1M window: if the unit
    auto-compacts before it seals, the model finished the task off its own summary and the unit has
    silently become the arm it was meant to be contrasted against. Re-read the unit's own stream,
    persist what it says (compaction events, the maximum model-visible context), and return a
    failure reason when a raw unit did not stay raw — so the pass refuses the arm instead of
    averaging a mixed treatment. A failing verdict is also stamped onto the unit's sealed row (see
    :func:`flag_context`), so the row says what the arm says. Returns ``None`` when there is nothing
    to object to.

    Takes the plain :class:`Cut` rather than a materialized :class:`CutSource`: the verdict needs
    only the arm's name and whether it is a raw segment, so an already-run arm can be re-audited
    from its persisted streams alone — no transcript, no prefix, no containers."""
    if cut is None:
        return None
    audit = stream_audit(stream_path)
    audit["arm"] = cut.name
    audit["raw_segment"] = cut.raw
    prov.mkdir(parents=True, exist_ok=True)
    (prov / "context_audit.json").write_text(json.dumps(audit, indent=2))
    if cut.raw and not audit["stayed_raw"]:
        reason = (f"raw prefix auto-compacted before the seal "
                  f"({audit['compactions_before_seal']} compaction event(s), max context "
                  f"{audit['max_context_tokens']} tokens) — this unit measures the summarised arm")
        flag_context(prov, reason)
        return reason
    return None


def _context_check(cut: Optional[CutSource], stream_path: Path, prov: Path) -> Optional[str]:
    """:func:`audit_unit_context` for a unit that has just run, whose cut is the materialized one."""
    return audit_unit_context(cut.cut if cut is not None else None, stream_path, prov)


def _warm_start_split(units: List[tuple[int, int]], *, stream_dir: Path,
                      concurrency: int) -> tuple[Optional[tuple[int, int]], List[tuple[int, int]]]:
    """Pick the ONE unit to run alone first so the rest of the pass reads a warm prompt cache.

    Every unit of an arm boots the same conversation prefix (same tool array, same system prompt,
    same ``--resume`` transcript; the tool array is pinned by ``TOOL_SEARCH`` precisely so that
    stays true), so the first unit to send it pays to WRITE that prefix into the prompt cache and
    every later unit only pays to READ it. Launching ``concurrency`` units simultaneously means
    ``concurrency`` simultaneous misses: each one writes the whole prefix, and only the last write
    is of any use to anybody. Measured over a 4-wide context arm, each of the four openers wrote
    ~1M cache-creation tokens and cost ~$15.73 against a ~$6.81 steady state: ~$8.92 of avoidable
    spend per extra cold unit, ~$27 per arm. Running one unit to completion first collapses the
    burst to a single write; that unit re-reads the prefix on every turn of its own episode, so the
    entry is fresh (a cache read renews the TTL) when the followers launch moments later.

    The warming unit must be the first unit that will ACTUALLY run: a pass is re-run to retry its
    failures and :func:`_heldout_unit` skips any unit that already holds its sealed score, so
    warming on unit 0 of a resumed pass can warm nothing at all. Returns ``(None, units)`` — run
    everything in one batch, exactly as before — whenever a warm start would buy nothing: a serial
    pass, and a pass with at most one unit left to run (the warming round trip IS that unit)."""
    if concurrency <= 1:
        return None, units
    # The same condition the unit itself short-circuits on, and read-only: it counts the sealed rows
    # in the unit's own provenance without touching them.
    todo = [k for k, (i, idx) in enumerate(units)
            if not _unit_score(stream_dir / f"prov-{i:03d}", idx)[0]]
    if len(todo) < 2:
        return None, units
    k = todo[0]
    # Everything else keeps its original order, including the already-scored units ahead of the
    # warming one: they launch no agent, so their placement costs nothing.
    return units[k], units[:k] + units[k + 1:]


def _heldout_pass(*, run_id: str, arm: str, cp: str, indices: List[int], src: Optional[Path],
                  home_src: Optional[Path], stream_dir: Path, disallowed: Optional[List[str]],
                  system: str, oauth: str, wandb_key: Optional[str], project: str,
                  concurrency: int = 1, resume: Optional[str] = None,
                  keep_context: bool = False, drop_memory: bool = False,
                  strip_thinking: bool = False,
                  server_name: str = "tasks", cut: Optional[CutSource] = None,
                  prompt: str = KICKOFF) -> dict:
    """One held-out pass over ``indices``: a FRESH broker + agent per task, each mounting a throwaway
    copy of the checkpoint's workdir self (``src``; ``None`` = empty workdir for control) AND its
    FILTERED memory home (``home_src``; ``None`` = empty seed home). The (checkpoint × task) units
    are independent, so they run through a bounded ``ThreadPoolExecutor`` (``concurrency`` at a
    time); each unit is a blocking ``run_agent``/``docker run``. Results accumulate per-unit and are
    rolled up at the end. No cross-task in-context carryover, and the training self is never handed
    to a held-out pass.

    Every requested unit must end holding its one sealed score: a unit that died (non-zero exit, no
    row, a row for another task) is collected as a failure and the pass raises
    :class:`HeldoutIncomplete` — AFTER all units have run, so the scored ones are on disk and a
    re-run of the same command completes the arm instead of restarting it.

    ``cut`` makes this a transcript-prefix arm: the prefix is validated and its MANIFEST written
    BEFORE any unit is launched (an invalid cut costs nothing), and every unit then mounts its own
    copy of that prefix beside the contemporaneous memory home.

    The units all boot the SAME cached conversation prefix, so the pass starts with a WARM START:
    the first unit that will actually run goes alone, and only once it has finished does the rest of
    the pass go wide (see :func:`_warm_start_split` for the cache economics). Which units run, what
    they are handed, and how their outcomes are collected are untouched by that split."""
    failures: List[dict] = []
    if cut is not None:
        write_cut_manifest(cut, stream_dir, home_src=home_src,
                           container=_dns_safe(f"selfopt-broker-{run_id}-{arm}-ho-{cp}-000"))
    # Every unit is submitted with the same arguments but its own (position, task) pair.
    unit_kw = dict(run_id=run_id, arm=arm, cp=cp, src=src, home_src=home_src,
                   stream_dir=stream_dir, disallowed=disallowed, system=system, oauth=oauth,
                   wandb_key=wandb_key, project=project, resume=resume, keep_context=keep_context,
                   drop_memory=drop_memory, strip_thinking=strip_thinking, server_name=server_name,
                   cut=cut, prompt=prompt)
    warm, rest = _warm_start_split(list(enumerate(indices)), stream_dir=stream_dir,
                                   concurrency=concurrency)
    if warm is not None:
        print(f"==> held-out {cp}: unit {warm[0]:03d} runs alone first to warm the shared "
              f"conversation prefix in the prompt cache; the other {len(rest)} unit(s) follow "
              f"<= {concurrency} at a time", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        # Batches run back-to-back through the SAME pool, and a batch is fully drained before the
        # next is submitted — that drain is the whole warm start. It also means a warming unit that
        # DIES does not abort the arm: its failure is collected like any other, and the remaining
        # units still run (a cold follower is the worst case, which is where this started).
        for batch in ([[warm], rest] if warm is not None else [rest]):
            futures = {pool.submit(_heldout_unit, i=i, idx=idx, **unit_kw): (i, idx)
                       for i, idx in batch}
            for fut in as_completed(futures):
                i, idx = futures[fut]
                try:
                    bad = fut.result()
                except Exception as exc:  # a unit that died before it could report on itself
                    bad = {"unit": i, "task_idx": idx, "rc": None,
                           "reason": f"{type(exc).__name__}: {exc}",
                           "stderr": _err_excerpt(stream_dir / f"stream-{i:03d}.err.txt")}
                if bad is not None:
                    failures.append(bad)
    agg = _aggregate_units([stream_dir / f"prov-{i:03d}" for i in range(len(indices))])
    if agg["n"] != len(indices) and not failures:
        # Belt-and-braces: the rolled-up count must equal the requested count even if every unit
        # reported itself healthy.
        failures.append({"unit": -1, "task_idx": -1, "rc": None, "stderr": "",
                         "reason": f"rolled-up n={agg['n']} != {len(indices)} requested"})
    if failures:
        raise HeldoutIncomplete(cp=cp, requested=len(indices), scored=agg["n"],
                                failures=sorted(failures, key=lambda f: f["unit"]))
    return agg


def phase_treatment(run_id: str, stream: List[int], heldout_size: int, *, oauth: str,
                    wandb_key: Optional[str], project: str, root: Path,
                    session_id: str, resume: bool = False,
                    arm_label: str = "treatment",
                    system_prompt: Optional[str] = None) -> dict:
    # Drives the persistent-session treatment arm. ``arm_label`` names the run subdir + every
    # container (kept parametric so runs stay run-scoped and can go concurrently without collision);
    # ``system_prompt`` is the standing objective ("Get Better" + the task-stream loop).
    system_prompt = system_prompt or TREATMENT_PROMPT
    rd = root / arm_label
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
    bname = f"selfopt-broker-{run_id}-{arm_label}-train"
    session_mode: Optional[str] = None
    try:
        start_broker(bname, split="train", prov=prov, run_name=f"{run_id}-{arm_label}-train",
                     self_ro=self_dir, home_ro=self_home, queue_size=len(stream),
                     wandb_key=wandb_key, project=project, resume=resume)
        if not resume:
            print(f"==> [{arm_label}] train: persistent 'Get Better' agent over the stream",
                  flush=True)
            run_agent(work=self_dir, prompt=KICKOFF, append_system_prompt=system_prompt,
                      oauth=oauth, stream_path=stream_path, broker_name=bname, home=self_home,
                      session_id=session_id)
            session_mode = "initial"
        else:
            before = train_scored(prov)
            sid = read_session_id(root, stream_path) or session_id
            print(f"==> [{arm_label}] train RESUME: continue session {sid or '(none captured)'} from "
                  f"task {before + 1} (broker skipped {before} already-scored)", flush=True)
            if sid:
                rc = run_agent(work=self_dir, prompt=CONTINUE_PROMPT,
                               append_system_prompt=system_prompt,
                               oauth=oauth, stream_path=stream_path, broker_name=bname,
                               home=self_home, resume=sid, append=True)
                after = train_scored(prov)
                if rc == 0 or after > before:
                    session_mode = "resumed-session"
                    print(f"==> [{arm_label}] train RESUME: SAME session continued "
                          f"(scored {before} -> {after})", flush=True)
            if session_mode is None:
                # The session could not be revived (none captured, or the resume failed) — WARM
                # RESTART: a fresh ``claude -p`` on the SAME persisted self + native home (the
                # on-disk self is kept; only the in-context conversation is lost). The broker still
                # dispenses only the unscored tasks.
                print(f"==> [{arm_label}] train RESUME: session not revivable — WARM RESTART "
                      "(fresh claude -p; self/ + self_home kept)", flush=True)
                run_agent(work=self_dir, prompt=KICKOFF, append_system_prompt=system_prompt,
                          oauth=oauth, stream_path=stream_path, broker_name=bname, home=self_home,
                          append=True)
                session_mode = "warm-restart"
    finally:
        _rm(bname)
    train = aggregate(prov / "results.jsonl", split="train")

    # -- held-out curve at start / mid / end (web+bash-off), each a throwaway copy of the checkpoint
    #    self AND its filtered memory home, paired at the same task boundary. The training self is
    #    never handed to a held-out pass.
    cps = resolve_checkpoints(prov, self_dir, self_home)
    pre = content_hash(self_dir)
    indices = list(public_split().heldout)[:heldout_size]
    curve: dict = {}
    for cp in ("start", "mid", "end"):
        info = cps[cp]
        print(f"==> [{arm_label}] held-out {cp}: web/bash-off probe of self {info['self_hash']} "
              f"+ home {info['home_hash']}", flush=True)
        agg = _heldout_pass(run_id=run_id, arm=arm_label, cp=cp, indices=indices,
                            src=info["self"], home_src=info["home"],
                            stream_dir=rd / "heldout" / cp,
                            disallowed=_HELDOUT_OFF, system=HELDOUT_PROMPT, oauth=oauth,
                            wandb_key=wandb_key, project=project)
        curve[cp] = {"mean_reward": agg["mean_reward"], "n": agg["n"],
                     "measured_self": info["self_hash"], "measured_home": info["home_hash"]}
    assert content_hash(self_dir) == pre, "held-out mutated the training self — isolation broken"
    return {"train": {"n": train["n"], "mean_reward": train["mean_reward"]},
            "heldout_curve": curve, "self_unchanged_by_heldout": True,
            "session_mode": session_mode}


def phase_control(run_id: str, stream: List[int], heldout_size: int, *, oauth: str,
                  wandb_key: Optional[str], project: str, root: Path) -> dict:
    rd = root / "control"
    prov = rd / "prov"
    # -- train: a FRESH agent per task (empty workdir, no persistence, no "Get Better").
    bname = f"selfopt-broker-{run_id}-control-train"
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
                       home_src=None, stream_dir=rd / "heldout",
                       disallowed=_HELDOUT_OFF, system=CONTROL_PROMPT, oauth=oauth,
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
        # The persistent-session (treatment) arm lives in a subdir named for the arm; the session id
        # to revive is under that subdir.
        sess_sub = arm if arm == "treatment" else "treatment"
        session_id = read_session_id(root, root / sess_sub / "stream.jsonl") or ""
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
                                                session_id=session_id, resume=resuming,
                                                arm_label="treatment",
                                                system_prompt=TREATMENT_PROMPT)
        if arm in ("control", "both"):
            out["control"] = phase_control(run_id, stream, heldout_size, oauth=oauth,
                                           wandb_key=wandb_key, project=project, root=root)
    finally:
        # Sweep only THIS run's broker containers (train + per-task held-out), scoped by run_id — so
        # a concurrently-running study's brokers are never touched.
        leftover = subprocess.run(["docker", "ps", "-aq", "--filter",
                                   f"name=selfopt-broker-{run_id}"],
                                  capture_output=True, text=True).stdout.split()
        if leftover:
            subprocess.run(["docker", "rm", "-f", *leftover], capture_output=True)
    print(json.dumps({"result": out}, indent=2))


if __name__ == "__main__":
    main()
