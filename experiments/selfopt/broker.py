"""The curriculum broker over AutomationBench (issue #57; adapts the ``selfopt-poc`` broker
from Wordle to AutomationBench).

The broker **is the environment the agent talks to**. It:

  - **Dispenses** a non-repeating stream of AutomationBench **public train** tasks
    (``get_task`` pops the next queued train index and starts a hidden :class:`ServedEpisode`),
    returning only the task framing ``{env, instructions, budget, tools}`` — never the task
    index, the target end-state, or a handle.
  - **Holds out** a disjoint subset (the private-set proxy) that is *never* placed in the train
    queue (:mod:`.split`), so training can't touch it. A held-out broker (``SELFOPT_SPLIT=heldout``)
    dispenses that pool for the authoritative generalization eval.
  - **Serves** AutomationBench's native tools (``api_search`` / ``api_fetch`` / ``base64_encode``
    / ``done`` / ``terminate``) directly, routed to the active episode.
  - **Scores authoritatively.** AutomationBench is a ``score``-terminal env: calling ``done``
    atomically **seals** the episode and runs ``finalize``, which scores the sealed ``WorldState``
    server-side and returns core-owned :class:`TerminalEvidence` (RFC-009, #52). The broker reads
    the score off that evidence via the episode's ``terminal_feedback`` — it can never be forged
    by the agent, and because AutomationBench's scoring is deterministic + offline it needs **no
    OpenAI key**. The broker (not the agent) writes the provenance log and the self-snapshots.

Integrity holds by construction: the queue is built from the **train pool only**; held-out
indices are out of the queue, and the agent never sees an index or target.

Run it as an MCP server (spawned from a ``.mcp.json`` or in its own container)::

    SELFOPT_SPLIT=train  python -m experiments.selfopt.broker      # stdio
    SELFOPT_HTTP=1 SELFOPT_SPLIT=train python -m experiments.selfopt.broker  # http :9000
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP
from fastmcp.tools import ToolResult

from hgym.envs import make
from hgym.serve.episode import ServedEpisode
from hgym.serve.server import _build_tool

from . import config
from .sink import MetricSink, make_sink
from .snapshot import home_skip, snapshot
from .split import Split, make_split, train_stream

# --- Task universe + split (computed once, cheaply cached) ----------------------------

_ENV_CACHE: Dict[str, Any] = {}


def _public_env():
    """Construct (and cache) the AutomationBench public env — used for num_tasks and the tool
    manifest. Constructing it provisions the pinned upstream source on a cold cache."""
    if "env" not in _ENV_CACHE:
        _ENV_CACHE["env"] = make(config.ENV_NAME, config=config.ENV_CONFIG)
    return _ENV_CACHE["env"]


def public_split() -> Split:
    """The deterministic, disjoint train/held-out partition of the public tasks."""
    return make_split(_public_env().num_tasks)


def _score_terminal_name(env) -> Optional[str]:
    """The env's ``score``-terminal tool name (AutomationBench: ``done``), or None."""
    for m in env.describe().tools:
        if m.terminal_kind == "score":
            return m.name
    return None


# --- Broker ---------------------------------------------------------------------------

class Broker:
    """A single active episode at a time (async single-threaded), dispensing one split.

    ``split_name`` is ``"train"`` (the learning stream) or ``"heldout"`` (the authoritative
    generalization set). ``indices`` overrides the queue explicitly (tests / a capped held-out
    checkpoint); otherwise it is derived from the split.
    """

    def __init__(
        self,
        split_name: str = "train",
        *,
        queue_size: Optional[int] = None,
        indices: Optional[List[int]] = None,
        prov_dir: Optional[Path] = None,
        self_dir: Optional[Path] = None,
        home_dir: Optional[Path] = None,
        sink: Optional[MetricSink] = None,
        resume: bool = False,
    ) -> None:
        self.split_name = split_name
        self.split = public_split()
        if indices is not None:
            self.queue: List[int] = list(indices)
        elif split_name == "train":
            size = queue_size if queue_size is not None else int(
                os.environ.get("SELFOPT_QUEUE_SIZE", "15")
            )
            self.queue = train_stream(self.split, size)
        elif split_name == "heldout":
            cap = queue_size if queue_size is not None else int(
                os.environ.get("SELFOPT_HELDOUT_SIZE", str(len(self.split.heldout)))
            )
            self.queue = list(self.split.heldout)[:cap]
        else:
            raise ValueError(f"unknown split {split_name!r} (want 'train' or 'heldout')")

        self.prov_dir = Path(prov_dir or os.environ.get(
            "SELFOPT_PROV_DIR", config.RUNS_DIR / "provenance"
        ))
        self.snap_dir = self.prov_dir / "snapshots"
        self.results_path = self.prov_dir / "results.jsonl"
        # The agent's "self": the whole workdir. Snapshotted at task boundaries (external,
        # tamper-evident). Optional — held-out eval may not want per-task snapshots.
        self.self_dir = Path(self_dir) if self_dir is not None else (
            Path(os.environ["SELFOPT_SELF_DIR"]) if os.environ.get("SELFOPT_SELF_DIR") else None
        )
        # The OTHER half of "the self": the agent's native Claude Code home (``~/.claude``), where
        # the CLI persists memory/skills/settings. It is mounted (read-only) into this broker so
        # its durable self-surface is snapshotted at task boundaries too — the workdir and the home
        # dir TOGETHER compose the self. Only set for the persistent train phase (the sole session
        # with a persisted home); held-out/control leave it None (ephemeral per-container home).
        self.home_dir = Path(home_dir) if home_dir is not None else (
            Path(os.environ["SELFOPT_HOME_DIR"]) if os.environ.get("SELFOPT_HOME_DIR") else None
        )
        # The metric sink is BROKER-side (the scorer owns the authoritative number). Optional —
        # tests/in-proc use pass None and stay silent; the container passes a WandbSink so the
        # training-feedback metrics + self-snapshots stream live as each task seals.
        self.sink = sink

        self.consumed = 0
        self.seq = 0
        self.active: Optional[Dict[str, Any]] = None
        self._lock = asyncio.Lock()
        self._score_terminal = _score_terminal_name(_public_env())

        # RESUME: rebuild the SAME seeded stream (above) but start dispensing at the next UNscored
        # task. The provenance volume persists across an interrupted run, so already-scored train
        # tasks are exactly the train rows in results.jsonl — drop them from the queue (the stream
        # is non-repeating, so a scored index never recurs) and advance the seq counter past them so
        # snapshots/metrics keep numbering continuously. No task is re-run, none is skipped, and the
        # authoritative scoring is unchanged. Held-out/indices queues are per-checkpoint and fresh,
        # so resume only ever rewrites the train stream.
        if resume and split_name == "train" and self.results_path.exists():
            scored = {
                json.loads(line)["task_idx"]
                for line in self.results_path.read_text().splitlines()
                if line.strip() and json.loads(line).get("split") == "train"
            }
            if scored:
                remaining = [i for i in self.queue if i not in scored]
                skipped = len(self.queue) - len(remaining)
                self.queue = remaining
                self.seq = skipped
                self.consumed = skipped

    # ----- provenance -----

    def _snapshot_self(self) -> Optional[str]:
        if self.self_dir is None:
            return None
        return snapshot(self.self_dir, self.snap_dir)

    def _snapshot_home(self) -> Optional[str]:
        """Snapshot the durable self-surface of ``~/.claude`` (memory/skills/settings), skipping
        transient logs/caches and — defense in depth — any credential file. None when no home dir
        is configured or it hasn't been created yet."""
        if self.home_dir is None or not self.home_dir.exists():
            return None
        return snapshot(self.home_dir, self.snap_dir, skip=home_skip)

    def _append_result(self, row: Dict[str, Any]) -> None:
        self.results_path.parent.mkdir(parents=True, exist_ok=True)
        with self.results_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ----- lifecycle -----

    async def _finalize(self, active: Dict[str, Any]) -> Dict[str, Any]:
        """End + authoritatively score the active episode, snapshot the self, record the row."""
        if active["finalized"]:
            return active["result"]
        ep: ServedEpisode = active["episode"]
        # AutomationBench is a score-terminal env: if the agent never called `done`, score its
        # *current partial state* by driving the score terminal on its behalf (the seal + rubric
        # give partial credit for whatever assertions it satisfied) — NOT `terminate`, which is a
        # no-score abort. If there is no score terminal (other envs), fall back to terminate.
        if not ep.terminated:
            try:
                if self._score_terminal is not None:
                    await ep.call(self._score_terminal, {})
                else:
                    await ep.call("terminate", {})
            except Exception:
                pass
        # Read the AUTHORITATIVE terminal feedback off the (sealed) episode — never a value the
        # agent supplied. AutomationBench emits reward == partial_credit, plus success.
        fb = {item["name"]: item["value"] for item in ep.terminal_feedback}
        reward = float(fb.get("reward", fb.get("partial_credit", 0.0)))
        success = bool(fb.get("success", False))
        self_hash_after = self._snapshot_self()
        home_hash_after = self._snapshot_home()
        result = {"reward": reward, "success": success, "feedback": fb}
        self._append_result({
            "seq": active["seq"],
            "handle": active["handle"],
            "split": self.split_name,
            "env": active["env"],
            "task_idx": active["idx"],           # provenance only — NEVER returned to the agent
            "self_hash_before": active["self_hash_before"],
            "self_hash_after": self_hash_after,
            "home_hash_before": active["home_hash_before"],
            "home_hash_after": home_hash_after,
            "reward": reward,
            "success": success,
            "feedback": fb,
        })
        # Stream the authoritative training-feedback LIVE as this task seals (broker-side, so the
        # number can't be forged). LocalSink by default; WandbSink when the container is keyed.
        if self.sink is not None:
            split = self.split_name
            self.sink.metric(f"{split}/reward", reward, step=active["seq"])
            self.sink.metric(f"{split}/success", float(success), step=active["seq"])
            self.sink.log({"event": f"{split}_task_scored", "seq": active["seq"],
                           "reward": reward, "success": success, "feedback": fb,
                           "self_hash_before": active["self_hash_before"],
                           "self_hash_after": self_hash_after,
                           "home_hash_before": active["home_hash_before"],
                           "home_hash_after": home_hash_after})
            # The self's evolution: log the whole-workdir snapshot as an artifact (the diffs are
            # the narrative). Snapshots live on the broker-only provenance volume.
            if self_hash_after is not None:
                snap = self.snap_dir / self_hash_after
                if snap.exists():
                    self.sink.artifact(snap, name=f"self-{active['seq']:03d}-{self_hash_after}",
                                       kind="workdir")
            # The OTHER half of the self: the native Claude Code home (~/.claude) snapshot —
            # memory/skills the CLI persisted, credential-free (see snapshot.home_skip).
            if home_hash_after is not None:
                home_snap = self.snap_dir / home_hash_after
                if home_snap.exists():
                    self.sink.artifact(home_snap,
                                       name=f"home-{active['seq']:03d}-{home_hash_after}",
                                       kind="claude-home")
        try:
            await ep.close()
        except Exception:
            pass
        active["finalized"] = True
        active["result"] = result
        return result

    async def get_task(self) -> Dict[str, Any]:
        async with self._lock:
            # Pulling a new task while one is unfinished == abandoning it: score it first so
            # every dispensed task lands exactly one result row.
            if self.active is not None and not self.active["finalized"]:
                await self._finalize(self.active)
            if not self.queue:
                return {"done": True, "remaining": 0, "consumed": self.consumed}

            idx = self.queue.pop(0)
            self.seq += 1
            self_hash = self._snapshot_self()  # the self (workdir) as it will play this task
            home_hash = self._snapshot_home()  # the self (native ~/.claude) as it will play this
            ep = await ServedEpisode.start(config.ENV_NAME, task=idx, env_config=config.ENV_CONFIG)
            handle = uuid.uuid4().hex[:8]
            self.active = {
                "handle": handle, "env": config.ENV_NAME, "idx": idx, "episode": ep,
                "self_hash_before": self_hash, "home_hash_before": home_hash, "seq": self.seq,
                "finalized": False, "result": None,
            }
            self.consumed += 1
            spec = ep.describe()
            return {  # ONLY the framing — no index, no target, no handle.
                "env": spec.env_name,
                "instructions": spec.instructions,
                "budget": spec.horizon,
                "tools": [
                    {"name": t.name, "description": t.description, "input_schema": t.input_schema}
                    for t in spec.tools
                ],
                "remaining_after": len(self.queue),
            }

    async def dispatch(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """Route a native tool call to the active episode; finalize on termination."""
        async with self._lock:
            active = self.active
            if active is None or active["finalized"]:
                return ToolResult(content=json.dumps(
                    {"error": "No active task. Call `get_task` first."}))
            ep: ServedEpisode = active["episode"]
            call = await ep.call(tool_name, arguments or {})
            payload: Dict[str, Any] = {"content": call.content, "terminated": call.terminated}
            if call.terminated:
                payload["result"] = await self._finalize(active)
                payload["hint"] = "Task over. Call `get_task` for the next one."
            return ToolResult(content=json.dumps(payload))

    def queue_info(self) -> Dict[str, Any]:
        return {
            "split": self.split_name,
            "remaining": len(self.queue),
            "consumed": self.consumed,
            "active": self.active is not None and not self.active["finalized"],
        }


# --- FastMCP server -------------------------------------------------------------------

def build_broker_server(broker: Broker) -> FastMCP:
    server: FastMCP = FastMCP(name="tasks")

    @server.tool
    async def get_task() -> Dict[str, Any]:
        """Pop the next queued task and start its (hidden) episode.

        Returns the task framing — ``{env, instructions, budget, tools}`` — and NEVER the
        target end-state, task index, or a handle. When the queue drains, returns
        ``{done: true}``. Carry out the returned task with the native tools it lists
        (``api_search`` / ``api_fetch`` / ``base64_encode``), then call ``done`` to submit +
        score it; they route to the active task automatically."""
        return await broker.get_task()

    @server.tool
    async def queue_info() -> Dict[str, Any]:
        """Report ``{split, remaining, consumed, active}`` for the curriculum queue."""
        return broker.queue_info()

    # Expose AutomationBench's native tools directly, routed to the active episode. Probing the
    # env's real manifest keeps the advertised JSON-Schema exact (not hardcoded).
    registered: Dict[str, Any] = {}
    for manifest in _public_env().describe().tools:
        if manifest.name in registered:
            continue
        registered[manifest.name] = manifest
        server.add_tool(_build_tool(manifest, broker.dispatch))
    return server


def _indices_from_env() -> Optional[List[int]]:
    """An explicit queue from ``SELFOPT_INDICES=3,7,42`` (control dispenses one train task per
    fresh process). The indices must belong to the requested split — enforced below."""
    spec = os.environ.get("SELFOPT_INDICES")
    if not spec:
        return None
    return [int(x) for x in spec.split(",") if x.strip()]


def main() -> None:
    split_name = os.environ.get("SELFOPT_SPLIT", "train")
    indices = _indices_from_env()
    if indices is not None:
        # Integrity: an explicit queue may only draw from the named split's pool, so a held-out
        # index can never be dispensed as "train" (and vice versa).
        allowed = set(make_split(_public_env().num_tasks).train if split_name == "train"
                      else make_split(_public_env().num_tasks).heldout)
        bad = [i for i in indices if i not in allowed]
        if bad:
            raise SystemExit(f"SELFOPT_INDICES {bad} are not in the {split_name!r} pool")
    # Broker-side metric sink. SELFOPT_WANDB=1 + WANDB_API_KEY → WandbSink (live streaming);
    # otherwise LocalSink (JSONL under the provenance volume). make_sink degrades to LocalSink
    # with a warning if wandb/key are missing — an unkeyed run never crashes. The run name ties
    # every arm/checkpoint of one study together (override with SELFOPT_RUN_NAME).
    run_name = os.environ.get("SELFOPT_RUN_NAME", f"broker-{split_name}")
    sink = make_sink(run_name)
    resume = os.environ.get("SELFOPT_RESUME", "").lower() in ("1", "true", "yes")
    broker = Broker(split_name, indices=indices, sink=sink, resume=resume)
    server = build_broker_server(broker)
    if os.environ.get("SELFOPT_HTTP"):
        server.run(
            transport="http",
            host=os.environ.get("SELFOPT_HOST", "0.0.0.0"),
            port=int(os.environ.get("SELFOPT_PORT", "9000")),
        )
    else:
        server.run(transport="stdio")


if __name__ == "__main__":
    main()
