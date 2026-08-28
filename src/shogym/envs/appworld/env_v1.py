"""``appworld`` on the env-as-center core: one AppWorld task, plus one authored chore, as one task.

AppWorld is a simulated phone and its nine apps, with a person's whole digital life in the
databases behind them and a natural-language instruction that takes tens of API calls to carry
out. This port rents all of that and adds one paragraph to the end of every instruction.

The paragraph asks the agent to keep a filing log in a corner of the world no scenario touches,
and the values it asks for are computed from the world's own data by a house convention nobody
states. Four choices are left open and every one of them is constructible from what the world
shows; none is named. So the base task measures whether the agent can operate the world, and the
appended chore measures something the base task cannot: whether the agent can work out an unstated
rule, and what a grade on one attempt is worth to it on the next.

  - **describe**: a :class:`TaskSpec` whose instructions are the task's own, the world's own
    conventions for driving it, and the appended paragraph, byte-identical on every task.
  - **serve**: ``execute`` runs Python against the world; ``submit`` is the ``score`` terminal.
  - **finalize + verify**: ``submit`` seals the episode, then ``finalize`` reads the end state,
    scores the filing against the drawn key, and publishes the three payload classes.

**Repeats are legal and a repeat is a repeat.** A task addressed by index is the same world every
time it is asked for, down to the state of the generator the world draws from: the backlog is a
deterministic function of the task, the key is a deterministic function of the task and the draw,
and the process's randomness is put aside at the start of an episode and handed back at the end.
AppWorld saves databases and not generator state, so a port that only replayed the databases would
serve two worlds that agreed on their contents and disagreed on their next draw.

**One world, one process.** AppWorld freezes the clock for the whole interpreter and holds every
app's database engine on a class attribute, so two worlds in one process are one world being
unfrozen by the other. Each episode gets a worker process on its own loopback port, gated by a
token the agent never sees (:mod:`shogym.envs.appworld.worker`).

This module imports **nothing** from upstream at load time, so ``import shogym`` (which imports it
to register the env) stays offline. The corpus and the app sources are provisioned when an
``appworld`` env is *constructed*; see :mod:`shogym.envs.appworld.adapter`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import zlib
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from shogym.core import Env
from shogym.envs.appworld import adapter, payload, world
from shogym.envs.appworld.ledger import build_backlog
from shogym.envs.appworld.scorer import Verdicts, draw_key, leg_of, score
from shogym.envs.registration import register
from shogym.feedback.wire import NOTICE_FEEDBACK_NAME, REPORT_FEEDBACK_NAME
from shogym.mcp import MCPServerSpec
from shogym.task import TaskSpec
from shogym.trajectory import Trajectory
from shogym.types import (
    EpisodeFeedback,
    FeedbackCollection,
    FunctionConfig,
    InferenceFeedback,
)

if TYPE_CHECKING:
    from shogym.serve.lifecycle import FinalizeRequest, TerminalEvidence

EXECUTE_TOOL_NAME = "execute"

# The env's `score` terminal. Calling it seals the episode and only then scores it, so a verdict
# never exists for a world that can still be written to.
SUBMIT_TOOL_NAME = "submit"

#: How many blocks of code an episode may run. Gold solutions for this split make a minimum of 5
#: API calls, a median of 25 and a maximum of 649, and a block may make as many calls as it likes,
#: so this is a budget on turns rather than on work.
DEFAULT_HORIZON = 60

#: The draw. It fixes the convention and the four stored slots for every task, and nothing else in
#: the port reads it. Two runs that share it are graded against the same rules; two that do not
#: are two different experiments and their scores are not comparable.
DEFAULT_PULSE = 0

APPWORLD_SPEC = MCPServerSpec(
    name="appworld",
    transport="in_process",
    module="shogym.envs.appworld.mcp_server",
)

_TOOL_GUIDE = """\
# Tools
- `execute(code)`: run one block of Python in the world. The shell persists between calls, so a
  name you bind in one block is still bound in the next. Only what you `print` comes back.
- `submit()`: end the task and record what you did. Call it once the instruction is carried out
  and the filing log is updated. There is no second submission and no need to call `terminate`
  afterward."""

_WORLD_GUIDE = """\
# The world
You drive nine apps through one object, `apis`, which is already bound in your shell. Start from
its documentation, which is part of the world rather than something to guess at:

```python
print(apis.api_docs.show_app_descriptions())
print(apis.api_docs.show_api_descriptions(app_name="todoist"))
print(apis.api_docs.show_api_doc(app_name="todoist", api_name="create_task"))
```

Most APIs need an access token. Your supervisor's own accounts are the ones to use:

```python
print(apis.supervisor.show_profile())
print(apis.supervisor.show_account_passwords())
```

Log in to an app with the supervisor's email and that app's password, and pass the token you get
back to the app's other APIs.

When the instruction asks a question, record the answer with
`apis.supervisor.complete_task(answer=...)`; when it does not, call it with no answer. Do that
before you `submit`."""


@register("appworld")
class AppWorldEnv(Env):
    """One AppWorld task with the filing chore appended, served as one shogym task.

    Config (all optional, via ``shogym.make("appworld", config=...)`` / ``env_config``):
      - ``pulse``: which draw the conventions come from. The default is the published one; a
        different value is a different experiment and its scores are not comparable.
      - ``report``: which payload class rides the report channel, ``graded`` (the receipt) or
        ``drawn`` (a receipt whose verdicts were sampled rather than computed). The notice channel
        always carries the digest. All three are the same length on the wire.
      - ``horizon``: how many blocks of code an episode may run.
    """

    mcp_servers = (APPWORLD_SPEC,)
    function_name = "agent"
    score_terminal_tool = SUBMIT_TOOL_NAME

    def __init__(
        self,
        pulse: int = DEFAULT_PULSE,
        report: str = payload.GRADED,
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        if report not in (payload.GRADED, payload.DRAWN):
            raise ValueError(
                f"report must be {payload.GRADED!r} or {payload.DRAWN!r}, got {report!r}; the "
                "digest is the notice channel's and is never the report's"
            )
        adapter.ensure_apps()
        self._pulse = int(pulse)
        self._report = report
        self._original = adapter.ensure_corpus() / "data"
        # Read once, from the corpus this run actually serves, and used for three things that all
        # have to agree: the name of the served cache, the name of the grader's cache, and the run
        # fingerprint a resumed record is checked against. They used to be able to disagree, so a
        # process pointed at a second corpus computed a fingerprint for that one and then reused
        # and served task material derived from the first.
        self._corpus = adapter.corpus_digest(self._original.parent)
        served, graded = (
            adapter.derived_root(self._corpus),
            adapter.graded_root(self._corpus),
        )
        adapter.stamp_cache(served, source=self._corpus)
        adapter.stamp_cache(graded, source=self._corpus)
        self._derived = world.derive_root(original=self._original, derived=served / "data")
        # The grader's view of the same corpus, with the answers linked back in. Only the grading
        # process is ever given this root; the world an agent drives is given the other one.
        self._graded = world.derive_root(original=self._original, derived=graded / "data")
        self._task_ids = adapter.task_ids()
        self._backlogs: Dict[str, Any] = {}
        self.function = FunctionConfig(example_system_template=_static_instructions())
        # The step budget the serve layer enforces is one past the configured block budget, so
        # that `submit` always has a slot to run in. The spare slot is not another block: the
        # serve layer dispatches the call that reaches the horizon and cannot tell an `execute`
        # from a terminal, so `execute` counts its own calls and refuses past `blocks` without
        # touching the world (see `mcp_server.execute`).
        self._blocks = int(horizon)
        self._config_digest = run_fingerprint(
            pulse=self._pulse,
            report=self._report,
            blocks=self._blocks,
            corpus=self._corpus,
        )
        super().__init__(horizon=self._blocks + 1, num_tasks=len(self._task_ids))

    # ----- task loading -----

    def _load_task(self, task_idx: Optional[int]) -> Dict[str, Any]:
        """Resolve one task: which one it is, and whose accounts its world is driven with."""
        if task_idx is None:
            task_idx = int(self.np_random.integers(0, len(self._task_ids)))
        if not 0 <= task_idx < len(self._task_ids):
            # Negatives too: Python would index backwards into a real task while the record said
            # `-1`, so a task that ran would be filed as one that does not exist.
            raise ValueError(
                f"Task index {task_idx} is out of range for {len(self._task_ids)} tasks"
            )
        task_id = self._task_ids[task_idx]
        specs = adapter.task_specs(self._original.parent, task_id)
        return {
            "task_idx": task_idx,
            "task_id": task_id,
            "supervisor_email": specs["supervisor"]["email"],
        }

    @property
    def config_digest(self) -> str:
        """The run fingerprint (see :func:`run_fingerprint`), for a runner to record and compare."""
        return self._config_digest

    def _backlog(self, task_id: str, specs: Dict[str, Any]):
        """The backlog seeded into ``task_id``'s world.

        Drawn from the task identity alone, so it is the same on every machine, in every process
        and under every feedback regime. Every task in the manifest has one; a task that does not
        is not in the manifest."""
        if task_id in self._backlogs:
            return self._backlogs[task_id]
        reference = dt.datetime.fromisoformat(specs["datetime"]).date()
        backlog = build_backlog(_backlog_seed(task_id), reference)
        if backlog is None:
            raise RuntimeError(
                f"no backlog for {task_id} separates the conventions, but the manifest lists it; "
                f"the manifest at {adapter.MANIFEST} and the generator disagree"
            )
        self._backlogs[task_id] = backlog
        return backlog

    # ----- session lifecycle -----

    def _begin_session(self, session_id: str, task: Dict[str, Any]) -> None:
        """Start a worker for this episode, seed the task's world if it is new, and open it.

        Seeding runs here rather than at load time because the rows are written by the worker's
        interpreter: AppWorld cannot be imported beside shogym, so the only process that can write
        a database log is the one that owns the worlds. It is idempotent, so the second time a
        task is served the derived world is already on disk and nothing is written."""
        from shogym.envs.appworld import mcp_server

        task_id = str(task["task_id"])
        # An absolute path, which AppWorld joins onto its own output root and so replaces it
        # outright. That is the whole of what keeps one episode's end state, logs and evaluator
        # artifacts out of the tree every other episode is served from: a shared output tree is a
        # place a later episode, including the other arm of a pair, can read an earlier grade.
        experiment = str(adapter.episode_outputs(session_id))
        # The shared, pristine seeded world, written once per task. It needs a worker of its own
        # because only the runtime's interpreter can write a database log, and it gets one only on
        # the cold path: a task already derived needs no worker to say so.
        if not world.already_derived(
            derived=self._derived, graded=self._graded, task_id=task_id
        ):
            seeder = adapter.Worker.spawn(self._derived.parent)
            try:
                self._derive(seeder, task_id)
            finally:
                seeder.close()
        # This episode's own view of it. Not the shared tree: see `world.derive_view` for why an
        # episode that writes through its served inputs must not be writing through the next
        # episode's, or the other arm of its own pair's.
        view = world.derive_view(
            derived=self._derived,
            view=adapter.episode_view(session_id),
            task_id=task_id,
        )
        worker = adapter.Worker.spawn(view)
        try:
            worker.call(
                "open", task_id=task_id, experiment=experiment, seed=_world_seed(task_id)
            )
        except Exception:
            worker.close()
            raise
        mcp_server.begin_session(
            session_id,
            mcp_server.Session(
                worker=worker,
                task_id=task_id,
                supervisor_email=str(task["supervisor_email"]),
                experiment=experiment,
                view=str(view),
                budget=self._blocks,
            ),
        )

    def _end_session(self, session_id: str) -> None:
        import shutil

        from shogym.envs.appworld import mcp_server

        session = mcp_server.end_session(session_id)
        if session is None:
            return
        session.worker.close()
        # The episode's served view goes with it too, for the reason it existed: a view that
        # outlived its episode is a directory the next one could be given by mistake.
        shutil.rmtree(session.view, ignore_errors=True)
        # The episode's output tree goes with the episode. It holds this episode's end state and
        # its logs, and leaving it behind gives a later episode something of an earlier one's to
        # find; a directory that only ever grows is retention by omission rather than by policy.
        # After the worker is closed, so nothing is still writing into what is being removed.
        shutil.rmtree(session.experiment, ignore_errors=True)
        # And the copy the grader was given, which is the same tree under a second name.
        shutil.rmtree(session.experiment + ".graded", ignore_errors=True)

    def _derive(self, worker: adapter.Worker, task_id: str) -> None:
        """Make sure the seeded copy of ``task_id``'s world exists, writing it if it does not.

        Checked before the backlog is drawn rather than after. Drawing one costs about a second,
        and a run that serves a task a second time has the world it needs already on disk."""
        if world.already_derived(
            derived=self._derived, graded=self._graded, task_id=task_id
        ):
            return
        specs = adapter.task_specs(self._original.parent, task_id)
        backlog = self._backlog(task_id, specs)
        rows = world.seeding(
            backlog,
            supervisor_email=specs["supervisor"]["email"],
            moment=specs["datetime"],
            tag=task_id,
        )
        world.derive_task(
            original=self._original,
            derived=self._derived,
            graded=self._graded,
            task_id=task_id,
            write_log=lambda source, into: worker.call(
                "seed", **rows, from_dbs=str(source), into=str(into)
            ),
        )

    # ----- describe -----

    def describe(self, task_id: Optional[str] = None) -> TaskSpec:
        spec = super().describe(task_id)
        idx = self._resolve_idx(task_id)
        if idx is None:
            return spec
        return spec.model_copy(update={"instructions": self._instructions(idx)})

    def _resolve_idx(self, task_id: Optional[str]) -> Optional[int]:
        if task_id is None:
            return None
        try:
            idx = int(task_id)
        except (TypeError, ValueError):
            return None
        return idx if 0 <= idx < len(self._task_ids) else None

    def _instructions(self, task_idx: int) -> str:
        """One task's instructions: the world's own, then the task's, then the appended paragraph.

        The paragraph goes last and is byte-identical everywhere, so an agent reading its
        hundredth task reads the same words it read on its first and nothing about the position
        is in the text."""
        specs = adapter.task_specs(self._original.parent, self._task_ids[task_idx])
        supervisor = specs["supervisor"]
        who = (
            f"You are working for {supervisor['first_name']} {supervisor['last_name']} "
            f"({supervisor['email']}, {supervisor['phone_number']}). "
            f"Today is {specs['datetime']}."
        )
        return (
            f"{_static_instructions()}\n\n"
            f"# Your supervisor\n{who}\n\n"
            f"# The instruction\n{specs['instruction']}\n\n"
            f"{world.APPENDED_PARAGRAPH}"
        )

    # ----- finalize (seal-before-verdict) -----

    async def finalize(  # pyright: ignore[reportIncompatibleVariableOverride]
        self, req: "FinalizeRequest"
    ) -> "TerminalEvidence":
        """Score the **sealed** episode and return core-owned evidence.

        The world is read, the base task's own checks are collected, and the filing is compared
        against the drawn key here, in the serving process. The key is never sent to the world:
        the worker's protocol has no field for it and no comparison in it, so a world that an
        agent had complete control of still could not be made to say what the key was.

        The verdict carries all three payload classes, because publishing them is what the port
        is for: they become the episode feedback a feedback policy decides the fate of. A stream
        answers a terminating call from its policy and never from this dict."""
        from shogym.envs.appworld import mcp_server
        from shogym.serve.lifecycle import TerminalEvidence

        session = mcp_server.get_session(req.session_id)
        if session is None:
            raise RuntimeError(f"no open world for session {req.session_id}")
        specs = adapter.task_specs(self._original.parent, session.task_id)
        backlog = self._backlog(session.task_id, specs)
        key = draw_key(leg_of(session.task_id), self._pulse)

        # Off the event loop. Each of these blocks on another process, one of them the world's
        # own evaluator, and a coroutine that never yields would stop every other episode this
        # serving process is running and would make the serve layer's deadline unable to fire on
        # this one.
        #
        # **Nothing here asks the world anything.** The world's process is the process that runs
        # agent-authored code, so a reply from it saying that it had stopped, or flushed, or that
        # a value was such-and-such, is a reply the episode could have written. There is no seal
        # command, no quiesce command and no read command any more. The host stops the worker's
        # process group, confirms from the process table that the group is empty, and grades what
        # is on disk.
        #
        # **What is on disk is the world at the end of the last block, because upstream puts it
        # there.** `AppWorld.execute` ends with its own save into the episode's output tree and
        # `initialize` writes one before any block runs, so an episode that ran N blocks is graded
        # on the state after block N, and an episode that ran none is graded on its opening state.
        # Work an agent's thread does after its last block is lost rather than scored, which is
        # the same rule the block budget states.
        #
        # **A stop that cannot be confirmed ends the episode unscored.** Not signalled, not
        # reaped, or a process table that would not answer: each of those leaves a tree something
        # may still be writing to, and a number computed over it is a number about no instant of
        # the episode. Raising here is what makes that an infrastructure closure rather than a
        # score: the stream files the row `finalize_error`, unscored, with nothing in `observed`,
        # so neither feedback arm has a payload to reveal (see `TaskStream._unsealed_row`).
        await asyncio.to_thread(session.worker.close, confirm=True)
        # A tree of regular files, or no grade. See `adapter.snapshot_outputs`: the grading
        # process is pointed at the root that holds the answers, so a link left under the output
        # tree would resolve there.
        snapshot = await asyncio.to_thread(
            adapter.snapshot_outputs,
            Path(session.experiment),
            into=Path(session.experiment + ".graded"),
        )
        graded = await asyncio.to_thread(
            adapter.grade,
            root=self._graded.parent,
            task_id=session.task_id,
            outputs=snapshot,
            ignore=world.ADDED_MODELS,
            filing={
                "supervisor_email": session.supervisor_email,
                "project": world.PROJECT_NAME,
                "title": world.LOG_TITLE,
                "label": world.LOG_LABEL,
            },
        )
        checks = graded["checks"]
        read = {
            "filing": graded["filing"],
            "world_digest": graded["world_digest"],
            "rng_digest": graded["rng_digest"],
        }
        filing = world.Filing(**{**read["filing"], "lines": tuple(read["filing"]["lines"])})
        verdicts = score(
            backlog=backlog,
            key=key,
            filing=filing,
            assertions=[(check_id, passed) for check_id, passed in checks],
        )
        rendered = {
            name: payload.render(
                task_id=session.task_id, verdicts=verdicts, cell=name, pulse=self._pulse
            )
            for name in (self._report, payload.DIGEST)
        }
        return TerminalEvidence(
            source=req.source,
            status="ok",
            verdict={
                **_numbers(verdicts),
                "payload_class": self._report,
                "world_digest": str(read["world_digest"]),
                "rng_digest": str(read["rng_digest"]),
                REPORT_FEEDBACK_NAME: rendered[self._report],
                NOTICE_FEEDBACK_NAME: rendered[payload.DIGEST],
            },
            diagnostic=(
                f"scored source={req.source} ledger={verdicts.ledger_fraction:.4f} "
                f"pinned={verdicts.pinned_fraction:.4f} exercised={verdicts.exercise_fraction:.4f}"
            ),
        )

    # ----- verify -----

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: "Optional[TerminalEvidence]" = None,
    ) -> FeedbackCollection:
        """Publish the episode's outcome off the core-owned terminal ``evidence``.

        ``ledger_fraction`` is the headline and ``pinned_fraction`` is its control: the four
        stored slots are scored the same way and cannot move past one over their option count
        whatever the agent learns, so a run in which they move with the headline is a run whose
        headline is measuring something else.

        ``report`` and ``notice`` are the matched pair a feedback policy chooses between. Both are
        always published, whatever regime the run is serving, because the env does not know the
        regime and may not: an env that published only the one its run was going to reveal would
        have made the record depend on the treatment."""
        fb = FeedbackCollection()
        if not terminated:
            return fb
        if evidence is not None and evidence.finalize_error:
            # **A failed terminal publishes the failure and nothing else.** It used to publish a
            # row of zeroed fractions and an empty receipt beside them, which is a scored-looking
            # row for an episode that was never scored: the zeros average into a mean and the
            # empty receipt is still an item a paired policy selects and reveals. There is no
            # verdict behind this episode, so the honest record of it is that fact alone, and the
            # stream files the row unscored on the core's own stamp rather than on this item.
            fb.episode.append(EpisodeFeedback(name="finalize_error", value=True))
            fb.inference.append(
                InferenceFeedback(name="config_digest", value=self._config_digest, step=0)
            )
            return fb
        verdict = evidence.verdict if evidence is not None else {}
        for name in (
            "ledger_fraction",
            "pinned_fraction",
            "exercise_fraction",
            "parse_fraction",
            "assertion_fraction",
        ):
            fb.episode.append(EpisodeFeedback(name=name, value=float(verdict.get(name) or 0.0)))
        for name in ("distinct_bands", "filing_rows", "duration_set", "checks"):
            fb.episode.append(EpisodeFeedback(name=name, value=float(verdict.get(name) or 0.0)))
        for name in ("payload_class", "world_digest", "rng_digest"):
            fb.episode.append(EpisodeFeedback(name=name, value=str(verdict.get(name) or "")))
        fb.episode.append(
            EpisodeFeedback(name=REPORT_FEEDBACK_NAME, value=str(verdict.get(REPORT_FEEDBACK_NAME) or ""))
        )
        fb.episode.append(
            EpisodeFeedback(name=NOTICE_FEEDBACK_NAME, value=str(verdict.get(NOTICE_FEEDBACK_NAME) or ""))
        )
        # Recorded, never surfaced. Run identity has to be on every row, because that is what a
        # resumed directory is checked against; and it has to be off every wire, because it is a
        # short digest over a small integer pulse and an agent handed one could enumerate pulses
        # until it matched and then compute every later key. Inference level is exactly that
        # contract: the record keeps it, and no feedback policy can reveal it, including the one
        # that reveals everything else.
        fb.inference.append(
            InferenceFeedback(name="config_digest", value=self._config_digest, step=0)
        )
        return fb


# ----- pure helpers -----


def _numbers(verdicts: Verdicts) -> Dict[str, float]:
    """The fractions a scored episode publishes, with the base task's own beside them."""
    assertions = [item for item in verdicts.items if item.kind == "assertion"]
    return {
        "ledger_fraction": verdicts.ledger_fraction,
        "pinned_fraction": verdicts.pinned_fraction,
        "exercise_fraction": verdicts.exercise_fraction,
        "parse_fraction": verdicts.parse_fraction,
        "assertion_fraction": (
            sum(1 for item in assertions if item.passed) / len(assertions) if assertions else 0.0
        ),
        "distinct_bands": float(verdicts.distinct_bands),
        "filing_rows": float(verdicts.filing_rows),
        "duration_set": float(verdicts.duration_set),
        "checks": float(len(verdicts.items)),
    }


def _static_instructions() -> str:
    """The durable, task-independent framing published by ``describe(task_id=None)``."""
    return f"{_WORLD_GUIDE}\n\n{_TOOL_GUIDE}"


#: Bumped by hand when a change to this port would make two runs' scores mean different things
#: without changing any of its inputs: the scorer's rules, the payload's layout, the seeded
#: backlog's shape, or how the state behind a score is read. It is in the run fingerprint so that
#: "the same pulse" is not mistaken for "the same measurement" across such a change.
#:
#: 3: the filing and the world digest are read from the stopped episode's own persisted tree by
#: the grading process, rather than asked of the serving world over the protocol. Two runs either
#: side of that read different bytes at different moments, so their rows are not one measurement
#: whatever else they agree on.
SCORING_VERSION = 3


def run_fingerprint(*, pulse: int, report: str, blocks: int, corpus: str = "") -> str:
    """Everything two runs must agree on for their rows to be one measurement.

    The draw and the payload class decide what a score *means*; the block budget decides what an
    episode had the chance to do; the corpus, the interpreter and the derivation decide what world
    it happened in; every constant a payload is generated from decides what the agent was told;
    and :data:`SCORING_VERSION` decides how it was read. ``corpus`` is what the corpus actually
    holds rather than what the pin says it should (see :func:`~adapter.corpus_digest`), because
    the root is whatever the environment points at and a repointed one would otherwise pass for
    the pinned one. A provenance directory reopened under a
    different one of those takes incomparable rows, and none of them is visible anywhere else in a
    run's record.

    **On every row, and on no wire.** It is published at inference level, which the record keeps
    and no feedback policy reveals, not even the one that reveals everything else. Both halves
    matter: a resumed directory is checked against what its rows say, and a digest over a usually
    small integer pulse is one an agent handed it could enumerate until it matched, after which
    every later key is computable.

    **Enforcing it is the runner's, not this env's.** A stream validates a resumed directory's
    queue and its feedback regime; an env is handed a task and does not know which directory it is
    being served into, so it cannot refuse a resume. What it can do is publish one value that says
    whether two rows belong to one measurement, which is what a runner has to compare."""
    material = "|".join(
        [
            str(pulse),
            report,
            str(blocks),
            str(SCORING_VERSION),
            adapter.DATA_VERSION,
            str(adapter.DERIVATION_VERSION),
            adapter.DATA_BUNDLE_SHA256,
            corpus,
            adapter.UPSTREAM_VERSION,
            adapter.MANIFEST.read_text(),
            payload.PASS_COUNTS_FILE.read_text(),
            # Every constant a published payload is generated from, and this one was missing.
            # `DRAWN_BASIS` seeds the drawn arm's whole visible vector: changing it re-rolls
            # every drawn payload, which is a change to the treatment an agent is under, and a
            # record could resume across it under an unchanged identity.
            payload.DRAWN_BASIS,
        ]
    )
    return hashlib.sha256(material.encode()).hexdigest()[:16]


def _backlog_seed(task_id: str) -> int:
    """The seed a task's backlog is drawn from: the task identity and nothing else.

    A checksum rather than ``hash``, which Python randomises per process, so a backlog built
    today is the backlog built tomorrow."""
    return zlib.crc32(task_id.encode())


def _world_seed(task_id: str) -> int:
    """The seed the world's own generator is started from.

    Named for the task and for nothing else. Not the session, not the run, and not the feedback
    regime: a seed that named the arm would deal two arms of one task two different worlds, and
    the difference between them would be a difference the treatment did not make."""
    return zlib.crc32(f"appworld|{task_id}".encode()) & 0x7FFFFFFF


__all__ = [
    "APPWORLD_SPEC",
    "DEFAULT_HORIZON",
    "DEFAULT_PULSE",
    "EXECUTE_TOOL_NAME",
    "SCORING_VERSION",
    "SUBMIT_TOOL_NAME",
    "AppWorldEnv",
    "run_fingerprint",
]
