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
unfrozen by the other. Each episode gets a worker in a container of its own, reached over the pipe
pair its parent made and by nothing else: there is no network in there and so no port to find and
no token to need (:mod:`shogym.envs.appworld.worker`).

This module imports **nothing** from upstream at load time, so ``import shogym`` (which imports it
to register the env) stays offline. Upstream is not installed on this machine at all; the image
and the corpus are provisioned when an ``appworld`` env is *constructed*; see
:mod:`shogym.envs.appworld.adapter`.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import os
import stat
import threading
import time
import zlib
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Sequence

from shogym.core import Env
from shogym.envs.appworld import adapter, container, payload, world
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

#: How long finalization waits for an accepted call to come back before it gives up on grading
#: this episode. Short, because it runs inside the serve layer's own release bound, and long
#: enough that an ordinary block finishing its save is waited for rather than raced.
_SETTLE_SECONDS = 30.0

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

    #: The item a runner may compare against the identity it is filing rows under. Declared here
    #: rather than assumed there: `config_digest` is an ordinary name any env may publish as a
    #: metric, and a module that decided what its number meant would turn another env's successful
    #: terminal into an unscored failure. An env that declares nothing is not checked.
    #:
    #: The value is readable off the env itself (`config_digest` below), so a serve layer can fold
    #: it into the identity a run is filed under before the first task is dispensed, rather than
    #: waiting for the first row to publish one.
    identity_feedback_name = "config_digest"

    def __init__(
        self,
        pulse: int = DEFAULT_PULSE,
        report: str = payload.GRADED,
        horizon: int = DEFAULT_HORIZON,
    ) -> None:
        if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
            # Eagerly, because the two ways of getting this wrong both half-work. A budget of zero
            # disabled the guard entirely and let one block through before the serve layer ended
            # the episode; a negative one refused the first block and still spent the call that
            # ends the horizon. A budget is a count of blocks, and the smallest honest one is one.
            raise ValueError(f"horizon must be a positive whole number of blocks, got {horizon!r}")
        if report not in (payload.GRADED, payload.DRAWN):
            raise ValueError(
                f"report must be {payload.GRADED!r} or {payload.DRAWN!r}, got {report!r}; the "
                "digest is the notice channel's and is never the report's"
            )
        # Before anything else, and at construction rather than at the first `execute`. Every
        # episode's world runs in a container; there is no host fallback, because a worker on the
        # host runs agent-authored code as the user running the run. A machine without Docker
        # cannot serve this env, and an hour into a run is the wrong time to find that out.
        container.require_docker()
        adapter.ensure_image()
        # What the last run left behind, on a thread of its own. See `_housekeep`.
        _housekeep()
        self._pulse = int(pulse)
        self._report = report
        self._original = adapter.ensure_corpus() / "data"
        self._task_ids = adapter.task_ids()
        # Read once, from the corpus this run actually serves, and used for three things that all
        # have to agree: the name of the served cache, the name of the grader's cache, and the run
        # fingerprint a resumed record is checked against. They used to be able to disagree, so a
        # process pointed at a second corpus computed a fingerprint for that one and then reused
        # and served task material derived from the first.
        #
        # **The roster's authored text comes back from the same read, and is what this env serves
        # for the rest of its life.** The digest and the cache names were fixed here while the
        # instructions, the supervisors and the dates went on being reread from the live corpus
        # every time a task was described, seeded or scored. So a corpus edited after construction
        # served new authored text under the old fingerprint, out of caches named for the old
        # bytes, and nothing in the record said so. An env that has stated what corpus it is
        # serving has to go on serving that one (see `adapter.corpus_snapshot`).
        snapshot = adapter.corpus_snapshot(self._original.parent, task_ids=self._task_ids)
        self._corpus = snapshot.digest
        self._specs = snapshot.specs
        # What the image turned out to be, read once here and used for three things that have to
        # agree: both cache names, both stamps, and the run fingerprint. It is the image that
        # writes a task's seeded database log, so a cache it did not write is a world this run did
        # not make; and reading it twice is a second control call for an answer that cannot have
        # changed in between.
        self._runtime = adapter.runtime_digest()
        # Bound to the corpus this snapshot was taken of, and handed to both derivations. A task
        # is materialised on its first use, which can be hours and two hundred episodes after the
        # digest above was computed, and the bytes it copies are the world an agent is served and
        # the baseline it is graded against. Pinning the authored text was never enough for those.
        self._source_check = partial(snapshot.verify, self._original.parent)
        served, graded = (
            adapter.derived_root(self._corpus, runtime=self._runtime),
            adapter.graded_root(self._corpus, runtime=self._runtime),
        )
        adapter.stamp_cache(served, source=self._corpus, runtime=self._runtime)
        adapter.stamp_cache(graded, source=self._corpus, runtime=self._runtime)
        self._derived = world.derive_root(
            original=self._original, derived=served / "data", verify=self._source_check
        )
        # The grader's view of the same corpus, with the answers linked back in. Only the grading
        # process is ever given this root; the world an agent drives is given the other one.
        self._graded = world.derive_root(
            original=self._original, derived=graded / "data", verify=self._source_check
        )
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
            # Passed in rather than read again: the env has already read it, reading it is a
            # control call, and the answer cannot have changed between the two reads.
            runtime=self._runtime,
            # What machine an episode was given. Captured once for this process and passed to
            # every launch, so an environment changed mid-run cannot move it and a run relaunched
            # under a changed one does not pass for the earlier measurement.
            resources="|".join(container.limits()),
        )
        super().__init__(horizon=self._blocks + 1, num_tasks=len(self._task_ids))

    # ----- task loading -----

    def _task_specs(self, task_id: str) -> Dict[str, Any]:
        """One task's shipped specification, as this env's corpus held it when it was fingerprinted.

        The pinned snapshot rather than the file, and that is the whole point: the instruction, the
        supervisor and the datetime decide what an agent is asked to do, what world is seeded and
        which key it is graded against, and all three used to be reread from the live corpus long
        after this env had fixed the digest it publishes and the cache names it serves out of. A
        corpus edited in place then produced new authored text under an unchanged fingerprint, so
        two episodes of one run could be two different tasks that a record calls one.

        Refusing the change instead would need the corpus rehashed on every read, which is two
        seconds a time, and re-fingerprinting would give one env two identities. Serving what was
        read is the contract that costs nothing and can be stated: an env serves the corpus it was
        constructed against, and a corpus that changed is a new env away."""
        try:
            return self._specs[task_id]
        except KeyError:
            raise KeyError(
                f"{task_id} is not in this env's roster, so no specification for it was read from "
                f"the corpus at construction"
            ) from None

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
        specs = self._task_specs(task_id)
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
        task is served the derived world is already on disk and nothing is written.

        **The backlog is drawn here too, and unconditionally.** It used to be drawn only where a
        task had to be seeded, so a task served a second time reached ``finalize`` without one and
        drew it there: at the top of a coroutine, on the loop every other episode of the run is
        running on, for between a tenth of a second and three seconds of auditing depending on the
        task. The serve layer runs this hook in a thread, which is where work of that shape
        belongs, and drawing a backlog twice is free because it is kept.

        **Nothing is left behind by a session that does not open.** The episode's own served view
        is written before the worker exists and the worker exists before the session is published,
        so a spawn that fails, a world that will not open, or anything else that raises in between
        used to leave a copied task directory, and sometimes a live worker, with nothing holding a
        reference to either: ``_end_session`` cleans up what a *published* session names, and there
        is no published session on this path. Everything made here is made under one guard that
        discards all of it, which is the same teardown, owed by whoever failed."""
        from shogym.envs.appworld import mcp_server

        task_id = str(task["task_id"])
        # Drawn here rather than where a task has to be seeded. A task served a second time
        # reached `finalize` without one and drew it there: at the top of a coroutine, on the loop
        # every other episode is running on, for between a tenth of a second and three seconds
        # depending on the task. This hook runs in a thread, and drawing a backlog twice is free
        # because it is kept.
        self._backlog(task_id, self._task_specs(task_id))
        # One output tree per episode, outside every served corpus, mounted alone into this
        # episode's container at a fixed name. The world is told its experiment *is* that
        # directory: AppWorld joins an experiment name onto its own output root, so an absolute
        # one replaces the root, and inside the container the absolute one is the mount point.
        outputs = adapter.episode_outputs(session_id)
        experiment = container.OUTPUTS_MOUNT
        view = adapter.episode_view(session_id)
        # **Everything made before the session exists is made under this guard**, and the guard
        # starts at the first claim rather than at the spawn: a claim can fail too, and so can
        # deriving a task the corpus does not have, and both used to fail after a tree had been
        # made with nothing holding it. The env's own close finds no session, so nothing else was
        # going to remove any of them.
        worker: Optional[adapter.Worker] = None
        try:
            # **Claimed before it exists, and every tree this port generates is.** A sweep
            # racing this construction has to find an owner rather than an untouched directory,
            # and `_reclaimable` refuses to guess about a tree the control plane never heard of,
            # so a directory that appeared before its record is one nothing will ever take.
            _claim_tree(outputs)
            # Claimed and not created: finalization is what makes the copy handed to the grader,
            # and the claim has to be older than the directory rather than older than the copy
            # into it. A record naming a directory that never gets made is one small file, which
            # teardown drops and the sweep collects.
            _claim_tree(_snapshot_of(outputs), create=False)
            # Deriving comes first, and has to: the world's container mounts this one task's tree,
            # so the tree has to exist before there is a container to mount it into. Seeding is a
            # container of its own, which is also why it no longer needs this episode's worker.
            self._derive(task_id)
            # This episode's own view of the derived corpus. Under the container the served tree
            # is mounted read-only, so the write this exists to contain cannot happen at all; it
            # is kept because it is the property at the layer below, and a run of this env without
            # the container is a run where it is the only thing holding it.
            _claim_tree(view)
            view = world.derive_view(
                derived=self._derived, view=view, task_id=task_id
            )
            worker = adapter.Worker.spawn(view, task_id=task_id, outputs=outputs)
            worker.call(
                "open", task_id=task_id, experiment=experiment, seed=_world_seed(task_id)
            )
        except BaseException:
            if worker is not None:
                worker.close()
            _discard(view, outputs, _snapshot_of(outputs))
            raise
        mcp_server.begin_session(
            session_id,
            mcp_server.Session(
                worker=worker,
                task_id=task_id,
                outputs=outputs,
                view=str(view),
                supervisor_email=str(task["supervisor_email"]),
                experiment=experiment,
                budget=self._blocks,
            ),
        )

    def _end_session(self, session_id: str) -> None:
        from shogym.envs.appworld import mcp_server

        session = mcp_server.get_session(session_id)
        if session is None:
            return
        try:
            # Never raises here: teardown's close is best effort by contract, and a container it
            # could not remove belongs to the reaper rather than to this call.
            session.worker.close()
        finally:
            # The directories go whatever the container did. A view that outlived its episode is
            # a directory the next one could be given by mistake, and an output tree that only
            # ever grows is retention by omission rather than by policy. In a `finally` because
            # the failure that stops the close is exactly the failure that would otherwise leave
            # them, and the handle is dropped last, so nothing between here and there loses it.
            _discard(Path(session.view), session.outputs, _snapshot_of(session.outputs))
            mcp_server.end_session(session_id)
            # Whatever that teardown could not finish is written down, and this is what comes back
            # for it. Construction used to be the only thing that ever started a pass, and every
            # failure that defers work happens after one: a run holding a single env for a whole
            # queue recorded work nothing in the process would ever return to. On a thread, so an
            # episode's end does not wait for a sweep (see `_housekeep`).
            _housekeep()

    def _derive(self, task_id: str) -> None:
        """Make sure the seeded copy of ``task_id``'s world exists, writing it if it does not.

        Checked before the backlog is drawn rather than after. Drawing one costs about a second,
        and a run that serves a task a second time has the world it needs already on disk."""
        if world.already_derived(
            derived=self._derived, graded=self._graded, task_id=task_id
        ):
            return
        specs = self._task_specs(task_id)
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
            write_log=lambda source, into: adapter.seed(
                root=self._derived.parent, source_dbs=source, into=into, rows=rows
            ),
            verify=self._source_check,
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
        specs = self._task_specs(self._task_ids[task_idx])
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
        specs = self._task_specs(session.task_id)
        # Off the loop, and it is the first thing this coroutine does that is not a dictionary
        # lookup. `_begin_session` has normally drawn this already, in the thread the serve layer
        # runs session setup in, and then this costs nothing; what it must not be is the coroutine
        # itself running an audit that takes up to three seconds on a task whose world was already
        # on disk, before it has yielded once. Every other episode of the run is on this loop, and
        # so is the `wait_for` that is supposed to be able to time this one out.
        backlog = await asyncio.to_thread(self._backlog, session.task_id, specs)
        key = draw_key(leg_of(session.task_id), self._pulse)

        # Off the event loop. Each of these blocks on another process, one of them the world's
        # own evaluator, and a coroutine that never yields would stop every other episode this
        # serving process is running and would make the serve layer's deadline unable to fire on
        # this one.
        #
        # **Nothing here asks the world anything.** The world's process is the process that runs
        # agent-authored code, so a reply from it saying that it had stopped, or flushed, or that
        # a value was such-and-such, is a reply the episode could have written. There is no seal
        # command and no quiesce command any more. The host stops the container, confirms it with
        # the daemon, and grades what is on disk.
        #
        # **What is on disk is the world at the end of the last block, because upstream puts it
        # there.** `AppWorld.execute` ends with its own save into the episode's output tree and
        # `initialize` writes one before any block runs, so an episode that ran N blocks is graded
        # on the state after block N, and an episode that ran none is graded on its opening state.
        # Work an agent's thread does after its last block is lost rather than scored, which is
        # the same rule the block budget states.
        # **A poisoned worker is not gradeable, however cleanly it was stopped.** A timeout or a
        # broken frame interrupts a command, and upstream ends every command with its own save
        # into the tree about to be graded: that saver clears its destination and writes several
        # pieces in sequence, so a stop in the middle of it leaves a tree that is stable and
        # partial. Confirmed absence proves that writing has stopped, not that it finished, so the
        # episode is refused here rather than scored on whatever the interruption left.
        # **A terminal may overtake an ordinary call, and stopping on top of one is not allowed.**
        # The serve layer lets a terminal jump the queue on purpose, so a submit can arrive while
        # an `execute` is inside the save upstream ends every block with. Removing the container
        # then leaves a tree that is stable and partial. So the call is waited for, bounded, and a
        # world that will not settle is poisoned rather than stopped underneath.
        if not await asyncio.to_thread(session.worker.settle, _SETTLE_SECONDS):
            session.worker.poisoned = (
                f"a call was still running {_SETTLE_SECONDS:.0f}s after this episode was sealed, "
                "so stopping the world now would interrupt whatever it was writing"
            )
        if session.worker.poisoned:
            await asyncio.to_thread(session.worker.close, confirm=False)
            raise RuntimeError(
                "this episode cannot be scored: its world was interrupted mid-command "
                f"({session.worker.poisoned}), so what it persisted may be half of a save"
            )
        await asyncio.to_thread(session.worker.close, confirm=True)
        # A tree of regular files, or no grade. See `adapter.snapshot_outputs`: the grader's
        # namespace holds the answers, so a link left under the output tree would resolve there.
        # The copy runs in a thread, and cancelling an `await` does not stop a thread: a
        # finalization the deadline gave up on would otherwise leave one walking an episode's
        # tree, holding the file handles and the disk it needs. The flag is what the thread stops
        # for, checked once per file.
        abandon = threading.Event()
        try:
            snapshot = await asyncio.to_thread(
                adapter.snapshot_outputs,
                session.outputs,
                into=_snapshot_of(session.outputs),
                stop=abandon,
            )
        except BaseException:
            abandon.set()
            raise
        # A whole save, or no grade. The reply that said a block finished is not evidence, so
        # what is on the stopped tree is checked instead: every log the task's own inputs have,
        # and none of them cut off mid-write.
        try:
            await asyncio.to_thread(
                adapter.verify_snapshot,
                snapshot,
                task_id=session.task_id,
                expected=self._derived / "tasks" / session.task_id / "dbs",
                # What the host sent, which is the half of this the world cannot write.
                blocks=session.calls,
                stop=abandon,
            )
        except BaseException:
            abandon.set()
            raise
        graded = await asyncio.to_thread(
            adapter.grade,
            graded=self._graded.parent,
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
        headline is measuring something else. The headline is published a second time as
        ``reward``, which is the name a durable row's summary is read from and which this port
        used to leave empty on every scored row.

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
                InferenceFeedback(
                    name="config_digest",
                    value=self._config_digest,
                    step=trajectory[-1].index if trajectory else 0,
                )
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
        # The headline again, under the name the record reads a headline from. `ledger_fraction`
        # is what this port calls its headline and the only thing here that is one, but a durable
        # row's summary is filled from `reward` or `partial_credit` and from nothing else, so
        # without this every complete appworld run recorded `score is not None` with `reward` and
        # `success` both empty, and every shipped reader counted it as `scored 0/N`. An alias and
        # not a rename: `ledger_fraction` stays, because it is the name the scorer, this README
        # and the analysis all use, and a run whose rows lost it would not be readable as one of
        # this port's. It changes no wire either, because both paired policies reveal the one
        # channel they are named for and neither of them is this.
        fb.episode.append(
            EpisodeFeedback(name="reward", value=float(verdict.get("ledger_fraction") or 0.0))
        )
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
        if evidence is not None and evidence.finalize_error:
            fb.episode.append(EpisodeFeedback(name="finalize_error", value=True))
        # Recorded, never surfaced. Run identity has to be on every row, because that is what a
        # resumed directory is checked against; and it has to be off every wire, because it is a
        # short digest over a small integer pulse and an agent handed one could enumerate pulses
        # until it matched and then compute every later key. Inference level is exactly that
        # contract: the record keeps it, and no feedback policy can reveal it, including the one
        # that reveals everything else.
        # On the row's own step, which is the last one the trajectory recorded. A fixed zero was
        # rejected by the trace store on any terminal row past the first step, and the store's
        # refusal was swallowed as degraded persistence: the trace then held no terminal row at
        # all, and a later read of it reported an episode that never ended.
        fb.inference.append(
            InferenceFeedback(
                name="config_digest",
                value=self._config_digest,
                step=trajectory[-1].index if trajectory else 0,
            )
        )
        return fb


#: How many entries teardown will delete before it leaves a tree for the sweep. Teardown runs on
#: the serve layer's own bounded release, so an unbounded walk over a tree an episode wrote is a
#: walk that can outlast the bound and be re-entered on the loop.
_DISCARD_MAX_NODES = 50_000


def _discard(*paths: Path) -> None:
    """Remove what an episode owned, bounded, and never raise while doing it.

    **Bounded, because this runs inside somebody else's deadline.** The serve layer gives a
    session's release sixty seconds and then abandons the wait, and an unbounded `rmtree` over a
    tree an episode wrote can outlast that and be entered a second time. So the count is capped:
    a tree past it is left where it is and swept at the next construction, which is a directory
    nobody is reading rather than a deletion nobody can bound.

    The snapshot copy is always small, because :func:`adapter.snapshot_outputs` refused anything
    that was not; the original is the one that can be large, and it is the one this may leave.

    **Every way out of here leaves the tree owned by somebody.** A tree this declines to walk, a
    walk that raised, and a removal that did not remove are three different failures and were two
    different bugs: the first two left an ended episode's tree still naming the live serving
    process, so the sweep could not take it until that process exited, and the third erased the
    ownership records after a removal that had ignored its own errors, which left whatever
    remained unknown to the control plane and therefore never retried. The records are dropped
    only once the root is confirmed absent."""

    for path in paths:
        try:
            nodes = 0
            for _ in path.rglob("*"):
                nodes += 1
                if nodes > _DISCARD_MAX_NODES:
                    break
            over = nodes > _DISCARD_MAX_NODES
        except OSError:
            # The walk itself failed, so this teardown is not going to remove the tree either. It
            # is still an episode that ended, and saying so is what lets a later sweep take it.
            _mark_ended(path)
            continue
        if over:
            # Left where it is, and *said to be over*: without that it still named the serving
            # process, which is alive, so the sweep meant to reclaim it could not until that
            # process exited.
            _mark_ended(path)
            continue
        if not _remove_tree(path):
            # What is left is a partial tree, and the honest record of it is an ended episode a
            # sweep will come back to rather than no record at all.
            _mark_ended(path)
            continue
        _forget_tree(path)


def _remove_tree(root: Path) -> bool:
    """Remove ``root`` and say whether it is gone, repairing owner access once if it is not.

    **A retry that changes nothing is not a retry.** `/outputs` is bound into the served container
    writable and the container runs as this user, so a block of agent-authored code can leave a
    directory it owns with no traversal bit on it. `shutil.rmtree` cannot enter that directory, and
    `ignore_errors=True` means it says nothing about having failed; every later pass then made the
    same call, got the same result and left the same bytes, so a tree that was recorded as still
    somebody's was one nothing was ever going to reclaim.

    The repair is the owner's own bits, which is all it takes: the process that cannot enter the
    directory is the one that owns it. It is bounded like the walk that precedes it, and it is
    attempted once, so a tree that survives two removals with a permission repair between them is
    a tree this leaves where it is and records rather than one it keeps grinding at.

    Symbolic links are never followed and never chmod-ed: unlinking one needs the bits on its
    parent, and following one would put this walk somewhere the episode chose."""
    import shutil

    shutil.rmtree(root, ignore_errors=True)
    if not os.path.lexists(root):
        return True
    _restore_owner_access(root)
    shutil.rmtree(root, ignore_errors=True)
    return not os.path.lexists(root)


def _restore_owner_access(root: Path) -> None:
    """Give the owner back traversal and write on what is left of a tree. Bounded, never raises.

    Each directory is chmod-ed *before* it is listed, which is the whole point and is why this is
    not an `os.walk`: a walk cannot enumerate a directory it cannot enter, so it would skip the
    one node the repair exists for."""
    pending = [root]
    spent = 0
    while pending:
        path = pending.pop()
        spent += 1
        if spent > _DISCARD_MAX_NODES:
            return
        try:
            mode = path.lstat().st_mode
        except OSError:
            continue
        if stat.S_ISLNK(mode):
            continue
        wanted = stat.S_IRWXU if stat.S_ISDIR(mode) else stat.S_IRUSR | stat.S_IWUSR
        if mode & wanted != wanted:
            try:
                os.chmod(path, stat.S_IMODE(mode) | wanted)
            except OSError:
                continue
        if stat.S_ISDIR(mode):
            try:
                pending.extend(path / name for name in os.listdir(path))
            except OSError:
                continue


#: What the control plane records about a per-episode tree. ``owner`` names the process that made
#: it, and ``ended`` says the episode is over and the tree is reclaimable whoever is still running.
#:
#: **Beside the tree, never inside it.** An episode's output tree is mounted into its container
#: writable, so a marker kept in there is cleanup authority the episode could rewrite: a forged
#: owner would have made a sibling constructor remove a live tree, or kept a dead one for ever.
_OWNER_KIND = "owner"
_ENDED_KIND = "ended"

#: How long a tree marked ended is left before a sweep takes it. Short, because the episode that
#: owned it is over; not zero, because a teardown that declined to walk a tree may still be
#: writing the marker as another construction reads it.
_ENDED_GRACE_SECONDS = 60.0

#: How many trees one construction will remove, and how long it will spend. The sweep runs where
#: an env is built, which a stream may do on its serving loop, so it is bounded like everything
#: else that runs there.
_SWEEP_MAX_TREES = 64
_SWEEP_SECONDS = 5.0


#: Whether a housekeeping pass is already running. One at a time, because a stream that builds an
#: env per task would otherwise start one per construction, and they would all be sweeping the same
#: two directories and the same ledger.
_HOUSEKEEPING = threading.Lock()

#: A wake that arrived while a pass was already running, and that the pass may not have seen.
#: Deferred work is written down before the wake is sent, so a thread that finds this set after a
#: pass has work to go back for, whatever that pass concluded. Set before the lock is tried and
#: read after it is released, which is what makes a rejected wake impossible to lose.
_HOUSEKEEPING_AGAIN = threading.Event()


#: How long a housekeeping thread waits between passes, and how many it will make. Deferred work
#: is written down when it is deferred, so a pass that leaves some is followed by another rather
#: than by nothing: the case is a teardown failure, which by definition happens after the pass a
#: construction started. Capped so the thread ends: what is still there when the passes run out is
#: still written down, and the next episode's end wakes a new one.
_HOUSEKEEPING_INTERVAL_SECONDS = 5.0
_HOUSEKEEPING_MAX_PASSES = 12


def _housekeep() -> None:
    """Clear what this run and the last one left behind, on a thread nothing waits on.

    Two jobs. Containers whose parent process is gone, which is the case teardown cannot reach: a
    run that died while a world was wedged inside a command leaves a worker that never gets back
    to the read that would tell it its parent had gone, so it never exits and ``--rm`` never
    fires. And the per-episode trees teardown declined to walk, plus whatever a crash left behind.

    **On a thread, because the caller may be an event loop.** Both are Docker control calls and
    filesystem walks over what a previous run left, and construction runs in a call a serve layer
    makes while it is dispensing, before the ``await`` that opens the episode, so every sibling
    episode, every deadline and the other arm of a pair waited out the whole of it. Each pass is
    bounded as well (see :func:`container.reap` and :func:`_sweep_leftovers`), and a bounded stall
    on the loop that dispenses tasks is still a stall on it.

    **And it recurs while there is work, because deferring work is not doing it.** Construction was
    the only thing that ever started a pass, and the failures that write work down happen after it:
    a container the daemon would not remove is disowned during a teardown, and a tree that could not
    be walked is marked ended there too. A run that builds one env and serves two hundred tasks
    from it, which is what this port's README recommends, therefore recorded deferred work that
    nothing in the process was ever going to come back to. Every episode's end asks for a pass, and
    a pass that leaves work behind schedules the next one itself.

    One pass at a time, and failures are swallowed: this is housekeeping, and an env that could
    not tidy up after a previous run is an env that can still serve."""
    # **Recorded before the lock is tried, which is what makes a rejected wake survive.** A caller
    # that found the flag held used to simply return, so a teardown that recorded a disowned
    # container or an ended tree in the window between the running pass reading "no work left" and
    # releasing the lock had its wake dropped, and the thread then exited on a conclusion that was
    # already out of date. On a run's last episode nothing came after it.
    _HOUSEKEEPING_AGAIN.set()
    if not _HOUSEKEEPING.acquire(blocking=False):
        return

    def _held() -> None:
        try:
            while True:
                # Cleared before the pass rather than after it, so work recorded *during* the pass
                # sets it again and is seen below rather than being cleared away unlooked at.
                _HOUSEKEEPING_AGAIN.clear()
                _housekeeping_passes()
                if not _HOUSEKEEPING_AGAIN.is_set():
                    break
        finally:
            _HOUSEKEEPING.release()
        # The one window the loop above cannot close is between its last read of the flag and the
        # release: a wake arriving there finds the lock still held and gives up. Read again once
        # the lock is gone, where whoever takes it next can act on it, and this call is the one
        # that has just seen it.
        if _HOUSEKEEPING_AGAIN.is_set():
            _housekeep()

    try:
        threading.Thread(target=_held, name="shogym-appworld-housekeeping", daemon=True).start()
    except RuntimeError:
        # A process that cannot start a thread is one shutting down, and there is nothing here
        # worth failing a construction over.
        _HOUSEKEEPING.release()


def _housekeeping_passes() -> None:
    """Sweep until nothing is outstanding or the passes run out.

    Synchronous, and holding nothing, so that what it does can be tested without a thread and
    without a clock; the lock belongs to :func:`_housekeep`, which is what decides that one pass
    is running at a time."""
    for attempt in range(_HOUSEKEEPING_MAX_PASSES):
        if attempt:
            time.sleep(_HOUSEKEEPING_INTERVAL_SECONDS)
        try:
            container.reap()
            _sweep_leftovers(
                adapter.episodes_home(),
                adapter.cache_root() / f"views-{adapter.DATA_VERSION}",
            )
        except Exception:  # noqa: BLE001 — housekeeping never fails a construction
            pass
        if not _deferred_work():
            return


def _deferred_work() -> bool:
    """Whether anything is still written down as somebody's to clean up.

    The two places work is deferred to: the ledger of containers nobody could remove, and the
    ended sidecars a teardown that could not finish leaves behind."""
    try:
        if container.outstanding():
            return True
    except OSError:
        return False
    try:
        return any(adapter.control_home().glob(f"*.{_ENDED_KIND}"))
    except OSError:
        return False


def _snapshot_of(outputs: Path) -> Path:
    """Where the copy handed to the grader goes for the episode whose output tree is ``outputs``.

    One definition, because three places name this directory and one of them claims it before it
    exists: a name spelled out at each use is a name that can disagree with the claim."""
    return Path(str(outputs) + ".graded")


def _claim_tree(root: Path, *, create: bool = True) -> None:
    """Record which process a per-episode tree belongs to, before the tree can be seen.

    **The sidecar is published first, and a claim that cannot be made fails the setup.** This
    created the root and then wrote the record, and swallowed every ``OSError`` on the way, so two
    things it exists to rule out were reachable: an interruption between the two left a directory
    the control plane had never heard of, and a control home that could not be written left one
    silently, with the episode carrying on around it. `_reclaimable` refuses to guess about a tree
    with no claim, by design, so either of those is a directory no sweep will ever take.

    The order is the other way round now: the record goes down, and only then is the root made. A
    crash between them leaves a record naming a directory that does not exist, which costs one
    small file and which the sweep collects; the reverse left bytes nobody would ever remove.

    ``create=False`` is for a tree something else will make later, which is the copy handed to the
    grader: the claim is what has to exist first, and the directory is finalization's to create."""
    from shogym.envs.appworld import container as container_module

    marker = adapter.control_file(root, _OWNER_KIND)
    marker.write_text(f"{os.getpid()} {container_module.process_birth(os.getpid())}\n")
    if not create:
        return
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError:
        # There is no tree to own, so the record is not left standing for one. Raised on, because
        # a session that cannot make its own output directory has nothing to serve.
        try:
            marker.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _mark_ended(root: Path) -> None:
    """Say that the episode which owned ``root`` is over, so a sweep may take the tree.

    The case is a teardown that declined to walk an oversized tree. Without this the tree still
    named a live process, which is the serving one, so the sweep kept it until that process
    exited: the path meant to reclaim what teardown could not delete could not reclaim it while
    the server ran."""
    try:
        adapter.control_file(root, _ENDED_KIND).write_text(f"{time.time():.0f}\n")
    except OSError:
        pass


def _forget_tree(root: Path) -> None:
    """Drop a tree's control-plane records, once the tree itself is gone."""
    for kind in (_OWNER_KIND, _ENDED_KIND):
        try:
            adapter.control_file(root, kind).unlink(missing_ok=True)
        except OSError:
            pass


def _reclaimable(root: Path) -> bool:
    """Whether a per-episode tree is nobody's any more.

    Two ways, and neither of them is age. An episode that ended says so, and after a short grace
    the tree is reclaimable however alive the server is; that is what a teardown which declined to
    walk an oversized tree leaves behind. Otherwise the question is whether the process that made
    it is still running, which is the same question and the same evidence the container sweep
    asks: a pid and the birth stamp that says it is the same process.

    A tree the control plane says nothing about predates this and is left alone rather than
    guessed about, and every read is bounded because these files are the control plane's own and
    are two short lines."""
    from shogym.envs.appworld import container as container_module

    ended = adapter.control_file(root, _ENDED_KIND)
    try:
        stamp = ended.read_text(errors="replace")[:64].strip()
    except OSError:
        stamp = ""
    if stamp:
        try:
            return time.time() - float(stamp) > _ENDED_GRACE_SECONDS
        except ValueError:
            return False
    try:
        owner = adapter.control_file(root, _OWNER_KIND).read_text(errors="replace")[:64].split()
    except OSError:
        return False
    if not owner or not owner[0].isdigit():
        return False
    return not container_module._process_is_alive(
        int(owner[0]), owner[1] if len(owner) > 1 else ""
    )


def _sweep_leftovers(*homes: Path) -> None:
    """Remove per-episode trees that are nobody's, bounded in trees and in seconds.

    **Age is not evidence.** This removed anything whose directory had not been touched for an
    hour, and an episode can legitimately run longer: sixty blocks at five minutes each is the
    default budget, a view's root is static from the moment it is built, and a database written
    three levels down does not touch the root above it. A sibling arm's construction would then
    have deleted a live episode's mounted tree.

    Bounded in both directions, because this runs where an env is constructed and a stream may
    construct one on its serving loop. The deadline is checked before each removal *and* the
    removals are capped, so one large tree cannot hold a construction open past the bound by more
    than the single deletion it was already inside."""

    began = time.monotonic()
    removed = 0
    for home in homes:
        try:
            entries = sorted(home.iterdir())
        except OSError:
            continue
        for entry in entries:
            if removed >= _SWEEP_MAX_TREES or time.monotonic() - began > _SWEEP_SECONDS:
                return
            if not entry.is_dir() or not _reclaimable(entry):
                continue
            removed += 1
            if not _remove_tree(entry):
                # Removed as much as it could and no more. The records stay, so the next sweep
                # still knows this tree is nobody's and comes back to it; erasing them here made
                # the leftover unknown to the control plane and therefore permanent.
                continue
            _forget_tree(entry)
    _sweep_records(homes)


def _sweep_records(homes: Sequence[Path]) -> None:
    """Drop control-plane records for trees that are gone and whose owner is too.

    A claim is written before its tree is made, so an interruption between the two leaves a record
    naming a directory that does not exist. That is the right way round, and it is still a file:
    without this the control home grows one small record per crash, for ever. A record is dropped
    only when its tree is absent *and* its owner is gone, which is the same question the tree sweep
    above asks and the same evidence."""
    home_by_name = {home.name: home for home in homes}
    try:
        records = sorted(adapter.control_home().iterdir())
    except OSError:
        return
    for record in records:
        name = record.name
        for kind in (_OWNER_KIND, _ENDED_KIND):
            if not name.endswith(f".{kind}"):
                continue
            stem = name[: -len(f".{kind}")]
            parent, _, leaf = stem.partition("--")
            home = home_by_name.get(parent)
            if home is None or not leaf:
                continue
            root = home / leaf
            if not os.path.lexists(root) and _reclaimable(root):
                _forget_tree(root)


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
#: backlog's shape. It is in the run fingerprint so that "the same pulse" is not mistaken for
#: "the same measurement" across such a change.
#:
#: 3 is the move from a worker process on the host to a worker container, which changed where the
#: filing and the world's digest are read from as well as what the world runs in.
#:
#: 4 is grading the state upstream persists at the end of every block rather than one this port
#: asked the world to write at the seal: two runs that differ across it read a score off two
#: different moments of the same episode.
SCORING_VERSION = 4


def run_fingerprint(
    *,
    pulse: int,
    report: str,
    blocks: int,
    corpus: str = "",
    resources: str = "",
    runtime: Optional[str] = None,
) -> str:
    """Everything two runs must agree on for their rows to be one measurement.

    The draw and the payload class decide what a score *means*; the block budget decides what an
    episode had the chance to do; the corpus and the interpreter decide what world it happened in;
    and :data:`SCORING_VERSION` decides how it was read. ``corpus`` is what the corpus actually
    holds rather than what the pin says it should (see :func:`~adapter.corpus_digest`), because
    the root is whatever the environment points at and a repointed one would otherwise pass for
    the pinned one. ``runtime`` is the image the world actually ran in, as the daemon has it,
    because the tag is a digest over this repository's inputs and says nothing about a base image
    re-pushed under its pin, a transitive version that resolved differently on the day, or the
    same tag built on another platform. A provenance directory reopened under a
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
            # The commit the pinned release claims to be cut from. Nothing on this machine can
            # check that claim against the wheel, so this is the only thing it is load-bearing
            # for: a run under a changed pin is a different measurement, whether or not the
            # artifact behind the version moved with it.
            adapter.UPSTREAM_SHA,
            # What the image turned out to hold, rather than the one version that was asked for.
            # The Dockerfile asks for a release and the resolver answers with a transitive set
            # that depends on the day and on the architecture, so two runs could sit under one
            # tag and one identity with different worlds underneath them.
            runtime if runtime is not None else adapter.runtime_digest(),
            adapter.MANIFEST.read_text(),
            payload.PASS_COUNTS_FILE.read_text(),
            # Every constant a published payload is generated from, and this one was missing.
            # `DRAWN_BASIS` seeds the drawn arm's whole visible vector: changing it re-rolls
            # every drawn payload, which is a change to the treatment an agent is under, and a
            # record could resume across it under an unchanged identity.
            payload.DRAWN_BASIS,
            # The text the agent is actually given. These are authored treatment, not scenery: an
            # edit to the guide or to the appended chore changes what every episode was asked to
            # do, and the digest said nothing about it.
            _WORLD_GUIDE,
            _TOOL_GUIDE,
            world.APPENDED_PARAGRAPH,
            # What decides the seeded backlog. It already names the derived cache, so changing it
            # served a different world under a fingerprint that had not moved.
            adapter._generator_digest(),
            # What machine an episode was given. Captured once for this process and passed to
            # every launch, so an environment changed mid-run cannot move it and a run relaunched
            # under a changed one does not pass for the earlier measurement.
            resources,
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
