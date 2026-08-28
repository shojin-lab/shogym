"""End-to-end: drive served ``appworld`` episodes, and check the hazards that would corrupt them.

The whole path (derive a seeded world, open it in a worker of its own, drive it with ``execute``,
then call ``submit``, the ``score`` terminal, which seals and scores in one step) runs against a
real AppWorld world in a real subprocess. It needs the provisioned interpreter and the corpus;
the module skips when the machine has neither and cannot get them, exactly as the tarball-
provisioned ports' tests do, and ``SHOGYM_REQUIRE_UPSTREAM=1`` removes that escape.

Four hazards are checked here because none of them can be checked anywhere else. A repeat must be
a repeat, in the databases *and* in the state of the generator the world draws from. No outcome
may reach the observation stream before the payload is delivered, and the grading routes AppWorld
publishes without authentication must be out of the agent's reach. And the two arms' payloads must
still match on encoded bytes when the world the agent wrote into holds values that are not ASCII.
"""

from __future__ import annotations

import datetime as dt
import json
import urllib.error
import urllib.request
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests._fixtures.upstream_gate import provisioned

from shogym.envs.appworld import adapter  # noqa: E402

CORPUS = provisioned(adapter.ensure_corpus, package="appworld", extra="appworld")
provisioned(adapter.ensure_apps, package="appworld", extra="appworld")

import shogym  # noqa: E402
from shogym.envs.appworld import ledger, payload, world  # noqa: E402
from shogym.envs.appworld.scorer import draw_key, leg_of  # noqa: E402
from shogym.envs.appworld.worker import TOKEN_HEADER  # noqa: E402
from shogym.serve import ServedEpisode  # noqa: E402

TASK = 0


def task_id(position: int = TASK) -> str:
    return adapter.task_ids()[position]


def backlog_for(position: int = TASK) -> ledger.Backlog:
    specs = adapter.task_specs(CORPUS, task_id(position))
    built = ledger.build_backlog(
        zlib.crc32(task_id(position).encode()),
        dt.datetime.fromisoformat(specs["datetime"]).date(),
    )
    assert built is not None
    return built


async def play(blocks: List[str], *, position: int = TASK, **config: Any) -> Dict[str, Any]:
    """Serve one episode, run ``blocks``, end it, and report everything it produced."""
    env = shogym.make("appworld", config=config)
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=position)
    try:
        spec = episode.describe()
        outputs = []
        for block in blocks:
            result = await episode.call("execute", {"code": block})
            outputs.append(
                {"content": result.content, "meta": dict(result.meta or {})}
            )
        terminal = await episode.call("submit", {})
        return {
            "instructions": spec.instructions,
            "tools": [tool.name for tool in spec.tools],
            "outputs": outputs,
            "verdict": json.loads(terminal.content),
            "feedback": {
                item["name"]: item["value"]
                for item in (terminal.meta.get("shogym/feedback") or [])
            },
        }
    finally:
        await episode.close()


def filing_block(position: int = TASK, *, pulse: int = 0, section: bool = True) -> str:
    """Code that files the log the drawn convention asks for. The reference solution."""
    built = backlog_for(position)
    key = draw_key(leg_of(task_id(position)), pulse)
    lines = "\n".join(
        f"{request.reference}, {band}"
        for request, band in zip(built.requests, built.key(key.convention))
    )
    chosen, color, unit, priority = key.slots
    return f'''
pw = [x for x in apis.supervisor.show_account_passwords() if x["account_name"] == "todoist"][0]
token = apis.todoist.login(
    username=apis.supervisor.show_profile()["email"], password=pw["password"]
)["access_token"]
project = [
    p for p in apis.todoist.show_projects(access_token=token, page_limit=50)
    if p["name"] == {world.PROJECT_NAME!r}
][0]["project_id"]
sections = {{
    s["name"]: s["section_id"]
    for s in apis.todoist.show_sections(access_token=token, project_id=project)
}}
filed = apis.todoist.create_task(
    access_token=token, project_id=project,
    section_id=sections[{chosen!r}] if {section!r} else None,
    title={world.LOG_TITLE!r}, description="""{lines}""",
    duration=1.0, duration_unit={unit!r}, priority={priority!r},
)
label = apis.todoist.create_label(access_token=token, name={world.LOG_LABEL!r}, color={color!r})
apis.todoist.add_label_to_task(
    access_token=token, task_id=filed["task_id"], label_id=label["label_id"]
)
print("filed")
'''


# ----- the treatment reaches the agent -----


async def test_the_appended_paragraph_is_in_the_instructions_verbatim() -> None:
    env = shogym.make("appworld")
    first = env.describe("0").instructions
    second = env.describe("5").instructions
    assert world.APPENDED_PARAGRAPH in first
    assert world.APPENDED_PARAGRAPH in second
    # Byte-identical everywhere and last, so an agent reading its hundredth task reads the words
    # it read on its first and nothing about the position is in the text.
    assert first.endswith(world.APPENDED_PARAGRAPH)
    assert second.endswith(world.APPENDED_PARAGRAPH)
    assert first != second


async def test_the_world_holds_the_backlog_the_paragraph_points_at() -> None:
    built = backlog_for()
    played = await play(
        [
            '''
pw = [x for x in apis.supervisor.show_account_passwords() if x["account_name"] == "todoist"][0]
token = apis.todoist.login(
    username=apis.supervisor.show_profile()["email"], password=pw["password"]
)["access_token"]
project = [
    p for p in apis.todoist.show_projects(access_token=token, page_limit=50)
    if p["name"] == "Task Log"
][0]
listing = apis.todoist.show_tasks(
    access_token=token, project_id=project["project_id"], page_limit=50
)
rows = listing["no_section_tasks"]
import json
print(json.dumps({
    "description": project["description"],
    "sections": [
        s["name"]
        for s in apis.todoist.show_sections(access_token=token, project_id=project["project_id"])
    ],
    "titles": sorted(r["title"] for r in rows),
    "bands": sorted({r.get("description") or "" for r in rows}),
}))
'''
        ]
    )
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    assert seen["description"] == built.description
    assert seen["sections"] == list(ledger.SECTIONS)
    assert seen["titles"] == sorted(request.reference for request in built.requests)
    # Nothing seeded carries a band, a section or a due date: one that did would set the leak to
    # one on every task at once and nothing downstream would notice.
    for description in seen["bands"]:
        for band in ledger.BANDS:
            assert band not in description


async def test_filing_the_drawn_convention_scores_the_whole_ledger() -> None:
    played = await play([filing_block()])
    feedback = played["feedback"]
    assert feedback["ledger_fraction"] == 1.0
    assert feedback["pinned_fraction"] == 1.0
    assert feedback["exercise_fraction"] == 1.0
    assert feedback["parse_fraction"] == 1.0
    assert feedback["filing_rows"] == 1.0
    assert feedback["report"].count(payload.PASS) >= 33


async def test_the_scenario_s_own_checks_survive_the_appended_chore() -> None:
    # The chore adds three models to the world. A scenario that asserted "nothing else changed"
    # would fail on the port rather than on the agent if those were not ignored, so an episode
    # that files a perfect log must not score worse on the base task than one that files nothing.
    filed = await play([filing_block()])
    idle = await play([])
    assert filed["feedback"]["assertion_fraction"] >= idle["feedback"]["assertion_fraction"]


# ----- hazard: the same task twice is the same world, generator state included -----


async def test_the_same_task_served_twice_yields_the_same_world_and_the_same_rng_state() -> None:
    block = 'print(apis.supervisor.show_profile()["email"])'
    first = await play([block, filing_block()])
    second = await play([block, filing_block()])
    for one, two in zip(first["outputs"], second["outputs"]):
        assert one["content"] == two["content"]
    # The databases, and the state of the generator the world draws from. AppWorld saves the
    # first and not the second, and every login draws from it, so a port that compared only the
    # databases would call two worlds identical while their next draw differed.
    assert first["feedback"]["world_digest"] == second["feedback"]["world_digest"]
    assert first["feedback"]["rng_digest"] == second["feedback"]["rng_digest"]
    assert first["feedback"]["ledger_fraction"] == second["feedback"]["ledger_fraction"]


async def test_two_different_tasks_are_two_different_worlds() -> None:
    first = await play([], position=0)
    second = await play([], position=1)
    assert first["feedback"]["world_digest"] != second["feedback"]["world_digest"]


# ----- hazard: no outcome before the payload, and no reachable grading route -----


async def test_no_tool_result_carries_an_outcome_before_the_terminal() -> None:
    played = await play(['print("hello")', filing_block()])
    for step in played["outputs"]:
        assert "shogym/feedback" not in step["meta"]
        for word in ("PASS", "FAIL", "ledger_fraction", "SUBMISSION RECEIPT"):
            assert word not in step["content"]


async def test_the_tool_surface_cannot_grade_anything() -> None:
    env = shogym.make("appworld")
    names = {tool.name for tool in env.describe("0").tools}
    assert names == {"execute", "submit", "terminate"}


async def test_the_world_process_answers_nothing_without_its_token() -> None:
    # AppWorld's own environment server publishes evaluate, save_state and load_state with no
    # authentication at all. This one refuses every command without the token the serving process
    # holds, so an agent that found the port still cannot ask the world what the answer was.
    worker = adapter.Worker.spawn(adapter.derived_root() / "data")
    try:
        # The two commands this protocol has, and three it does not: a token is checked before
        # the path is looked up, so an unknown command is refused as an unauthenticated one and
        # never as a hint about what the protocol carries.
        for command in ("open", "execute", "evaluate", "quiesce", "read"):
            request = urllib.request.Request(
                f"http://127.0.0.1:{worker.port}/{command}",
                data=b"{}",
                headers={"Content-Type": "application/json"},
            )
            with pytest.raises(urllib.error.HTTPError) as refused:
                urllib.request.urlopen(request, timeout=30)
            assert refused.value.code == 403
        # And a wrong token is refused exactly as a missing one is.
        request = urllib.request.Request(
            f"http://127.0.0.1:{worker.port}/evaluate",
            data=b"{}",
            headers={"Content-Type": "application/json", TOKEN_HEADER: "not-the-token"},
        )
        with pytest.raises(urllib.error.HTTPError) as refused:
            urllib.request.urlopen(request, timeout=30)
        assert refused.value.code == 403
    finally:
        worker.close()


async def test_the_port_and_the_token_are_absent_from_every_standard_surface() -> None:
    """What the port and the token are kept off, which is every surface this env answers on: the
    instructions, a tool's schema, a tool's result and the terminal's own metadata. Argv and the
    environment are checked beside these, in the probe test below.

    That is the whole of the claim, and the name says so now because the old one ("no
    agent-visible surface") claimed more than any test here can. Agent-authored code runs *as* the
    worker process and the token is that process's own handler state, so it is a gate against
    another process on this machine and never a secret from the code running in this one (see
    `worker.TOKEN_HEADER`)."""
    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        from shogym.envs.appworld import mcp_server

        session = mcp_server.get_session(episode.session_id)
        assert session is not None
        secrets = (str(session.worker.port), session.worker.token)
        surfaces = [episode.describe().instructions]
        result = await episode.call("execute", {"code": 'print("hello")'})
        surfaces.append(result.content)
        surfaces.append(json.dumps(episode.describe().model_dump(), default=str))
        terminal = await episode.call("submit", {})
        surfaces.append(terminal.content)
        surfaces.append(json.dumps(terminal.meta or {}))
        for surface in surfaces:
            for secret in secrets:
                assert secret not in surface
    finally:
        await episode.close()


async def test_agent_authored_code_cannot_reach_what_the_boundary_hides() -> None:
    """The probes a curious agent actually runs, and what each of them may not find.

    The code an agent writes runs *as* the worker process, so the worker is not a sandbox and this
    does not pretend otherwise. What it is is a set of things deliberately not put where that code
    can reach: the token and the corpus root are not on the command line, the answers are not in
    the process and not in the corpus tree it was given, and the serving process's environment was
    left behind. Each probe below is one the review named."""
    # Written the way the review wrote them. AppWorld's own guard refuses a plain `import sys`
    # and lets `__import__("sys")` through, which is the whole reason the guard is not a boundary
    # and these properties have to hold without it.
    argv_probe = """
print(json.dumps(__import__("sys").argv))
"""
    env_probe = """
print(json.dumps(dict(__import__("os").environ)))
"""
    disk_probe = """
_io, _os = __import__("io"), __import__("os")
root = _os.environ["APPWORLD_ROOT"]


def _read(path):
    try:
        return _io.open(path).read()[:40]
    except Exception as failure:
        return type(failure).__name__


print(json.dumps({
    "root": root,
    "readable": _read(root + "/data/tasks/%s/specs.json"),
    "answers": _read(root + "/data/tasks/%s/ground_truth/answer.json"),
}))
""" % (task_id(), task_id())
    played = await play([argv_probe, env_probe, disk_probe])
    argv = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    environment = json.loads(json.loads(played["outputs"][1]["content"])["output"])
    filesystem = json.loads(json.loads(played["outputs"][2]["content"])["output"])

    # 1. Nothing on the command line but the interpreter, the script and the subcommand. The
    #    token and the root arrive on stdin, which is read once and closed.
    assert argv[-1] == "serve"
    joined = " ".join(argv)
    assert "--token" not in joined and "--root" not in joined
    assert filesystem["root"] not in joined

    # 2. The serving process's environment did not come along, so an inherited provider key is
    #    not sitting there for the taking.
    # The allow-list, what the worker sets for itself, and the handful the platform and the
    # interpreter inject into every process no matter what they were handed.
    platform = {"__CF_USER_TEXT_ENCODING", "__PYVENV_LAUNCHER__", "PYTHONHASHSEED"}
    allowed = (
        set(adapter._ENV_ALLOW_LIST)
        # `PYTHONDONTWRITEBYTECODE` is what keeps every cache in the runtime a hash-based one, so
        # the runtime digest can leave `__pycache__` out and still say what executes.
        | {"HOME", "APPWORLD_CACHE", "APPWORLD_ROOT", "PYTHONDONTWRITEBYTECODE"}
        | platform
    )
    assert set(environment) <= allowed, sorted(set(environment) - allowed)
    assert not [
        name
        for name in environment
        if any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    ]

    # 3. The answers are not in the tree the world was served from. They ship in the corpus in
    #    plaintext, so the tree an agent's world is given simply does not carry them. Both halves
    #    matter: the corpus really is readable from here, and the answers really are not in it.
    assert filesystem["readable"].startswith("{")
    assert filesystem["answers"] == "FileNotFoundError"


async def test_the_served_tree_has_no_path_that_leads_to_the_answers() -> None:
    """The route a reviewer actually walked: follow a link out of the served tree, then take the
    sibling. It worked, because `specs.json` was a symlink into the corpus and the corpus holds
    every task's answers next door.

    **One invariant, stated where it can be checked.** This test used to demand that the served
    root hold no symlinks at all, which was a proxy for the thing that matters and stopped being
    equivalent to it when an episode got a served view of its own: the view names the shared
    derived base rather than copying 134 MB per episode. A symlink is not the defect; a symlink
    whose *target* has the answers as a sibling is. So what is asserted here is the target
    contract, walked the way the reviewer walked it: step through each link, ask the directory it
    landed in for this task, and find the task there and no `ground_truth` beside it. The
    read-only half of the same invariant, which is what keeps one arm of a pair out of the
    other's inputs, is asserted below and at
    `test_appworld_runtime.py::test_two_episodes_of_one_task_do_not_share_their_served_inputs`."""
    probe = """
_io, _os = __import__("io"), __import__("os")
root = _os.environ["APPWORLD_ROOT"]
task = root + "/data/tasks/%(task)s"


def _read(path):
    try:
        return _io.open(path).read()[:40]
    except Exception as failure:
        return type(failure).__name__


def _mode(path):
    # The permissions as the worker itself sees them, rather than a write attempt, because a write
    # cannot be measured from in here: upstream's guard replaces `io.open` with one that refuses
    # every write mode whatever the path, and null-patches `os.open`, which then reports success
    # and creates nothing. Both would answer about the guard rather than about the filesystem.
    # What a 0o555 directory does to a write is proved on the host, in
    # `test_appworld_runtime.py::test_two_episodes_of_one_task_do_not_share_their_served_inputs`.
    try:
        return oct(_os.stat(path).st_mode & 0o777)
    except Exception as failure:
        return type(failure).__name__


def _target(path):
    try:
        return _os.readlink(path)
    except Exception as failure:
        return type(failure).__name__


# The shared entries named by name, because AppWorld's own guard null-patches `os.listdir` inside
# an `execute` call and there is nothing here to enumerate them with.
shared = ["base_dbs", "datasets", "api_docs"]
# Take every link out of the served root, step to the directory its target sits in, and ask that
# directory for this task. That is the reviewer's walk exactly: `specs.json` was a link into the
# corpus, and the corpus keeps every task's answers in the folder beside its specs.
neighbours = []
answers = []
writable = []
parents = []
for name in shared:
    if not _os.path.islink(root + "/data/" + name):
        continue
    here = _os.path.dirname(_os.path.realpath(root + "/data/" + name))
    # The directory the link lands *in*, not the entry it lands on. These links are absolute, so
    # what this episode resolves is the name as much as the bytes, and a name lives in its parent:
    # a writable parent is a rename away from putting something else under `base_dbs` for this
    # episode and for the other arm of its pair.
    parents.append(_mode(here))
    if _os.path.exists(here + "/tasks/%(task)s/specs.json"):
        neighbours.append(name)
    if _os.path.exists(here + "/tasks/%(task)s/ground_truth"):
        answers.append(name)
    # Continue the same walk into the shared task cache the links land beside: it is the pristine
    # source every later episode's view is copied out of, so anything writable in it reaches the
    # next episode and the other arm of this one's pair.
    for reachable in (
        "/tasks",
        "/tasks/%(task)s",
        "/tasks/%(task)s/specs.json",
        "/tasks/%(task)s/dbs",
        "/tasks/%(task)s/dbs/gmail.jsonl",
        "/tasks/%(task)s/dbs/todoist.jsonl",
    ):
        path = here + reachable
        if _os.path.exists(path) and (_os.stat(path).st_mode & 0o222):
            writable.append(name + reachable)
print(json.dumps({
    "specs": _read(task + "/specs.json"),
    "specs_is_link": _os.path.islink(task + "/specs.json"),
    "specs_target": _target(task + "/specs.json"),
    "db_is_link": _os.path.islink(task + "/dbs/gmail.jsonl"),
    "sibling": _read(task + "/ground_truth/test_data.json"),
    "neighbours": sorted(set(neighbours)),
    "answers_beyond_a_link": sorted(set(answers)),
    "writable_shared_inputs": sorted(set(writable)),
    "shared_parent_modes": sorted(set(parents)),
    # The base every episode shares. A write through it would be in the next episode's inputs.
    "shared_mode": _mode(root + "/data/api_docs"),
    "shared_file_mode": _mode(root + "/data/base_dbs/admin.db"),
    "own_mode": _mode(task),
}))
""" % {"task": task_id()}
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])

    # The corpus is genuinely readable from here, so the negatives below mean something.
    assert seen["specs"].startswith("{")
    # And nothing in it names where it came from.
    assert seen["specs_is_link"] is False
    assert seen["specs_target"] == "OSError"
    assert seen["db_is_link"] is False
    assert seen["sibling"] == "FileNotFoundError"
    # Links are allowed; a link whose target has the answers next door is not. The walk really did
    # land somewhere holding this task, which is what makes the second assertion mean something:
    # on the head this test was written against, that somewhere was the corpus, and the corpus
    # keeps every task's answers in the folder beside its specs.
    assert seen["neighbours"] == ["api_docs", "base_dbs", "datasets"]
    assert seen["answers_beyond_a_link"] == []
    # And nothing the walk reaches is writable, which is the other half of the invariant: the
    # shared base was sealed and the shared task cache beside it was not, so a served worker could
    # edit the pristine copy the next episode is built from.
    assert seen["writable_shared_inputs"] == []
    # And the directory those names live in is read-only too, which sealing each entry did not
    # do. The links are absolute, so what this episode resolves is the name as much as the bytes:
    # under an owner-writable parent, `base_dbs` could be renamed aside and a directory of the
    # episode's own choosing put there under the same name, and every view resolving it afterwards
    # would follow — including the other arm of this one's pair. What a 0o555 directory does to a
    # rename is proved on the host, in
    # `test_appworld_runtime.py::test_the_shared_parent_cannot_be_renamed_around`.
    assert seen["shared_parent_modes"] == ["0o555"]
    # The episode owns its task and nothing else. The shared base an episode links to is sealed
    # read-only, so a write through it cannot reach the next episode's starting inputs or the
    # other arm of its own pair's; its own task copy is writable and goes with the episode.
    assert seen["shared_mode"] == "0o555"
    assert seen["shared_file_mode"] == "0o444"
    assert int(seen["own_mode"], 8) & 0o200


async def test_the_graders_tree_is_not_a_neighbour_of_the_served_one() -> None:
    """The independent route: the grader's view used to be a fixed name beside the served root,
    so knowing one gave you the other. It now lives under a different parent with an unguessable
    name. This raises the cost and does not close it, and the README says so; what it checks is
    that the tree is no longer derivable from what the worker is handed.

    **The substitution is made from the derived corpus and not from `APPWORLD_ROOT`.** Since each
    episode is served a view of its own, `APPWORLD_ROOT` names that view and has no `seeded` in
    it, so the old probe substituted nothing and then asked whether the served root existed. It
    does, so the test asserted the failure it was written to catch. The derived corpus is still
    reachable from inside, by following one of the view's shared links, so the guess is made from
    there: that is the name the grader's tree used to sit beside."""
    from shogym.envs.appworld import adapter as adapter_module

    served, graded = adapter_module.derived_root(), adapter_module.graded_root()
    assert graded.parent != served.parent
    assert str(served.parent) not in str(graded)
    # Sixteen hex characters of the name are drawn, not derived from anything public.
    assert not str(graded).startswith(str(adapter_module.cache_root()))
    probe = """
_os = __import__("os")
root = _os.environ["APPWORLD_ROOT"]
base = _os.path.dirname(_os.path.dirname(_os.path.realpath(root + "/data/base_dbs")))
print(json.dumps({
    "base": base,
    "guess": _os.path.exists(base.replace("seeded", "graded")),
    "beside_the_view": _os.path.exists(_os.path.dirname(root) + "/graded"),
}))
"""
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    # The walk landed on the derived corpus, so the substitution below is made on the name it
    # would have to be made on.
    assert seen["base"] == str(served)
    assert seen["guess"] is False
    assert seen["beside_the_view"] is False


@pytest.mark.xfail(
    strict=True,
    reason=(
        "shojin-lab/shogym#138: the worker is a scrubbed process and not an OS namespace, so "
        "agent-authored code reads whatever the user running the port can read. The containerised "
        "worker turns this green; remove the marker with it."
    ),
)
async def test_a_read_outside_the_served_tree_is_refused() -> None:
    """The property the closed routes do not give, reproduced so it cannot be forgotten.

    Everything the served tree could have led to has been taken out of it: no answers, no symlink
    ancestry, no neighbouring grader. None of that stops a read of a path an agent names outright,
    because the code it writes runs as the worker, with the worker's filesystem, as the same user
    as the run. A mount namespace is what makes this fail; nothing short of one does.

    Expected to fail today, and strictly: when it starts passing, the boundary exists and the
    marker is what should go.

    **The sentinel is a host-only file, deliberately.** This read `/etc/hosts`, which a correctly
    isolated container has its own readable copy of, so the test would have gone green on a
    container that was working exactly as intended and told us nothing about whether the boundary
    held. The file below is written by this test on the host, outside every tree the worker is
    given, so it can only be read by a process sharing the host's mount namespace."""
    sentinel = adapter.cache_root().parent / f"host-only-{uuid.uuid4().hex}"
    sentinel.write_text("readable only from the host namespace")
    try:
        probe = f"""
_io = __import__("io")
try:
    _io.open({str(sentinel)!r}).read()
    print(json.dumps({{"read": True}}))
except Exception as failure:
    print(json.dumps({{"read": False, "why": type(failure).__name__}}))
"""
        played = await play([probe])
        seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
        assert seen["read"] is False
    finally:
        sentinel.unlink(missing_ok=True)


async def test_the_answers_are_not_in_the_process_that_runs_agent_code() -> None:
    """The world is built without ground truth, so there is no evaluator to call and no expected
    value to walk to from a frame. Grading happens in a second process that never runs agent
    code, reading the end state off disk."""
    probe = """
_gc, _sys = __import__("gc"), __import__("sys")
kind = _sys.modules["appworld.environment"].AppWorld
live = [o for o in _gc.get_objects() if isinstance(o, kind)]
print(json.dumps({
    "worlds": len(live),
    "ground_truth": [w.task.ground_truth is not None for w in live],
}))
"""
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    assert seen["worlds"] >= 1
    assert seen["ground_truth"] == [False] * seen["worlds"]
    # And the base task is still graded, by the process that does hold them.
    assert played["feedback"]["checks"] > 0


# ----- hazard: the payloads match on bytes even when the world does not speak ASCII -----


async def test_the_two_arms_match_on_encoded_bytes_with_non_ascii_in_the_world() -> None:
    hostile = "​Sectioń«\U0001f600"
    played = await play(
        [
            f'''
pw = [x for x in apis.supervisor.show_account_passwords() if x["account_name"] == "todoist"][0]
token = apis.todoist.login(
    username=apis.supervisor.show_profile()["email"], password=pw["password"]
)["access_token"]
project = [
    p for p in apis.todoist.show_projects(access_token=token, page_limit=50)
    if p["name"] == "Task Log"
][0]["project_id"]
filed = apis.todoist.create_task(
    access_token=token, project_id=project, title="Filing",
    description="""{hostile}, Routine""", priority="high",
)
print("filed")
'''
        ]
    )
    feedback = played["feedback"]
    report, notice = feedback["report"], feedback["notice"]
    assert len(report.encode()) == len(notice.encode())
    assert len(json.dumps(report)) == len(json.dumps(notice))
    assert report.isascii() and notice.isascii()
    assert hostile not in report and hostile not in notice


async def test_a_block_budget_of_n_allows_exactly_n_blocks_to_touch_the_world() -> None:
    """Exercised, not read off the constructor.

    The serve layer dispatches the call that *reaches* the horizon and cannot tell an `execute`
    from a terminal, so a budget of N published as N + 1 let call N + 1 be another block, changing
    the world after the budget it was to be scored under had run out. `execute` therefore counts
    its own calls and refuses past the budget without touching the world.

    The proof is behavioural: the over-budget call is the one that would have filed a perfect
    ledger, and the episode scores zero on it."""
    env = shogym.make("appworld", config={"horizon": 2})
    # One slot past the block budget, and the slot exists so a terminal always has somewhere to go.
    assert env.describe("0").horizon == 3
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        used = [
            json.loads((await episode.call("execute", {"code": "print(%d)" % n})).content)
            for n in (1, 2)
        ]
        assert [step["calls"] for step in used] == [1, 2]
        assert [step["output"].strip() for step in used] == ["1", "2"]
        # Call three reaches the horizon, so the serve layer runs it and then finalizes. What it
        # must not do is change the world, and this one would have filed a perfect log.
        terminal = await episode.call("execute", {"code": filing_block()})
        verdict = json.loads(terminal.content)
        assert verdict["ledger_fraction"] == 0.0
        assert verdict["exercise_fraction"] == 0.0
        assert verdict["filing_rows"] == 0.0
    finally:
        await episode.close()


async def test_the_same_block_inside_the_budget_does_file_the_log() -> None:
    """The other half, without which the test above passes for the wrong reason."""
    env = shogym.make("appworld", config={"horizon": 2})
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        await episode.call("execute", {"code": filing_block()})
        terminal = await episode.call("submit", {})
        feedback = {
            item["name"]: item["value"]
            for item in (terminal.meta.get("shogym/feedback") or [])
        }
        assert feedback["ledger_fraction"] == 1.0
        assert feedback["filing_rows"] == 1.0
    finally:
        await episode.close()


async def test_one_episodes_grade_is_not_readable_by_the_next(tmp_path: Path) -> None:
    """The failure this closes is the one the paired design cannot survive: the placebo member of
    a pair reading the receipt of its twin.

    Upstream's evaluator writes a report beside the episode's output by default, quoting the
    requirement prose and the values behind it, and every worker used to be handed the same root
    to find it under. Report writing is off, and an episode's output tree is now named absolutely
    and lives outside any served corpus, so there is nothing of one episode inside another's
    world."""
    first = await play([filing_block()])
    assert first["feedback"]["assertion_fraction"] >= 0.0  # the grader really ran

    probe = """
_io, _os = __import__("io"), __import__("os")
root = _os.environ["APPWORLD_ROOT"]


def _read(path):
    try:
        return _io.open(path).read()[:60]
    except Exception as failure:
        return type(failure).__name__


print(json.dumps({
    "root": root,
    "experiments": _read(root + "/experiments/outputs"),
    "report": _read(root + "/experiments/outputs/report.md"),
}))
"""
    second = await play([probe])
    seen = json.loads(json.loads(second["outputs"][0]["content"])["output"])
    # Nothing of any episode's output is inside the tree a world is served from.
    assert seen["experiments"] in ("FileNotFoundError", "IsADirectoryError", "NotADirectoryError")
    assert seen["report"] == "FileNotFoundError"
    # And no evaluator report exists anywhere under the served corpus.
    served = adapter.derived_root()
    assert list(served.rglob("report.md")) == []


async def test_one_episodes_write_is_not_in_the_next_episodes_world() -> None:
    """The served inputs are per episode, so a write through one does not start the next.

    The derived corpus was one deterministic global root and every worker was handed it, with its
    files writable by the process that runs agent-authored code and nothing putting them back. A
    write through episode A's served view was therefore still there in episode B's starting inputs.
    Two arms of a pair are the same task served at the same time, so the arm meant to differ only
    in what it was told could also differ in the world it was given, and that is a difference the
    treatment did not make.

    **The write is made from here rather than from inside `execute`, and it has to be.** Upstream's
    own guard replaces `io.open` with one that refuses every write mode and null-patches `os.open`
    so that it reports success and creates nothing, so no code running inside an episode can write
    a file or report truthfully that it failed to. What the route through the guard would have been
    testing is upstream's guard; what matters here is the filesystem. So the pathname is the one
    the worker was handed, the worker is asked to confirm it sees the write, and the episode that
    follows is asked what it sees at the same place."""
    task = task_id()
    marker = "written by an earlier episode"
    served = 'root + "/data/tasks/%s/dbs/gmail.jsonl"' % task
    report = (
        '_os = __import__("os")\n'
        'root = _os.environ["APPWORLD_ROOT"]\n'
        '_io = __import__("io")\n'
        'print(json.dumps({"root": root, "body": _io.open(%s).read()[:64]}))\n' % served
    )

    env = shogym.make("appworld")
    first = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        before = json.loads(json.loads((await first.call("execute", {"code": report})).content)["output"])
        target = Path(before["root"]) / "data" / "tasks" / task / "dbs" / "gmail.jsonl"
        assert before["body"] != marker
        # Through the pathname the worker is actually given, while that worker is running.
        target.write_text(marker)
        after = json.loads(json.loads((await first.call("execute", {"code": report})).content)["output"])
        # The write really did land in the world this episode is being served, so the negative
        # below is about isolation rather than about a write that never happened.
        assert after["body"] == marker
    finally:
        await first.close()

    second = json.loads(json.loads((await play([report]))["outputs"][0]["content"])["output"])
    assert second["body"] != marker, "the second episode started in the first one's leftovers"
    # Two episodes, two served roots. One shared root is what carried the write.
    assert before["root"] != second["root"]
    # The view the first episode wrote through is gone with the episode that owned it.
    assert not target.exists()
    # And the pristine copy the views are built from never saw it.
    pristine = adapter.derived_root() / "data" / "tasks" / task / "dbs" / "gmail.jsonl"
    assert pristine.read_text()[:64] != marker


async def test_the_world_stops_before_it_is_graded() -> None:
    """Sealing closes the tool surface, so the worker is stopped and reaped before anything reads
    what it left: the filing, the digests and the evaluator all read one state.

    What that does *not* prove is that nothing of the episode's is still running. Agent code runs
    in the worker and is free to spawn, and a descendant of it survives this stop. Closing that
    needs a namespace rather than a signal, which is shojin-lab/shogym#140."""
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        session = mcp_server.get_session(episode.session_id)
        assert session is not None
        worker = session.worker
        assert worker.process.poll() is None
        await episode.call("submit", {})
        # Graded, and the process that could have changed the world was reaped before a byte of
        # what it wrote was read.
        assert worker.process.poll() is not None
    finally:
        await episode.close()
    # The copy the grader was given goes with the episode too.
    assert not Path(str(adapter.episode_outputs(episode.session_id)) + ".graded").exists()


# ----- the matched pair, through a stream -----


async def test_information_hands_back_the_receipt_and_placebo_the_digest(
    tmp_path: Path,
) -> None:
    from shogym.serve.stream import Information, Placebo, TaskRef, TaskStream

    answers = {}
    for name, policy in (("information", Information()), ("placebo", Placebo())):
        stream = TaskStream(
            shogym.make,
            [TaskRef("appworld", TASK)],
            prov_dir=tmp_path / name,
            feedback=policy,
            # The port's own fingerprint, which is what makes two of these runs one measurement:
            # the draw, the payload class, the block budget, the corpus contents and the scoring
            # version. A resume under a changed pulse or a repointed corpus is refused by it.
            identity=shogym.make("appworld").config_digest,
        )
        async with stream:
            await stream.get_task()
            await stream.dispatch("execute", {"code": filing_block()})
            terminal = await stream.dispatch("submit", {})
            answers[name] = json.loads(terminal.content[0].text)
        assert [row.feedback_regime for row in stream.results] == [name]

    revealed = {
        name: {item["name"]: item["value"] for item in answer["feedback"]}
        for name, answer in answers.items()
    }
    # One item each, under one public name, and neither arm is handed the numbers. The env files
    # its two versions under two names so the record can tell them apart; what reaches the agent
    # is named the same in both, or the control would announce its own arm in the field name.
    from shogym.feedback.wire import CHANNEL_FEEDBACK_NAME

    assert set(revealed["information"]) == {CHANNEL_FEEDBACK_NAME}
    assert set(revealed["placebo"]) == {CHANNEL_FEEDBACK_NAME}
    receipt = revealed["information"][CHANNEL_FEEDBACK_NAME]
    digest = revealed["placebo"][CHANNEL_FEEDBACK_NAME]
    assert "SUBMISSION RECEIPT" in receipt and "SUBMISSION RECEIPT" in digest
    assert payload.PASS in receipt and payload.PASS not in digest
    assert len(receipt.encode()) == len(digest.encode())
    # And the two answers are the same size on the wire, not just the two values.
    assert len(json.dumps(answers["information"])) == len(json.dumps(answers["placebo"]))


async def test_the_documented_launch_serves_each_arm_of_the_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The command in this port's README, run, both arms of it.

    The test above builds a `TaskStream` by hand, which is not what anybody launches. What the
    README documents is the Claude Code quickstart: an MCP config that spawns `serve.py`, and an
    environment that says which env, which tasks and which regime. That path served
    `Immediate()` and nothing else, so the documented way to run this env handed one agent the
    receipt, its placebo and every numeric grade at once. It was neither arm of the pair it was
    documented as running, and no test noticed because none of them ran it.

    So this drives the quickstart's own module the way its own MCP server does, once per arm, and
    asks the three questions the launch has to answer: does the regime reach the stream, does each
    arm reveal only its own channel, and does each arm land a row of its own that says which arm
    it was. The rows are read back with the quickstart's reader, which is also what makes this a
    check on the summary the reader prints.

    **The environment below is the README's, not this test's.** The arm now travels in the MCP
    config's `env` block rather than on the agent's command line, so what a launch hands this
    server is whatever those documented commands write into that block; typing the same variables
    out again here would check a second document. `test_appworld_launch.py` runs the commands, and
    holds the other half of the separation: that the agent's own process is byte-identical across
    the two arms."""
    import importlib

    from fastmcp import Client

    from tests.envs.test_appworld_launch import documented_arms

    from examples.claude_code import results as results_mod
    from examples.claude_code import serve as serve_mod
    from shogym.feedback.wire import CHANNEL_FEEDBACK_NAME
    from shogym.serve.stream import build_stream_server

    configured = documented_arms(tmp_path / "documented")
    assert sorted(configured) == ["information", "placebo"]

    directories: List[Path] = []
    try:
        for arm in ("information", "placebo"):
            for name, value in configured[arm].items():
                monkeypatch.setenv(name, value)
            # The constants are read from the environment at import, which is what a launch does.
            importlib.reload(serve_mod)
            assert (serve_mod.ENV, serve_mod.TASKS, serve_mod.FEEDBACK) == (
                "appworld",
                [TASK],
                arm,
            )
            # Each arm names its own directory, without being told to.
            directories.append(serve_mod.new_run_dir(runs=tmp_path))

            stream = serve_mod.build_stream(prov_dir=tmp_path / arm)
            assert stream.feedback.regime == arm
            async with stream:
                async with Client(build_stream_server(stream, name="shogym")) as client:
                    dispensed = json.loads(
                        (await client.call_tool("get_task", {})).content[0].text
                    )
                    assert set(dispensed) == {"env", "instructions", "budget", "tools"}
                    terminal = json.loads((await client.call_tool("submit", {})).content[0].text)

            revealed = {item["name"]: item["value"] for item in terminal["feedback"]}
            # One item, under the one public name, and never the numbers.
            assert set(revealed) == {CHANNEL_FEEDBACK_NAME}
            assert "SUBMISSION RECEIPT" in revealed[CHANNEL_FEEDBACK_NAME]

            rows = results_mod.rows(tmp_path / arm)
            assert [row.feedback_regime for row in rows] == [arm]
            assert rows[0].score is not None
            # And the row is summarisable, which is what the reader counts and prints.
            assert rows[0].score.reward is not None
    finally:
        # The module's constants are process-wide, so they go back to what the environment says.
        monkeypatch.undo()
        importlib.reload(serve_mod)

    assert directories[0] != directories[1], "the two arms would have shared one record"
