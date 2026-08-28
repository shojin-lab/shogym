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

import asyncio
import contextlib
import datetime as dt
import json
import re
import uuid
import zlib
from pathlib import Path
from typing import Any, Dict, List

import pytest

from tests._fixtures.upstream_gate import provisioned

from shogym.envs.appworld import adapter  # noqa: E402

# Docker first: every world runs in a container and there is no host fallback, so a machine
# without a daemon cannot run any of this. `ensure_image` raises the provisioning error the gate
# reads as "not provisioned here", and `SHOGYM_REQUIRE_UPSTREAM=1` turns it back into a failure.
provisioned(adapter.ensure_image, package="appworld", extra="appworld")
CORPUS = provisioned(adapter.ensure_corpus, package="appworld", extra="appworld")

import shogym  # noqa: E402
from shogym.envs.appworld import ledger, payload, world  # noqa: E402
from shogym.envs.appworld.scorer import draw_key, leg_of  # noqa: E402
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


async def test_agent_authored_code_cannot_reach_what_the_boundary_hides() -> None:
    """The probes a curious agent actually runs, and what each of them may not find.

    These are the round-two probes, unchanged in what they ask and changed in what the answer
    means. They used to check that things had not been *put* where agent code could reach them,
    on a worker that was the same uid as the run and could read whatever the run could. The world
    is a container now, so the same probes are checking that a filesystem holds one task's tree
    and nothing else."""
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

    # 1. Nothing on the command line but the script and the subcommand. There is no token and no
    #    root to leak there any more: the root is a mount point named the same in every episode.
    assert argv[-1] == "serve"
    joined = " ".join(argv)
    assert "--token" not in joined and "--root" not in joined
    assert filesystem["root"] == "/corpus"

    # 2. The serving process's environment did not come along. Not because it was filtered: a
    #    container is given the image's own environment and what `docker run -e` names, so an
    #    inherited provider key was never offered to it.
    from shogym.envs.appworld import container as container_module

    image = {"GPG_KEY", "HOSTNAME", "PATH", "PYTHON_SHA256", "PYTHON_VERSION", "PYTHONUNBUFFERED"}
    ours = {"APPWORLD_ROOT", "APPWORLD_CACHE", "HOME", "LANG", "PYTHONDONTWRITEBYTECODE"}
    # Docker's client injects these from whatever proxy profile is configured, so this port passes
    # them empty rather than leaving them: the names may be present, the values may not.
    ours |= set(container_module._PROXY_VARIABLES)
    assert not [
        name for name in container_module._PROXY_VARIABLES if environment.get(name)
    ], "a configured proxy reached the world"
    assert set(environment) <= image | ours, sorted(set(environment) - image - ours)
    # Nothing secret-shaped among the names this port passes. The base image's own `GPG_KEY` is
    # excluded by name and not by luck: it is the published CPython release signing key that every
    # official python image carries, so it is part of the machinery rather than part of the run.
    assert not [
        name
        for name in set(environment) - image
        if any(word in name.upper() for word in ("KEY", "TOKEN", "SECRET", "PASSWORD"))
    ]

    # 3. The answers are not in the tree the world was served from. Both halves matter: the served
    #    tree really is readable from here, and the answers really are not in it.
    assert filesystem["readable"].startswith("{")
    assert filesystem["answers"] == "FileNotFoundError"


async def test_the_served_tree_has_no_path_that_leads_to_the_answers() -> None:
    """The route a reviewer actually walked: follow a link out of the served tree, then take the
    sibling. It worked, because `specs.json` was a symlink into the corpus and the corpus holds
    every task's answers next door.

    Nothing in the served tree is a symlink now, and the corpus it was derived from is not mounted,
    so a link would not resolve even if one were left. Each probe below is one step of that walk."""
    probe = """
_io, _os = __import__("io"), __import__("os")
root = _os.environ["APPWORLD_ROOT"]
task = root + "/data/tasks/%s"


def _read(path):
    try:
        return _io.open(path).read()[:40]
    except Exception as failure:
        return type(failure).__name__


def _target(path):
    try:
        return _os.readlink(path)
    except Exception as failure:
        return type(failure).__name__


print(json.dumps({
    "specs": _read(task + "/specs.json"),
    "specs_is_link": _os.path.islink(task + "/specs.json"),
    "specs_target": _target(task + "/specs.json"),
    "db_is_link": _os.path.islink(task + "/dbs/gmail.jsonl"),
    "sibling": _read(task + "/ground_truth/test_data.json"),
    "root_links": [n for n in ("base_dbs", "datasets", "api_docs")
                   if _os.path.islink(root + "/data/" + n)],
}))
""" % task_id()
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])

    # The served tree is genuinely readable from here, so the negatives below mean something.
    assert seen["specs"].startswith("{")
    # And nothing in it names where it came from.
    assert seen["specs_is_link"] is False
    assert seen["specs_target"] == "OSError"
    assert seen["db_is_link"] is False
    assert seen["root_links"] == []
    assert seen["sibling"] == "FileNotFoundError"


async def test_the_graders_tree_is_not_a_neighbour_of_the_served_one() -> None:
    """The independent route: the grader's view used to be a fixed name beside the served root, so
    knowing one gave you the other. It moved under a different parent with an unguessable name,
    which raised the cost and did not close it.

    **The route is not merely expensive now, it is not a route.** The probe used to substitute
    `graded` for `seeded` in a host path the worker could reach by following one of its view's
    shared links. Inside the container those links land on mount points: the corpus's shared parts
    are mounted at `/corpus/data/<name>` directly, so what a walk out of the served tree reaches is
    a name this container was given rather than a name on the machine. There is nothing to
    substitute into, and the tree itself is absent either way, which is asserted at length in
    :func:`test_the_run_the_grader_the_repository_and_the_corpus_are_absent`."""
    from shogym.envs.appworld import adapter as adapter_module

    served, graded = adapter_module.derived_root(), adapter_module.graded_root()
    # On the host the two are still far apart, which is what the unguessable name is for.
    assert graded.parent != served.parent
    assert str(served.parent) not in str(graded)
    probe = """
_os = __import__("os")
root = _os.environ["APPWORLD_ROOT"]
base = _os.path.dirname(_os.path.dirname(_os.path.realpath(root + "/data/base_dbs")))
print(json.dumps({
    "root": root,
    "base": base,
    "beside_the_view": _os.path.exists(_os.path.dirname(root) + "/graded"),
    "graded_by_name": _os.path.exists("/corpus/../graded"),
}))
"""
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    # The walk ends inside this container's own namespace rather than on the machine, so the
    # substitution has no host name to be made on.
    assert seen["root"] == "/corpus"
    assert seen["base"] == "/corpus"
    # The name the substitution needed is not in anything the world can reach: there is no
    # `seeded` to turn into `graded`, because the walk ends on a mount point.
    assert "seeded" not in seen["base"]
    assert seen["beside_the_view"] is False
    assert seen["graded_by_name"] is False


async def test_a_read_outside_the_served_tree_is_refused() -> None:
    """The acceptance test the audit hook was pulled in favour of, and it passes now.

    It was marked expected-to-fail and strict, on the reasoning that everything the served tree
    could have led to had been taken out of it and none of that stopped a read of a path an agent
    names outright: the code it wrote ran as the worker, with the worker's filesystem, as the same
    user as the run. A mount namespace is what makes this fail, and nothing short of one does. The
    worker is in one now, so the marker is gone with it.

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
    """The served inputs are per episode, and under the container they are not writable at all.

    The derived corpus was one deterministic global root and every worker was handed it, with its
    files writable by the process that runs agent-authored code and nothing putting them back. A
    write through episode A's served view was therefore still there in episode B's starting
    inputs. Two arms of a pair are the same task served at the same time, so the arm meant to
    differ only in what it was told could also differ in the world it was given, and that is a
    difference the treatment did not make.

    Two things close it and this checks both. The view is per episode, so one episode's writes
    have nowhere to reach the next from; and the mount is read-only, so there is no write to
    contain. The second is the stronger and is what the branch below said the container would
    bring.

    **The refusal is read off the mount and not off the attempt.** Upstream's own guard replaces
    `io.open` with one that refuses every write mode, so an attempt from inside `execute` fails
    whatever the mount says, and a test that only tried to write would pass on a container mounted
    read-write. So the kernel's own view of the mount is what is asserted, and the attempt is kept
    beside it as the thing an episode would actually do.

    Written through the pathname the worker is actually given, which is the route an episode has,
    rather than through a path this test worked out for itself."""
    task = task_id()
    marker = "written by an earlier episode"
    views = adapter.cache_root() / f"views-{adapter.DATA_VERSION}"
    # Snapshotted rather than assumed empty: what this checks is that the views *this test* makes
    # do not outlive their episodes, not that the machine started tidy.
    before = {entry.name for entry in views.iterdir()} if views.exists() else set()
    probe = (
        '_io, _os = __import__("io"), __import__("os")\n'
        'root = _os.environ["APPWORLD_ROOT"]\n'
        'target = root + "/data/tasks/%s/dbs/gmail.jsonl"\n'
        'mounts = _io.open("/proc/self/mountinfo").read()\n'
        'served = [line for line in mounts.splitlines() if "/corpus/data/tasks/" in line]\n'
    ) % task
    scribble = probe + (
        "\n"
        "def _write():\n"
        '    try:\n'
        '        _io.open(target, "w").write(%r)\n'
        '        return "wrote"\n'
        "    except Exception as failure:\n"
        "        return type(failure).__name__\n"
        "\n"
        '\n'
        'print(json.dumps({"root": root, "write": _write(), "served": served,'
        ' "body": _io.open(target).read()[:64]}))\n'
    ) % marker
    read_back = probe + (
        'print(json.dumps({"root": root, "body": _io.open(target).read()[:64]}))\n'
    )

    first = json.loads(json.loads((await play([scribble]))["outputs"][0]["content"])["output"])
    second = json.loads(json.loads((await play([read_back]))["outputs"][0]["content"])["output"])

    # The kernel's own word: the task tree is mounted read-only, which is the fact the attempt
    # below cannot establish on its own.
    assert first["served"], "no served task mount to read"
    for line in first["served"]:
        assert " ro," in line or line.endswith(" ro"), line
    # And the attempt an episode would make does not succeed either.
    assert first["write"] != "wrote", "the served tree was writable"
    assert first["body"] != marker
    assert second["body"] != marker, "the second episode started in the first one's leftovers"
    # Two episodes, two views on the host, and each removed with the episode that owned it. The
    # root inside the container is the same fixed mount point for both, which is the point: it
    # names nothing about which episode is behind it.
    assert first["root"] == second["root"] == "/corpus"
    after = {entry.name for entry in views.iterdir()} if views.exists() else set()
    assert after <= before, sorted(after - before)
    # And the pristine copies either side of the served view never saw it.
    pristine = adapter.derived_root() / "data" / "tasks" / task / "dbs"
    for entry in sorted(pristine.iterdir()):
        assert entry.read_text()[:64] != marker


async def test_the_world_stops_before_it_is_graded() -> None:
    """Sealing closes the tool surface and does not stop work an earlier call left running. The
    worker is terminated before the read is scored, so the evaluator reads a snapshot nothing can
    still be writing to."""
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        session = mcp_server.get_session(episode.session_id)
        assert session is not None
        worker = session.worker
        assert worker.process.poll() is None
        await episode.call("submit", {})
        # Graded, and the process that could have changed the world is already gone.
        assert worker.process.poll() is not None
    finally:
        await episode.close()


# ----- the matched pair, through a stream -----


async def test_the_world_has_no_network_to_be_reached_over() -> None:
    """The worker used to answer on a loopback port, and the token was the whole of what kept a
    second process on the machine from asking a live world to evaluate itself. There is no port
    now: the container is started with ``--network none`` and the parent talks to it over the pipe
    pair it created, which no other process can open.

    ``/proc/net/route`` is the kernel's own routing table for this container. Empty is what
    having no network looks like from inside, and it is the honest thing to assert: a fresh
    network namespace still lists the kernel's unconfigured tunnel pseudo-devices in
    ``/proc/net/dev``, so counting interfaces would be asserting a detail of Docker rather than
    the property. What matters is that there is no interface carrying traffic and nowhere to send
    it."""
    probe = """
_io, _socket = __import__("io"), __import__("socket")


def _read(path):
    try:
        return _io.open(path).read()
    except Exception as failure:
        return type(failure).__name__


def _reach(host, port):
    try:
        _socket.create_connection((host, port), timeout=3).close()
        return "connected"
    except Exception as failure:
        return type(failure).__name__


print(json.dumps({
    "routes": [line for line in _read("/proc/net/route").splitlines()[1:] if line.strip()],
    "interfaces": sorted(
        line.split(":")[0].strip()
        for line in _read("/proc/net/dev").splitlines()[2:]
        if ":" in line
    ),
    "outbound": _reach("1.1.1.1", 53),
    "host_loopback": _reach("127.0.0.1", 22),
}))
"""
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    assert seen["routes"] == [], seen["routes"]
    # And no ethernet interface: what Docker would have attached is exactly what is not there.
    assert not [name for name in seen["interfaces"] if name.startswith("eth")]
    assert seen["outbound"] != "connected"
    # And the container's own loopback is not the host's, so a service on the host is not one
    # step away either.
    assert seen["host_loopback"] != "connected"


async def test_the_container_name_and_the_served_root_reach_no_agent_visible_surface() -> None:
    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        from shogym.envs.appworld import mcp_server

        session = mcp_server.get_session(episode.session_id)
        assert session is not None
        # The container's name is the only handle a process on the host could use to `exec` into
        # a live world, and the served root is the one host path that would say where the cache is.
        secrets = (session.worker.container, str(session.worker.root))
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


async def test_the_run_the_grader_the_repository_and_the_corpus_are_absent(
    tmp_path: Path,
) -> None:
    """The whole of shogym#138 in one probe, and the word that matters is *absent*.

    Every path here was readable from a worker on the host, because the worker was the user
    running the run. None of them is a mount of the episode's container, so each one is not a
    file the world may not open: it is a file that is not there. That is the difference between a
    property of the arrangement and a property of the machine, and it is why each assertion below
    is ``FileNotFoundError`` rather than ``PermissionError``.

    The last of them is the expected-failure the port shipped with. An audit hook was supposed to
    record a read like this and did not record one made through a live ``execute``; the read now
    fails because the path does not exist, and the record catches it as well (see
    :func:`test_the_record_catches_a_read_made_through_a_live_execute`)."""
    from shogym.envs.appworld import adapter as adapter_module

    run_tree = tmp_path / "provenance"
    run_tree.mkdir()
    (run_tree / "results.jsonl").write_text('{"report": "the true report"}\n')

    absent = {
        "run_tree": str(run_tree / "results.jsonl"),
        "grader_root": str(adapter_module.graded_root() / "data" / "tasks" / task_id()),
        "grader_parent": str(adapter_module.private_home()),
        "corpus": str(CORPUS / "data" / "tasks" / task_id() / "ground_truth" / "test_data.json"),
        "cache": str(adapter_module.cache_root()),
        "repository": str(Path(shogym.__file__).parent / "envs" / "appworld" / "scorer.py"),
        "another_episode": str(adapter_module.derived_root() / "data" / "tasks"),
        "home": str(Path.home() / ".ssh"),
    }
    probe = """
_io, _os = __import__("io"), __import__("os")
paths = json.loads(%r)


def _read(path):
    try:
        return _io.open(path).read()[:40]
    except Exception as failure:
        return type(failure).__name__


print(json.dumps({
    "read": {name: _read(path) for name, path in paths.items()},
    "exists": {name: _os.path.exists(path) for name, path in paths.items()},
}))
""" % json.dumps(absent)
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    for name in absent:
        assert seen["exists"][name] is False, (name, seen["exists"][name])
        assert seen["read"][name] in ("FileNotFoundError", "NotADirectoryError"), (
            name,
            seen["read"][name],
        )
    # Every one of those files really is there, on the host, for the process running this test.
    # The assertions above are about a different filesystem, not about a tidy machine.
    assert (run_tree / "results.jsonl").exists()
    assert adapter_module.private_home().exists()
    assert (CORPUS / "data" / "tasks" / task_id() / "ground_truth").exists()


async def test_what_the_world_can_learn_about_the_host_is_names_and_not_contents() -> None:
    """The claim, narrowed to what is true, and pinned by a test rather than by prose.

    `/proc/self/mountinfo` is readable from inside, and it names the host-side source of every
    bind: the cache root, the private home and its tag, and this episode's own output directory.
    None of those is openable from here, none of them is the Docker socket, and none of them is a
    key, a pulse or a grade. So the boundary is about *contents*, not about *names*, and the port
    says so rather than implying that a path an agent cannot use is a path it cannot see.

    What matters more than the disclosure is that none of those names encodes the feedback arm.
    Two arms of one pair differ in their session identifiers and in nothing else a mount is named
    after, so a world cannot read its own treatment off its filesystem."""
    probe = """
_io, _os = __import__("io"), __import__("os")
mounts = _io.open("/proc/self/mountinfo").read()
print(json.dumps({
    "mountinfo": [line.split(" - ")[0][-160:] for line in mounts.splitlines()
                  if "/corpus" in line or "/outputs" in line],
    "hostname": _os.environ.get("HOSTNAME", ""),
    "socket": _os.path.exists("/var/run/docker.sock"),
}))
"""
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    disclosed = " ".join(seen["mountinfo"])
    # The disclosure is real and is what the port documents: host-side names are visible.
    assert "/corpus" in disclosed and "/outputs" in disclosed
    # What it is not: a Docker socket, or a name that says which arm this episode is.
    assert seen["socket"] is False
    for arm in ("information", "placebo", "graded", "drawn", "report", "notice"):
        assert arm not in disclosed.lower(), arm
    # And the container's own short id is not handed out in the environment: the hostname is a
    # constant, so `HOSTNAME` says nothing about which container this is.
    assert seen["hostname"] in ("", "worker")


async def test_the_machine_a_world_reads_about_is_not_the_host(tmp_path: Path) -> None:
    """A container's `/proc` is mostly the machine's, and this is where that stopped being true.

    The kernel virtualizes the process tree per namespace and virtualizes almost nothing else, so
    `/proc/cpuinfo`, `/proc/meminfo`, `/proc/uptime`, `/proc/stat` and `/proc/loadavg` were the
    host's own, readable from ordinary `execute` output. None of that is ground truth, a grade, a
    pulse or an arm label, and both arms on one host read the same numbers; what it is, is a
    description of the machine rather than of the world, and a pair split across two machines
    would read two descriptions of two machines under one identity.

    Fixed files are mounted over the ones the runtime will let a bind cover. What it will not cover
    is documented in the port's README rather than implied to be covered: an arbitrary path inside
    `/proc` is refused (`cannot be mounted because it is inside /proc`), so the boot identifier,
    the kernel version string and the processor count as `sched_getaffinity` reports it are still
    the machine's.

    Both arms are read and compared rather than the values being asserted, because what the pair
    needs is that they are the same, and the constants are the port's to change."""
    from shogym.serve.stream import Information, Placebo, TaskRef, TaskStream

    probe = """
_io = __import__("io")
def _read(path):
    try:
        return _io.open(path).read()
    except Exception as exc:
        return "unreadable: %s" % type(exc).__name__
print(json.dumps({
    "cpuinfo": _read("/proc/cpuinfo"),
    "meminfo": _read("/proc/meminfo"),
    "uptime": _read("/proc/uptime"),
    "stat": _read("/proc/stat").splitlines()[:1],
    "loadavg": _read("/proc/loadavg"),
}))
"""
    seen = {}
    identity = shogym.make("appworld").config_digest
    for name, policy in (("information", Information()), ("placebo", Placebo())):
        stream = TaskStream(
            shogym.make,
            [TaskRef("appworld", TASK)],
            prov_dir=tmp_path / name,
            feedback=policy,
            identity=identity,
        )
        async with stream:
            await stream.get_task()
            answered = await stream.dispatch("execute", {"code": probe})
            answer = json.loads(json.loads(answered.content[0].text)["content"])
            seen[name] = json.loads(answer["output"])
            await stream.dispatch("submit", {})

    # Byte for byte the same in both arms, which is the paired claim, and none of it is this
    # machine's, which is the boundary one.
    assert seen["information"] == seen["placebo"]
    read = seen["information"]
    assert "model name\t: neutral" in read["cpuinfo"]
    assert read["meminfo"].startswith("MemTotal:        1048576 kB")
    assert read["uptime"].split() == ["0.00", "0.00"]
    assert read["loadavg"].split() == ["0.00", "0.00", "0.00", "1/1", "1"]
    assert read["stat"] == ["cpu  0 0 0 0 0 0 0 0 0 0"]
    # Well formed rather than blank, because the world's own dependencies parse them: a truncated
    # one would be an exception on import rather than a masked fact.


async def test_the_two_arms_see_the_same_surface_from_inside_execute(tmp_path: Path) -> None:
    """The paired claim, compared rather than asserted.

    Two arms of a pair differ in what a terminating call hands back and in nothing else. The
    surface an episode can read from inside `execute` is where that could quietly stop being true:
    a mount named for its arm, an environment variable that differs, a filesystem that says which
    policy is serving. So both arms are run and their surfaces are compared, with the parts that
    are per episode by design masked out, rather than the words of the arms being grepped for."""
    from shogym.serve.stream import Information, Placebo, TaskRef, TaskStream

    probe = """
_io, _os = __import__("io"), __import__("os")
mounts = _io.open("/proc/self/mountinfo").read()
print(json.dumps({
    "mounts": sorted(line.split(" - ")[0].split(" ", 4)[-1] for line in mounts.splitlines()
                     if "/corpus" in line or "/outputs" in line),
    "env": sorted("%s=%s" % (k, v) for k, v in _os.environ.items()),
    "root": sorted(n for n in ("corpus", "outputs", "scratch", "opt", "tmp")
                   if _os.path.exists("/" + n)),
    "socket": _os.path.exists("/var/run/docker.sock"),
}))
"""
    seen = {}
    identity = shogym.make("appworld").config_digest
    for name, policy in (("information", Information()), ("placebo", Placebo())):
        stream = TaskStream(
            shogym.make,
            [TaskRef("appworld", TASK)],
            prov_dir=tmp_path / name,
            feedback=policy,
            identity=identity,
        )
        async with stream:
            await stream.get_task()
            answered = await stream.dispatch("execute", {"code": probe})
            answer = json.loads(json.loads(answered.content[0].text)["content"])
            seen[name] = json.loads(answer["output"])
            await stream.dispatch("submit", {})

    def _masked(surface):
        # The session identifier is per episode by design and is in the output mount's host path.
        # Everything else has to match, including every environment variable.
        return {
            "mounts": sorted(re.sub(r"[0-9a-f-]{20,}", "<episode>", line)
                             for line in surface["mounts"]),
            "env": surface["env"],
            "root": surface["root"],
            "socket": surface["socket"],
        }

    assert _masked(seen["information"]) == _masked(seen["placebo"])
    # And the masking did not hide everything: the surfaces really do name the mounts.
    assert any("/corpus" in line for line in seen["information"]["mounts"])
    # Neither arm's own word is anywhere in what either of them can read. The surfaces only: the
    # keys of this dictionary are the test's own names for them.
    both = json.dumps(list(seen.values()))
    for word in ("information", "placebo", "graded", "drawn", "report", "notice"):
        assert word not in both.lower(), word


async def test_only_this_episodes_task_and_output_tree_are_mounted() -> None:
    """One task, not the roster; one experiment's outputs, not the run's.

    The served corpus holds 318 tasks and a run's output tree holds one directory per episode.
    Mounting either wholesale would put a sibling episode's world, and every other task's derived
    tree, one ``listdir`` away. The mount set is per episode instead, so ``data/tasks`` inside the
    container holds exactly one entry."""
    from shogym.envs.appworld import adapter as adapter_module

    # Served here rather than assumed. Selecting this test alone against a fresh cache used to
    # raise `FileNotFoundError` on a tasks directory nothing had created yet, and it passed in
    # full-file order only because an earlier test happened to derive a sibling first. A boundary
    # test that depends on collection order is a boundary test that can go quiet.
    await play([], position=TASK)
    await play([], position=TASK + 1)
    derived = adapter_module.derived_root()
    others = [
        entry.name
        for entry in sorted((derived / "data" / "tasks").iterdir())
        if entry.is_dir() and entry.name != task_id()
    ]
    probe = """
_os = __import__("os")
root = _os.environ["APPWORLD_ROOT"]
print(json.dumps({
    "mine": _os.path.exists(root + "/data/tasks/%s/specs.json"),
    "others": [n for n in json.loads(%r)
               if _os.path.exists(root + "/data/tasks/" + n)],
}))
""" % (task_id(), json.dumps(others))
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    assert seen["mine"] is True
    assert seen["others"] == []
    # And the test is not vacuous: the host really does hold another task's derived tree, because
    # this test served one.
    assert task_id(TASK + 1) in others


async def test_activity_an_earlier_block_started_cannot_change_the_graded_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The scored state and the graded state have to be one state, not two that agreed.

    A block can start a thread, and the thread outlives the block: AppWorld runs an agent's code
    under an alarm on the main thread and does nothing about what that code started. The filing
    and the world's digest used to be read off the live world in a `read` command, with the
    container removed only after the answer came back, so a thread still writing during the read,
    or between the digest and the removal, changed the bytes a grader opened afterwards. Nothing
    downstream would have noticed: the receipt would describe one state and the base checks
    another.

    The order is quiesce, seal, remove, read now, and the last of those happens in a container the
    world cannot reach. The window this watches is the one that used to be open: the tree is
    digested on the host on either side of the grading container's own run, and both have to equal
    what the run recorded."""
    from shogym.envs.appworld import adapter as adapter_module
    from shogym.envs.appworld import mcp_server
    from shogym.envs.appworld.worker import _directory_digest

    spinner = """
_t = __import__("threading")
state = {"writes": 0}
pw = [x for x in apis.supervisor.show_account_passwords() if x["account_name"] == "todoist"][0]
tok = apis.todoist.login(
    username=apis.supervisor.show_profile()["email"], password=pw["password"]
)["access_token"]
proj = [
    p for p in apis.todoist.show_projects(access_token=tok, page_limit=50)
    if p["name"] == "Task Log"
][0]["project_id"]


def _write():
    while state["writes"] < 400:
        try:
            apis.todoist.create_task(
                access_token=tok, project_id=proj,
                title="Filing", description="RQ-0001, Routine",
            )
        except Exception:
            pass
        state["writes"] += 1


_t.Thread(target=_write, daemon=True).start()
print("started")
"""
    watched: Dict[str, str] = {}
    real = adapter_module.grade

    def _watching(**kwargs: Any) -> Any:
        # Both sides of the grading container's own run, on the host, so this sees the tree the
        # grader read rather than a report about it. The episode's output tree is removed at
        # teardown, which is why the reading happens here rather than after the call returns.
        dbs = str(kwargs["outputs"] / "tasks" / kwargs["task_id"] / "dbs")
        watched["before"] = _directory_digest(dbs)
        answer = real(**kwargs)
        watched["after"] = _directory_digest(dbs)
        return answer

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    monkeypatch.setattr(adapter_module, "grade", _watching)
    try:
        await episode.call("execute", {"code": spinner})
        # The thread is demonstrably writing into the world while the episode is still open, so
        # what follows is an assertion about a moving target rather than a quiet one.
        seen = await episode.call("execute", {"code": 'print(state["writes"])'})
        assert int(json.loads(seen.content)["output"].strip()) > 0
        terminal = await episode.call("submit", {})
    finally:
        monkeypatch.undo()
        await episode.close()
    feedback = {
        item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
    }
    # Nothing moved across the grading window, and what the run recorded is that same tree.
    assert watched["before"] == watched["after"]
    assert feedback["world_digest"] == watched["before"]


async def test_a_removal_the_daemon_did_not_confirm_fails_the_episode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Failing closed, because the alternative is a grade with nothing behind it.

    Removal used to swallow every failure. The daemon owns the container, so a `docker rm -f` that
    returned nonzero, or timed out, left a world that might still be writing to the output tree
    finalization was about to mount into the grader. That is exactly the invariant the ordering
    exists to establish, so an unconfirmed removal has to end the episode rather than grade it,
    and it has to leave the worker unclosed so teardown can try again."""
    from shogym.envs.appworld import container as container_module
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    worker = session.worker
    # The container really is removed; what is simulated is a daemon that will not say so.
    monkeypatch.setattr(container_module, "absent", lambda name: False)
    try:
        terminal = await episode.call("submit", {})
        feedback = {
            item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
        }
        # Not graded, and not even zeroed: a failed terminal publishes the failure and nothing
        # else, so there is no fraction to average and no receipt for a policy to select.
        assert feedback.get("finalize_error") is True
        assert "ledger_fraction" not in feedback
        assert "report" not in feedback and "notice" not in feedback
    finally:
        monkeypatch.undo()
        await episode.close()
    # Teardown, which is best effort rather than confirming, still cleaned up after the failure.
    # That the confirming close leaves the worker retryable is checked where it can be seen
    # without teardown racing the assertion: `test_a_close_that_cannot_confirm_stays_retryable`.
    assert container_module.absent(worker.container)


async def test_the_container_is_gone_when_the_episode_is() -> None:
    """Teardown removes the container, and does it on the failing paths too.

    Killing the ``docker run`` client does not stop a container: the daemon owns it. So close
    removes it by name, and it has to still do that when the world it was holding is wedged and
    when the episode never got as far as opening one."""
    from shogym.envs.appworld import container as container_module
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    name = session.worker.container
    assert container_module.running(name)
    await episode.call("submit", {})
    await episode.close()
    assert not container_module.running(name)

    # And a worker whose caller crashes between spawn and close: `close` is what teardown calls
    # from its `finally`, and it is the same call.
    worker = adapter.Worker.spawn(
        adapter.derived_root(),
        task_id=task_id(),
        outputs=adapter.episode_outputs("teardown-probe"),
    )
    assert container_module.running(worker.container)
    worker.close()
    assert not container_module.running(worker.container)


async def test_the_worlds_container_is_gone_before_the_grader_starts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The ordering, exercised at the container rather than at the process.

    `read` flushes the end state into the episode's output directory, which is the one directory
    the world may write and the one the grading container reads. A world still running while that
    happens is agent-authored code with a live handle on the bytes it is about to be scored on: a
    thread left behind by an earlier block could rewrite the flushed state after the flush and
    before the grader opened it, and no assertion downstream would notice.

    :func:`test_the_world_stops_before_it_is_graded` asserts the worker process is gone. This
    asserts the container is, which is the stronger of the two and the one that holds when agent
    code has started processes of its own: removing a container ends everything in its namespace,
    where signalling a process group ends everything that stayed in the group."""
    from shogym.envs.appworld import adapter as adapter_module
    from shogym.envs.appworld import container as container_module
    from shogym.envs.appworld import mcp_server

    alive: List[bool] = []
    real = adapter_module.grade

    def _watching(**kwargs: Any) -> Any:
        alive.append(any(container_module.running(name) for name in names))
        return real(**kwargs)

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    names = [session.worker.container]
    monkeypatch.setattr(adapter_module, "grade", _watching)
    try:
        await episode.call("execute", {"code": filing_block()})
        terminal = await episode.call("submit", {})
    finally:
        await episode.close()
    # The grader ran, and the world's container was already gone when it did.
    assert alive == [False], alive
    feedback = {
        item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
    }
    # And stopping the world first did not cost the episode its grade: the base task's own checks
    # are still collected, from the state the world flushed before it was stopped.
    assert feedback["checks"] > 0
    assert feedback["ledger_fraction"] == 1.0


async def test_a_failed_setup_leaves_no_tree_behind() -> None:
    """A spawn or an open that fails owns what it made until something else does.

    The view and the output tree are created before the session is registered, so a failure
    between the two left both on disk with nothing holding them: the env's own close finds no
    session, and neither is ever removed. On a paired run that is two directories per failed
    episode, for the life of the machine."""
    from shogym.envs.appworld import adapter as adapter_module

    env = shogym.make("appworld")
    views = adapter_module.cache_root() / f"views-{adapter_module.DATA_VERSION}"
    outputs = adapter_module.episodes_home()
    before = (
        {entry.name for entry in views.iterdir()} if views.exists() else set(),
        {entry.name for entry in outputs.iterdir()} if outputs.exists() else set(),
    )
    session_id = "failed-setup-probe"
    task = env._load_task(TASK)
    # A world that cannot open: the task the env was told to serve is not one the corpus has.
    with pytest.raises(Exception):
        env._begin_session(session_id, {**task, "task_id": "no_such_task_1"})
    after = (
        {entry.name for entry in views.iterdir()} if views.exists() else set(),
        {entry.name for entry in outputs.iterdir()} if outputs.exists() else set(),
    )
    assert after[0] <= before[0], sorted(after[0] - before[0])
    assert after[1] <= before[1], sorted(after[1] - before[1])


async def test_teardown_never_raises_when_docker_will_not_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Teardown's close is best effort by contract, and the contract has to hold on the paths that
    actually fail.

    A control call that times out or finds no CLI used to raise straight out of teardown, past the
    pipes and the directories it was there to release. The container it could not remove belongs
    to the sweep; the handles belong here, and dropping them is not optional."""
    from shogym.envs.appworld import container as container_module
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    worker, view, outputs = session.worker, Path(session.view), session.outputs
    real = container_module.remove

    def _refuses(name: str, *, confirm: bool = False) -> None:
        raise container_module.DockerError("the docker CLI is not on PATH")

    monkeypatch.setattr(container_module, "remove", _refuses)
    try:
        await episode.close()
    finally:
        monkeypatch.setattr(container_module, "remove", real)
    # The episode closed, the directories are gone, and the pipes are shut, even though the
    # removal could not be made.
    assert not view.exists()
    assert not outputs.exists()
    assert worker.process.poll() is not None
    real(worker.container)


async def test_a_runaway_block_does_not_slow_its_sibling_arm() -> None:
    """Two arms of a pair are two containers on one host, so one has to be bounded against the
    other.

    A pid limit bounds a fork bomb and nothing else: a block that spins takes whatever the machine
    will give it. The quota is what keeps the arm that was supposed to differ only in what it was
    told from also differing in how much machine it got."""
    import time

    burn = """
_t = __import__("threading")
state = {"on": True}


def _spin():
    while state["on"]:
        sum(i * i for i in range(100000))


for _ in range(8):
    _t.Thread(target=_spin, daemon=True).start()
print("burning")
"""
    quiet = shogym.make("appworld")
    busy = shogym.make("appworld")
    calm = await ServedEpisode.open_env(quiet, env_name="appworld", task=TASK)
    loud = await ServedEpisode.open_env(busy, env_name="appworld", task=TASK)
    try:
        # A baseline from the quiet arm before anything is burning.
        start = time.monotonic()
        await calm.call("execute", {"code": "print(1)"})
        alone = time.monotonic() - start
        await loud.call("execute", {"code": burn})
        start = time.monotonic()
        await calm.call("execute", {"code": "print(2)"})
        beside = time.monotonic() - start
    finally:
        await loud.close()
        await calm.close()
    # **A ceiling, not a reservation.** `--cpus` is a CFS bound on how much a container may use,
    # not a set of cpus reserved for it: two arms on one host share the same processors, and on a
    # loaded machine each still slows the other. What the quota buys is that neither can take more
    # than its share, so what this asserts is the absence of starvation rather than independence.
    # Arms that must not influence each other at all need disjoint cpusets or separate hosts, and
    # the README says so.
    assert beside < max(1.0, alone * 8 + 0.5), (alone, beside)


async def test_several_worlds_alive_at_once_stay_several_worlds() -> None:
    """One env per episode, which is what `open_env` asks for, and one closed while the rest run.

    Two worlds in one interpreter are one world the other keeps unfreezing, and two episodes
    sharing a container would be the same failure with a bigger boundary around it. This shares
    nothing: each episode gets its own env, as a production run does, and the interesting moment
    is the one the previous version of this test skipped by closing them together. A sibling has
    to survive its neighbour's whole teardown, which now stops a container, confirms the stop with
    the daemon, snapshots a tree and grades it."""
    from shogym.envs.appworld import container as container_module
    from shogym.envs.appworld import mcp_server

    positions = (0, 1, 0)
    envs = [shogym.make("appworld") for _ in positions]
    episodes = await asyncio.gather(
        *(
            ServedEpisode.open_env(env, env_name="appworld", task=position)
            for env, position in zip(envs, positions)
        )
    )
    try:
        names = []
        for episode in episodes:
            session = mcp_server.get_session(episode.session_id)
            assert session is not None
            names.append(session.worker.container)
        assert len(set(names)) == 3
        assert all(container_module.running(name) for name in names)
        # Concurrent, not merely alive at the same time: each world answers its own question
        # while the others are open.
        answers = await asyncio.gather(
            *(
                episode.call("execute", {"code": "print(%d)" % index})
                for index, episode in enumerate(episodes)
            )
        )
        assert [
            json.loads(answer.content)["output"].strip() for answer in answers
        ] == ["0", "1", "2"]
        # Two episodes of one task are two worlds, not one shared one.
        await episodes[0].call("execute", {"code": "marker = 'first'"})
        third = await episodes[2].call("execute", {"code": "print(globals().get('marker'))"})
        assert json.loads(third.content)["output"].strip() == "None"

        # One goes through the whole of a production teardown while the others are mid-episode.
        await episodes[1].call("submit", {})
        await episodes[1].close()
        assert not container_module.running(names[1])
        # And a survivor is still a working world afterwards, not merely a live container.
        assert all(container_module.running(name) for name in (names[0], names[2]))
        after = await episodes[0].call("execute", {"code": "print(marker)"})
        assert json.loads(after.content)["output"].strip() == "first"
        scored = await episodes[2].call("submit", {})
        assert json.loads(scored.content)["checks"] > 0
    finally:
        await asyncio.gather(*(episode.close() for episode in episodes))
    assert not any(container_module.running(name) for name in names)


async def test_the_terminal_row_reaches_the_trace_after_an_ordinary_execute(
    tmp_path: Path,
) -> None:
    """The run fingerprint rides on the row, and a row the trace store refuses is a row nobody has.

    The fingerprint is published as inference feedback, which the store requires to carry the step
    of the row it is on. It carried a fixed zero, so on any terminal row past the first step the
    store refused the record; the refusal was caught, flagged as degraded persistence, and the
    call returned success. What was left was a trace with no terminal row in it: a later read
    reported an episode that never ended, with no feedback and no identity, and nothing anywhere
    said so.

    So this drives an episode that takes a step before it ends, which is the case that failed, and
    reads the trace back."""
    from shogym.trace import load_traces

    trace = tmp_path / "episode.jsonl"
    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(
        env, env_name="appworld", task=TASK, trace_path=trace
    )
    try:
        await episode.call("execute", {"code": filing_block()})
        await episode.call("submit", {})
    finally:
        await episode.close()

    rows = load_traces(trace)
    terminal = [row for row in rows if row.get("terminated")]
    assert terminal, "the terminal row was refused and the failure was swallowed"
    last = terminal[-1]
    names = {item["name"] for item in last["feedback"]}
    assert "ledger_fraction" in names
    # The row is past the first step, which is the case that failed.
    assert last["step"] >= 1
    # The identity is on the row, at the row's own step, which is what the store was refusing.
    identity = [item for item in last["feedback"] if item["name"] == "config_digest"]
    assert identity and identity[0]["step"] == last["step"]
    assert identity[0]["value"] == env.config_digest


# ----- the matched pair, through a stream -----


async def test_the_identity_a_row_is_filed_under_carries_what_this_env_said_it_was(
    tmp_path: Path,
) -> None:
    """The env's half of the identity is the env's own, and the caller's half is never parsed.

    The check used to be containment: this env's digest had to occur somewhere inside the caller's
    string, which is not a comparison of anything. A name with unrelated text around a digest
    passed and was stamped on a scored row, and a name composing several fields could match by
    accident. What a record is filed under is a record now: the caller's opaque name, and beside
    it what each env in the queue said about itself, read off the env at construction under the
    item the env declares (`identity_feedback_name`, which this env sets and answers to).

    So the value reaches the ownership claim before the first task is dispensed rather than
    waiting for a row to publish one, and a caller may call itself whatever it likes without the
    record losing track of which configuration produced its rows. What a resume is held to is that
    member, which is the test below."""
    from shogym.serve.stream import Information, TaskRef, TaskStream

    stream = TaskStream(
        shogym.make,
        [TaskRef("appworld", TASK)],
        prov_dir=tmp_path / "named",
        feedback=Information(),
        identity="a name this env never produced",
    )
    async with stream:
        # Read while the claim is held, and before any row exists: this is the window a run killed
        # early leaves behind, and it used to hold nothing but the caller's own string.
        claim = json.loads((tmp_path / "named" / "claim.json").read_text())
        await stream.get_task()
        await stream.dispatch("submit", {})

    identity = claim["run_identity"]
    assert identity["caller"] == "a name this env never produced"
    published = [
        item["value"]
        for row in stream.results
        for item in row.observed
        if item.get("name") == "config_digest"
    ]
    # What the claim recorded before the first task is what the env went on to publish on the row.
    assert identity["envs"] == {"appworld": published[0]}
    # And the episode scored: an env that says what it is does not need its caller to repeat it.
    assert [row.score is not None for row in stream.results] == [True]


async def test_a_resume_is_refused_under_a_changed_draw_and_under_a_changed_deadline(
    tmp_path: Path,
) -> None:
    """Two things decide whether rows belong to one record, and only one of them was checked.

    The draw, the payload class and the corpus decide what a score *means*, and the env says so.
    The deadline and the capacity decide what an episode was allowed to do: a deadline turns a
    slow episode into a timeout rather than a score, and a capacity changes the tool surface and
    the scheduling. Two directories that differ in either hold rows about two different
    opportunities, so both are in what a resume is checked against."""
    from shogym.serve.stream import Information, TaskRef, TaskStream

    prov = tmp_path / "run"
    first = shogym.make("appworld")
    # The opportunity is part of what the caller names, not something the stream appends: a
    # deadline decides whether a slow episode is scored or timed out, and a capacity decides the
    # tool surface, so two directories that differ in either hold rows about two different
    # opportunities. Composing it here keeps `identity` a string the stream compares and never
    # reads, which is what stops it inferring a format it cannot verify.
    stream = TaskStream(
        shogym.make,
        [TaskRef("appworld", TASK)],
        prov_dir=prov,
        feedback=Information(),
        identity=f"{first.config_digest}|deadline=600.0",
        deadline=600.0,
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch("submit", {})
    assert len(stream.results) == 1

    # A different draw. The env says so itself, and its digest moves with it.
    other = shogym.make("appworld", config={"pulse": 7})
    assert other.config_digest != first.config_digest
    with pytest.raises(ValueError) as changed_draw:
        TaskStream(
            shogym.make,
            [TaskRef("appworld", TASK)],
            prov_dir=prov,
            feedback=Information(),
            identity=f"{other.config_digest}|deadline=600.0",
            deadline=600.0,
            resume=True,
        )
    assert "run identity" in str(changed_draw.value)

    # The same env, a different deadline. Nothing about the score changed; what an episode was
    # allowed to do did.
    with pytest.raises(ValueError) as changed_deadline:
        TaskStream(
            shogym.make,
            [TaskRef("appworld", TASK)],
            prov_dir=prov,
            feedback=Information(),
            identity=f"{first.config_digest}|deadline=30.0",
            deadline=30.0,
            resume=True,
        )
    assert "run identity" in str(changed_deadline.value)

    # And the unchanged one resumes, so the refusals above are about the change rather than about
    # resuming at all.
    TaskStream(
        shogym.make,
        [TaskRef("appworld", TASK)],
        prov_dir=prov,
        feedback=Information(),
        identity=f"{first.config_digest}|deadline=600.0",
        deadline=600.0,
        resume=True,
    )


async def test_a_forged_completion_neither_earns_a_block_nor_moves_the_grade() -> None:
    """The reply an episode might forge, what it would buy, and what actually stops it today.

    The protocol's writer stays open in the interpreter that runs agent-authored code and nothing
    can close it, so a frame carrying the next request's identifier would reach the parent before
    the real handler finished. Two things mean that buys nothing. The budget is spent when a
    request goes out rather than when an answer comes back, so a forged answer consumes a block
    instead of granting one; and what is graded is what upstream persisted at the end of a block
    that actually ran, so a block whose answer was forged wrote nothing and the state graded is
    the state before it.

    **And the route is closed as well, by upstream rather than by this port.** Every primitive
    that could put bytes on that descriptor is null-patched by AppWorld's own guard: `os.write`
    returns `None` and writes nothing, `io.open` refuses write modes. That is evidence and not a
    boundary, which is why the two properties above are what the design rests on: the guard lets
    `__import__("sys")` through and reads whatever it likes, so it is not a thing to build on."""
    probe = """
_io, _os = __import__("io"), __import__("os")
pipes = []
for fd in range(3, 24):
    try:
        if _os.readlink("/proc/self/fd/%d" % fd).startswith("pipe:"):
            pipes.append(fd)
    except Exception:
        pass
body = b'{"id": 3, "output": {"output": "forged"}}'


def _try(what):
    try:
        return {"wrote": what()}
    except Exception as failure:
        return {"raised": type(failure).__name__}


routes = {}
for fd in pipes:
    routes["os.write.%d" % fd] = _try(lambda fd=fd: _os.write(fd, body))
    routes["io.open.%d" % fd] = _try(
        lambda fd=fd: _io.open("/proc/self/fd/%d" % fd, "wb").write(body)
    )
    routes["fdopen.%d" % fd] = _try(lambda fd=fd: _os.fdopen(_os.dup(fd), "wb").write(body))
print(json.dumps({"pipes": pipes, "routes": routes, "write": str(_os.write)}))
"""
    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    try:
        first = await episode.call("execute", {"code": probe})
        seen = json.loads(json.loads(first.content)["output"])
        # The descriptors really are there, which is what the port concedes and does not hide.
        assert seen["pipes"], seen
        # And no route from inside puts a byte on one: upstream's guard null-patches the write
        # primitives, so every attempt either wrote nothing or raised.
        assert "disabled" in seen["write"], seen["write"]
        for name, outcome in seen["routes"].items():
            assert outcome.get("wrote") in (None, 0), (name, outcome)
        # The block that would file a perfect ledger, unaffected by any of it.
        second = await episode.call("execute", {"code": filing_block()})
        spent = json.loads(second.content)
        terminal = await episode.call("submit", {})
    finally:
        await episode.close()
    feedback = {
        item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
    }
    # Two requests went out, so two blocks are spent, whatever any answer said.
    assert spent["calls"] == 2
    assert feedback["ledger_fraction"] == 1.0


async def test_a_timed_out_call_leaves_the_grade_refused_rather_than_taken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The timeout path used to mark the container absent without asking, and finalization's own
    gate then returned early: the tree being graded could still have had the timed-out command
    writing into it. Unusable and absent are two facts, and only the second may open a grade."""
    from shogym.envs.appworld import container as container_module
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    worker = session.worker
    monkeypatch.setattr(adapter, "_CALL_TIMEOUT_SECONDS", 0.5)
    # A daemon that will not confirm the removal, from the moment the timeout tries it.
    monkeypatch.setattr(container_module, "absent", lambda name: False)
    try:
        # A block that runs long enough to outlast the call timeout. `time.sleep` is null-patched
        # by upstream's guard, so the work has to be work.
        with contextlib.suppress(Exception):
            await episode.call(
                "execute", {"code": "print(sum(i * i for i in range(400000000)))"}
            )
        assert worker.poisoned
        # Unusable, and not claimed absent: the gate before grading still has to ask.
        assert worker.closed is False
        terminal = await episode.call("submit", {})
        feedback = {
            item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
        }
        # Refused, not graded: finalization's gate asked and did not get an answer, and a failed
        # terminal publishes that fact alone.
        assert feedback.get("finalize_error") is True
        assert "checks" not in feedback
        assert "report" not in feedback and "notice" not in feedback
    finally:
        monkeypatch.undo()
        await episode.close()
    container_module.remove(worker.container)


async def test_an_interrupted_world_is_not_graded_even_when_it_stopped_cleanly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Confirmed absence proves writing stopped, not that it finished.

    Upstream ends every command with its own save into the tree about to be graded, and that saver
    clears its destination and writes several pieces in sequence. A timeout kills the command, and
    the removal that follows can succeed perfectly: what is left is a tree that is stable and
    partial, and grading it is scoring an episode on half a save. The earlier version of this test
    forced the removal to fail, which is the other branch; this one lets cleanup do exactly what
    it is supposed to."""
    from shogym.envs.appworld import container as container_module
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    worker = session.worker
    monkeypatch.setattr(adapter, "_CALL_TIMEOUT_SECONDS", 0.5)
    try:
        with contextlib.suppress(Exception):
            await episode.call(
                "execute", {"code": "print(sum(i * i for i in range(400000000)))"}
            )
        # The removal worked: this is the ordinary cleanup path, not the refusing one.
        assert worker.poisoned
        assert worker.closed is True
        assert container_module.absent(worker.container)
        terminal = await episode.call("submit", {})
        feedback = {
            item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
        }
        # And still not graded, because what it persisted may be half of a save.
        assert feedback.get("finalize_error") is True
        assert "checks" not in feedback
        assert "report" not in feedback and "notice" not in feedback
    finally:
        monkeypatch.undo()
        await episode.close()


async def test_a_terminal_that_overtakes_a_block_does_not_grade_the_interrupted_save() -> None:
    """The race the serve layer creates on purpose, and what finalization does about it.

    A terminal may overtake an ordinary call: a deadline has to be able to end an episode whose
    block is not coming back. What must not follow is removing the container while upstream is
    inside the save it ends every block with, because that leaves a tree that is stable and
    partial and a grade taken over it is a grade of half a save.

    So finalization waits for the accepted call, bounded, and a world that will not settle is
    refused rather than stopped underneath. This submits while a block is still running."""
    import asyncio as _asyncio

    from shogym.envs.appworld import env_v1
    from shogym.envs.appworld import mcp_server

    env = shogym.make("appworld")
    episode = await ServedEpisode.open_env(env, env_name="appworld", task=TASK)
    session = mcp_server.get_session(episode.session_id)
    assert session is not None
    original = env_v1._SETTLE_SECONDS
    env_v1._SETTLE_SECONDS = 1.0
    slow = "print(sum(i * i for i in range(400000000)))"
    try:
        block = _asyncio.create_task(episode.call("execute", {"code": slow}))
        await _asyncio.sleep(1.0)
        terminal = await episode.call("submit", {})
        block.cancel()
        with contextlib.suppress(BaseException):
            await block
        feedback = {
            item["name"]: item["value"] for item in (terminal.meta.get("shogym/feedback") or [])
        }
    finally:
        env_v1._SETTLE_SECONDS = original
        await episode.close()
    # The block was still running, so the episode is refused rather than scored over whatever the
    # stop interrupted.
    assert feedback.get("finalize_error") is True
    assert "checks" not in feedback
    assert "report" not in feedback and "notice" not in feedback


async def test_the_resolver_a_world_reads_says_nothing_about_the_host() -> None:
    """Docker writes one from the host's resolver configuration even with no network at all.

    The file it wrote named a nameserver and said it was based on the host's. There is nothing to
    resolve in a container with no network, so what that file holds is host metadata and nothing
    else; a fixed one is mounted over it."""
    probe = """
_io = __import__("io")
print(json.dumps({"resolv": _io.open("/etc/resolv.conf").read()}))
"""
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    assert "no resolver" in seen["resolv"]
    assert "nameserver" not in seen["resolv"]
    assert "host" not in seen["resolv"].lower().replace("this container", "")


async def test_the_proxy_profile_a_client_is_configured_with_reaches_no_world(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Docker's client adds these; this port never passed them, which is how they were missed.

    A proxy URL can carry credentials or an internal host name, by Docker's own documentation, so
    a machine whose Docker profile configures one was handing that string to every episode's
    environment, where the adversarial probe prints the whole of it. `--network none` stops the
    proxy being used and does nothing about the string being there."""
    config = tmp_path / "docker"
    config.mkdir()
    (config / "config.json").write_text(
        json.dumps(
            {
                "proxies": {
                    "default": {
                        "httpProxy": "http://user:secret@proxy.internal:3128",
                        "httpsProxy": "http://user:secret@proxy.internal:3128",
                        "noProxy": "internal.example",
                    }
                }
            }
        )
    )
    monkeypatch.setenv("DOCKER_CONFIG", str(config))
    probe = """
_os = __import__("os")
print(json.dumps({
    "proxyish": {k: v for k, v in _os.environ.items() if "proxy" in k.lower()},
}))
"""
    played = await play([probe])
    seen = json.loads(json.loads(played["outputs"][0]["content"])["output"])
    # The names may be there; the values may not, and nothing of the profile may be.
    assert all(value == "" for value in seen["proxyish"].values()), seen["proxyish"]
    assert "secret" not in json.dumps(seen)
    assert "proxy.internal" not in json.dumps(seen)


async def test_a_resume_is_refused_under_changed_machine_limits(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """What machine an episode was given is part of what its row measured.

    The limits decide latency, what a call timeout means and whether a world is killed for
    allocating, so a record whose rows ran under two of them is a record whose mean is about
    neither. They are captured once per process and recorded, so a relaunch under a changed one
    does not pass for the earlier measurement."""
    from shogym.envs.appworld import container as container_module
    from shogym.serve.stream import Information, TaskRef, TaskStream

    prov = tmp_path / "run"
    container_module.limits.cache_clear()
    first = shogym.make("appworld")
    stream = TaskStream(
        shogym.make,
        [TaskRef("appworld", TASK)],
        prov_dir=prov,
        feedback=Information(),
        identity=first.config_digest,
    )
    async with stream:
        await stream.get_task()
        await stream.dispatch("submit", {})

    # A machine with more of it. Same draw, same corpus, same image.
    container_module.limits.cache_clear()
    monkeypatch.setenv("SHOGYM_APPWORLD_CPUS", "8")
    roomier = shogym.make("appworld")
    try:
        assert roomier.config_digest != first.config_digest
        with pytest.raises(ValueError) as refused:
            TaskStream(
                shogym.make,
                [TaskRef("appworld", TASK)],
                prov_dir=prov,
                feedback=Information(),
                identity=roomier.config_digest,
                resume=True,
            )
        assert "run identity" in str(refused.value)
    finally:
        monkeypatch.undo()
        container_module.limits.cache_clear()


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

