"""``orca_bench`` describe: the task contract publishes ``instruction.md`` and nothing else.

This is the port's load-bearing correctness property. Every ORCA-bench task ships its own answer
in ``task.toml``: the feature flag that caused the incident, each root-cause event and its time,
and (for a control task) the quiet window that says there is no incident to find. Upstream is
safe by accident: Harbor never mounts ``task.toml`` into the agent's container. A shogym env has
no such accident to rely on, so the guarantee has to be tested.

Offline against a synthetic dataset in the real on-disk shape (the real one is downloaded on
demand and never vendored, and committing real task files would publish real answers). The same
assertions run against a real task in ``test_orca_bench_dataset.py``, behind the ``network`` mark.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

import shogym
from shogym.envs.orca_bench.tasks import load_index, load_metadata
from tests._fixtures import orca_bench_dataset as synth


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return synth.write_dataset(tmp_path / "orca")


@pytest.fixture
def env(dataset: Path):
    return shogym.make("orca_bench", config={"dataset_dir": str(dataset)})


def leaked_answers(text: str, answers: List[str]) -> List[str]:
    """Which of ``answers`` appear in ``text``. The redaction property is: none of them."""
    return [answer for answer in answers if answer and answer in text]


def test_env_is_registered() -> None:
    assert "orca_bench" in shogym.registered_envs()


def test_describe_publishes_the_instruction_verbatim(env, dataset: Path) -> None:
    for task in synth.TASKS:
        spec = env.describe(task.name)
        instruction = (dataset / task.name / "instruction.md").read_text(encoding="utf-8")
        assert spec.instructions.startswith(instruction.rstrip())


def test_describe_leaks_no_answer_string(env, dataset: Path) -> None:
    # The whole published contract, not just the instructions: tool descriptions and reference
    # templates go out on the same wire.
    for index, task in enumerate(synth.TASKS):
        answers = synth.answers(task)
        assert answers, f"{task.name} has no answers to check, so the test would be vacuous"
        for selector in (task.name, str(index)):
            blob = env.describe(selector).model_dump_json()
            assert leaked_answers(blob, answers) == [], f"{task.name} via {selector!r}"


def test_describe_publishes_no_metadata_label_either(env, dataset: Path) -> None:
    """Past the instruction, the footer carries only the task's public id and the dataset pin.

    ``section`` is not an answer, but it does say whether a task is a control (that there is no
    incident to find), and the snapshot id timestamps the window the incident sits in. So the
    check is that *nothing* from ``[metadata]`` reaches the footer, not that the known-bad keys
    are filtered out one by one."""
    for task in synth.TASKS:
        spec = env.describe(task.name)
        instruction = (dataset / task.name / "instruction.md").read_text(encoding="utf-8")
        footer = spec.instructions[len(instruction.rstrip()) :]
        metadata = load_metadata(dataset / task.name)
        values = [str(v) for v in metadata.values() if isinstance(v, str) and len(v) >= 4]
        assert leaked_answers(footer, values) == [], f"{task.name} footer: {footer!r}"
        assert task.name in footer  # the public id is the one thing it does carry


def test_the_leak_check_is_not_vacuous(env) -> None:
    """A describe() that appended the answer would be caught by exactly the check above."""
    task = synth.TASKS[0]
    spec = env.describe(task.name)
    leaky = spec.model_copy(
        update={"instructions": f"{spec.instructions}\n- root cause: {task.flag}"}
    )
    assert leaked_answers(leaky.model_dump_json(), synth.answers(task)) == [task.flag]


def test_describe_defaults_to_the_configured_task(dataset: Path) -> None:
    env = shogym.make("orca_bench", config={"dataset_dir": str(dataset), "task": 2})
    assert synth.TASKS[2].name in env.describe().instructions
    by_name = shogym.make(
        "orca_bench", config={"dataset_dir": str(dataset), "task": synth.TASKS[1].name}
    )
    assert synth.TASKS[1].name in by_name.describe().instructions


def test_describe_advertises_the_score_terminal(env) -> None:
    spec = env.describe("0")
    kinds = {tool.name: tool.terminal_kind for tool in spec.tools}
    assert kinds["submit_report"] == "score"
    assert kinds["terminate"] == "abort"
    assert kinds["exec"] == "none"


def test_duplicate_task_identities_are_refused(dataset: Path) -> None:
    """A task list is a set of identities.

    Two entries with one name collapse in the name lookup and in the position map, so the second
    copy answers to the first one's id: `load_task(0)` reports itself as task 1 and the footer
    `describe("0")` publishes names a task the caller did not ask for."""
    refs = load_index(dataset)
    with pytest.raises(ValueError, match=refs[1].name):
        shogym.make("orca_bench", config={"tasks": [refs[1], refs[1]]})


def test_a_duplicate_identity_error_names_both_sources(tmp_path: Path, dataset: Path) -> None:
    # Same name, different directories: the message has to say which two, since the whole
    # difficulty is that they are indistinguishable by identity.
    elsewhere = synth.write_dataset(tmp_path / "elsewhere")
    first = load_index(dataset)[0]
    second = load_index(elsewhere)[0]
    assert first.name == second.name and first.task_dir != second.task_dir

    with pytest.raises(ValueError) as raised:
        shogym.make("orca_bench", config={"tasks": [first, second]})
    message = str(raised.value)
    assert str(first.task_dir) in message and str(second.task_dir) in message


# ----- task identity: the id an episode is recorded under must select the task it ran -----


@pytest.fixture
def sliced(dataset: Path):
    """An env over a *slice* of the dataset, offset from the canonical order (tasks 1, 2, 3)."""
    refs = load_index(dataset)
    return shogym.make("orca_bench", config={"tasks": refs[1:4]}), refs


def published_instruction(spec) -> str:
    """The upstream instruction a published contract carries, without the provenance footer."""
    return spec.instructions.split("\n\n# This run\n")[0]


def test_the_loaded_task_id_describes_the_loaded_task(sliced) -> None:
    """The round trip the serve layer actually performs: load a task, then describe the id it
    reported. Publishing another task's instructions while the backend runs the loaded one is
    silent and unrecoverable, so it is pinned for every position of a non-trivial slice."""
    env, refs = sliced
    for position in range(env.num_tasks):
        loaded = env.load_task(position)
        spec = env.describe(str(loaded["task_idx"]))
        expected = (Path(loaded["task_dir"]) / "instruction.md").read_text(encoding="utf-8")
        assert published_instruction(spec) == expected.rstrip(), loaded["task_name"]
        assert spec.instructions.count(loaded["task_name"]) == 1


def test_the_default_task_id_describes_the_default_task(dataset: Path) -> None:
    """A one-task slice: the id reported for the default task must resolve, not raise."""
    refs = load_index(dataset)
    env = shogym.make("orca_bench", config={"tasks": [refs[2]]})
    loaded = env.load_task(None)
    assert loaded["task_name"] == refs[2].name
    spec = env.describe(str(loaded["task_idx"]))
    assert refs[2].name in spec.instructions


def test_the_dataset_index_is_provenance_and_never_a_selector(sliced) -> None:
    """The canonical position is recorded, and is deliberately not the served id.

    Under this slice the two differ for every task, which is what makes the round-trip test
    above non-vacuous: resolving the canonical index as a selector would publish a different
    task."""
    env, refs = sliced
    for position in range(env.num_tasks):
        loaded = env.load_task(position)
        assert loaded["task_idx"] == position
        assert loaded["dataset_index"] == refs[1 + position].dataset_index
        assert loaded["dataset_index"] != loaded["task_idx"]


def test_the_footer_publishes_the_served_id(sliced) -> None:
    # The footer's index is the id a harness can pass back to describe(), not the canonical one.
    env, _refs = sliced
    loaded = env.load_task(0)
    footer = env.describe(str(loaded["task_idx"])).instructions.split("\n\n# This run\n")[1]
    assert f"(index {loaded['task_idx']})" in footer
    assert f"(index {loaded['dataset_index']})" not in footer


@pytest.mark.parametrize("selector", ["nope", "99", "-1"])
def test_unknown_selectors_are_refused(env, selector: str) -> None:
    # Never silently serve another task under a bogus public id: that records the episode against
    # a task it did not run, and describe() of that id then refuses to resolve it.
    with pytest.raises(ValueError):
        env.describe(selector)


def test_out_of_range_index_is_refused_at_load(env) -> None:
    with pytest.raises(ValueError):
        env.load_task(99)
