"""How a yc_bench attempt ends under the durable stream, and what the agent is told.

The scaffolding these Activities are built on is checked in ``tests/test_env_grading.py``. What is
checked here is this environment's own half, and one decision above all: the benchmark's headline
is the company's funds in cents, which is not a score, so the canonical component is whether the
run finished its year solvent and the funds stay in the evidence. Beside that, the submission
names the company rather than the numbers a grade is made of, a seal that arrives after the sim
has been torn down refuses instead of publishing the unseeded fallback, and an ordinary generation
over the environment delivers the score the seal committed.
"""

from __future__ import annotations

import importlib
import json
from typing import Any, Dict

import pytest

from tests._fixtures.upstream_gate import gate

gate("shogym.envs.yc_bench.adapter", package="yc_bench", extra="yc_bench")

pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402

import shogym  # noqa: E402
from examples.claude_code import serve as serve_mod  # noqa: E402
from shogym.envs._grading import MemoryCaptures  # noqa: E402
from shogym.envs.yc_bench import mcp_server  # noqa: E402
from shogym.envs.yc_bench.protocol_v2 import (  # noqa: E402
    CANONICALIZATION_VERSION,
    YC_BENCH_GRADE,
    _score,
    configuration_digest,
    yc_bench_terminal,
)
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    durable_client,
    environment_grade,
    environment_terminal,
    open_gateway,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    GradeAttemptInput,
    SealAttemptInput,
)

ATTEMPT = "b" * 32
SEAL_ID = "a" * 64

# One year of a company, as the sim reports it: solvent at the horizon, with money in the bank.
_SURVIVED: Dict[str, Any] = {
    "seeded": True,
    "survived": True,
    "final_funds_cents": 41_500_000,
    "tasks_succeeded": 12,
    "tasks_failed": 1,
    "horizon_reached": True,
    "terminal_reason": "horizon_end",
    "sim_time": "2026-01-01T00:00:00",
    "horizon_end": "2026-01-01T00:00:00",
}


def seeded(session_id: str = "a-session") -> str:
    """One seeded simulation in this process, at the start of its year."""
    mcp_server.begin_session(
        session_id,
        seed=1,
        config_name="default",
        start_date="2025-01-01",
        horizon_years=None,
        company_name="BenchCo",
    )
    return session_id


async def seal_and_grade(route: Any) -> Any:
    """Drive this port's two Activities as functions, which is enough for everything but the arc."""
    _version, activities = yc_bench_terminal(route, store=MemoryCaptures())
    sealed = await activities[0](
        SealAttemptInput(
            attempt_id=ATTEMPT,
            seal_id=SEAL_ID,
            native_terminal_name="submit",
            canonicalization_version=CANONICALIZATION_VERSION,
        )
    )
    graded = await activities[1](
        GradeAttemptInput(
            attempt_id=ATTEMPT,
            seal_id=SEAL_ID,
            submission_digest="c" * 64,
            canonical_submission_text=sealed.canonical_submission_text,
            environment_recovery_token=sealed.environment_recovery_token,
        )
    )
    return sealed, graded


def test_the_score_is_the_outcome_and_the_funds_are_not_a_score() -> None:
    """The v1 headline is cash in cents, which has no ceiling and a real negative branch.

    A score is one number in the unit interval, so publishing funds would mean inventing a
    maximum nobody measured against or clamping to one, and a clamped headline names a company
    that did not exist. The canonical component is the outcome instead, and the money stays in
    the verdict where a reader with the run's store can find it.
    """
    assert YC_BENCH_GRADE.stand_in is False
    assert YC_BENCH_GRADE.score_component == "success"
    published = {number.name for number in YC_BENCH_GRADE.public_components}
    assert published == {"survived", "horizon_reached"}
    assert "final_funds_cents" not in published

    survived = _score(_SURVIVED)
    assert survived["success"] == 1.0
    assert (survived["survived"], survived["horizon_reached"]) == (1.0, 1.0)
    assert survived["final_funds_cents"] == 41_500_000.0
    assert survived["decode_state"] == "decoded"

    bankrupt = _score({**_SURVIVED, "survived": False, "terminal_reason": "bankruptcy"})
    assert (bankrupt["success"], bankrupt["survived"]) == (0.0, 0.0)
    assert bankrupt["decode_state"] == "decoded"


def test_a_company_still_running_when_the_agent_stopped_is_credited_with_nothing() -> None:
    """Upstream's terminal gate, kept whole: ``submit`` is callable on turn one.

    An agent that sealed while the company was solvent and the year unfinished would bank the
    starting money without operating anything, so a solvent state on its own is worth nothing and
    reads as a filing that said nothing rather than as a year that ended badly.

    The gate takes the credit and not the facts. The company was solvent when it was sealed, so
    that is what the number beside the zero says: a body that answered this run with ``survived 0``
    would be saying the same of a company that went broke, which is a different run and a worse
    one.
    """
    premature = _score({**_SURVIVED, "horizon_reached": False, "terminal_reason": None})
    assert premature["success"] == 0.0
    assert (premature["survived"], premature["horizon_reached"]) == (1.0, 0.0)
    assert premature["decode_state"] == "ambiguous_zero"


async def test_the_environment_and_not_the_stand_in_is_what_a_generation_is_built_over() -> None:
    """The two halves are one fact, and this environment declares both.

    The composition guard is checked here against this environment rather than restated: an env
    that claimed the grade and brought no terminal is refused before a world is served.
    """
    episode = await ServedEpisode.start("yc_bench", task=0, ends_on_horizon=False)
    try:
        assert environment_grade(episode) == YC_BENCH_GRADE
        environment = environment_terminal(episode)
        assert environment.canonicalization_version == CANONICALIZATION_VERSION
        assert len(environment.activities) == 4
        assert environment.configuration_digest == configuration_digest(
            task_split="train",
            config_name="default",
            start_date="2025-01-01",
            horizon_years=None,
            company_name="BenchCo",
            seeds=list(range(1, 17)),
        )

        setattr(episode.env, "protocol_v2_terminal", None)
        with pytest.raises(ValueError, match="protocol_v2_terminal"):
            environment_grade(episode)
    finally:
        await episode.close()


async def test_the_submission_names_the_company_and_the_numbers_stay_out_of_it() -> None:
    """``submit`` takes no arguments, so what was filed is the end state, named by its own bytes.

    The canonical text is what the digest covers and what the payload renderer is handed, so the
    funds and the outcome are not in it: a run composed to withhold its score would otherwise be
    handing the renderer the material of the score in the field beside the one it withheld.
    """
    session = seeded()
    try:
        sealed, graded = await seal_and_grade(lambda _a: (None, session))
    finally:
        mcp_server.end_session(session)

    submission = json.loads(sealed.canonical_submission_text)
    assert submission["canonicalization_version"] == CANONICALIZATION_VERSION
    assert list(submission["submission"]) == ["end_state_sha256"]
    assert "funds" not in sealed.canonical_submission_text

    # A company sealed at the start of its year has not finished one, so it earned nothing, and
    # the money it still has is reported as the money it still has.
    assert graded.score == 0.0
    assert graded.public_components == {"survived": 1.0, "horizon_reached": 0.0}
    assert graded.decode_state == "ambiguous_zero"
    assert graded.grade == YC_BENCH_GRADE


async def test_a_seal_that_arrives_after_the_sim_was_torn_down_refuses() -> None:
    """``read_verdict`` answers a session that is not here with the unseeded fallback, which is
    faithful where the DB is open by construction. Under the stream the session can be gone
    before the seal arrives, and the fallback published there is a result nobody read.
    """
    request = SealAttemptInput(
        attempt_id=ATTEMPT,
        seal_id=SEAL_ID,
        native_terminal_name="submit",
        canonicalization_version=CANONICALIZATION_VERSION,
    )
    _version, nowhere = yc_bench_terminal(lambda _a: None, store=MemoryCaptures())
    with pytest.raises(ApplicationError, match="no world this process opened"):
        await nowhere[0](request)

    session = seeded("a-session-that-goes-away")
    mcp_server.end_session(session)
    _version, gone = yc_bench_terminal(lambda _a: (None, session), store=MemoryCaptures())
    with pytest.raises(ApplicationError, match="has been let go"):
        await gone[0](request)


def test_the_documented_env_swap_reaches_this_environments_own_grader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The quickstarts swap envs through one variable, and this one is a swap worth making."""
    monkeypatch.setenv("SHOGYM_ENV", "yc_bench")
    try:
        assert importlib.reload(serve_mod).ENV == "yc_bench"
    finally:
        monkeypatch.delenv("SHOGYM_ENV", raising=False)
        importlib.reload(serve_mod)
    assert "yc_bench" in shogym.registered_envs()
    assert shogym.make("yc_bench").protocol_v2_grade() == YC_BENCH_GRADE


@pytest.mark.network
async def test_an_ordinary_generation_tells_the_agent_the_score_this_run_earned() -> None:
    """The whole arc, through the real gateway and the real stream: task, command, ack, payload.

    Every other test here calls the Activities as functions, and production reaches them one way
    only: the stream accepts the terminal, runs the seal and the grade inside the transaction that
    accepted it, mints the acknowledgement from what they returned, and releases a body built under
    the policy the obligation resolved to. A year that was never run earns nothing, and the body
    says so under the two names this environment declared.
    """
    episode = await ServedEpisode.start("yc_bench", task=0, ends_on_horizon=False)
    running = False
    try:
        async with durable_client() as client:
            running = True
            environment = environment_terminal(episode)
            async with stream_worker(client, activities=environment.activities):
                await _drive_the_arc(client, episode, environment)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")
    finally:
        await episode.close()


async def _drive_the_arc(client: Any, episode: ServedEpisode, environment: Any) -> None:
    """Operate the company once, file it, and read what the generation says it was worth."""
    gateway = await open_gateway(client, episode, environment=environment)
    await gateway.close_queue()
    task = json.loads(await gateway.pull({}))
    attempt = task["attempt_id"]

    status = await gateway.environment(
        "run_command", {"attempt_id": attempt, "arguments": {"command": "yc-bench company status"}}
    )
    assert json.loads(status.content[0].text)["ok"] is True

    ack = json.loads(await gateway.terminal({"attempt_id": attempt, "arguments": {}}))
    assert ack["kind"] == "seal_ack"
    assert ack["canonicalization_version"] == CANONICALIZATION_VERSION

    payload = json.loads(await gateway.pull({}))
    assert payload["kind"] == "payload"
    assert payload["body"] == f"attempt {attempt}\nscore 0\nhorizon_reached 0\nsurvived 1"
    assert json.loads(await gateway.pull({}))["kind"] == "done"
    await gateway.aclose()
