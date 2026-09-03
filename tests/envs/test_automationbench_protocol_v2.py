"""How an automationbench attempt ends under the durable stream, and what the agent is told.

The scaffolding these Activities are built on is checked in ``tests/test_env_grading.py``. What is
checked here is this environment's own half: the rubric is the one a v1 run uses, the submission
names the workspace rather than the verdict, a seal that arrives after the world has been let go
refuses instead of publishing a zero, and an ordinary generation over it delivers the score the
seal committed. The last of those runs the whole arc through the real gateway and a real durable
service, because the seal, the grade and the body are three Activities inside one transaction and
calling them as functions would prove none of that.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from tests._fixtures.upstream_gate import gate

_adapter = gate(
    "shogym.envs.automationbench.adapter",
    package="automationbench",
    extra="automationbench",
)

pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402

import shogym  # noqa: E402
from examples.claude_code import serve as serve_mod  # noqa: E402
from shogym.envs._grading import MemoryCaptures  # noqa: E402
from shogym.envs.automationbench import mcp_server  # noqa: E402
from shogym.envs.automationbench.env_v1 import task_name  # noqa: E402
from shogym.envs.automationbench.protocol_v2 import (  # noqa: E402
    AUTOMATIONBENCH_GRADE,
    CANONICALIZATION_VERSION,
    automationbench_terminal,
    configuration_digest,
)
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import FilesystemBlobStore  # noqa: E402
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
from shogym.serve.protocol_v2.policy import PolicyViolation, check_grade_result  # noqa: E402

ATTEMPT = "b" * 32
SEAL_ID = "a" * 64
_CONTACT = "003001"
_UPDATE_URL = (
    f"https://yourinstance.salesforce.com/services/data/v61.0/sobjects/Contact/{_CONTACT}"
)
_TASK: Dict[str, Any] = {
    "example_id": 999001,
    "task": "test.sf_contact_phone_update",
    "prompt": [{"role": "user", "content": "Update Jordan Lee's phone to +1-555-0101."}],
    "answer": "",
    "info": {
        "zapier_tools": ["salesforce_contact_update"],
        "initial_state": {
            "salesforce": {
                "contacts": [
                    {
                        "id": _CONTACT,
                        "first_name": "Jordan",
                        "last_name": "Lee",
                        "email": "jordan.lee@acme.example.com",
                        "phone": "+1-555-0000",
                    }
                ]
            }
        },
        "assertions": [
            {
                "type": "salesforce_field_equals",
                "collection": "contacts",
                "record_id": _CONTACT,
                "field": "phone",
                "value": "+1-555-0101",
            }
        ],
    },
}
_CONFIG = {"tasks": [_TASK], "max_steps": 50}
_FILING = {"method": "PATCH", "url": _UPDATE_URL, "body": json.dumps({"Phone": "+1-555-0101"})}

#: The same contact under three assertions the world it starts in satisfies none of, beside a
#: filing that sets two of them. Two of three is a fraction with no exact decimal expansion, which
#: is what makes the resolution this environment declares visible in the number the rubric returns.
_PARTIAL_INFO: Dict[str, Any] = {
    **_TASK["info"],
    "assertions": [
        _TASK["info"]["assertions"][0],
        {
            "type": "salesforce_field_equals",
            "collection": "contacts",
            "record_id": _CONTACT,
            "field": "email",
            "value": "jordan.lee@acme.test",
        },
        {
            "type": "salesforce_field_equals",
            "collection": "contacts",
            "record_id": _CONTACT,
            "field": "last_name",
            "value": "Leigh",
        },
    ],
}
_PARTIAL_BODY = json.dumps({"Phone": "+1-555-0101", "Email": "jordan.lee@acme.test"})

_SHEETS = "https://sheets.googleapis.com/v4/spreadsheets/"


def _get(url: str, **params: Any) -> Any:
    """One read of an endpoint, the way the recorded run made every one of its calls."""
    arguments: Dict[str, Any] = {"method": "GET", "url": url}
    if params:
        arguments["params"] = json.dumps(params)
    return ("api_fetch", arguments)


def _sheet(name: str) -> Any:
    """One guess at the spreadsheet the recorded run spent most of its budget looking for."""
    return _get(_SHEETS + name)


def _search(query: str, top_k: int) -> Any:
    return ("api_search", {"query": query, "top_k": top_k})


#: The action sequence of the recorded cell-1 attempt at automationbench task 156, in order.
#: The run had a budget of 52 and spent every one of it on a read: it never found the spreadsheet
#: the task named, never sent an email, and left the workspace exactly as it was seeded, so the
#: rubric scored it at nothing. They are replayed against that task's own world and rubric, so
#: what the ending is read from is the recording whole: these fifty-two calls, no state changed,
#: no ``done`` called, and the last call reaching the horizon.
_TASK_156_CALLS = [
    _get("https://www.googleapis.com/drive/v3/files", pageSize=100),
    _search("trial extension requests queue", 8),
    _sheet("ss_trials"),
    _sheet("ss_trial_extensions"),
    _sheet("ss_extensions"),
    _get("https://gmail.googleapis.com/gmail/v1/users/me/messages", maxResults=50),
    _get("https://api.airtable.com/v0/meta/bases"),
    _get("https://slack.com/api/conversations.list", limit=200),
    _sheet("ss_trial_ops"),
    _sheet("ss_queue"),
    _sheet("ss_policy"),
    _sheet("ss_trial_policy"),
    _sheet("ss_trial_requests"),
    _search("spreadsheets list all available", 10),
    _sheet("ss_trial"),
    _sheet("ss_trial_ext"),
    _sheet("ss_trial_queue"),
    _sheet("ss_trial_extension"),
    _sheet("ss_trial_mgmt"),
    _sheet("ss_extension_policy"),
    _sheet("ss_requests"),
    _sheet("ss_extension_requests"),
    _sheet("ss_accounts"),
    _sheet("ss_trial_ext_requests"),
    _sheet("ss_ops"),
    _sheet("ss_trialext"),
    _get("https://gmail.googleapis.com/gmail/v1/users/me/messages/msg_mkt_noise_016"),
    _sheet("ss_trial_extensions/values/Policy!A1:Z100"),
    _sheet("ss_ext"),
    _sheet("ss_trial_ops_queue"),
    _sheet("zzz_definitely_not_real/values/Policy!A1:Z10"),
    _sheet("ss_trial_extensions/values/A1:Z100"),
    _sheet("trial_extensions"),
    _sheet("trials"),
    _sheet("sheet_trials"),
    _sheet("ss_trial_extension_queue"),
    _sheet("ss_growth"),
    _sheet("ss_customer_success"),
    _search("subscription trial end extend billing", 8),
    _sheet("trial_ops"),
    _sheet("ss_trial_ext_policy"),
    _sheet("ss_ext_requests"),
    _get(
        "https://gmail.googleapis.com/gmail/v1/users/me/messages",
        q="trial",
        maxResults=50,
        includeSpamTrash=True,
    ),
    _search("labels list mailbox", 3),
    _sheet("ss_extension"),
    _sheet("extensions"),
    _sheet("Trial%20Extensions"),
    _sheet("ss_trial_extensions_queue"),
    _sheet("ss_te"),
    _sheet("ss_trial_requests_queue"),
    _get("https://api.helpscout.net/v2/mailboxes"),
    _get("https://api.intercom.io/conversations"),
]


def sealing(session_id: str = "a-session") -> str:
    """One seeded world with the task's assertion satisfied, in this process."""
    from shogym.envs.automationbench import adapter

    mcp_server.begin_session(session_id, info=_TASK["info"])
    session = mcp_server._session_for(session_id)
    assert session is not None
    adapter.api_fetch(session.world, "PATCH", _UPDATE_URL, None, _FILING["body"])
    return session_id


async def seal_and_grade(route: Any, store: Any, blob_root: Optional[str] = None) -> Any:
    """Drive this port's two Activities as functions, which is enough for everything but the arc."""
    _version, activities = automationbench_terminal(route, store=store)
    sealed = await activities[0](
        SealAttemptInput(
            attempt_id=ATTEMPT,
            seal_id=SEAL_ID,
            native_terminal_name="done",
            canonicalization_version=CANONICALIZATION_VERSION,
            blob_root=blob_root,
        )
    )
    graded = await activities[1](
        GradeAttemptInput(
            attempt_id=ATTEMPT,
            seal_id=SEAL_ID,
            submission_digest="c" * 64,
            canonical_submission_text=sealed.canonical_submission_text,
            environment_recovery_token=sealed.environment_recovery_token,
            blob_root=blob_root,
        )
    )
    return sealed, graded


async def test_the_environment_and_not_the_stand_in_is_what_a_generation_is_built_over() -> None:
    """The two halves are one fact, and this environment declares both.

    Under the stand-in ``done`` carries nothing, so every attempt scores what an empty filing is
    worth however much of the workflow was carried out. The composition guard is what makes the
    two halves inseparable, and it is checked here against this environment rather than restated:
    an env that claimed the grade and brought no terminal is refused before a world is served.
    """
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config=_CONFIG, ends_on_horizon=False
    )
    try:
        declared = environment_grade(episode)
        assert declared == AUTOMATIONBENCH_GRADE
        assert declared.stand_in is False
        assert declared.score_component == "partial_credit"

        environment = environment_terminal(episode)
        assert environment.canonicalization_version == CANONICALIZATION_VERSION
        assert len(environment.activities) == 4
        assert environment.configuration_digest == configuration_digest(
            domain="public", max_steps=50, tasks=[_TASK]
        )
    finally:
        await episode.close()


async def test_an_environment_that_claims_this_grade_without_the_terminal_is_refused() -> None:
    """The guard, over this environment: half of the pair is not a grader."""
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config=_CONFIG, ends_on_horizon=False
    )
    try:
        setattr(episode.env, "protocol_v2_terminal", None)
        with pytest.raises(ValueError, match="protocol_v2_terminal"):
            environment_grade(episode)
    finally:
        await episode.close()


async def test_the_submission_is_the_workspace_and_the_verdict_stays_out_of_it() -> None:
    """``done`` takes no arguments, so what was filed is the world, named by its own bytes.

    The canonical text is what the digest covers and what the payload renderer is handed, so the
    numbers a grade is made of are not in it: a run composed to withhold its score would otherwise
    be handing the renderer the score in the field beside the one it withheld.
    """
    session = sealing()
    try:
        sealed, graded = await seal_and_grade(lambda _a: (None, session), MemoryCaptures())
    finally:
        mcp_server.end_session(session)

    submission = json.loads(sealed.canonical_submission_text)
    assert submission["canonicalization_version"] == CANONICALIZATION_VERSION
    assert list(submission["submission"]) == ["world_sha256"]
    assert "partial_credit" not in sealed.canonical_submission_text
    assert "salesforce" not in sealed.canonical_submission_text.lower()

    assert graded.score == 1.0
    assert graded.public_components == {"success": 1.0}
    assert graded.decode_state == "decoded"
    assert graded.grade == AUTOMATIONBENCH_GRADE


async def test_a_workspace_the_agent_never_changed_is_a_filing_that_said_nothing() -> None:
    """A run that changed nothing and a run that changed the wrong things both score zero.

    They are not the same filing, and the number alone cannot say so, so the decode state does.
    """
    session = "an-untouched-session"
    mcp_server.begin_session(session, info=_TASK["info"])
    try:
        _sealed, graded = await seal_and_grade(lambda _a: (None, session), MemoryCaptures())
    finally:
        mcp_server.end_session(session)
    assert graded.score == 0.0
    assert graded.decode_state == "ambiguous_zero"


async def test_a_headline_finer_than_this_environment_declared_is_not_what_it_publishes(
    tmp_path: Path,
) -> None:
    """Two assertions of three is a fraction with no end, and the rubric hands over all of it.

    The digits past the fourth say nothing about how many assertions passed, and a body prints
    what it is given. So the number that crosses is the fraction at the resolution this environment
    declared, and the division it came out of stays in the evidence a harness resolves. It has to
    be that way round: the same number unrounded is one the stream refuses, and the attempt ends
    over it rather than the agent being told what the workspace it left was worth.
    """
    from shogym.envs.automationbench import adapter

    session = "a-session-that-half-worked"
    mcp_server.begin_session(session, info=_PARTIAL_INFO)
    try:
        live = mcp_server._session_for(session)
        assert live is not None
        adapter.api_fetch(live.world, "PATCH", _UPDATE_URL, None, _PARTIAL_BODY)
        _sealed, graded = await seal_and_grade(
            lambda _a: (None, session), MemoryCaptures(), blob_root=str(tmp_path)
        )
    finally:
        mcp_server.end_session(session)

    assert graded.score == 0.6667
    assert graded.public_components == {"success": 0.0}

    verdict = json.loads(FilesystemBlobStore(tmp_path).read(graded.evidence.sha256))
    assert verdict["partial_credit"] == 2 / 3
    with pytest.raises(PolicyViolation, match="decimal places"):
        check_grade_result(
            score=verdict["partial_credit"],
            components={"success": verdict["success"]},
            grade=AUTOMATIONBENCH_GRADE,
        )


async def test_a_seal_that_arrives_after_the_world_was_let_go_refuses() -> None:
    """A missing session scores a clean zero, and a seal must not publish it.

    ``score_session`` answers a session that is not here with zeros. Under the stream the world
    can be gone before the seal arrives, and the zero it would publish is a grade the environment
    never took.
    """
    _version, activities = automationbench_terminal(lambda _a: None, store=MemoryCaptures())
    request = SealAttemptInput(
        attempt_id=ATTEMPT,
        seal_id=SEAL_ID,
        native_terminal_name="done",
        canonicalization_version=CANONICALIZATION_VERSION,
    )
    with pytest.raises(ApplicationError, match="no world this process opened"):
        await activities[0](request)

    session = sealing("a-session-that-goes-away")
    mcp_server.end_session(session)
    _version, gone = automationbench_terminal(lambda _a: (None, session), store=MemoryCaptures())
    with pytest.raises(ApplicationError, match="has been let go"):
        await gone[0](request)


async def test_a_worker_that_replaced_the_one_which_sealed_has_nothing_to_grade() -> None:
    """A world that is this process goes with it, and a grade over nothing is refused."""
    session = sealing("a-session-for-two-workers")
    try:
        sealed, _graded = await seal_and_grade(lambda _a: (None, session), MemoryCaptures())
    finally:
        mcp_server.end_session(session)
    _version, replaced = automationbench_terminal(lambda _a: None, store=MemoryCaptures())
    with pytest.raises(ApplicationError, match="holds nothing sealed"):
        await replaced[1](
            GradeAttemptInput(
                attempt_id=ATTEMPT,
                seal_id=SEAL_ID,
                submission_digest="c" * 64,
                canonical_submission_text=sealed.canonical_submission_text,
                environment_recovery_token=sealed.environment_recovery_token,
            )
        )


def test_the_documented_env_swap_reaches_this_environments_own_grader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The quickstarts swap envs through one variable, and this one is a swap worth making.

    An env with no grader of its own composes over the stand-in, and a run that wanted the default
    honest body gets a refusal rather than a withheld score. So what the variable reaches has to
    be an environment whose grade is its own.
    """
    monkeypatch.setenv("SHOGYM_ENV", "automationbench")
    try:
        assert importlib.reload(serve_mod).ENV == "automationbench"
    finally:
        monkeypatch.delenv("SHOGYM_ENV", raising=False)
        importlib.reload(serve_mod)
    assert "automationbench" in shogym.registered_envs()
    env = shogym.make("automationbench", config=_CONFIG)
    assert env.protocol_v2_grade() == AUTOMATIONBENCH_GRADE


@pytest.mark.network
async def test_an_ordinary_generation_tells_the_agent_the_score_this_world_earned() -> None:
    """The whole arc, through the real gateway and the real stream: task, filing, ack, payload.

    Every other test here calls the Activities as functions, and production reaches them one way
    only: the stream accepts the terminal, runs the seal and the grade inside the transaction that
    accepted it, mints the acknowledgement from what they returned, and releases a body built under
    the policy the obligation resolved to. The default is honest, so the body is the score the seal
    committed and the one number this environment declared beside it.
    """
    episode = await ServedEpisode.start(
        "automationbench", task=0, env_config=_CONFIG, ends_on_horizon=False
    )
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
    """Carry the workflow out, file it, and read what the generation says it was worth."""
    gateway = await open_gateway(client, episode, environment=environment)
    await gateway.close_queue()
    task = json.loads(await gateway.pull({}))
    attempt = task["attempt_id"]
    assert "Jordan Lee" in task["body"]

    await gateway.environment("api_fetch", {"attempt_id": attempt, "arguments": _FILING})
    ack = json.loads(await gateway.terminal({"attempt_id": attempt, "arguments": {}}))
    assert ack["kind"] == "seal_ack"
    assert ack["canonicalization_version"] == CANONICALIZATION_VERSION

    payload = json.loads(await gateway.pull({}))
    assert payload["kind"] == "payload"
    assert payload["body"] == f"attempt {attempt}\nscore 1\nsuccess 1"
    assert json.loads(await gateway.pull({}))["kind"] == "done"
    # And the row says the agent filed, which is what the horizon-filed row below is read against.
    [record] = await _records(gateway)
    assert (record.terminal_tool, record.terminal_source) == ("done", "agent")
    await gateway.aclose()


def _public_task(index: int) -> Dict[str, Any]:
    """One row of the public benchmark, at the index a run selects it by.

    The row carries its own deterministic noise, seeded upstream by ``example_id``, and its
    ``info`` is stored as JSON the way the dataset stores it. Both are read here rather than
    restated, so what this pins is the task a run at that index gets rather than a copy of it.
    """
    row = dict(_adapter.load_domain_tasks("public")[index])
    info = row.get("info", {})
    row["info"] = json.loads(info) if isinstance(info, str) else info
    return row


async def _records(gateway: Any) -> Any:
    """The generation's own rows, asked of the stream this gateway is serving."""
    from shogym.serve.protocol_v2.kernel.workflow import StreamWorkflow

    return list(await gateway._stream.handle.query(StreamWorkflow.attempt_records))


@pytest.mark.network
async def test_a_run_that_spends_its_budget_is_graded_where_the_budget_ran_out() -> None:
    """The recorded cell-1 attempt at task 156, replayed against this protocol.

    That attempt made fifty-two environment calls, changed nothing, and never called ``done``;
    the fifty-second call's own result carried the termination, the reward, the success flag and
    the feedback, because the episode graded the partial state at its horizon. This environment
    says the same thing under the durable stream, and this is where that is checked: the last
    call is dispatched and answered, the filing is made for the attempt as its step commits, the
    grade is the rubric's over the world the attempt left, and the honest body reaches the agent
    on the pull after it.

    The four outcomes are the recorded ones. The world was untouched, so the rubric scores it at
    nothing and the success flag is false, and this environment says which zero that is: a filing
    that said nothing rather than one that was read and got no credit.

    It is the public benchmark's own row 156 that is played, loaded the way a run loads it, and
    the row is named here as well as counted: an action list replayed against some other world
    and some other rubric would stay green while the reproduction it claims to cover broke.
    """
    row = _public_task(156)
    info = row["info"]
    assert (row["example_id"], task_name(row)) == (1121, "marketing.trial_extension_processing")
    assert len(info["assertions"]) == 20
    episode = await ServedEpisode.start("automationbench", task=156, ends_on_horizon=False)
    running = False
    try:
        assert episode.describe().horizon == len(_TASK_156_CALLS)
        assert row["prompt"][-1]["content"] in episode.describe().instructions
        async with durable_client() as client:
            running = True
            environment = environment_terminal(episode)
            assert environment.horizon_ending == "graded"
            async with stream_worker(client, activities=environment.activities):
                await _spend_the_budget(client, episode, environment)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")
    finally:
        await episode.close()


async def _spend_the_budget(client: Any, episode: ServedEpisode, environment: Any) -> None:
    """Make every call the recorded attempt made, and read what the horizon came to."""
    gateway = await open_gateway(client, episode, environment=environment)
    await gateway.close_queue()
    attempt = json.loads(await gateway.pull({}))["attempt_id"]

    for tool, arguments in _TASK_156_CALLS[:-1]:
        result = await gateway.environment(
            tool, {"attempt_id": attempt, "arguments": arguments}
        )
        assert len(result.content) == 1

    # The call that reaches the horizon. Its own observation comes back, and behind it the
    # acknowledgement of the filing that call ended: the attempt is over, and nothing in those
    # bytes says what it scored.
    tool, arguments = _TASK_156_CALLS[-1]
    result = await gateway.environment(tool, {"attempt_id": attempt, "arguments": arguments})
    assert len(result.content) == 2
    ack = json.loads(result.content[1].text)
    assert ack["kind"] == "seal_ack"
    assert ack["canonicalization_version"] == CANONICALIZATION_VERSION
    assert "score" not in result.content[1].text

    # The reward, the success flag and the feedback, on the pull after it.
    payload = json.loads(await gateway.pull({}))
    assert payload["kind"] == "payload"
    assert payload["body"] == f"attempt {attempt}\nscore 0\nsuccess 0"
    assert json.loads(await gateway.pull({}))["kind"] == "done"

    [record] = await _records(gateway)
    assert record.state == "ack_presented"
    assert record.final_failure is None
    assert (record.score, record.decode_state) == (0.0, "ambiguous_zero")
    assert (record.terminal_tool, record.terminal_source) == ("done", "horizon")
    assert record.payload_delivered is True
    await gateway.aclose()
