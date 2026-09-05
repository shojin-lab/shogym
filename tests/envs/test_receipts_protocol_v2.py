"""How a receipts attempt ends under the durable stream, and what the agent is told.

The scaffolding these Activities are built on is checked in ``tests/test_env_grading.py``. What
is checked here is this environment's own half. The score a generation commits is the number a
v1 run reports for the same filing, and that is checked by running both paths over one task for
a filing that is right, one that is wrong, one that leaves records out and one nothing can be
read from, rather than by restating the arithmetic. The acknowledgement says a filing landed and
nothing about what it was worth. The cells the seal rendered stay under the seal, where a payload
policy will read them, and reach neither the acknowledgement nor the grade.

A filing that says nothing is refused rather than sealed, on both paths and for the same reason,
which is the tool's own schema. The fork behind an accepted one is rendered once however many
times that filing is sent, and that is counted rather than inferred from the bytes.

Two properties are about addressing rather than grading, and they are here because they are what
lets one generation work both siblings of a family. The roster is flat, so a position names a
family and a sibling; and the environment's configuration says nothing about which sibling,
because a world whose configuration is not the one a generation was started as is refused, and
the second position is worked in a world of its own.

The last test runs the whole arc through the real gateway and a real durable service, because
the seal, the grade and the body are three Activities inside one transaction and calling them as
functions would prove none of that.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import threading
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema
import pytest

pytest.importorskip("temporalio")

from temporalio.exceptions import ApplicationError  # noqa: E402

from shogym.envs._grading import GRADED, SEALED, DirectoryCaptures  # noqa: E402
from shogym.envs.receipts import bank as bank_mod  # noqa: E402
from shogym.envs.receipts import bundle as bundle_mod  # noqa: E402
from shogym.envs.receipts.env_v1 import (  # noqa: E402
    SIBLINGS,
    ReceiptsV1Env,
    sibling,
)
from shogym.envs.receipts.generators.ledger import GENERATOR  # noqa: E402
from shogym.envs.receipts.protocol_v2 import (  # noqa: E402
    CANONICALIZATION_VERSION,
    CELLS,
    RECEIPTS_GRADE,
    cells_for,
    configuration_digest,
    receipts_terminal,
)
from shogym.serve.episode import ServedEpisode  # noqa: E402
from shogym.serve.protocol_v2 import FLOOR_HORIZON, GRADED_HORIZON  # noqa: E402
from shogym.serve.protocol_v2.gateway import (  # noqa: E402
    _check_graded_horizon,
    durable_client,
    environment_grade,
    environment_horizon_ending,
    environment_terminal,
    open_gateway,
    stream_start,
    stream_worker,
    terminal_manifest,
)
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    ABANDONED,
    FinalizeRequest,
    GradeAttemptInput,
    SealAttemptInput,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    DELIVER,
    HONEST,
    KERNEL_STAND_IN_GRADE,
    POLICIES,
    PolicyViolation,
    check_grade,
    check_grade_result,
)
from tests._fixtures.receipts_bundle import verified_bundle  # noqa: E402

ATTEMPT = "b" * 32


def seal_of(name: str) -> str:
    """One test's own seal id.

    Distinct per test on purpose. The records a seal writes are kept on disk beside the bundle's
    forks now, and the bundle is the module's, so two tests filing under one key would be reading
    each other's evidence rather than their own.
    """
    return sha256(name.encode("utf-8")).hexdigest()


@pytest.fixture(scope="module")
def frozen_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One admission bundle that actually verifies, shared by the module."""
    return verified_bundle(tmp_path_factory.mktemp("bundles"))


def _config(frozen_bundle: Path) -> Dict[str, Any]:
    """What a generation over a dealt family is configured as: a bundle and nothing else."""
    return {"bundle": str(frozen_bundle)}


async def _episode(task: int, frozen_bundle: Path, tmp_path: Path) -> ServedEpisode:
    """One world of this environment, at one roster position, for a generation to seal.

    ``ends_on_horizon`` is false because under this protocol the stream is what ends an
    attempt: an episode that graded itself at its own horizon would end one behind the stream's
    back, and the gateway refuses to open a generation on such an episode.
    """
    return await ServedEpisode.start(
        "receipts_v1",
        task=task,
        env_config=_config(frozen_bundle),
        ends_on_horizon=False,
        trace_path=tmp_path / f"run-{task}.jsonl",
    )


def _filing(env: ReceiptsV1Env, position: int) -> str:
    """The filing of an agent that applied the drawn convention exactly, at one position."""
    ordinal, index = env._position(position)
    task = env.instance(ordinal).side(sibling(index))
    return "\n".join(
        f"{identifier},{value}"
        for identifier, value in zip(GENERATOR.row_identifiers(task.table), task.key)
    )


def _spoiled(filing: str) -> str:
    """The same filing with every record but the first answered wrongly."""
    lines = filing.splitlines()
    return "\n".join([lines[0]] + [f"{line.split(',')[0]},Wrong" for line in lines[1:]])


def _partial(filing: str) -> str:
    """The same filing with every record but the first left out."""
    return filing.splitlines()[0]


def _all_wrong(filing: str) -> str:
    """The same records, every one of them answered wrongly.

    A filing the parser reads whole and the scorer credits with nothing. Its zero has to stay
    distinct from the zero a filing nothing could be read from comes to, and the decode state
    is where that distinction lives.
    """
    return "\n".join(f"{line.split(',')[0]},Wrong" for line in filing.splitlines())


def _second_wrong(filing: str) -> str:
    """The same filing with the second record answered wrongly, and nothing else touched.

    A filing no other test in this module sends, so the fork behind it is one this test's own
    seals render rather than one the store already holds.
    """
    lines = filing.splitlines()
    return "\n".join([lines[0], f"{lines[1].split(',')[0]},Wrong"] + lines[2:])


def _last_wrong(filing: str) -> str:
    """The same filing with the last record answered wrongly, and nothing else touched.

    A filing no other test files, so the fork behind it is one this test's seal renders rather
    than one the store already holds: what is being counted is a render, and a fork committed
    by an earlier filing of the same bytes is replayed and never built.
    """
    lines = filing.splitlines()
    return "\n".join(lines[:-1] + [f"{lines[-1].split(',')[0]},Wrong"])


def _store(episode: ServedEpisode) -> DirectoryCaptures:
    """The records this environment's own seals are written to, opened again from the disk."""
    return DirectoryCaptures(episode.env.seals)


async def seal_and_grade(
    episode: ServedEpisode,
    filing: str,
    *,
    seal_id: str,
    activities: Optional[List[Any]] = None,
    blob_root: Optional[str] = None,
) -> Any:
    """Drive this port's two Activities as functions, which is enough for everything but the arc.

    The Activities are the environment's own, asked for the way a generation asks for them, so
    the store under test is the one production uses rather than one this file chose.
    """
    if activities is None:
        _version, activities, _digest = episode.env.protocol_v2_terminal(
            lambda _a: (episode.env, episode.session_id)
        )
    sealed = await activities[0](
        SealAttemptInput(
            attempt_id=ATTEMPT,
            seal_id=seal_id,
            native_terminal_name="submit_filing",
            canonicalization_version=CANONICALIZATION_VERSION,
            native_arguments={"filing": filing},
            blob_root=blob_root,
        )
    )
    graded = await activities[1](
        GradeAttemptInput(
            attempt_id=ATTEMPT,
            seal_id=seal_id,
            submission_digest="d" * 64,
            canonical_submission_text=sealed.canonical_submission_text,
            environment_recovery_token=sealed.environment_recovery_token,
            blob_root=blob_root,
        )
    )
    return sealed, graded


async def test_the_environment_and_not_the_stand_in_is_what_a_generation_is_built_over(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The two halves are one fact, and this environment declares both.

    Under the stand-in a filing is worth what its shape is worth, so a table of wrong answers
    and a table of right ones come to the same number. The composition guard is what makes the
    two halves inseparable, and it is checked here against this environment rather than
    restated: an env that claimed the grade and brought no terminal is refused before a world
    is served.
    """
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        declared = environment_grade(episode)
        assert declared == RECEIPTS_GRADE
        assert declared.stand_in is False
        assert declared.score_component == "component_score"

        environment = environment_terminal(episode)
        assert environment.canonicalization_version == CANONICALIZATION_VERSION
        assert len(environment.activities) == 4
        assert environment.configuration_digest == configuration_digest(
            genre="ledger",
            side=None,
            source=bundle_mod.load(frozen_bundle).digest,
            dealable=True,
        )

        setattr(episode.env, "protocol_v2_terminal", None)
        with pytest.raises(ValueError, match="protocol_v2_terminal"):
            environment_grade(episode)
    finally:
        await episode.close()


def test_the_roster_is_one_position_for_every_sibling_of_every_family(
    frozen_bundle: Path,
) -> None:
    """A position names a family and a sibling, and nothing else has to be configured.

    The environment used to serve one sibling of every family and take which one from its
    configuration. A generation cannot change its environment's configuration between
    positions, so that environment could serve A or B and never both, and the point measurement
    is A then B. What replaces it is arithmetic on the position, which is why the sibling a
    position names is asked of one function rather than written at the call sites.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    assert env.num_tasks == len(env._ordinals) * len(SIBLINGS)
    first = env.instance(env._ordinals[0])
    second = env.instance(env._ordinals[1])
    assert env.describe("0").task_id == first.a.task_id
    assert env.describe("1").task_id == first.b.task_id
    assert env.describe("2").task_id == second.a.task_id

    assert (sibling(0), sibling(1)) == ("a", "b")
    with pytest.raises(ValueError, match="no sibling 2"):
        sibling(len(SIBLINGS))

    # And the narrowing the v1 path and the CLI still have serves one sibling of every family,
    # which is a different roster and says so in what the environment is configured as.
    narrowed = ReceiptsV1Env(**_config(frozen_bundle), side="b")
    assert narrowed.num_tasks == len(narrowed._ordinals)
    assert narrowed.describe("0").task_id == first.b.task_id
    assert configuration_digest(
        genre="ledger", side="b", source=bundle_mod.load(frozen_bundle).digest, dealable=True
    ) != configuration_digest(
        genre="ledger", side=None, source=bundle_mod.load(frozen_bundle).digest, dealable=True
    )


async def test_one_configuration_serves_both_siblings_of_a_family(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The two worlds a family needs are one environment, which is what the gateway checks.

    Each task after the first is worked in a world of its own, and a world whose
    canonicalization version or configuration digest is not the one the generation was started
    as is refused before the task is presented. So the two things compared there are compared
    here, over the two positions of one family: the same configuration, and two different
    siblings of one drawn convention.
    """
    first = await _episode(0, frozen_bundle, tmp_path)
    second = await _episode(1, frozen_bundle, tmp_path)
    try:
        a = environment_terminal(first)
        b = environment_terminal(second)
        assert a.canonicalization_version == b.canonicalization_version
        assert a.configuration_digest == b.configuration_digest

        env = ReceiptsV1Env(**_config(frozen_bundle))
        instance = env.instance(env._ordinals[0])
        assert first.describe().task_id == instance.a.task_id
        assert second.describe().task_id == instance.b.task_id
        assert instance.a.task_id != instance.b.task_id
        assert first.describe().instructions != second.describe().instructions
    finally:
        await second.close()
        await first.close()


async def test_a_filing_sealed_under_the_stream_scores_what_the_v1_path_scores(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """One task, five filings, two paths to a number, and it is the same number every time.

    The v1 path parses the filing, scores it and reports ``component_score`` as episode
    feedback. The stream seals the same filing and commits the score the generation publishes.
    One filing agrees trivially, so the matrix is the five kinds this environment distinguishes:
    every record right, every record but one wrong, every record wrong, most records left out,
    and a filing nothing can be read from.

    The last two rows are the ones that carry the distinction a number cannot. A filing read
    whole and credited with nothing and a filing nothing could be read from are both zero, and
    they are different facts about the attempt, so the decode state is compared beside the score
    rather than left to be inferred from it.

    A sixth kind, a filing that says nothing at all, has no number on either path: it is refused
    before it can be sealed, which is what the refusal test below is for. A seventh, an attempt
    that files no terminal at all, has no seal to compare and takes the floor instead, which is
    what the floor test at the end of this file is for.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    whole = _filing(env, 0)
    filings = {
        "every record right": whole,
        "one record right": _spoiled(whole),
        "every record wrong": _all_wrong(whole),
        "most records left out": _partial(whole),
        "nothing readable": "I could not work out the rule",
    }
    expected = {
        "every record right": (1.0, 1.0, "decoded"),
        "one record right": (None, 0.0, "decoded"),
        "every record wrong": (0.0, 0.0, "decoded"),
        "most records left out": (None, 0.0, "decoded"),
        "nothing readable": (0.0, 0.0, "ambiguous_zero"),
    }

    read: Dict[str, Dict[str, Any]] = {}
    for name, filing in filings.items():
        v1 = await _episode(0, frozen_bundle, tmp_path)
        try:
            await v1.call("submit_filing", {"filing": filing})
            feedback = {item["name"]: item["value"] for item in v1.terminal_feedback}
        finally:
            await v1.close()

        v2 = await _episode(0, frozen_bundle, tmp_path)
        try:
            _sealed, graded = await seal_and_grade(
                v2, filing, seal_id=seal_of(f"parity {name}")
            )
        finally:
            await v2.close()

        score, solved, decode = expected[name]
        assert graded.score == feedback["component_score"], name
        assert graded.public_components == {"solved": solved}, name
        assert feedback["solved"] is bool(solved), name
        assert graded.decode_state == decode, name
        assert graded.grade == RECEIPTS_GRADE
        if score is None:
            assert 0.0 < graded.score < 1.0, name
        else:
            assert graded.score == score, name
        read[name] = feedback

    # The two partial filings can come to one number and are not one filing, and the v1 reading
    # is what says so: one named every record and got one right, the other named one record.
    assert read["one record right"]["rows_filed"] > read["most records left out"]["rows_filed"]
    assert read["most records left out"]["rows_omitted"] > 0.0
    # The two zeros are two different endings of an attempt, and only one of them was read.
    assert read["every record wrong"]["rows_filed"] > 0.0
    assert "no_filing" not in read["every record wrong"]
    assert "rows_filed" not in read["nothing readable"]
    assert read["nothing readable"]["no_filing"] == "no_known_identifier"


async def test_a_filing_that_says_nothing_is_refused_rather_than_sealed(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """An empty call is a mistake the agent can correct, and it is that on both paths.

    A served episode holds every required string to being non-blank on top of the schema; a
    durable generation holds a terminal call to the schema and nothing else. So the rule is in
    the schema, where both of them read it, and the same empty call is refused on both rather
    than being a correctable mistake on one and a seal worth nothing on the other. A seal is the
    one answer no later call can take back, and an attempt that filed nothing has not filed.

    What the refusal leaves behind is the task: the episode is still open afterwards and the
    filing that follows is sealed and scored, which is the whole point of refusing rather than
    ending it at zero.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        schema = {
            manifest.name: manifest.input_schema for manifest in episode.describe().tools
        }["submit_filing"]
        for empty in ("", " ", "\n\t "):
            with pytest.raises(jsonschema.ValidationError):
                jsonschema.validate({"filing": empty}, schema)

            answer = await episode.call("submit_filing", {"filing": empty})
            assert answer.terminated is False
            assert json.loads(answer.content)["validation_error"] is True
            assert episode.terminal_feedback == []

        # The task the refusal left open is still there to be filed.
        result = await episode.call("submit_filing", {"filing": _filing(env, 0)})
        assert result.terminated
        feedback = {item["name"]: item["value"] for item in episode.terminal_feedback}
        assert feedback["component_score"] == 1.0
    finally:
        await episode.close()


async def test_the_acknowledgement_says_the_filing_landed_and_nothing_it_was_worth(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """What the canonical text covers is the agent's own act, read back canonically.

    It is what the digest is taken over and what a payload renderer is handed, so the score is
    not in it and neither is the answer key: a run composed to serve a placebo would otherwise
    be handing the renderer the corrections in the field beside the one it withheld.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal, index = env._position(0)
    task = env.instance(ordinal).side(sibling(index))
    filing = _spoiled(_filing(env, 0))

    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        sealed, graded = await seal_and_grade(
            episode, filing, seal_id=seal_of("acknowledgement")
        )
    finally:
        await episode.close()

    submission = json.loads(sealed.canonical_submission_text)
    assert submission["canonicalization_version"] == CANONICALIZATION_VERSION
    assert list(submission["submission"]) == ["filing"]
    assert submission["submission"]["filing"] == filing
    assert "component_score" not in sealed.canonical_submission_text
    assert "solved" not in sealed.canonical_submission_text
    assert "PASS" not in sealed.canonical_submission_text
    # The records this filing got wrong carry the answer, and the answer is not in what the
    # acknowledgement commits to. The first record was filed correctly, so it is not checked.
    identifiers = GENERATOR.row_identifiers(task.table)
    for identifier, band in list(zip(identifiers, task.key))[1:]:
        if band:
            assert f"{identifier},{band}" not in sealed.canonical_submission_text
    assert graded.score < 1.0


async def test_a_retried_seal_returns_the_first_seals_numbers_and_renders_nothing_again(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """One filing is sealed once, and a second call under that key reads what the first wrote.

    An Activity is retried, and a filing that was read a second time would move the digest the
    acknowledgement already committed to and score a table nobody filed. The retry here carries
    a different filing, which is the strongest form of the same call: what comes back is the
    first filing's canonical text, the first filing's score and the first filing's cells.

    The render is counted rather than inferred. Byte equality says the cells agree; it does not
    say a second render never happened, and a second render is the failure this is about: two
    branches of one fork holding bytes that were built twice is exactly what a committed fork
    exists to prevent. So the bank's renderer is watched, and one filing is one render.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    whole = _filing(env, 0)
    filing = _last_wrong(whole)
    rows = len(whole.splitlines())
    key = seal_of("retry")
    renders: List[str] = []
    rendered = bank_mod.render_fork

    def counted(generator: Any, instance: Any, side: str, raw: Any) -> Any:
        renders.append(str(raw))
        return rendered(generator, instance, side, raw)

    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        setattr(bank_mod, "render_fork", counted)
        first, graded = await seal_and_grade(episode, filing, seal_id=key)
        again, regraded = await seal_and_grade(episode, whole, seal_id=key)
        # And one more seal of its own, carrying the same filing: a fork the store already holds
        # is replayed rather than built, which is the other half of rendering once.
        elsewhere, _also = await seal_and_grade(
            episode, filing, seal_id=seal_of("retry, another seal")
        )
        store = _store(episode)
    finally:
        setattr(bank_mod, "render_fork", rendered)
        await episode.close()

    expected = round((rows - 1) / rows, 6)
    assert again.canonical_submission_text == first.canonical_submission_text
    assert again.environment_recovery_token == first.environment_recovery_token
    assert (regraded.score, regraded.public_components) == (expected, {"solved": 0.0})
    assert graded.score == expected
    assert again.canonical_submission_text.count("Wrong") == 1
    # One render, for the filing that sealed. The retry read the record it wrote, and the second
    # seal read the committed fork, so neither of them built a cell.
    assert renders == [filing]
    assert cells_for(store, key) == cells_for(store, seal_of("retry, another seal"))
    assert elsewhere.canonical_submission_text == first.canonical_submission_text


async def test_two_seals_of_one_filing_arriving_together_render_the_fork_once(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Once is once under contention, and that is what the fork's claim is for.

    Two seals of one filing can arrive together, and each of them looks for a committed fork
    before it builds one. Without a claim both find nothing and both render, which is what the
    sequential tests above cannot see: they would come to the same bytes, because a render is a
    function of the filing and the instance, but the environment says the cells are made once
    and once is what is checked here rather than what they came to.

    The two seals are held at the door of the fork store until both are there, so the race is
    the arrangement under test rather than a timing accident. What is asserted is one render,
    two seals that each answer, and one set of cells under both of them.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    filing = _second_wrong(_filing(env, 0))
    keys = (seal_of("together one"), seal_of("together two"))
    renders: List[str] = []
    counting = threading.Lock()
    both_here = threading.Barrier(2, timeout=60)
    rendered = bank_mod.render_fork
    committed = bank_mod.fork_for

    def counted(generator: Any, instance: Any, side: str, raw: Any) -> Any:
        with counting:
            renders.append(str(raw))
        return rendered(generator, instance, side, raw)

    def together(*arguments: Any, **named: Any) -> Any:
        # Both seals are inside the fork store, and neither has written, which is the moment a
        # claim decides and nothing else does.
        both_here.wait()
        return committed(*arguments, **named)

    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        setattr(bank_mod, "render_fork", counted)
        setattr(bank_mod, "fork_for", together)
        answers = await asyncio.gather(
            seal_and_grade(episode, filing, seal_id=keys[0]),
            seal_and_grade(episode, filing, seal_id=keys[1]),
        )
        store = _store(episode)
    finally:
        setattr(bank_mod, "fork_for", committed)
        setattr(bank_mod, "render_fork", rendered)
        await episode.close()

    assert renders == [filing]
    sealed = [answer[0] for answer in answers]
    graded = [answer[1] for answer in answers]
    assert sealed[0].canonical_submission_text == sealed[1].canonical_submission_text
    assert graded[0].score == graded[1].score
    cells = [cells_for(store, key) for key in keys]
    assert cells[0] is not None
    assert cells[0] == cells[1]


async def test_the_cells_are_kept_under_the_seal_that_rendered_them(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """What one seal rendered is on the disk under that seal, and the bytes are the fork's own.

    The key is the seal rather than the attempt. Two executions of one attempt carry one public
    id between them and have a filing and a fork each, so an answer looked up by attempt would
    be whichever of them sealed first; that is checked here by sealing two filings for one
    attempt id and asking for each.

    Where it is kept matters as much as what it is keyed by. The records go to the bank's fork
    store, so a reader that opens that directory later gets what was sealed rather than nothing,
    which is what a process holding the only copy would have left behind.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal, index = env._position(0)
    filing = _spoiled(_filing(env, 0))
    other = _partial(_filing(env, 0))
    first, second = seal_of("cells one"), seal_of("cells two")

    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        sealed, graded = await seal_and_grade(episode, filing, seal_id=first)
        await seal_and_grade(episode, other, seal_id=second)
        seals = episode.env.seals
    finally:
        await episode.close()

    rendered = bank_mod.render_fork(GENERATOR, env.instance(ordinal), sibling(index), filing)
    expected = {
        kind: rendered.agent_bytes(kind).decode("ascii") for kind in rendered.digests
    }
    assert set(expected) == {"graded", "placebo", "oracle"}

    # Opened again from the disk, after the store that wrote them is gone.
    store = DirectoryCaptures(seals)
    held = store.held(first, SEALED)
    assert held is not None
    assert held[CELLS] == expected
    assert cells_for(store, first) == expected

    # The second seal of the same attempt is its own filing and its own cells, and asking for
    # one does not answer with the other.
    later = cells_for(store, second)
    assert later is not None
    assert later != expected
    assert cells_for(store, seal_of("a seal nothing was filed under")) is None

    # And none of it crosses. The acknowledgement commits to the filing, the grade publishes
    # two numbers, and the verdict beside them names the cells by digest rather than carrying
    # them.
    for cell in expected.values():
        assert cell not in sealed.canonical_submission_text
    verdict = store.held(first, GRADED)
    assert verdict is not None
    assert verdict["cell_digests"] == dict(rendered.digests)
    assert set(graded.public_components) == {"solved"}


def test_the_grade_publishes_the_score_and_whether_the_table_was_solved() -> None:
    """The roster is closed, and what the receipt says row by row is not on it.

    A published number is a field an agent reads, so the environment declares the closed set of
    them before it grades anything. Whether the whole table was right is one whole number; the
    per-row verdicts and the corrections are the receipt's, and an arm that carries them carries
    the environment's own rendered cell rather than a number a body printed.
    """
    check_grade(RECEIPTS_GRADE)
    assert RECEIPTS_GRADE.stand_in is False
    assert RECEIPTS_GRADE.score_component == "component_score"
    assert RECEIPTS_GRADE.score_places == 6
    published = {number.name: number for number in RECEIPTS_GRADE.public_components}
    assert set(published) == {"solved"}
    assert (published["solved"].minimum, published["solved"].maximum) == (0.0, 1.0)
    assert published["solved"].places == 0

    check_grade_result(score=0.041667, components={"solved": 0.0}, grade=RECEIPTS_GRADE)
    with pytest.raises(PolicyViolation, match="rows_filed"):
        check_grade_result(
            score=1.0, components={"rows_filed": 24.0}, grade=RECEIPTS_GRADE
        )
    with pytest.raises(PolicyViolation, match="decimal places"):
        check_grade_result(score=0.5, components={"solved": 0.5}, grade=RECEIPTS_GRADE)


async def test_the_horizon_is_the_floor_because_the_filing_is_the_agents(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """An attempt that reaches this horizon filed nothing, and nothing is what it is worth.

    A graded horizon files the terminal as the last step commits, and the filing it makes is
    the world the attempt left: the gateway writes no arguments into it. This terminal is the
    filing, so there would be nothing for the gateway to put in the call, and a generation whose
    terminal declares arguments is refused a graded horizon where it is composed. What the floor
    records for such an ending is the number the v1 path reports for it.
    """
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        assert environment_horizon_ending(episode) == FLOOR_HORIZON
        spec = episode.describe()
        environment = environment_terminal(episode)
        with pytest.raises(ValueError, match="filing"):
            _check_graded_horizon(
                spec,
                terminal_manifest(spec),
                environment._replace(horizon_ending=GRADED_HORIZON),
            )
    finally:
        await episode.close()


async def test_a_seal_that_arrives_after_the_world_was_let_go_refuses(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """A filing with no instance to answer scores nothing, and a seal must not publish it.

    The two refusals are separate. A seal that resolves to no world reached a process that never
    served this attempt; a seal that resolves to a world whose session is gone reached the right
    process too late. Both would otherwise be a zero, and a zero says the agent got no record
    right when what happened is that nobody read the table.
    """
    key = seal_of("no world")
    request = SealAttemptInput(
        attempt_id=ATTEMPT,
        seal_id=key,
        native_terminal_name="submit_filing",
        canonicalization_version=CANONICALIZATION_VERSION,
        native_arguments={"filing": "CLM-1,x"},
    )
    elsewhere = DirectoryCaptures(tmp_path / "another-machine")
    _version, nowhere = receipts_terminal(lambda _a: None, store=elsewhere)
    with pytest.raises(ApplicationError, match="no world this process opened"):
        await nowhere[0](request)

    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        _version, gone = receipts_terminal(
            lambda _a: (episode.env, "a-session-that-is-not-open"), store=elsewhere
        )
        with pytest.raises(ApplicationError, match="has been let go"):
            await gone[0](request)

        filing = _filing(ReceiptsV1Env(**_config(frozen_bundle)), 0)
        sealed, _graded = await seal_and_grade(
            episode, filing, seal_id=seal_of("worker replaced")
        )
        seals = episode.env.seals
    finally:
        await episode.close()

    grading = GradeAttemptInput(
        attempt_id=ATTEMPT,
        seal_id=seal_of("worker replaced"),
        submission_digest="d" * 64,
        canonical_submission_text=sealed.canonical_submission_text,
        environment_recovery_token=sealed.environment_recovery_token,
    )
    # A Worker that replaced the one which sealed grades what was sealed, because the record is
    # on the disk beside the fork rather than in the process that wrote it. The world is gone
    # and it is not needed: the grade is a projection of the record.
    _version, replaced = receipts_terminal(lambda _a: None, store=DirectoryCaptures(seals))
    assert (await replaced[1](grading)).score == 1.0
    # A machine that holds nothing under that key has nothing to grade, and says so rather than
    # publishing the zero an empty reading would come to.
    _version, elsewhere_now = receipts_terminal(lambda _a: None, store=elsewhere)
    with pytest.raises(ApplicationError, match="holds nothing sealed"):
        await elsewhere_now[1](grading)


async def test_an_ordinary_generation_over_this_environment_may_publish_its_score(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The honest body is what an ordinary generation stamps, and only over a real grader.

    A generation composed over the stand-in has no verdict to publish and is refused the honest
    policy where it is composed. That refusal is what the declaration in this port lifts, so it
    is checked here against this environment rather than assumed from the roster.
    """
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        spec = episode.describe()
        terminal = terminal_manifest(spec)
        composed = stream_start(
            spec, terminal, claim_hash="e" * 64, grade=RECEIPTS_GRADE
        )
        [row] = composed.dispositions
        assert row.kind == DELIVER
        assert POLICIES[row.policy_digest].exposure == HONEST

        with pytest.raises(ValueError, match="which is a stand-in"):
            stream_start(spec, terminal, claim_hash="e" * 64, grade=KERNEL_STAND_IN_GRADE)
    finally:
        await episode.close()


@pytest.mark.network
async def test_a_generation_serves_a_family_as_two_positions_and_pays_each_score(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The whole arc, through the real gateway and the real stream, over A and then B.

    Every other test here calls the Activities as functions, and production reaches them one way
    only: the stream accepts the terminal, runs the seal and the grade inside the transaction
    that accepted it, mints the acknowledgement from what they returned, and releases a body
    built under the policy the obligation resolved to.

    Two positions rather than one, because that is the arrangement the point measurement needs
    and the one the environment could not serve before: the second task is worked in a world of
    its own, and a world whose configuration is not the one this generation was started as is
    refused before it is presented. So a generation that gets to B at all is a generation whose
    two siblings came from one environment.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    bodies = [env.describe("0").instructions, env.describe("1").instructions]
    filings = [_filing(env, 0), _filing(env, 1)]
    episode = await _episode(0, frozen_bundle, tmp_path)
    opened: List[ServedEpisode] = []

    async def next_world(_attempt_id: str) -> ServedEpisode:
        """The world the second position is worked in, at the position it names."""
        world = await _episode(1, frozen_bundle, tmp_path)
        opened.append(world)
        return world

    running = False
    try:
        async with durable_client() as client:
            running = True
            environment = environment_terminal(episode)
            async with stream_worker(client, activities=environment.activities):
                await _drive_the_arc(
                    client, episode, environment, bodies, filings, next_world
                )
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")
    finally:
        for world in opened:
            await world.close()
        await episode.close()


async def _drive_the_arc(
    client: Any,
    episode: ServedEpisode,
    environment: Any,
    bodies: List[str],
    filings: List[str],
    next_world: Any,
) -> None:
    """File both siblings, and read what the generation says each one was worth."""
    spec = episode.describe()
    composed = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash="f" * 64,
        bodies=bodies,
        grade=environment.grade,
    )
    gateway = await open_gateway(
        client, episode, start=composed, environment=environment, open_episode=next_world
    )
    await gateway.close_queue()

    scores = []
    for position, filing in enumerate(filings):
        task = json.loads(await gateway.pull({}))
        assert task["kind"] == "task"
        assert task["body"] == bodies[position]
        attempt = task["attempt_id"]

        if position == 0:
            # A filing that says nothing is refused where a call's arguments are checked, which
            # is in front of the stream: nothing is claimed, nothing is sealed, and the task is
            # still the agent's to file. The refusal is the transport's, so it is asked for the
            # way the served surface asks for it rather than through the terminal below.
            for empty in ("", "   "):
                with pytest.raises(Exception, match="invalid_message"):
                    gateway.check_native_arguments(
                        gateway.terminal_tool,
                        {"attempt_id": attempt, "arguments": {"filing": empty}},
                    )
            assert [row.state for row in await _records(gateway)][0] == "active"

        answer = await gateway.terminal(
            {"attempt_id": attempt, "arguments": {"filing": filing}}
        )
        ack = json.loads(answer)
        assert ack["kind"] == "seal_ack"
        assert ack["canonicalization_version"] == CANONICALIZATION_VERSION
        assert "score" not in answer
        assert "PASS" not in answer

        payload = json.loads(await gateway.pull({}))
        assert payload["kind"] == "payload"
        assert payload["body"] == f"attempt {attempt}\nscore 1\nsolved 1"
        scores.append(attempt)

    assert json.loads(await gateway.pull({}))["kind"] == "done"

    rows = await _records(gateway)
    assert [row.attempt_id for row in rows] == scores
    assert [row.score for row in rows] == [1.0, 1.0]
    assert [row.decode_state for row in rows] == ["decoded", "decoded"]
    assert all(row.final_failure is None for row in rows)
    assert all(row.terminal_tool == "submit_filing" for row in rows)
    await gateway.aclose()


@pytest.mark.network
async def test_an_attempt_that_files_nothing_takes_the_floor_and_leaves_no_seal(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The one ending this environment has for an attempt nobody filed, and what it records.

    Nothing an agent can call over this environment spends the step budget: the filing is the
    terminal and the reserved abort is not served, so an attempt that never files stays where it
    is until a controller or a deadline ends it. That ending is the floor, and the floor is not a
    seal: it writes a failure and a zero, and it writes no submission, no decode state, no seal
    ordinal, no acknowledgement and no cells, because nothing was read and nothing was rendered.

    The v1 path reports zero for the same ending, through its own no-filing reason code. The
    numbers agree and the records do not, which is the honest way round: one of them is a filing
    that said nothing and the other is no filing at all.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    episode = await _episode(0, frozen_bundle, tmp_path)
    running = False
    try:
        async with durable_client() as client:
            running = True
            environment = environment_terminal(episode)
            async with stream_worker(client, activities=environment.activities):
                await _leave_it_unfiled(client, episode, env)
    except Exception as error:  # noqa: BLE001 - re-raised below unless the service never came up
        if running:
            raise
        pytest.skip(f"the durable service is unavailable: {error}")
    finally:
        await episode.close()


async def _leave_it_unfiled(client: Any, episode: ServedEpisode, env: ReceiptsV1Env) -> None:
    """Present one task, file nothing, end the attempt the way a controller ends one."""
    spec = episode.describe()
    composed = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash="a" * 64,
        bodies=[env.describe("0").instructions],
        grade=environment_terminal(episode).grade,
    )
    gateway = await open_gateway(
        client, episode, start=composed, environment=environment_terminal(episode)
    )
    await gateway.close_queue()

    attempt = json.loads(await gateway.pull({}))["attempt_id"]
    [before] = await _records(gateway)
    assert before.state == "active"
    sealed_before = _seal_records(episode.env.seals)

    ended = await gateway._stream.finalize(
        FinalizeRequest(
            request_id=secrets.token_hex(16), attempt_id=attempt, reason=ABANDONED
        )
    )
    assert ended.reason == ABANDONED

    [row] = await _records(gateway)
    assert row.state == "final_failed"
    assert row.final_failure == ABANDONED
    assert row.score == 0.0
    assert row.terminal_tool is None
    assert row.terminal_source is None
    assert row.submission_digest is None
    assert row.decode_state is None
    assert row.seal_ordinal is None
    assert row.ack_delivered is False
    # And this ending rendered nothing and kept nothing: no seal record appeared while it was
    # made, because no filing reached a seal and a seal is what renders.
    assert _seal_records(episode.env.seals) == sealed_before
    await gateway.aclose()


def _seal_records(seals: Path) -> set:
    """The seals this environment's store holds, by name."""
    return {held.name for held in seals.iterdir()} if seals.is_dir() else set()


async def _records(gateway: Any) -> Any:
    """The generation's own rows, which are the rows a reader of the run answers with."""
    from shogym.serve.protocol_v2.kernel.workflow import StreamWorkflow

    return list(await gateway._stream.handle.query(StreamWorkflow.attempt_records))
