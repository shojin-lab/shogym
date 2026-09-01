"""End-to-end: serve one side of an admitted instance and file an answer.

``submit_filing`` is the env's score terminal: the call validates its args, seals
the episode and runs ``finalize``, so a filing is graded once and on an episode
that can no longer be continued. Nothing here reaches the network.

The property this file exists for is the last one: the terminal returns no verdict.
What one graded receipt is worth is the quantity the environment exists to measure,
so an environment that handed the grade back at the terminal would put a receipt in
every arm, including the arm that is supposed to be empty.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts import streams
from shogym.envs.receipts.env_v1 import ReceiptsV1Env
from shogym.envs.receipts.generators.ledger import GENERATOR
from shogym.envs.receipts.protocol import option_mentions
from shogym.serve import ServedEpisode


@pytest.fixture(scope="module")
def frozen_bundle(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """One admission bundle that actually verifies, shared by the module.

    A bank filled by the registered bars, a screen artifact and a review pack. The
    bank and its population are real. The screen's score rows and the pack's render
    bytes are SYNTHETIC: structurally valid, and standing in for a pilot nobody ran
    here and renders nobody read. They are written rather than asserted because the
    production open path recomputes everything mechanical about them, so a fixture
    that only claimed the stages had happened would be testing the claim, not the path.
    """
    room = tmp_path_factory.mktemp("bundles")
    bank, held = bank_mod.materialized(GENERATOR, streams.new_master_key(), 2)
    outcomes = room / "screen.json"
    outcomes.write_text(json.dumps(_screen_artifact()), encoding="utf-8")
    pack = _review_pack(room, held, bank)
    built = bundle_mod.build(room / "bundles", GENERATOR, bank, outcomes, pack)
    assert bundle_mod.verify(built, GENERATOR).problems == ()
    return built.root


def _screen_artifact(pairs: int = 40) -> dict:
    """A pilot run and the bars it is judged against, as one artifact.

    Three numbers say what was measured; they do not say what it was measured on or
    what it had to clear, so both travel with the rows.
    """
    return {
        "family": GENERATOR.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
            for i in range(pairs)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
        "floor": 0.0, "floor_rule": "drop",
        "candidates_screened": 1, "selection_note": "",
    }


def _review_pack(room: Path, held: bank_mod.Population, bank: bank_mod.Bank) -> Path:
    """A pack that covers what the family declares, with artifacts of a plausible size.

    Every surface, every option of every axis, every filing shape, every row count the
    bank holds, and a counterfactual render. The bytes stand in for what a reviewer
    actually read; what is being exercised is the coverage rule, not the reading.
    """
    from shogym.envs.receipts.checks import FILING_CLASSES
    from shogym.envs.receipts.review import required_coverage

    coverage = required_coverage(
        GENERATOR, FILING_CLASSES,
        [i.a.n_rows for i in held.instances] + [i.b.n_rows for i in held.instances],
    )
    envelope_size = min(i.envelope.size for i in held.instances)
    folder = room / "renders"
    folder.mkdir(exist_ok=True)
    renders = []
    for index, (category, key) in enumerate(coverage.required):
        kind = "task" if category == "surface" else "cell"
        floor = 400 if kind == "task" else envelope_size
        artifact = folder / f"{index:03d}.txt"
        artifact.write_text("R" * (floor + 8), encoding="utf-8")
        renders.append({
            "category": category, "key": key, "kind": kind,
            "path": f"renders/{artifact.name}",
        })
    pack = room / "review-pack.json"
    pack.write_text(
        json.dumps({
            "reviewer": "test",
            "checklist": ["surface templates", "every option", "filing shapes"],
            "seeds": [0, 1],
            "family": GENERATOR.name,
            "bank": bank_mod.bank_identity(bank),
            "renders": renders,
        }),
        encoding="utf-8",
    )
    return pack


def _config(frozen_bundle: Path, side: str = "a") -> dict:
    return {"bundle": str(frozen_bundle), "side": side}


def _feedback(episode: ServedEpisode) -> dict:
    return {item["name"]: item["value"] for item in episode.terminal_feedback}


async def _episode(task: int, frozen_bundle: Path, tmp_path: Path) -> ServedEpisode:
    """A served episode whose durable records live under the test's own directory.

    A score terminal builds its finalization store beside the trace, so a test that
    names no trace shares the user's cache and pays a scan of everything in it.
    """
    return await ServedEpisode.start(
        "receipts_v1",
        task=task,
        env_config=_config(frozen_bundle),
        trace_path=tmp_path / "run.jsonl",
    )


def _filing(env: ReceiptsV1Env, ordinal: int, side: str = "a") -> str:
    task = env.instance(ordinal).side(side)
    return "\n".join(
        f"{i},{v}" for i, v in zip(GENERATOR.row_identifiers(task.table), task.key)
    )


async def test_describe_surfaces_the_schedule_and_the_filing_tool(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        spec = episode.describe()
        by_name = {t.name: t for t in spec.tools}
        # Exactly these two. A subset check is not a pinned surface: an extra
        # advertised tool would pass the assertion that exists to catch one.
        assert set(by_name) == {"submit_filing", "terminate"}
        assert by_name["submit_filing"].terminal_kind == "score"
        assert by_name["terminate"].terminal_kind == "abort"
        assert "SCHEDULE" in spec.instructions
        assert "POLICY EXTRACT" in spec.instructions
    finally:
        await episode.close()


async def test_the_task_spec_carries_no_convention_and_no_answer(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Everything that could reproduce the hidden rule stays controller-side."""
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env.instance(env._ordinals[0]).ordinal
    instance = env.instance(ordinal)
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        published = json.dumps(episode.describe().model_dump())
        # the rule is not stated
        assert "HOUSE CONVENTIONS" not in published
        assert option_mentions(GENERATOR.AXES, published) == []
        # nor is any record's answer: the band table names every band, which the task
        # needs, but no record is ever paired with the one it takes
        identifiers = GENERATOR.row_identifiers(instance.a.table)
        checked = 0
        for identifier, band in zip(identifiers, instance.a.key):
            # One option on the `missing` axis is the empty band, and "CLM-1051,"
            # with nothing after it is just the record's own line in the schedule.
            # There is no answer to leak on those rows, so there is nothing to look
            # for either.
            if not band:
                continue
            checked += 1
            assert f"{identifier},{band}" not in published
            assert f"{identifier}: {band}" not in published
        assert checked >= len(identifiers) - 4
    finally:
        await episode.close()


async def test_a_correct_filing_scores_one_and_the_content_carries_no_verdict(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        result = await episode.call("submit_filing", {"filing": _filing(env, ordinal)})
        assert result.terminated
        payload = json.loads(result.content)
        assert payload["filed"] is True
        assert set(payload) == {"filed", "rows", "finalize_error"}

        fb = _feedback(episode)
        assert fb["component_score"] == 1.0
        assert fb["solved"] is True
        assert fb["rows_omitted"] == 0.0
    finally:
        await episode.close()


async def test_the_whole_terminal_result_carries_no_verdict_on_either_channel(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Every byte the terminal hands back, over a real MCP client.

    A tool result has two channels and the content is only one of them. The serve
    layer attaches episode feedback to the result's `_meta["shogym/feedback"]`
    sidecar, so an env that returns no verdict in its content and takes the default
    would still hand the agent's own process the score. This env declares
    `inband_terminal_feedback = False`, and this asserts the whole result rather than
    half of it: content, metadata, and no scored number anywhere in either.
    """
    from fastmcp import Client

    from shogym.serve.server import build_server

    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        async with Client(build_server(episode)) as client:
            result = await client.call_tool(
                "submit_filing", {"filing": _filing(env, ordinal)}
            )
            content = json.loads(result.content[0].text)  # type: ignore[union-attr]
            assert set(content) == {"filed", "rows", "finalize_error"}
            meta = dict(result.meta or {})
            assert meta == {"shogym/terminate": True}
            # The whole result object, not the two fields a reader thinks of: a
            # structured payload beside the text would be a third place a score
            # could ride out on, and an error flag is part of what the agent reads.
            assert result.structured_content is None
            assert result.is_error is False
            # And the whole result, as bytes, names nothing evaluative.
            everything = json.dumps(
                {
                    "content": [block.text for block in result.content],  # type: ignore[union-attr]
                    "meta": meta,
                }
            )
            for word in ("component_score", "solved", "rows_filed", "rows_omitted",
                         "grade_error", "1.0"):
                assert word not in everything
        # The controller still has the number: it is verified, traced and reported.
        assert _feedback(episode)["component_score"] == 1.0
    finally:
        await episode.close()


async def test_a_partial_filing_scores_the_fraction_it_got_right(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    lines = _filing(env, ordinal).splitlines()
    spoiled = [lines[0]] + [f"{line.split(',')[0]},Wrong" for line in lines[1:]]
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        await episode.call("submit_filing", {"filing": "\n".join(spoiled)})
        fb = _feedback(episode)
        # the component score is rounded to six places, which is registered, so the
        # expected value is rounded the same way rather than compared loosely
        assert fb["component_score"] == round(1.0 / len(lines), 6)
        assert fb["solved"] is False
    finally:
        await episode.close()


async def test_a_filing_carrying_bytes_the_serializer_refuses_still_seals(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """The tool takes any string, so the seal has to survive any string.

    A value the fixed-width serializer would refuse used to raise out of `finalize`,
    which fails the episode closed and commits no fork at all: an agent that preferred
    not to be receipted filed one accented band name. The parser folds to printable
    ASCII, so the filing scores, the fork commits, and the terminal is ordinary.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    lines = _filing(env, ordinal).splitlines()
    identifier = lines[0].split(",")[0]
    lines[0] = f"{identifier},Rout\u00efne \x1b[7m\x00"
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        result = await episode.call("submit_filing", {"filing": "\n".join(lines)})
        payload = json.loads(result.content)
        assert payload["finalize_error"] is False
        assert payload["filed"] is True
        fb = _feedback(episode)
        assert "grade_error" not in fb
        # One row was spoiled and the rest were right, so the score is the fraction.
        assert fb["component_score"] == round((len(lines) - 1) / len(lines), 6)
    finally:
        await episode.close()


@pytest.mark.parametrize("value", ["\ud800", "\udfff"], ids=["high", "low"])
async def test_a_lone_surrogate_seals_and_commits_a_fork(
    frozen_bundle: Path, tmp_path: Path, value: str
) -> None:
    """A JSON string can carry a code point UTF-8 cannot encode.

    The value folds to a printable character in the parser and renders three ordinary
    cells, but the fork was keyed by a digest taken before the parser ran, and strict
    UTF-8 refused it there: the seal raised, the episode failed closed and the link
    kept no cell at all. The key is total over Python strings now, so the folding is
    the only thing that decides what a surrogate becomes.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    lines = _filing(env, ordinal).splitlines()
    lines[0] = f"{lines[0].split(',')[0]},{value}"
    forks = frozen_bundle.parent / "forks"
    before = set(forks.rglob("fork-*.json")) if forks.is_dir() else set()
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        result = await episode.call("submit_filing", {"filing": "\n".join(lines)})
        payload = json.loads(result.content)
        assert payload["finalize_error"] is False
        assert payload["filed"] is True
        fb = _feedback(episode)
        assert "grade_error" not in fb
        assert fb["component_score"] == round((len(lines) - 1) / len(lines), 6)
    finally:
        await episode.close()
    assert len(set(forks.rglob("fork-*.json")) - before) == 1


async def test_a_failure_inside_finalize_leaves_grade_error_and_no_fork(
    frozen_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The act this environment owns is atomic, and its failure carries no score.

    Render, envelope check, hash and commit are one act. When it fails there is no
    committed fork, so a scalar would say the link produced one it does not have.
    """
    from shogym.envs.receipts import bank as bank_mod

    def refuse(*args: object, **kwargs: object) -> None:
        raise ValueError("the renderer dropped a row")

    monkeypatch.setattr(bank_mod, "fork_for", refuse)
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    forks = frozen_bundle.parent / "forks"
    before = set(forks.rglob("fork-*.json")) if forks.is_dir() else set()
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        result = await episode.call("submit_filing", {"filing": _filing(env, ordinal)})
        payload = json.loads(result.content)
        assert payload["finalize_error"] is True
        fb = _feedback(episode)
        assert fb == {"grade_error": True}
    finally:
        await episode.close()
    assert set(forks.rglob("fork-*.json")) == before


async def test_a_core_verifier_failure_leaves_a_committed_fork_and_no_feedback(
    frozen_bundle: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The other failure shape, pinned rather than described.

    `finalize` returns, the fork is committed and valid, and the core verifier fails
    afterwards. Core's contract on that path is to publish no feedback at all, so the
    episode carries an EMPTY feedback list beside a committed fork: `finalize_error`
    means the transaction failed, not that there is no fork and not that `grade_error`
    is present.
    """
    def refuse(*args: object, **kwargs: object) -> None:
        raise ValueError("the verifier could not score this")

    monkeypatch.setattr(ReceiptsV1Env, "_verify", refuse)
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    filed = _filing(env, ordinal) + "\nZZZ-9001,Routine"
    forks = frozen_bundle.parent / "forks"
    before = set(forks.rglob("fork-*.json")) if forks.is_dir() else set()
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        result = await episode.call("submit_filing", {"filing": filed})
        payload = json.loads(result.content)
        assert payload["finalize_error"] is True
        assert episode.terminal_feedback == []
    finally:
        await episode.close()
    # The fork the env committed before the verifier ran is still there, and it is one.
    assert len(set(forks.rglob("fork-*.json")) - before) == 1


async def test_an_unreadable_filing_is_reason_coded_not_a_low_score(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    episode = await _episode(1, frozen_bundle, tmp_path)
    try:
        await episode.call("submit_filing", {"filing": "I could not work out the rule"})
        fb = _feedback(episode)
        assert fb["component_score"] == 0.0
        assert fb["no_filing"] == "no_known_identifier"
        assert "rows_filed" not in fb
    finally:
        await episode.close()


async def test_terminating_without_filing_records_that_none_arrived(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    episode = await _episode(1, frozen_bundle, tmp_path)
    try:
        result = await episode.call("terminate", {})
        assert result.terminated
        fb = _feedback(episode)
        assert fb["component_score"] == 0.0
        assert fb["no_filing"] == "unreadable"
    finally:
        await episode.close()


async def test_side_b_files_and_seals_through_the_same_path(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """B is a measured path, so the serve path on B is measured too.

    The env seals either side, and an author's mistake in a cell only B renders would
    otherwise surface as a branch finalization failure at seal time, which the chain
    records as an outcome and which has nothing to do with what the learner did.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle, "b"))
    ordinal = env._ordinals[0]
    episode = await ServedEpisode.start(
        "receipts_v1",
        task=0,
        env_config=_config(frozen_bundle, "b"),
        trace_path=tmp_path / "run.jsonl",
    )
    try:
        result = await episode.call(
            "submit_filing", {"filing": _filing(env, ordinal, "b")}
        )
        assert result.terminated
        payload = json.loads(result.content)
        assert payload["filed"] is True
        assert payload["finalize_error"] is False
        fb = _feedback(episode)
        assert fb["component_score"] == 1.0
        assert fb["solved"] is True
    finally:
        await episode.close()


async def test_a_second_seal_replays_the_fork_it_wrote_outside_the_bundle(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """One filing, one render, and the bytes live where the bundle is not.

    A bundle is frozen at its digest, so anything written inside one would make it
    disagree with its own manifest. And rendering again on a retry is how two branches
    of one fork come to hold different bytes, so the second seal has to read back the
    first one's file rather than remake it.
    """
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    filed = _filing(env, ordinal) + "\nZZZ-0001,Routine"
    forks = frozen_bundle.parent / "forks"

    def written() -> set[Path]:
        return set(forks.rglob("fork-*.json")) if forks.is_dir() else set()

    before = written()
    stamp: int | None = None
    for attempt in range(2):
        episode = await _episode(0, frozen_bundle, tmp_path / f"seal-{attempt}")
        try:
            await episode.call("submit_filing", {"filing": filed})
            assert _feedback(episode)["component_score"] == 1.0
        finally:
            await episode.close()
        fresh = sorted(written() - before)
        assert len(fresh) == 1
        if stamp is None:
            stamp = fresh[0].stat().st_mtime_ns
        else:
            # The second seal read the first one's bytes back rather than remaking them.
            assert fresh[0].stat().st_mtime_ns == stamp
            # Outside the bundle: beside it, never inside the addressed directory.
            assert frozen_bundle not in fresh[0].parents


def test_the_two_sides_are_two_envs_over_one_convention(frozen_bundle: Path) -> None:
    side_a = ReceiptsV1Env(**_config(frozen_bundle, "a"))
    side_b = ReceiptsV1Env(**_config(frozen_bundle, "b"))
    ordinal = side_a._ordinals[0]
    assert side_a.instance(ordinal).convention == side_b.instance(ordinal).convention
    assert side_a.instance(ordinal).a.task_id != side_b.instance(ordinal).b.task_id
    assert side_a.dealable


def test_the_env_refuses_a_side_that_is_not_a_or_b(frozen_bundle: Path) -> None:
    with pytest.raises(ValueError, match="sides a and b"):
        ReceiptsV1Env(bundle=str(frozen_bundle), side="c")


async def test_the_terminal_reports_the_rows_the_filing_named(
    frozen_bundle: Path, tmp_path: Path
) -> None:
    """Not the size of the table. The canonical form allocates one value per printed
    row including the omissions, so its length would tell an agent it had filed rows
    it never mentioned."""
    env = ReceiptsV1Env(**_config(frozen_bundle))
    ordinal = env._ordinals[0]
    one_line = _filing(env, ordinal).splitlines()[0]
    episode = await _episode(0, frozen_bundle, tmp_path)
    try:
        result = await episode.call("submit_filing", {"filing": one_line})
        payload = json.loads(result.content)
        assert payload["rows"] == 1
        feedback = _feedback(episode)
        assert feedback["rows_filed"] == 1.0
        assert feedback["rows_omitted"] == float(env.instance(ordinal).side("a").n_rows - 1)
    finally:
        await episode.close()


def test_a_named_bundle_that_is_not_there_fails_loudly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="there is no bundle at"):
        ReceiptsV1Env(bundle=str(tmp_path / "nothing"))


def test_the_environment_refuses_a_bank_that_was_never_bundled(
    tmp_path: Path,
) -> None:
    """Gates passing is necessary and is not admission.

    A bank with no room screen and no recorded human read may not be served: a chain
    run on one would have no defined family treatment, and the failure would be
    invisible because every published field would look ordinary. A bank is not a
    bundle, so there is nothing for production to open.
    """
    path = tmp_path / "raw.json"
    bank_mod.save_bank(
        bank_mod.materialize(GENERATOR, streams.new_master_key(), 1), path
    )
    with pytest.raises(ValueError, match="there is no bundle at"):
        ReceiptsV1Env(bundle=str(path.parent))
    # and the development environment says so in its own name rather than by a flag
    from shogym.envs.receipts.env_v1 import ReceiptsDevEnv

    dev = ReceiptsDevEnv(bank=str(path))
    assert not dev.dealable
    assert dev.num_tasks == 1


def test_the_published_task_id_is_the_opaque_one(frozen_bundle: Path) -> None:
    """The selector stays controller-side; what goes out encodes no ordinal."""
    env = ReceiptsV1Env(bundle=str(frozen_bundle))
    published = env.describe("0").task_id
    assert published == env.instance(env._ordinals[0]).a.task_id
    assert published != "0"
    assert len(published) == 16
    assert set(published) <= set("0123456789abcdef")
