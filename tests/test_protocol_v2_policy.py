"""What a generation is allowed to tell its agent, and how both mistakes are made loud.

Two mistakes are possible and each one is silent by nature. An ordinary run can end up blinded
because nobody said what to deliver, and an experiment can end up unblinded for the same reason.
Neither is visible in a transcript: a body that says nothing and a body that says the score both
look like a payload arriving. So the tests here are about refusals and records rather than about
bytes going out. A generation that has not resolved what it delivers is not created, a body built
under something other than what was resolved does not become an acknowledgement, and every run
carries the policy it served under where a reader can find it.

The stream tests drive a real workflow on Temporal's time-skipping environment, like the rest of
the kernel's, and skip when that server is not there.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from hashlib import sha256
from pathlib import Path
from typing import Any, AsyncIterator, Optional

import pytest

pytest.importorskip("temporalio")

import pytest_asyncio  # noqa: E402
from temporalio import activity  # noqa: E402
from temporalio.client import (  # noqa: E402
    WorkflowFailureError,
    WorkflowHistory,
    WorkflowUpdateFailedError,
)
from temporalio.converter import default as default_converter  # noqa: E402
from temporalio.exceptions import ApplicationError  # noqa: E402
from temporalio.testing import WorkflowEnvironment  # noqa: E402

from shogym.serve.protocol_v2 import (  # noqa: E402
    Payload,
    PullRequest,
    TerminalMetadata,
    visible_bytes,
)
from shogym.serve.protocol_v2.kernel import (  # noqa: E402
    SEAL_UNUSABLE,
    STREAM_TASK_QUEUE,
    ConsumerClaim,
    GeneratePayloadBundleInput,
    OfferedMessage,
    SealRequest,
    StreamHandle,
    StreamStart,
    StreamWorkflow,
    TaskItem,
    TerminalTool,
    assignments_for,
    generate_payload_bundle_activity,
    grade_attempt_activity,
    protocol_error_code,
    seal_attempt_activity,
    start_stream,
    stream_replayer,
    stream_worker,
)
from shogym.serve.protocol_v2.kernel.activities import (  # noqa: E402
    GENERATE_PAYLOAD_BUNDLE,
    GRADE_ATTEMPT,
)
from shogym.serve.protocol_v2.kernel.messages import (  # noqa: E402
    SEAL_RENDERER,
    GradeAttemptInput,
    GradeAttemptResult,
    configuration_hash,
)
from shogym.serve.protocol_v2.policy import (  # noqa: E402
    BLINDED_RECEIPT_V1,
    DELIVER,
    EXPERIMENT,
    HONEST_V1,
    KERNEL_STAND_IN_GRADE,
    LEGACY,
    LEGACY_PLACEHOLDER_V1,
    HONEST_V1_DIGEST,
    ORDINARY,
    PLACEBO_RECEIPT_V1,
    PLATFORM_DEFAULT,
    POLICIES,
    REGISTERED,
    SELECTABLE,
    SINGLETON_SLOT,
    WITHHOLD,
    GradeIdentity,
    MatchedFamily,
    PayloadDisposition,
    PolicyProvenance,
    PolicyViolation,
    PublicGrade,
    PublishedNumber,
    check_dispositions,
    disposition_key,
    policy_digest,
    policy_name_of,
    policy_preimage,
    published_grade,
    render_body,
    roster_digest,
)

CLAIM_HASH = "d" * 64
CONSUMER = ConsumerClaim(consumer_id="harness-1", claim_hash=CLAIM_HASH)
TRANSCRIPT_BLOB = "e" * 64
PROVIDER_TURN_BLOB = "f" * 64
CHECKPOINT_BLOB = "9" * 64
FILING = {"answer": "42"}
TASK_BODY = "file the report"

# The generation this file drives, with every public identifier fixed before it serves anything.
ATTEMPT = "00000000000000000000000000000100"
DIGEST = "be479ed7d985a6fd522c999eab03a639486f7c7762d1c0479e07b78b69aa4d91"

# The generation the checked-in fixture history recorded. The name is the one that history was
# written under, because a history is replayed as the workflow it was.
RECORDED_BEFORE_POLICIES = "stream/recorded-before-policies/1"

# What the legacy configuration hash formula produced before a generation carried a policy. A
# run recorded then presents this value when its owner resumes it, so the formula it was taken
# under has to keep producing it for a start that declares no profile.
LEGACY_CONFIGURATION = "e884f10027a18f613045200cf29bdd02a00c0764944b6fd7ac8b825a3fe2e1c5"

# A grader that says its number is the environment's own, which is what an honest body needs, and
# that declares the one number it publishes beside that score, with the domain that number lies
# in.
GUESSES_USED = PublishedNumber(name="guesses_used", minimum=0, maximum=6)
# And a measure that is a fraction, declared to the resolution it is a measure at. The range
# leaves every digit a double holds available; how many of them mean anything is the other half
# of the domain and is declared here with the first.
LEDGER_FRACTION = PublishedNumber(
    name="ledger_fraction", minimum=0, maximum=1, places=4
)
REAL_GRADE = GradeIdentity(
    grader_id="double-grade",
    grader_version="1",
    stand_in=False,
    score_component="answer",
    # The headline carries the same half of a domain the components do. This grader scores in
    # quarters, so two places is the resolution it declares and everything below them is room.
    score_places=2,
    public_components=(GUESSES_USED, LEDGER_FRACTION),
)
# The same grader, one version on. Two runs whose bodies published different graders are not one
# generation, so this is what the identity is checked against.
REAL_GRADE_TWO = replace(REAL_GRADE, grader_version="2")


def oid(value: int) -> str:
    return f"{value:032x}"


def checking(rows: Any, **declared: Any) -> None:
    """Check a roster with the authority its profile is entitled to, unless one is given.

    Most of these tests are about coverage, shape or the profile matrix rather than about who
    was entitled to the profile, so the registration or the stamp comes from the same helper the
    builder uses. A test whose subject is the authority passes its own.
    """
    declared.setdefault("provenance", entitling(declared["profile"], list(rows)))
    check_dispositions(list(rows), **declared)


def entitling(profile: str, rows: Any) -> Optional[PolicyProvenance]:
    """The authority a generation under ``profile`` is created with.

    A profile is a word until something stands behind it, so every start built here carries the
    registration or the stamp that entitles it to the one it claims, exactly as the builder
    produces. A test whose subject is the authority passes its own.
    """
    if profile == LEGACY:
        return None
    if profile == EXPERIMENT:
        return PolicyProvenance(
            authority=REGISTERED,
            roster_digest=roster_digest(list(rows)),
            experiment_id="the_subject_of_this_run",
        )
    return PolicyProvenance(
        authority=PLATFORM_DEFAULT,
        roster_digest=roster_digest(list(rows)),
        descriptor_digest=HONEST_V1_DIGEST,
    )


def make_start(
    *,
    profile: str = LEGACY,
    grade: Optional[GradeIdentity] = None,
    dispositions: Any = (),
    bodies: Any = (TASK_BODY,),
    provenance: Any = "the one this profile is entitled to",
    families: Any = (),
) -> StreamStart:
    """One generation, composed here so a test can say exactly what it resolved to."""
    tasks = [
        TaskItem(
            task_position=index,
            attempt_id=oid(0x100 + index * 4),
            task_message_id=oid(0x101 + index * 4),
            ack_message_id=oid(0x102 + index * 4),
            payload_position=index,
            payload_message_id=oid(0x103 + index * 4),
            body=body,
        )
        for index, body in enumerate(bodies)
    ]
    return StreamStart(
        configuration_hash="c" * 64,
        consumer_claim_hash=CLAIM_HASH,
        initial_cursor=oid(1),
        done_message_id=oid(2),
        id_key_hex="ab" * 32,
        hidden_execution_id="execution-1",
        canonicalization_version="kernel.1",
        terminal_tool=TerminalTool(
            public_tool_name="submit", native_terminal_name="submit", argument_names=["answer"]
        ),
        tasks=tasks,
        profile=profile,
        grade=grade,
        dispositions=list(dispositions),
        provenance=(
            entitling(profile, dispositions)
            if provenance == "the one this profile is entitled to"
            else provenance
        ),
        families=list(families),
    )


def delivering(
    policy: Any, *, position: int = 0, slot: str = SINGLETON_SLOT
) -> PayloadDisposition:
    """One obligation an experiment registered a delivery under ``policy`` for."""
    return PayloadDisposition(
        attempt_id=oid(0x100 + position * 4),
        payload_position=position,
        branch_slot=slot,
        kind=DELIVER,
        policy_digest=policy_digest(policy),
        cell=policy.cells[0],
    )


def stamped(*, position: int = 0) -> PayloadDisposition:
    """The row an ordinary generation converts one obligation's silence into."""
    return replace(
        delivering(HONEST_V1, position=position), resolution_source=PLATFORM_DEFAULT
    )


# The descriptor.


def test_a_policy_is_named_by_the_bytes_that_say_what_it_is() -> None:
    """The digest names a preimage, and the preimage says what a body may contain.

    A name is mutable and a hash with nothing behind it says only that something was hashed. So
    the preimage is canonical bytes carrying the renderer this policy names, its version, the
    cells it declares and the projection its body is drawn from, and the digest is that.
    """
    preimage = json.loads(policy_preimage(HONEST_V1).decode("utf-8"))
    assert preimage["policy_name"] == "honest-v1"
    assert preimage["renderer_id"] == "kernel-honest-1"
    assert preimage["renderer_version"] == "1"
    assert preimage["exposure"] == "honest"
    assert preimage["cells"] == ["honest"]
    assert [field["type"] for field in preimage["projection"]] == [
        "public_attempt_id",
        "unit_interval",
        "named_finite_numbers",
    ]
    # How a number becomes text is declared too. A body that says a score is a claim about the
    # number the seal committed, and a format is what decides whether it is still that number.
    assert preimage["number_format"] == "shortest_roundtrip"
    assert POLICIES[policy_digest(HONEST_V1)] is HONEST_V1
    assert len({policy_digest(policy) for policy in POLICIES.values()}) == len(POLICIES)

    # The placeholder is how a recorded history is read and never something a new run may ask
    # for. The two concealing policies are ones a run may ask for, and only an experiment may.
    assert LEGACY_PLACEHOLDER_V1.policy_name not in SELECTABLE
    assert set(SELECTABLE) == {"honest-v1", "blinded-receipt-v1", "placebo-receipt-v1"}
    # The placebo is a record of its own rather than a second name for the concealed cell: two
    # registrations are what a family's byte count is a check over, and one is not.
    assert policy_digest(PLACEBO_RECEIPT_V1) != policy_digest(BLINDED_RECEIPT_V1)
    assert PLACEBO_RECEIPT_V1.cells == ("placebo",)
    assert policy_name_of(None) == "legacy-placeholder-v1"
    assert policy_name_of("") == "legacy-placeholder-v1"
    assert policy_name_of("0" * 64).startswith("unknown-policy:")


def test_what_a_generation_delivers_is_part_of_what_it_is() -> None:
    """The policy digest is inside the identity a resume is held to, and legacy stays legacy.

    A run that changed what its bodies were allowed to say is a different generation, so the
    dispositions are folded into the configuration hash. A run recorded before any of this
    carried no dispositions and hashed a fixed set of keys, and it still hashes exactly those:
    a formula that grew a key would refuse every resume of every generation recorded under it.
    """
    honest = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    blinded = make_start(
        profile=EXPERIMENT, grade=REAL_GRADE, dispositions=[delivering(BLINDED_RECEIPT_V1)]
    )
    assert configuration_hash(honest) != configuration_hash(blinded)
    assert configuration_hash(honest) != configuration_hash(make_start())
    assert configuration_hash(honest) == configuration_hash(replace(honest))

    # The grader the generation was built over is in it too: honesty is a claim about a number,
    # and two runs whose bodies published different graders are not one generation. So is the
    # roster of what that grader may publish beside the score, because it is the whole of what a
    # body may name.
    assert configuration_hash(honest) != configuration_hash(replace(honest, grade=REAL_GRADE_TWO))
    assert configuration_hash(honest) != configuration_hash(
        replace(honest, grade=replace(REAL_GRADE, public_components=()))
    )

    # And how fine the headline is, which is the half of the score's domain the environment
    # declares. The same grader scoring in whole numbers and scoring to four decimals admits
    # different verdicts, so the two are different generations and a resume of one is not a
    # resume of the other.
    coarse = replace(honest, grade=replace(REAL_GRADE, score_places=0))
    fine = replace(honest, grade=replace(REAL_GRADE, score_places=4))
    assert configuration_hash(coarse) != configuration_hash(fine)
    assert configuration_hash(honest) not in {
        configuration_hash(coarse),
        configuration_hash(fine),
    }
    assert configuration_hash(make_start()) == LEGACY_CONFIGURATION


# Resolution, and both loud mistakes.


def test_an_ordinary_run_is_stamped_honest_before_the_generation_exists() -> None:
    """Omission is converted where the generation is built, so no omission is left to inherit."""
    from shogym.serve.protocol_v2.gateway import stream_start, terminal_manifest

    spec = _spec()
    start = stream_start(
        spec, terminal_manifest(spec), claim_hash=CLAIM_HASH, bodies=["one"], grade=REAL_GRADE
    )
    assert start.profile == ORDINARY
    [row] = start.dispositions
    assert row.kind == DELIVER
    assert row.policy_digest == policy_digest(HONEST_V1)
    assert row.cell == "honest"
    assert row.resolution_source == PLATFORM_DEFAULT
    assert row.branch_slot == SINGLETON_SLOT


def test_an_ordinary_run_that_delivers_nothing_says_so_rather_than_saying_nothing() -> None:
    """A position with no obligation is withheld under a reason, which is a row like any other."""
    from shogym.serve.protocol_v2.gateway import stream_start, terminal_manifest

    spec = _spec()
    start = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        bodies=["one", "two"],
        without_payload=(1,),
        grade=REAL_GRADE,
    )
    kinds = [row.kind for row in start.dispositions]
    assert kinds == [DELIVER, WITHHOLD]
    assert start.dispositions[1].reason == "roster_creates_no_obligation"
    assert all(row.resolution_source == PLATFORM_DEFAULT for row in start.dispositions)


def test_an_ordinary_run_cannot_be_handed_a_registered_policy() -> None:
    """Concealment is something a run declares itself to be doing, not an argument it takes."""
    from shogym.serve.protocol_v2.gateway import stream_start, terminal_manifest

    spec = _spec()
    with pytest.raises(ValueError, match="belongs to an experiment"):
        stream_start(
            spec,
            terminal_manifest(spec),
            claim_hash=CLAIM_HASH,
            grade=REAL_GRADE,
            dispositions=lambda rows: [
                PayloadDisposition(
                    attempt_id=rows[0].attempt_id,
                    payload_position=0,
                    kind=DELIVER,
                    policy_digest=policy_digest(BLINDED_RECEIPT_V1),
                    cell="graded",
                )
            ],
        )


def test_an_experiment_with_a_position_nobody_covered_is_not_created() -> None:
    """There is no default under the experiment profile, so an omission is a refusal."""
    from shogym.serve.protocol_v2.gateway import stream_start, terminal_manifest

    spec = _spec()
    with pytest.raises(ValueError, match="has no default policy"):
        stream_start(
            spec,
            terminal_manifest(spec),
            claim_hash=CLAIM_HASH,
            profile=EXPERIMENT,
            experiment="the_subject_of_this_run",
            grade=REAL_GRADE,
        )
    with pytest.raises(ValueError, match="has no disposition"):
        stream_start(
            spec,
            terminal_manifest(spec),
            claim_hash=CLAIM_HASH,
            bodies=["one", "two"],
            profile=EXPERIMENT,
            experiment="the_subject_of_this_run",
            grade=REAL_GRADE,
            dispositions=lambda rows: [
                PayloadDisposition(
                    attempt_id=rows[0].attempt_id,
                    payload_position=0,
                    kind=DELIVER,
                    policy_digest=policy_digest(HONEST_V1),
                    cell="honest",
                )
            ],
        )


def test_a_disposition_is_keyed_by_its_branch_and_only_this_branch_resolves() -> None:
    """A row is keyed by the attempt, the position and the branch, and one branch exists.

    A row per obligation could hold one child's cell and would silently be every child's, so the
    branch is in the key. What a row for another branch would be is a precommitment, and there is
    nothing here to hold one to: no fork declares the slots it creates, nothing maps a slot to a
    child, and nothing stops a child from resolving itself again. A record carrying one would
    claim an assignment that never controlled an exposure, so it is refused until the fork that
    consumes it arrives with the roster it is checked against.
    """
    row = delivering(BLINDED_RECEIPT_V1)
    assert disposition_key(row) == f"{ATTEMPT}/0/{SINGLETON_SLOT}"
    assert disposition_key(replace(row, branch_slot="b")) == f"{ATTEMPT}/0/b"

    for extra in (
        delivering(BLINDED_RECEIPT_V1, slot="b"),
        PayloadDisposition(
            attempt_id="f" * 32,
            payload_position=999,
            branch_slot="future-child",
            kind=DELIVER,
            policy_digest=policy_digest(BLINDED_RECEIPT_V1),
            cell="graded",
        ),
    ):
        with pytest.raises(PolicyViolation, match="nothing has created"):
            checking(
                [delivering(BLINDED_RECEIPT_V1), extra],
                profile=EXPERIMENT,
                obligations={ATTEMPT: 0},
                silent={},
                grade=REAL_GRADE,
            )

    with pytest.raises(PolicyViolation, match="two dispositions on branch"):
        checking(
            [delivering(HONEST_V1), delivering(BLINDED_RECEIPT_V1)],
            profile=EXPERIMENT,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )


def test_a_delivery_and_an_obligation_have_to_agree() -> None:
    """A row that owes a payload delivers, a row that owes none withholds, and never the other."""
    with pytest.raises(PolicyViolation, match="owes no payload"):
        checking(
            [delivering(HONEST_V1)],
            profile=EXPERIMENT,
            obligations={},
            silent={ATTEMPT: 0},
            grade=REAL_GRADE,
        )
    withheld = PayloadDisposition(
        attempt_id=ATTEMPT, payload_position=0, kind=WITHHOLD, reason="nothing here"
    )
    with pytest.raises(PolicyViolation, match="owes a payload"):
        checking(
            [withheld],
            profile=EXPERIMENT,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )


def test_a_row_says_which_profile_stamped_it_and_the_two_do_not_mix() -> None:
    """Both mistakes are a row that claims the other profile's answer.

    An ordinary generation is the platform's conversion of its own silence, so its rows deliver
    the honest policy under a reason the roster gave and say the platform stamped them. An
    experiment is what it registered, so its rows say they were registered. A run whose rows mix
    the two is one whose record cannot say who decided what its agent was told, which is the
    thing this roster exists to make impossible.
    """
    checking(
        [stamped()], profile=ORDINARY, obligations={ATTEMPT: 0}, silent={}, grade=REAL_GRADE
    )
    with pytest.raises(PolicyViolation, match="delivers honest-v1"):
        checking(
            [replace(delivering(BLINDED_RECEIPT_V1), resolution_source=PLATFORM_DEFAULT)],
            profile=ORDINARY,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )
    with pytest.raises(PolicyViolation, match="was registered"):
        checking(
            [delivering(HONEST_V1)],
            profile=ORDINARY,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )
    with pytest.raises(PolicyViolation, match="stamped by the platform"):
        checking(
            [stamped()],
            profile=EXPERIMENT,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )

    # And the reason an ordinary run withholds under is its roster's, not one somebody wrote.
    platform = PayloadDisposition(
        attempt_id=ATTEMPT,
        payload_position=0,
        kind=WITHHOLD,
        reason="roster_creates_no_obligation",
        resolution_source=PLATFORM_DEFAULT,
    )
    checking(
        [platform], profile=ORDINARY, obligations={}, silent={ATTEMPT: 0}, grade=REAL_GRADE
    )
    with pytest.raises(PolicyViolation, match="withholds under the reason"):
        checking(
            [replace(platform, reason="the arm says nothing here")],
            profile=ORDINARY,
            obligations={},
            silent={ATTEMPT: 0},
            grade=REAL_GRADE,
        )


def test_the_honest_policy_is_refused_over_a_stand_in_grader() -> None:
    """Honesty is a claim about a number, and the stand-in's number is not the environment's."""
    from shogym.serve.protocol_v2.gateway import stream_start, terminal_manifest

    with pytest.raises(PolicyViolation, match="which is a stand-in"):
        checking(
            [delivering(HONEST_V1)],
            profile=EXPERIMENT,
            obligations={ATTEMPT: 0},
            silent={},
            grade=KERNEL_STAND_IN_GRADE,
        )
    spec = _spec()
    with pytest.raises(ValueError, match="which is a stand-in"):
        stream_start(spec, terminal_manifest(spec), claim_hash=CLAIM_HASH)

    # A generation that delivers nothing at all is not making the claim, so it is not refused.
    pinned = stream_start(
        spec, terminal_manifest(spec), claim_hash=CLAIM_HASH, evaluation_only=True
    )
    assert [row.kind for row in pinned.dispositions] == [WITHHOLD]
    assert pinned.dispositions[0].reason == "release_plan_creates_no_obligation"

    # And a blinded body is what an experiment over the stand-in may register instead.
    registered = stream_start(
        spec,
        terminal_manifest(spec),
        claim_hash=CLAIM_HASH,
        profile=EXPERIMENT,
        experiment="the_subject_of_this_run",
        dispositions=lambda rows: [
            PayloadDisposition(
                attempt_id=rows[0].attempt_id,
                payload_position=0,
                kind=DELIVER,
                policy_digest=policy_digest(BLINDED_RECEIPT_V1),
                cell="graded",
            )
        ],
    )
    assert registered.dispositions[0].policy_digest == policy_digest(BLINDED_RECEIPT_V1)


def test_a_composition_is_held_to_the_environment_it_is_opened_over() -> None:
    """A controller composes where the environment is not, and the two meet at the gateway."""
    from shogym.serve.protocol_v2.gateway import _check_honest_over

    honest = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    _check_honest_over(honest, REAL_GRADE)
    over_the_stand_in = make_start(
        profile=ORDINARY, grade=KERNEL_STAND_IN_GRADE, dispositions=[stamped()]
    )
    with pytest.raises(ValueError, match="which is a stand-in"):
        _check_honest_over(over_the_stand_in, KERNEL_STAND_IN_GRADE)

    # And a composition built over one grader is not opened over another. Its honest bodies
    # publish the grader it names, so the second grader's numbers would go out under the first
    # one's name and the record would describe a measurement nobody took.
    claimed = GradeIdentity(
        grader_id="claimed-grader",
        grader_version="99",
        stand_in=False,
        score_component="claimed",
    )
    actual = replace(claimed, grader_id="actual-grader", grader_version="1")
    composed = make_start(profile=ORDINARY, grade=claimed, dispositions=[stamped()])
    with pytest.raises(ValueError, match="composed over claimed-grader/99"):
        _check_honest_over(composed, actual)
    for drifted in (
        replace(REAL_GRADE, grader_version="2"),
        replace(REAL_GRADE, score_component="something_else"),
        replace(
            REAL_GRADE,
            public_components=(GUESSES_USED, PublishedNumber(name="partial_credit", minimum=0, maximum=1)),
        ),
    ):
        with pytest.raises(ValueError, match="is being opened over"):
            _check_honest_over(honest, drifted)


def test_a_policy_this_build_cannot_render_is_not_resolved_to() -> None:
    """There is no renderer to fall back to, so an unknown digest is a refusal at creation."""
    unknown = PayloadDisposition(
        attempt_id=ATTEMPT,
        payload_position=0,
        kind=DELIVER,
        policy_digest="0" * 64,
        cell="honest",
    )
    with pytest.raises(PolicyViolation, match="does not implement"):
        checking(
            [unknown],
            profile=EXPERIMENT,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )
    legacy = replace(unknown, policy_digest=policy_digest(LEGACY_PLACEHOLDER_V1), cell="graded")
    with pytest.raises(PolicyViolation, match="not a policy a generation may be created under"):
        checking(
            [legacy],
            profile=EXPERIMENT,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )
    wrong_cell = replace(unknown, policy_digest=policy_digest(HONEST_V1), cell="placebo")
    with pytest.raises(PolicyViolation, match="declares the cells"):
        checking(
            [wrong_cell],
            profile=EXPERIMENT,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
        )


# The body.


def test_the_honest_body_is_the_score_and_the_numbers_published_beside_it() -> None:
    """One attempt, one score, and the environment's own numbers, in a fixed order."""
    body = render_body(
        HONEST_V1,
        grade=PublicGrade(attempt_id=ATTEMPT, score=0.5, components={"guesses_used": 4.0}),
        payload_position=0,
        submission_digest=DIGEST,
    )
    assert body == f"attempt {ATTEMPT}\nscore 0.5\nguesses_used 4"
    whole = render_body(
        HONEST_V1,
        grade=PublicGrade(attempt_id=ATTEMPT, score=1.0, components={}),
        payload_position=0,
        submission_digest=DIGEST,
    )
    assert whole == f"attempt {ATTEMPT}\nscore 1"


def test_the_body_says_the_number_the_seal_committed_and_not_a_rounding_of_it() -> None:
    """A score printed to fixed places is a claim about a different number.

    An environment that scores fractions commits values no decimal place count covers: a third
    of a ledger is one, and printed to six places it is a number the run never recorded. So the
    format the policy declares is the shortest text that reads back as the same number, and what
    is checked here is that reading it back gives what was committed.
    """
    for score in (1 / 3, 2 / 3, 0.1 + 0.2, 7 / 9, 1e-7, 0.25, 1.0, 0.0):
        body = render_body(
            HONEST_V1,
            grade=PublicGrade(attempt_id=ATTEMPT, score=score, components={"guesses_used": 1 / 7}),
            payload_position=0,
            submission_digest=DIGEST,
        )
        printed = dict(line.split(" ", 1) for line in body.splitlines())
        assert float(printed["score"]) == score
        assert float(printed["guesses_used"]) == 1 / 7
    assert "0.3333333333333333" in render_body(
        HONEST_V1,
        grade=PublicGrade(attempt_id=ATTEMPT, score=1 / 3),
        payload_position=0,
        submission_digest=DIGEST,
    )


def test_the_projection_admits_numbers_and_nothing_a_grader_wrote() -> None:
    """A field list would let a convention or a target through under an allowed name.

    So the projection is a type rather than a set of names: the score is one number in the unit
    interval, the components are finite numbers under token names, and a value that is not one
    of those is a body that does not get built.
    """
    for components in (
        {"expected": float("nan")},
        {"expected": float("inf")},
        {"CONVENTION": 1.0},
        {"note; expected CRANE": 1.0},
    ):
        with pytest.raises(PolicyViolation):
            render_body(
                HONEST_V1,
                grade=PublicGrade(attempt_id=ATTEMPT, score=1.0, components=components),
                payload_position=0,
                submission_digest=DIGEST,
            )
    with pytest.raises(PolicyViolation, match="unit interval"):
        render_body(
            HONEST_V1,
            grade=PublicGrade(attempt_id=ATTEMPT, score=1.5),
            payload_position=0,
            submission_digest=DIGEST,
        )
    with pytest.raises(PolicyViolation, match="at most"):
        render_body(
            HONEST_V1,
            grade=PublicGrade(
                attempt_id=ATTEMPT,
                score=1.0,
                components={f"metric_{index}": float(index) for index in range(17)},
            ),
            payload_position=0,
            submission_digest=DIGEST,
        )


def test_a_number_is_published_under_a_name_the_environment_declared_and_no_other() -> None:
    """The token grammar admits ``target_crane``, and a declared roster is what does not.

    A component name is the one thing in a grade that the grader writes as text and the body
    prints as text. A type cannot tell ``guesses_used`` from ``expected_crane`` and neither can a
    grammar, so what the agent may be told the names of is declared with the grader, before
    anything is graded, and a number arriving under any other name is refused where the
    projection is built.
    """
    grade = published_grade(
        attempt_id=ATTEMPT,
        score=0.5,
        components={"guesses_used": 3.0},
        grade=REAL_GRADE,
    )
    assert grade.components == {"guesses_used": 3.0}
    with pytest.raises(PolicyViolation, match="target_crane"):
        published_grade(
            attempt_id=ATTEMPT,
            score=0.0,
            components={"target_crane": 1.0},
            grade=REAL_GRADE,
        )
    # And a roster naming something a body could not print is refused where it is declared.
    with pytest.raises(PolicyViolation, match="named by a token"):
        published_grade(
            attempt_id=ATTEMPT,
            score=0.0,
            components={},
            grade=replace(
                REAL_GRADE,
                public_components=(
                    PublishedNumber(name="the target is CRANE", minimum=0, maximum=1),
                ),
            ),
        )


def test_a_blinded_renderer_is_given_no_grade_to_withhold() -> None:
    """The restriction is the value's shape rather than the renderer's conduct."""
    body = render_body(
        BLINDED_RECEIPT_V1, grade=None, payload_position=0, submission_digest=DIGEST
    )
    assert body == f"receipt 0 for {DIGEST[:16]}"
    with pytest.raises(PolicyViolation, match="cannot be given a grade"):
        render_body(
            BLINDED_RECEIPT_V1,
            grade=PublicGrade(attempt_id=ATTEMPT, score=1.0),
            payload_position=0,
            submission_digest=DIGEST,
        )
    with pytest.raises(PolicyViolation, match="given none"):
        render_body(HONEST_V1, grade=None, payload_position=0, submission_digest=DIGEST)


async def test_the_renderer_builds_what_the_request_asked_for_and_echoes_it() -> None:
    """The digest decides which body is built, and the candidate says which one ran."""
    honest = await generate_payload_bundle_activity(
        _bundle_request(
            policy_digest=policy_digest(HONEST_V1),
            cell="honest",
            grade=PublicGrade(attempt_id=ATTEMPT, score=1.0, components={"guesses_used": 3.0}),
        )
    )
    [candidate] = honest.candidates
    assert candidate.body == f"attempt {ATTEMPT}\nscore 1\nguesses_used 3"
    assert candidate.policy_digest == policy_digest(HONEST_V1)
    assert candidate.renderer_id == "kernel-honest-1"
    assert candidate.renderer_version == "1"
    assert candidate.cell == "honest"

    # A request from a history recorded before policies existed renders what that history holds.
    legacy = await generate_payload_bundle_activity(_bundle_request())
    [recorded] = legacy.candidates
    assert recorded.body == f"receipt 0 for {DIGEST[:16]}"
    assert recorded.renderer_id == "kernel-receipt-1"
    assert (recorded.policy_digest, recorded.renderer_version) == ("", "")

    with pytest.raises(ApplicationError, match="implements no payload policy"):
        await generate_payload_bundle_activity(_bundle_request(policy_digest="0" * 64))


def _bundle_request(
    *,
    policy_digest: str = "",
    cell: str = "",
    grade: Optional[PublicGrade] = None,
) -> GeneratePayloadBundleInput:
    return GeneratePayloadBundleInput(
        attempt_id=ATTEMPT,
        payload_position=0,
        payload_message_id=oid(0x103),
        submission_digest=DIGEST,
        canonical_submission_text="answer='42'",
        policy_digest=policy_digest,
        cell=cell,
        public_grade=grade,
    )


# The stream.


@pytest_asyncio.fixture
async def env() -> AsyncIterator[WorkflowEnvironment]:
    try:
        environment = await WorkflowEnvironment.start_time_skipping()
    except Exception as error:  # noqa: BLE001 - an absent test server is a skip, not a failure
        pytest.skip(f"the Temporal test server is unavailable: {error}")
    async with environment:
        yield environment


class Caller:
    """One authenticated consumer, keeping its cursor so a test reads as protocol steps."""

    def __init__(self, stream: StreamHandle, cursor: str) -> None:
        self.stream = stream
        self.cursor = cursor
        self._counter = 0

    def next_id(self) -> str:
        self._counter += 1
        return oid(0x1000 + self._counter)

    async def pull(self) -> OfferedMessage:
        return await self.stream.pull(
            PullRequest(request_id=self.next_id(), last_presented_cursor=self.cursor)
        )

    async def present(self, message: OfferedMessage) -> None:
        ack = await self.stream.present(
            message,
            attestation_id=self.next_id(),
            transcript_blob=TRANSCRIPT_BLOB,
            provider_turn_blob=PROVIDER_TURN_BLOB if message.kind == "seal_ack" else None,
            task_start_checkpoint_blob=CHECKPOINT_BLOB if message.kind == "task" else None,
        )
        self.cursor = ack.cursor

    async def take(self) -> OfferedMessage:
        message = await self.pull()
        await self.present(message)
        return message

    async def seal(self) -> OfferedMessage:
        return await self.stream.seal(
            SealRequest(
                metadata=TerminalMetadata(
                    request_id=self.next_id(),
                    last_presented_cursor=self.cursor,
                    attempt_id=ATTEMPT,
                ),
                public_tool_name="submit",
                native_terminal_name="submit",
                native_arguments=dict(FILING),
            )
        )


async def open_stream(
    environment: WorkflowEnvironment, start: StreamStart, *, workflow_id: str
) -> Caller:
    stream = await start_stream(environment.client, start, workflow_id=workflow_id)
    receipt = await stream.claim_consumer(CONSUMER)
    return Caller(stream, receipt.initial_cursor)


async def failed(awaitable: Any) -> str:
    """Return the failure type a call that could not be completed carries."""
    try:
        await awaitable
    except WorkflowUpdateFailedError as error:
        cause = error.cause
        assert isinstance(cause, ApplicationError), cause
        return str(cause.type)
    raise AssertionError("the call was accepted")


@activity.defn(name=GRADE_ATTEMPT)
async def a_real_grade(request: GradeAttemptInput) -> GradeAttemptResult:
    """A grader that reaches an environment, says which grader it is, and has its own number."""
    return replace(
        await grade_attempt_activity(request),
        score=0.25,
        grade=REAL_GRADE,
        public_components={"guesses_used": 3.0},
    )


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def a_renderer_that_never_learned_about_policies(
    request: GeneratePayloadBundleInput,
) -> Any:
    """The build a rolling deployment leaves behind: it renders the body it always rendered."""
    return await generate_payload_bundle_activity(
        replace(request, policy_digest="", cell="", public_grade=None)
    )


async def _substituting(request: GeneratePayloadBundleInput, body: str) -> Any:
    """Return the bundle this request asked for, carrying ``body`` and describing itself as such.

    Every measurement is recomputed over the substituted bytes and the echo is left exactly as
    the request asked for it. This is what a faulty, half upgraded or replaced renderer can
    always produce: metadata is the easy half to get right, and correlating it with a body it
    does not describe costs nothing.
    """
    bundle = await generate_payload_bundle_activity(request)
    [candidate] = bundle.candidates
    serialized = visible_bytes(
        Payload(
            message_id=request.payload_message_id, attempt_id=request.attempt_id, body=body
        )
    )
    return replace(
        bundle,
        candidates=[
            replace(
                candidate,
                body=body,
                inner_sha256=sha256(body.encode("utf-8")).hexdigest(),
                visible_sha256=sha256(serialized).hexdigest(),
                visible_byte_count=len(serialized),
            )
        ],
    )


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def a_renderer_that_echoes_honestly_and_conceals(
    request: GeneratePayloadBundleInput,
) -> Any:
    """The honest echo around the receipt that says nothing about the work."""
    return await _substituting(
        request, f"receipt {request.payload_position} for {request.submission_digest[:16]}"
    )


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def a_renderer_that_echoes_blindly_and_publishes(
    request: GeneratePayloadBundleInput,
) -> Any:
    """The blinded echo around a body carrying a score, which that arm was never to be told."""
    return await _substituting(request, f"attempt {request.attempt_id}\nscore 1")


@activity.defn(name=GENERATE_PAYLOAD_BUNDLE)
async def a_renderer_that_builds_in_another_wrapper(
    request: GeneratePayloadBundleInput,
) -> Any:
    """A Worker whose candidates travel in a wrapper this generation's arm was not built in."""
    bundle = await generate_payload_bundle_activity(request)
    [candidate] = bundle.candidates
    return replace(
        bundle, candidates=[replace(candidate, match_group="another_wrapper")]
    )


@activity.defn(name=GRADE_ATTEMPT)
async def a_grade_from_another_grader(request: GradeAttemptInput) -> GradeAttemptResult:
    """A real grader, and not the one this generation was built over."""
    return replace(
        await grade_attempt_activity(request),
        score=0.25,
        grade=REAL_GRADE_TWO,
        public_components={"guesses_used": 3.0},
    )


@activity.defn(name=GRADE_ATTEMPT)
async def a_grade_that_names_its_target(request: GradeAttemptInput) -> GradeAttemptResult:
    """A grader smuggling text past the type, in the one field a body prints verbatim."""
    return replace(
        await grade_attempt_activity(request),
        score=0.0,
        grade=REAL_GRADE,
        public_components={"target_crane": 1.0},
    )


@activity.defn(name=GRADE_ATTEMPT)
async def a_fractional_grade(request: GradeAttemptInput) -> GradeAttemptResult:
    """A grader whose score is a fraction, as a ledger's is, at the resolution it declared."""
    return replace(
        await grade_attempt_activity(request),
        score=0.33,
        grade=REAL_GRADE,
        public_components={"ledger_fraction": 0.6667},
    )


@activity.defn(name=GRADE_ATTEMPT)
async def a_grade_that_writes_under_its_own_resolution(
    request: GradeAttemptInput,
) -> GradeAttemptResult:
    """A grader hiding bytes in the digits of a score that is otherwise a valid one.

    The number lies in the unit interval and the roster is the declared one, so range and name
    are no help here: what is wrong with it is that this grader said its measure runs to two
    places and this value runs to seventeen.
    """
    return replace(
        await grade_attempt_activity(request),
        score=int.from_bytes(b"CRANE", "big") / 2**40,
        grade=REAL_GRADE,
        public_components={"guesses_used": 3.0},
    )


@pytest.mark.network
async def test_an_ordinary_generation_tells_the_agent_the_score_it_committed(
    env: WorkflowEnvironment,
) -> None:
    """The payload carries the score the seal made authoritative, and the record says so.

    The stream keeps the authoritative record, the acknowledgement says nothing about the work,
    and the body released afterwards is the score the environment gave and the numbers it
    published beside it.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[seal_attempt_activity, a_real_grade, generate_payload_bundle_activity],
    ):
        caller = await open_stream(env, start, workflow_id="stream/policy-honest/1")
        await caller.take()
        ack = await caller.seal()
        await caller.present(ack)
        payload = await caller.pull()
        assert payload.kind == "payload"
        assert f'"body":"attempt {ATTEMPT}\\nscore 0.25\\nguesses_used 3"' in (
            payload.visible_text
        )
        await caller.present(payload)

        state = await caller.stream.stream_state()
        assert state.dispositions == {
            f"{ATTEMPT}/0/{SINGLETON_SLOT}": "deliver:honest-v1:honest:platform_default"
        }
        assert (state.profile, state.experiment_id) == (ORDINARY, None)
        [record] = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        assert record.score == 0.25
        assert record.payload_policy == "honest-v1"
        assert record.payload_disposition == "deliver:honest-v1:honest:platform_default"
        assert (record.profile, record.payload_resolution_source) == (
            ORDINARY,
            PLATFORM_DEFAULT,
        )


@pytest.mark.network
async def test_a_candidate_from_the_wrong_renderer_ends_the_attempt(
    env: WorkflowEnvironment,
) -> None:
    """A body built under something other than what was resolved is not acknowledged.

    A Worker running code the generation did not ask for cannot echo what it was asked for, and
    the seal compares the echo before it commits. The ending is the settled one for a result the
    seal cannot vouch for: the attempt is finalized, the capacity comes back, nothing is
    acknowledged or materialized, and the generation goes on serving. Its reason is its own, so
    a reader can tell a wrong renderer from a result that was merely malformed.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_real_grade,
            a_renderer_that_never_learned_about_policies,
        ],
    ):
        caller = await open_stream(env, start, workflow_id="stream/policy-stale-renderer/1")
        await caller.take()
        assert await failed(caller.seal()) == "RendererDescriptorMismatch"

        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: SEAL_RENDERER}
        assert state.capacity_in_use == 0
        assert state.materialization_count == 0
        assert state.pending_message_id is None

        # The generation keeps serving, which is the whole of what the ending cost.
        await caller.stream.close_queue()
        assert (await caller.pull()).kind == "done"


@pytest.mark.network
@pytest.mark.parametrize("substitution", ["honest_echo_blinded_body", "blinded_echo_honest_body"])
async def test_a_candidate_whose_body_is_not_its_policys_ends_the_attempt(
    env: WorkflowEnvironment, substitution: str
) -> None:
    """A correct echo around the wrong body is not a body this generation may serve.

    The echo says which policy built the candidate and the body is what the agent reads, and
    nothing about a self-reported echo makes the second follow from the first: a renderer that
    is faulty, substituted or half upgraded can copy the requested descriptor around any bytes
    at all. So the authority builds the body again out of the policy it resolved and the grade
    it committed, and the candidate is that or the attempt ends. Both directions are the same
    failure: an ordinary run whose agent is told nothing, and a blinded arm whose agent is told
    its score.
    """
    ordinary = substitution == "honest_echo_blinded_body"
    start = (
        make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
        if ordinary
        else make_start(
            profile=EXPERIMENT,
            grade=REAL_GRADE,
            dispositions=[delivering(BLINDED_RECEIPT_V1)],
        )
    )
    renderer = (
        a_renderer_that_echoes_honestly_and_conceals
        if ordinary
        else a_renderer_that_echoes_blindly_and_publishes
    )
    async with stream_worker(
        env.client, activities=[seal_attempt_activity, a_real_grade, renderer]
    ):
        caller = await open_stream(
            env, start, workflow_id=f"stream/policy-substituted/{substitution}"
        )
        await caller.take()
        assert await failed(caller.seal()) == "RendererDescriptorMismatch"

        state = await caller.stream.stream_state()
        assert state.attempts[ATTEMPT] == "final_failed"
        assert state.final_failures == {ATTEMPT: SEAL_RENDERER}
        assert state.capacity_in_use == 0
        assert state.materialization_count == 0


@pytest.mark.network
async def test_a_grade_from_another_grader_is_not_published_under_this_ones_name(
    env: WorkflowEnvironment,
) -> None:
    """The grader a generation publishes is the one it was built over, checked where it arrives.

    The identity is inside what the generation is and the honest body publishes that grader's
    number, so a result from a different grader would put the second grader's numbers out under
    the first one's name and leave a record describing a measurement nobody took.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_grade_from_another_grader,
            generate_payload_bundle_activity,
        ],
    ):
        caller = await open_stream(env, start, workflow_id="stream/policy-other-grader/1")
        await caller.take()
        assert await failed(caller.seal()) == "UnusableActivityResult"
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
        assert state.materialization_count == 0


@pytest.mark.network
async def test_a_number_the_environment_never_declared_is_not_served(
    env: WorkflowEnvironment,
) -> None:
    """A component name is text a grader wrote, and a body prints it.

    The value's type keeps a string out and the token grammar keeps punctuation out, and neither
    can tell a measure from a hint: ``target_crane`` is a token and a number under it is finite.
    What the run holds instead is the roster the environment declared before it graded anything,
    and a number under any other name ends the attempt rather than reaching a body.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_grade_that_names_its_target,
            generate_payload_bundle_activity,
        ],
    ):
        caller = await open_stream(env, start, workflow_id="stream/policy-smuggled-name/1")
        await caller.take()
        assert await failed(caller.seal()) == "UnusableActivityResult"
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
        assert state.materialization_count == 0


@pytest.mark.network
async def test_a_score_finer_than_the_grader_declared_is_not_served(
    env: WorkflowEnvironment,
) -> None:
    """The headline is held to a resolution the way the numbers beside it are.

    The score's range is the protocol's and is closed already, and a name is no help here at all:
    there is one score and every generation prints it. What is left open without a declared
    resolution is the width of the number, and a unit-interval double is wide enough to write
    five bytes in. So the environment says how fine its measure is, the seal checks the value
    against it, and a score carrying digits below that ends the attempt instead.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_grade_that_writes_under_its_own_resolution,
            generate_payload_bundle_activity,
        ],
    ):
        caller = await open_stream(env, start, workflow_id="stream/policy-fine-score/1")
        await caller.take()
        assert await failed(caller.seal()) == "UnusableActivityResult"
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
        assert state.materialization_count == 0
        [record] = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        # And the smuggled number is nowhere in the record either: what an ended attempt
        # carries is the floor.
        assert (record.score, record.final_failure) == (0.0, SEAL_UNUSABLE)


@pytest.mark.network
async def test_the_body_carries_the_fraction_the_seal_committed(
    env: WorkflowEnvironment,
) -> None:
    """An environment that scores fractions has them published as the numbers they are.

    The record holds a float and the body holds text, and the two say the same number: what the
    agent reads parses back to what the run committed. A body that rounded would report a score
    this generation never recorded, which is a different claim about the same attempt. The
    rounding that does happen is the grader's own, declared before the run and applied to what
    it commits rather than to what it prints.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_fractional_grade,
            generate_payload_bundle_activity,
        ],
    ):
        caller = await open_stream(env, start, workflow_id="stream/policy-fraction/1")
        await caller.take()
        ack = await caller.seal()
        await caller.present(ack)
        payload = await caller.pull()
        body = json.loads(payload.visible_text)["body"]
        printed = dict(line.split(" ", 1) for line in body.splitlines())

        [record] = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        assert record.score == 0.33
        assert float(printed["score"]) == record.score
        assert float(printed["ledger_fraction"]) == 0.6667


@pytest.mark.network
async def test_a_stand_in_grade_is_not_published_as_a_verdict(env: WorkflowEnvironment) -> None:
    """A generation that resolved to publish the grade refuses the stand-in's number.

    The declaration is checked when the generation is created, and this is the same check where
    the number actually arrives: a Worker registering the stand-ins for an environment that
    grades for itself would otherwise have its receipt say the filing scored one because it was
    not empty.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(env.client):
        caller = await open_stream(env, start, workflow_id="stream/policy-stand-in/1")
        await caller.take()
        assert await failed(caller.seal()) == "UnusableActivityResult"
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
        assert state.capacity_in_use == 0


@pytest.mark.network
@pytest.mark.parametrize("exposure", ["blinded", "withheld"])
async def test_the_grader_is_checked_for_a_seal_no_body_publishes(
    env: WorkflowEnvironment, exposure: str
) -> None:
    """The identity is a property of the score, not of what the agent is told about it.

    A concealed cell's number is the outcome that arm reports and a position owing no payload
    still records one, so a stand-in or another implementation substituted behind either would
    put a number nobody took into the record, with nothing about it visible in any body. The
    seal that cannot say the score is its generation's is the seal that cannot go on.
    """
    if exposure == "blinded":
        start = make_start(
            profile=EXPERIMENT, grade=REAL_GRADE, dispositions=[delivering(BLINDED_RECEIPT_V1)]
        )
    else:
        withheld = PayloadDisposition(
            attempt_id=ATTEMPT,
            payload_position=0,
            kind=WITHHOLD,
            reason="roster_creates_no_obligation",
            resolution_source=PLATFORM_DEFAULT,
        )
        start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[withheld])
        start = replace(
            start,
            assignments=assignments_for(
                start.tasks, start.release, without_payload=[ATTEMPT]
            ),
        )
    for workflow_id, grader in (
        (f"stream/policy-identity-{exposure}/1", grade_attempt_activity),
        (f"stream/policy-identity-{exposure}/2", a_grade_from_another_grader),
    ):
        async with stream_worker(
            env.client,
            activities=[seal_attempt_activity, grader, generate_payload_bundle_activity],
        ):
            caller = await open_stream(env, start, workflow_id=workflow_id)
            await caller.take()
            assert await failed(caller.seal()) == "UnusableActivityResult"
            state = await caller.stream.stream_state()
            assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
            assert state.materialization_count == 0
            [record] = await caller.stream.handle.query(StreamWorkflow.attempt_records)
            # Nothing of that grader's reached the record: the score is the floor an ended
            # attempt carries rather than the number this seal was handed.
            assert (record.score, record.final_failure) == (0.0, SEAL_UNUSABLE)


@pytest.mark.network
@pytest.mark.parametrize("malformed", ["score", "component", "roster", "boolean"])
async def test_a_grade_that_is_not_numbers_ends_the_attempt_rather_than_the_decoding(
    env: WorkflowEnvironment, malformed: str
) -> None:
    """A result that is not a grade is an ending with a reason, not a generation that stops.

    A field typed as a number is a field the decoder has to make a number of, so a grader
    returning a string or an object under one would fail the activation rather than the check:
    the generation would retry that activation for ever, record nothing about why, and answer no
    query while it did. And a boolean is a number to a decoder and not to a measurement, so one
    would be published as a score of one. Both are the same failure, and both end the attempt
    with the reason in the record and the generation still answering.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def a_grade_that_is_not_one(request: GradeAttemptInput) -> Any:
        graded = await grade_attempt_activity(request)
        return replace(
            graded,
            grade=REAL_GRADE,
            score={"score": "the filing was good"} if malformed == "score" else True,
            public_components={
                "score": [1, 2, 3],
                "component": {"guesses_used": {"nested": 1}},
                "roster": ["guesses_used", 3],
                "boolean": {"guesses_used": True},
            }[malformed],
        )

    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_grade_that_is_not_one,
            generate_payload_bundle_activity,
        ],
    ):
        caller = await open_stream(env, start, workflow_id=f"stream/policy-shape-{malformed}/1")
        await caller.take()
        assert await failed(caller.seal()) == "UnusableActivityResult"
        # The generation is still answering, which is the half a failed decoding takes away.
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
        assert state.materialization_count == 0
        await caller.stream.close_queue()
        assert (await caller.pull()).kind == "done"


@pytest.mark.network
@pytest.mark.parametrize(
    "malformed",
    [
        "grade_is_text",
        "grade_is_a_list",
        "grader_id_is_nested",
        "roster_entry_is_a_list",
        "roster_is_a_map",
        "grade_has_a_field_of_its_own",
        "evidence_is_text",
        "evidence_digest_is_nested",
        "attempt_id_is_nested",
        "decode_state_is_a_list",
        "there_is_no_result",
        "the_result_is_another_protocols",
        "the_score_is_wider_than_a_double",
        "a_component_is_wider_than_a_double",
        "decode_state_is_longer_than_a_word",
    ],
)
async def test_a_result_that_is_not_a_grade_ends_the_attempt_rather_than_the_decoding(
    env: WorkflowEnvironment, malformed: str
) -> None:
    """The same rule as the numbers, for the identity and the reference beside them.

    These are the fields a result carries that are objects rather than values, and an object is
    where a shape can be nested or a list. Decoding one is the step before any line of this
    generation's own runs, so a mistake there is not an ending at all: the activation fails, it
    is retried for ever, nothing is recorded about why, and the generation stops answering even
    the questions that write nothing. So each of these is bounded here, and what the bound is
    for is the difference between a refusal and a hang.

    The result itself and the size of what it carries are the same question asked of the other
    two ways a Worker can hand this side something it cannot read. A Worker that answered with
    nothing leaves the generation reading fields off nothing. One built against an earlier
    protocol answers with fields this side would otherwise accept and publish as its own. And a
    value too large is one no later check can refuse cleanly: the refusal would be written with
    the value in it, or the arithmetic that decides it raises on the way, and either way the
    ending the result was owed turns back into the hang.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def a_result_that_is_not_a_grade(request: GradeAttemptInput) -> Any:
        if malformed == "there_is_no_result":
            return None
        graded = await grade_attempt_activity(request)
        roster = [{"name": "guesses_used", "minimum": 0, "maximum": 6, "places": 0}]
        identity = {
            "grader_id": "double-grade",
            "grader_version": "1",
            "stand_in": False,
            "score_component": "answer",
            "score_places": 2,
            "public_components": roster,
        }
        # A result that is right in every other way, so what each case changes is the one field
        # it is about.
        fields: Any = {
            "score": 1.0,
            "grade": REAL_GRADE,
            "public_components": {"guesses_used": 3.0},
        }
        broken: Any = {
            "grade_is_text": {"grade": "private expected CRANE"},
            "grade_is_a_list": {"grade": [identity]},
            "grader_id_is_nested": {
                "grade": {**identity, "grader_id": {"was": "double-grade"}}
            },
            "roster_entry_is_a_list": {
                "grade": {**identity, "public_components": [["guesses_used", 0, 6]]}
            },
            "roster_is_a_map": {
                "grade": {**identity, "public_components": {"guesses_used": 6}}
            },
            "grade_has_a_field_of_its_own": {"grade": {**identity, "expected_crane": 1}},
            "evidence_is_text": {"evidence": "the verdict said CRANE"},
            "evidence_digest_is_nested": {
                "evidence": {"sha256": {"was": "0" * 64}, "size": 2, "media_type": "j"}
            },
            "attempt_id_is_nested": {"attempt_id": {"was": ATTEMPT}},
            "decode_state_is_a_list": {"decode_state": ["decoded"]},
            "the_result_is_another_protocols": {"protocol_version": 1},
            # An integer no double can be made of. Every check that would refuse it as a score
            # has to weigh it first, and weighing it is where it raises.
            "the_score_is_wider_than_a_double": {"score": 10**400},
            "a_component_is_wider_than_a_double": {
                "public_components": {"guesses_used": 10**400}
            },
            # And text long enough to be the whole of an answer rather than a word in one.
            "decode_state_is_longer_than_a_word": {"decode_state": "decoded" * 4096},
        }
        fields.update(broken[malformed])
        return replace(graded, **fields)

    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_result_that_is_not_a_grade,
            generate_payload_bundle_activity,
        ],
    ):
        caller = await open_stream(
            env, start, workflow_id=f"stream/policy-object-{malformed}/1"
        )
        await caller.take()
        # Bounded, because the failure this is about is one that never comes back.
        sealed = asyncio.wait_for(failed(caller.seal()), timeout=30)
        assert await sealed == "UnusableActivityResult"
        state = await asyncio.wait_for(caller.stream.stream_state(), timeout=30)
        assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
        assert state.materialization_count == 0
        [record] = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        assert (record.score, record.final_failure) == (0.0, SEAL_UNUSABLE)


async def test_a_history_recorded_before_policies_replays_to_what_it_recorded() -> None:
    """The compatibility branch is pinned by a history this code did not write.

    The fixture is a complete generation recorded by a build from before policies existed, kept
    as the bytes that server wrote. It is the only thing that can hold the branch: a test
    that constructs a legacy start with the current code proves that the current code agrees
    with itself. So this one replays the recorded history, which is what a deployment does to
    every open stream, and checks the three facts that make the branch worth keeping. The start
    decodes as a generation that resolved nothing. The configuration hash the current formula
    takes over it is the one the old code committed into that history, so a resume of such a run
    still presents what it presented. And the body it recorded is the placeholder receipt, which
    is what its records name it.
    """
    recorded = json.loads(
        (Path(__file__).parent / "_fixtures" / "recorded_before_policies.json").read_text()
    )
    history = WorkflowHistory.from_json(RECORDED_BEFORE_POLICIES, recorded)
    started = history.events[0].workflow_execution_started_event_attributes
    start = default_converter().payload_converter.from_payload(
        started.input.payloads[0], StreamStart
    )
    assert (start.profile, start.dispositions, start.provenance) == (LEGACY, [], None)
    assert policy_name_of(None) == "legacy-placeholder-v1"

    committed = [
        json.loads(payload.data)
        for event in history.events
        for payload in (
            event.workflow_execution_update_completed_event_attributes.outcome.success.payloads
        )
    ]
    assert configuration_hash(start) == committed[0]["configuration_hash"]

    [served] = [row["visible_text"] for row in committed if row.get("kind") == "payload"]
    assert json.loads(served)["body"] == f"receipt 0 for {DIGEST[:16]}"

    await stream_replayer().replay_workflow(history)


@pytest.mark.network
@pytest.mark.parametrize(
    "unresolved",
    [
        "uncovered",
        "legacy_with_rows",
        "stand_in",
        "ordinary_blinded",
        "ordinary_registered",
        "ordinary_invented_reason",
        "experiment_stamped",
        "unbound_precommitment",
        "experiment_without_a_registration",
        "ordinary_claiming_a_registration",
        "a_registration_over_other_rows",
        "a_family_no_row_is_a_cell_of",
    ],
)
async def test_a_generation_that_has_not_resolved_what_it_delivers_does_not_start(
    env: WorkflowEnvironment, unresolved: str
) -> None:
    """The refusal is at the stream and not only in whatever composed the generation.

    A builder is a function a caller chooses, so what makes an ordinary run ordinary cannot be
    which function ran. The generation carries its own profile and its own dispositions, and the
    stream holds it to the matrix: an ordinary run delivers the honest policy under the platform's
    stamp and withholds under its roster's own reason, an experiment's rows are registered, every
    position is resolved, no row resolves a branch nothing created, and a generation over a
    stand-in publishes no grade. The profile itself is held the same way: it is a word a caller
    wrote, so the authority behind it has to be there, has to be the right kind, and has to have
    answered for these rows. Each of these starts is one a composition could hand a public entry
    point, and none of them is served.
    """
    withheld = PayloadDisposition(
        attempt_id=ATTEMPT,
        payload_position=0,
        kind=WITHHOLD,
        reason="the arm says nothing here",
        resolution_source=PLATFORM_DEFAULT,
    )
    start = {
        "uncovered": make_start(profile=ORDINARY, grade=REAL_GRADE),
        "legacy_with_rows": make_start(profile=LEGACY, dispositions=[delivering(HONEST_V1)]),
        "stand_in": make_start(
            profile=ORDINARY, grade=KERNEL_STAND_IN_GRADE, dispositions=[stamped()]
        ),
        "ordinary_blinded": make_start(
            profile=ORDINARY,
            grade=REAL_GRADE,
            dispositions=[
                replace(delivering(BLINDED_RECEIPT_V1), resolution_source=PLATFORM_DEFAULT)
            ],
        ),
        "ordinary_registered": make_start(
            profile=ORDINARY, grade=REAL_GRADE, dispositions=[delivering(HONEST_V1)]
        ),
        "ordinary_invented_reason": make_start(
            profile=ORDINARY, grade=REAL_GRADE, dispositions=[withheld]
        ),
        "experiment_stamped": make_start(
            profile=EXPERIMENT, grade=REAL_GRADE, dispositions=[stamped()]
        ),
        "unbound_precommitment": make_start(
            profile=EXPERIMENT,
            grade=REAL_GRADE,
            dispositions=[
                delivering(HONEST_V1),
                delivering(BLINDED_RECEIPT_V1, slot="future-child"),
            ],
        ),
        "experiment_without_a_registration": make_start(
            profile=EXPERIMENT,
            grade=REAL_GRADE,
            dispositions=[delivering(BLINDED_RECEIPT_V1)],
            provenance=None,
        ),
        "ordinary_claiming_a_registration": make_start(
            profile=ORDINARY,
            grade=REAL_GRADE,
            dispositions=[stamped()],
            provenance=entitling(EXPERIMENT, [stamped()]),
        ),
        "a_registration_over_other_rows": make_start(
            profile=EXPERIMENT,
            grade=REAL_GRADE,
            dispositions=[delivering(BLINDED_RECEIPT_V1)],
            provenance=entitling(EXPERIMENT, [delivering(HONEST_V1)]),
        ),
        "a_family_no_row_is_a_cell_of": make_start(
            profile=EXPERIMENT,
            grade=REAL_GRADE,
            dispositions=[replace(delivering(BLINDED_RECEIPT_V1), family_id="the_dose_arm")],
        ),
    }[unresolved]
    async with stream_worker(env.client):
        handle = await env.client.start_workflow(
            StreamWorkflow.run,
            start,
            id=f"stream/policy-unresolved/{unresolved}",
            task_queue=STREAM_TASK_QUEUE,
        )
        with pytest.raises(WorkflowFailureError) as caught:
            await handle.result()
        assert protocol_error_code(caught.value.cause) == "configuration_mismatch"


@pytest.mark.network
async def test_a_generation_created_now_says_what_it_delivers(env: WorkflowEnvironment) -> None:
    """The legacy profile is a decode of an old history and not a shape a new run may take.

    A generation created under it would resolve nothing, serve the placeholder receipt, and hold
    no record of having decided to, which is the blinding a resolved generation makes impossible.
    The stream has to keep accepting the shape so replays work, so the refusal is where a
    generation is created.
    """
    with pytest.raises(ValueError, match="legacy profile"):
        await start_stream(env.client, make_start(), workflow_id="stream/policy-fresh-legacy/1")

    # And the same refusal inside the authority, for a caller that starts the exported workflow
    # itself. The marker is what tells a new execution from a replay of an old history: this one
    # writes it and is refused, and the recorded history that has none keeps its branch.
    async with stream_worker(env.client):
        handle = await env.client.start_workflow(
            StreamWorkflow.run,
            make_start(),
            id="stream/policy-fresh-legacy/2",
            task_queue=STREAM_TASK_QUEUE,
        )
        with pytest.raises(WorkflowFailureError) as caught:
            await handle.result()
        assert protocol_error_code(caught.value.cause) == "configuration_mismatch"


# Who was entitled to the profile, and the matched arms an experiment registers.


def test_a_profile_is_a_claim_something_has_to_stand_behind() -> None:
    """Somebody chooses the profile, so the choice arrives with the authority that made it.

    An experiment names the experiment that registered its rows and an ordinary run names the
    platform default it was stamped from, and both name the digest of the rows that authority
    answered for. What that closes is a label asserted bare: a composition cannot call itself an
    experiment with nothing behind the word, an ordinary run cannot carry a registration, and a
    registration cannot be pointed at answers it was not made over.
    """
    from shogym.serve.protocol_v2.gateway import stream_start, terminal_manifest

    spec = _spec()
    with pytest.raises(ValueError, match="which experiment registered it"):
        stream_start(
            spec,
            terminal_manifest(spec),
            claim_hash=CLAIM_HASH,
            profile=EXPERIMENT,
            grade=REAL_GRADE,
            dispositions=lambda rows: [
                PayloadDisposition(
                    attempt_id=rows[0].attempt_id,
                    payload_position=0,
                    kind=DELIVER,
                    policy_digest=policy_digest(HONEST_V1),
                    cell="honest",
                )
            ],
        )
    with pytest.raises(ValueError, match="under the experiment"):
        stream_start(
            spec,
            terminal_manifest(spec),
            claim_hash=CLAIM_HASH,
            grade=REAL_GRADE,
            experiment="a_run_that_registered_nothing",
        )

    stamp = entitling(ORDINARY, [stamped()])
    registration = entitling(EXPERIMENT, [delivering(BLINDED_RECEIPT_V1)])
    for profile, rows, authority, complaint in (
        (ORDINARY, [stamped()], None, "names no authority at all"),
        (EXPERIMENT, [delivering(BLINDED_RECEIPT_V1)], None, "names no authority at all"),
        (ORDINARY, [stamped()], registration, "different set of rows"),
        (
            ORDINARY,
            [stamped()],
            replace(stamp, authority=REGISTERED),
            "says its rows came from",
        ),
        (
            ORDINARY,
            [stamped()],
            replace(stamp, experiment_id="a_registration_nobody_made"),
            "registered nothing",
        ),
        (
            ORDINARY,
            [stamped()],
            replace(stamp, descriptor_digest="0" * 64),
            "another descriptor",
        ),
        (
            EXPERIMENT,
            [delivering(BLINDED_RECEIPT_V1)],
            replace(registration, experiment_id=""),
            "names the experiment it was made under",
        ),
        (
            EXPERIMENT,
            [delivering(BLINDED_RECEIPT_V1)],
            replace(registration, authority=PLATFORM_DEFAULT),
            "says its rows came from",
        ),
        (LEGACY, [], stamp, "named no authority for one"),
    ):
        with pytest.raises(PolicyViolation, match=complaint):
            check_dispositions(
                rows,
                profile=profile,
                obligations={ATTEMPT: 0} if rows else {},
                silent={},
                grade=REAL_GRADE,
                provenance=authority,
            )


def test_a_matched_family_holds_its_cells_to_one_shape() -> None:
    """A matched arm is a comparison only if its cells differ in what they say and nothing else.

    So the arm declares the cells it holds, the group they are built in and the exact byte count
    they come to, ahead of the run, and every leg of it declares the same. The cells are rendered
    where the arm is registered and held to one length there, because a leg builds the one cell it
    serves and would never see the counterpart it is supposed to match. A family of one cell has
    no counterpart to check at all, and a policy that prints a number is not a cell any of them
    can hold, because the text of a number is as long as the number is and no count holds across
    it. And a row claiming a family this generation never declared is a cell of nothing.
    """
    row = replace(delivering(BLINDED_RECEIPT_V1), family_id="the_dose_arm")
    family = MatchedFamily(
        family_id="the_dose_arm",
        match_group="kernel",
        cells=(
            (policy_digest(BLINDED_RECEIPT_V1), "graded"),
            (policy_digest(PLACEBO_RECEIPT_V1), "placebo"),
        ),
        visible_byte_count=175,
    )
    checking(
        [row],
        profile=EXPERIMENT,
        obligations={ATTEMPT: 0},
        silent={},
        grade=REAL_GRADE,
        families=[family],
    )
    # The other leg of the same arm, which registers the same family and serves the other cell.
    checking(
        [replace(delivering(PLACEBO_RECEIPT_V1), family_id="the_dose_arm")],
        profile=EXPERIMENT,
        obligations={ATTEMPT: 0},
        silent={},
        grade=REAL_GRADE,
        families=[family],
    )

    one_cell = replace(family, cells=(family.cells[0],))
    for declared, rows, complaint in (
        ([family], [replace(row, family_id="another_arm")], "declares no such family"),
        (
            [replace(family, cells=((policy_digest(HONEST_V1), "honest"),) + family.cells)],
            [replace(delivering(HONEST_V1), family_id="the_dose_arm")],
            "prints a number",
        ),
        ([replace(family, cells=())], [row], "separates the cells it holds"),
        ([one_cell], [row], "separates the cells it holds"),
        ([replace(family, visible_byte_count=0)], [row], "declares no byte count"),
        ([replace(family, match_group="another_wrapper")], [row], "renders every candidate"),
        ([family, family], [row], "declared twice"),
        (
            [
                replace(
                    family,
                    cells=(
                        (policy_digest(BLINDED_RECEIPT_V1), "honest"),
                        family.cells[1],
                    ),
                )
            ],
            [row],
            "declares the cells",
        ),
        ([family], [delivering(BLINDED_RECEIPT_V1)], "no row of it is one of that family"),
    ):
        with pytest.raises(PolicyViolation, match=complaint):
            checking(
                rows,
                profile=EXPERIMENT,
                obligations={ATTEMPT: 0},
                silent={},
                grade=REAL_GRADE,
                families=declared,
            )

    # And a matched arm is something an experiment runs. A platform stamp is not a comparison.
    with pytest.raises(PolicyViolation, match="comparison an experiment"):
        checking(
            [stamped()],
            profile=ORDINARY,
            obligations={ATTEMPT: 0},
            silent={},
            grade=REAL_GRADE,
            families=[family],
        )

    # The cells are measured against each other where the arm is registered, because a leg
    # builds the one cell it serves and never sees the counterpart it is supposed to match. The
    # two that ship come to one body at every position, which is what makes them an arm.
    for position in (0, 11):
        bodies = {
            render_body(
                policy,
                grade=None,
                payload_position=position,
                submission_digest=DIGEST,
            )
            for policy in (BLINDED_RECEIPT_V1, PLACEBO_RECEIPT_V1)
        }
        assert len({len(body) for body in bodies}) == 1


def test_a_roster_name_a_body_could_split_or_shadow_is_refused() -> None:
    """The names and the domains are declared before the run, and both are checked.

    A name matched to the end of a line is a name with a newline after it, which is a body whose
    line count the grader chose. A name the body writes itself is a second line saying what the
    authority already said. And a number with a declared name and no declared range is a field
    wide enough to write text in: five big-endian bytes of a large integer spell a word.
    """
    for roster, complaint in (
        (
            (PublishedNumber(name="target_crane\n", minimum=0, maximum=1),),
            "named by a token",
        ),
        ((PublishedNumber(name="score", minimum=0, maximum=1),), "writes score itself"),
        ((PublishedNumber(name="attempt", minimum=0, maximum=1),), "writes attempt itself"),
        (
            (PublishedNumber(name="guesses_used", minimum=6, maximum=0),),
            "which is no range at all",
        ),
    ):
        with pytest.raises(PolicyViolation, match=complaint):
            published_grade(
                attempt_id=ATTEMPT,
                score=0.0,
                components={},
                grade=replace(REAL_GRADE, public_components=roster),
            )

    with pytest.raises(PolicyViolation, match="between 0 and 6"):
        published_grade(
            attempt_id=ATTEMPT,
            score=1.0,
            components={"guesses_used": 289142820421.0},
            grade=REAL_GRADE,
        )
    # And a range on its own is not the whole domain. A measure declared between zero and one
    # still arrives with every digit a double holds unless the environment said how fine it is,
    # and the digits under a measurement are where something that is not one gets written: the
    # five big-endian bytes of this fraction times two to the fortieth spell a word.
    with pytest.raises(PolicyViolation, match="decimal places"):
        published_grade(
            attempt_id=ATTEMPT,
            score=1.0,
            components={"guesses_used": 2.5},
            grade=REAL_GRADE,
        )
    with pytest.raises(PolicyViolation, match="decimal places"):
        published_grade(
            attempt_id=ATTEMPT,
            score=1.0,
            components={"ledger_fraction": 0.26297386322858074},
            grade=REAL_GRADE,
        )
    published_grade(
        attempt_id=ATTEMPT,
        score=1.0,
        components={"ledger_fraction": 0.263, "guesses_used": 2.0},
        grade=REAL_GRADE,
    )
    with pytest.raises(PolicyViolation, match="decimal places"):
        published_grade(
            attempt_id=ATTEMPT,
            score=1.0,
            components={},
            grade=replace(
                REAL_GRADE, public_components=(replace(GUESSES_USED, places=15),)
            ),
        )
    for score in (1.5, -0.5, float("nan")):
        with pytest.raises(PolicyViolation, match="in the unit interval"):
            published_grade(
                attempt_id=ATTEMPT, score=score, components={}, grade=REAL_GRADE
            )


@pytest.mark.network
async def test_a_registered_honest_cell_and_a_stamped_one_read_back_apart(
    env: WorkflowEnvironment,
) -> None:
    """The same body is two different facts, and the record says which.

    An analysis counting these rows is asking which of the two mistakes a run made, if either.
    The policy name cannot answer it: honest-v1 is the right answer for an ordinary run the
    platform stamped and for an experiment that registered an informative cell. So the source is
    in the disposition, the profile is in both records, and the experiment that registered the
    run is in the state.
    """
    registered = make_start(
        profile=EXPERIMENT, grade=REAL_GRADE, dispositions=[delivering(HONEST_V1)]
    )
    async with stream_worker(env.client):
        caller = await open_stream(env, registered, workflow_id="stream/policy-registered/1")
        state = await caller.stream.stream_state()
        assert state.dispositions == {
            f"{ATTEMPT}/0/{SINGLETON_SLOT}": "deliver:honest-v1:honest:registered"
        }
        assert (state.profile, state.experiment_id) == (EXPERIMENT, "the_subject_of_this_run")
        [record] = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        assert record.payload_policy == "honest-v1"
        assert (record.profile, record.payload_resolution_source) == (EXPERIMENT, REGISTERED)


@pytest.mark.network
async def test_a_cell_that_does_not_come_to_its_familys_shape_ends_the_attempt(
    env: WorkflowEnvironment,
) -> None:
    """The declared count is what makes an arm's cells indistinguishable, so it is enforced.

    A candidate longer than its counterpart is one an agent could pick out without reading it,
    and a candidate built in another group is one a harness could. Both are the comparison the
    registration says is being run measuring something else, so the attempt ends the way any
    candidate the seal cannot vouch for does.
    """
    served = visible_bytes(
        Payload(message_id=oid(0x103), attempt_id=ATTEMPT, body=f"receipt 0 for {DIGEST[:16]}")
    )
    row = replace(delivering(BLINDED_RECEIPT_V1), family_id="the_dose_arm")
    family = MatchedFamily(
        family_id="the_dose_arm",
        match_group="kernel",
        cells=(
            (policy_digest(BLINDED_RECEIPT_V1), "graded"),
            (policy_digest(PLACEBO_RECEIPT_V1), "placebo"),
        ),
        visible_byte_count=len(served),
    )
    matched = make_start(
        profile=EXPERIMENT, grade=REAL_GRADE, dispositions=[row], families=[family]
    )
    grading = [seal_attempt_activity, a_real_grade, generate_payload_bundle_activity]
    async with stream_worker(env.client, activities=grading):
        caller = await open_stream(env, matched, workflow_id="stream/policy-family/1")
        await caller.take()
        await caller.present(await caller.seal())
        assert (await caller.pull()).kind == "payload"

    # The same arm, declaring a count its cells do not come to.
    async with stream_worker(env.client, activities=grading):
        caller = await open_stream(
            env,
            make_start(
                profile=EXPERIMENT,
                grade=REAL_GRADE,
                dispositions=[row],
                families=[replace(family, visible_byte_count=len(served) + 1)],
            ),
            workflow_id="stream/policy-family/2",
        )
        await caller.take()
        assert await failed(caller.seal()) == "MatchedFamilyMismatch"
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_RENDERER}
        assert state.materialization_count == 0

    # And the same arm, registered exactly as the first one was, built by a Worker whose wrapper
    # is not the one this build renders in. The group is checked where the family is registered,
    # which is what stops an arm nothing could ever build; this is the other half, where a
    # candidate comes back in a group the registration was refused for.
    async with stream_worker(
        env.client,
        activities=[
            seal_attempt_activity,
            a_real_grade,
            a_renderer_that_builds_in_another_wrapper,
        ],
    ):
        caller = await open_stream(env, matched, workflow_id="stream/policy-family/3")
        await caller.take()
        assert await failed(caller.seal()) == "MatchedFamilyMismatch"
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_RENDERER}
        assert state.materialization_count == 0


@pytest.mark.network
@pytest.mark.parametrize("malformed", ["score", "component"])
async def test_a_verdict_outside_what_the_grader_declared_ends_the_attempt(
    env: WorkflowEnvironment, malformed: str
) -> None:
    """A result is a decoded value, so the projection is applied to what arrived.

    A field declared as a number comes back as a number whatever was put in it, and the score
    the seal commits is authoritative for the run whether or not any body publishes it. So the
    numbers are held to what the environment said they are before any of them is committed,
    rather than at the renderer where only the published ones would be seen.
    """

    @activity.defn(name=GRADE_ATTEMPT)
    async def a_malformed_grade(request: GradeAttemptInput) -> GradeAttemptResult:
        return replace(
            await grade_attempt_activity(request),
            score=2.0 if malformed == "score" else 1.0,
            grade=REAL_GRADE,
            public_components={"guesses_used": 3.0 if malformed == "score" else 289142820421.0},
        )

    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    async with stream_worker(
        env.client,
        activities=[seal_attempt_activity, a_malformed_grade, generate_payload_bundle_activity],
    ):
        caller = await open_stream(env, start, workflow_id=f"stream/policy-malformed/{malformed}")
        await caller.take()
        assert await failed(caller.seal()) == "UnusableActivityResult"
        state = await caller.stream.stream_state()
        assert state.final_failures == {ATTEMPT: SEAL_UNUSABLE}
        # The floor an ended attempt scores, and never the number that arrived.
        [record] = await caller.stream.handle.query(StreamWorkflow.attempt_records)
        assert (record.score, record.final_failure) == (0.0, SEAL_UNUSABLE)


@pytest.mark.network
async def test_the_descriptor_a_generation_named_is_read_back_before_it_is_handed_on(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The preimage is a required object, not a convenience installed beside the run.

    What a run has to be able to prove years later is what its bodies were allowed to say, and a
    digest with nothing behind it says only that something was hashed. So the descriptor goes
    into the run's own store when the generation is created, and it is among the references a
    claim reads back before the generation is handed to a new owner: a store that can no longer
    produce it is refused exactly as one that lost a transcript is.
    """
    from shogym.serve.protocol_v2.blobs import FilesystemBlobStore
    from shogym.serve.protocol_v2.gateway import install_policies

    blobs = FilesystemBlobStore.under(tmp_path / "run")
    start = replace(
        make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()]),
        blob_root=str(blobs.root),
    )
    install_policies(blobs, start)
    assert blobs.read(HONEST_V1_DIGEST) == policy_preimage(HONEST_V1)

    async with stream_worker(env.client):
        caller = await open_stream(env, start, workflow_id="stream/policy-descriptor/1")
        claim = dict(
            configuration_hash=configuration_hash(start),
            previous_epoch=1,
            claimant_id="the-next-owner",
            reason="resume",
        )
        blobs.path_for(HONEST_V1_DIGEST).write_bytes(b"a descriptor saying something else")
        claimant = StreamHandle(caller.stream.handle)
        with pytest.raises(WorkflowUpdateFailedError) as caught:
            await claimant.claim_ownership(**claim)
        assert protocol_error_code(caught.value.cause) == "invalid_message"

        # And the same claim, once the store can produce what the generation resolved to.
        install_policies(blobs, start)
        receipt = await claimant.claim_ownership(**claim)
        assert receipt.ownership_epoch == 2


@pytest.mark.network
async def test_a_generation_whose_descriptor_was_never_installed_does_not_start(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """The claim that creates a generation reads the store, like every claim after it.

    A composition that installs its descriptors and one that only names them are the same start
    to a workflow, and the second is a run whose record says what its bodies were allowed to
    contain over a store that cannot produce the saying. Asking at creation is what makes that a
    generation that never serves, rather than something the resume months later is the first to
    notice.
    """
    from shogym.serve.protocol_v2.blobs import FilesystemBlobStore

    blobs = FilesystemBlobStore.under(tmp_path / "run")
    start = replace(
        make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()]),
        blob_root=str(blobs.root),
    )
    async with stream_worker(env.client):
        with pytest.raises(WorkflowUpdateFailedError) as caught:
            await open_stream(env, start, workflow_id="stream/policy-uninstalled/1")
        assert protocol_error_code(caught.value.cause) == "invalid_message"

    # A store holding something else under that name is the same refusal: the claim reads the
    # bytes rather than asking whether a file is there.
    blobs.path_for(HONEST_V1_DIGEST).parent.mkdir(parents=True, exist_ok=True)
    blobs.path_for(HONEST_V1_DIGEST).write_bytes(b"a descriptor saying something else")
    async with stream_worker(env.client):
        with pytest.raises(WorkflowUpdateFailedError) as caught:
            await open_stream(env, start, workflow_id="stream/policy-uninstalled/2")
        assert protocol_error_code(caught.value.cause) == "invalid_message"


@pytest.mark.network
async def test_the_counterpart_a_matched_arm_names_is_kept_and_read_back(
    env: WorkflowEnvironment, tmp_path: Path
) -> None:
    """An arm has to be able to produce the cell it was matched against, not only the one it ran.

    A leg builds one cell of a family and never renders the counterpart it is supposed to be
    indistinguishable from. So a directory holding only the served descriptor can say what this
    body was allowed to contain and not what the comparison was between, which is the half of the
    record that makes the run a comparison at all. The counterpart's preimage is therefore a
    required object under the same rule the served one is: it goes in when the generation is
    created, and it is read back before the generation is handed to any owner.
    """
    from shogym.serve.protocol_v2.blobs import FilesystemBlobStore
    from shogym.serve.protocol_v2.gateway import install_policies

    served = visible_bytes(
        Payload(message_id=oid(0x103), attempt_id=ATTEMPT, body=f"receipt 0 for {DIGEST[:16]}")
    )
    counterpart = policy_digest(PLACEBO_RECEIPT_V1)
    blobs = FilesystemBlobStore.under(tmp_path / "run")
    start = replace(
        make_start(
            profile=EXPERIMENT,
            grade=REAL_GRADE,
            dispositions=[replace(delivering(BLINDED_RECEIPT_V1), family_id="the_dose_arm")],
            families=[
                MatchedFamily(
                    family_id="the_dose_arm",
                    match_group="kernel",
                    cells=(
                        (policy_digest(BLINDED_RECEIPT_V1), "graded"),
                        (counterpart, "placebo"),
                    ),
                    visible_byte_count=len(served),
                )
            ],
        ),
        blob_root=str(blobs.root),
    )

    # The served cell installed and the counterpart left out is a generation that never starts.
    blobs.put(policy_preimage(BLINDED_RECEIPT_V1), media_type="application/json")
    async with stream_worker(env.client):
        with pytest.raises(WorkflowUpdateFailedError) as caught:
            await open_stream(env, start, workflow_id="stream/policy-counterpart/1")
        assert protocol_error_code(caught.value.cause) == "invalid_message"

    install_policies(blobs, start)
    assert blobs.read(counterpart) == policy_preimage(PLACEBO_RECEIPT_V1)

    async with stream_worker(env.client):
        caller = await open_stream(env, start, workflow_id="stream/policy-counterpart/2")
        claim = dict(
            configuration_hash=configuration_hash(start),
            previous_epoch=1,
            claimant_id="the-next-owner",
            reason="resume",
        )
        blobs.path_for(counterpart).write_bytes(b"a counterpart saying something else")
        claimant = StreamHandle(caller.stream.handle)
        with pytest.raises(WorkflowUpdateFailedError) as caught:
            await claimant.claim_ownership(**claim)
        assert protocol_error_code(caught.value.cause) == "invalid_message"
        # The refusal touched nothing: the owner it would have replaced still holds the run.
        assert (await caller.stream.stream_state()).ownership_epoch == 1

        # And the same claim, once the store can produce the cell this arm was matched against.
        install_policies(blobs, start)
        receipt = await claimant.claim_ownership(**claim)
        assert receipt.ownership_epoch == 2


@pytest.mark.network
async def test_a_resume_composed_for_another_resolution_is_refused_before_it_owns_anything(
    env: WorkflowEnvironment,
) -> None:
    """How fine the headline is is part of what the generation is, so a claim is held to it.

    A grader that scores in whole numbers and one that scores to two decimals admit different
    verdicts under the same name, so a process composed for the second is not the owner of a run
    recorded under the first. The skew is a refusal at the claim rather than a disagreement at
    the next seal: an owner that took the generation would fence the one holding it, keep the
    resolution the history recorded, and end the next otherwise good attempt.
    """
    start = make_start(profile=ORDINARY, grade=REAL_GRADE, dispositions=[stamped()])
    elsewhere = replace(start, grade=replace(REAL_GRADE, score_places=1))
    async with stream_worker(env.client):
        caller = await open_stream(env, start, workflow_id="stream/policy-resolution/1")
        claimant = StreamHandle(caller.stream.handle)
        claim = dict(previous_epoch=1, claimant_id="the-next-owner", reason="resume")
        with pytest.raises(WorkflowUpdateFailedError) as caught:
            await claimant.claim_ownership(
                configuration_hash=configuration_hash(elsewhere), **claim
            )
        assert protocol_error_code(caught.value.cause) == "configuration_mismatch"
        assert (await caller.stream.stream_state()).ownership_epoch == 1

        # And the same claim composed over the resolution this generation declared.
        receipt = await claimant.claim_ownership(
            configuration_hash=configuration_hash(start), **claim
        )
        assert receipt.ownership_epoch == 2


# The environment.


def test_wordle_scores_the_game_that_was_played() -> None:
    """The word was found within the allowed guesses, or it was not.

    The score is the game's own result rather than a fact about the shape of a filing, which is
    the whole reason a generation over this environment may publish it. The progress measure
    stays in the verdict, and what the agent is told beside the score is how many guesses went.
    """
    from shogym.envs.wordle.protocol_v2 import WORDLE_GRADE, _score

    assert WORDLE_GRADE.stand_in is False
    assert WORDLE_GRADE.score_component == "check_answer"

    solved = _score({"target": "crane", "entries": ["crane"]})
    assert solved["check_answer"] == 1.0
    assert solved["guesses_used"] == 1.0
    assert solved["decode_state"] == "decoded"

    near = _score({"target": "crane", "entries": ["crate"]})
    assert near["check_answer"] == 0.0
    assert near["partial_credit"] == 0.8

    nothing = _score({"target": "crane", "entries": []})
    assert nothing["check_answer"] == 0.0
    assert nothing["guesses_used"] == 0.0
    assert nothing["decode_state"] == "ambiguous_zero"

    # The budget is what it is: a seventh guess cannot be the one that found the word.
    late = _score({"target": "crane", "entries": ["stump"] * 6 + ["crane"]})
    assert late["check_answer"] == 0.0
    assert late["guesses_used"] == 6.0


@pytest.mark.network
@pytest.mark.parametrize("found", [True, False])
async def test_a_wordle_play_that_uses_every_guess_is_still_filed_and_scored(
    env: WorkflowEnvironment, found: bool
) -> None:
    """The play this environment tells the agent to make is the one it grades.

    Wordle promises six tries, and a v1 episode of it ends and verifies on the sixth guess rather
    than waiting for a call after it. So this env declares its horizon a graded ending, and the
    sixth guess is the whole of the ending: it is dispatched, its own board comes back, and
    behind it the acknowledgement of the filing that guess made. The two ends of the game are the
    same arc, because what is at stake is the filing rather than the score: a sixth guess that
    finds the word and a sixth that does not both have a result this generation commits.
    """
    from shogym.envs.wordle.utils import load_words
    from shogym.serve.episode import ServedEpisode
    from shogym.serve.protocol_v2.gateway import environment_terminal, open_gateway

    episode = await ServedEpisode.start("wordle_v1", task=0, ends_on_horizon=False)
    try:
        answer = str(episode.env.load_task(0)["answer"])
        misses = [word for word in load_words() if word != answer][:5]
        played = misses + [answer if found else "zzzzz"]
        environment = environment_terminal(episode)
        assert environment.horizon_ending == "graded"
        async with stream_worker(env.client, activities=environment.activities):
            gateway = await open_gateway(env.client, episode, environment=environment)
            await gateway.close_queue()
            task = json.loads(await gateway.pull({}))
            attempt = task["attempt_id"]
            for word in played[:-1]:
                result = await gateway.environment(
                    "guess", {"attempt_id": attempt, "arguments": {"word": word}}
                )
                assert json.loads(result.content[0].text)["valid"] is True
                assert len(result.content) == 1

            # The sixth guess spends the last of the budget and files with it. Its own board is
            # the first item of the result and the acknowledgement is the second, and nothing in
            # those bytes says what the play scored.
            result = await gateway.environment(
                "guess", {"attempt_id": attempt, "arguments": {"word": played[-1]}}
            )
            assert json.loads(result.content[0].text)["valid"] is True
            assert len(result.content) == 2
            ack = json.loads(result.content[1].text)
            assert ack["kind"] == "seal_ack"
            assert "score" not in result.content[1].text

            payload = json.loads(await gateway.pull({}))
            assert payload["kind"] == "payload"
            # And what it says is the game's own result, from the grader this generation was
            # built over rather than from a floor an ending would have written.
            assert payload["body"] == (
                f"attempt {attempt}\nscore {1 if found else 0}\nguesses_used 6"
            )
            state = await gateway.stream_state()
            assert state.attempts[attempt] == "ack_presented"
            assert state.final_failures == {}
            await gateway.aclose()
    finally:
        await episode.close()


def test_a_worker_that_replaced_the_one_which_sealed_grades_the_play_it_sealed(
    tmp_path: Path,
) -> None:
    """The seal and the grade are two Activities, and a Worker can be replaced between them.

    A capture that lived only in the process which took it would leave the stream holding a
    successful seal and the grade behind it saying there was no play, so a valid attempt would
    end as a seal failure. The record is written down instead: a new store over the same
    directory is a new Worker on the same machine, and it grades the play that was sealed even
    with the world that was played in gone. A machine that holds no such record still refuses,
    which is the answer any environment gives for evidence that is not where it is asked for.
    """
    from shogym.envs.wordle import mcp_server
    from shogym.envs.wordle.protocol_v2 import SealStore, _grade, _seal
    from shogym.serve.protocol_v2.kernel.messages import GradeAttemptInput, SealAttemptInput

    seal_id = "1" * 64
    session = "session-for-the-seal"
    mcp_server.begin_session(session, "crane")
    mcp_server.guess("crane", session)
    sealed = _seal(
        lambda attempt_id: (None, session),
        SealStore(tmp_path / "seals"),
        SealAttemptInput(
            attempt_id=ATTEMPT,
            seal_id=seal_id,
            native_terminal_name="submit",
            canonicalization_version="shogym.wordle.1",
        ),
    )
    request = GradeAttemptInput(
        attempt_id=ATTEMPT,
        seal_id=seal_id,
        submission_digest=DIGEST,
        canonical_submission_text=sealed.canonical_submission_text,
        environment_recovery_token=sealed.environment_recovery_token,
    )
    # The world goes with the Worker that had it. What is left is the record.
    mcp_server.end_session(session)
    graded = _grade(SealStore(tmp_path / "seals"), request)
    assert graded.score == 1.0
    assert graded.public_components == {"guesses_used": 1.0}
    assert graded.grade is not None and graded.grade.grader_id == "wordle-grade-v2"

    # And what the agent filed is the words it played. The target is beside them in the record
    # this store holds and is not a field of the submission the digest covers.
    assert json.loads(sealed.canonical_submission_text)["guesses"] == ["crane"]
    assert "target" not in sealed.canonical_submission_text

    with pytest.raises(ApplicationError, match="no play sealed"):
        _grade(SealStore(tmp_path / "another-machine"), request)


# The prose.


def test_no_quickstart_promises_that_the_agent_is_not_told_its_score() -> None:
    """The default is honest, so no document that describes it may promise concealment.

    Five harness quickstarts say the same three things in the same words, and a change made in
    one of them is a change made in one of them. The claim is checked rather than reviewed.
    """
    root = Path(__file__).resolve().parents[1]
    documents = sorted((root / "examples").glob("*/README.md")) + [root / "README.md"]
    assert len(documents) == 6
    for path in documents:
        text = path.read_text(encoding="utf-8")
        for promise in (
            "The agent is not told it",
            "a record the agent never sees",
            "says nothing about how good it was",
        ):
            assert promise not in text, f"{path.name} still promises concealment: {promise!r}"


def _spec() -> Any:
    """The served surface of an environment, as a composition test needs one."""
    from shogym.task import TaskSpec, ToolManifest

    return TaskSpec(
        env_name="filing",
        task_id="7",
        instructions="File the report.",
        tools=[
            ToolManifest(
                name="submit",
                description="File it.",
                input_schema={
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                },
                terminal_kind="score",
            )
        ],
    )
