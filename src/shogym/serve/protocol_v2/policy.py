"""What a payload body may say, as a record rather than as a convention.

A generation delivers bytes to a model, and what those bytes are allowed to contain is a
decision somebody made. This module is where that decision is written down. A
:class:`PayloadPolicy` names the renderer that builds a body, the version of it, the cells it
declares, and the projection its body is drawn from. Its preimage is canonical bytes, its digest
names those bytes, and the digest is folded into what the generation is, so two runs that
recorded the same policy delivered under the same rules and a run that changed renderers has a
different identity.

Four policies ship. ``honest-v1`` tells the agent its score and the numbers the grader published
beside it, and it is what an ordinary run delivers. ``blinded-receipt-v1`` says that a filing was
answered and nothing about the work in it, and only an experiment may select it, which is what
makes concealment a thing a run is recorded as having chosen. ``placebo-receipt-v1`` is the cell
an arm needs opposite that one: a second registered policy whose body says exactly what the
concealed cell's body says, so an arm has two cells that differ in which of them a leg was
assigned and in nothing a model can read. It is registered separately rather than being a second
name for the first, because what a matched family checks is that two independently recorded
policies come to one shape, and a check over one record is a check over nothing.
``legacy-placeholder-v1`` is the receipt this kernel used to build before a body could say
anything, and it exists so a history recorded then still replays to the bytes it recorded. It is
not selectable at all: a generation created now cannot ask for it.

The projection is the whole of what reaches an honest body. :class:`PublicGrade` is a closed type
carrying the attempt the authority assigned, one score in the unit interval, and finite numbers
under token names. A grader's free strings, its evidence, its diagnostics and its private
verdict have nowhere in that type to travel, so a renderer holding it cannot publish them and a
policy that admits more is a different policy with a different digest.

A :class:`PayloadDisposition` is the other half: one obligation's resolved answer, either
``DELIVER`` under a named policy and cell or ``WITHHOLD`` under a reason. It is keyed by the
obligation and the branch it is for, never by the obligation alone, because two branches of one
fork share a public attempt and a payload position while their exposure is exactly what differs
between them. Until a fork declares the slots it creates, the one branch a generation may
resolve is the one it serves: a row naming a slot nothing has created is a claim about an
exposure nothing can be held to, so it is refused rather than recorded.

The profile is what closes the two ways a run comes to serve the wrong body. An ordinary run's
rows are the platform's conversion of its own silence and may say nothing but the honest policy
under a closed set of reasons; an experiment's rows are registered, every one of them. Which of
the two a generation is is carried in its rows rather than in whichever function composed it,
and the matrix is checked where a generation is built and again where it starts.

A profile is a word, though, and a word a caller writes is a claim rather than a fact. So each
one comes with the :class:`PolicyProvenance` that entitles it: an experiment names the
experiment it registered under and the digest of the exact rows it registered, and an ordinary
run names the platform default it was stamped from and that same digest over its own rows.
Neither label can be asserted bare, and neither can be moved onto a different set of rows than
the one its authority answered for.

A :class:`MatchedFamily` is what an experiment declares when the cells of one arm have to be
indistinguishable by shape as well as by wrapper: it fixes the match group its candidates are
built in and the exact model-visible byte count each of them comes to, so a graded cell and a
concealed one cannot be told apart by length. The honest policy publishes a number whose text
is as long as the number is, so it is not a cell any matched family may hold. A family holds two
cells at least, and every one of them is rendered where the family is registered and held to the
same length as the rest: a declaration that measured only the cell a leg happened to serve would
be measuring one receipt and calling it a comparison.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from hashlib import sha256
from typing import Dict, List, Optional, Sequence, Tuple

from shogym.serve.protocol_v2.jcs import encode as canonical_json

#: A generation whose payload policy the platform stamped. Every obligation delivers the honest
#: body and nothing registers anything, which is what an ordinary run is.
ORDINARY = "ordinary"
#: A generation whose payload policy an experiment registered, row by row. There is no default
#: under it: an obligation the roster did not cover is a generation that will not be created.
EXPERIMENT = "experiment"
#: A generation recorded before a policy was a fact about one. It has no dispositions and its
#: bodies are the placeholder receipt, which is what its history holds and what a replay of that
#: history has to keep producing.
LEGACY = "legacy"
PROFILES = (ORDINARY, EXPERIMENT, LEGACY)

#: What a disposition resolves to. A delivery names the policy and the cell; a withholding names
#: why nothing is delivered. Neither is implied by silence.
DELIVER = "deliver"
WITHHOLD = "withhold"

#: The branch a disposition is for, where the generation has one branch. A fork gives its
#: children slots of their own and each child's exposure is its own row under the same
#: obligation; a generation that forks nothing resolves every obligation on this one.
SINGLETON_SLOT = "singleton"

#: Where a disposition came from. The platform stamps an ordinary run's; an experiment registers
#: its own. A reader can tell a converted omission from a declared choice.
PLATFORM_DEFAULT = "platform_default"
REGISTERED = "registered"

#: Why an ordinary run delivers nothing against a position. The set is closed because the reason
#: is the platform's own conversion of a roster it already holds, so a run that withheld under a
#: reason nobody stamped is a run somebody wrote a row for by hand.
NO_RELEASE = "release_plan_creates_no_obligation"
NO_OBLIGATION = "roster_creates_no_obligation"
PLATFORM_REASONS = (NO_RELEASE, NO_OBLIGATION)

#: What an honest body may be drawn from: the score's exposure class, and the placeholder's.
HONEST = "honest"
BLINDED = "blinded"

#: The wrapper a candidate is built in. Every body this kernel renders travels in the same
#: envelope, so this is the one group a matched family's cells can be declared in: a family built
#: in another is one this build could never produce a cell for, which is a registration to refuse
#: rather than a run to start and fail.
KERNEL_MATCH_GROUP = "kernel"

#: How a policy that publishes numbers writes them. A whole number prints as itself and every
#: other prints as the shortest text that reads back as the same number, so the score a body says
#: is the score the seal committed rather than a rounding of it that no longer names it.
SHORTEST_ROUNDTRIP = "shortest_roundtrip"
#: What a policy whose body carries no number out of a grade declares instead.
NO_GRADE_NUMBERS = "no_grade_numbers"

# A component's name, as the honest projection admits it. The grader supplies numbers under these
# and nothing else: a name is a token from this grammar or the body is not built. The match is
# whole, because a grammar anchored at the end still admits a name with a newline after it, and a
# body whose line count is a function of what the grader named is a body that can be split.
_TOKEN = re.compile(r"[a-z][a-z0-9_]{0,31}")

# The names an honest body writes itself. A component under one of them would be a second line
# saying what the authority already said, so a grader does not declare one.
_RESERVED = ("attempt", "score")

# How many numbers one honest body may carry beside the score. The bound is on the projection
# rather than on the bytes, because what a body is allowed to say is a property of the policy and
# a byte count is a property of one grade.
_MOST_COMPONENTS = 16

# How fine a published number may declare itself. A double carries about seventeen digits and a
# measurement carries as many as it means; the gap between the two is where a number stops being
# one. The bound is on the declaration, so an environment that wants more room has to ask for it
# in the identity its generation is built over rather than in a value it returns.
_MOST_PLACES = 6


class PolicyViolation(ValueError):
    """Material a policy does not admit, refused where the body would have been built."""


@dataclass(frozen=True)
class PayloadPolicy:
    """The content-addressed record of what one payload body may say.

    ``renderer_id`` and ``renderer_version`` are the implementation this policy names. They are
    in the preimage, so a run that changed which code builds its bodies changed the digest its
    generation recorded rather than keeping the name and moving what is under it.

    ``cells`` is the finite family this policy declares. An assignment selects one of them and a
    cell this policy never named is not a cell an assignment may carry.

    ``projection`` is the schema of what reaches the renderer, field by field, as a type rather
    than as a field name. A name says which value was picked; a type says what a value may be,
    which is the half that keeps a convention or a target out of a body.

    ``number_format`` is how a number in that projection becomes text. It is declared rather than
    left to the renderer because a body that says a score is making a claim about the number the
    seal committed, and a format that drops digits makes a claim about a different number.
    """

    policy_name: str
    policy_version: str
    renderer_id: str
    renderer_version: str
    exposure: str
    cells: Tuple[str, ...]
    projection: Tuple[Tuple[str, str], ...]
    number_format: str = NO_GRADE_NUMBERS


def policy_preimage(policy: PayloadPolicy) -> bytes:
    """Return the canonical bytes a policy's digest names.

    The preimage is retained rather than merely hashed. A digest says that something was hashed,
    and what a run has to be able to prove years later is what that something said.
    """
    return canonical_json(
        {
            "policy_name": policy.policy_name,
            "policy_version": policy.policy_version,
            "renderer_id": policy.renderer_id,
            "renderer_version": policy.renderer_version,
            "exposure": policy.exposure,
            "cells": list(policy.cells),
            "projection": [{"field": name, "type": kind} for name, kind in policy.projection],
            "number_format": policy.number_format,
        }
    )


def policy_digest(policy: PayloadPolicy) -> str:
    """Return the digest that names this policy's preimage."""
    return sha256(policy_preimage(policy)).hexdigest()


HONEST_V1 = PayloadPolicy(
    policy_name="honest-v1",
    policy_version="1",
    renderer_id="kernel-honest-1",
    renderer_version="1",
    exposure=HONEST,
    cells=(HONEST,),
    projection=(
        ("attempt", "public_attempt_id"),
        ("score", "unit_interval"),
        ("components", "named_finite_numbers"),
    ),
    number_format=SHORTEST_ROUNDTRIP,
)

BLINDED_RECEIPT_V1 = PayloadPolicy(
    policy_name="blinded-receipt-v1",
    policy_version="1",
    renderer_id="kernel-receipt-1",
    renderer_version="1",
    exposure=BLINDED,
    cells=("graded",),
    projection=(
        ("receipt", "payload_position"),
        ("filing", "submission_digest_prefix"),
    ),
)

PLACEBO_RECEIPT_V1 = PayloadPolicy(
    policy_name="placebo-receipt-v1",
    policy_version="1",
    renderer_id="kernel-receipt-1",
    renderer_version="1",
    exposure=BLINDED,
    cells=("placebo",),
    projection=(
        ("receipt", "payload_position"),
        ("filing", "submission_digest_prefix"),
    ),
)

LEGACY_PLACEHOLDER_V1 = PayloadPolicy(
    policy_name="legacy-placeholder-v1",
    policy_version="1",
    renderer_id="kernel-receipt-1",
    renderer_version="1",
    exposure=BLINDED,
    cells=("graded",),
    projection=(
        ("receipt", "payload_position"),
        ("filing", "submission_digest_prefix"),
    ),
)

HONEST_V1_DIGEST = policy_digest(HONEST_V1)
BLINDED_RECEIPT_V1_DIGEST = policy_digest(BLINDED_RECEIPT_V1)
PLACEBO_RECEIPT_V1_DIGEST = policy_digest(PLACEBO_RECEIPT_V1)
LEGACY_PLACEHOLDER_V1_DIGEST = policy_digest(LEGACY_PLACEHOLDER_V1)

#: Every policy this kernel can render, by the digest that names it. A digest that is not in here
#: is a policy this build does not implement, and the answer to one is a refusal rather than
#: whatever body the renderer would have produced without it.
POLICIES: Dict[str, PayloadPolicy] = {
    HONEST_V1_DIGEST: HONEST_V1,
    BLINDED_RECEIPT_V1_DIGEST: BLINDED_RECEIPT_V1,
    PLACEBO_RECEIPT_V1_DIGEST: PLACEBO_RECEIPT_V1,
    LEGACY_PLACEHOLDER_V1_DIGEST: LEGACY_PLACEHOLDER_V1,
}

#: The policies a generation created now may select, by name. The placeholder is renderable and
#: not selectable: it is what a recorded history means, not something a new run may ask for. The
#: blinded receipt is selectable and never a default: an ordinary run stamps the honest policy,
#: so the only way to serve a body that says nothing about the work is for an experiment to
#: register one and be recorded as having done so.
SELECTABLE: Dict[str, PayloadPolicy] = {
    HONEST_V1.policy_name: HONEST_V1,
    BLINDED_RECEIPT_V1.policy_name: BLINDED_RECEIPT_V1,
    PLACEBO_RECEIPT_V1.policy_name: PLACEBO_RECEIPT_V1,
}


@dataclass(frozen=True)
class PolicyProvenance:
    """What entitles a generation to the profile it claims.

    A profile is a word, and a word is something a caller writes. What makes it a fact is this:
    an experiment names the experiment it registered under, an ordinary run names the platform
    default it was stamped from, and both name the digest of the exact rows that authority
    answered for. So an ordinary composition cannot call itself an experiment without a
    registration to point at, an experiment's rows cannot be swapped for another set under the
    same registration, and an ordinary run's honesty is recorded as the platform's own stamp
    rather than as a choice somebody made.

    ``roster_digest`` covers the rows and nothing else. Where they sit in the queue is the
    schedule's business and is covered by the configuration hash already; what this binds is
    which answers this authority gave.
    """

    authority: str
    roster_digest: str
    experiment_id: str = ""
    descriptor_digest: str = ""


@dataclass(frozen=True)
class MatchedFamily:
    """The cells of one experiment arm that have to be indistinguishable by shape.

    A graded cell, a placebo and an oracle are a comparison only if what separates them is what
    they say. A body that is longer than its counterpart is a body an agent can tell apart
    without reading it, and a wrapper built in another group is one a harness can tell apart
    without reading it either. So a matched arm declares both here, ahead of the run: the group
    its candidates are built in, and the exact model-visible byte count each cell comes to.

    ``cells`` is the closed set of policy and cell pairs this family holds, so a row that claims
    the family is one of the cells the family declared rather than any body an experiment liked.
    A policy that publishes a number is not admissible: the text of a number is as long as the
    number is, so no fixed count can hold across the cells of a family that contains one.
    """

    family_id: str
    match_group: str
    cells: Tuple[Tuple[str, str], ...]
    visible_byte_count: int


@dataclass(frozen=True)
class PublishedNumber:
    """One number a grader may publish beside its score, and the domain it lies in.

    A name on its own says which measure a number is and nothing about what it may be. An
    unbounded field is a channel: an integer wide enough to carry text carries text, and a body
    that printed it would be publishing whatever a faulty or substituted grader put there. So
    the environment declares the range each of its numbers lies in before it grades anything, and
    a value outside that range ends the attempt rather than going out.

    ``places`` is the other half of the domain, and the half a range alone leaves open. A number
    is printed as the shortest text that reads back as itself, so a measure declared between zero
    and one still arrives with as many digits as a double can hold, and the digits below the ones
    a measurement means are room. What the environment declares here is how fine its number is:
    zero places is a whole number, and a fraction says how many decimals of it are a measurement.
    A value carrying more than that is not this number at a finer resolution, it is something
    else written in this number's field, so the attempt ends instead.
    """

    name: str
    minimum: float
    maximum: float
    places: int = 0


@dataclass(frozen=True)
class GradeIdentity:
    """What an environment says its grader is, before a generation is built over it.

    ``stand_in`` is the fact a default hangs on. The kernel's grade computes from the shape of a
    filing and reaches no environment, so a body that published its number as the environment's
    verdict would be publishing a transport fixture as a grade. An environment says here whether
    the number a seal commits is its own.

    ``score_component`` names what that number is, in the environment's own terms, so a reader
    of a run knows which of an environment's measures the headline was.

    ``score_places`` is how fine that number is, and it is here for the reason a published
    component's resolution is on the component. The score's range is fixed by the protocol
    rather than declared, so the range is not the half that was left open: a measure between
    zero and one still arrives with as many digits as a double holds, and the digits below the
    ones the measure means are room. So the environment says how many decimals of its headline
    are a measurement, zero being a whole number, and a score carrying more than that is not
    this measure at a finer resolution. The default is the strict one: an environment that
    declares nothing here publishes whole numbers, because the field a grader did not describe
    is the field a substituted grader writes in.

    ``public_components`` is the closed roster of what this grader may publish beside the score.
    A name is a token and the roster is declared here, ahead of any grading, because a name is
    text the grader chose and a body prints it: a roster fixed before the run is what keeps
    ``expected_crane`` out of a payload while ``guesses_used`` goes through. Each entry carries
    the domain its number lies in as well as its name, because a declared name with an
    undeclared range is still a field wide enough to write text in. The roster is part of the
    grade identity, so it is inside what the generation is and a reader can see what the agent
    could have been told.
    """

    grader_id: str
    grader_version: str
    stand_in: bool
    score_component: str
    score_places: int = 0
    public_components: Tuple[PublishedNumber, ...] = ()


#: What an environment that brings no terminal of its own is graded by. It is a stand-in, and it
#: says so, which is what stops an honest body from being built over it.
KERNEL_STAND_IN_GRADE = GradeIdentity(
    grader_id="kernel-stand-in",
    grader_version="1",
    stand_in=True,
    score_component="decoded",
)


@dataclass(frozen=True)
class PayloadDisposition:
    """One obligation's resolved answer, for one branch, fixed before the generation serves.

    The key is the attempt, the payload position and the branch slot together. An obligation on
    its own is the wrong key: a fork's children share the public attempt and the position they
    are exposed at, and which of them is told what is exactly the thing that differs, so a single
    row per obligation could hold one child's cell and would silently be every child's.

    ``resolution_source`` says whether this row was stamped by the platform for an ordinary run
    or registered by an experiment. Both are explicit rows in the generation's state; the column
    is there so a reader can tell a converted omission from a declared choice rather than having
    to infer which builder ran.

    ``family_id`` names the matched family this row is a cell of, where it is one. A row in a
    family is held to that family's group and byte count when its candidate is built, so the
    cells of one arm cannot be told apart by shape.
    """

    attempt_id: str
    payload_position: int
    kind: str
    branch_slot: str = SINGLETON_SLOT
    policy_digest: Optional[str] = None
    cell: Optional[str] = None
    reason: Optional[str] = None
    resolution_source: str = REGISTERED
    family_id: str = ""


def disposition_key(disposition: PayloadDisposition) -> str:
    """Return the key one disposition occupies: the obligation, and the branch it is for."""
    return (
        f"{disposition.attempt_id}/{disposition.payload_position}/{disposition.branch_slot}"
    )


def describe_disposition(disposition: PayloadDisposition) -> str:
    """Return one disposition as the line a harness reads it as.

    The source is part of the line rather than a column beside it. An honest body an experiment
    registered as its informative cell and an honest body the platform stamped over a run that
    said nothing are the same delivery and two different facts, and a reader that cannot tell
    them apart cannot say which of the two mistakes a run made.
    """
    if disposition.kind == WITHHOLD:
        return f"{WITHHOLD}:{disposition.reason}:{disposition.resolution_source}"
    return (
        f"{DELIVER}:{policy_name_of(disposition.policy_digest)}:{disposition.cell}"
        f":{disposition.resolution_source}"
    )


def policy_name_of(digest: Optional[str]) -> str:
    """Return the policy a digest names, as a name, or say the digest is one nothing implements.

    An absent digest is the recorded history that carried no policy at all, and it reads as the
    placeholder, never as honest. A digest this build does not hold is reported as unknown rather
    than resolved to something convenient.
    """
    if not digest:
        return LEGACY_PLACEHOLDER_V1.policy_name
    policy = POLICIES.get(digest)
    return f"unknown-policy:{digest[:16]}" if policy is None else policy.policy_name


def descriptor_digests(
    dispositions: Sequence[PayloadDisposition], families: Sequence[MatchedFamily]
) -> List[str]:
    """Return every descriptor this generation has to be able to produce, in one order.

    A run's own rows are the obvious half: they say what its bodies were allowed to contain, and
    a digest with nothing behind it says only that something was hashed.

    The cells of its matched families are the other half, and they are the half a leg does not
    serve. An arm is a comparison, one leg builds one cell of it, and what makes the leg's record
    a comparison rather than a body with a note attached is the counterpart it was matched
    against. A reader holding this directory alone can recover the counterpart's digest from the
    family either way; what it cannot recover, unless the preimage is here too, is what that
    counterpart was allowed to say. So the counterparts are required objects for the same reason
    the served descriptor is, and they are read back at every claim under the same rule.

    Every cell of a declared family resolves to a policy this build implements, which is checked
    where the family is, so this adds no name the store could not be given the bytes for.
    """
    named = {row.policy_digest for row in dispositions if row.policy_digest}
    named.update(digest for family in families for digest, _ in family.cells)
    return sorted(named)


@dataclass(frozen=True)
class PublicGrade:
    """The whole of what an honest renderer may see.

    ``attempt_id`` comes from the obligation and not from the grader, so a body's identity is the
    authority's own and a body can be held to the wrapper it travels in.

    ``score`` is the canonical component, in the unit interval. ``components`` are the other
    numbers the environment published for the agent, under token names. Nothing here is a string
    an environment wrote, which is what makes the type rather than a review the thing that keeps
    a convention, a target, a rationale or an exception out of a body.
    """

    attempt_id: str
    score: float
    components: Dict[str, float] = field(default_factory=dict)


def published_grade(
    *,
    attempt_id: str,
    score: float,
    components: Dict[str, float],
    grade: GradeIdentity,
) -> PublicGrade:
    """Return what a renderer may see of one verdict, or refuse what the grader declared none of.

    A name is the one thing in a grade that a grader writes as text and a body prints as text.
    The type keeps a string out of the values and the token grammar keeps punctuation out of the
    names, and neither of those stops ``expected_crane``. What stops it is that the environment
    said before the run which names it publishes, so a name that was not declared is a number
    this generation never agreed to carry and the attempt ends rather than the body going out.
    """
    check_grade(grade)
    check_grade_result(score=score, components=components, grade=grade)
    return PublicGrade(attempt_id=attempt_id, score=score, components=dict(components))


def check_grade_result(
    *, score: float, components: Dict[str, float], grade: GradeIdentity
) -> None:
    """Refuse a verdict that is not the shape this grader declared it would produce.

    A result arrives as a decoded value rather than as the object a grader built, and decoding
    is a conversion: a field declared as a number comes back as one whatever was put in it. So
    what the environment said its numbers are is checked here, against the value that arrived,
    before any of it is committed or printed. The score is one finite number in the unit
    interval at the resolution its grader declared, and each published number is a finite number
    under a declared name inside the range declared for it, whole where the declaration says
    whole. The score is held to its resolution for the reason a component is: the range it lies
    in leaves every digit a double holds available, and a body prints the ones a measure does
    not mean alongside the ones it does.
    """
    if not _finite(score) or not 0.0 <= float(score) <= 1.0:
        raise PolicyViolation(
            f"{grade.grader_id} committed {score!r}, and a score is one number in the unit "
            "interval"
        )
    if round(float(score), grade.score_places) != float(score):
        raise PolicyViolation(
            f"{grade.grader_id} scores to {grade.score_places} decimal places, and this one is "
            f"{score!r}: the digits under a measure's own resolution are room for something "
            "that is not the measure"
        )
    declared = {number.name: number for number in grade.public_components}
    undeclared = sorted(name for name in components if name not in declared)
    if undeclared:
        raise PolicyViolation(
            f"{grade.grader_id} published {undeclared} beside its score, and what a body may "
            f"carry is the roster this environment declared: {sorted(declared)}"
        )
    for name in sorted(components):
        value = components[name]
        number = declared[name]
        if not _finite(value):
            raise PolicyViolation(
                f"a published component is a finite number and {name} carried {value!r}"
            )
        if not number.minimum <= float(value) <= number.maximum:
            raise PolicyViolation(
                f"{grade.grader_id} publishes {name} between {number.minimum} and "
                f"{number.maximum}, and this one is {value!r}"
            )
        if round(float(value), number.places) != float(value):
            raise PolicyViolation(
                f"{grade.grader_id} publishes {name} to {number.places} decimal places, and this "
                f"one is {value!r}: the digits under a measure's own resolution are room for "
                "something that is not the measure"
            )


def check_grade(grade: GradeIdentity) -> None:
    """Refuse a grade identity whose declared roster is not one a body could print."""
    if not 0 <= grade.score_places <= _MOST_PLACES:
        raise PolicyViolation(
            f"a score is written to between 0 and {_MOST_PLACES} decimal places, and "
            f"{grade.grader_id} declares {grade.score_places}: a field as fine as the "
            "arithmetic under it is a field with room in it"
        )
    names = [number.name for number in grade.public_components]
    if len(names) > _MOST_COMPONENTS:
        raise PolicyViolation(
            f"a grade publishes at most {_MOST_COMPONENTS} numbers beside its score, and "
            f"{grade.grader_id} declares {len(names)}"
        )
    if len(set(names)) != len(names):
        raise PolicyViolation(
            f"{grade.grader_id} declares one of its published numbers twice, and a roster names "
            "each of them once"
        )
    for number in grade.public_components:
        if not _TOKEN.fullmatch(number.name):
            raise PolicyViolation(
                f"a published component is named by a token and {grade.grader_id} declares one "
                f"named {number.name!r}"
            )
        if number.name in _RESERVED:
            raise PolicyViolation(
                f"an honest body writes {number.name} itself, so it is not a name a grader "
                f"publishes a number under, and {grade.grader_id} declares one"
            )
        if not _finite(number.minimum) or not _finite(number.maximum):
            raise PolicyViolation(
                f"{grade.grader_id} declares {number.name} over a range that is not two finite "
                "numbers"
            )
        if number.minimum > number.maximum:
            raise PolicyViolation(
                f"{grade.grader_id} declares {number.name} between {number.minimum} and "
                f"{number.maximum}, which is no range at all"
            )
        if not 0 <= number.places <= _MOST_PLACES:
            raise PolicyViolation(
                f"a published number is written to between 0 and {_MOST_PLACES} decimal places, "
                f"and {grade.grader_id} declares {number.name} to {number.places}: a field as "
                "fine as the arithmetic under it is a field with room in it"
            )


def render_body(
    policy: PayloadPolicy,
    *,
    grade: Optional[PublicGrade],
    payload_position: int,
    submission_digest: str,
) -> str:
    """Return the body this policy declares, or refuse the material it does not admit.

    The refusal is the point. A projection that admitted whatever it was handed would be a field
    list rather than a schema, so a score outside the unit interval, a name that is not a token,
    a value that is not a finite number, and a component roster past the declared bound are each
    a body that does not get built.
    """
    if policy.exposure == HONEST:
        if grade is None:
            raise PolicyViolation(
                f"{policy.policy_name} publishes the grade and this obligation was given none"
            )
        return _honest_body(grade, policy.number_format)
    if grade is not None:
        raise PolicyViolation(
            f"{policy.policy_name} is blinded and cannot be given a grade to render"
        )
    return f"receipt {payload_position} for {submission_digest[:16]}"


def _honest_body(grade: PublicGrade, number_format: str) -> str:
    """Return the honest receipt: the attempt, the score, and the published numbers."""
    if not _finite(grade.score) or not 0.0 <= grade.score <= 1.0:
        raise PolicyViolation(
            f"the honest score is one number in the unit interval, and this one is {grade.score!r}"
        )
    if len(grade.components) > _MOST_COMPONENTS:
        raise PolicyViolation(
            f"an honest body carries at most {_MOST_COMPONENTS} published numbers, and this "
            f"grade published {len(grade.components)}"
        )
    lines = [
        f"attempt {grade.attempt_id}",
        f"score {_number(grade.score, number_format)}",
    ]
    for name in sorted(grade.components):
        if not _TOKEN.fullmatch(name) or name in _RESERVED:
            raise PolicyViolation(
                f"a published component is named by a token this body does not write itself, "
                f"and this one is named {name!r}"
            )
        value = grade.components[name]
        if not _finite(value):
            raise PolicyViolation(
                f"a published component is a finite number and {name} carried {value!r}"
            )
        lines.append(f"{name} {_number(value, number_format)}")
    return "\n".join(lines)


def number_text(value: float) -> str:
    """Return one number as the exact text this kernel writes numbers in.

    A record that carries a number carries it as text, because the canonical encoding a digest
    is taken over holds whole numbers only and a range written as a rounding of itself would
    name a different range. It is the same rule the honest body prints under, so a bound and the
    value it bounds are written the same way.
    """
    return _number(value, SHORTEST_ROUNDTRIP)


def _finite(value: object) -> bool:
    """True iff this is a real number a body can print exactly once.

    An integer wider than a double is one of the values this answers no about rather than one it
    raises on. It is a predicate the checks read to decide what to refuse, so a value that made
    it raise would take the refusal with it and leave whatever called the check unfinished.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except OverflowError:
        return False


def _number(value: float, number_format: str) -> str:
    """Return one number as the text this policy declares numbers are written in.

    A whole number prints as itself and everything else prints as the shortest text that reads
    back as the same number. Both are exact, which is the property that matters: a body saying a
    score is a claim about the number the seal committed, and one that printed a rounding of it
    would be reporting a number the run never recorded. The text is a function of the value
    alone, so the same grade renders to the same bytes wherever it is built.
    """
    if number_format != SHORTEST_ROUNDTRIP:
        raise PolicyViolation(
            f"a body that prints a number declares how, and {number_format!r} is not a format "
            "this build writes numbers in"
        )
    number = float(value)
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    return repr(number)


def roster_digest(dispositions: List[PayloadDisposition]) -> str:
    """Return the digest of the exact rows an authority answered for.

    Order is not one of the facts, so the rows are sorted by the obligation they resolve. What
    the digest covers is every field of every row, which is what makes a registration a
    commitment to these answers rather than to the count of them.
    """
    return sha256(
        canonical_json(
            [
                {
                    "attempt_id": row.attempt_id,
                    "payload_position": row.payload_position,
                    "branch_slot": row.branch_slot,
                    "kind": row.kind,
                    "policy_digest": row.policy_digest,
                    "cell": row.cell,
                    "reason": row.reason,
                    "resolution_source": row.resolution_source,
                    "family_id": row.family_id,
                }
                for row in sorted(dispositions, key=disposition_key)
            ]
        )
    ).hexdigest()


def check_provenance(
    provenance: Optional[PolicyProvenance],
    *,
    profile: str,
    dispositions: List[PayloadDisposition],
) -> None:
    """Refuse a profile nothing entitles this generation to.

    Somebody has to choose which kind of run this is, and the choice arrives as a word. What
    stops the word from being the whole of it is that each profile has an authority behind it
    and the authority has to be named: an experiment says which experiment registered these
    rows, an ordinary run says it was stamped from the platform default and which descriptor
    that is. Both say the digest of the rows they answered for, so a registration cannot be
    pointed at a different set of answers than the one it was made over.

    A history recorded before any of this has no authority to name and none is admitted for it,
    for the same reason it has no rows: it predates the question rather than answering it.
    """
    if profile == LEGACY:
        if provenance is not None:
            raise PolicyViolation(
                "a generation that recorded no profile named no authority for one, and a "
                "provenance under one is a run that was created after profiles existed"
            )
        return
    if provenance is None:
        raise PolicyViolation(
            f"a {profile} generation says what entitles it to that profile, and this one names "
            "no authority at all"
        )
    if provenance.roster_digest != roster_digest(dispositions):
        raise PolicyViolation(
            "the authority this generation names answered for a different set of rows than the "
            "ones it carries"
        )
    if profile == EXPERIMENT:
        if provenance.authority != REGISTERED:
            raise PolicyViolation(
                f"an {EXPERIMENT} generation is created from what an experiment registered, and "
                f"this one says its rows came from {provenance.authority!r}"
            )
        if not _TOKEN.fullmatch(provenance.experiment_id):
            raise PolicyViolation(
                "an experiment registration names the experiment it was made under, and this "
                f"one names {provenance.experiment_id!r}"
            )
        if provenance.descriptor_digest:
            raise PolicyViolation(
                "an experiment registers the policy of each row, so there is no platform "
                "descriptor for it to have been stamped from"
            )
        return
    if provenance.authority != PLATFORM_DEFAULT:
        raise PolicyViolation(
            f"an {ORDINARY} generation is the platform's own conversion of its silence, and this "
            f"one says its rows came from {provenance.authority!r}"
        )
    if provenance.experiment_id:
        raise PolicyViolation(
            f"an {ORDINARY} generation registered nothing, and this one names the experiment "
            f"{provenance.experiment_id!r}: a run that registered an experiment is one"
        )
    if provenance.descriptor_digest != HONEST_V1_DIGEST:
        raise PolicyViolation(
            f"an {ORDINARY} generation is stamped from {HONEST_V1.policy_name}, and this one "
            "names another descriptor as the default it was stamped from"
        )


def check_families(
    families: List[MatchedFamily],
    *,
    profile: str,
    dispositions: List[PayloadDisposition],
) -> None:
    """Refuse a matched family that could not hold, and a row claiming one that is not there.

    What a matched family promises is that its cells differ in what they say and in nothing
    else. Three things make that a fact rather than a declaration.

    The family holds two cells at least, each under a policy this build implements and lets a
    generation select. One cell is a comparison with nothing: a record naming it would carry the
    word family over a run that had no counterpart registered anywhere.

    Every one of those cells is rendered here, for each obligation that claims the family, and
    they are required to come to one body length. That is the cross-cell half, and it is checked
    where the family is registered rather than where one leg serves: a generation is one leg and
    builds one cell, so the counterpart it is supposed to be indistinguishable from is never
    built by the run that would notice. The bodies are measured under two different submission
    digests as well, because a renderer whose length is a function of what it was handed is one
    no single measurement would catch.

    The family also declares the group its cells are built in and the exact model-visible byte
    count each of them reaches, and the candidate for a row in the family is held to both when it
    is built. Equal body lengths and one wrapper make the cells' visible counts equal to each
    other; the served candidate's count being the declared one is what makes them equal to the
    number the arm registered, and the two together are what hold the legs of an arm to one
    another. A family holding a policy that prints a number promises something no fixed count can
    keep, so a policy that publishes the grade is not a cell any of them may hold.

    An ordinary run declares none. A matched family is a comparison an experiment is running,
    and a platform stamp is not a comparison.
    """
    declared: Dict[str, MatchedFamily] = {}
    for family in families:
        if profile != EXPERIMENT:
            raise PolicyViolation(
                f"a matched family is a comparison an {EXPERIMENT} generation registers, and "
                f"this {profile} generation declares {family.family_id!r}"
            )
        if not _TOKEN.fullmatch(family.family_id) or not _TOKEN.fullmatch(family.match_group):
            raise PolicyViolation(
                f"a matched family and the group its cells are built in are named by tokens, and "
                f"this one is {family.family_id!r} in {family.match_group!r}"
            )
        if family.family_id in declared:
            raise PolicyViolation(
                f"the matched family {family.family_id} is declared twice, and a family is one "
                "set of cells"
            )
        declared[family.family_id] = family
        if len(family.cells) < 2 or len(set(family.cells)) != len(family.cells):
            raise PolicyViolation(
                f"the matched family {family.family_id} declares {len(family.cells)} cells, and "
                "a family separates the cells it holds: two of them at least, each of them once"
            )
        if family.match_group != KERNEL_MATCH_GROUP:
            raise PolicyViolation(
                f"this build renders every candidate in {KERNEL_MATCH_GROUP!r}, and the matched "
                f"family {family.family_id} declares its cells are built in "
                f"{family.match_group!r}"
            )
        if family.visible_byte_count <= 0:
            raise PolicyViolation(
                f"the matched family {family.family_id} declares no byte count for its cells, "
                "and what makes its cells indistinguishable is that they come to the same one"
            )
        for digest, cell in family.cells:
            policy = POLICIES.get(digest)
            if policy is None or policy.policy_name not in SELECTABLE:
                raise PolicyViolation(
                    f"the matched family {family.family_id} holds a cell under a policy this "
                    "build does not implement or does not let a generation select"
                )
            if cell not in policy.cells:
                raise PolicyViolation(
                    f"{policy.policy_name} declares the cells {list(policy.cells)}, and the "
                    f"matched family {family.family_id} holds {cell!r}"
                )
            if policy.number_format != NO_GRADE_NUMBERS:
                raise PolicyViolation(
                    f"{policy.policy_name} prints a number, whose text is as long as the number "
                    f"is, so it is not a cell the matched family {family.family_id} can hold to "
                    "one byte count"
                )
    claimed: Dict[str, str] = {}
    for row in dispositions:
        if not row.family_id:
            continue
        family = declared.get(row.family_id)
        if family is None:
            raise PolicyViolation(
                f"the row for attempt {row.attempt_id} is a cell of the matched family "
                f"{row.family_id!r}, and this generation declares no such family"
            )
        if (row.policy_digest, row.cell) not in family.cells:
            raise PolicyViolation(
                f"the row for attempt {row.attempt_id} claims the matched family "
                f"{row.family_id}, and what it delivers is not one of that family's cells"
            )
        claimed[family.family_id] = row.attempt_id
        _check_cells_match(family, row.payload_position)
    for family_id in sorted(declared):
        if family_id not in claimed:
            raise PolicyViolation(
                f"this generation declares the matched family {family_id} and no row of it is "
                "one of that family's cells, so the arm it names is one this generation is not in"
            )


def _check_cells_match(family: MatchedFamily, payload_position: int) -> None:
    """Render every cell of one family for one obligation, and refuse two lengths.

    The measurement is of the body rather than of the serialized result, because every candidate
    for one obligation travels in the same wrapper: the message it becomes carries the position's
    own preallocated identifiers whichever cell built it, so two bodies of one length are two
    results of one length. Two submission digests are used because a body whose length depends on
    what it was handed would come to one length under one of them and to another under a filing
    nobody has made yet.
    """
    lengths: Dict[int, List[str]] = {}
    for digest, cell in family.cells:
        policy = POLICIES[digest]
        for submission in ("0" * 64, "f" * 64):
            body = render_body(
                policy,
                grade=None,
                payload_position=payload_position,
                submission_digest=submission,
            )
            lengths.setdefault(len(body.encode("utf-8")), []).append(
                f"{policy.policy_name}/{cell}"
            )
    if len(lengths) > 1:
        shapes = ", ".join(
            f"{sorted(set(names))} at {length}" for length, names in sorted(lengths.items())
        )
        raise PolicyViolation(
            f"the cells of the matched family {family.family_id} come to different lengths at "
            f"payload position {payload_position} ({shapes}), and cells an agent can tell apart "
            "by length are not a comparison"
        )


def check_dispositions(
    dispositions: List[PayloadDisposition],
    *,
    profile: str,
    obligations: Dict[str, int],
    silent: Dict[str, int],
    grade: GradeIdentity,
    provenance: Optional[PolicyProvenance] = None,
    families: Optional[List[MatchedFamily]] = None,
) -> None:
    """Refuse a roster of dispositions that does not resolve this generation.

    ``obligations`` are the positions this generation owes a payload against and ``silent`` are
    the ones it does not, both by attempt. The cross checks are the two directions of one rule:
    a row that owes a payload delivers under a policy, a row that owes none withholds under a
    reason, and neither may claim the other's answer. A generation cannot be created with an
    obligation nobody resolved, nor with two answers to one obligation on one branch.

    ``provenance`` is what entitles the generation to the profile it claims, and ``families``
    are the matched arms it registered. Both are checked here, with the rows, because a profile
    and a family are claims about these rows and are worth nothing said over another set.

    ``profile`` decides which rows are admissible at all, and it is checked here rather than
    trusted from the builder that produced them. An ordinary generation carries the platform's
    own conversion of its silence: the honest policy where it owes a payload, one of the closed
    platform reasons where it does not, and both marked as stamped. An experiment carries
    registered rows and nothing else. A row that mixes the two is a run whose record does not say
    who decided what its agent was told, which is the failure this whole roster exists to make
    impossible.

    A row for a branch this generation does not serve is refused. Such a row is a precommitment,
    and there is no fork here to declare the slots it would create, map them to children, or stop
    a child from resolving itself again: an audit surface that recorded one would be claiming an
    assignment that never controlled an exposure. The key keeps its branch, because the fork this
    is for is the reason two rows can share an obligation, and the roster it will be bound to is
    the thing that has to arrive with it.
    """
    if profile == LEGACY:
        if dispositions:
            raise PolicyViolation(
                "a generation that recorded no profile resolved nothing, and rows under one are "
                "a run that was created after policies existed and says it was not"
            )
        check_provenance(provenance, profile=profile, dispositions=[])
        check_families(list(families or []), profile=profile, dispositions=[])
        return
    if profile not in (ORDINARY, EXPERIMENT):
        raise PolicyViolation(f"a generation is {ORDINARY} or {EXPERIMENT}, not {profile!r}")
    check_provenance(provenance, profile=profile, dispositions=dispositions)
    check_families(list(families or []), profile=profile, dispositions=dispositions)
    check_grade(grade)
    positions = {**obligations, **silent}
    seen: Dict[str, PayloadDisposition] = {}
    for row in dispositions:
        key = disposition_key(row)
        if key in seen:
            raise PolicyViolation(
                f"attempt {row.attempt_id} has two dispositions on branch {row.branch_slot}, "
                "and one obligation on one branch resolves once"
            )
        seen[key] = row
        _check_shape(row, grade=grade)
        _check_source(row, profile=profile)
        served = obligations if row.kind == DELIVER else silent
        other = silent if row.kind == DELIVER else obligations
        if row.branch_slot != SINGLETON_SLOT:
            raise PolicyViolation(
                f"the row for attempt {row.attempt_id} resolves the branch "
                f"{row.branch_slot!r}, and this generation serves one branch: a row for a slot "
                "nothing has created is a precommitment nothing can be held to"
            )
        if row.attempt_id in other:
            raise PolicyViolation(
                f"attempt {row.attempt_id} is a row this generation "
                f"{'owes no payload against' if row.kind == DELIVER else 'owes a payload against'}"
                f", so its disposition cannot be {row.kind}"
            )
        if served.get(row.attempt_id) != row.payload_position:
            raise PolicyViolation(
                f"the disposition for attempt {row.attempt_id} names payload position "
                f"{row.payload_position}, which is not the position its roster row assigns"
            )
    for attempt_id in sorted(positions):
        if f"{attempt_id}/{positions[attempt_id]}/{SINGLETON_SLOT}" not in seen:
            raise PolicyViolation(
                f"attempt {attempt_id} has no disposition, and a generation serves no payload "
                "position it has not resolved"
            )


def _check_source(row: PayloadDisposition, *, profile: str) -> None:
    """Hold one row to the profile its generation declared.

    The two directions are the two mistakes. An ordinary run that carried a registered row would
    be a run somebody blinded without declaring an experiment, so an ordinary row says the honest
    policy, one of the closed platform reasons, and that the platform stamped it. An experiment
    row that claimed the platform stamped it would be a registration nobody made, so an
    experiment's rows are registered, all of them. What is left after both is a record in which a
    reader can tell a converted omission from a declared choice, which is the whole point of
    keeping the column.
    """
    if profile == EXPERIMENT:
        if row.resolution_source != REGISTERED:
            raise PolicyViolation(
                f"the row for attempt {row.attempt_id} says it was stamped by the platform, and "
                f"an {EXPERIMENT} generation is created from what it registered"
            )
        return
    if row.resolution_source != PLATFORM_DEFAULT:
        raise PolicyViolation(
            f"the row for attempt {row.attempt_id} was registered, and an {ORDINARY} generation "
            "is the platform's own conversion of its silence: a registered row belongs to an "
            f"{EXPERIMENT} run"
        )
    if row.kind == DELIVER and row.policy_digest != HONEST_V1_DIGEST:
        raise PolicyViolation(
            f"an {ORDINARY} generation delivers {HONEST_V1.policy_name} against every payload it "
            f"owes, and the row for attempt {row.attempt_id} delivers "
            f"{policy_name_of(row.policy_digest)}"
        )
    if row.kind == WITHHOLD and row.reason not in PLATFORM_REASONS:
        raise PolicyViolation(
            f"an {ORDINARY} generation withholds under the reason its roster gave, and the row "
            f"for attempt {row.attempt_id} names {row.reason!r}"
        )


def _check_shape(row: PayloadDisposition, *, grade: GradeIdentity) -> None:
    """Refuse one disposition that does not say what a disposition says."""
    if row.kind == WITHHOLD:
        if not row.reason or row.policy_digest is not None or row.cell is not None:
            raise PolicyViolation(
                f"a withholding names why and nothing else, and the row for attempt "
                f"{row.attempt_id} does not"
            )
        return
    if row.kind != DELIVER:
        raise PolicyViolation(
            f"a disposition is {DELIVER} or {WITHHOLD}, and the row for attempt "
            f"{row.attempt_id} is {row.kind!r}"
        )
    policy = POLICIES.get(row.policy_digest or "")
    if policy is None:
        raise PolicyViolation(
            f"the row for attempt {row.attempt_id} delivers under a policy this build does not "
            f"implement, and there is no renderer to fall back to"
        )
    if policy.policy_name not in SELECTABLE:
        raise PolicyViolation(
            f"{policy.policy_name} is how a recorded history is read and not a policy a "
            "generation may be created under"
        )
    if row.cell not in policy.cells:
        raise PolicyViolation(
            f"{policy.policy_name} declares the cells {list(policy.cells)}, and the row for "
            f"attempt {row.attempt_id} names {row.cell!r}"
        )
    if policy.exposure == HONEST and grade.stand_in:
        raise PolicyViolation(
            f"{policy.policy_name} publishes the environment's grade, and this environment is "
            f"scored by {grade.grader_id}, which is a stand-in: its number is a fact about the "
            "shape of a filing rather than about the work in it"
        )
