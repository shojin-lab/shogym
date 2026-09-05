"""How a receipts attempt ends under the durable stream, and what it is worth.

The stream's stand-in grade computes from the shape of a filing: a terminal call carrying
something scores one and an empty one scores nothing. A filing is a table of answers, so under
the stand-in every attempt that files at all is worth one however many records it got right, and
the fraction it did get right, which is the quantity this environment exists to measure, reaches
the stream nowhere. That number cannot be published to an agent as a grade, because it is not one.

So this port replaces the environment's half of the terminal. The seal does what a v1 episode's
finalize does, once and under the seal id: it canonicalizes the filing, renders the fork through
the bank's own path, and scores what the parser made, and every later call under that key reads
what the first one wrote. The grade is a projection of that record, which is what makes a retry
return the first seal's numbers rather than a second reading of a world that may no longer be
there.

The seal reads the filing out of the terminal call rather than out of the world, and that is why
these two Activities are written here rather than assembled from :func:`terminal_for`. A receipts
world holds a table and an answer key, and it never holds the agent's filing at all: the terminal
call does not reach the environment under this protocol, so what an attempt filed exists in the
call and in the record this seal writes from it. The world is still resolved, because it is what
says which instance the filing is against, and a seal that reaches no world refuses rather than
scoring an empty table.

The headline is ``component_score``: the fraction of records the filing got right, equal weight
per row, over the printed row count, which is the number a v1 run seals. It goes out at six
places because that is where the scorer rounds it. The resolution is the scorer's own rather than
this port's second opinion, so the number a generation commits is the number a v1 run reports for
the same filing and not a coarser one that would have to be explained.

Beside it a body may say whether the whole table was right, and nothing else. The per-row
verdicts and the corrections are the receipt's, and the receipt is not this environment's to
deliver: an experiment decides which cell of the fork a branch is served, and an environment that
handed the grade back at the terminal would be putting a receipt in every arm, including the one
that is meant to be empty. So the acknowledgement carries the submission digest and the version it
was captured under and no verdict, and the three cells stay under the seal.

The seal id is what those cells are keyed by, and it is the only name here that identifies one of
them. A public attempt id does not: two executions of one attempt share it, and each has a filing
and a fork of its own, so cells looked up under it would be whichever execution wrote first. A
seal id is minted from the hidden execution, the ordinal of that execution and the attempt
together, which is exactly the identity a rendered cell belongs to. Where the records go is the
bank's own fork store, beside the committed fork whose bytes they are, so a Worker that replaced
the one which sealed reads what was sealed and a process that stopped did not take it with it.
Carrying that reference through the kernel to a renderer is the next piece of work rather than
this one: what this owes is cells that are still there, under a key one seal owns.

The horizon is the floor, and that is a decision rather than an omission. A graded horizon files
the terminal for the attempt as its last step commits, and the filing it makes is the world the
attempt left: the gateway writes no arguments into it, and it refuses a generation whose terminal
declares any. ``submit_filing`` declares the filing itself, so there is nothing at this horizon
for a gateway to file, and an attempt that reaches it filed nothing. That is what the floor
records, and it is the number a v1 episode reports for the same ending.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from shogym.envs._grading import (
    GRADED,
    SEALED,
    CaptureStore,
    WorldRoute,
    check_recovery_token,
    configuration_digest as digest_of,
    encoded,
    graded_result,
    installed,
    recovery_token,
    refusal,
    terminal_activities,
)
from shogym.serve.protocol_v2.kernel.messages import (
    GradeAttemptInput,
    GradeAttemptResult,
    SealAttemptInput,
    SealAttemptResult,
)
from shogym.serve.protocol_v2.policy import GradeIdentity, PublishedNumber

#: The version this environment declares for the submission its terminal captures. It names what
#: goes into the canonical text and what does not, so a run recorded under it is comparable with
#: another run recorded under it and with nothing else.
CANONICALIZATION_VERSION = "shogym.receipts.1"

#: The argument the score terminal is filed with. The stream holds a terminal call to the names
#: the tool declares, so this is where the filing arrives and the only place it exists.
FILING_ARGUMENT = "filing"

#: What the three cells this seal rendered are held under, inside the record the seal wrote.
CELLS = "cells"

#: How fine the headline is. The scorer rounds the fraction of records the filing got right to six
#: places, so six is what the number means: declaring fewer would publish a rounding of the
#: environment's own number rather than the number, and a run under this grade and a v1 run over
#: the same filing would then report two different scores for one filing.
_SCORE_PLACES = 6

#: What this environment's grader is. The score is the scorer's own verdict over the filing rather
#: than a fact about its shape, which is what lets a generation over this environment publish it.
#: The roster is the one number a body may print beside that score: the whole table was right or it
#: was not, which is a whole number between nothing and one, and a roster entry says so by
#: declaring no decimal places at all. What the receipt says row by row is not on the roster and
#: never becomes a number a body prints: an arm that carries the verdicts carries them as the
#: environment's own rendered cell, at the byte count the envelope fixes.
RECEIPTS_GRADE = GradeIdentity(
    grader_id="receipts-grade-v2",
    grader_version="1",
    stand_in=False,
    score_component="component_score",
    score_places=_SCORE_PLACES,
    public_components=(
        PublishedNumber(name="solved", minimum=0.0, maximum=1.0),
    ),
)


def receipts_terminal(
    route: WorldRoute, *, store: CaptureStore
) -> Tuple[str, List[Any]]:
    """The version this environment declares, and the Activities that end an attempt in it.

    ``route`` says which world an attempt was worked in, and it is asked when a seal has to read
    one rather than now: these Activities are registered once and a generation may serve several
    tasks, each in a world of its own.

    ``store`` is where the records go, and there is no default. What a seal writes here is the
    cells a later arm delivers as well as the numbers, so a caller has to say where they are kept
    rather than have somewhere chosen for it: a store this process holds and nothing else would
    make the evidence of an attempt a property of the Worker that happened to seal it. The
    environment answers with a directory in the bank's own fork store.
    """

    def seal(request: SealAttemptInput) -> SealAttemptResult:
        """Capture what this attempt filed, or return what its key already holds."""
        if request.canonicalization_version != CANONICALIZATION_VERSION:
            raise refusal(
                f"this world is captured as {CANONICALIZATION_VERSION!r} and the generation was "
                f"started as {request.canonicalization_version!r}",
                "CanonicalizationMismatch",
            )
        state = store.once(request.seal_id, SEALED, lambda: _filed(route, request))
        text = encoded(
            {
                "canonicalization_version": CANONICALIZATION_VERSION,
                "submission": _submission(state),
            }
        )
        return SealAttemptResult(
            attempt_id=request.attempt_id,
            seal_id=request.seal_id,
            canonicalization_version=CANONICALIZATION_VERSION,
            canonical_submission_text=text,
            canonical_submission=installed(request.blob_root, text),
            environment_recovery_token=recovery_token(request.seal_id, state),
        )

    def grade(request: GradeAttemptInput) -> GradeAttemptResult:
        """Score the capture, or return the score this key was already given."""
        state = store.held(request.seal_id, SEALED)
        if state is None:
            raise refusal(
                f"this machine holds nothing sealed under {request.seal_id}, so there is nothing "
                "here to grade",
                "NoSealedCapture",
            )
        check_recovery_token(request.seal_id, state, request.environment_recovery_token)
        verdict = store.once(request.seal_id, GRADED, lambda: _verdict(state))
        return graded_result(request, verdict=verdict, grade=RECEIPTS_GRADE)

    return CANONICALIZATION_VERSION, terminal_activities(seal=seal, grade=grade)


def cells_for(store: CaptureStore, seal_id: str) -> Optional[Dict[str, str]]:
    """The cells one seal rendered, or ``None`` where nothing was sealed under that key.

    The key is the seal rather than the attempt, because the seal is what one render belongs to:
    two executions of one attempt carry one public id between them and a filing and a fork each,
    so an answer looked up by attempt would be whichever of them wrote first. The bytes are the
    fork's own, as the seal committed them, and nothing here renders: a cell built a second time
    is how two branches of one fork come to differ.
    """
    held = store.held(seal_id, SEALED)
    if held is None:
        return None
    cells = held.get(CELLS) or {}
    return {str(kind): str(text) for kind, text in cells.items()}


def _filed(route: WorldRoute, request: SealAttemptInput) -> Dict[str, Any]:
    """Read the world this attempt was worked in, and what the filing in this call comes to.

    The two refusals are separate on purpose. A seal that resolves to no world at all reached a
    process that never served this attempt, which is what a resumed generation sees for every
    attempt; a seal that resolves to a world whose session is gone reached the right process after
    it let the world go. Both would otherwise be a filing scored against an instance nobody looked
    up, and a zero there is a grade: it says the agent got no record right, and what happened is
    that nobody read the table it was answering.
    """
    world = route(request.attempt_id)
    if world is None:
        raise refusal(
            f"attempt {request.attempt_id} was worked in no world this process opened, so there "
            "is nothing here to seal",
            "NoWorldHere",
        )
    env, session_id = world
    filed = env.sealed_filing(session_id, request.native_arguments.get(FILING_ARGUMENT))
    if filed is None:
        raise refusal(
            f"the world attempt {request.attempt_id} was worked in has been let go, so the "
            "instance this filing answers cannot be read: a score for a table nobody read is not "
            "a score",
            "NoSessionHere",
        )
    return filed


def _submission(state: Dict[str, Any]) -> Dict[str, Any]:
    """What the agent filed: the parser's canonical reading of it, and nothing the scorer made.

    The canonical text is what the digest covers and what a payload renderer is handed, so the
    numbers a grade is made of are not in it, and neither is the key: a run composed to withhold
    its score would otherwise be handing the renderer the score in the field beside the one it
    withheld, and one composed to serve a placebo would be handing it the answers.
    """
    return {"filing": str(state["filing"])}


def _verdict(state: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict this port commits, out of the record the seal wrote.

    ``decode_state`` is how the filing read rather than whether anything went wrong. A filing
    nothing scorable could be made of and a filing that was read and got no record right are both
    worth nothing and are different facts about the attempt, and both are answers the grade stands
    on: neither is retried and the score behind them is the score.

    What the cells say stays out of it. The verdict is installed as the run's own evidence, where
    a harness can resolve it and a renderer cannot, and the numbers on it are counts of what the
    reading had to decide: nothing here names an option of the convention or a record's correct
    value in any form, because a fork's own cells are where those live.
    """
    return {
        "component_score": float(state["component_score"]),
        "solved": 1.0 if bool(state["solved"]) else 0.0,
        "rows_filed": float(state["rows_filed"]),
        "rows_omitted": float(state["rows_omitted"]),
        "no_filing": state["no_filing"],
        "task_id": str(state["task_id"]),
        "filing_digest": str(state["filing_digest"]),
        "cell_digests": dict(state["cell_digests"]),
        "decode_state": "ambiguous_zero" if state["no_filing"] else "decoded",
    }


def configuration_digest(
    *, genre: str, side: Optional[str], source: str, dealable: bool
) -> str:
    """Return the digest of what this environment is configured as.

    The source decides which instances could be dealt and under which conventions, because a
    bundle is frozen at its digest and a bank is named by its own bytes, and the genre decides
    what a family means. A generation resumed against a different one of those is a different
    measurement, and it is refused rather than scored against a draw nobody made for it.

    Whether the source was admitted is here as well as which source it is. An unbundled bank and a
    verified bundle are two different claims about the same instances, and a run that resumed
    across that line would be reporting scores from a development draw under a generation composed
    over a dealt one.

    Which sibling a task is comes from the position it was served at rather than from here, so a
    generation over a whole family declares nothing about siblings at all: what would otherwise
    happen is that A and B could never sit in one generation, because the world opened for the
    second position would carry a configuration the generation was not started as and be refused.
    The narrowing an environment can still be given is here, because an environment narrowed to
    one sibling serves a different roster and a position in it means something else.
    """
    return digest_of(
        {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "genre": genre,
            "side": side,
            "source_digest": source,
            "dealable": bool(dealable),
        }
    )


__all__ = [
    "CANONICALIZATION_VERSION",
    "CELLS",
    "FILING_ARGUMENT",
    "RECEIPTS_GRADE",
    "cells_for",
    "configuration_digest",
    "receipts_terminal",
]
