"""How an automationbench attempt ends under the durable stream, and what it is worth.

``done`` takes no arguments. What the agent filed is the workspace it left behind, so there is
nothing in the call to hash and the submission has to be synthesized from the world: it is that
world's own digest, taken at the seal and named in the acknowledgement before the score reaches
anything. The verdict is the upstream rubric's, run against this session's private world through
the one scoring entry point the port already had, so there is no second copy of the scorer here to
drift from the one a v1 run uses.

The rubric runs at the seal rather than at the grade, and that is a property of this world rather
than a shortcut. A ``WorldState`` the served tools mutated does not survive a serialize and
revalidate round trip: part of what the rubric reads is recorded outside the model's declared
fields, and revalidation can reject values a live world legitimately holds. So the one moment the
end state can be read is while the session is open, and what is kept under the seal id is what was
read then. The grade is a pure projection of that record, which is what makes a retry return the
first call's numbers rather than a second look at a world that may no longer be there.

The headline is ``partial_credit``: the fraction of the task's assertions the end state satisfies,
which is the number a v1 run reports as its reward. Beside it a body may say whether every
assertion passed, and nothing else: the assertions themselves, the target values and the world are
the answer key, and they stay in the verdict this port installs as evidence.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from shogym.envs._grading import (
    CaptureStore,
    Grader,
    WorldRoute,
    configuration_digest as digest_of,
    terminal_for,
)
from shogym.serve.protocol_v2.policy import GradeIdentity, PublishedNumber

#: The version this environment declares for the submission its terminal captures. It names what
#: goes into the canonical text and what does not, so a run recorded under it is comparable with
#: another run recorded under it and with nothing else.
CANONICALIZATION_VERSION = "shogym.automationbench.1"

#: How fine the headline is. ``partial_credit`` is the count of assertions the end state satisfied
#: over the count that were scored, and a division of two small whole numbers runs to as many
#: digits as a double holds. The digits past the fourth say nothing about how many assertions
#: passed: they are room, in a field an agent reads. So the number that goes out is that fraction
#: to four places, and the division behind it stays in the verdict this port installs as evidence,
#: where a harness resolving it finds the number the rubric actually returned.
_SCORE_PLACES = 4

#: What this environment's grader is. The score is the upstream rubric's own verdict over the
#: workspace the agent left, rather than a fact about the shape of a filing, which is what lets a
#: generation over this environment publish it. The roster is the one number a body may print
#: beside that score, declared with the domain it lies in: an assertion set is either wholly
#: satisfied or it is not, so the number is a whole one between nothing and one, which is what a
#: roster entry says by declaring no decimal places at all. The headline is a fraction of the same
#: assertions, so it is declared at the resolution the fraction means rather than at the one a
#: division produces.
AUTOMATIONBENCH_GRADE = GradeIdentity(
    grader_id="automationbench-grade-v2",
    grader_version="1",
    stand_in=False,
    score_component="partial_credit",
    score_places=_SCORE_PLACES,
    public_components=(
        PublishedNumber(name="success", minimum=0.0, maximum=1.0),
    ),
)


def automationbench_terminal(
    route: WorldRoute, *, store: Optional[CaptureStore] = None
) -> Tuple[str, List[Any]]:
    """The version this environment declares, and the Activities that end an attempt in it.

    ``route`` says which world an attempt was worked in, and it is asked when a seal has to read
    one rather than now: these Activities are registered once and a generation may serve several
    tasks, each in a world of its own.
    """
    from shogym.envs.automationbench import mcp_server

    return terminal_for(
        Grader(
            version=CANONICALIZATION_VERSION,
            grade=AUTOMATIONBENCH_GRADE,
            read=mcp_server.sealed_state,
            submission=_submission,
            score=_score,
        ),
        route,
        store=store,
    )


def _submission(state: Dict[str, Any]) -> Dict[str, Any]:
    """What the agent filed: the workspace it left, named by its own bytes and not carried.

    The digest is the whole of it. The state beside it holds the rubric's numbers, and those are
    what the grade publishes under the policy a generation resolved to rather than something a
    submission hands the renderer whatever that policy says.
    """
    return {"world_sha256": str(state["world_sha256"])}


def _score(state: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict this port commits, out of the rubric numbers the seal read.

    ``decode_state`` is how the filing read rather than whether anything went wrong. A workspace
    the agent never changed is a filing that said nothing, which is a different fact from a
    workspace it changed and got no credit for, and both are answers the grade stands on: neither
    is retried and the score behind them is the score.
    """
    return {
        "partial_credit": float(state["partial_credit"]),
        "success": float(state["success"]),
        "world_sha256": str(state["world_sha256"]),
        "decode_state": "ambiguous_zero" if bool(state["untouched"]) else "decoded",
    }


def configuration_digest(
    *, domain: str, max_steps: int, tasks: List[Dict[str, Any]]
) -> str:
    """Return the digest of what this environment is configured as.

    The upstream commit decides what the rubric is, the domain and the rows decide what could be
    asked, and the step budget decides what could be done about it. A generation resumed against a
    different one of those is a different measurement, and it is refused rather than scored against
    a rule nobody drew for it. Each row is covered by the identifiers upstream gives it and by the
    world it seeds, which is the material a score depends on.
    """
    from shogym.envs.automationbench import adapter

    return digest_of(
        {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "upstream_sha": adapter.UPSTREAM_SHA,
            "domain": domain,
            "max_steps": int(max_steps),
            "task_count": len(tasks),
            "tasks": [
                {
                    "example_id": str(row.get("example_id")),
                    "task": str(row.get("task")),
                    "info": row.get("info", {}),
                }
                for row in tasks
            ],
        }
    )


__all__ = [
    "AUTOMATIONBENCH_GRADE",
    "CANONICALIZATION_VERSION",
    "automationbench_terminal",
    "configuration_digest",
]
