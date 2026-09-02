"""How a yc_bench attempt ends under the durable stream, and what it is worth.

``submit`` takes no arguments. What the agent filed is the company it leaves behind, so the seal
reads the sim's own end state off the session's database while there is still one to read, keeps
it under the seal id, and names it in the acknowledgement by its digest. The grade is taken from
that record and from nothing live, which is what lets a retry return the first call's numbers
rather than a second read of a database that may already be gone.

The headline is not the benchmark's headline, and that is a decision rather than an omission. A v1
run reports ``reward`` as the company's final funds in cents, which is a number with no upper
bound and a genuine negative branch: bankruptcy is funds below zero. A score is one number in the
unit interval, so publishing funds would mean either inventing a ceiling nobody measured against
or clamping, and a clamped headline reports a company that did not exist. The canonical score is
``success`` instead: the run reached the end of its year solvent. The funds stay in the verdict
this port installs as evidence, where a reader with the run's store can find them and a body
cannot print them.

Upstream's terminal gate is kept whole. ``submit`` is callable at any time, so a solvent state on
its own is worth nothing: an agent that sealed on turn one would bank the starting two hundred
thousand without operating the company. Credit needs the sim to have actually ended, at the
horizon or in bankruptcy, and a premature seal reads as a filing that said nothing.

The gate decides the credit and not the facts. A body that answered a company sealed with money in
the bank with ``survived 0`` would be telling the agent something untrue about the world it left,
and it would say the same thing for a company that went broke. So the two published numbers are the
sim's own, as the seal read them, and the zero a premature filing earns is in the score.
"""

from __future__ import annotations

from hashlib import sha256
from typing import Any, Dict, List, Optional, Sequence, Tuple

from shogym.envs._grading import (
    CaptureStore,
    Grader,
    WorldRoute,
    configuration_digest as digest_of,
    encoded,
    terminal_for,
)
from shogym.serve.protocol_v2.policy import GradeIdentity, PublishedNumber

#: The version this environment declares for the submission its terminal captures. It names what
#: goes into the canonical text and what does not, so a run recorded under it is comparable with
#: another run recorded under it and with nothing else.
CANONICALIZATION_VERSION = "shogym.yc_bench.1"

#: The sim's own endings. Only these two are a genuine end of the one-year run, and anything else
#: means the agent stopped the company early, which upstream scores as nothing.
_TERMINAL_REASONS = ("horizon_end", "bankruptcy")

#: What this environment's grader is. The score is the run's own outcome rather than a fact about
#: the shape of a filing, which is what lets a generation over this environment publish it. The
#: two numbers a body may say beside it are what the sim measured of the state that was sealed,
#: each a whole number between nothing and one: the company was solvent, and the year ran out. The
#: score is whole by the same declaration and for the same reason: the run either finished its year
#: solvent or it did not, so a headline arriving with digits after the point is not a finer answer
#: to that question, it is something else written in the field the answer goes in.
YC_BENCH_GRADE = GradeIdentity(
    grader_id="yc-bench-grade-v2",
    grader_version="1",
    stand_in=False,
    score_component="success",
    score_places=0,
    public_components=(
        PublishedNumber(name="survived", minimum=0.0, maximum=1.0),
        PublishedNumber(name="horizon_reached", minimum=0.0, maximum=1.0),
    ),
)


def yc_bench_terminal(
    route: WorldRoute, *, store: Optional[CaptureStore] = None
) -> Tuple[str, List[Any]]:
    """The version this environment declares, and the Activities that end an attempt in it.

    ``route`` says which world an attempt was worked in, and it is asked when a seal has to read
    one rather than now: these Activities are registered once and a generation may serve several
    tasks, each in a simulation of its own.
    """
    from shogym.envs.yc_bench import mcp_server

    return terminal_for(
        Grader(
            version=CANONICALIZATION_VERSION,
            grade=YC_BENCH_GRADE,
            read=mcp_server.sealed_state,
            submission=_submission,
            score=_score,
        ),
        route,
        store=store,
    )


def _submission(state: Dict[str, Any]) -> Dict[str, Any]:
    """What the agent filed: the company it left, named by its own bytes and not carried.

    The end state is the whole of the material a grade is taken from, so what the canonical text
    holds is a name for it. A body that publishes the score publishes it because the policy the
    obligation resolved to says so, and never because a submission handed the renderer the numbers
    a blinded run was composed to withhold.
    """
    return {"end_state_sha256": sha256(encoded(state).encode("utf-8")).hexdigest()}


def _score(state: Dict[str, Any]) -> Dict[str, Any]:
    """The verdict this port commits, out of the end state the seal read.

    A solvent company before the horizon is a run the agent stopped rather than one it finished, so
    nothing is credited for it: the score needs a sim that actually ended, exactly as the v1 scorer
    needs one. The two numbers beside the score are measurements rather than credit, so they are
    the sim's own answers for the state that was sealed, and a company with money in the bank is
    published as one whether or not its year ran out. The funds, the task counts and the clock stay
    here in the evidence and reach no body.
    """
    ended = str(state.get("terminal_reason") or "") in _TERMINAL_REASONS
    survived = bool(state.get("survived"))
    horizon_reached = bool(state.get("horizon_reached"))
    return {
        "success": 1.0 if ended and survived and horizon_reached else 0.0,
        "survived": 1.0 if survived else 0.0,
        "horizon_reached": 1.0 if horizon_reached else 0.0,
        "final_funds_cents": float(state.get("final_funds_cents") or 0),
        "tasks_succeeded": float(state.get("tasks_succeeded") or 0),
        "tasks_failed": float(state.get("tasks_failed") or 0),
        "terminal_reason": state.get("terminal_reason"),
        "sim_time": state.get("sim_time"),
        "decode_state": "decoded" if ended else "ambiguous_zero",
    }


def configuration_digest(
    *,
    task_split: str,
    config_name: str,
    start_date: str,
    horizon_years: Optional[int],
    company_name: str,
    seeds: Sequence[int],
) -> str:
    """Return the digest of what this environment is configured as.

    The upstream commit decides what the simulation is, the preset and the horizon decide how long
    a year lasts and what it costs, and the seeds decide which market a task draws. A generation
    resumed against a different one of those is a different measurement, and it is refused rather
    than scored against a rule nobody drew for it.
    """
    from shogym.envs.yc_bench import adapter

    return digest_of(
        {
            "canonicalization_version": CANONICALIZATION_VERSION,
            "upstream_sha": adapter.UPSTREAM_SHA,
            "task_split": task_split,
            "config_name": config_name,
            "start_date": start_date,
            "horizon_years": horizon_years,
            "company_name": company_name,
            "seeds": [int(seed) for seed in seeds],
        }
    )


__all__ = [
    "CANONICALIZATION_VERSION",
    "YC_BENCH_GRADE",
    "configuration_digest",
    "yc_bench_terminal",
]
