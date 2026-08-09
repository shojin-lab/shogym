"""``orca_bench`` grading: judge preflight, verdict parsing, and the scored feedback.

The shapes here are modelled on three real verifier runs recorded during the port's feasibility
spike (a correct report, a wrong report, and a judge that failed), with the payloads rewritten
around invented flags so no real answer is committed. The one that matters is the third: upstream
writes ``{"reward": 0.0}`` for a failed judge, byte-identical to what it writes for a wrong
report, and this port must not inherit that ambiguity.

All offline: no stack, no key, no network.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Dict, List, Optional

import pytest

from shogym.envs.orca_bench import backend, judge
from shogym.envs.orca_bench.env_v1 import public_verdict, score_evidence
from shogym.serve.lifecycle import TerminalEvidence

# A graded run: reward.json always carries `reward`, plus the rollups that are defined.
REWARD_PARTIAL = {"reward": 0.4166666666666667, "rca_accuracy": 0, "hallucinate_any": 0}
REWARD_WRONG = {"reward": 0.0, "rca_accuracy": 0, "hallucinate_any": 1}
REWARD_ALL_CAUSES = {"reward": 1.0, "rca_accuracy": 1, "hallucinate_any": 0}
# A control task: hallucination is undefined there (no plausible root cause to match against), so
# the verifier omits the key rather than writing a null Harbor would reject.
REWARD_CONTROL_PASS = {"reward": 1.0, "rca_accuracy": 0}
# The judge's failure shape, verbatim in structure: reward.json is a bare zero and the details
# file carries the exception.
REWARD_JUDGE_FAILED = {"reward": 0.0}
# What the verifier writes when it cannot read the agent's report: the same shape a dead judge
# model produces, which is the whole difficulty.
DETAILS_MISSING_REPORT = {
    "reward": 0.0,
    "error": "FileNotFoundError: [Errno 2] No such file or directory: '/app/report.md'",
    "traceback": "Traceback (most recent call last):\n  ...\n",
}
DETAILS_JUDGE_FAILED = {
    "reward": 0.0,
    "error": (
        "BadRequestError: Error code: 400 - {'error': {'message': \"The requested model "
        "'openai-gpt-5.4' does not exist.\"}}"
    ),
    "traceback": "Traceback (most recent call last):\n  ...\n",
}


def _details(*scores: int) -> Dict[str, Any]:
    """A judged details file with one rubric per score (the shape reward is derived from)."""
    return {
        "mode": "llm_judge",
        "model": "gpt-5.6-terra",
        "nested": {
            "rubrics": [
                {"feature_flag": f"syntheticFlag{i}", "score": score}
                for i, score in enumerate(scores)
            ]
        },
    }


def _evidence(verdict: Optional[Dict[str, Any]], *, status: str = "ok") -> TerminalEvidence:
    return TerminalEvidence(source="explicit_tool", status=status, verdict=verdict)


def _published(
    verdict: judge.OrcaVerdict, *, is_control: bool = False, config: Optional[judge.JudgeConfig] = None
) -> Dict[str, Any]:
    """The public verdict as `finalize` builds it, judged by a config resolved against an empty
    environment (so the tests never depend on the process's own OPENAI_* variables)."""
    return public_verdict(
        verdict, is_control=is_control, judge=(config or judge.JudgeConfig()).resolve({})
    )


# ----- judge preflight -----


def test_preflight_refuses_the_retired_upstream_default() -> None:
    config = judge.JudgeConfig(model=judge.UPSTREAM_DEFAULT_JUDGE_MODEL)
    with pytest.raises(judge.JudgePreflightError, match="retired it"):
        judge.validate_judge_config(config)


def test_preflight_refuses_a_missing_key_but_not_a_base_url() -> None:
    config = judge.JudgeConfig()
    with pytest.raises(judge.JudgePreflightError, match="OPENAI_API_KEY"):
        judge.preflight_judge(config, environ={})
    judge.preflight_judge(config, environ={"OPENAI_API_KEY": "sk-test"})
    judge.preflight_judge(judge.JudgeConfig(base_url="http://localhost:1234/v1"), environ={})


def test_preflight_refuses_an_unknown_effort() -> None:
    with pytest.raises(judge.JudgePreflightError, match="judge_effort"):
        judge.validate_judge_config(judge.JudgeConfig(effort="ludicrous"))


def test_a_keyless_endpoint_still_gets_a_credential_the_sdk_will_accept() -> None:
    """The keyless path has to survive the verifier, not just the preflight.

    The verifier constructs its client as ``AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))``
    unconditionally, and the SDK refuses to construct on a missing or empty key before it ever
    reaches the endpoint. So a config the preflight accepts as keyless must hand the verifier a
    non-empty placeholder, or the endpoint is never contacted and every task lands on the
    judge-error grade."""
    config = judge.JudgeConfig(base_url="http://localhost:1234/v1")
    judge.preflight_judge(config, environ={})
    assert config.environment({}) == {
        "OPENAI_API_KEY": judge.PLACEHOLDER_API_KEY,
        "OPENAI_BASE_URL": "http://localhost:1234/v1",
    }


def test_a_real_key_is_never_replaced_by_the_placeholder() -> None:
    config = judge.JudgeConfig(base_url="http://localhost:1234/v1")
    assert config.environment({"OPENAI_API_KEY": "sk-test"})["OPENAI_API_KEY"] == "sk-test"


def test_the_placeholder_is_only_for_an_explicit_endpoint() -> None:
    # With no base_url there is nowhere keyless to point at, so a placeholder would just send an
    # invalid credential to the real API. That case must keep failing the preflight instead.
    config = judge.JudgeConfig()
    assert config.environment({}) == {}
    with pytest.raises(judge.JudgePreflightError, match="OPENAI_API_KEY"):
        judge.preflight_judge(config, environ={})


def test_preflight_and_environment_agree_on_a_base_url_from_the_process_environment() -> None:
    # `environment()` honors OPENAI_BASE_URL from the environment, so the preflight must too:
    # otherwise the documented keyless setup is refused for a config that would have worked.
    config = judge.JudgeConfig()
    environ = {"OPENAI_BASE_URL": "http://localhost:1234/v1"}
    judge.preflight_judge(config, environ=environ)
    assert config.environment(environ) == {
        "OPENAI_API_KEY": judge.PLACEHOLDER_API_KEY,
        "OPENAI_BASE_URL": "http://localhost:1234/v1",
    }


def test_judge_config_carries_the_verifier_arguments() -> None:
    config = judge.JudgeConfig()
    assert config.cli_args() == ["--model", judge.DEFAULT_JUDGE_MODEL, "--effort", "high"]
    assert config.model != judge.UPSTREAM_DEFAULT_JUDGE_MODEL
    env = config.environment({"OPENAI_API_KEY": "sk-test"})
    assert env == {"OPENAI_API_KEY": "sk-test"}


def test_env_construction_refuses_a_judge_that_cannot_grade(tmp_path: Path) -> None:
    import shogym
    from tests._fixtures import orca_bench_dataset as synth

    root = synth.write_dataset(tmp_path / "orca")
    with pytest.raises(judge.JudgePreflightError):
        shogym.make(
            "orca_bench",
            config={
                "dataset_dir": str(root),
                "judge_model": judge.UPSTREAM_DEFAULT_JUDGE_MODEL,
            },
        )
    # ...but a valid model needs no key to construct or describe: only serving does.
    env = shogym.make("orca_bench", config={"dataset_dir": str(root)})
    assert env.describe("0").instructions


class _ShiftingEnviron(Mapping[str, str]):
    """An environment that answers once and then forgets, so a second read is visible.

    The finding is that three separate readings of ``OPENAI_BASE_URL`` decided three different
    things. A mapping that gives a different answer the second time turns that into an assertion
    instead of a race."""

    def __init__(self, key: str, first: str, base: Optional[Dict[str, str]] = None) -> None:
        self._key = key
        self._first = first
        self._base = dict(base or {})
        self.reads = 0

    def __getitem__(self, key: str) -> str:
        if key == self._key:
            self.reads += 1
            if self.reads > 1:
                raise KeyError(key)
            return self._first
        return self._base[key]

    def __iter__(self):
        return iter({**self._base, self._key: self._first})

    def __len__(self) -> int:
        return len(self._base) + 1


def test_one_reading_of_the_environment_decides_the_whole_judge() -> None:
    """Preflight, the verifier's environment and the audited endpoint come from one snapshot.

    Three reads of a mutable environment can disagree, and this one is the worst kind of
    disagreement: the run is equipped with one endpoint and the score is audited against another,
    which is precisely the confusion round four was supposed to end."""
    endpoint = "https://judge-a.internal/v1"
    shifting = _ShiftingEnviron("OPENAI_BASE_URL", endpoint)
    snapshot = judge.environment_snapshot(shifting)
    assert shifting.reads == 1

    config = judge.JudgeConfig()
    judge.preflight_judge(config, environ=snapshot)  # keyless, so it passes only via the endpoint
    resolved = config.resolve(snapshot)
    assert resolved.endpoint_id == judge.JudgeConfig(base_url=endpoint).endpoint_id({})
    assert resolved.environment["OPENAI_BASE_URL"] == endpoint
    assert resolved.environment["OPENAI_API_KEY"] == judge.PLACEHOLDER_API_KEY
    assert shifting.reads == 1  # still one: everything above derives from the snapshot

    # And a caller who hands over a live mapping gets one reading as well, rather than one per
    # derivation: this is the shape the finding was about.
    live = _ShiftingEnviron("OPENAI_BASE_URL", endpoint)
    from_live = config.resolve(live)
    assert live.reads == 1
    assert from_live.endpoint_id == resolved.endpoint_id
    assert from_live.environment["OPENAI_BASE_URL"] == endpoint


def test_the_session_takes_exactly_one_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """And the episode takes that one reading itself, sharing it with everything downstream."""
    import shogym
    from shogym.envs.orca_bench import judge as judge_module
    from shogym.envs.orca_bench import mcp_server
    from tests._fixtures import orca_bench_dataset as synth

    endpoint = "https://judge-a.internal/v1"
    taken: List[Mapping[str, str]] = []
    real_snapshot = judge_module.environment_snapshot

    def _snapshot(environ: Optional[Mapping[str, str]] = None) -> Mapping[str, str]:
        # Stands in for the process environment on the episode's own reading, and passes an
        # existing snapshot straight through, exactly as the real one does.
        snapshot = real_snapshot(environ if environ is not None else {"OPENAI_BASE_URL": endpoint})
        taken.append(snapshot)
        return snapshot

    preflighted: List[Optional[Mapping[str, str]]] = []
    real_preflight = judge_module.preflight_judge

    def _preflight(config: judge.JudgeConfig, *, environ: Optional[Mapping[str, str]] = None) -> None:
        preflighted.append(environ)
        real_preflight(config, environ=environ)

    monkeypatch.setattr(judge_module, "environment_snapshot", _snapshot)
    monkeypatch.setattr(judge_module, "preflight_judge", _preflight)
    captured: Dict[str, Any] = {}
    monkeypatch.setattr(
        mcp_server, "begin_session", lambda _session_id, **kwargs: captured.update(kwargs)
    )

    env = shogym.make(
        "orca_bench", config={"dataset_dir": str(synth.write_dataset(tmp_path / "orca"))}
    )
    env.begin_session("session-1", env.load_task(0))

    assert taken, "the episode never read the environment"
    # One reading: every derivation downstream is the same frozen object, not a fresh look.
    assert len({id(snapshot) for snapshot in taken}) == 1
    assert preflighted == [taken[0]], "the preflight decided on a different reading"
    resolved = captured["judge"]
    assert resolved.environment["OPENAI_BASE_URL"] == endpoint
    assert resolved.endpoint_id == judge.JudgeConfig(base_url=endpoint).endpoint_id({})


# ----- verdict parsing -----


def test_a_graded_run_parses_its_rollups() -> None:
    verdict = judge.parse_verdict(REWARD_PARTIAL, _details(2, 3, 0, 0))
    assert verdict.judge_error is False
    assert verdict.reward == pytest.approx(0.4166666, rel=1e-5)
    assert verdict.rca_accuracy is False
    assert verdict.hallucinate_any is False
    assert verdict.mode == "llm_judge"
    # rca_depth is the per-rubric mean the reward is derived from (reward = depth / 3).
    assert verdict.rca_depth == pytest.approx(1.25)
    assert verdict.rca_depth is not None and verdict.rca_depth / 3 == pytest.approx(verdict.reward)


def test_a_wrong_report_is_an_honest_zero() -> None:
    verdict = judge.parse_verdict(REWARD_WRONG, _details(0, 0))
    assert verdict.judge_error is False
    assert verdict.reward == 0.0
    assert verdict.hallucinate_any is True
    assert verdict.success(is_control=False) is False


def test_naming_every_cause_is_the_success_criterion() -> None:
    verdict = judge.parse_verdict(REWARD_ALL_CAUSES, _details(3, 3))
    assert verdict.rca_accuracy is True
    assert verdict.success(is_control=False) is True


def test_a_control_task_succeeds_on_reward_not_on_rca_accuracy() -> None:
    """The verifier synthesizes one ``(no_incident)`` rubric whose ``feature_flag_match`` is
    hard-coded False, so ``rca_accuracy`` is 0 on every control task however well the agent did.
    Reading success off it would score all 138 controls as failures."""
    verdict = judge.parse_verdict(REWARD_CONTROL_PASS, _details(3))
    assert verdict.rca_accuracy is False
    assert verdict.success(is_control=True) is True
    assert verdict.success(is_control=False) is False
    missed = judge.parse_verdict({"reward": 0.0, "rca_accuracy": 0}, _details(0))
    assert missed.success(is_control=True) is False


@pytest.mark.parametrize(
    "reward,details",
    [
        # The recorded failure: a bare zero plus the exception in the details file.
        (REWARD_JUDGE_FAILED, DETAILS_JUDGE_FAILED),
        # The same bare zero with no details file at all: indistinguishable from a wrong answer
        # by reward alone, which is the whole point.
        (REWARD_JUDGE_FAILED, None),
        # The verifier never wrote anything.
        (None, None),
        # A judgement that scored no rubric: `rca_accuracy` is the key that says one was scored.
        ({"reward": 0.0}, {"mode": "llm_judge", "nested": {"rubrics": []}}),
        # Junk in the reward file.
        ({"reward": "nan-ish"}, None),
    ],
)
def test_a_judge_failure_is_an_explicit_grade_never_a_silent_zero(
    reward: Optional[Dict[str, Any]], details: Optional[Dict[str, Any]]
) -> None:
    verdict = judge.parse_verdict(reward, details)
    assert verdict.judge_error is True
    assert verdict.judge_error_message
    assert verdict.reward == 0.0
    assert verdict.rca_accuracy is None
    assert verdict.success(is_control=False) is False
    assert verdict.success(is_control=True) is False


def test_the_judge_failure_message_names_the_cause() -> None:
    verdict = judge.parse_verdict(REWARD_JUDGE_FAILED, DETAILS_JUDGE_FAILED)
    assert "openai-gpt-5.4" in verdict.judge_error_message


def test_a_wrong_answer_and_a_failed_judge_are_told_apart() -> None:
    # Both write reward 0.0; only one of them is a grade.
    wrong = judge.parse_verdict(REWARD_WRONG, _details(0, 0))
    failed = judge.parse_verdict(REWARD_JUDGE_FAILED, DETAILS_JUDGE_FAILED)
    assert wrong.reward == failed.reward == 0.0
    assert (wrong.judge_error, failed.judge_error) == (False, True)


# ----- a grade has to be in the contract to be a grade -----


@pytest.mark.parametrize(
    "reward_json,why",
    [
        ({"reward": 2.0, "rca_accuracy": 1}, "above the documented range"),
        ({"reward": -0.5, "rca_accuracy": 1}, "below it"),
        ({"reward": float("nan"), "rca_accuracy": 1}, "not a number at all"),
        ({"reward": float("inf"), "rca_accuracy": 1}, "not finite"),
        ({"reward": 1.0, "rca_accuracy": None}, "no rubric verdict, only the key"),
        ({"reward": 1.0, "rca_accuracy": 2}, "not a 0/1 rollup"),
        ({"reward": 1.0, "rca_accuracy": "1"}, "not a number"),
        ({"reward": 1.0, "rca_accuracy": 1, "hallucinate_any": 7}, "rollup out of contract"),
    ],
)
def test_a_metric_outside_its_contract_is_a_judge_error(
    reward_json: Dict[str, Any], why: str
) -> None:
    """These are verifier failure shapes, not grades.

    Reward is documented as a normalized [0, 1] score and the rollups as flat 0/1 values, so a
    payload outside that says the verifier did something other than grade this report. Accepting
    it would put a number nobody can interpret into a published result, and truthiness would turn
    `rca_accuracy: 2` into a pass."""
    verdict = judge.parse_verdict(reward_json, _details(3))
    assert verdict.judge_error is True, why
    assert verdict.reward == 0.0
    assert verdict.rca_accuracy is None
    assert verdict.success(is_control=False) is False


def test_nan_cannot_slip_through_a_range_check() -> None:
    # Every comparison with NaN is false, so `value < 0 or value > 1` waves it through. The check
    # is written as "is it finite and inside the range", and this pins that it stays that way.
    assert not (float("nan") < 0.0 or float("nan") > 1.0)  # the naive form accepts it
    assert judge.parse_verdict({"reward": float("nan"), "rca_accuracy": 1}).judge_error is True


def test_a_rejected_payload_is_named_in_private_only_and_truncated() -> None:
    huge = {"reward": 3.0, "rca_accuracy": 1, "noise": "x" * 5000}
    verdict = judge.parse_verdict(huge, _details(3))
    assert "3.0" in verdict.judge_error_message
    assert len(verdict.judge_error_message) < 600  # the payload is summarized, not pasted
    published = _published(verdict)
    assert "noise" not in json.dumps(published) and "3.0" not in json.dumps(published)


@pytest.mark.parametrize(
    "value,expected", [(1, True), (0, False), (True, True), (False, False), (1.0, True), (0.0, False)]
)
def test_the_rollups_accept_exactly_zero_and_one(value: Any, expected: bool) -> None:
    verdict = judge.parse_verdict({"reward": 0.5, "rca_accuracy": value}, _details(2))
    assert verdict.judge_error is False
    assert verdict.rca_accuracy is expected


@pytest.mark.parametrize("bound", [0.0, 1.0, 0.4166666666666667])
def test_rewards_inside_the_documented_range_are_grades(bound: float) -> None:
    verdict = judge.parse_verdict({"reward": bound, "rca_accuracy": 0}, _details(1))
    assert verdict.judge_error is False and verdict.reward == bound


def test_read_verdict_reads_the_two_files(tmp_path: Path) -> None:
    import json

    (tmp_path / judge.REWARD_FILENAME).write_text(json.dumps(REWARD_ALL_CAUSES))
    (tmp_path / judge.REWARD_DETAILS_FILENAME).write_text(json.dumps(_details(3, 3)))
    verdict = judge.read_verdict(tmp_path)
    assert verdict.judge_error is False and verdict.rca_accuracy is True
    # A verifier directory with nothing in it is a judge error, not a zero.
    assert judge.read_verdict(tmp_path / "absent").judge_error is True


# ----- a failed submission is a failure, not an excluded attempt -----


def test_a_missing_report_is_an_ordinary_failure_not_a_judge_error(tmp_path: Path) -> None:
    """The exclusion must not be reachable from the agent's own side of the boundary.

    Upstream's verifier reads `/app/report.md` inside the same blanket `except` that wraps the
    judge call, so an absent report produces the identical `{reward: 0.0, error, traceback}` shape
    a dead judge model does. Judge errors are filtered out of results by this port's own contract,
    so an agent that simply never writes a report would convert a failed attempt into no attempt
    at all. That has to be a graded zero."""
    reason = judge.inspect_report(tmp_path / "report.md")
    assert reason and "no report" in reason.lower()

    verdict = judge.parse_verdict(REWARD_JUDGE_FAILED, DETAILS_MISSING_REPORT, submission_error=reason)
    assert verdict.judge_error is False, "an absent report is not a grading-infrastructure failure"
    assert verdict.submission_error == reason
    assert verdict.reward == 0.0
    assert verdict.rca_accuracy is False
    assert verdict.success(is_control=False) is False
    assert verdict.success(is_control=True) is False

    values = _values(score_evidence(_evidence(_published(verdict))))
    assert values["verified"] is True, "the attempt was graded, not lost"
    assert "judge_error" not in values
    assert values["submission_error"] == reason


def test_an_empty_report_is_a_valid_submission(tmp_path: Path) -> None:
    # The required answer for a control task is an empty file. The check for "nothing here to
    # grade" must not eat the one submission whose emptiness is the point.
    empty = tmp_path / "report.md"
    empty.write_text("", encoding="utf-8")
    assert judge.inspect_report(empty) is None


def test_a_report_that_is_not_a_regular_file_is_a_failed_submission(tmp_path: Path) -> None:
    directory = tmp_path / "as-a-directory"
    directory.mkdir()
    assert judge.inspect_report(directory)

    target = tmp_path / "elsewhere.md"
    target.write_text("# report", encoding="utf-8")
    link = tmp_path / "as-a-symlink"
    link.symlink_to(target)
    reason = judge.inspect_report(link)
    assert reason and "regular file" in reason


def test_an_unreadable_report_is_a_failed_submission(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_bytes(b"\xff\xfe not utf-8 \x00")
    reason = judge.inspect_report(report)
    assert reason and "read" in reason.lower()


def test_an_oversized_report_is_a_failed_submission(tmp_path: Path) -> None:
    """The same manufactured exclusion is available at the other end: a report too large for the
    judge's request fails inside that blanket handler too."""
    report = tmp_path / "report.md"
    report.write_bytes(b"x" * (judge.MAX_REPORT_BYTES + 1))
    reason = judge.inspect_report(report)
    assert reason and str(judge.MAX_REPORT_BYTES) in reason
    assert str(judge.MAX_REPORT_BYTES + 1) in reason

    report.write_bytes(b"x" * judge.MAX_REPORT_BYTES)
    assert judge.inspect_report(report) is None  # exactly at the bound is still a submission


def test_the_bound_is_far_above_a_real_report() -> None:
    # Sized from what the judge actually consumes, not invented: the recorded prompts carry ~65 KB
    # of rubrics plus a report of a few KB.
    assert judge.MAX_REPORT_BYTES == 256 * 1024


def test_a_real_judge_failure_is_still_a_judge_error() -> None:
    # The sweep's other side: nothing about this fix should reclassify grading-infrastructure
    # failures, which stay excluded.
    verdict = judge.parse_verdict(REWARD_JUDGE_FAILED, DETAILS_JUDGE_FAILED)
    assert verdict.judge_error is True and verdict.submission_error == ""


async def test_a_finalized_missing_report_is_graded_not_excluded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end: the backend reports the submission problem and the episode is scored on it."""
    import shogym
    from shogym.envs.orca_bench import mcp_server
    from shogym.serve.lifecycle import FinalizeRequest
    from tests._fixtures import orca_bench_dataset as synth

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setattr(mcp_server, "begin_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mcp_server,
        "finalize_session",
        lambda _session_id: {
            "reward": REWARD_JUDGE_FAILED,
            "details": DETAILS_MISSING_REPORT,
            "submission_error": "no report was written at /app/report.md",
        },
    )

    env = shogym.make(
        "orca_bench", config={"dataset_dir": str(synth.write_dataset(tmp_path / "orca"))}
    )
    env.begin_session("session-1", env.load_task(0))
    evidence = await env.finalize(
        FinalizeRequest(source="explicit_tool", finalization_id="f1", session_id="session-1")
    )

    assert evidence.status == "ok", "a failed submission is a grade, not a finalize error"
    assert evidence.verdict["judge_error"] is False
    assert evidence.verdict["success"] is False and evidence.verdict["reward"] == 0.0
    assert "report" in evidence.verdict["submission_error"]


def _capture_within(report: Path, seconds: float) -> judge.CapturedReport:
    """Capture on a daemon thread, so a regression that blocks fails this test fast.

    A blocking ``open`` cannot be cancelled from the caller, so the guard has to be a thread that
    the interpreter will not wait for. `asyncio.wait_for` is what this repo uses for hang-risk
    tests, but it would leave a non-daemon executor thread stuck in the syscall and the process
    would then hang at exit, which is exactly what a regression check must not do."""
    outcome: List[judge.CapturedReport] = []
    worker = threading.Thread(target=lambda: outcome.append(judge.capture_report(report)), daemon=True)
    worker.start()
    worker.join(timeout=seconds)
    assert outcome, f"capture_report did not return within {seconds}s"
    return outcome[0]


def test_a_fifo_report_is_refused_rather_than_waited_on(tmp_path: Path) -> None:
    """A named pipe with no writer must not strand the terminal call.

    Opening a FIFO read-only blocks until a writer appears, so `mkfifo /app/report.md` and then
    submitting would park the finalize inside the syscall: no verdict, and a teardown that drains
    the evaluator waits on it too. The diagnostic open is non-blocking, and the regular-file check
    happens before any read, so a FIFO opens immediately and is refused as an ordinary failed
    submission."""
    fifo = tmp_path / "report.md"
    os.mkfifo(fifo)

    captured = _capture_within(fifo, seconds=10.0)
    assert captured.problem and "regular file" in captured.problem
    assert captured.data is None
    # And it is a graded zero, not the exclusion a stranded or crashed finalize would produce.
    verdict = judge.parse_verdict(None, None, submission_error=captured.problem)
    assert verdict.judge_error is False and verdict.reward == 0.0


def test_a_device_report_is_refused_rather_than_waited_on() -> None:
    # The same hazard through a different file type: character devices can also block on open.
    captured = _capture_within(Path("/dev/zero"), seconds=10.0)
    assert captured.problem and "regular file" in captured.problem


def test_a_regular_file_reads_whole_under_the_non_blocking_flag(tmp_path: Path) -> None:
    """O_NONBLOCK is specified to have no effect on regular files, and this pins it on every
    platform the suite runs on rather than taking the specification's word for it."""
    report = tmp_path / "report.md"
    payload = ("# Incident Report\n" + "x" * 63 + "\n") * (judge.MAX_REPORT_BYTES // 82)
    payload = (payload + "y" * judge.MAX_REPORT_BYTES)[: judge.MAX_REPORT_BYTES].encode("utf-8")
    assert len(payload) == judge.MAX_REPORT_BYTES
    report.write_bytes(payload)

    captured = _capture_within(report, seconds=10.0)
    assert captured.problem == ""
    assert captured.data == payload, "the non-blocking read returned a short or altered report"


# ----- the bytes that were sealed are the bytes that get graded -----


class _FakeBackend:
    """A backend whose report lives at a path the "agent" can still change after the seal."""

    def __init__(self, report: Path) -> None:
        self.report = report
        self.verified_with: Optional[judge.CapturedReport] = None

    def capture_report(self) -> judge.CapturedReport:
        return judge.capture_report(self.report)

    def run_verifier(self, captured: judge.CapturedReport) -> Dict[str, Any]:
        self.verified_with = captured
        graded = REWARD_ALL_CAUSES if captured.data else REWARD_CONTROL_PASS
        return {"reward": graded, "details": _details(3)}

    def exec(self, command: str, *, timeout: Optional[float] = None) -> Dict[str, Any]:
        raise AssertionError("not used")

    def read_file(self, path: str) -> Dict[str, Any]:
        raise AssertionError("not used")

    def write_file(self, path: str, content: str) -> Dict[str, Any]:
        raise AssertionError("not used")

    def teardown(self) -> None:
        return None


def test_the_report_is_captured_once_and_the_verifier_gets_those_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Sealing stops MCP calls, not processes the agent already started in its container.

    A pre-check that stats the live path and then lets the verifier reopen it grades whatever is
    there the second time. A watcher started before the seal can delete the report between the two
    reads and recreate the judge-error exclusion, or swap the bytes so the score belongs to
    something that did not exist when the episode was sealed. So the artifact is read once, and
    the bytes that were read are the bytes that are graded."""
    from shogym.envs.orca_bench import mcp_server

    report = tmp_path / "report.md"
    report.write_text("# Incident Report\n\nthe submission as sealed\n", encoding="utf-8")
    backend = _FakeBackend(report)
    monkeypatch.setitem(mcp_server._sessions, "session-1", backend)

    original = report.read_bytes()
    real_capture = judge.capture_report

    def _capture_then_let_the_agent_move(path: Path) -> judge.CapturedReport:
        captured = real_capture(path)
        report.unlink()  # the watcher fires in the window the old shape left open
        report.write_text("something else entirely\n", encoding="utf-8")
        return captured

    monkeypatch.setattr(judge, "capture_report", _capture_then_let_the_agent_move)
    payload = mcp_server.finalize_session("session-1")

    assert backend.verified_with is not None
    assert backend.verified_with.data == original, "the verifier was given the later bytes"
    assert payload is not None and not payload.get("submission_error")
    verdict = judge.parse_verdict(
        payload.get("reward"), payload.get("details"),
        submission_error=str(payload.get("submission_error") or ""),
    )
    assert verdict.judge_error is False and verdict.reward == 1.0


def test_a_report_removed_after_the_seal_cannot_manufacture_an_exclusion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shogym.envs.orca_bench import mcp_server

    report = tmp_path / "report.md"
    report.write_text("# Incident Report\n", encoding="utf-8")
    backend = _FakeBackend(report)
    monkeypatch.setitem(mcp_server._sessions, "session-1", backend)

    real_capture = judge.capture_report

    def _capture_then_delete(path: Path) -> judge.CapturedReport:
        captured = real_capture(path)
        report.unlink()
        return captured

    monkeypatch.setattr(judge, "capture_report", _capture_then_delete)
    payload = mcp_server.finalize_session("session-1")

    assert payload is not None and not payload.get("submission_error")
    assert judge.parse_verdict(payload.get("reward"), payload.get("details")).judge_error is False


def test_the_verifier_never_runs_on_a_submission_that_failed_capture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from shogym.envs.orca_bench import mcp_server

    class _RefusingBackend(_FakeBackend):
        def run_verifier(self, captured: judge.CapturedReport) -> Dict[str, Any]:
            raise AssertionError("the verifier ran on a submission that could not be captured")

    backend = _RefusingBackend(tmp_path / "absent.md")
    monkeypatch.setitem(mcp_server._sessions, "session-1", backend)
    payload = mcp_server.finalize_session("session-1")

    assert payload is not None
    assert "no report" in payload["submission_error"]
    assert payload["reward"] is None and payload["details"] is None


def test_capture_refuses_a_symlink_without_following_it(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere.md"
    target.write_text("# not the submission", encoding="utf-8")
    link = tmp_path / "report.md"
    link.symlink_to(target)

    captured = judge.capture_report(link)
    assert captured.problem and "regular file" in captured.problem
    assert captured.data is None, "a refused capture holds no bytes"


def test_capture_holds_the_bytes_it_validated(tmp_path: Path) -> None:
    report = tmp_path / "report.md"
    report.write_bytes(b"# Incident Report\n")
    captured = judge.capture_report(report)
    assert captured.problem == "" and captured.data == b"# Incident Report\n"
    assert captured.source == str(report)

    empty = tmp_path / "empty.md"
    empty.write_bytes(b"")
    assert judge.capture_report(empty).problem == ""  # the control-task submission

    oversized = tmp_path / "oversized.md"
    oversized.write_bytes(b"x" * (judge.MAX_REPORT_BYTES + 1))
    over = judge.capture_report(oversized)
    assert over.problem and str(judge.MAX_REPORT_BYTES) in over.problem
    assert over.data is None


# ----- scored feedback -----


def _values(fb) -> Dict[str, Any]:
    return {item.name: item.value for item in fb.episode}


def test_score_evidence_publishes_the_benchmark_numbers() -> None:
    verdict = judge.parse_verdict(REWARD_ALL_CAUSES, _details(3, 3))
    values = _values(score_evidence(_evidence(_published(verdict))))
    assert values["reward"] == 1.0
    assert values["success"] is True
    assert values["verified"] is True
    assert values["rca_accuracy"] is True
    assert values["hallucinate_any"] is False
    assert "judge_error" not in values


def test_score_evidence_flags_a_judge_error_instead_of_averaging_it_in() -> None:
    verdict = judge.parse_verdict(REWARD_JUDGE_FAILED, DETAILS_JUDGE_FAILED)
    values = _values(
        score_evidence(_evidence(_published(verdict), status="finalize_error"))
    )
    assert values["judge_error"] is True
    assert values["reward"] == 0.0
    assert values["success"] is False
    assert values["verified"] is False
    # Undefined rollups are omitted rather than reported as False.
    assert "rca_accuracy" not in values


def test_score_evidence_handles_an_abort() -> None:
    values = _values(score_evidence(None))
    assert values == {"reward": 0.0, "success": False, "verified": False}


# ----- which scoring function produced the score -----


def test_a_score_carries_the_judge_that_produced_it() -> None:
    """Changing the judge model changes the scoring function, so two runs that differ only in
    the judge must not produce identical evidence. The provenance rides on the verdict, which is
    what the trace carries; the private diagnostic is not an auditable surface."""
    details = _details(3)
    details.update({"model": "custom-judge", "reasoning_effort": "low"})
    verdict = judge.parse_verdict(REWARD_ALL_CAUSES, details)
    assert (verdict.judge_model, verdict.judge_effort) == ("custom-judge", "low")

    custom = _published(verdict)
    default = _published(judge.parse_verdict(REWARD_ALL_CAUSES, _details(3)))
    assert custom["judge_model"] == "custom-judge" and custom["judge_effort"] == "low"
    assert default["judge_model"] == judge.DEFAULT_JUDGE_MODEL
    assert custom != default


def test_the_judge_that_ran_wins_over_the_one_that_was_configured() -> None:
    # reward-details.json records what actually scored the report; the config is what was asked
    # for. A run whose judge differed from the configuration must report the judge that ran.
    details = _details(3)
    details.update({"model": "actually-ran", "reasoning_effort": "medium"})
    published = _published(
        judge.parse_verdict(REWARD_ALL_CAUSES, details),
        config=judge.JudgeConfig(model="was-configured", effort="high"),
    )
    assert published["judge_model"] == "actually-ran"
    assert published["judge_effort"] == "medium"


def test_a_judge_error_still_names_the_configured_judge() -> None:
    # The failure shape records no model, so the configuration is the only thing that can say
    # which judge failed. A judge-error grade with no provenance is unactionable.
    published = _published(
        judge.parse_verdict(REWARD_JUDGE_FAILED, DETAILS_JUDGE_FAILED),
        config=judge.JudgeConfig(model="configured-judge", effort="low"),
    )
    assert published["judge_error"] is True
    assert published["judge_model"] == "configured-judge"
    assert published["judge_effort"] == "low"


def test_a_custom_endpoint_is_identified_without_being_disclosed() -> None:
    """A private endpoint's URL may name internal infrastructure, so it is published as a stable
    digest: enough to tell two runs apart, not enough to read the host out of a trace."""
    assert judge.JudgeConfig().endpoint_id({}) == "default"
    private = judge.JudgeConfig(base_url="https://judge.internal.example/v1")
    identifier = private.endpoint_id({})
    assert identifier.startswith("custom:")
    assert identifier == private.endpoint_id({})  # stable across calls
    assert "internal.example" not in identifier and "https" not in identifier
    assert identifier != judge.JudgeConfig(base_url="https://other/v1").endpoint_id({})
    # An endpoint from the process environment counts the same as a configured one.
    environ = {"OPENAI_BASE_URL": "https://judge.internal.example/v1"}
    assert judge.JudgeConfig().endpoint_id(environ) == identifier


def test_score_evidence_publishes_the_judge_provenance() -> None:
    # The trace surface: `result_from_trace(...).value("judge_model")` has to answer "scored by
    # what?", so the provenance is emitted as feedback and not only inside the verdict dict.
    details = _details(3)
    details.update({"model": "custom-judge", "reasoning_effort": "low"})
    published = _published(
        judge.parse_verdict(REWARD_ALL_CAUSES, details),
        config=judge.JudgeConfig(base_url="https://judge.internal.example/v1"),
    )
    values = _values(score_evidence(_evidence(published)))
    assert values["judge_model"] == "custom-judge"
    assert values["judge_effort"] == "low"
    assert values["judge_endpoint"].startswith("custom:")


# ----- the hidden control label is not a benchmark number -----


def test_the_public_verdict_never_carries_the_control_label() -> None:
    """Whether a task is a control is the answer, not a score.

    ``describe()`` withholds even ``section`` because it gives that away; publishing the same fact
    in the verdict would hand it back at the end of the episode. The verdict is returned to the
    caller verbatim and persisted in the trace, so a reader who has graded a task once would know
    the definitive label for every later run of it. It is used to derive ``success`` and goes no
    further.
    """
    graded = judge.parse_verdict(REWARD_CONTROL_PASS, _details(3))
    control = _published(graded, is_control=True)
    incident = _published(judge.parse_verdict(REWARD_ALL_CAUSES, _details(3, 3)), is_control=False)

    for published in (control, incident):
        assert "is_control" not in published
        assert not any(isinstance(value, bool) and key.endswith("control") for key, value in published.items())
    # The label is still what decided `success`: this control passed on reward alone, with the
    # strict all-causes metric structurally 0.
    assert control["success"] is True and control["rca_accuracy"] is False

    assert set(control) == {
        "reward",
        "success",
        "judge_error",
        "rca_accuracy",
        "judge_model",
        "judge_effort",
        "judge_endpoint",
    }
    # The one remaining structural difference between the two, and it is upstream's: the
    # benchmark leaves `hallucinate_any` undefined on a control task, so the verifier omits it
    # there. Emitting a made-up value to hide that would misreport a published metric.
    assert set(incident) - set(control) == {"hallucinate_any"}
    assert set(control) - set(incident) == set()


def test_the_scored_feedback_never_carries_the_control_label() -> None:
    values = _values(score_evidence(_evidence(_published(
        judge.parse_verdict(REWARD_CONTROL_PASS, _details(3)), is_control=True
    ))))
    assert "is_control" not in values
    assert values["success"] is True


async def test_a_finalized_control_task_publishes_no_control_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End to end through the real ``finalize``, on a task that really is a control."""
    import shogym
    from shogym.envs.orca_bench import mcp_server
    from shogym.serve.lifecycle import FinalizeRequest
    from tests._fixtures import orca_bench_dataset as synth

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setattr(mcp_server, "begin_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mcp_server,
        "finalize_session",
        lambda _session_id: {"reward": REWARD_CONTROL_PASS, "details": _details(3)},
    )

    env = shogym.make(
        "orca_bench", config={"dataset_dir": str(synth.write_dataset(tmp_path / "orca"))}
    )
    control = next(ref for ref in env.refs if ref.is_control)
    task = env.load_task(env.position_of(control))
    assert task["is_control"] is True
    env.begin_session("session-1", task)
    evidence = await env.finalize(
        FinalizeRequest(source="explicit_tool", finalization_id="f1", session_id="session-1")
    )

    # The serve layer hands `evidence.verdict` to the caller verbatim and persists it in the
    # trace, so this is the surface that matters.
    assert "is_control" not in evidence.verdict
    assert json.dumps(evidence.verdict).find("is_control") == -1
    assert evidence.verdict["success"] is True
    # The private diagnostic keeps it: it never leaves the durable store, and without it the
    # record cannot explain why a task with `rca_accuracy` false scored a success.
    assert evidence.diagnostic is not None and "control" in evidence.diagnostic


def test_a_graded_control_is_inferable_from_the_published_metrics() -> None:
    """Documents a disclosure this port accepts, rather than fighting it.

    `success` is derived class-dependently because the benchmark's own metrics are: a control
    passes on reward alone, since the verifier hard-codes `feature_flag_match` false in the
    synthesized no-incident rubric. So `success` true beside `rca_accuracy` false is a
    combination only a control can produce, and `hallucinate_any` is absent exactly on controls.
    Publishing upstream's per-task numbers publishes facts about the task's class.

    The alternatives were rejected on purpose: a class-independent `success` would diverge from
    what the leaderboard means by these metrics, and invented values for undefined ones would
    misreport them. Redaction's promise is about the attempt, before and during, and that is
    `describe`'s contract, which stays intact. This test exists so the after-the-fact disclosure
    is a pinned decision rather than an accident. See the env README.
    """
    control = _published(judge.parse_verdict(REWARD_CONTROL_PASS, _details(3)), is_control=True)
    assert (control["success"], control["rca_accuracy"]) == (True, False)
    assert "hallucinate_any" not in control

    # No incident task can show that pair: there, success IS rca_accuracy.
    for reward, details in (
        (REWARD_ALL_CAUSES, _details(3, 3)),
        (REWARD_PARTIAL, _details(2, 3, 0, 0)),
        (REWARD_WRONG, _details(0, 0)),
    ):
        incident = _published(judge.parse_verdict(reward, details), is_control=False)
        assert incident["success"] == incident["rca_accuracy"]
        assert "hallucinate_any" in incident


def test_public_verdict_hides_the_judge_transcript() -> None:
    verdict = judge.parse_verdict(REWARD_PARTIAL, _details(2, 3, 0, 0))
    published = _published(verdict)
    assert set(published) == {
        "reward",
        "success",
        "judge_error",
        "rca_accuracy",
        "hallucinate_any",
        "judge_model",
        "judge_effort",
        "judge_endpoint",
    }
    # The transcript stays private: the rubric depth, the mode, and the judge's own reasoning are
    # never published, only which judge produced the numbers.
    assert "rca_depth" not in published and "mode" not in published
    assert "judge_prompt" not in published and "nested" not in published


# ----- the endpoint recorded is the one that scored -----


async def test_the_recorded_endpoint_is_the_session_s_not_the_readback_s(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Provenance is captured when the verifier is equipped, not re-read when the score is.

    An episode can run for an hour, and the process environment is not frozen for it. Resolving
    ``OPENAI_BASE_URL`` again at grading time records whatever it says then, which may be an
    endpoint that never saw this report. The audit field has to name the scoring function that
    ran.
    """
    import shogym
    from shogym.envs.orca_bench import env_v1, mcp_server
    from shogym.serve.lifecycle import FinalizeRequest
    from tests._fixtures import orca_bench_dataset as synth

    at_session_start = "https://judge-a.internal/v1"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_BASE_URL", at_session_start)
    monkeypatch.setattr(mcp_server, "begin_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        mcp_server,
        "finalize_session",
        lambda _session_id: {"reward": REWARD_ALL_CAUSES, "details": _details(3)},
    )

    env = shogym.make(
        "orca_bench", config={"dataset_dir": str(synth.write_dataset(tmp_path / "orca"))}
    )
    task = env.load_task(0)
    env.begin_session("session-1", task)

    monkeypatch.setenv("OPENAI_BASE_URL", "https://judge-b.internal/v1")
    evidence = await env.finalize(
        FinalizeRequest(source="explicit_tool", finalization_id="f1", session_id="session-1")
    )

    expected = judge.JudgeConfig().endpoint_id({"OPENAI_BASE_URL": at_session_start})
    assert evidence.verdict["judge_endpoint"] == expected
    assert env_v1.score_evidence(evidence).episode
    values = _values(env_v1.score_evidence(evidence))
    assert values["judge_endpoint"] == expected


def test_a_resolved_judge_freezes_the_endpoint_it_was_resolved_with() -> None:
    # The pure half: resolution happens once, against a given environment, and the record does
    # not consult the environment again.
    resolved = judge.JudgeConfig().resolve({"OPENAI_BASE_URL": "https://judge-a.internal/v1"})
    assert resolved.endpoint_id == judge.JudgeConfig(
        base_url="https://judge-a.internal/v1"
    ).endpoint_id({})
    assert resolved.environment["OPENAI_BASE_URL"] == "https://judge-a.internal/v1"
    # ...and it carries what the verifier process is given, so the audited endpoint and the
    # equipped one cannot drift apart.
    assert resolved.environment["OPENAI_API_KEY"] == judge.PLACEHOLDER_API_KEY
    assert resolved.model == judge.DEFAULT_JUDGE_MODEL and resolved.effort == "high"


# ----- the phase-2 seam -----


def test_the_backend_seam_reaches_the_live_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`backend.create_backend` is the seam phase 1 declared and phase 2 fills.

    The live implementation is imported lazily, so everything else about this env stays reachable
    on a machine with no Docker; asked to serve without a daemon, it says which piece is missing
    rather than failing somewhere further in."""
    from shogym.envs.orca_bench import compose_backend

    monkeypatch.setattr(compose_backend, "docker_available", lambda: False)
    with pytest.raises(backend.BackendUnavailableError, match="Docker daemon"):
        backend.create_backend(
            tmp_path,
            judge=judge.JudgeConfig().resolve({"OPENAI_API_KEY": "sk-test"}),
                snapshot="20260423T050139Z-4f4aceafe624e619",
        )


def test_the_environment_image_is_pinned_by_digest() -> None:
    assert backend.SNAPSHOT_IMAGE_DIGEST.startswith("sha256:")
    assert len(backend.SNAPSHOT_IMAGE_DIGEST) == len("sha256:") + 64
    assert backend.SNAPSHOT_IMAGE_PINNED.endswith(backend.SNAPSHOT_IMAGE_DIGEST)
    assert "@" in backend.SNAPSHOT_IMAGE_PINNED and ":" in backend.SNAPSHOT_IMAGE
