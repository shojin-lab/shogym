"""The latentgym port's pure pieces: the inert channel, the reward, and the success verdict.

Offline and upstream-free: ``env_v1`` imports nothing from the provisioned source at module load,
which is itself one of the things asserted here.
"""

from __future__ import annotations

import json

from shogym.envs.latentgym.env_v1 import _as_reward, _succeeded, notice_for
from shogym.feedback.wire import NOTICE_FEEDBACK_NAME, REPORT_FEEDBACK_NAME

REPORTS = [
    "Episode 1 finished. You selected 'red' — correct! Score: 0.955",
    "Episode 4 finished. That wasn't the maximum. Score: 0.048. The maximum was 0.751 at "
    "position 2 (of 10). All values: [0.324, 0.151, 0.751]",
    "x",
    "a b",
    "This episode has been recorded. The next task is drawn the same way.",
]


# ----- the inert channel matches the report it stands in for -----


def test_a_notice_is_exactly_as_long_as_its_report_on_the_wire() -> None:
    # The match is the whole reason the control arm is a control: an agent that reads five words
    # in one arm and forty in the other has been told which arm it is in. Measured as the answer
    # is encoded, because that is the form the agent reads.
    for report in REPORTS:
        assert len(json.dumps(notice_for(report))) == len(json.dumps(report)), report


def test_the_match_survives_a_report_the_encoder_escapes() -> None:
    # The case that made this about encoded length rather than character count: upstream writes
    # its reports with a dash character the default encoder expands to six characters. A notice
    # built to the character count would arrive five characters short of every bandits report.
    escaped = "Episode 1 finished. You selected 'red' \u2014 correct! Score: 0.955"
    assert "\\u2014" in json.dumps(escaped)  # the encoder really does expand it
    assert len(json.dumps(notice_for(escaped))) == len(json.dumps(escaped))


def test_a_notice_carries_nothing_from_its_report() -> None:
    for report in REPORTS[:2]:
        notice = notice_for(report)
        for leak in ("Score", "red", "0.955", "0.048", "maximum", "Episode"):
            assert leak not in notice


def test_an_empty_report_gets_an_empty_notice() -> None:
    # There was no channel to match, so there is nothing to stand in for.
    assert notice_for("") == ""


def test_a_notice_is_the_same_for_every_report_of_a_length() -> None:
    # It is a function of the length and of nothing else, which is what makes it uninformative:
    # two episodes whose reports happen to be the same size are told the same thing.
    assert notice_for("a" * 80) == notice_for("b" * 80)


def test_a_notice_needs_no_escaping_itself() -> None:
    # The arithmetic above assumes it: plain ASCII with no quote or backslash, so the notice
    # encodes as itself and its character count is its wire count.
    for report in REPORTS:
        notice = notice_for(report)
        assert notice.isascii()
        assert '"' not in notice and "\\" not in notice


def test_the_two_channel_names_are_the_same_length() -> None:
    # The item name is on the wire beside the value, so an unequal pair would put the arms'
    # payloads at different byte lengths however well the values matched.
    assert len(REPORT_FEEDBACK_NAME) == len(NOTICE_FEEDBACK_NAME)


# ----- the reward is the env's own number -----


def test_a_reward_is_taken_as_published() -> None:
    assert _as_reward(0.955) == 0.955
    assert _as_reward(1) == 1.0
    # Not clamped: rescaling an upstream reward would be reporting an adjusted benchmark.
    assert _as_reward(1.5) == 1.5
    assert _as_reward(-0.25) == -0.25


def test_a_value_that_cannot_be_a_reward_is_no_reward() -> None:
    # `bool` is an `int` subclass, so an unguarded numeric read would record a reward of 1.0 no
    # env published. NaN and the infinities are not JSON and would break the wire.
    for junk in (None, "not a number", True, False, float("nan"), float("inf"), object()):
        assert _as_reward(junk) == 0.0


# ----- success is the core env's own verdict -----


def test_success_reads_the_core_envs_own_verdict() -> None:
    # A bandits episode won on turn 2 scores 0.97, so no arithmetic on the reward finds this.
    won = {"finished": True, "reward": 0.97, "info": {"is_correct": True}}
    lost = {"finished": True, "reward": 0.0, "info": {"is_correct": False}}
    assert _succeeded(won) is True
    assert _succeeded(lost) is False


def test_success_falls_back_to_full_credit() -> None:
    # Secretary pays partial credit for a miss, so anything short of full credit is a miss.
    assert _succeeded({"finished": True, "reward": 1.0, "info": {}}) is True
    assert _succeeded({"finished": True, "reward": 0.48, "info": {}}) is False


def test_an_unfinished_episode_never_succeeded() -> None:
    assert _succeeded({"finished": False, "reward": 1.0, "info": {"solved": True}}) is False


def test_a_non_boolean_verdict_is_not_read_as_one() -> None:
    # `EpisodeFeedbackValue` admits text, and `bool("false")` is True, so a verdict key holding
    # anything but a bool falls through to the reward rather than deciding by truthiness.
    assert _succeeded({"finished": True, "reward": 0.2, "info": {"solved": "false"}}) is False
    assert _succeeded({"finished": True, "reward": 1.0, "info": {"solved": "false"}}) is True
