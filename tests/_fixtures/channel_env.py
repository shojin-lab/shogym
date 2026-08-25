"""A ``score``-terminal fixture env that publishes the paired report/notice channels (tests only).

The two paired feedback policies select between two *named* episode-feedback items, so testing
them needs an env that publishes both. This is the score fixture plus exactly that: the same
tools, the same finalizer, the same ``correct`` verdict, and two more published items whose
values are fixed strings a test can assert on byte for byte.

Registered under ``_fixture_channel``. Importing this module registers it (idempotent within a
process); the registry entry is inert for every other test because nothing else constructs it.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from shogym.envs.registration import _ENV_REGISTRY, register
from shogym.feedback.wire import NOTICE_FEEDBACK_NAME, REPORT_FEEDBACK_NAME
from shogym.trajectory import Trajectory
from shogym.types import EpisodeFeedback, FeedbackCollection

from tests._fixtures.score_env import _FixtureScoreEnv

ENV_NAME = "_fixture_channel"

# Equal length, so a test can assert the two arms differ in content and in nothing else.
REPORT_TEXT = "the answer was 4 and you were right about it"
NOTICE_TEXT = "this episode has been recorded, and no more."


class _FixtureChannelEnv(_FixtureScoreEnv):
    """The score fixture, publishing a report and its inert stand-in beside the verdict."""

    def _verify(
        self,
        trajectory: Trajectory,
        task: Dict[str, Any],
        *,
        terminated: bool,
        evidence: Optional[Any] = None,
    ) -> FeedbackCollection:
        fb = super()._verify(trajectory, task, terminated=terminated, evidence=evidence)
        if terminated:
            fb.episode.append(EpisodeFeedback(name=REPORT_FEEDBACK_NAME, value=REPORT_TEXT))
            fb.episode.append(EpisodeFeedback(name=NOTICE_FEEDBACK_NAME, value=NOTICE_TEXT))
        return fb


if ENV_NAME not in _ENV_REGISTRY:
    register(ENV_NAME)(_FixtureChannelEnv)
