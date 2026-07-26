"""Keyed fidelity check: the real ``OpenAIJudge`` grades a served HLE episode as a human would.

Issue #33 asks HLE to give hgym its first *model-graded* verifier. The offline suite proves
the plumbing with a scripted judge; this test proves the **real LLM judge** actually
distinguishes right from wrong on a served episode: a paraphrased-but-correct answer whose
surface form defeats the exact-match fast path must be graded ``correct``, and a clearly
wrong answer ``incorrect``. The task is injected (a well-known fact), so this needs no gated
``cais/hle`` download — only an OpenAI key for the judge.

Skipped when ``OPENAI_API_KEY`` is absent, so offline CI stays green; run it with a key to
confirm real-judge fidelity.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("openai", reason="hle extra not installed")

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip("OPENAI_API_KEY not set; keyed HLE judge test skipped", allow_module_level=True)

from hgym.serve import ServedEpisode  # noqa: E402

# A fact unambiguous enough that a competent judge grades it deterministically, with a gold
# answer whose exact string differs from the (correct) response we submit — so the judge, not
# the exact-match fast path, must do the grading.
_TASKS = [
    {
        "id": "fidelity_1",
        "question": "In which year did the first human land on the Moon?",
        "answer": "1969",
        "answer_type": "exactMatch",
        "category": "history",
    }
]


async def _grade(answer: str, confidence: int) -> dict:
    # No `judge` in the config -> the env builds the default OpenAIJudge (real network call).
    # RFC 009: submit_answer is the `score` terminal; the call seals + finalizes and returns
    # the sanitized payload (score + judge_error). Correctness is read off terminal_feedback.
    episode = await ServedEpisode.start("hle", task=0, env_config={"tasks": _TASKS})
    try:
        result = await episode.call("submit_answer", {"answer": answer, "confidence": confidence})
        grade = json.loads(result.content)
        grade["feedback"] = {i["name"]: i["value"] for i in episode.terminal_feedback}
        return grade
    finally:
        await episode.close()


async def test_real_judge_accepts_a_paraphrased_correct_answer() -> None:
    # The gold answer is "1969"; a spelled-out paraphrase defeats the exact-match fast path,
    # so a `correct` verdict here proves the real LLM judge did the grading.
    grade = await _grade("It happened in the year nineteen sixty-nine.", confidence=80)
    assert grade["correct"] is True
    assert grade["judge_error"] is False
    assert grade["feedback"]["correct"] is True


async def test_real_judge_rejects_a_wrong_answer() -> None:
    grade = await _grade("The first Moon landing was in 1975.", confidence=80)
    assert grade["correct"] is False
    assert grade["feedback"]["correct"] is False
