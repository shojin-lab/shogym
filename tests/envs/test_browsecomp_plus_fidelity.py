"""Keyed fidelity check: the real ``OpenAIJudge`` (BrowseComp-Plus's own grader prompt) grades a
served ``browsecomp_plus`` episode as a human would. The offline suite proves the plumbing with a
scripted judge; this proves the **real LLM judge** distinguishes right from wrong on a served
episode. Tasks + an in-memory searcher are injected, so this needs no encrypted-dataset download,
no corpus, and no Java — only an OpenAI key for the judge.

Skipped when ``OPENAI_API_KEY`` is absent, so offline CI stays green.
"""

from __future__ import annotations

import json
import os

import pytest

pytest.importorskip("openai", reason="browsecomp_plus extra not installed")

if not os.getenv("OPENAI_API_KEY"):
    pytest.skip(
        "OPENAI_API_KEY not set; keyed browsecomp_plus judge test skipped",
        allow_module_level=True,
    )

from hgym.envs.browsecomp_plus.searcher import InMemorySearcher  # noqa: E402
from hgym.serve import ServedEpisode  # noqa: E402

_CORPUS = {"1": "Paris is the capital of France.", "2": "France is a country in Europe."}
_TASKS = [
    {
        "query_id": "fidelity_1",
        "query": "What is the capital of France?",
        "answer": "Paris",
        "qrel_gold": ["1"],
        "qrel_evidence": ["1"],
    }
]


async def _grade(answer: str, confidence: int) -> dict:
    # No `judge` in the config -> the env builds the default OpenAIJudge (real network call).
    # submit_answer is the score terminal: it seals + runs finalize (the real judge) + ends the
    # episode in one call. The returned content is the public-safe, sanitized verdict.
    config = {"tasks": _TASKS, "searcher": InMemorySearcher(_CORPUS)}
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=config)
    try:
        result = await episode.call("submit_answer", {"answer": answer, "confidence": confidence})
        assert result.terminated
        verdict = json.loads(result.content)
        verdict["feedback"] = {i["name"]: i["value"] for i in episode.terminal_feedback}
        return verdict
    finally:
        await episode.close()


async def test_real_judge_accepts_a_correct_answer() -> None:
    verdict = await _grade("The capital of France is Paris [1].", confidence=90)
    assert verdict["correct"] is True
    assert verdict.get("judge_error", False) is False
    assert verdict["feedback"]["correct"] is True


async def test_real_judge_rejects_a_wrong_answer() -> None:
    verdict = await _grade("The capital of France is Berlin.", confidence=90)
    assert verdict["correct"] is False
    assert verdict["feedback"]["correct"] is False
