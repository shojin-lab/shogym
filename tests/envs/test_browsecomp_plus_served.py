"""End-to-end: drive a served ``browsecomp_plus`` episode through shogym's serve layer with an
in-memory searcher + a scripted (injectable) judge, and check that the served retrieval tools,
the seal-before-verdict grade (``submit_answer`` seals → the env's ``finalize`` judges), and the
deterministic retrieval/citation metrics all flow into episode feedback — exercising the whole
env without Java, pyserini, a corpus download, or any network.

Fully offline: the searcher, judge, and tasks are injected via ``env_config``, so this runs in
the core suite (no ``browsecomp_plus`` extra, no key).
"""

from __future__ import annotations

import json

from shogym.envs.browsecomp_plus.judge import JudgeResult
from shogym.envs.browsecomp_plus.searcher import InMemorySearcher
from shogym.serve import ServedEpisode

# A tiny synthetic corpus: docids 1 and 2 are the evidence for the query; 9 is a distractor.
_CORPUS = {
    "1": "The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
    "2": "Paris is the capital and most populous city of France.",
    "9": "The Great Wall of China is a series of fortifications in northern China.",
}

_TASKS = [
    {
        "query_id": "q_paris",
        "query": "In which city is the Eiffel Tower, and what is that country's capital?",
        "answer": "Paris",
        "qrel_gold": ["1"],
        "qrel_evidence": ["1", "2"],
    }
]


class _ScriptedJudge:
    """A deterministic judge: correct iff the response mentions 'paris'. Records call count."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self, *, question: str, correct_answer: str, response: str) -> JudgeResult:
        self.calls += 1
        return JudgeResult(correct="paris" in response.lower(), extracted_answer="Paris")


def _config(judge=None) -> dict:
    return {
        "tasks": _TASKS,
        "searcher": InMemorySearcher(_CORPUS),
        "judge": judge or _ScriptedJudge(),
        "snippet_max_tokens": -1,  # no truncation in tests
    }


def _feedback(episode: ServedEpisode) -> dict:
    return {item["name"]: item["value"] for item in episode.terminal_feedback}


async def test_describe_surfaces_the_query_and_tools() -> None:
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=_config())
    try:
        spec = episode.describe()
        assert {"search", "get_document", "submit_answer", "terminate"} <= {t.name for t in spec.tools}
        assert "Eiffel Tower" in spec.instructions
    finally:
        await episode.close()


async def test_search_returns_ranked_docids() -> None:
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=_config())
    try:
        result = await episode.call("search", {"query": "Eiffel Tower Paris capital France"})
        hits = json.loads(result.content)
        docids = [h["docid"] for h in hits]
        # The Paris/Eiffel docs (1, 2) rank above the unrelated Great Wall doc (9, if present).
        assert docids[0] in {"1", "2"}
        assert set(docids) & {"1", "2"}
    finally:
        await episode.close()


async def test_get_document_returns_full_text() -> None:
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=_config())
    try:
        result = await episode.call("get_document", {"docid": "2"})
        doc = json.loads(result.content)
        assert doc["docid"] == "2"
        assert "capital" in doc["text"]
        missing = json.loads((await episode.call("get_document", {"docid": "404"})).content)
        assert missing["text"] is None
    finally:
        await episode.close()


async def test_full_episode_scores_correct_with_metrics() -> None:
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=_config(judge))
    try:
        await episode.call("search", {"query": "Eiffel Tower Paris capital France"})
        # submit_answer is the score terminal: it seals + runs finalize (the judge) + ends the
        # episode in one call. The returned content is the public-safe, sanitized verdict — no
        # judge reasoning/extracted_answer, no marker JSON.
        result = await episode.call(
            "submit_answer",
            {"answer": "The Eiffel Tower is in Paris [1], the capital of France [2].", "confidence": 90},
        )
        assert result.terminated
        verdict = json.loads(result.content)
        assert verdict["correct"] is True
        assert verdict["finalize_error"] is False
        assert "reasoning" not in verdict and "extracted_answer" not in verdict
        assert judge.calls == 1

        fb = _feedback(episode)
        assert fb["correct"] is True
        assert fb["confidence"] == 0.9
        assert abs(fb["calibration_error"] - 0.1) < 1e-9
        assert fb["retrieval_recall"] == 1.0  # searched docs cover evidence {1,2}
        assert fb["citation_recall"] == 1.0  # cited [1] and [2]
        assert fb["citation_precision"] == 1.0
    finally:
        await episode.close()


async def test_wrong_answer_scores_incorrect() -> None:
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=_config())
    try:
        result = await episode.call("submit_answer", {"answer": "It is in Berlin.", "confidence": 100})
        assert result.terminated
        assert json.loads(result.content)["correct"] is False
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert fb["calibration_error"] == 1.0
    finally:
        await episode.close()


async def test_second_submission_is_tombstoned_and_first_verdict_stands() -> None:
    # The seal makes single-submission structural: the first submit_answer seals + finalizes, so a
    # second one is tombstoned (never dispatched, never re-graded) and the first verdict stands.
    judge = _ScriptedJudge()
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=_config(judge))
    try:
        first = await episode.call("submit_answer", {"answer": "Berlin", "confidence": 100})
        assert first.terminated
        assert json.loads(first.content)["correct"] is False
        second = await episode.call("submit_answer", {"answer": "Paris", "confidence": 100})
        assert second.terminated
        assert "sealed" in second.content  # tombstone, not a fresh grade
        assert judge.calls == 1  # the second submission never reached the judge
        assert _feedback(episode)["correct"] is False
    finally:
        await episode.close()


async def test_judge_exception_scores_incorrect_without_crashing() -> None:
    class _BoomJudge:
        def __call__(self, *, question: str, correct_answer: str, response: str) -> JudgeResult:
            raise RuntimeError("judge unavailable")

    config = {**_config(), "judge": _BoomJudge()}
    episode = await ServedEpisode.start("browsecomp_plus", task=0, env_config=config)
    try:
        result = await episode.call("submit_answer", {"answer": "x", "confidence": 70})
        assert result.terminated
        verdict = json.loads(result.content)
        assert verdict["correct"] is False
        assert verdict["judge_error"] is True  # judge infra failure flagged (public-safe)
        assert "judge unavailable" not in result.content  # exception text is private diagnostic
        fb = _feedback(episode)
        assert fb["correct"] is False
        assert fb["judge_error"] is True
    finally:
        await episode.close()


async def test_preflight_raises_without_key_for_default_judge(monkeypatch) -> None:
    # Default judge (no judge=, no judge_base_url) + no OPENAI_API_KEY: beginning the episode must
    # raise early with an actionable message (an injected searcher still doesn't opt out the judge).
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = {"tasks": _TASKS, "searcher": InMemorySearcher(_CORPUS)}
    try:
        await ServedEpisode.start("browsecomp_plus", task=0, env_config=config)
        raised = False
    except RuntimeError as exc:
        raised = "OPENAI_API_KEY" in str(exc) and "judge_base_url" in str(exc)
    assert raised


async def test_make_and_describe_are_keyless(monkeypatch) -> None:
    import shogym

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    env = shogym.make("browsecomp_plus", config={"tasks": _TASKS, "searcher": InMemorySearcher(_CORPUS)})
    spec = env.describe("0")
    assert "Eiffel Tower" in spec.instructions
    assert {"search", "submit_answer", "terminate"} <= {t.name for t in spec.tools}


async def test_negative_task_index_is_rejected() -> None:
    import shogym

    env = shogym.make("browsecomp_plus", config=_config())
    import pytest

    with pytest.raises(ValueError, match="out of range"):
        env._load_task(-1)
