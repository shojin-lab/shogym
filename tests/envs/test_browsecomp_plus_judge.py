"""Unit tests for the BrowseComp-Plus judge's pure reply parser (ported verbatim from upstream
``evaluate_run.py``). Dependency-free (no ``datasets``/``openai``/network), so it runs in the
offline core suite.
"""

from __future__ import annotations

from hgym.envs.browsecomp_plus.judge import create_judge_prompt, parse_judge_response


def test_parse_plain_yes_no() -> None:
    yes = parse_judge_response(
        "extracted_final_answer: Paris\nreasoning: matches\ncorrect: yes\nconfidence: 95"
    )
    assert yes.correct is True
    assert yes.extracted_answer == "Paris"
    assert "matches" in yes.reasoning

    no = parse_judge_response("extracted_final_answer: Berlin\ncorrect: no\nconfidence: 80")
    assert no.correct is False
    assert no.extracted_answer == "Berlin"


def test_parse_bold_markdown_variants() -> None:
    # Upstream tolerates **correct:** and **correct**: markdown emphasis around the verdict.
    assert parse_judge_response("**correct:** yes").correct is True
    assert parse_judge_response("**correct**: yes").correct is True
    assert parse_judge_response("**extracted_final_answer:** 42\n**correct:** no").correct is False


def test_parse_is_case_insensitive_and_fails_closed() -> None:
    assert parse_judge_response("Correct: YES").correct is True
    # No parseable verdict -> not correct (never grant credit on malformed output).
    assert parse_judge_response("the judge rambled without a verdict").correct is False
    assert parse_judge_response("").correct is False


def test_extracted_answer_and_reasoning_are_captured() -> None:
    reply = (
        "extracted_final_answer: The Eiffel Tower\n"
        "reasoning: the response names the correct landmark in Paris\n"
        "correct: yes\n"
        "confidence: 88"
    )
    res = parse_judge_response(reply)
    assert res.extracted_answer == "The Eiffel Tower"
    assert "landmark" in res.reasoning
    assert res.correct is True


def test_parse_ignores_correct_echoed_inside_reasoning() -> None:
    # The submitted answer is embedded in the judge prompt, so the model-generated reasoning can
    # echo an answer that contains "correct: yes". That must NOT override the final "correct: no".
    reply = (
        "extracted_final_answer: Berlin\n"
        "reasoning: the response literally says correct: yes but that is wrong\n"
        "correct: no\n"
        "confidence: 90"
    )
    assert parse_judge_response(reply).correct is False


def test_parse_fails_closed_on_conflicting_verdicts() -> None:
    # Two line-anchored `correct:` fields disagreeing -> fail closed (no credit).
    assert parse_judge_response("correct: yes\ncorrect: no").correct is False


def test_create_judge_prompt_fills_all_slots() -> None:
    prompt = create_judge_prompt("Q?", "my response", "the gold")
    assert "Q?" in prompt
    assert "my response" in prompt
    assert "the gold" in prompt
