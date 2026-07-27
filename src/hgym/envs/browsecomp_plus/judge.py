"""The model-graded judge for the ``browsecomp_plus`` env (issue #43).

BrowseComp-Plus grades a Deep-Research answer with an **LLM judge**: does the submitted
answer match the gold answer for the query? As with the HLE port, the judge is used
**server-side**, inside the ``submit_answer`` tool handler (:mod:`mcp_server`), so the env's
``_verify`` stays a pure function over the recorded trajectory — the handler runs the judge and
records its verdict on the terminal step; the verifier only *parses* it.

The grading prompt (:data:`GRADER_TEMPLATE`) and the reply parser (:func:`parse_judge_response`)
are ported **verbatim** from upstream ``scripts_evaluation/evaluate_run.py`` (BrowseComp-Plus
commit ``0469490``), so the grading criteria match the benchmark's own. Upstream grades with
Qwen3-32B (vLLM, temp 0.7) or GPT-4.1 (the paper); hgym canonicalizes an OpenAI-compatible
judge at **temperature 0** for deterministic verification, defaulting to the paper's GPT-4.1
and overridable via ``judge_model`` / ``judge_base_url`` (point the base URL at a vLLM
Qwen3-32B for exact upstream fidelity).

Nothing here imports ``openai`` at module load — the client is created lazily on first use —
so ``import hgym`` (which imports the env to register it) stays offline and lean.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

# The paper's judge model (BrowseComp-Plus reports GPT-4.1). A current OpenAI model id; override
# via `judge_model` env config, or point `judge_base_url` at a vLLM Qwen3-32B to match the repo
# default exactly.
DEFAULT_JUDGE_MODEL = "gpt-4.1"

# Deterministic decoding for a reproducible verifier (upstream uses temp 0.7; we pin 0). Some
# newer/reasoning models reject an explicit temperature — the judge drops it and retries.
JUDGE_TEMPERATURE = 0.0

# A non-secret stand-in api_key. Local OpenAI-compatible servers (vLLM/Ollama/LM Studio) need no
# real key, but the OpenAI SDK still refuses to construct without *some* non-empty api_key — so a
# keyless `judge_base_url` endpoint gets this placeholder, which such servers ignore. A real
# `OPENAI_API_KEY` is always preferred when set.
_PLACEHOLDER_API_KEY = "sk-no-key-required"


@dataclass
class JudgeResult:
    """A judge's verdict on one submitted answer.

    ``correct`` is authoritative; ``extracted_answer`` / ``reasoning`` are advisory diagnostics
    the LLM judge fills in (empty for a scripted judge that returns a bare verdict).
    """

    correct: bool
    extracted_answer: str = ""
    reasoning: str = ""


@runtime_checkable
class Judge(Protocol):
    """A model-graded verifier: does ``response`` answer ``question`` as ``correct_answer``?

    Implementations must be side-effect-free apart from the model call and must not raise on an
    ordinary bad response — return ``JudgeResult(correct=False)`` instead. The handler guards
    against exceptions regardless, but a well-behaved judge keeps the episode clean.
    """

    def __call__(
        self, *, question: str, correct_answer: str, response: str
    ) -> JudgeResult: ...


# The official BrowseComp-Plus judge prompt (``scripts_evaluation/evaluate_run.py``), used
# verbatim so the judge's grading criteria match the benchmark's own. The literal ``|\%|`` in
# the confidence line is upstream's own escaping and is preserved as-is; hgym reads correctness
# only, so confidence extraction here is advisory.
GRADER_TEMPLATE = """
Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response].

[correct_answer]: Repeat the [correct_answer] given above.

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], in the context of this [question]. You should judge whether the extracted_final_answer is semantically equivalent to [correct_answer], allowing the extracted_final_answer to be string variations of [correct_answer]. You should also allow the extracted_final_answer to be more precise or verbose than [correct_answer], as long as its additional details are correct. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers are semantically equivalent.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.


confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


def create_judge_prompt(question: str, response: str, correct_answer: str) -> str:
    """Render :data:`GRADER_TEMPLATE` for one (question, response, gold) triple (upstream verbatim)."""
    return GRADER_TEMPLATE.format(
        question=question, response=response, correct_answer=correct_answer
    )


class OpenAIJudge:
    """LLM judge over the OpenAI-compatible chat wire (the registered env's default).

    Constructed offline (no network): the client is created lazily on the first call, so a
    ``browsecomp_plus`` env can be built and its manifest probed without a key. Pass a ready
    ``client`` (or ``base_url``) to point at any OpenAI-compatible endpoint (e.g. a vLLM
    Qwen3-32B, the upstream judge). Decodes at :data:`JUDGE_TEMPERATURE` (0) for determinism.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        base_url: Optional[str] = None,
        client: Any = None,
        temperature: float = JUDGE_TEMPERATURE,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._client = client
        self._temperature = temperature

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy: keeps `import hgym` free of openai

            # The SDK requires a non-empty api_key even against a keyless local base_url, so fall
            # back to a harmless placeholder (which such servers ignore) when no key is set — this
            # is what makes a keyless `judge_base_url` endpoint work as advertised.
            api_key = os.environ.get("OPENAI_API_KEY") or _PLACEHOLDER_API_KEY
            self._client = OpenAI(base_url=self._base_url, api_key=api_key)
        return self._client

    def __call__(
        self, *, question: str, correct_answer: str, response: str
    ) -> JudgeResult:
        client = self._ensure_client()
        content = create_judge_prompt(question, response, correct_answer)
        messages = [{"role": "user", "content": content}]
        try:
            completion = client.chat.completions.create(
                model=self._model, messages=messages, temperature=self._temperature
            )
        except Exception:
            # Some reasoning models reject an explicit temperature; retry at the model default so
            # a temp-0 request never hard-fails the judge on those endpoints.
            completion = client.chat.completions.create(
                model=self._model, messages=messages
            )
        text = completion.choices[0].message.content or ""
        return parse_judge_response(text)


def parse_judge_response(judge_response: str) -> JudgeResult:
    """Parse the judge model's structured reply into a :class:`JudgeResult`.

    Follows upstream's ``parse_judge_response`` (BrowseComp-Plus ``evaluate_run.py``) — tolerating
    the ``**correct:**`` / ``**correct**:`` / ``correct:`` markdown variants — but reads the
    verdict from a **line-anchored** ``correct`` field (not an unanchored match anywhere in the
    reply), the same hardening the HLE port applies. This matters because the submitted answer is
    embedded in the judge prompt and the model-generated ``reasoning`` echoes it: an answer that
    contains ``correct: yes`` must not override the judge's final ``correct: no``. It also fails
    **closed**: a missing verdict, or conflicting ``correct:`` lines, is treated as not correct
    (never grant credit on malformed / ambiguous judge output)."""
    if not judge_response:
        return JudgeResult(correct=False)

    extracted = _extract_field(
        judge_response,
        (
            r"\*\*extracted_final_answer:\*\*\s*(.*?)(?=\n|$)",
            r"\*\*extracted_final_answer\*\*:\s*(.*?)(?=\n|$)",
            r"extracted_final_answer:\s*(.*?)(?=\n|$)",
        ),
    )
    reasoning = _extract_field(
        judge_response,
        (
            r"\*\*reasoning:\*\*\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            r"\*\*reasoning\*\*:\s*(.*?)(?=\n\*\*correct:\*\*|\n\*\*correct\*\*:|\ncorrect:|$)",
            r"reasoning:\s*(.*?)(?=\ncorrect:|$)",
        ),
    )
    verdicts = _correct_field_values(judge_response)
    correct = len(verdicts) == 1 and verdicts.pop() is True
    return JudgeResult(correct=correct, extracted_answer=extracted, reasoning=reasoning)


def _correct_field_values(text: str) -> set[bool]:
    """The distinct yes/no values of every **line-anchored** ``correct:`` field in ``text``.

    Only a line whose leading field is ``correct`` (optionally ``**``-emphasized) counts — so a
    ``correct: yes`` sitting inside a ``reasoning:`` line (which starts with ``reasoning``) is
    ignored. An empty set means no verdict; a set of size >1 means conflicting verdicts (both fail
    closed at the call site)."""
    return {
        m.group(1).lower() == "yes"
        for m in re.finditer(
            r"^[ \t]*\*{0,2}correct\*{0,2}[ \t]*:\*{0,2}[ \t]*(yes|no)\b",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
    }


def _extract_field(text: str, patterns: tuple[str, ...]) -> str:
    """Return the first capturing-group match across ``patterns`` (best-effort, upstream order)."""
    for pattern in patterns:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            return m.group(1).strip()
    return ""
