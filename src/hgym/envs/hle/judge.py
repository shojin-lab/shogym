"""The model-graded judge for the HLE env — hgym's first model-graded verifier (issue #33).

The judge decides whether a submitted answer matches the gold answer for a Humanity's Last
Exam question. It is used **server-side**, inside the ``submit_answer`` tool handler
(``mcp_server``), so the env's ``_verify`` stays a pure function over the recorded
trajectory: the handler runs the judge and records its verdict on the terminal step; the
verifier only *parses* that verdict.

Two pieces, on purpose:

- ``exact_match`` — a deterministic, offline fast path (normalize + compare). The handler
  tries this first; a match short-circuits the LLM judge entirely (no network, no cost).
- :class:`Judge` — the injectable LLM-judge seam. The registered ``hle`` env defaults to
  :class:`OpenAIJudge`; offline tests inject a scripted judge (mirroring how the tau2 port
  injects a ``mock_response`` user simulator). A judge is only ever *called* when the exact
  fast path misses.

Nothing here imports ``openai`` at module load — the client is created lazily on first use —
so ``import hgym`` (which imports the env to register it) stays offline and lean.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

# The registered `hle` env's default judge model. A cheap, current OpenAI model (the same
# family the repo's quickstart uses); override via `judge_model` env config.
DEFAULT_JUDGE_MODEL = "gpt-5.4-nano"

# A non-secret stand-in api_key. Local OpenAI-compatible servers (Ollama/vLLM/LM Studio) need
# no real key, but the OpenAI SDK still refuses to construct without *some* non-empty api_key —
# so a keyless `judge_base_url` endpoint gets this placeholder, which such servers ignore. A
# real `OPENAI_API_KEY` is always preferred when set.
_PLACEHOLDER_API_KEY = "sk-no-key-required"

# Trailing punctuation/whitespace stripped before an exact-match comparison.
_STRIP_CHARS = " \t\n\r,.;:!?'\"`()[]{}"


@dataclass
class JudgeResult:
    """A judge's verdict on one submitted answer.

    ``correct`` is authoritative; ``extracted_answer`` / ``reasoning`` are advisory
    diagnostics the LLM judge fills in (empty for a scripted judge that returns a bare
    verdict).
    """

    correct: bool
    extracted_answer: str = ""
    reasoning: str = ""


@runtime_checkable
class Judge(Protocol):
    """A model-graded verifier: does ``response`` answer ``question`` as ``correct_answer``?

    Implementations must be side-effect-free apart from the model call and must not raise on
    an ordinary bad response — return ``JudgeResult(correct=False)`` instead. The handler
    guards against exceptions regardless, but a well-behaved judge keeps the episode clean.
    """

    def __call__(
        self, *, question: str, correct_answer: str, response: str
    ) -> JudgeResult: ...


def normalize_answer(text: Any) -> str:
    """Casefold, collapse internal whitespace, and strip surrounding punctuation.

    Deliberately conservative: it only equates answers that differ by case, spacing, or
    trailing punctuation. Anything semantic (unit conversions, paraphrase, numeric
    tolerance) is left to the LLM judge — the fast path must never *grant* credit an exact
    comparison wouldn't."""
    if text is None:
        return ""
    s = re.sub(r"\s+", " ", str(text)).strip()
    return s.strip(_STRIP_CHARS).casefold()


def exact_match(response: Any, correct_answer: Any) -> bool:
    """True iff ``response`` equals ``correct_answer`` after normalization.

    An empty gold answer never matches (so a blank submission can't earn credit against a
    blank field)."""
    gold = normalize_answer(correct_answer)
    return bool(gold) and normalize_answer(response) == gold


# The official HLE judge prompt (Center for AI Safety, `cais/hle`), used verbatim so the
# judge's grading criteria match the benchmark's own.
JUDGE_PROMPT = """Judge whether the following [response] to [question] is correct or not \
based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the \
extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on \
[correct_answer], focusing only on if there are meaningful differences between \
[correct_answer] and the extracted_final_answer. Do not comment on any background to the \
problem, do not attempt to solve the problem, do not argue for any answer different than \
[correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, \
or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. \
if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is \
incorrect.

confidence: The extracted confidence score between 0% and 100% from [response]. Put 100 if \
there is no confidence score available."""


class OpenAIJudge:
    """LLM judge over the OpenAI-compatible chat wire (the registered env's default).

    Constructed offline (no network): the client is created lazily on the first call, so an
    ``hle`` env can be built and its manifest probed without a key. Pass a ready ``client``
    (or ``base_url``) to point at any OpenAI-compatible endpoint.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        base_url: Optional[str] = None,
        client: Any = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._client = client

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
        content = JUDGE_PROMPT.format(
            question=question, correct_answer=correct_answer, response=response
        )
        completion = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
        )
        text = completion.choices[0].message.content or ""
        return parse_judge_response(text)


def parse_judge_response(text: str) -> JudgeResult:
    """Parse the judge model's structured reply into a :class:`JudgeResult`.

    ``correct`` is True only on an explicit ``yes`` on a **line whose field is** ``correct``
    — the verdict is read from the structured ``correct:`` field, not from an unanchored
    match anywhere in the reply. This matters because ``reasoning`` (model-generated) echoes
    the agent-controlled response, so a substring like ``correct: yes`` buried in the
    reasoning must not override a final ``correct: no``. It also fails **closed**: a missing
    verdict, or conflicting ``correct:`` lines, is treated as not correct (never grant credit
    on malformed judge output)."""
    verdicts = _correct_field_values(text)
    correct = len(verdicts) == 1 and verdicts.pop() is True
    extracted = _field(text, "extracted_final_answer")
    reasoning = _field(text, "reasoning")
    return JudgeResult(correct=correct, extracted_answer=extracted, reasoning=reasoning)


def _correct_field_values(text: str) -> set[bool]:
    """The distinct yes/no values of every line-anchored ``correct:`` **field** in ``text``.

    Only a line whose leading field name is ``correct`` counts — so ``correct: yes`` sitting
    inside a ``reasoning:`` line (which starts with ``reasoning``) is ignored. An empty set
    means no verdict; a set of size >1 means conflicting verdicts (both fail closed)."""
    return {
        m.group(1).lower() == "yes"
        for m in re.finditer(
            r"^[ \t]*correct[ \t]*:[ \t]*(yes|no)\b", text, re.IGNORECASE | re.MULTILINE
        )
    }


def _field(text: str, name: str) -> str:
    """Pull a single-line ``name: value`` field out of the judge reply (best-effort)."""
    m = re.search(rf"{re.escape(name)}\s*:\s*(.+)", text, re.IGNORECASE)
    return m.group(1).strip() if m is not None else ""
