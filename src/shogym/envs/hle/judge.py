"""The model-graded judge for the HLE env — shogym's first model-graded verifier (issue #33).

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
so ``import shogym`` (which imports the env to register it) stays offline and lean.
"""

from __future__ import annotations

import copy
import os
import re
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Protocol, runtime_checkable

# The registered `hle` env's default judge model, at that model's own default reasoning effort.
# A judge model is a scoring function, so this default is picked on measured grading quality (the
# measurement is in issue #122), not on price. Override via `judge_model` env config.
DEFAULT_JUDGE_MODEL = "gpt-5.6-luna"

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
    verdict). ``model`` is what the judge reports having graded with, which is what a score's
    provenance names: an alias, a router, or a local endpoint can answer as something other
    than what was asked for, and the score belongs to whatever actually ran.
    """

    correct: bool
    extracted_answer: str = ""
    reasoning: str = ""
    model: str = ""


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

    ``request_kwargs`` are sampling fields (``reasoning_effort``, say), sent verbatim and only
    when given: with none, the request carries the same fields it did before the parameter
    existed. Anything outside :data:`_ALLOWED_REQUEST_FIELDS` is refused at construction.
    """

    def __init__(
        self,
        model: str = DEFAULT_JUDGE_MODEL,
        *,
        base_url: Optional[str] = None,
        client: Any = None,
        request_kwargs: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self._model = model
        self._base_url = base_url
        self._client = client
        self._request_kwargs = _validated_request_kwargs(request_kwargs)

    @property
    def model(self) -> str:
        """The model id this judge grades with."""
        return self._model

    @property
    def request_kwargs(self) -> Mapping[str, Any]:
        """A copy of the extra request fields this judge sends.

        A copy all the way down, so neither what the caller still holds nor what it reads back
        here can change the request: grading stays one function for the length of a run."""
        return MappingProxyType(copy.deepcopy(dict(self._request_kwargs)))

    def _ensure_client(self) -> Any:
        if self._client is None:
            from openai import OpenAI  # lazy: keeps `import shogym` free of openai

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
            # Unset means absent, not null: with no kwargs this request carries the same fields
            # it did before the parameter existed, which keeps endpoints that reject unknown
            # fields working. A copy per call, because the client is arbitrary and one it can
            # edit would let one request change the next.
            **copy.deepcopy(dict(self._request_kwargs)),
        )
        text = completion.choices[0].message.content or ""
        # The model that answered, not the one that was asked for: provenance names what ran.
        return replace(
            parse_judge_response(text), model=str(getattr(completion, "model", "") or "")
        )


# Everything `request_kwargs` may set, and what each one does. The judge owns what it asks and
# the shape of the reply its parser reads; the caller owns how the model is sampled, which is
# this list. It is an allowlist and not a list of exclusions because the failure is silent and
# the surface grows: a request field that costs the reply its verdict line is scored as a wrong
# answer, not as a judge error, and any name nobody thought to exclude (a legacy one, a new SDK
# one) would arrive already permitted. Default-deny makes that a constructor error instead.
_ALLOWED_REQUEST_FIELDS: Mapping[str, str] = MappingProxyType(
    {
        "reasoning_effort": "how much the model reasons before it answers",
        "temperature": "how sharply the next token is drawn",
        "top_p": "how much of the distribution the next token is drawn from",
        "seed": "which draw the sampler makes",
        "frequency_penalty": "how much a token's own count discourages drawing it again",
        "presence_penalty": "how much a token having appeared discourages drawing it again",
    }
)


def _validated_request_kwargs(
    request_kwargs: Optional[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Deep-copy the caller's sampling fields, refusing every name that is not one.

    Deep, because a shallow copy leaves nested values shared with the caller: editing one after
    construction changes what a later episode is scored with, and no score shows it."""
    fields = copy.deepcopy(dict(request_kwargs or {}))
    unknown = sorted(name for name in fields if name not in _ALLOWED_REQUEST_FIELDS)
    if unknown:
        raise ValueError(
            f"judge request kwargs may not set {', '.join(repr(name) for name in unknown)}: only "
            f"sampling settings are settable ({', '.join(sorted(_ALLOWED_REQUEST_FIELDS))}). The "
            "judge owns the rest of the request and the shape of the reply it parses."
        )
    return MappingProxyType(fields)


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
