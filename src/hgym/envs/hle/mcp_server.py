"""In-process MCP server for the HLE env — one tool, ``submit_answer``, that grades server-side.

HLE has essentially no tool surface: an episode is a single ``submit_answer(answer,
confidence)`` call (plus the reserved ``terminate``). The **judge runs here, in the tool
handler**: on submission the handler tries a deterministic exact-match fast path and, on a
miss, calls the session's (injectable) LLM judge. It records the verdict + the caller's
confidence as a marked JSON result. The env's pure ``_verify`` later parses that marked
``submit_answer`` result off the recorded trajectory — so verification stays a pure function
and the model-grading side effect is confined to the served handler (the same
server-side-verdict pattern the tau2 port uses for ``done``).

State is keyed by ``_session_id`` (hgym injects the real id for tools that declare it), so
several episodes can share this module safely. Lifecycle:

- ``begin_session(session_id, question=…, correct_answer=…, judge=…)`` — the env pushes the
  question, its gold answer, and the judge to use when an episode starts.
- ``submit_answer(answer, confidence, _session_id)`` — grades and returns the marked verdict.
- ``end_session(session_id)`` — drops the per-session state on teardown.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from fastmcp import FastMCP

from hgym.envs.hle.judge import Judge, exact_match

# Marker stamped on the graded `submit_answer` result so the pure verifier can find the
# verdict in the recorded trajectory without trusting arbitrary tool output (mirrors tau2's
# `tau2_verdict`). Only a `submit_answer` step carrying this marker is trusted for credit.
GRADE_MARKER = "hle_grade"
SUBMIT_TOOL_NAME = "submit_answer"

# HLE confidence is a 0–100 integer; clamp out-of-range values rather than reject them.
MIN_CONFIDENCE = 0
MAX_CONFIDENCE = 100
DEFAULT_CONFIDENCE = 100

server: FastMCP = FastMCP(name="hle")

# session_id -> {question, correct_answer, judge, answer_type}
_sessions: Dict[str, Dict[str, Any]] = {}
_lock = threading.RLock()


def begin_session(
    session_id: str,
    *,
    question: str,
    correct_answer: str,
    judge: Judge,
    answer_type: str = "",
) -> None:
    """Register a fresh episode's question, gold answer, and judge.

    Called by the env (in-process) when an episode starts, before any tool is called.
    Idempotent within a ``session_id``."""
    with _lock:
        _sessions[session_id] = {
            "question": question,
            "correct_answer": correct_answer,
            "judge": judge,
            "answer_type": answer_type,
        }


def end_session(session_id: str) -> None:
    """Drop a finished episode's per-session state. Idempotent."""
    with _lock:
        _sessions.pop(session_id, None)


@server.tool
def submit_answer(
    answer: str, _session_id: str, confidence: int = 100
) -> Dict[str, Any]:
    """Submit your final answer. **This ends the episode** (RFC 009): submitting seals the
    episode, grades ``answer`` against the gold answer (an exact-match fast path first, then
    the LLM judge on a miss), and returns a **public-safe** result:

      - ``correct``: whether your answer was judged correct
      - ``judge_error``: ``True`` only if the grader itself failed (a fail-closed zero, not
        an honest wrong answer); otherwise ``False``

    There is no second submission and no separate ``terminate`` — call this exactly once.
    (Internally the handler records a fuller, server-owned marked grade on the sealed step
    for the pure verifier; the judge's reasoning/extracted answer are **not** returned to
    you — they are answer oracles, stripped by the serve layer before you see the result.)
    """
    conf = _clamp_confidence(confidence)
    with _lock:
        state = _sessions.get(_session_id)
        if state is None:
            return {
                "error": "session not initialized; env did not call begin_session",
                "correct": False,
                "confidence": conf,
            }
        if state.get("graded"):
            # Single-turn: only the first answer is graded. Return an **unmarked** notice so a
            # replacement submission can never become terminal credit (no feedback loop — the
            # first verdict stands). The horizon still lets the harness then `terminate`.
            return {
                "error": "answer already submitted; only the first answer is graded",
                "correct": False,
                "confidence": conf,
            }
        # Claim the single grading slot now, under the lock, so a concurrent or sequential
        # second call is rejected even though the (possibly slow) judge runs outside the lock.
        state["graded"] = True
        question = state["question"]
        gold = state["correct_answer"]
        judge: Judge = state["judge"]

    # Fast path: a normalized exact match short-circuits the LLM judge (offline, free).
    if exact_match(answer, gold):
        return _grade(True, conf, judged_by="exact_match", extracted_answer=str(answer))

    try:
        verdict = judge(question=question, correct_answer=gold, response=answer)
    except Exception as exc:
        # A judge failure (network, key, parse) must not crash the served episode. Score as
        # incorrect and record the reason; the run stays scoreable and attributable.
        return _grade(
            False,
            conf,
            judged_by="llm_judge_error",
            extracted_answer=str(answer),
            reasoning=f"judge error: {exc}",
        )
    return _grade(
        bool(verdict.correct),
        conf,
        judged_by="llm_judge",
        extracted_answer=verdict.extracted_answer or str(answer),
        reasoning=verdict.reasoning,
    )


def _grade(
    correct: bool,
    confidence: int,
    *,
    judged_by: str,
    extracted_answer: str = "",
    reasoning: str = "",
) -> Dict[str, Any]:
    return {
        GRADE_MARKER: True,
        "correct": correct,
        "confidence": confidence,
        "judged_by": judged_by,
        "extracted_answer": extracted_answer,
        "reasoning": reasoning,
    }


def _clamp_confidence(confidence: Optional[Any]) -> int:
    """Coerce ``confidence`` to an int in [0, 100]; junk falls back to 100."""
    try:
        value = int(confidence)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return DEFAULT_CONFIDENCE
    return max(MIN_CONFIDENCE, min(MAX_CONFIDENCE, value))


def reset_state() -> None:
    """Drop all sessions (test hygiene)."""
    with _lock:
        _sessions.clear()
