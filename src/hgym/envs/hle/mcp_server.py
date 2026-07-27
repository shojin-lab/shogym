"""In-process MCP server for the HLE env — it advertises the single ``submit_answer`` tool.

HLE has essentially no tool surface: an episode is a single ``submit_answer(answer,
confidence)`` call (plus the reserved ``terminate``). ``submit_answer`` is the env's **score
terminal**: the serve layer validates its args against the schema this tool advertises,
atomically **seals** the episode, then runs the env's ``finalize`` hook to grade — so this
handler body is **never dispatched** for a sealed episode. It exists only to publish the
tool's name and argument schema in the manifest ``describe()`` reads; grading (the exact-match
fast path + the injectable LLM judge) lives in the env's ``finalize``, and the per-episode
question / gold answer / judge live on the env, not here.

Because the seal makes ``submit_answer`` one-shot **structurally** (the first call seals; every
later call is tombstoned with no inward dispatch), this module needs no session state, no
grading, and no first-grade lock of its own.
"""

from __future__ import annotations

from typing import Any, Dict

from fastmcp import FastMCP

SUBMIT_TOOL_NAME = "submit_answer"

server: FastMCP = FastMCP(name="hle")


@server.tool
def submit_answer(answer: str, _session_id: str, confidence: int = 100) -> Dict[str, Any]:
    """Submit your final answer. **This ends the episode**: submitting seals the episode,
    grades ``answer`` against the gold answer (an exact-match fast path first, then the LLM
    judge on a miss), and returns a **public-safe** result:

      - ``correct``: whether your answer was judged correct
      - ``judge_error``: ``True`` only if the grader itself failed (a fail-closed zero, not an
        honest wrong answer); otherwise ``False``

    There is no second submission and no separate ``terminate`` — call this exactly once. The
    judge's reasoning / extracted answer are **not** returned to you (they are answer oracles,
    stripped by the serve layer before you see the result).
    """
    # Never reached for a sealed episode: the serve layer intercepts this `score` terminal,
    # validates + seals + runs the env's `finalize` instead of dispatching here. Kept as a
    # defensive, non-grading, no-oracle stub in case the tool is ever dispatched outside the
    # seal path.
    return {"submitted": True}
