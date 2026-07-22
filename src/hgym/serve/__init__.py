"""Serving an environment to an external harness (RFC 008 §3.2, §5).

A :class:`ServedEpisode` is the transport-independent engine: it wraps a reset env and
turns one incoming tool call into one env step, returning the tool's functional result
plus the feedback ``_meta`` sidecar (:mod:`hgym.feedback`) and recording the step to the
trace (:mod:`hgym.trace`). PR4 wraps this engine in a FastMCP stdio server (`hgym serve`)
so Claude Code / Codex / pi / Hermes can drive it; the engine itself is exercised
in-process, no subprocess required.
"""

from hgym.serve.episode import CallResult, ServedEpisode

__all__ = ["CallResult", "ServedEpisode"]
