"""shogym port of tau2-bench (RFC 008 env-as-center).

This package is *registration-safe to import without tau2 installed*: ``env_v1``
imports nothing from ``tau2`` at module load, so ``import shogym`` (which registers
``tau2_<domain>`` envs) stays offline. tau2 is imported lazily — only when a tau2
env is actually *constructed* (its in-process MCP server is probed) or served.

The port is a faithful wrap: it reuses tau2's Orchestrator, user simulator, domain
tools/tasks, and evaluator verbatim, and replaces only the *agent* — via tau2's own
``GymAgent`` control-inversion bridge (``tau2.gym``), which runs the Orchestrator on a
background thread and blocks the agent's turn until the shogym harness makes its next
MCP tool call.
"""
