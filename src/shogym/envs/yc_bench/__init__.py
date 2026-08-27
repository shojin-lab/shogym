"""shogym port of YC-Bench (RFC 008 env-as-center).

This package is *registration-safe to import without yc-bench installed*: ``env_v1`` imports
nothing from ``yc_bench`` at module load, so ``import shogym`` (which registers the ``yc_bench``
env) stays offline. yc-bench is imported lazily — only when the env is actually *constructed*
(its in-process MCP server is probed) or served.

The port reuses YC-Bench's deterministic sim engine, command
execution/validation, SQLite state, world seeding, and scoring verbatim (funnelled through
:mod:`shogym.envs.yc_bench.adapter`), and replaces only the *agent* — the shogym harness drives
the ``run_command`` / ``submit`` tools that the env's in-process MCP server exposes.
"""
