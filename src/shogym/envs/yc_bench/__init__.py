"""shogym port of YC-Bench (RFC 008 env-as-center).

This package is *registration-safe to import without yc-bench installed*: ``env_v1`` imports
nothing from ``yc_bench`` at module load, so ``import shogym`` (which registers the ``yc_bench``
env) stays offline. yc-bench is imported lazily — only when the env is actually *constructed*
(its in-process MCP server is probed) or served.

The port reuses YC-Bench's seeded sim engine, CLI entry point and command validation,
SQLite state/ORM, and ``_init_simulation`` world seeding verbatim (funnelled through
:mod:`shogym.envs.yc_bench.adapter`). It replaces upstream's own LLM agent loop
(``agent/loop.py``, driven by ``runner/main.py``) and supplies the command, terminal and
scoring layers around the sim: the shogym harness drives the ``run_command`` / ``submit``
tools that the env's in-process MCP server exposes.
"""
