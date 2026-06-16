"""The harness directory format, plus ``export_harness`` / ``load_harness``.

The **harness** is the editable projection of a rollout-and-agent configuration,
holding the environment and verifier fixed (RFC 000). It is what an optimizer
(human or coding agent) edits and re-runs. This module defines the on-disk format
and the two functions that write it from an env and read it back.

Layout (RFC 000 Section 5; surfaces are absent by default):

    harness/
    ├── harness.toml            # [inference] model + params; [limits] horizon
    ├── instruction/
    │   └── system.minijinja    # the system template (the only optimizable template)
    └── tools.toml              # OPTIONAL: extras MCP servers (the tool surface)

This PR (1 of the M1 stack) ships the format plus a faithful round-trip; later PRs
wire a loaded ``Harness`` into the agent (inference surface) and the function's
system prompt (instruction surface), and add per-surface hashing (observability).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Dict, List, Optional, TypeVar, Union

import tomli_w

from hgym.envs import make
from hgym.mcp.config import load_mcp_server_specs
from hgym.mcp.types import MCPServerSpec

_T = TypeVar("_T")


@dataclass
class Harness:
    """A loaded harness: the optimizable surfaces, decoupled from the env.

    Fields map to surfaces (RFC 000):

    - ``model`` + ``inference_params``: the **inference surface** (RFC 001 Section 6),
      everything an inference call takes besides messages and tools.
    - ``system_template``: the **instruction surface** (RFC 001), the system template
      text (or ``None`` if the env declares no system template).
    - ``extra_specs``: the **tool surface** extras (RFC 002), the optimizer-added MCP
      servers. Empty for a baseline harness.
    - ``horizon``: a **control**-surface limit (RFC 004); may only tighten the env's
      horizon, never loosen it (enforced where the harness is applied, a later PR).
    """

    model: str
    inference_params: Dict[str, Any] = field(default_factory=dict)
    system_template: Optional[str] = None
    extra_specs: List[MCPServerSpec] = field(default_factory=list)
    horizon: Optional[int] = None


def export_harness(
    env_name: str,
    model: str,
    path: Union[str, Path],
    *,
    inference_params: Optional[Dict[str, Any]] = None,
    env_config: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write a baseline harness directory for ``env_name`` to ``path``.

    Offline: builds the env to read its single chat function's system template and
    horizon, then writes ``harness.toml`` and ``instruction/system.minijinja``. It
    never writes ``tools.toml`` (a baseline harness has no extras; the optimizer
    creates that file when it engages the tool surface).

    Returns the harness directory path.
    """
    out = Path(path)
    env = make(env_name, config=env_config)
    try:
        fn_names = list(env.functions.keys())
        if len(fn_names) != 1:
            raise ValueError(
                f"export_harness expects a single-function env; {env_name!r} has "
                f"{len(fn_names)} functions: {fn_names!r}"
            )
        function = env.functions[fn_names[0]]
        system_template = getattr(function, "example_system_template", None)
        horizon = env.horizon
    finally:
        _run_sync(env.close())

    out.mkdir(parents=True, exist_ok=True)

    doc: Dict[str, Any] = {"inference": {"model": model, **(inference_params or {})}}
    if horizon is not None:
        doc["limits"] = {"horizon": horizon}
    (out / "harness.toml").write_text(tomli_w.dumps(doc))

    if system_template is not None:
        instruction = out / "instruction"
        instruction.mkdir(exist_ok=True)
        (instruction / "system.minijinja").write_text(system_template)

    return out


def load_harness(path: Union[str, Path]) -> Harness:
    """Read a harness directory into a :class:`Harness`.

    Surfaces are absent by default: a missing ``instruction/system.minijinja`` means
    the env declares no system template; a missing ``tools.toml`` means no extras.
    """
    src = Path(path)
    harness_toml = src / "harness.toml"
    if not harness_toml.exists():
        raise FileNotFoundError(f"no harness.toml in {src}")

    with harness_toml.open("rb") as f:
        doc = tomllib.load(f)

    inference = dict(doc.get("inference", {}))
    model = inference.pop("model", None)
    if model is None:
        raise ValueError(f"{harness_toml} is missing `inference.model`")
    horizon = doc.get("limits", {}).get("horizon")

    system_path = src / "instruction" / "system.minijinja"
    system_template = system_path.read_text() if system_path.exists() else None

    tools_path = src / "tools.toml"
    extra_specs: List[MCPServerSpec] = (
        load_mcp_server_specs(tools_path) if tools_path.exists() else []
    )

    return Harness(
        model=model,
        inference_params=inference,
        system_template=system_template,
        extra_specs=extra_specs,
        horizon=horizon,
    )


def _run_sync(coro: Awaitable[_T]) -> _T:
    """Run an async coroutine to completion from sync code (mirrors the helper in
    ``tool_using_env``): direct ``asyncio.run`` if no loop is running, else a worker
    thread with its own loop (async tests / notebooks)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)  # type: ignore[arg-type]
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()  # type: ignore[arg-type]
