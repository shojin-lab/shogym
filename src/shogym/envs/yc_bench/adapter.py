# yc-bench's source is provisioned at runtime into a cache dir (see `ensure_source`); it is
# intentionally absent from the base type-check / offline environment, so its imports are
# expected to be unresolved there.
# pyright: reportMissingImports=false
"""The single seam between shogym and yc-bench's *internal* modules.

yc-bench ships **no stable public API** — the port reaches into its sim engine, world
seeder, config loader, command-validation policy, and ORM models. To keep upstream drift
contained to one file (per issue #32's fidelity caveat), *every* ``yc_bench`` import lives
here, behind a small, stable surface the rest of the env calls:

  - :func:`build_db` — open a fresh per-session SQLite database.
  - :func:`seed_session` — seed one deterministic company/world (reuses yc-bench's own
    ``_init_simulation``, so the seeded world matches ``yc-bench run``'s attributes for a given
    seed/config/start-date).
  - :func:`run_cli` — execute one ``yc-bench <cmd>`` against the session DB, reusing
    yc-bench's command-validation policy and its real CLI entry point.
  - :func:`read_final_state` — read the authoritative terminal metrics (funds, survival,
    task outcomes) off the sim DB.

The port used to pin the upstream commit as a direct (``@ git+https://``) requirement, which
PyPI rejects outright and which therefore made all of shogym unpublishable. So this adapter
**provisions the pinned upstream source at runtime** into a gitignored cache
(``~/.cache/shogym/yc_bench/<sha>/``, overridable via ``YC_BENCH_SRC`` / ``SHOGYM_CACHE``) and
registers it directly in ``sys.modules`` — never onto ``sys.path``, because the yc-bench archive
root carries sibling top-level dirs (``docs/``, ``scripts/``, ``system_design/``, ``imgs/``) that
would shadow shogym's own packages. Only ``src/yc_bench`` is extracted, so the cache dir holds the
package and nothing else — which is what makes it safe to hand to the CLI subprocess on
``PYTHONPATH`` (see :func:`run_cli`). The mechanics live in :mod:`shogym.envs._upstream`, shared
with the automationbench and tau2 ports. Nothing from upstream is committed to shogym, and the SHA
pin — hence the fidelity guarantee — is unchanged; it just moved from a requirement string to
:data:`UPSTREAM_SHA` here.

yc-bench's *own* runtime dependencies used to be resolved transitively by pip through that direct
requirement. They are now declared explicitly by the ``yc_bench`` extra in ``pyproject.toml``
(the upstream's ``[project] dependencies``, verbatim at the pinned SHA), so
``pip install shogym[yc_bench]`` still installs exactly the same set.

Importing this module triggers provisioning (a one-time network fetch if the cache is cold) and
imports ``yc_bench``, so it is only ever imported when a ``yc_bench`` env is *constructed*
(manifest probe) or *served*, never by ``import shogym``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

from shogym.envs._upstream import ensure_package

# Fidelity pin: the upstream commit this port reproduces.
UPSTREAM_SHA = "e7d606789be4c52a34f9fa5b04ada4a2eaf9d731"
_TARBALL_URL = f"https://github.com/collinear-ai/yc-bench/archive/{UPSTREAM_SHA}.tar.gz"


def ensure_source() -> Path:
    """Ensure the upstream source is available and importable; return its containing directory.

    Idempotent and thread-safe. ``YC_BENCH_SRC`` overrides the cache with an existing checkout (a
    dir that *contains* a ``yc_bench`` package), so a provisioned/offline environment needs no
    network; otherwise the pinned tarball is fetched on the first call and reused thereafter.
    yc-bench is a src-layout project, so the package sits at ``<archive root>/src/yc_bench``."""
    return ensure_package(
        package="yc_bench", sha=UPSTREAM_SHA, tarball_url=_TARBALL_URL, archive_subdir="src"
    )


_SOURCE_DIR = ensure_source()

from yc_bench.agent.commands.policy import parse_bench_command  # noqa: E402
from yc_bench.config import load_config  # noqa: E402
from yc_bench.db.models.company import Company  # noqa: E402
from yc_bench.db.models.sim_state import SimState  # noqa: E402
from yc_bench.db.models.task import Task, TaskStatus  # noqa: E402
from yc_bench.db.session import (  # noqa: E402
    build_engine,
    build_session_factory,
    init_db,
    session_scope,
)
from yc_bench.runner.args import RunArgs  # noqa: E402
from yc_bench.runner.main import _init_simulation  # noqa: E402

# yc-bench's ORM/runtime helpers are dynamically typed at these call sites; treat the few
# attribute reads as ``Any`` rather than fighting upstream annotations. Runtime is covered by
# the served/fidelity tests.
_Any = Any

# The only ``yc-bench`` sub-command groups the harness may drive: the observe/act/sim/memory
# surface that operates the *already-seeded* session. This deliberately excludes ``run`` and
# ``start`` — YC-Bench's own LLM-driven agent loop and interactive quickstart. ``run`` would
# spawn an unbounded, credential-inheriting model loop that replaces DATABASE_URL and writes
# its own artifacts, none of it represented as the harness's individual, trace-attributable
# actions — so the port allowlists the operational groups and rejects everything else.
ALLOWED_COMMAND_GROUPS = frozenset(
    {
        "company",
        "employee",
        "market",
        "task",
        "sim",
        "finance",
        "report",
        "scratchpad",
        "client",
    }
)


def build_db(db_url: str) -> Tuple[Any, Any]:
    """Create (or open) the SQLite database at ``db_url`` and return ``(engine, factory)``.

    Reuses yc-bench's own engine/session construction so the schema and pragmas match a real
    ``yc-bench run`` exactly."""
    engine = build_engine(db_url)
    init_db(engine)
    factory = build_session_factory(engine)
    return engine, factory


@contextmanager
def _db_factory(session_factory: Any) -> Iterator[Any]:
    with session_scope(session_factory) as session:
        yield session


def seed_session(
    session_factory: Any,
    *,
    seed: int,
    config_name: str,
    start_date: str,
    horizon_years: Optional[int],
    company_name: str,
) -> str:
    """Seed one company/world into the session DB from the task seed; return the company id.

    Delegates to yc-bench's own ``_init_simulation`` (the exact function ``yc-bench run``
    calls), so for a fixed seed/config/start-date the seeded world — employees, clients,
    market tasks, horizon event, and initial ``SimState`` — carries the same *attributes* as
    upstream's. The row **ids** are not reproducible: upstream's ``services/seed_world.py``
    mints a ``uuid4()`` per company, employee, client and market task."""
    experiment_cfg = load_config(config_name)
    resolved_horizon = (
        horizon_years if horizon_years is not None else experiment_cfg.sim.horizon_years
    )
    args = RunArgs(
        model="shogym",  # unused by seeding (only the agent loop reads it)
        seed=seed,
        horizon_years=horizon_years,
        company_name=company_name,
        start_date=start_date,
        config_name=config_name,
    )
    company_id = _init_simulation(
        lambda: _db_factory(session_factory), args, experiment_cfg, resolved_horizon
    )
    return str(company_id)


# How the CLI subprocess is launched. yc-bench's `yc-bench` console script is generated by an
# *install* of the distribution, and the port no longer installs one (the source is provisioned
# into a cache instead), so the equivalent invocation is used: `python -m yc_bench` runs upstream's
# `yc_bench/__main__.py`, which calls the very same `yc_bench.cli:app_main` the console script
# points at. Same entry point, same argument parsing, same JSON on stdout.
#
# `-P` is the subprocess half of this package's sys.path hygiene, and it is not optional. `-m`
# prepends the *working directory* to sys.path, ahead of everything PYTHONPATH contributes — so a
# harness whose cwd happens to hold a `yc_bench/` directory would shadow the pinned source
# completely and silently, which is the same class of bug the parent process avoids by binding
# through sys.modules instead of sys.path. `-P` suppresses that prepend, leaving the provisioned
# source (below) as the first place the name resolves.
_CLI_LAUNCH = (sys.executable, "-P", "-m", "yc_bench")


def _subprocess_env(base_env: Dict[str, str]) -> Dict[str, str]:
    """The subprocess environment, with the provisioned source reachable by ``python -m``.

    The cache dir contains the ``yc_bench`` package and nothing else (only ``src/yc_bench`` is
    extracted), so prepending it to ``PYTHONPATH`` can shadow nothing — while still letting the
    subprocess import upstream from the same pinned source this process registered in
    ``sys.modules``. An inherited ``PYTHONPATH`` is preserved after it. (A ``YC_BENCH_SRC``
    override points somewhere the caller controls; upstream's own layout puts nothing but
    ``yc_bench`` beside it either.)

    ``PYTHONPATH`` alone would not be enough — see :data:`_CLI_LAUNCH` for why the subprocess also
    runs under ``-P``."""
    existing = base_env.get("PYTHONPATH")
    source = str(_SOURCE_DIR)
    return {
        **base_env,
        "PYTHONPATH": f"{source}{os.pathsep}{existing}" if existing else source,
    }


def validate_command(command: Any) -> Tuple[bool, Optional[str], Optional[List[str]]]:
    """Reuse yc-bench's command-validation policy (only top-level ``yc-bench`` commands)."""
    return parse_bench_command(command)


def run_cli(
    command: str,
    *,
    db_url: str,
    config_name: str,
    timeout_seconds: float,
    base_env: Dict[str, str],
) -> Dict[str, Any]:
    """Execute one ``yc-bench <cmd>`` against ``db_url`` and return a structured result.

    Mirrors upstream ``yc_bench.agent.commands.executor.run_command`` (same validation, same
    result shape) but injects ``DATABASE_URL`` / ``YC_BENCH_EXPERIMENT`` *explicitly* via the
    subprocess ``env`` rather than mutating ``os.environ`` — so concurrent sessions can never
    race on process-global state. yc-bench's real CLI entry point runs the command (see
    :data:`_CLI_LAUNCH`), so command parsing, execution, and JSON output are yc-bench's own."""
    ok, err, argv = validate_command(command)
    if not ok or argv is None:
        return {
            "ok": False,
            "exit_code": 2,
            "stdout": "",
            "stderr": err or "invalid command",
            "command": command if isinstance(command, str) else str(command),
        }

    argv = list(argv)
    # Allowlist the operational sub-command groups; reject `run` / `start` / anything else
    # *before* spawning, so a command that would launch YC-Bench's own agent loop (or an
    # interactive prompt) can never run. This keeps the surface offline and trace-attributable.
    group = argv[1] if len(argv) > 1 else None
    if group not in ALLOWED_COMMAND_GROUPS:
        allowed = ", ".join(sorted(ALLOWED_COMMAND_GROUPS))
        return {
            "ok": False,
            "exit_code": 2,
            "stdout": "",
            "stderr": (
                f"command group not permitted: {group!r}. "
                f"Only these yc-bench groups may be run: {allowed}."
            ),
            "command": command,
        }
    # `argv[0]` is the literal "yc-bench" the policy parsed; swap it for the equivalent
    # `python -m yc_bench` launch, leaving the parsed arguments untouched.
    argv = [*_CLI_LAUNCH, *argv[1:]]
    env = {
        **_subprocess_env(base_env),
        "DATABASE_URL": db_url,
        "YC_BENCH_EXPERIMENT": config_name,
    }

    try:
        proc = subprocess.run(
            argv,
            shell=False,
            text=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=env,
        )
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
            "command": command,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "ok": False,
            "exit_code": 124,
            "stdout": exc.stdout or "",
            "stderr": f"command timed out after {timeout_seconds} seconds",
            "command": command,
        }
    except Exception as exc:  # keep a broken command from killing the server
        return {
            "ok": False,
            "exit_code": 1,
            "stdout": "",
            "stderr": str(exc),
            "command": command,
        }


def read_final_state(session_factory: Any) -> Dict[str, Any]:
    """Read the authoritative terminal metrics off the sim DB.

    This is the sim's own state — the same rows ``company status`` / ``sim resume`` report —
    read directly so the terminal verdict can't be forged through the command surface."""
    with _db_factory(session_factory) as db:
        sim_state = db.query(SimState).first()
        if sim_state is None:
            return {
                "seeded": False,
                "survived": False,
                "final_funds_cents": 0,
                "tasks_succeeded": 0,
                "tasks_failed": 0,
                "horizon_reached": False,
                "terminal_reason": None,
                "sim_time": None,
                "horizon_end": None,
            }
        company: _Any = (
            db.query(Company).filter(Company.id == sim_state.company_id).one_or_none()
        )
        funds = int(company.funds_cents) if company is not None else 0
        succeeded = (
            db.query(Task)
            .filter(
                Task.company_id == sim_state.company_id,
                Task.status == TaskStatus.COMPLETED_SUCCESS,
            )
            .count()
        )
        failed = (
            db.query(Task)
            .filter(
                Task.company_id == sim_state.company_id,
                Task.status == TaskStatus.COMPLETED_FAIL,
            )
            .count()
        )
        horizon_reached = sim_state.sim_time >= sim_state.horizon_end
        survived = funds >= 0
        terminal_reason: Optional[str]
        if not survived:
            terminal_reason = "bankruptcy"
        elif horizon_reached:
            terminal_reason = "horizon_end"
        else:
            terminal_reason = None
        return {
            "seeded": True,
            "survived": survived,
            "final_funds_cents": funds,
            "tasks_succeeded": int(succeeded),
            "tasks_failed": int(failed),
            "horizon_reached": bool(horizon_reached),
            "terminal_reason": terminal_reason,
            "sim_time": sim_state.sim_time.isoformat(),
            "horizon_end": sim_state.horizon_end.isoformat(),
        }


__all__ = [
    "ALLOWED_COMMAND_GROUPS",
    "UPSTREAM_SHA",
    "ensure_source",
    "build_db",
    "seed_session",
    "run_cli",
    "validate_command",
    "read_final_state",
]
