# AutomationBench's source is provisioned at runtime into a cache dir (see `ensure_source`); it
# is intentionally absent from the base type-check / offline environment, so its imports are
# expected to be unresolved there.
# pyright: reportMissingImports=false
"""The single seam between hgym and the upstream ``automationbench`` package.

AutomationBench (MIT, © Zapier) can't be taken as an ordinary pinned git extra the way the
yc-bench and tau2 ports are: it declares ``requires-python >=3.13`` (hgym is hard-pinned to 3.12
because tau2 needs the stdlib ``audioop`` module removed in 3.13), and it depends on Prime
Intellect's heavy ``verifiers`` / ``anthropic`` agent-loop stack. So a ``pip``/``uv`` resolve of
``automation-bench`` under 3.12 is *unsatisfiable*.

The env-as-center port needs none of that loop. It reuses only the three deterministic,
``verifiers``-free pieces — the simulated tools + ``WorldState`` engine, the typed task defs, and
the pure rubric (all of which import fine on 3.12 with just ``pydantic`` + ``datasets``). So this
adapter **provisions the pinned upstream source at runtime** into a gitignored cache
(``~/.cache/hgym/automationbench/<sha>/``, overridable via ``AUTOMATIONBENCH_SRC`` /
``HGYM_CACHE``), registers it directly in ``sys.modules`` (never onto ``sys.path``, so the
checkout's sibling dirs can't shadow hgym's own packages), and imports from it — the same "fetch
upstream at runtime, never vendor it into the repo" shape the tau2 port uses for its data. Nothing
from upstream is committed to hgym.

It also re-hosts the two small, ``verifiers``-free helpers from upstream's ``runner.py``
(``strip_none_values`` + ``compute_allowed_services``) so the per-task ``WorldState`` seed and the
``allowed_services`` gate stay **byte-identical** to a real ``auto-bench`` run.

Importing this module triggers provisioning (a one-time network fetch if the cache is cold) and
imports ``automationbench``, so it is only ever imported when an ``automationbench`` env is
*constructed* (task load) or *served* — never by ``import hgym``.
"""

from __future__ import annotations

import copy
import importlib
import importlib.util
import io
import os
import sys
import tarfile
import tempfile
import threading
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Fidelity pin: the upstream commit this port reproduces (AutomationBench has no release tags).
UPSTREAM_SHA = "a321764ace3cfbe42289e6a13abef2f0f4f56fad"
_TARBALL_URL = f"https://github.com/zapier/AutomationBench/archive/{UPSTREAM_SHA}.tar.gz"
_DOWNLOAD_TIMEOUT_SECONDS = 120.0

_provision_lock = threading.Lock()


def _cache_root() -> Path:
    base = os.environ.get("HGYM_CACHE")
    root = Path(base) if base else Path.home() / ".cache" / "hgym"
    return root / "automationbench" / UPSTREAM_SHA


def _source_dir() -> Path:
    """The directory that *contains* the ``automationbench`` package.

    Honors ``AUTOMATIONBENCH_SRC`` (an existing checkout — a repo root, or a dir holding the
    package) so a provisioned/offline environment needs no network."""
    override = os.environ.get("AUTOMATIONBENCH_SRC")
    if override:
        return Path(override).expanduser().resolve()
    return _cache_root()


def _register_package(pkg_dir: Path) -> None:
    """Import the ``automationbench`` package from ``pkg_dir`` into ``sys.modules`` directly.

    Deliberately does **not** put ``pkg_dir``'s parent on ``sys.path``: the upstream checkout root
    can carry sibling top-level dirs (``tests/`` / ``visualizer/``) that would shadow hgym's own
    ``tests`` package. Registering the top package with its ``__path__`` set means every absolute
    ``from automationbench.x import y`` resolves through the package's own ``__path__`` — nothing
    leaks onto ``sys.path``. Idempotent."""
    if "automationbench" in sys.modules:
        return
    init = pkg_dir / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "automationbench", init, submodule_search_locations=[str(pkg_dir)]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load the automationbench package from {pkg_dir}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["automationbench"] = module
    spec.loader.exec_module(module)


def _download_source(dest: Path) -> None:
    """Fetch the pinned upstream tarball and extract it so ``dest/automationbench`` exists.

    Only the ``automationbench`` **package** is kept — the archive also carries top-level
    ``tests/`` / ``visualizer/`` dirs, which must never land on ``sys.path`` and shadow hgym's own
    ``tests`` package. Extracts atomically (temp dir + ``os.replace``) so concurrent provisioners
    can't observe a half-written tree, and uses tarfile's ``data`` filter to reject path
    traversal."""
    with urllib.request.urlopen(_TARBALL_URL, timeout=_DOWNLOAD_TIMEOUT_SECONDS) as resp:
        raw = resp.read()
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=str(dest.parent), prefix=".dl-") as tmp:
        tmp_path = Path(tmp)
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:gz") as tf:
            tf.extractall(tmp_path, filter="data")
        # The archive extracts to a single `AutomationBench-<sha>/` root; keep only its package.
        roots = [p for p in tmp_path.iterdir() if p.is_dir()]
        if len(roots) != 1 or not (roots[0] / "automationbench").is_dir():
            raise RuntimeError(
                f"unexpected AutomationBench archive layout: {[p.name for p in roots]}"
            )
        staged = tmp_path / ".staged"
        staged.mkdir()
        os.replace(roots[0] / "automationbench", staged / "automationbench")
        if dest.exists():  # a concurrent provisioner won the race — keep theirs
            return
        os.replace(staged, dest)


def ensure_source() -> Path:
    """Ensure the upstream source is available and importable; return its containing directory.

    Idempotent and thread-safe. If ``AUTOMATIONBENCH_SRC`` is set it is used as-is (no network).
    Otherwise the pinned tarball is downloaded into the cache on the first call and reused
    thereafter — so only the very first construction on a cold cache touches the network. The
    ``automationbench`` package is then registered directly in ``sys.modules`` (never onto
    ``sys.path``)."""
    src = _source_dir()
    with _provision_lock:
        if not (src / "automationbench").is_dir():
            if os.environ.get("AUTOMATIONBENCH_SRC"):
                raise RuntimeError(
                    f"AUTOMATIONBENCH_SRC={src} does not contain an 'automationbench' package"
                )
            _download_source(src)
        _register_package(src / "automationbench")
    return src


ensure_source()

from automationbench.rubric import (  # noqa: E402
    partial_credit as _partial_credit,
    task_completed_correctly as _task_completed_correctly,
)
from automationbench.schema.world import WorldState  # noqa: E402
from automationbench.tools.api.encode import base64_encode as _base64_encode  # noqa: E402
from automationbench.tools.api.fetch import api_fetch as _api_fetch  # noqa: E402
from automationbench.tools.api.search import api_search as _api_search  # noqa: E402

# ----- re-hosted verbatim from upstream automationbench/runner.py (the two verifiers-free
#       helpers that shape the per-task WorldState seed + service gate). Kept identical so
#       scoring matches a real auto-bench run. -----


def strip_none_values(obj: Any) -> Any:
    """Recursively strip ``None`` values from nested dicts and lists.

    Verbatim from upstream ``runner.strip_none_values``. HuggingFace ``Dataset`` normalizes
    schemas across rows, adding all possible keys and setting missing values to ``None``; this
    breaks Pydantic's ``default_factory`` since ``None`` is passed instead of the field being
    omitted. Upstream applies this in ``setup_state`` before constructing ``WorldState``, so the
    port applies it too."""
    if isinstance(obj, dict):
        return {k: strip_none_values(v) for k, v in obj.items() if v is not None}
    elif isinstance(obj, list):
        return [strip_none_values(item) for item in obj if item is not None]
    else:
        return obj


# Service field names on WorldState, longest first so prefix matching prefers
# "google_sheets" over a hypothetical "google" (verbatim from upstream runner).
_SERVICE_FIELDS = sorted(
    (str(f) for f in WorldState.model_fields if f != "meta"), key=len, reverse=True
)


def _service_for_name(name: str) -> Optional[str]:
    """Map an assertion type or tool name to its WorldState service field (verbatim)."""
    for field in _SERVICE_FIELDS:
        field = str(field)
        if name == field or name.startswith(field + "_"):
            return field
    return None


def compute_allowed_services(
    initial_state: dict, assertions: List[dict], zapier_tools: List[str]
) -> List[str]:
    """Derive the set of services a task's world is subscribed to (verbatim from upstream).

    A service is in-scope when the task seeds it (key present in ``initial_state``), asserts on
    it, or grants one of its Zapier tools. ``api_fetch`` rejects calls to out-of-scope services
    with a credentials error, closing the silent-diversion hole where writes to an unrelated
    vendor succeeded into untracked state."""
    allowed: set[str] = set()
    for key in initial_state:
        if key != "meta" and key in WorldState.model_fields:
            allowed.add(key)
    for a in assertions or []:
        service = _service_for_name(str(a.get("type", "")))
        if service:
            allowed.add(service)
    for tool_name in zapier_tools or []:
        service = _service_for_name(tool_name)
        if service:
            allowed.add(service)
    return sorted(allowed)


# ----- task loading (via the upstream domain datasets) -----


def load_domain_tasks(domain: str) -> List[Dict[str, Any]]:
    """Return one domain's tasks as a list of raw upstream task rows.

    Delegates to the upstream ``automationbench.domains.get_domain_dataset`` — the same loader
    ``auto-bench`` uses — so deterministic per-domain noise injection (seeded by ``example_id``)
    and the HuggingFace ``Dataset`` schema normalization + JSON-string ``info`` encoding are all
    reproduced verbatim. Each returned row has ``example_id``, ``task``, ``prompt`` (chat
    messages), ``answer``, and ``info`` (a JSON string, exactly as the dataset stores it)."""
    from automationbench.domains import (  # lazy: pulls in `datasets`
        DOMAIN_ALIASES,
        get_combined_dataset,
        get_domain_dataset,
    )

    if domain in DOMAIN_ALIASES:
        dataset = get_combined_dataset(DOMAIN_ALIASES[domain])
    else:
        dataset = get_domain_dataset(domain)
    return [dict(row) for row in dataset]


def available_domains() -> List[str]:
    from automationbench.domains import get_available_domains

    return get_available_domains()


# ----- per-task WorldState construction (re-hosts upstream setup_state, verifiers-free) -----


def build_world(info: Dict[str, Any]) -> Tuple[WorldState, Dict[str, Any], List[Dict[str, Any]]]:
    """Build a task's seeded ``WorldState`` from its ``info`` dict.

    Reproduces upstream ``AutomationBenchEnv.setup_state`` exactly, minus the verifiers plumbing:
    strip HuggingFace ``None`` padding from ``initial_state`` and ``assertions``, construct the
    ``WorldState``, and set ``meta.allowed_services`` from the seed / assertions / tool grant.
    Returns ``(world, stripped_initial_state, stripped_assertions)`` — the latter two are what
    :func:`score_state` needs to rerun the rubric (initial state gates free/negative assertions).
    """
    initial_state_dict = strip_none_values(info.get("initial_state", {}))
    assertions = [strip_none_values(a) for a in info.get("assertions", [])]
    world = WorldState(**initial_state_dict)
    world.meta.allowed_services = compute_allowed_services(
        initial_state_dict, assertions, info.get("zapier_tools", [])
    )
    return world, copy.deepcopy(initial_state_dict), assertions


# ----- served tool surface (the pinned `api` toolset) -----


def api_search(query: str, top_k: int = 5) -> str:
    """BM25 search over the upstream endpoint schemas (top-5 by default)."""
    return _api_search(query, top_k=top_k)


def api_fetch(
    world: WorldState,
    method: str,
    url: str,
    params: Optional[str] = None,
    body: Optional[str] = None,
) -> str:
    """Route a REST call into ``world`` via the upstream router (mutates ``world`` in place)."""
    return _api_fetch(world, method, url, params, body)


def base64_encode(text: str) -> str:
    """base64url-encode ``text`` (the Gmail body format); reused verbatim."""
    return _base64_encode(text)


# ----- pure scoring (reuses the upstream rubric verbatim) -----


def score_state(
    world_dump: Dict[str, Any],
    initial_state: Dict[str, Any],
    assertions: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """Score an end-state with the **reused** upstream rubric. Pure.

    Rebuilds ``WorldState`` from the recorded end-state dump, assembles the ``state`` dict shape
    ``partial_credit`` reads (``world`` / ``initial_state`` / ``info.assertions``), and calls the
    upstream ``partial_credit`` then ``task_completed_correctly`` — so the negative-assertion
    "must not shotgun" gate (negatives pass free in the initial world and only count when broken)
    and the pass-rate metric are byte-identical to ``auto-bench``. ``partial_credit`` must run
    first: it caches its score on the state, which ``task_completed_correctly`` reads back.

    Returns ``(partial_credit, task_completed_correctly)``.
    """
    world = WorldState(**world_dump)
    state: Dict[str, Any] = {
        "world": world,
        "initial_state": copy.deepcopy(initial_state),
        "info": {"assertions": assertions},
    }
    pc = float(_partial_credit(state))
    success = float(_task_completed_correctly(state))
    return pc, success


__all__ = [
    "UPSTREAM_SHA",
    "WorldState",
    "ensure_source",
    "strip_none_values",
    "compute_allowed_services",
    "load_domain_tasks",
    "available_domains",
    "build_world",
    "api_search",
    "api_fetch",
    "base64_encode",
    "score_state",
]
