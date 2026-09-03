# AutomationBench's source is provisioned at runtime into a cache dir (see `ensure_source`); it
# is intentionally absent from the base type-check / offline environment, so its imports are
# expected to be unresolved there.
# pyright: reportMissingImports=false
"""The single seam between shogym and the upstream ``automationbench`` package.

AutomationBench (MIT, © Zapier) cannot be resolved by ``pip`` at all under shogym's Python pin: it
declares ``requires-python >=3.13`` (shogym is hard-pinned to 3.12 because tau2 needs the stdlib
``audioop`` module removed in 3.13), and it depends on Prime Intellect's heavy ``verifiers`` /
``anthropic`` agent-loop stack. So a ``pip``/``uv`` resolve of ``automation-bench`` under 3.12 is
*unsatisfiable*.

The env-as-center port needs none of that loop. It reuses only the three offline,
``verifiers``-free pieces — the simulated tools + ``WorldState`` engine, the typed task defs, and
the pure rubric (all of which import fine on 3.12 with just ``pydantic`` + ``datasets``). So this
adapter **provisions the pinned upstream source at runtime** into a gitignored cache
(``~/.cache/shogym/automationbench/<sha>/``, overridable via ``AUTOMATIONBENCH_SRC`` /
``SHOGYM_CACHE``), registers it directly in ``sys.modules`` (never onto ``sys.path``, so the
checkout's sibling dirs can't shadow shogym's own packages), and imports from it. The mechanics
live in :mod:`shogym.envs._upstream`, shared with the tau2 and yc_bench ports (which adopted the
same pattern so shogym carries no direct-URL requirement and can be published to PyPI). Nothing
from upstream is committed to shogym.

It also re-hosts the two small, ``verifiers``-free helpers from upstream's ``runner.py``
(``strip_none_values`` + ``compute_allowed_services``) so the per-task ``WorldState`` seed and the
``allowed_services`` gate stay **byte-identical** to a real ``auto-bench`` run.

Importing this module triggers provisioning (a one-time network fetch if the cache is cold) and
imports ``automationbench``, so it is only ever imported when an ``automationbench`` env is
*constructed* (task load) or *served* — never by ``import shogym``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
from urllib.parse import unquote, urlparse

from shogym.envs._upstream import ensure_package

# Fidelity pin: the upstream commit this port reproduces. AutomationBench cuts no git tags and
# publishes no releases, so a release is named only in its changelog; this commit is the one whose
# message reads "1.0.6 release. Bug fixes."
UPSTREAM_SHA = "6d210543b7a046f0f451c828cd2dadef774276eb"
_TARBALL_URL = f"https://github.com/zapier/AutomationBench/archive/{UPSTREAM_SHA}.tar.gz"


def ensure_source() -> Path:
    """Ensure the upstream source is available and importable; return its containing directory.

    Idempotent and thread-safe. See :mod:`shogym.envs._upstream`: ``AUTOMATIONBENCH_SRC`` overrides
    the cache (no network), otherwise the pinned tarball is fetched into
    ``~/.cache/shogym/automationbench/<sha>/`` on the first call and reused thereafter, and the
    ``automationbench`` package is registered directly in ``sys.modules`` (never onto
    ``sys.path``). AutomationBench uses a flat layout, so the package sits at the archive root."""
    return ensure_package(
        package="automationbench", sha=UPSTREAM_SHA, tarball_url=_TARBALL_URL
    )


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
    reproduced verbatim. Each returned row has ``example_id``, ``prompt`` (chat messages),
    ``answer``, and ``info`` (a JSON string, exactly as the dataset stores it). The pinned upstream
    carries the task's name inside ``info`` as ``task_name``; the top-level ``task`` column older
    revisions had is absent from every shipped domain, the combined ``public`` alias included. Read
    the name through :func:`shogym.envs.automationbench.env_v1.task_name`, which covers both."""
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
    """Route a REST call into ``world`` via the upstream router (mutates ``world`` in place).

    One answer is completed on the way out: the Jira project search also reports the projects the
    task seeded (see :func:`_with_seeded_jira_projects`). Every other response is the router's own.
    """
    response = _api_fetch(world, method, url, params, body)
    if str(method).strip().upper() == "GET" and _is_project_search(url):
        return _with_seeded_jira_projects(world, params, response)
    return response


# ``JiraState.projects`` is a declared field that nothing reads. The project search is the only Jira
# lookup on the served surface, and it answers out of the recorded project *actions*, so a project
# the task seeded in that collection is invisible to lookup: the agent cannot learn its key, and a
# key it names in a request is one it brought rather than one it found. This completes the search's
# answer with those projects, leaving the action-log results the route already found as they were.
_PROJECT_SEARCH_PATH = "rest/api/3/project/search"
_ATLASSIAN_HOST_SUFFIX = ".atlassian.net"
_JIRA_INTERNAL_PREFIX = "jira/"


def _is_project_search(url: str) -> bool:
    """Whether ``url`` names the Jira project search, the way the router names it.

    The router parses the URL and percent-decodes each path segment, sends a path under an
    ``.atlassian.net`` host to Jira, and routes a host-less URL by its own leading segment instead.
    So the route is that whole path under an Atlassian host, or the internal path itself, and both
    an encoded spelling (``project%2Fsearch``) and a path parameter (``project/search;v=1``) reach
    it. The path's *tail* does not name it: Google Sheets takes an A1 range as its own last
    segment, a decoded slash inside that range stays part of the range, and such a read can end in
    these very segments while being someone else's route."""
    parsed = urlparse(str(url))
    path = "/".join(unquote(segment) for segment in parsed.path.split("/")).strip("/")
    if parsed.netloc:
        return parsed.netloc.endswith(_ATLASSIAN_HOST_SUFFIX) and path == _PROJECT_SEARCH_PATH
    return path == _JIRA_INTERNAL_PREFIX + _PROJECT_SEARCH_PATH


def _search_query(params: Optional[str]) -> str:
    """The literal Jira project search filters on, from the request's params.

    The route reads ``query`` and nothing else: its own ``searchByParameter`` filter is built from
    that one field, and a caller who spells that name instead is passing an unread keyword. So an
    empty or absent ``query`` is a search for everything, here as there."""
    parsed: Any = params
    if isinstance(parsed, str):
        try:
            parsed = json.loads(parsed)
        except ValueError:
            return ""
    if not isinstance(parsed, dict):
        return ""
    return str(parsed.get("query") or "")


def _with_seeded_jira_projects(world: WorldState, params: Optional[str], response: str) -> str:
    """Add the world's own Jira projects to a project search answer.

    A project matches the way Jira's own search matches, on a case-insensitive literal in its key
    or name, and an empty query matches everything. A project the route already returned is left
    alone, identified case-insensitively too so one key is never reported twice in two spellings,
    and a response that is not a search result (a 401 from the service gate, say) is returned
    untouched."""
    seeded = [p for p in (world.jira.projects or []) if isinstance(p, dict)]
    if not seeded:
        return response
    try:
        payload = json.loads(response)
    except ValueError:
        return response
    if not isinstance(payload, dict) or not isinstance(payload.get("values"), list):
        return response

    values: List[Any] = list(payload["values"])
    seen = {
        str(v.get("key") or v.get("project") or v.get("id")).casefold()
        for v in values
        if isinstance(v, dict)
    }
    query = _search_query(params).casefold()
    for project in seeded:
        record = {"id": project.get("key") or project.get("name"), **project}
        key = str(record.get("key") or record.get("id")).casefold()
        if key in seen:
            continue
        if query and not any(query in str(record.get(f, "")).casefold() for f in ("key", "name")):
            continue
        seen.add(key)
        values.append(record)

    payload["values"] = values
    payload["total"] = len(values)
    return json.dumps(payload)


def base64_encode(text: str) -> str:
    """base64url-encode ``text`` (the Gmail body format); reused verbatim."""
    return _base64_encode(text)


# ----- pure scoring (reuses the upstream rubric verbatim) -----


def score_state(
    world: Union[WorldState, Dict[str, Any]],
    initial_state: Dict[str, Any],
    assertions: List[Dict[str, Any]],
) -> Tuple[float, float]:
    """Score an end-state with the **reused** upstream rubric.

    Assembles the ``state`` dict shape ``partial_credit`` reads (``world`` / ``initial_state`` /
    ``info.assertions``) and calls the upstream ``partial_credit`` then
    ``task_completed_correctly``, so the negative-assertion "must not shotgun" gate (negatives
    pass free in the initial world and only count when broken) and the pass-rate metric are
    byte-identical to ``auto-bench``. ``partial_credit`` must run first: it caches its score on
    the state, which ``task_completed_correctly`` reads back.

    ``world`` is the **live** :class:`WorldState` the served tools mutated, scored as-is. That is
    what upstream's own runner does, and a serialize/revalidate round trip in between is not
    reliably a no-op. An ordinary write to a declared field does survive one; two things do not:

    - It can **lose evidence.** Part of what the rubric reads is recorded by the tool layer
      *outside* the model's declared fields, so it does not survive ``model_dump``. Scoring a
      reconstruction can silently read a world that has forgotten writes the agent made.
    - It can **raise.** A raise scores nothing at all. The pinned upstream closed the two failures
      this port had observed, by populating aliased fields by name and by validating on assignment,
      but not every way a tool can reach one: validating on assignment does not see an in-place
      mutation of the container an attribute already holds, and the tools reach several of those.
      A served world can still be one its own dump cannot rebuild.

    The rubric only reads the world (its handlers are pure and it caches its score on ``state``),
    and this runs on an already-sealed episode whose world is discarded immediately afterwards,
    so scoring the caller's object in place is safe and no defensive copy is taken.

    A mapping is still accepted, and is revalidated into a ``WorldState`` before scoring. That
    path is **lossy** for the reasons above and exists only so callers written against the older
    dump-taking signature keep working; prefer passing the live object.

    Returns ``(partial_credit, task_completed_correctly)``.
    """
    if not isinstance(world, WorldState):
        world = WorldState(**world)
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
