"""`extra_toolset` forwarding through the `make` factory (factory-level only).

Covers how the kwarg is *forwarded* by the env factory, not env tool-merging (that lives
in `tests/envs/test_tool_using_env.py`). Lightweight stubs — no MCP servers or API calls.
The runner-forwarding tests were removed with `hgym.runner`; these factory tests stay
because `make`'s `extra_toolset` interface is unchanged here (it is removed in a later PR).
"""

import pytest

from hgym.core import Env
from hgym.envs.registration import make, register
from hgym.mcp import MCPToolset

# What `_CaptureEnv` was last constructed with (module-level so its type is plain `dict`).
_CAPTURED: dict = {}


def _toolset() -> MCPToolset:
    """A real but empty toolset — satisfies the type without opening any server."""
    return MCPToolset(specs=[], tool_configs_by_server={})


@register("capture_extra_toolset_v0")
class _CaptureEnv(Env):
    """Records the kwargs `make` constructs it with; not a real env.

    Intentionally skips ``super().__init__`` — these tests only assert how ``make``
    forwards kwargs, so a valid env is unnecessary.
    """

    def __init__(self, *, semaphore=None, **kwargs) -> None:
        _CAPTURED.clear()
        _CAPTURED.update({"semaphore": semaphore, **kwargs})

    async def _reset(self, task_idx=None):  # pragma: no cover - never called
        raise NotImplementedError

    async def _step(self, action):  # pragma: no cover - never called
        raise NotImplementedError

    async def close(self):  # pragma: no cover - never called
        pass


def test_make_forwards_extra_toolset():
    ts = _toolset()
    make("capture_extra_toolset_v0", extra_toolset=ts)
    assert _CAPTURED.get("extra_toolset") is ts


def test_make_omits_extra_toolset_when_none():
    # Envs that don't accept `extra_toolset` must not receive it by default.
    make("capture_extra_toolset_v0")
    assert "extra_toolset" not in _CAPTURED


def test_make_rejects_extra_toolset_passed_twice():
    with pytest.raises(ValueError, match="only once"):
        make(
            "capture_extra_toolset_v0",
            config={"extra_toolset": _toolset()},
            extra_toolset=_toolset(),
        )


def test_make_accepts_extra_toolset_via_config():
    # The pre-existing path (in `config`) still works and reaches the env.
    ts = _toolset()
    make("capture_extra_toolset_v0", config={"extra_toolset": ts})
    assert _CAPTURED.get("extra_toolset") is ts
