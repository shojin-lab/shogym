"""The environment registry: ``register`` decorator + ``make`` factory."""

from typing import Any, Callable, Dict, Optional, TypeVar

from shogym.core import Env

_ENV_REGISTRY: Dict[str, type] = {}

_EnvT = TypeVar("_EnvT", bound=Env)


def make(env_name: str, config: Optional[Dict[str, Any]] = None) -> Env:
    """Instantiate a registered environment by name, passing ``config`` to its constructor.

    Stamps the registered name on the instance so ``env.describe()`` can report it.
    """
    if env_name not in _ENV_REGISTRY:
        raise ValueError(f"Environment '{env_name}' is not registered.")
    env = _ENV_REGISTRY[env_name](**(config or {}))
    env._registered_name = env_name
    return env


def identity_feedback_name(env_name: str) -> str:
    """The feedback item ``env_name`` declares as the identity of what produced a row.

    Read off the class rather than off an instance, because a record is checked when it is
    replayed as well as when it is written, and a replay has no live env. An env that declares
    nothing gets the empty string, which means nothing about its rows is read as an identity:
    the name is an ordinary one that any environment may publish as a metric, and a module that
    decided what it meant would turn another env's successful terminal into an unscored failure.
    """
    cls = _ENV_REGISTRY.get(env_name)
    return str(getattr(cls, "identity_feedback_name", "") or "") if cls is not None else ""


def registered_envs() -> list:
    """Return the names of all registered environments."""
    return list(_ENV_REGISTRY.keys())


def register(name: str) -> Callable[[type[_EnvT]], type[_EnvT]]:
    """Decorator: register an :class:`~shogym.core.Env` subclass under a unique ``name``."""

    def decorator(cls: type[_EnvT]) -> type[_EnvT]:
        if name in _ENV_REGISTRY:
            raise ValueError(f"An environment with name '{name}' is already registered.")
        if not issubclass(cls, Env):
            raise TypeError(
                f"Cannot register {cls.__name__} because it is not a subclass of Env."
            )
        _ENV_REGISTRY[name] = cls
        return cls

    return decorator
