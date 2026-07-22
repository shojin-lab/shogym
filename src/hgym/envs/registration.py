"""The environment registry: ``register`` decorator + ``make`` factory."""

from typing import Any, Callable, Dict, Optional, TypeVar

from hgym.core import Env

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


def registered_envs() -> list:
    """Return the names of all registered environments."""
    return list(_ENV_REGISTRY.keys())


def register(name: str) -> Callable[[type[_EnvT]], type[_EnvT]]:
    """Decorator: register an :class:`~hgym.core.Env` subclass under a unique ``name``."""

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
