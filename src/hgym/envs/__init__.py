from hgym.envs.registration import make, register, registered_envs

# Importing an env's `env_v1` module runs its `@register(...)` decorators. These imports
# must stay offline: neither pulls in heavy/optional deps at module load. In particular
# `hgym.envs.tau2.env_v1` imports nothing from `tau2` at top level (tau2 is loaded lazily,
# only when a tau2 env is constructed or served), so `import hgym` works without the
# `tau2` extra installed.
from hgym.envs.tau2 import env_v1 as tau2_env_v1  # noqa: F401 — triggers registration
from hgym.envs.wordle import env_v1 as wordle_env_v1  # noqa: F401 — triggers registration

__all__ = [
    "make",
    "register",
    "registered_envs",
    "tau2_env_v1",
    "wordle_env_v1",
]
