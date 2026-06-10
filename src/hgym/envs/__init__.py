from hgym.envs.registration import make, register, registered_envs
from hgym.envs.wordle import env_v1 as wordle_env_v1  # noqa: F401 — triggers registration

__all__ = [
    "make",
    "register",
    "registered_envs",
    "wordle_env_v1",
]
