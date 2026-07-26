from hgym.envs.registration import make, register, registered_envs

# Importing an env's `env_v1` module runs its `@register(...)` decorators. These imports
# must stay offline: neither pulls in heavy/optional deps at module load. In particular
# `hgym.envs.tau2.env_v1` imports nothing from `tau2` at top level (tau2 is loaded lazily,
# only when a tau2 env is constructed or served), so `import hgym` works without the
# `tau2` extra installed.
# `hgym.envs.hle.env_v1` imports nothing from the `hle` extra (datasets/openai) at top level
# — the dataset loads lazily on env construction and the judge's client on first call — so
# `import hgym` stays offline without the extra installed.
# `hgym.envs.frontier_bench.env_v1` imports nothing Docker-related at top level — the task
# metadata loads from vendored files on env construction and Docker is touched only when an
# episode is served — so `import hgym` stays offline without a Docker daemon.
from hgym.envs.frontier_bench import env_v1 as frontier_bench_env_v1  # noqa: F401 — registration
from hgym.envs.hle import env_v1 as hle_env_v1  # noqa: F401 — triggers registration
from hgym.envs.tau2 import env_v1 as tau2_env_v1  # noqa: F401 — triggers registration
from hgym.envs.wordle import env_v1 as wordle_env_v1  # noqa: F401 — triggers registration
from hgym.envs.yc_bench import env_v1 as yc_bench_env_v1  # noqa: F401 — triggers registration

__all__ = [
    "frontier_bench_env_v1",
    "hle_env_v1",
    "make",
    "register",
    "registered_envs",
    "tau2_env_v1",
    "wordle_env_v1",
    "yc_bench_env_v1",
]
