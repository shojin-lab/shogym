from shogym.envs.registration import make, register, registered_envs

# Importing an env's `env_v1` module runs its `@register(...)` decorators. These imports
# must stay offline: neither pulls in heavy/optional deps at module load. In particular
# `shogym.envs.tau2.env_v1` imports nothing from `tau2` at top level (tau2 is loaded lazily,
# only when a tau2 env is constructed or served), so `import shogym` works without the
# `tau2` extra installed.
# `shogym.envs.hle.env_v1` imports nothing from the `hle` extra (datasets/openai) at top level
# — the dataset loads lazily on env construction and the judge's client on first call — so
# `import shogym` stays offline without the extra installed.
# `shogym.envs.automationbench.env_v1` imports nothing from the vendored `automationbench` package
# at top level — the package (and `datasets`) load lazily on env construction/serve — so
# `import shogym` stays offline without the `automationbench` extra installed.
# `shogym.envs.appworld.env_v1` imports nothing from the `appworld` extra at top level. The app
# sources the wheel ships packed, and the pinned data bundle, are provisioned only when the env is
# *constructed*, so `import shogym` stays offline and needs neither the extra nor the corpus.
from shogym.envs.appworld import env_v1 as appworld_env_v1  # noqa: F401 — triggers registration
from shogym.envs.automationbench import (  # noqa: F401 — triggers registration
    env_v1 as automationbench_env_v1,
)

# `shogym.envs.browsecomp_plus.env_v1` likewise imports nothing from the `browsecomp_plus` extra
# (datasets/openai/pyserini) at top level — the encrypted queries decrypt in memory on env
# construction, the BM25 index and judge client load lazily — so `import shogym` stays offline.
from shogym.envs.browsecomp_plus import env_v1 as browsecomp_plus_env_v1  # noqa: F401 — triggers registration

# `shogym.envs.frontier_bench.env_v1` imports nothing Docker-related at top level — the task
# metadata loads from vendored files on env construction and Docker is touched only when an
# episode is served — so `import shogym` stays offline without a Docker daemon.
from shogym.envs.frontier_bench import env_v1 as frontier_bench_env_v1  # noqa: F401 — registration
from shogym.envs.hle import env_v1 as hle_env_v1  # noqa: F401 — triggers registration
from shogym.envs.tau2 import env_v1 as tau2_env_v1  # noqa: F401 — triggers registration
from shogym.envs.wordle import env_v1 as wordle_env_v1  # noqa: F401 — triggers registration
from shogym.envs.yc_bench import env_v1 as yc_bench_env_v1  # noqa: F401 — triggers registration

__all__ = [
    "appworld_env_v1",
    "automationbench_env_v1",
    "browsecomp_plus_env_v1",
    "frontier_bench_env_v1",
    "hle_env_v1",
    "make",
    "register",
    "registered_envs",
    "tau2_env_v1",
    "wordle_env_v1",
    "yc_bench_env_v1",
]
