# LatentGym's source (and the two upstream packages it imports) is provisioned at runtime into a
# cache dir (see `ensure_source`); it is intentionally absent from the base type-check / offline
# environment, so its imports are expected to be unresolved there.
# pyright: reportMissingImports=false
"""The single seam between shogym and the upstream ``latentgym`` package.

LatentGym (MIT, © Namkoong Lab) publishes no distribution: it is a research checkout whose
``latentgym`` package sits at the repository root beside two vendored dependencies it imports,
``TextArena/textarena`` and ``skyrl-gym/skyrl_gym``. So this adapter **provisions the pinned
upstream source at runtime** into a gitignored cache, exactly as the tau2 / yc_bench /
automationbench ports do. See :mod:`shogym.envs._upstream` for the mechanics, and the port's
README for the three ``*_SRC`` overrides that skip the fetch. Nothing from upstream is committed
to shogym.

The port reuses upstream's own pieces and adds no game logic of its own:

- ``SingleEpisodeEnv``: one game, driven a turn at a time by
  :mod:`shogym.envs.latentgym.mcp_server`.
- ``LatentDefinition``: the hidden rule that decides each episode's configuration.
- ``PromptTemplate``: the framing an agent reads, including the multi-episode framing that is
  the whole point of the suite.
- ``FeedbackHandler``: how an ending is written up. The ``standard`` handler formats the
  in-episode turns; the ``information`` handler writes the end-of-episode report.

What it does **not** reuse is ``MultiEpisodeEnv`` and the ``make_env`` factory around it, which
play a whole trajectory in one process against an inline agent loop. A shogym task is one episode
and the harness is the loop, so this resolves the trajectory's episode configurations itself and
plays them one at a time. That resolution is the ~10 lines re-hosted verbatim below
(:func:`_generated_configs`), kept identical so a family's episodes are the ones a real LatentGym
run of the same ``(env, latent, seed, length)`` would have played.

Only ``generator`` latents are supported, because only they run: a ``filter`` latent selects its
episodes from a candidate pool built by a ``latentgym.data`` package that the public repository
does not ship, and upstream refuses to construct one without a pool file. That is 83 of the 421
registered latents, across four of the seven registered core envs (``bandits``, ``secretary``,
``number_guessing``, and 6 of ``wordladder``'s 20).

Importing this module triggers provisioning (a one-time network fetch if the cache is cold) and
imports the three upstream packages, so it is only ever imported when a ``latentgym`` env is
*constructed* or *served*, never by ``import shogym``.
"""

from __future__ import annotations

import importlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from shogym.envs._upstream import ensure_package

# Fidelity pin: the upstream commit this port reproduces (LatentGym has no release tags).
UPSTREAM_SHA = "4f5c79749523545434975da87eb441c54fc49d4b"
_TARBALL_URL = f"https://github.com/namkoong-lab/LatentGym/archive/{UPSTREAM_SHA}.tar.gz"

# The three packages the checkout carries, and where each sits inside the archive. `textarena`
# backs the bandits and secretary core envs; `skyrl_gym` is imported by `latentgym.core`'s own
# `__init__`, so it is needed even though this port never touches the trajectory env that uses it.
_PACKAGES: Tuple[Tuple[str, str], ...] = (
    ("latentgym", ""),
    ("textarena", "TextArena"),
    ("skyrl_gym", "skyrl-gym"),
)

# The core envs whose latents are all `generator` mode, so every one of them runs without the
# absent `latentgym.data` pools. Ordered as upstream registers them.
EXECUTABLE_CORE_ENVS: Tuple[str, ...] = (
    "bandits",
    "secretary",
    "number_guessing",
    "wordladder",
)

# The handler that formats each turn *inside* an episode. Both registered handlers pass raw
# step feedback straight through in all four executable core envs, so this choice changes no
# byte of what an agent reads mid-episode. It is named rather than assumed so the in-episode
# channel is regime-free by construction rather than by coincidence.
STEP_FEEDBACK_ID = "standard"

# The handler that writes the end-of-episode report. It is the richer of the two: it names the
# ground truth whether the agent succeeded or not, which is the only end-of-episode channel in
# the suite that tells an agent something its own transcript does not already contain.
REPORT_FEEDBACK_ID = "information"


def ensure_source() -> Path:
    """Ensure the upstream source is available and importable; return LatentGym's own directory.

    Idempotent and thread-safe. Provisions all three packages the checkout carries (see
    :data:`_PACKAGES`) from the one pinned tarball. Each is fetched and cached under its own
    name, so a cold cache downloads the archive once per package."""
    directories = [
        ensure_package(
            package=package,
            sha=UPSTREAM_SHA,
            tarball_url=_TARBALL_URL,
            archive_subdir=subdir,
        )
        for package, subdir in _PACKAGES
    ]
    return directories[0]


ensure_source()

from latentgym.core.feedback import FeedbackHandler  # noqa: E402
from latentgym.core.latent import LatentDefinition  # noqa: E402
from latentgym.core.prompt import PromptTemplate  # noqa: E402
from latentgym.core.registry import (  # noqa: E402
    _ENV_REGISTRY,
    get_feedback_class,
    get_latent,
    get_prompt_class,
    list_latents,
)
from latentgym.core.single_episode_env import SingleEpisodeEnv  # noqa: E402


def _register_core_envs() -> None:
    """Import the four executable core-env packages, which is what registers their latents,
    prompts and feedback handlers.

    Imported one by one rather than through ``latentgym.envs``, whose ``__init__`` also imports
    wordle, hangman and mastermind. Those three register only ``filter`` latents, so nothing they
    add is reachable here, and wordle's core env pulls in a TextArena module this port would
    otherwise never load."""
    for name in EXECUTABLE_CORE_ENVS:
        importlib.import_module(f"latentgym.envs.{name}")


_register_core_envs()


# ----- re-hosted from upstream latentgym/core/registry.py `make_env` (the generator branch) -----


def _generated_configs(
    latent: "LatentDefinition",
    env_params: Dict[str, Any],
    seed: int,
    num_episodes: int,
) -> List[Dict[str, Any]]:
    """Resolve a family's episode configurations. Verbatim from upstream ``make_env``.

    The seeding is the load-bearing part and is reproduced exactly: upstream seeds the **global**
    ``random`` module (not a local generator) because the latent generator functions call the
    module-level API directly, and it restores the caller's state afterwards. A latent is also
    free to carry state forward through ``context``, which is what lets a rule like "the best arm
    depends on the previous episode's" exist at all, so configurations are generated as a whole
    trajectory, in order, and a single episode is an index into that trajectory rather than
    something that can be generated on its own."""
    old_state = random.getstate()
    random.seed(seed)
    try:
        configs: List[Dict[str, Any]] = []
        context: Dict[str, Any] = {}
        for episode_idx in range(num_episodes):
            configs.append(
                latent.generate_episode_config(env_params, episode_idx, num_episodes, context)
            )
    finally:
        random.setstate(old_state)
    return configs


# ----- the family: one (core env, latent, prompt) trajectory, resolved -----


@dataclass(frozen=True)
class Family:
    """One ``(core env, latent, prompt, seed, length)`` trajectory, resolved to its episodes.

    A family is what a LatentGym run *is*: a sequence of episodes drawn under one hidden rule,
    where the point is to infer the rule from the early ones and exploit it in the later ones.
    This holds the whole sequence, so an episode is addressed by its position in it: index 3 is
    the same starting world every time it is asked for, and asking twice is legal."""

    core_env: str
    latent: str
    prompt: str
    seed: int
    configs: Tuple[Dict[str, Any], ...]
    env_params: Dict[str, Any]

    @property
    def length(self) -> int:
        return len(self.configs)

    @property
    def max_turns(self) -> int:
        """The longest any one episode of this family may run.

        Upstream's registered default, unless a latent wrote a budget into an episode's own
        configuration, which several do. The longest of them is what a per-episode step budget has
        to cover, so it is the family's answer rather than the first episode's."""
        default = int(self.env_params.get("max_turns_per_episode", 0) or 0)
        return max(
            [
                default,
                *(int(cfg.get("max_turns_per_episode", default) or 0) for cfg in self.configs),
            ]
        )

    def config(self, episode_idx: int) -> Dict[str, Any]:
        """The episode at ``episode_idx``, as its own copy: a core env's ``reset`` is free to keep
        a reference to what it is handed, and the family outlives the episode."""
        return dict(self.configs[episode_idx])

    def new_core_env(self) -> "SingleEpisodeEnv":
        """A fresh core env for one episode. Fresh per episode because upstream's core envs hold
        the whole game in instance state and reuse it across ``reset`` calls."""
        return _ENV_REGISTRY[self.core_env].env_class()

    def prompt_template(self) -> "PromptTemplate":
        return get_prompt_class(self.core_env, self.prompt)()

    def step_handler(self) -> "FeedbackHandler":
        return get_feedback_class(self.core_env, STEP_FEEDBACK_ID)()

    def report_handler(self) -> "FeedbackHandler":
        return get_feedback_class(self.core_env, REPORT_FEEDBACK_ID)()


def load_family(
    *,
    core_env: str,
    latent: str,
    prompt: str,
    seed: int,
    length: int,
) -> Family:
    """Resolve a family, refusing anything that cannot be played rather than failing at serve time.

    Every name is checked against upstream's registries here, at env construction, so a typo is an
    error before a queue is built rather than a stopped stream three tasks in. A ``filter`` latent
    is refused for the reason its own error gives: it selects its episodes out of a candidate pool
    that the public repository does not ship."""
    if core_env not in _ENV_REGISTRY:
        raise ValueError(
            f"unknown LatentGym core env {core_env!r}; registered: {sorted(_ENV_REGISTRY)}"
        )
    if length < 1:
        raise ValueError(f"a family needs at least one episode, got length={length}")
    definition = get_latent(core_env, latent)
    if definition.latent_mode != "generator":
        raise ValueError(
            f"latent {latent!r} on {core_env!r} is {definition.latent_mode!r} mode, which draws "
            "its episodes from a candidate pool built by the `latentgym.data` package the public "
            f"repository does not ship. Generator latents for {core_env!r}: "
            f"{generator_latents(core_env)}"
        )
    # `latent_id` is read back out of `env_params` by the full-info prompt templates to pick their
    # hint, so upstream writes it in beside the registered defaults. Same here, same place.
    env_params = {**_ENV_REGISTRY[core_env].env_params, "latent_id": latent}
    # Resolved and discarded, for the refusal alone: a bad prompt id has to raise here rather
    # than at the first `describe`, where it would stop a run instead of a construction.
    get_prompt_class(core_env, prompt)
    configs = _generated_configs(definition, env_params, seed, length)
    return Family(
        core_env=core_env,
        latent=latent,
        prompt=prompt,
        seed=seed,
        configs=tuple(configs),
        env_params=env_params,
    )


def generator_latents(core_env: str) -> List[str]:
    """The latents of ``core_env`` that run without a candidate pool, in registration order."""
    return [lat.id for lat in list_latents(core_env) if lat.latent_mode == "generator"]


# ----- the RNG a served episode plays under -----


def episode_seed(family: Family, episode_idx: int) -> str:
    """A stable seed for one episode of one family.

    Upstream's core envs draw from the **global** ``random`` module while a game is being played (a
    bandit's Bernoulli pulls, a secretary's discarded default draws), so an episode's realized
    outcomes are a function of that module's state and nothing else. Left alone, the same task
    served twice would deal the same hidden world and then different pulls out of it, which is not
    the same task. Seeding per episode from the family and the index makes a repeat a repeat: the
    same actions in the same order produce the same game, whoever plays it."""
    return f"latentgym|{family.core_env}|{family.latent}|{family.seed}|{episode_idx}"


__all__ = [
    "EXECUTABLE_CORE_ENVS",
    "Family",
    "REPORT_FEEDBACK_ID",
    "STEP_FEEDBACK_ID",
    "UPSTREAM_SHA",
    "ensure_source",
    "episode_seed",
    "generator_latents",
    "load_family",
]
