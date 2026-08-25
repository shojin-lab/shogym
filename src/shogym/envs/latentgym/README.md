# `latentgym`: LatentGym, one episode per task

A shogym port of [**LatentGym**](https://github.com/namkoong-lab/LatentGym) (Namkoong Lab,
arXiv 2606.15306): seven small text games, each played as a *run* of episodes drawn under one
hidden rule (a **latent**) that the agent is meant to infer from the early episodes and exploit
in the later ones. A **family** is one `(core env, latent)` pair.

Like every shogym env this **describes** a task, **serves** its tools over MCP, and **verifies**
a recorded trajectory while an external harness drives the tools. See
[`../README.md`](../README.md). What is specific here is the mapping and the feedback channel:

- **A task is one episode of a family**, addressed by its position. A queue of `[0, 1, 2, …]` is
  the trajectory a LatentGym run would have played, dispensed one task at a time, with the
  harness as the loop instead of an inline agent driver. Positions are stable and repeats are
  legal: task 3 is the same starting world every time it is asked for, so one task can be served
  twice to two agents and the two compared.
- **The end-of-episode write-up is feedback, not observation.** Upstream concatenates it onto the
  last turn's tool output, so every agent reads it whatever regime the run claims to be serving.
  Here it is published as episode feedback, and a [feedback
  policy](#the-two-channels-information-and-placebo) decides whether it reaches the agent.

`latentgym` is a **score-terminal** env: `done` seals the episode and only then scores it (see the
shared [terminal lifecycle](../README.md#terminal-lifecycle-seal-terminal-score-terminal-abort)).

## Running it

> Requires **Python 3.12 + the `latentgym` extra**. No key and no dataset download: the pinned
> upstream source is fetched once into `~/.cache/shogym/` on the first construction and is fully
> offline thereafter.

### Construct + serve

```python
import shogym

env = shogym.make("latentgym")                            # bandits / loyal_favorite_0, 32 episodes
env = shogym.make("latentgym", config={
    "core_env": "bandits",       # bandits | secretary | number_guessing | wordladder
    "latent": "cold_hand",       # a generator latent of that core env
    "prompt": "some_info",       # no_info | some_info | full_info
    "seed": 7,
    "length": 32,                # episodes in the family == tasks it offers
})
spec = env.describe("3")                                  # episode 3: framing, opening, tools
```

Serve it as a stdio MCP server any harness can spawn:

```bash
uv run python -m shogym.cli serve latentgym --task 0 --trace ./shogym_logs/latentgym.jsonl
```

**Drive it with Claude Code** via the quickstart at
[`examples/claude_code/`](../../../../examples/claude_code/), with two variables set:

```bash
SHOGYM_ENV=latentgym SHOGYM_TASKS=0,1,2,3 claude -p "$(cat PROMPT.txt)" \
    --mcp-config .mcp.json --strict-mcp-config \
    --allowedTools 'mcp__shogym__*' --permission-mode dontAsk
```

The queue order is the run order: serve `0,1,2,3` and the agent plays the family in sequence,
which is the only order in which the latent is learnable.

### The tools

- **`act(action: str)`**: one turn. The action goes in square brackets exactly as the game's own
  rules describe (`[red]`, `[select red]`, `[accept]`, `[continue]`, `[500]`, `[cart]`); upstream's
  own parsers read it, so nothing about the action grammar is this port's invention. Answers with
  `observation` (what the game said back), `turn`, and `episode_over`.
- **`done()`**: the **score terminal**. Call it once `act` reports the episode is over. It seals
  the episode, scores it, and ends the task. Calling it early records a task with no outcome
  rather than a zero the agent played for.
- **`terminate()`**: the reserved abort every env serves. Not needed after `done`.

The horizon is the game's own turn budget plus two, so a run that means to end its task explicitly
can always reach `done`. Reaching the horizon instead finalizes on whatever the game recorded.

## The two channels: `Information` and `Placebo`

Every ended episode publishes five episode-level feedback items:

| name | what it is |
|---|---|
| `reward` | the core env's own reward, read from what its `step` returned when the game ended |
| `success` | the core env's own boolean verdict where it publishes one, else full credit |
| `turns` | turns the episode took, on this benchmark the *more* sensitive readout (see below) |
| `report` | upstream's `information` write-up: the verdict **and the ground truth**, win or lose |
| `notice` | an inert, length-matched stand-in for `report`, carrying no evaluative content |

The last two are a matched pair, and they are what the [feedback
policies](../../serve/stream.py) `Information` and `Placebo` select between:

```python
from shogym.serve import Information, Placebo, TaskStream

TaskStream(shogym.make, tasks, prov_dir=..., feedback=Information())   # the treatment arm
TaskStream(shogym.make, tasks, prov_dir=..., feedback=Placebo())       # the matched control
```

`Information()` answers a terminating call with the `report` item and nothing else; `Placebo()`
answers with the `notice` item and nothing else. Both names are six characters and the notice is
built to the report's exact length *as the terminal answer encodes it*. Upstream writes its
reports with a dash character the encoder expands to six characters, so matching the character
count would leave every bandits notice five characters short. The two arms therefore differ in the
*content* of the channel and in nothing else: not its size, not its shape, not whether it is there
at all.

`Never()` (the default) opens no channel, and `Immediate()` hands back every item above, including
the ground truth and the numbers.

Both items are published on every episode whatever regime the run is serving, because the env does
not know the regime and may not: an env that published only the item its run was going to reveal
would have made its own record depend on the treatment.

## What is worth measuring

Two findings from a pilot on this instrument are worth knowing before picking a family, because
neither is visible from upstream's own metadata:

**On bandits, the bare verdict is not news.** The correct/wrong verdict is emitted by the core env
*inside* the episode, as the observation on the selecting turn, and the reward is
`1.0 - turns * 0.015` on a correct pick and `0` otherwise, so an agent that can count its own
turns already knows its score before any end-of-episode text exists. What the `report` channel adds
there is the identity of the best arm on a miss and the full hidden probability vector always,
which is the only channel that lets the cross-episode latent be inferred from evidence rather than
from a lucky streak. **Secretary is the opposite:** an agent cannot tell whether it accepted the
maximum, because the unseen remainder of the sequence is what decides it, so even a bare verdict is
information there.

**Report turns alongside reward.** An agent that has inferred the latent stops exploring, and that
shows up in `turns` well before it shows up in a reward already near its ceiling. A family can
look like a non-learner by score and a complete learner by turn count.

**Do not pick a latent by its declared complexity.** Upstream's `easy`/`hard`/`very_hard` labels
were close to anti-correlated with measured difficulty in both bandits and secretary. Pick by
measured behaviour.

## Fidelity & deviations

Pinned to upstream commit `4f5c797` (`adapter.UPSTREAM_SHA`). LatentGym publishes no distribution,
so the port provisions the pinned source at runtime. See
[`../_upstream.py`](../_upstream.py) and the `latentgym` extra in
[`pyproject.toml`](../../../../pyproject.toml).

**Reused verbatim:** the core envs (game dynamics, action parsing, rewards, rules text), the
latents, the prompt templates, the feedback handlers, and the episode-configuration resolution
(re-hosted as `adapter._generated_configs`, seeding included, so a family's episodes are the ones a
real run of the same `(env, latent, seed, length)` would have played).

**Not reused:** `MultiEpisodeEnv` and the `make_env` factory around it, which play a whole
trajectory in one process against an inline agent loop. A shogym task is one episode and the
harness is the loop.

Four deviations, each deliberate:

1. **83 of 421 latents, 4 of 7 core envs.** A `filter` latent draws its episodes from a candidate
   pool built by a `latentgym.data` package the public repository does not ship, and upstream
   refuses to construct one without a pool file. That rules out wordle, hangman and mastermind
   entirely and 14 of wordladder's 20 latents. The port refuses such a latent at construction with
   a message naming the ones that do run, rather than failing when the task comes up.
2. **The end-of-episode report is withheld from the observation stream** and published as feedback
   (above). Upstream appends it to the final turn's text.
3. **A stated score that disagrees with the reward is corrected.** Upstream's secretary handlers
   print `Score: 0.000` on every miss while the core env awards `0.5 * accepted / max`, so an agent
   is told it scored nothing on an episode worth 0.4. The port rewrites a `Score:` claim that
   disagrees with the reward the task is actually scored on, in every env. A handler that already
   agrees is left byte-identical. Scores themselves are always read from the core env's own return
   value, never parsed back out of feedback text.
4. **Each episode plays under its own seeded RNG.** Upstream's core envs draw from the global
   `random` module while a game is being played, so the same task served twice would deal the same
   hidden world and then different outcomes out of it. The port installs a per-episode state
   derived from the family and the index around every call into upstream, which is what makes a
   repeat a repeat.

The cross-episode framing is preserved rather than flattened: an episode's instructions carry the
family's own multi-episode framing and name this episode's place in it (`--- Episode 8 of 32 ---`),
because a run whose framing said "episode 1 of 1" thirty-two times would not be this benchmark.
What an agent carries between episodes is the harness's context, which is the thing being measured.

## Requirements

Python 3.12 and the `latentgym` extra (`uv sync` installs it via the default dev group; otherwise
`pip install 'shogym[latentgym]'`). The extra is what the two vendored upstream packages need to
import: TextArena's dependencies (the bandits and secretary core envs import it) and skyrl-gym's
(`latentgym.core`'s own `__init__` imports it).

The pinned source is fetched on the first construction into `~/.cache/shogym/latentgym/<sha>/`,
`~/.cache/shogym/textarena/<sha>/` and `~/.cache/shogym/skyrl_gym/<sha>/`: three fetches of the
one archive, once. To skip the network entirely, point each at an existing checkout:

```bash
export LATENTGYM_SRC=/path/to/LatentGym
export TEXTARENA_SRC=/path/to/LatentGym/TextArena
export SKYRL_GYM_SRC=/path/to/LatentGym/skyrl-gym
```

## Layout

| File | Role |
|---|---|
| `env_v1.py` | `LatentGymEnv` (the registered `latentgym`): task loading, instructions, `finalize`, and the two published channels. |
| `adapter.py` | The single seam to upstream: provisioning, the family, and the re-hosted episode-configuration resolution. |
| `mcp_server.py` | The in-process FastMCP server backing `act` / `done`; holds the live game, its RNG, and the withheld report, keyed by session id. |
