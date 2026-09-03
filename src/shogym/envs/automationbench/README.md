# `automationbench` — AutomationBench, an offline cross-app workflow env

[**AutomationBench**](https://github.com/zapier/AutomationBench) (MIT, © Zapier) served through
shogym at upstream commit `6d21054`: an agent gets a natural-language instruction and carries out
a realistic cross-application business workflow over a fully **simulated** world of ~47 SaaS
apps; scoring checks — programmatically, **end-state only** — whether the right data landed in
the right systems. No LLM judge, no live SaaS, no network at eval time. The port is offline and
keyless and the rubric is deterministic over a given end-state, but a run is not reproducible
step for step: upstream mints `uuid4` ids (e.g. Google Drive create/copy) and reads
`datetime.now()`, so the same tool sequence can land different ids and timestamps.

Like every shogym env this **describes** a task, **serves** its tools over MCP, and **verifies**
a recorded trajectory while an external harness drives the tools — see
[`../README.md`](../README.md). Upstream ships its own loop (a Prime Intellect `verifiers`
`StatefulToolEnv`); this port throws that away and reuses only the three offline,
`verifiers`-free pieces (the simulated tools + `WorldState` engine, the typed task defs, and
the pure rubric). The runnable demo is
[`examples/`](../../../../examples/).

## Running it

> Requires **Python 3.12 + the `automationbench` extra**; the upstream source is provisioned at
> runtime on first construction (a one-time network fetch, or point at a local checkout). Fully
> offline at eval time, and the rubric is deterministic over the sealed end-state — no API key.
> See [Requirements](#requirements).

### Construct + serve

```python
import shogym

env = shogym.make("automationbench", config={"domain": "simple"})  # or "public" (default), "sales", …
spec = env.describe("0")          # instructions (the task request) + tool manifest + horizon
```

Serve it over MCP for a harness (Claude Code, Codex, pi, …) to drive:

```bash
uv run python -m shogym.cli serve automationbench --task 0 --trace shogym_logs/automationbench.jsonl
```

The harness discovers and calls endpoints with **`api_search`** / **`api_fetch`** (plus the
`base64_encode` helper), then calls **`done`** — the env's **score terminal**: it seals the
episode, scores the frozen `WorldState` in `finalize`, and ends the episode in one call (no
separate `terminate`). See the shared
[terminal lifecycle](../README.md#terminal-lifecycle-seal-terminal-score-terminal-abort). shogym
reads the verdict off the trace via `shogym.result_from_trace(...)`.

**Config** (via `shogym.make(name, config)` / `env_config`): `domain` (a domain name — `sales` /
`marketing` / `operations` / `support` / `finance` / `hr` / `simple` — or the `public` alias,
the default, expanding to the six public domains = the 600 distributed tasks), `tasks` (an
explicit list of raw upstream task rows — bypasses the domain loaders, used by the offline
tests), and `max_steps` (the *advertised* tool-call budget, default 50 — upstream's
`--max-steps` default and the "~50 tool-using turns" the task prompts quote). It is not an
enforced cap on ordinary calls: the shogym horizon is `max_steps + 2` and `Episode.call`
dispatches the ordinary call that reaches the horizon before sealing it as the terminal step, so
`max_steps + 2` ordinary calls can execute. A `done` issued while the episode is still open is
handled before the horizon is consulted and so is never preempted; the `+2` reserve is what
keeps it reachable after the advertised budget, since a `done` sent after the horizon has sealed
is tombstoned.

### Quickstart

Any quickstart under [`examples/`](../../../../examples/) serves this env: one MCP endpoint
hands out a queue of tasks and scores each one server-side. Point it here with the single
variable at the top of its `serve.py`:

```python
ENV = "automationbench"
```

Score an in-process episode directly:

```python
from shogym.serve import ServedEpisode

episode = await ServedEpisode.start("automationbench", task=0)
# ... harness calls api_search / api_fetch / base64_encode, then done (seals + scores) ...
result = episode.terminal_feedback   # reward (== partial_credit), partial_credit, success
```

## Requirements

The Python pin and the `uv sync` / `pip install` / `import shogym` mechanics are the shared
[requirements boilerplate](../README.md#requirements-boilerplate). The `automationbench` extra
pulls `datasets` (the domain task loaders build a HuggingFace `Dataset`). On top of that:

- **The pinned upstream source is provisioned at runtime** into `~/.cache/shogym/automationbench/<sha>/`
  on first construction — a one-time network fetch. Set `AUTOMATIONBENCH_SRC` to a local checkout
  (a dir containing the `automationbench` package) to skip the fetch, or `SHOGYM_CACHE` to relocate
  the cache. Upstream is **not** a pip dependency: it declares `requires-python >=3.13` (shogym is
  pinned `<3.13` for tau2's `audioop`) and pulls the heavy `verifiers` / `anthropic` agent-loop
  stack — an unsatisfiable resolve on 3.12 — so the port provisions and imports only the
  `verifiers`-free pieces it needs.
- **No API key.** Scoring is pure / offline / deterministic; the harness brings the model, and
  shogym drops it entirely.

## How it works

### describe → TaskSpec

`describe(task_id)` returns a `TaskSpec` whose `instructions` are the task's own chat `prompt`
(the workflow request) plus a short guide to the `api` tool surface and the `done` finish
protocol; `tools` is the served manifest; `horizon` is `max_steps + 2` (see [Config](#running-it)).

### Tools (served over MCP)

The port pins **one** upstream toolset variant — the **`api`** toolset (the simplest), served
in-process against a per-session `WorldState`:

- `api_search(query, top_k=5)` — BM25 search over the upstream endpoint schemas (top-5 by
  default) to discover an endpoint's URL, method, params, and body shape.
- `api_fetch(method, url, params, body)` — route a REST call into this session's `WorldState`,
  mimicking the real SaaS API response. **All** state mutation happens here, through upstream's
  own router; the `allowed_services` gate (a task's world is "subscribed" only to services it
  seeds / asserts on / grants a tool for) is preserved, so a call to an out-of-scope service
  fails like an unconnected account (a 401) instead of silently mutating untracked state.
- `base64_encode(text)` — the Gmail body encoding helper; a local no-API utility.
- `done()` — the env's **score terminal**. Calling it makes the serve layer validate its (empty)
  args, atomically **seal** the episode, then run the env's `finalize` hook — which scores the
  sealed `WorldState` **server-side** and returns core-owned `TerminalEvidence` carrying just the
  score numbers (never the assertions, target values, or world). Because the seal freezes the world
  before scoring, there is no `done`-then-fix loop to defend against. The harness does **not** call
  `terminate` after `done`.

### finalize + verify

`done` is the `score` terminal. On a `done` call (or at the horizon) the serve layer seals the
episode, then calls `finalize`, which scores the now-frozen session `WorldState` with
AutomationBench's **own** rubric — `partial_credit` then `task_completed_correctly` — against the
private session world. The free/negative-assertion "must not shotgun" gate is preserved: a
negative assertion passes "free" in the initial world (so it drops out of the denominator) and
only counts as a failure if the agent *breaks* it (e.g. emails a forbidden recipient). `finalize`
returns core-owned `TerminalEvidence` whose verdict is only the score numbers; the pure `_verify`
reads those numbers off the trusted evidence (never the trajectory) into episode feedback. An
explicit `terminate` is a no-score abort (clean zero); the horizon scores whatever **partial**
state the workspace is in.

Under the durable stream the horizon does the same thing, and this env says so: it declares its
horizon a **graded ending**. The call that spends the last of the budget is dispatched and
answered, the server then files `done` for the attempt, and the attempt is sealed and graded on
the partial state. The acknowledgement comes back in that call's own result and the score reaches
the agent as the next payload; the record says the horizon filed rather than the agent.

## Tasks

The public benchmark ships **600 tasks** across 6 domains (`sales`, `marketing`, `operations`,
`support`, `finance`, `hr`) plus a `simple` domain; the private leaderboard set is not
distributed. Select tasks with `config={"domain": ...}`:

- a single domain name, or the **`public`** alias (default) that expands to the six public
  domains — the 600 distributed tasks;
- `tasks=[...]` injects explicit upstream task rows (used by the offline tests to build the env
  without the domain loaders).

Each task carries a `prompt` (the request), a `zapier_tools` allowlist, an `initial_state`
`WorldState` seed, and typed `assertions`. Per-domain **noise** (distractor records) is injected
deterministically, seeded by each task's `example_id`, so a domain loads identically every time.

## Scoring

Two episode-feedback signals flow off the terminal evidence `finalize` produces:

- `partial_credit` (also surfaced as `reward`) — the fraction of scored assertions satisfied
  (0.0–1.0), the dense training signal;
- `success` — `task_completed_correctly`: `True` iff **every** scored assertion passed (the
  strict benchmark pass/fail).

Both come from AutomationBench's own rubric run over the sealed end-state. Read the score back
off the trace:

```python
import shogym
result = shogym.result_from_trace("shogym_logs/automationbench.jsonl", env="automationbench", task="0")
print(result.terminated, result.value("success"), result.value("partial_credit"))
```

`result_from_trace` treats `env` / `task` / `session_id` as **filters** — see
[Reading a score back](../README.md#reading-a-score-back-result_from_trace) for the shared
semantics (give each run its own trace file for a guaranteed 1:1 mapping).

## Fidelity & deviations

- **One toolset variant.** Upstream offers three (`scripts/eval.py` `--toolset`): `api`,
  `zapier` and `limited_zapier`, where `zapier` *is* the meta-tool discovery mode
  (`runner.py` sets `use_meta_tools = toolset == "zapier"`). This port pins **`api`** and
  reproduces it, including the BM25 top-5 `api_search`. Upstream publishes a single table and
  scores its official leaderboard on a held-out private set, so treat a number from here as an
  `api`-toolset result rather than as comparable to any published per-variant figure.
- **Pinned upstream source.** The port provisions and imports the pinned upstream commit
  [`6d21054`](https://github.com/zapier/AutomationBench/commit/6d210543b7a046f0f451c828cd2dadef774276eb),
  reusing only the `verifiers`-free pieces (simulated tools + `WorldState`, typed task defs, the
  pure rubric); upstream's `verifiers` / `anthropic` agent loop is discarded. The pin is the
  default runtime provisioner's; `AUTOMATIONBENCH_SRC` points at a local checkout whose revision
  nothing checks, so that documented override is a trusted, unversioned path.
- **End-state-only scoring, verbatim rubric.** Scoring is AutomationBench's own rubric run over
  the sealed end-state — no LLM judge, no live SaaS. The free/negative-assertion gate is preserved.
- **Public set only.** The 600 public-domain tasks are distributed; the private leaderboard set
  is not, so it is out of scope here.

## Gotchas

- **First construction fetches the upstream source** (a one-time network hit into
  `~/.cache/shogym`); set `AUTOMATIONBENCH_SRC` for a fully offline / air-gapped run.
- **`done` seals the episode; scoring is protected by the seal.** A `done` call atomically seals
  the episode before `finalize` scores it, so the world can no longer be mutated once scoring
  begins — reading a score and retrying to a better state is impossible by construction (the seal),
  not by a hand-written one-shot guard. `finalize` returns only the score numbers; the assertions,
  targets, and world never leave the server.
- **Offline vs keyed tests.** Per the shared
  [offline-vs-keyed split](../README.md#tests-offline-vs-keyed), automationbench has **no keyed
  path** — scoring is deterministic and offline. The suite skips the served tests if the upstream
  source can't be provisioned, so the core suite stays green.

## Layout

```
src/shogym/envs/automationbench/
  env_v1.py      # @register("automationbench"): describe / task-load / finalize (seal) + verify (evidence)
  mcp_server.py  # in-process MCP: api_search / api_fetch / base64_encode / done, per-session WorldState + score_session
  adapter.py     # the single seam: provisions + imports upstream, re-hosts setup_state helpers, reuses the rubric
```
