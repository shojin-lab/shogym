# automationbench — Zapier's AutomationBench as an offline hgym env

A faithful, deterministic, **offline** wrap of [AutomationBench](https://github.com/zapier/AutomationBench)
(MIT, © Zapier): an agent gets a natural-language instruction and carries out a realistic
cross-application business workflow over a fully **simulated** world of ~47 SaaS apps; scoring
checks — programmatically, **end-state only** — whether the right data landed in the right
systems. No LLM judge, no live SaaS, no network at eval time.

Like every hgym env this is [env-as-center](https://github.com/anndvision/hgym/wiki/RFC-008-Environment-as-Center-of-Gravity):
the env **describes** a task, **serves** its tools over MCP, and **verifies** a recorded
trajectory with a pure function — no agent loop, model, or prompt inside. Upstream ships its own
loop (a Prime Intellect `verifiers` `StatefulToolEnv`); this port throws that away and reuses only
the three deterministic, `verifiers`-free pieces (the simulated tools + `WorldState` engine, the
typed task defs, and the pure rubric). See the runnable demo:
[`examples/automationbench/claude_code/`](../../../../examples/automationbench/claude_code/README.md).

## Running it

```python
import hgym

env = hgym.make("automationbench", config={"domain": "simple"})  # or "public" (default), "sales", …
spec = env.describe("0")          # instructions (the task request) + tool manifest + horizon
```

Serve it over MCP for a harness (Claude Code, Codex, pi, …) to drive:

```bash
uv run python -m hgym.cli serve automationbench --task 0 --trace hgym_logs/automationbench.jsonl
```

The [`examples/automationbench/claude_code/`](../../../../examples/automationbench/claude_code/README.md)
demo has Claude Code play a served episode end-to-end; hgym scores off the trace. Score an
in-process episode directly:

```python
from hgym.serve import ServedEpisode

episode = await ServedEpisode.start("automationbench", task=0)
# ... harness calls api_search / api_fetch / base64_encode, then done (seals + scores) ...
result = episode.terminal_feedback   # reward (== partial_credit), partial_credit, success
```

## Requirements

- The **`automationbench` extra** (`pip install "hgym[automationbench]"`, or `uv sync` for dev):
  it pulls `datasets` (the domain task loaders build a HuggingFace `Dataset`). Python **3.12**
  (the project pin).
- The pinned upstream source (commit
  [`a321764`](https://github.com/zapier/AutomationBench/commit/a321764ace3cfbe42289e6a13abef2f0f4f56fad))
  is **provisioned at runtime** into `~/.cache/hgym/automationbench/<sha>/` on first construction
  — a one-time network fetch. Set `AUTOMATIONBENCH_SRC` to a local checkout (a dir containing the
  `automationbench` package) to skip the fetch, or `HGYM_CACHE` to relocate the cache. Upstream is
  **not** a pip dependency: it declares `requires-python >=3.13` (hgym is pinned `<3.13` for
  tau2's `audioop`) and pulls the heavy `verifiers` / `anthropic` agent-loop stack — an
  unsatisfiable resolve on 3.12 — so the port provisions and imports only the `verifiers`-free
  pieces it needs.
- **No API key.** Scoring is pure/offline/deterministic; the harness brings the model, and hgym
  drops it entirely.

## How it works

### describe → TaskSpec

`describe(task_id)` returns a `TaskSpec` whose `instructions` are the task's own chat `prompt`
(the workflow request) plus a short guide to the `api` tool surface and the `done` finish
protocol; `tools` is the served manifest; `horizon` is `max_steps + 2` (default `max_steps=50`,
upstream's `--max-steps` default and the "~50 tool-using turns" budget the task prompts advertise
— `+2` keeps the horizon a hair above the budget so a run can still call `done` explicitly).

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
- `done()` — the env's **`score` terminal**. Calling it makes the serve layer validate its (empty)
  args, atomically **seal** the episode, then run the env's `finalize` hook — which scores the
  sealed `WorldState` **server-side** and returns core-owned `TerminalEvidence` carrying just the
  score numbers (never the assertions, target values, or world). Because the seal freezes the world
  before scoring, there is no `done`-then-fix loop to defend against — the old one-shot guard is now
  structural. The harness does **not** call `terminate` after `done`.

### finalize + verify (seal-before-verdict)

`done` is the `score` terminal, so scoring runs through the **seal-before-verdict** lifecycle. On a
`done` call (or at the horizon) the serve layer seals the episode, then calls `finalize`, which
scores the now-frozen session `WorldState` with AutomationBench's **own** rubric — `partial_credit`
then `task_completed_correctly` — against the private session world. The free/negative-assertion
"must not shotgun" gate is preserved: a negative assertion passes "free" in the initial world (so
it drops out of the denominator) and only counts as a failure if the agent *breaks* it (e.g. emails
a forbidden recipient). `finalize` returns core-owned `TerminalEvidence` whose verdict is only the
score numbers; the pure `_verify` reads those numbers off the trusted evidence (never the
trajectory) into episode feedback. An explicit `terminate` is a no-score abort (clean zero); the
horizon scores whatever **partial** state the workspace is in.

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

Both come from AutomationBench's own rubric run over the sealed end-state, so difficulty tracks
the pinned upstream `api`-toolset variant.

## Gotchas

- **One toolset variant.** Upstream scores differ across its toolset variants (`api`, `zapier`,
  `limited_zapier`, meta-tools); this port pins the `api` variant and reproduces it (including the
  BM25 top-5 `api_search`). Numbers are comparable to the `api`-toolset leaderboard, not the
  meta-tool ones.
- **First construction fetches the upstream source** (a one-time network hit into
  `~/.cache/hgym`); set `AUTOMATIONBENCH_SRC` for a fully offline / air-gapped run. The offline
  test suite skips these tests if the source can't be provisioned, so the core suite stays green.
- **`done` seals the episode; scoring is protected by the seal.** A `done` call atomically seals
  the episode before `finalize` scores it, so the world can no longer be mutated once scoring
  begins — reading a score and retrying to a better state is impossible by construction (the seal),
  not by a hand-written one-shot guard. `finalize` returns only the score numbers; the assertions,
  targets, and world never leave the server.

## Layout

```
src/hgym/envs/automationbench/
  env_v1.py      # @register("automationbench"): describe / task-load / finalize (seal) + verify (evidence)
  mcp_server.py  # in-process MCP: api_search / api_fetch / base64_encode / done, per-session WorldState + score_session
  adapter.py     # the single seam: provisions + imports upstream, re-hosts setup_state helpers, reuses the rubric
  README.md      # this file
```
