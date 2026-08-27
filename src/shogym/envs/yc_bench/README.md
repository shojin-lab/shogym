# `yc_bench` — YC-Bench, CEO of a simulated AI startup

[**YC-Bench**](https://github.com/collinear-ai/yc-bench) (Collinear AI's long-horizon
deterministic benchmark) served through shogym at upstream commit `e7d6067`. YC-Bench puts an
agent in charge of a simulated AI startup for one year: starting with **$200,000**, it issues
`yc-bench` CLI commands against a
deterministic, SQLite-backed discrete-event simulation — accepting tasks from a marketplace,
assigning employees, advancing the clock, and managing cash flow — until **bankruptcy**
(funds < 0) or the **one-year horizon**. The score is how the company ends up.

Like every shogym env this **describes** a task, **serves** its tools over MCP, and **verifies**
a recorded trajectory while an external harness drives the tools — see
[`../README.md`](../README.md). YC-Bench ships **no agent loop**: it explicitly expects an
*external driver* to advance the sim, feed CLI results back, and collect the next commands.
shogym's harness *is* that driver, so the port reuses YC-Bench's sim engine, command
execution, SQLite state, world seeding and scoring verbatim, and replaces only the *agent* (with
the harness, through the served tools). The runnable demo is
[`examples/`](../../../../examples/).

## Running it

> Requires **Python 3.12 + the `yc_bench` extra** — see [Requirements](#requirements). Unlike
> tau2, YC-Bench needs **no data download**: its whole world is generated deterministically
> from the seed and the sim runs in-process, so a served episode is fully offline (no API key).

### Construct + serve

```python
import shogym

env = shogym.make("yc_bench")                              # train split
env_test = shogym.make("yc_bench", config={"task_split": "test"})
spec = env.describe("0")                                  # rules + this run's seed + tools
```

Serve it as a stdio MCP server any harness can drive:

```bash
uv run python -m shogym.cli serve yc_bench --task 0 --trace ./shogym_logs/yc_bench.jsonl
```

The harness runs the year with **`run_command`** (browse the market, accept/assign/dispatch
tasks, `yc-bench sim resume` to advance the clock), then calls **`submit`** when the run is
over. `submit` is the env's **score terminal**: shogym seals the episode, reads the final funds /
survival / task outcomes straight off the sim in `finalize`, scores it, and ends the episode in
one step — there is no separate stop call (`terminate` remains available as the *abort*
terminal, which ends the episode without crediting anything). See the shared
[terminal lifecycle](../README.md#terminal-lifecycle-seal-terminal-score-terminal-abort). shogym
reads the verdict off the trace via `shogym.result_from_trace(...)`.

**Config** (via `shogym.make(name, config)` / `env_config`): `task_split` (`"train"`/`"test"`),
`config_name` (YC-Bench preset name or `.toml` path, default `"default"`), `max_commands`
(command budget = the shogym horizon, default 4000), `horizon_years` (default: the preset's
`sim.horizon_years`), `start_date` / `company_name` (seeding params, defaults match
`yc-bench run`), and `command_timeout_seconds`.

### Quickstart

Any quickstart under [`examples/`](../../../../examples/) serves this env: one MCP endpoint
hands out a queue of tasks and scores each one server-side. Point it here with the single
variable at the top of its `serve.py`:

```python
ENV = "yc_bench"
```

The sim itself is fully offline and needs no OpenAI or YC-Bench key; the harness still
makes its own model calls.

## Requirements

The Python pin and the `uv sync` / `pip install` / `import shogym` mechanics are the shared
[requirements boilerplate](../README.md#requirements-boilerplate); the `yc_bench` extra is in
the default `dev` group, so `uv sync` includes it. On top of that:

- **The pinned upstream source is provisioned at runtime** into
  `~/.cache/shogym/yc_bench/<sha>/` on first construction — a one-time network fetch (~4 MB
  downloaded, ~700 KB kept). Set `YC_BENCH_SRC` to a local checkout (a dir containing the
  `yc_bench` package) to skip the fetch, or `SHOGYM_CACHE` to relocate the cache. Upstream is
  **not** a pip dependency: PyPI forbids direct (`@ git+…`) references, and the `yc-bench`
  release on PyPI is not the pinned commit — so the port fetches and imports the pinned source
  instead, exactly as `automationbench` does. The `yc_bench` extra declares upstream's own
  runtime dependencies explicitly, since pip no longer resolves them transitively.
- **No data, no API key** for a served episode. YC-Bench generates its world deterministically
  from the seed and runs its sim in-process, so once the source is cached the whole served path
  (seed → commands → verdict) is offline. (A YC-Bench *model* key is only needed by upstream's
  own agent loop, which this port replaces with the harness.)
- **Heavy extra.** YC-Bench depends on litellm / streamlit / matplotlib / plotly (its own
  runner/dashboard). They install with the extra but are never imported by the served path
  (only the sim engine, CLI commands, and ORM are), so command execution stays light.

## How it works

YC-Bench expects an external driver to advance its sim and collect the agent's commands.
shogym's harness is that driver — mapped onto shogym's env-as-center trio. All `yc_bench` imports
are funnelled through a single **adapter** (`adapter.py`), so upstream API drift touches one
file.

### describe → TaskSpec

`env.describe(task_id)` publishes the task contract the harness reads: the **rules** (the
startup-sim objective, the command loop, and the key mechanics — deadlines, payroll growth,
client trust, adversarial clients), this run's **seed / preset**, and the **tool manifest**
(`run_command` (`terminal_kind: none`), `submit` (`terminal_kind: score`), plus the reserved
`terminate` (`terminal_kind: abort`)).

### Tools (served over MCP)

The env's in-process MCP server (`mcp_server.py`) seeds a **private, throwaway SQLite database
per episode** on `begin_session` (one company/world, seeded deterministically from the task's
seed via YC-Bench's own `_init_simulation`, so seeding is bit-identical to `yc-bench run`) and
drops it on `end_session`. It exposes two tools:

- **`run_command(command)`** — mirrors upstream's `run_command("yc-bench
  <cmd>")`: it validates the command with YC-Bench's own command policy, **allowlists the
  operational sub-command groups** (`company`, `employee`, `market`, `task`, `sim`, `finance`,
  `report`, `scratchpad`, `client`), then runs YC-Bench's real CLI entry point
  (`python -P -m yc_bench`, i.e. the same `yc_bench.cli:app_main` its console script points at,
  with the provisioned source on the subprocess's `PYTHONPATH` — `-P` keeps the working
  directory off `sys.path`, where it would otherwise sit *ahead* of it) against *this
  session's* DB (`DATABASE_URL` injected explicitly per call — never via process-global env, so
  concurrent episodes can't race) and returns the CLI's JSON. Every observe / task / sim /
  memory command is reached through this one tool. `yc-bench run` (YC-Bench's own
  credential-inheriting LLM agent loop) and `yc-bench start` (interactive) are **rejected
  before any subprocess is spawned**, keeping the surface offline and trace-attributable.
- **`submit()`** — the env's **score terminal**. Its call is not an ordinary step: the serve
  layer validates it, **atomically seals** the episode, then runs the env's `finalize` hook,
  which reads the authoritative final metrics (survival, final funds, task outcomes) straight
  off the **live** sim DB and returns them as core-owned `TerminalEvidence`. Because the read
  happens on a frozen, un-continuable episode and the verdict is stamped by the core (never
  surfaced as forgeable tool output), the terminal score can't be gamed by inspecting a verdict
  and issuing more commands.

### finalize + verify

`finalize` runs on the already-sealed episode. It reads the sim's final state through the
session while the SQLite engine is still live — the serve layer disposes the session only
*after* `finalize` returns — and returns the verdict as `TerminalEvidence`. shogym's pure
`_verify` then scores from `evidence.verdict` (never from tool output), defensively: a missing
/ non-terminal verdict scores a non-surviving zero. The one-shot / "trust only the `submit`
step" guard is **structural** — a `score` terminal seals on its first call, so there is no
second submission to guard against and no trajectory to scan for a forged marker.

## Tasks

A task **is a world seed**. The seed selects the market tasks generated for the world (upstream
fixes employees/clients across seeds), so a seed fully reproduces an instance. `yc_bench` ships
two disjoint seed banks — `train` (seeds 1–16) and `test` (seeds 9001–9016) — selected by
`task_split` (default `train`). Task indices are relative to the chosen split.

## Scoring

Scoring is read from YC-Bench's own final sim state (the same rows `company status` reports),
surfaced by `_verify` as episode feedback:

- **`reward`** / **`final_funds_cents`** — the company's final funds in cents (the benchmark's
  objective; can be negative on bankruptcy).
- **`survived`** — `True` iff final funds ≥ 0 (not bankrupt).
- **`horizon_reached`** — did the sim reach the one-year horizon.
- **`success`** — `survived and horizon_reached` (finished the year solvent).
- **`tasks_succeeded`** / **`tasks_failed`** — completed-task outcome counts.

**Funds are credited only from a genuine terminal state.** `submit` can seal the episode at any
time, so the verdict is credited for reward *only* when the sim actually ended — the one-year
horizon (`terminal_reason == "horizon_end"`) or bankruptcy (`"bankruptcy"`). A solvent,
pre-horizon `submit` (the agent stopping early) is treated as **premature** and scores
`reward = 0.0`, `survived = False`, `success = False` — exactly like a missing verdict — so
submitting on turn one can't bank the starting $200k without operating the company. This
terminal-state gate lives in the scorer; the horizon path finalizes the same way (it credits
the end-state only if the sim's own `terminal_reason` says it genuinely ended).

Read the score back off the trace:

```python
import shogym
result = shogym.result_from_trace("shogym_logs/yc_bench.jsonl", env="yc_bench", task="0")
print(result.terminated, result.value("success"), result.value("final_funds_cents"))
```

`result_from_trace` treats `env` / `task` / `session_id` as **filters** — see
[Reading a score back](../README.md#reading-a-score-back-result_from_trace) for the shared
semantics (give each run its own trace file for a guaranteed 1:1 mapping).

## Fidelity & deviations

- **Verbatim reuse.** YC-Bench's sim engine, command execution/validation, SQLite state, world
  seeding, and scoring are reused **verbatim** — only the agent is swapped for the harness.
  There are **zero shogym core changes**; the whole port is additive under
  `src/shogym/envs/yc_bench/`.
- **Pinned to a commit SHA.** The extra is pinned to an upstream **commit SHA** — YC-Bench has
  no stable public API, so a pin makes upstream drift a contained maintenance cost.
- **`run` / `start` rejected.** `run_command` allowlists only the operational sub-command
  groups; `yc-bench run` (upstream's own credential-inheriting LLM agent loop) and `yc-bench
  start` (interactive) are rejected before any subprocess spawns — that agent loop is exactly
  what this port replaces with the harness.
- **Determinism check.** For a fixed seed + command sequence the final funds are reproducible
  run to run and match a direct `yc-bench` seeding of the same seed — the served/determinism
  tests assert this.

## Gotchas

- **`sim resume` needs an active task.** Matching upstream, `yc-bench sim resume` refuses to
  advance the clock with no active task — the agent must `market browse → task accept → task
  assign → task dispatch` first. The command returns an error payload (`ok: false`); it does
  not raise.
- **Completion is a single step:** `submit` is the score terminal — it seals the episode,
  reads the sim's final state, scores it, and ends the run in one call. No separate `terminate`
  is needed (that stays available only as the *abort* terminal, which scores nothing).
- **Long horizon.** A full year is many `run_command` / `sim resume` turns; the shogym horizon is
  `max_commands + 1` (default 4001 steps) — `max_commands` non-terminal commands plus one
  reserved slot so the terminal `submit` is never preempted by the horizon. Raise `max_commands`
  for verbose policies.
- **One company per DB.** YC-Bench stores a single simulation per database, so each episode gets
  its own private SQLite file (seeded on `begin_session`, deleted on `end_session`). Sessions
  never share state.
- **Offline vs keyed tests.** Follows the shared
  [offline-vs-keyed split](../README.md#tests-offline-vs-keyed): the pure-`verify` unit tests run
  in the core suite with no extra; the served + determinism tests need the `yc_bench` extra but
  no API key (the sim is deterministic and in-process), and are `importorskip`-gated so the core
  3.12 suite stays green without it.

## Layout

A source map for orientation:

| File | Role |
|---|---|
| `env_v1.py` | The registered `yc_bench` env: `describe` (rules + seed + manifest), the train/test seed banks, the `finalize` hook (reads final sim metrics on the sealed episode), and the pure `_verify` scorer (verdict from evidence, with the genuine-terminal-state gate). |
| `mcp_server.py` | The in-process MCP server: `run_command` (allowlisted `yc-bench` CLI against a per-session SQLite DB) + `submit` (the score terminal), seeding/dropping the private DB per session. |
| `adapter.py` | The single seam funnelling all `yc_bench` imports (sim engine, CLI, ORM, `_init_simulation` seeding), so upstream API drift touches one file. |
</content>
