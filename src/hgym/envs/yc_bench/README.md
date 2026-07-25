# YC-Bench — CEO of a simulated AI startup, wrapped faithfully in hgym

A faithful hgym port of [**YC-Bench**](https://github.com/collinear-ai/yc-bench) (Collinear
AI's long-horizon deterministic benchmark). YC-Bench puts an agent in charge of a simulated AI
startup for one year: starting with **$200,000**, it issues `yc-bench` CLI commands against a
deterministic, SQLite-backed discrete-event simulation — accepting tasks from a marketplace,
assigning employees, advancing the clock, and managing cash flow — until **bankruptcy**
(funds < 0) or the **one-year horizon**. The score is how the company ends up.

YC-Bench ships **no agent loop**: it explicitly expects an *external driver* to advance the
sim, feed CLI results back, and collect the next commands. hgym's harness *is* that driver, so
the port is a clean wrap — YC-Bench's sim engine, command execution/validation, SQLite state,
world seeding, and scoring are reused verbatim; only the *agent* is replaced (by the harness,
through the served tools).

## Running it

> Requires **Python 3.12 + the yc_bench extra** — see [Requirements](#requirements). Unlike
> tau2, YC-Bench needs **no data download**: its whole world is generated deterministically
> from the seed and the sim runs in-process, so a served episode is fully offline (no API key).

### Construct + serve

```python
import hgym

env = hgym.make("yc_bench")                              # train split
env_test = hgym.make("yc_bench", config={"task_split": "test"})
spec = env.describe("0")                                  # rules + this run's seed + tools
```

Serve it as a stdio MCP server any harness can drive:

```bash
uv run python -m hgym.cli serve yc_bench --task 0 --trace ./hgym_logs/yc_bench.jsonl
```

The harness runs the year with **`run_command`** (browse the market, accept/assign/dispatch
tasks, `yc-bench sim resume` to advance the clock), calls **`submit`** when the run is over
(its result reports the final funds / survival / task outcomes), then **`terminate`** to end
the hgym episode; hgym reads the verdict off the trace via `hgym.result_from_trace(...)`.

**Config** (via `hgym.make(name, config)` / `env_config`): `task_split` (`"train"`/`"test"`),
`config_name` (YC-Bench preset name or `.toml` path, default `"default"`), `max_commands`
(command budget = the hgym horizon, default 4000), `horizon_years` (default: the preset's
`sim.horizon_years`), `start_date` / `company_name` (seeding params, defaults match
`yc-bench run`), and `command_timeout_seconds`.

### Claude Code example

The runnable end-to-end demo (Claude Code plays a served YC-Bench year; hgym scores off the
trace) lives in [`examples/yc_bench/claude_code/`](../../../../examples/yc_bench/claude_code/):

```bash
# Fully offline sim — no OpenAI/YC-Bench key (the Claude harness still makes model calls):
uv run python examples/yc_bench/claude_code/run.py --task 0
uv run python examples/yc_bench/claude_code/run.py --task 0 --transcript
```

## Requirements

- **Python 3.12.** The project is pinned to 3.12 (`requires-python = ">=3.12,<3.13"`); YC-Bench
  requires `>=3.12` and installs cleanly there.
- **`uv sync`** builds the venv with YC-Bench: the `yc_bench` extra is also listed in the
  default `dev` dependency-group, so `uv sync` / `uv run …` include it without a manual
  `--extra` flag. (`pip install hgym` stays lean; `pip install hgym[yc_bench]` adds it
  explicitly.) The extra is pinned to an upstream **commit SHA** — YC-Bench has no stable
  public API, so a pin makes upstream drift a contained maintenance cost.
- **No data, no API key** for a served episode. YC-Bench generates its world deterministically
  from the seed and runs its sim in-process, so the whole served path (seed → commands →
  verdict) is offline. (A YC-Bench *model* key is only needed by upstream's own agent loop,
  which this port replaces with the harness.)

## How it works

YC-Bench expects an external driver to advance its sim and collect the agent's commands.
hgym's harness is that driver — mapped onto hgym's env-as-center trio. All `yc_bench` imports
are funnelled through a single **adapter** (`adapter.py`), so upstream API drift touches one
file.

### describe → TaskSpec

`env.describe(task_id)` publishes the task contract the harness reads: the **rules** (the
startup-sim objective, the command loop, and the key mechanics — deadlines, payroll growth,
client trust, adversarial clients), this run's **seed / preset**, and the **tool manifest**
(`run_command`, `submit`, plus the reserved `terminate`).

### Tools (served over MCP)

The env's in-process MCP server (`mcp_server.py`) seeds a **private, throwaway SQLite database
per episode** on `begin_session` (one company/world, seeded deterministically from the task's
seed via YC-Bench's own `_init_simulation`, so seeding is bit-identical to `yc-bench run`) and
drops it on `end_session`. It exposes two tools:

- **`run_command(command)`** — the faithful mirror of upstream's `run_command("yc-bench
  <cmd>")`: it validates the command with YC-Bench's own command policy, **allowlists the
  operational sub-command groups** (`company`, `employee`, `market`, `task`, `sim`, `finance`,
  `report`, `scratchpad`, `client`), then runs the real `yc-bench` console-script against *this
  session's* DB (`DATABASE_URL` injected explicitly per call — never via process-global env, so
  concurrent episodes can't race) and returns the CLI's JSON. Every observe / task / sim /
  memory command is reached through this one tool. `yc-bench run` (YC-Bench's own
  credential-inheriting LLM agent loop) and `yc-bench start` (interactive) are **rejected
  before any subprocess is spawned**, keeping the surface offline and trace-attributable.
- **`submit()`** — an hgym terminal tool that reads the authoritative final metrics
  (survival, final funds, task outcomes) straight off the sim DB and returns a **marked**
  verdict. (`terminate` then ends the hgym episode.)

### verify

hgym's pure `_verify` parses that verdict off the recorded terminal **`submit`** step into
episode feedback — defensively: a missing / forged / malformed verdict scores a
non-surviving zero, and **only a `submit` step is trusted** (a marked payload forged onto a
`run_command` result grants no credit, mirroring how wordle trusts only its recorded `guess`
argument and tau2 only its `done` verdict).

---

YC-Bench's sim engine, command execution/validation, SQLite state, world seeding, and scoring
are reused **verbatim** — only the agent is swapped. There are **zero hgym core changes**; the
whole port is additive under `src/hgym/envs/yc_bench/`. `import hgym` registers the `yc_bench`
env **without importing yc_bench**, so the core stays lean and offline; yc-bench is loaded only
when the env is constructed or served.

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

**Funds are credited only from a genuine terminal state.** `submit` is callable at any time,
so the verdict is trusted for reward *only* when the sim actually ended — the one-year horizon
(`terminal_reason == "horizon_end"`) or bankruptcy (`"bankruptcy"`). A solvent, pre-horizon
`submit` (the agent stopping early) is treated as **premature** and scores `reward = 0.0`,
`survived = False`, `success = False` — exactly like a missing verdict — so submitting on turn
one can't bank the starting $200k without operating the company.

**Fidelity check:** for a fixed seed + command sequence, the final funds are reproducible run
to run (the deterministic sim is preserved) and match a direct `yc-bench` seeding of the same
seed — the served/determinism tests assert this.

## Gotchas

- **`sim resume` needs an active task.** Faithful to upstream, `yc-bench sim resume` refuses to
  advance the clock with no active task — the agent must `market browse → task accept → task
  assign → task dispatch` first. The command returns an error payload (`ok: false`); it does
  not raise.
- **Only operational command groups are allowed.** `run_command` allowlists the groups that
  operate the seeded session and rejects `yc-bench run` / `yc-bench start` — `run` would launch
  YC-Bench's own LLM agent loop (inheriting provider credentials, replacing `DATABASE_URL`, and
  doing unbounded external model work outside the trace), which is exactly what this port
  replaces with the harness.
- **Completion is a two-step:** `submit` (reads the sim's final state) then `terminate` (ends
  the hgym episode). The verdict is only ever surfaced on — and trusted from — a `submit` step.
- **Long horizon.** A full year is many `run_command` / `sim resume` turns; the hgym horizon is
  `max_commands + 2` (default 4002 steps). Raise `max_commands` for verbose policies.
- **One company per DB.** YC-Bench stores a single simulation per database, so each episode gets
  its own private SQLite file (seeded on `begin_session`, deleted on `end_session`). Sessions
  never share state.
- **Heavy extra.** YC-Bench depends on litellm / streamlit / matplotlib / plotly (its own
  runner/dashboard). They install with the extra but are never imported by the served path
  (only the sim engine, CLI commands, and ORM are), so command execution stays light.
- **Offline vs keyed tests.** The pure-`verify` unit tests run in the core suite with no extra;
  the served + determinism tests need the `yc_bench` extra but no API key (the sim is
  deterministic and in-process), and are `importorskip`-gated so the core 3.12 suite stays
  green without it.
