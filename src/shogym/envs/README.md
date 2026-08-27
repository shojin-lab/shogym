# shogym environments

This directory holds shogym's environments. Every env follows the same **env-as-center** design,
in which an environment does exactly three things:

1. **describe** a task — publish a `TaskSpec` (instructions + tool manifest + horizon);
2. **serve** its essential tools over [MCP](https://modelcontextprotocol.io) (in-process,
   stdio, or HTTP);
3. **verify** a *recorded trajectory* of tool calls with a pure function.

There is no agent loop, no observation stream, and no model inside the env. An external
harness (Claude Code, Codex, pi, Hermes, or a small in-process loop) drives the tools; the
env only describes, serves, and scores. Hold `(env, task)` fixed, swap the harness, and the
delta in the trace is attributable to the harness.

The machinery every env shares lives here — the terminal lifecycle vocabulary, how a score is
read back off a trace, the requirements boilerplate, and the offline-vs-keyed test split. Each
env's own README states only what is *specific* to that env and links back here for the rest.

## Terminal lifecycle: seal, terminal, score-terminal, abort

Every env ends an episode through the same lifecycle, and each env README uses this vocabulary
verbatim:

- **terminal** — a tool call that *ends* the episode. Each tool advertises a `terminal_kind`
  in the manifest: `none` (an ordinary step), `score`, or `abort`.
- **seal** — ending an episode is atomic: the first terminal call **seals** the episode,
  freezing it so it can never be continued. A verdict only ever exists for an already-sealed,
  un-continuable episode, so an agent cannot read a score and then revise.
- **score terminal** (`terminal_kind: score`) — a tool whose call seals the episode and then
  runs the env's **`finalize`** hook, which produces the verdict as core-owned
  `TerminalEvidence` (never forgeable tool output). Examples: `submit_answer`, `done`,
  `submit`. The env's pure `verify` scores from that evidence, not from marker JSON on the
  trajectory.
- **abort terminal** (`terminal_kind: abort`) — the reserved **`terminate`** tool every env
  serves. It ends the episode with **no** score (a premature, no-credit end). After a score
  terminal has already sealed, a later `terminate` is a tombstoned no-op.
- **horizon** — reaching the step cap also ends the episode. A seal env runs the same
  `finalize` (source `horizon`); a marker env (wordle) simply stops and scores the recorded
  trajectory.

A **marker/trivial** env (wordle) has no `finalize`: its `verify` is a pure function over the
recorded trajectory itself, and the episode ends on `terminate` or the horizon. Every other
env is a **seal env**: the score terminal seals, `finalize` produces the verdict, and `verify`
reads it.

## Reading a score back (`result_from_trace`)

Every served tool call appends one row to the JSONL trace; the episode scores ride out on the
terminal result's `_meta` sidecar and the terminal trace row. To read the score back, filter
the trace to the run and take its terminal row:

```python
import shogym
result = shogym.result_from_trace("shogym_logs/run.jsonl", env="<env>", task="0")
print(result.terminated, result.value("<metric>"))
```

`result_from_trace` treats `env` / `task` / `session_id` as **filters** (not just labels):
each narrows the rows before the terminal row is chosen, so a shared, append-only trace can't
let another run supply a stale result. For a guaranteed 1:1 mapping, give each run its own
trace file. The in-process `shogym.evaluate` path and the external `shogym serve` path converge on
the same `EvalResult` this way.

## Requirements boilerplate

Every env that needs an optional extra states its own data / keys, but the pin-and-install
mechanics are identical:

- **Python 3.12.** The project is pinned to 3.12 (`requires-python = ">=3.12,<3.13"`, with a
  committed `.python-version`) — the tau2 port needs it.
- **`uv sync` includes the extra.** Each env's extra is also listed in the default `dev`
  dependency-group, so `uv sync` / `uv run …` install it without a manual `--extra` flag.
- **`pip install shogym` stays lean.** A plain install pulls no env extra; add one explicitly
  with `pip install 'shogym[<extra>]'`.
- **`import shogym` registers without importing.** Importing shogym registers every env but
  imports **none** of the optional deps — an env's heavy/optional imports happen lazily only
  when that env is constructed or served, so `import shogym` works offline without any extra.
- **Some upstreams are fetched, not installed.** The `tau2`, `yc_bench`, and `automationbench`
  ports declare no pip dependency on their upstream: shogym cannot carry a direct (`@ git+…`)
  requirement and still publish to PyPI, so each fetches its SHA-pinned upstream source once
  into `~/.cache/shogym/<package>/<sha>/` when the env is first constructed or served, and its
  extra declares upstream's own runtime dependencies explicitly. `<PACKAGE>_SRC` points at a
  local checkout instead; `SHOGYM_CACHE` relocates the cache. Provisioning **refuses** to bind an
  upstream name that is already imported from somewhere else rather than reporting success over
  the top of it — `tau2` on PyPI is an unrelated project, so "a module called `tau2` is
  importable" is not the same claim as "the pinned tau2-bench is loaded".

## Tests: offline vs keyed

Each env's test suite splits the same way. The **pure** verifier / metric / helper tests and
the **served, scripted-offline** tests (an injected judge, a scripted user simulator, an
in-memory searcher, or a deterministic in-process sim) run in the core offline suite with no
API key; a **keyed fidelity test** — the real model judge or live user simulator grading a
served episode — is skipped unless `OPENAI_API_KEY` is set. So the core 3.12 suite stays green
offline.

The three runtime-provisioned envs add one wrinkle: their test modules must import a production
adapter before they can collect anything, and that import needs the extra and (on a cold cache)
the network. Those two failures skip the module, and **nothing else does** — upstream drift, a
gap in a hand-maintained extra, a corrupt cache or a plain `NameError` all fail, because a
regression that deletes an env's tests while the run stays green is worse than a red one. CI sets
`SHOGYM_REQUIRE_UPSTREAM=1`, which removes even the environmental skip: that runner has the
extras and the network, so there is nothing left for it to legitimately skip.

## Available environments

| Env | What it is | README |
|---|---|---|
| `wordle_v1` | The reference env-as-center environment — Wordle in the smallest honest form (`guess` + reserved `terminate`, a pure trajectory verifier). No extra deps; runs on core shogym. | [`wordle/README.md`](wordle/README.md) |
| `tau2_mock`, `tau2_airline`, `tau2_retail`, `tau2_telecom`, `tau2_banking_knowledge` | [τ²-bench](https://github.com/sierra-research/tau2-bench) served through shogym at upstream source commit `1d244f5`, with domain data from an unversioned `TAU2_DATA_DIR` checkout — tool-using customer-service agents across domains, scored by tau2's own evaluator. Needs the `tau2` extra + tau2 data. | [`tau2/README.md`](tau2/README.md) |
| `yc_bench` | [YC-Bench](https://github.com/collinear-ai/yc-bench) served through shogym at upstream commit `e7d6067` — operate a simulated AI startup for one year via a single `run_command` tool, scored on survival, funds, and tasks completed. Needs the `yc_bench` extra (deterministic in-process sim — no data or key). | [`yc_bench/README.md`](yc_bench/README.md) |
| `hle` | [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle) served through shogym — a single-turn, expert-level question answered via one `submit_answer` tool, graded server-side with HLE's own judge prompt (exact-match fast path, then an OpenAI model judge; shogym's first model-graded verifier). Needs the `hle` extra, `OPENAI_API_KEY`, and gated `cais/hle` access. | [`hle/README.md`](hle/README.md) |
| `browsecomp_plus` | [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus) served through shogym at upstream commit `0469490` for the qrels and the copied evaluation code, with separately pinned Hugging Face revisions for the queries and the BM25 index — answer reasoning-heavy queries against a fixed ~100K-doc corpus via `search` / `get_document` / `submit_answer`, graded by an LLM judge plus deterministic retrieval-recall / citation metrics. Needs the `browsecomp_plus` extra, `OPENAI_API_KEY`, Java 21, and gated dataset access. | [`browsecomp_plus/README.md`](browsecomp_plus/README.md) |
| `automationbench` | [AutomationBench](https://github.com/zapier/AutomationBench) served through shogym at upstream commit `a321764` — carry out a cross-application business workflow over a fully simulated world of ~47 SaaS apps via an `api` tool surface, scored end-state-only by upstream's own rubric. Needs the `automationbench` extra (offline / deterministic — no key). | [`automationbench/README.md`](automationbench/README.md) |
| `frontier_bench` | [Frontier-Bench](https://github.com/harbor-framework/frontier-bench) served through shogym at upstream commit `eb4af26c` — operate a per-task Docker container through a shell (`exec` / `read_file` / `write_file` / `done`), scored by the task's own verifier over the container end-state. A CPU-only, single-container slice (5 tasks). Needs the `frontier_bench` extra + a local Docker daemon (no key or data download). | [`frontier_bench/README.md`](frontier_bench/README.md) |

Runnable end-to-end demos (Claude Code drives a served env; shogym scores off the trace) live
under [`examples/`](../../../examples/), one per harness. Each serves a
whole queue over a single MCP endpoint, and the environment it serves is one variable at the top
of its `serve.py` — so every env listed above runs from any of them.

## Adding an env: the README template

Every env ships its own `README.md` in the same shape, so a reader (or an optimizer) can move
between envs without relearning the layout. A new env MUST follow this canonical order and
reuse the exact shared headings below. The intro, terminal vocabulary, `result_from_trace`
semantics, requirements boilerplate, and offline-vs-keyed note are **not** restated per env —
each env keeps a one-liner and links back to the matching section of this README.

```
# `<registered_env_id>` — <BenchmarkName>, <hook>
<upstream> served through shogym[ at <pinned commit or release>]. <one line: what the task is.>
<one line: what the port reuses from upstream verbatim, and what it replaces.>
<one-liner: env-as-center → link to this README; the runnable example link.>

## Running it       ← FIRST content section: construct → serve over MCP → drive it
   ### Construct + serve
   ### Claude Code example
## Requirements     (optional — extras / data / keys; link here for the shared pin/install boilerplate)
## How it works     describe → serve → verify, as subsections:
   ### describe → TaskSpec
   ### Tools (served over MCP)
   ### finalize + verify        (seal env)  — or  ### verify  (marker/trivial like wordle)
## Tasks            env-specific — dataset / domains / splits
## Scoring          the metrics + the `result_from_trace` read-back (link here for filter semantics)
## Fidelity & deviations   every deviation from upstream, in one place
## Gotchas          (optional — the sharp edges)
## Layout           (optional appendix, at the end — a source-map table)
```

**Shared headings that MUST match verbatim across envs** (a reader greps for these):
`Running it`, `Requirements`, `How it works`, `Scoring`, `Fidelity & deviations`, `Gotchas`,
`Layout`. `Requirements`, `Gotchas`, and `Layout` are optional — include them when the env has
extras / keys / data, sharp edges, or a source map worth drawing.

- **The opener summarises the pin; `Fidelity & deviations` is authoritative.** Name the commit
  or release in the opener when the port pins one, and give the exact pin and its limits below.
  The clause is optional: `hle` loads the `cais/hle` split at whatever revision resolves, so its
  opener drops it and the next line names what it loads instead. An opener never implies a pin
  the code does not make.
- **`How it works`** always carries `describe → TaskSpec` then `Tools (served over MCP)`, then
  a verifier subsection: **`finalize + verify`** for a seal env (the score terminal seals,
  `finalize` produces the verdict, `verify` reads it) or **`verify`** for a marker/trivial env
  (wordle — a pure verifier over the trajectory, no `finalize`). Use the heading bare: no
  parentheticals, no arrows.
- **`Fidelity & deviations`** is the one place every deviation from upstream lives: the exact
  pinned upstream commit / tag and what it does not cover, the retriever / judge / variant /
  model choices, any deferred scope (multimodal, dense retrieval, keyed variants), and any
  intentional delta from the original benchmark. Every question of the form "what does this port change" is answered here, and
  nothing fidelity-related hides in `Gotchas` or `Requirements`.
