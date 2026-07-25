# hgym environments

This directory holds hgym's environments. Every env follows the **env-as-center** design
([RFC 008](https://github.com/anndvision/hgym/wiki/RFC-008-Environment-as-Center-of-Gravity)):
an environment does exactly three things —

1. **describe** a task — publish a `TaskSpec` (instructions + tool manifest + horizon);
2. **serve** its essential tools over [MCP](https://modelcontextprotocol.io) (in-process,
   stdio, or HTTP);
3. **verify** a *recorded trajectory* of tool calls with a pure function.

There is no agent loop, no observation stream, and no model inside the env. An external
harness (Claude Code, Codex, pi, Hermes, or a small in-process loop) drives the tools; the
env only describes, serves, and scores. Hold `(env, task)` fixed, swap the harness, and the
delta in the trace is attributable to the harness.

## Available environments

| Env | What it is | README |
|---|---|---|
| `wordle_v1` | The reference env-as-center environment — Wordle in the smallest honest form (`guess` + reserved `terminate`, a pure trajectory verifier). No extra deps; runs on core hgym. | [`wordle/README.md`](wordle/README.md) |
| `tau2_mock`, `tau2_airline`, `tau2_retail`, `tau2_telecom`, `tau2_banking_knowledge` | A faithful port of [τ²-bench](https://github.com/sierra-research/tau2-bench) — tool-using customer-service agents across domains, scored by tau2's own evaluator. Needs the `tau2` extra + tau2 data (Python 3.12). | [`tau2/README.md`](tau2/README.md) |

Runnable end-to-end demos (Claude Code drives a served env; hgym scores off the trace) live
under [`examples/`](../../../examples/): [`examples/wordle/claude_code/`](../../../examples/wordle/claude_code/README.md)
and [`examples/tau2/claude_code/`](../../../examples/tau2/claude_code/README.md).

## Adding an env: the README template

Every env ships its own `README.md` in the same shape, so a reader (or an optimizer) can move
between envs without relearning the layout. New envs (e.g. yc, HLE) MUST follow this canonical
order and reuse the exact shared headings below. `wordle_v1` and tau2-bench are the two
worked examples.

```
# <name> — <one-line>
<intro: the env-as-center framing (describe / serve / verify) + a link to the example>

## Running it      ← FIRST content section: instantiate → serve over MCP → Claude Code example → evaluate
## Requirements    (optional — extras / Python pin / API keys / data provisioning)
## How it works    describe → serve → verify, as three subsections:
   ### describe → TaskSpec
   ### Tools (served over MCP)
   ### verify
## Tasks           env-specific — dataset / domains / splits
## Scoring         the metrics and how they flow off the recorded trace
## Gotchas         (optional — the sharp edges)
## Layout          (optional appendix, at the end — a source map)
```

**Shared headings that MUST match verbatim across envs** (a reader greps for these):
`Running it`, `Requirements`, `How it works`, `Scoring`, `Gotchas`, `Layout`. `Requirements`,
`Gotchas`, and `Layout` are optional — include them when the env has extras/keys/data, sharp
edges, or a source map worth a map. `How it works` always carries the three
`describe → TaskSpec` / `Tools (served over MCP)` / `verify` subsections, in that order.
