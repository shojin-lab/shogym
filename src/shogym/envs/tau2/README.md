# `tau2_*` — τ²-bench, tool-using customer-service agents

[**τ²-bench**](https://github.com/sierra-research/tau2-bench) (Sierra's benchmark for tool-using
customer-service agents) served through shogym at upstream source commit `1d244f5`. The domain
policies, tasks and DBs come from a checkout the caller supplies, by `TAU2_DATA_DIR` or by
upstream's source-relative `<checkout>/data` fallback; shogym version-checks neither. tau2 puts
an agent in a domain (airline, retail, telecom, …) where it uses domain tools and converses
with a simulated user to resolve a task, then scores the run with a deterministic evaluator
(final DB state, the actions taken) plus optional NL assertions.

Like every shogym env this **describes** a task, **serves** its tools over MCP, and **verifies**
a recorded trajectory while an external harness drives the tools — see
[`../README.md`](../README.md). tau2's own contract is different (its `Orchestrator` *drives
the agent*); this port bridges the two by reusing tau2's `Orchestrator`, user simulator, domain
tools/tasks and evaluator verbatim and replacing only the agent. The runnable demo is
[`examples/`](../../../../examples/).

## Running it

> Requires **Python 3.12 + the `tau2` extra**, a tau2 `data/` checkout (by `TAU2_DATA_DIR`, or
> by a full `TAU2_SRC` clone that carries its own `data/`), and a one-time fetch of the pinned
> upstream source on first construction — see [Requirements](#requirements) below.

### Construct + serve

```python
import shogym

env = shogym.make("tau2_mock")          # needs tau2 data (TAU2_DATA_DIR, or a full TAU2_SRC clone)
spec = env.describe("0")               # task 0: policy + this task's ticket + the tool manifest
```

Serve it as a stdio MCP server any harness can drive:

```bash
uv run python -m shogym.cli serve tau2_mock --task 0 --trace ./shogym_logs/tau2_mock.jsonl
```

The harness completes the task with the domain tools, then calls **`done`** — tau2's **score
terminal**. Calling `done` seals the shogym episode and runs tau2's evaluator in `finalize` (its
result reports the score) in one step; the episode ends there (no separate `terminate` needed).
`terminate` remains available as an explicit abort (a no-score end). See the shared
[terminal lifecycle](../README.md#terminal-lifecycle-seal-terminal-score-terminal-abort). shogym
reads the verdict off the trace via `shogym.result_from_trace(...)`.

**Config** (via `shogym.make(name, config)` / `env_config`): `task_split` (`"train"`/`"test"`),
`max_steps` (default 100, matching upstream), `user_llm` / `user_llm_args` (non-solo user
simulator — pass `user_llm_args={"mock_response": "…"}` for a deterministic **offline** user),
and `evaluation_type` (default `"all"`; use `"env"` for an offline run of an NL-basis domain).

### Quickstart

Any quickstart under [`examples/`](../../../../examples/) serves this env: one MCP endpoint
hands out a queue of tasks and scores each one server-side. Point it here with the single
variable at the top of its `serve.py`:

```python
ENV = "tau2_mock"
```

The mock domain needs no key and no network of its own once tau2's `data/` is reachable, by
`TAU2_DATA_DIR` or by a full `TAU2_SRC` clone. It is not strictly network-free: serving imports
tau2's `registry`, which reaches `agent.llm_agent` → `utils.llm_utils` → `litellm`, and litellm
fetches a model-cost map on import unless `LITELLM_LOCAL_MODEL_COST_MAP=true` is set, falling
back to a bundled copy when that fails.
A real (non-solo) domain additionally needs `OPENAI_API_KEY` for tau2's own user simulator,
which is a real cost.

## Requirements

The Python pin and the `uv sync` / `pip install` / `import shogym` mechanics are the shared
[requirements boilerplate](../README.md#requirements-boilerplate); the `tau2` extra is in the
default `dev` group, so `uv sync` includes it. On top of that:

- **The pinned upstream source is provisioned at runtime** into `~/.cache/shogym/tau2/<sha>/`
  on first construction — a one-time network fetch (~93 MB downloaded, ~5 MB kept; only
  `src/tau2` is extracted, the archive's ~700 MB of benchmark `data/` is filtered out). Set
  `TAU2_SRC` to a local checkout (a dir containing the `tau2` package) to skip the fetch, or
  `SHOGYM_CACHE` to relocate the cache. Upstream is **not** a pip dependency: PyPI forbids
  direct (`@ git+…`) references, and the `tau2` name on PyPI belongs to an unrelated
  magnetochemistry library — so the port fetches and imports the pinned source instead, exactly
  as `automationbench` does. The `tau2` extra declares upstream's own runtime dependencies
  explicitly (its base deps plus its `[gym]` and `[knowledge]` extras), since pip no longer
  resolves them transitively.
- **tau2 data**, separately. tau2 does **not** ship its `data/` in the install, and the default
  runtime fetch extracts `src/tau2` alone, so a provisioned-from-cache run must supply the data
  itself. Two routes work: set `TAU2_DATA_DIR` to a tau2-bench `data/` checkout, or point
  `TAU2_SRC` at a **full** clone's `src/`, whose sibling `data/` upstream finds on its own
  (`tau2/utils/utils.py` falls back to `<checkout>/data` when `TAU2_DATA_DIR` is unset). shogym
  version-checks neither.
- **`OPENAI_API_KEY`** — required for the **default/live user simulator** on non-solo domains
  (it's an OpenAI LLM), and for evaluator paths that call a judge (NL assertions, mostly
  `retail`) or dense retrieval (`banking_knowledge` only; `retail` ships no retrieval code). It is
  *not* needed to run a non-solo domain with a scripted offline user
  (`user_llm_args={"mock_response": "…"}`), nor for the solo `mock` domain.

## How it works

tau2's `Orchestrator` *drives the agent* (it asks the agent for its next action, executes
tools, invokes the user simulator, checks termination). shogym's harness *drives tool calls*.
The bridge (`mcp_server.py`) reconciles the two by **replacing only the agent** — mapped onto
shogym's env-as-center trio:

### describe → TaskSpec

`env.describe(task_id)` publishes the task contract the harness reads: the domain **policy**
(tau2's agent system prompt), the specific task's **ticket** (what the user wants), and the
**tool manifest** — the domain tools, plus `send_message` on non-solo domains, plus the
reserved `done` and `terminate`.

### Tools (served over MCP)

The env's in-process MCP server hosts tau2's `Orchestrator` on a **background thread** (one
per episode) and reuses tau2's own `GymAgent` — a `HalfDuplexAgent` whose
`generate_next_message` **blocks until an external `set_action`**. Each incoming MCP tool call
becomes the agent's next action:

- a **domain tool** → an `AssistantMessage` with one tool call; the tool output flows back as
  the MCP result;
- **`send_message`** (non-solo domains) → an `AssistantMessage` with text routed to the
  **user simulator**; its result is the user's reply;
- **`done`** → the env's **score terminal**. The serve layer validates it, atomically seals the
  episode, then runs this env's `finalize` hook: tau2 agent-stop, the Orchestrator finalizes,
  and tau2's **evaluator** scores its final state — **exactly once**. The verdict is returned
  as the tool result and the episode ends there.

### finalize + verify

`done` is a `score` terminal, so shogym scores from **core-owned terminal evidence**, not marker
JSON. On the sealed episode the serve layer runs `finalize`, a tau2-owned atomic `finalize_once`
on the background Orchestrator: if tau2 already stopped autonomously (max-step, or the user
simulator ended the conversation) it returns the stashed verdict; otherwise it delivers `done`
once, waits, and runs `evaluate_simulation` once — never double-stopping the Orchestrator. The
evaluator's exception text (on failure) is a private diagnostic, never surfaced to the agent.
`_verify` then scores from `evidence.verdict`. Reaching the shogym horizon runs the same
`finalize` (source `horizon`), so a hit cap scores tau2's evaluator over the completed run,
never an independent premature zero; an explicit `terminate` (abort) is a no-score premature
end.

## Tasks

Each registered env is a tau2 domain; its tasks are tau2's benchmark tasks for that domain,
selected by `task_split`.

| Env | Domain | Mode | Notes |
|---|---|---|---|
| `tau2_mock` | mock | **solo** (tau2 `DummyUser`) | tiny, fully offline — no user-sim LLM |
| `tau2_airline` | airline | non-solo | user simulator (LLM) |
| `tau2_retail` | retail | non-solo | user simulator; NL-assertion reward |
| `tau2_telecom` | telecom | non-solo | user operates device tools; user simulator |
| `tau2_banking_knowledge` | banking_knowledge | non-solo | pinned to the offline `bm25_grep` retrieval variant |

**Solo vs non-solo:** in solo mode the agent works autonomously (tau2's `DummyUser`, no
user-simulator LLM), so the episode runs offline. Non-solo domains drive tau2's LLM user
simulator, so an episode makes model calls (see [Requirements](#requirements)).

**Splits:** `airline` / `retail` / `telecom` expose tau2's declared `train`/`test` splits
verbatim; `mock` and `banking_knowledge` declare no train/test holdout, so both splits return
the full task set.

## Scoring

Scoring is tau2's own `evaluate_simulation`, run server-side over tau2's final state; shogym
never reimplements it. `_verify` surfaces it as episode feedback:

- **`reward`** — tau2's reward (the product of the applicable components in the task's
  `reward_basis`); `1.0` is a full pass.
- **`success`** — `reward >= 1.0`.
- **`db_match`** — did the final database state match the task's expected state.
- **`action_match_proportion`** — fraction of the task's expected agent actions that matched.

A premature end — the harness calls `terminate` (abort) instead of `done` — scores
`reward = 0.0`, `success = False`. An evaluator failure fails closed to `reward = 0.0` with an
`eval_error` flag (the exception text is a private diagnostic, never shown to the agent).

Read the score back off the trace:

```python
import shogym
result = shogym.result_from_trace("shogym_logs/tau2_mock.jsonl", env="tau2_mock", task="0")
print(result.terminated, result.value("reward"), result.value("success"))
```

`result_from_trace` treats `env` / `task` / `session_id` as **filters** — see
[Reading a score back](../README.md#reading-a-score-back-result_from_trace) for the shared
semantics (give each run its own trace file for a guaranteed 1:1 mapping).

## Fidelity & deviations

- **What is reused, and what is not.** tau2's `Orchestrator`, user simulator,
  domains/tools/tasks, and **evaluator** are reused **verbatim**, and the agent is swapped for
  the harness. `reward` and `db_match` are tau2's own numbers off `evaluate_simulation`
  (`RewardInfo.reward`, `DBCheck.db_match`). shogym derives the rest: `action_match_proportion`
  is a ratio it computes over upstream's per-action `action_checks`, `success` is
  `reward >= 1.0`, and an abort or a missing verdict scores `reward = 0.0`. There are **zero
  shogym core changes**; the whole port is additive under `src/shogym/envs/tau2/`.
- **Pinned to source commit `1d244f5`.** The pin covers tau2's Python source only, and the
  default provisioner extracts `src/tau2` alone, so it ships no benchmark data at all. The
  domain policies, tasks and DBs arrive by one of two caller-controlled routes: `TAU2_DATA_DIR`,
  or upstream's source-relative fallback (`tau2/utils/utils.py` resolves `<checkout>/data` when
  that variable is unset), which a `TAU2_SRC=/repo/src` clone satisfies on its own. shogym
  checks no revision or hash on either route, and `TAU2_SRC` likewise swaps the source for a
  checkout nothing version-checks, so two runs under the same commit label can serve different
  benchmark content.
- **Python 3.12 pin rationale.** The pinned tau2 revision imports the stdlib `audioop` module,
  removed in 3.13 — this is why the shared project pin is `>=3.12,<3.13`.
- **banking uses `bm25_grep`.** `tau2_banking_knowledge` is pinned to the offline `bm25_grep`
  retrieval variant (rank-bm25, no embeddings) so it constructs/serves offline. The benchmark
  default (`alltools`, dense OpenAI embeddings) needs a key at construction time and is a keyed
  follow-up.
- **NL-assertion reward is keyed.** `retail` carries `NL_ASSERTION` in nearly every task's
  `reward_basis` (112 of 114), so its *full* reward needs a judge LLM. `banking_knowledge`
  carries it in exactly one task of 97 (the rest are `DB` or `ACTION`), so a keyless run loses
  almost nothing there. Offline runs use
  `evaluation_type="env"` to score the deterministic DB component only.
- **Splits verbatim.** `airline` / `retail` / `telecom` use tau2's declared `train`/`test`
  splits verbatim (no positional slicing, no leakage). `mock` and `banking_knowledge` declare no
  holdout, so both splits return the full task set.
- **Upstream's max-step is what normally binds, and it scores zero.** The same `max_steps` goes
  to tau2's `Orchestrator` while the shogym horizon is `max_steps + 2`, so for ordinary valid
  actions upstream reaches its own message-hop budget first: it sets `MAX_STEPS`, and its
  evaluator scores every termination outside `AGENT_STOP` / `USER_STOP` as a premature
  `reward = 0.0`. The bridge evaluates and stashes that verdict as soon as it sees the stop, and
  `finalize_once` returns the stored verdict rather than sending `done`, so reaching the outer
  horizon afterwards does not convert the run to `AGENT_STOP`. The outer horizon binds first only
  when calls are rejected before they advance tau2 (a malformed or disallowed call consumes a
  shogym step but no tau2 step); in that case `finalize` does deliver `done`, tau2 stops as
  `AGENT_STOP`, and the evaluator scores the completed run. Either way shogym invents no zero of
  its own; upstream's is passed through.

## Gotchas

- **Solo-mode task limits.** Some `mock` tasks start mid-conversation (an
  `initial_state.message_history` ending in an agent→user turn) and require a *user* reply —
  e.g. `mock` task 4 (`update_task_with_message_history`). In solo mode there is no user
  (`DummyUser`), so tau2 terminates immediately and the task scores **0.0**. Use a
  solo-solvable task (e.g. `mock` task 0 scores 1.0) or a non-solo domain for conversational
  tasks.
- **Completion is one step:** `done` is the score terminal — it seals the episode, runs tau2's
  evaluator, and ends the episode in a single call. A `terminate` after `done` is a no-op
  (the episode is already sealed); `terminate` *instead of* `done` is an abort (premature zero).
- **Offline vs keyed tests.** Follows the shared
  [offline-vs-keyed split](../README.md#tests-offline-vs-keyed): the pure-`verify` unit tests and
  the mock/served + non-solo (`mock_response` user) tests run offline; the keyed fidelity test —
  which replays gold agent actions through both this bridge and upstream tau2's `AgentGymEnv` and
  asserts identical scores — is skipped unless `OPENAI_API_KEY` is set.

## Layout

| File | Role |
|---|---|
| `env_v1.py` | The registered `tau2_*` envs (one per domain): `describe` (policy + ticket + manifest), the `finalize` hook (tau2 agent-stop → `evaluate_simulation` once, on the sealed episode), and the pure `_verify` scorer (verdict from evidence). |
| `mcp_server.py` | The bridge: hosts tau2's `Orchestrator` + `GymAgent` on a per-episode background thread, turning each MCP tool call into the agent's next action; `done` is the score terminal. |
| `mock_server.py` | The solo `mock` domain server (tau2 `DummyUser`, fully offline). |
| `airline_server.py` / `retail_server.py` / `telecom_server.py` | The non-solo domain servers (domain tools + `send_message` to the LLM user simulator). |
| `banking_knowledge_server.py` | The `banking_knowledge` domain server, pinned to the offline `bm25_grep` retrieval variant. |
