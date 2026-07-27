# tau2-bench — tool-using customer-service agents, wrapped faithfully in hgym

A faithful hgym port of [**τ²-bench**](https://github.com/sierra-research/tau2-bench)
(Sierra's benchmark for tool-using customer-service agents). tau2 puts an agent in a domain
(airline, retail, telecom, …) where it uses domain tools and converses with a simulated user
to resolve a task, then scores the run with a deterministic evaluator (final DB state, the
actions taken) plus optional NL assertions.

hgym's contract is different from tau2's: an hgym env **describes** a task, **serves** its
tools as an MCP server, and **verifies** a recorded trajectory — an external harness (Claude
Code, Codex, …) drives the tools. This port bridges the two while keeping tau2's benchmark
intact.

## Running it

> Requires **Python 3.12 + the tau2 extra** and tau2 data — see [Requirements](#requirements)
> below. The Claude Code example handles both for you.

### Construct + serve

```python
import hgym

env = hgym.make("tau2_mock")          # needs tau2 data (set TAU2_DATA_DIR)
spec = env.describe("0")               # task 0: policy + this task's ticket + the tool manifest
```

Serve it as a stdio MCP server any harness can drive:

```bash
uv run python -m hgym.cli serve tau2_mock --task 0 --trace ./hgym_logs/tau2_mock.jsonl
```

The harness completes the task with the domain tools, then calls **`done`** — tau2's **score
terminal**. Calling `done` seals the hgym episode and runs tau2's evaluator (its result
reports the score) in one step; the episode ends there (no separate `terminate` needed).
`terminate` remains available as an explicit abort (a no-score end). hgym reads the verdict
off the trace via `hgym.result_from_trace(...)`.

**Config** (via `hgym.make(name, config)` / `env_config`): `task_split` (`"train"`/`"test"`),
`max_steps` (default 100, matching upstream), `user_llm` / `user_llm_args` (non-solo user
simulator — pass `user_llm_args={"mock_response": "…"}` for a deterministic **offline** user),
and `evaluation_type` (default `"all"`; use `"env"` for an offline run of an NL-basis domain).

### Claude Code example

The runnable end-to-end demo (Claude Code plays a served tau2 env; hgym scores off the
trace) lives in [`examples/tau2/claude_code/`](../../../../examples/tau2/claude_code/):

```bash
# mock domain — offline (no OpenAI key, no user-sim cost); auto-downloads tau2 data:
uv run python examples/tau2/claude_code/run.py --task 0

# a real (non-solo) domain — needs OPENAI_API_KEY for tau2's user simulator (real cost):
export OPENAI_API_KEY=sk-...
uv run python examples/tau2/claude_code/run.py --domain telecom --task 0 --transcript
```

## Requirements

- **Python 3.12.** The project is pinned to 3.12 (`requires-python = ">=3.12,<3.13"`, plus a
  committed `.python-version`) because the pinned tau2 revision imports the stdlib `audioop`
  module, removed in 3.13.
- **`uv sync`** builds the venv with tau2: the `tau2` extra is also listed in the default
  `dev` dependency-group, so `uv sync` / `uv run …` include it without a manual `--extra`
  flag. (`pip install hgym` stays lean; `pip install hgym[tau2]` adds tau2 explicitly.)
- **tau2 data.** tau2 does **not** ship its `data/` in the install, so it must be
  provisioned: either set `TAU2_DATA_DIR` to a tau2 data checkout, or use the Claude Code
  example, which **lazy-downloads** the pinned data to `~/.cache/hgym/tau2-bench` on first run.
- **`OPENAI_API_KEY`** — required for the **default/live user simulator** on non-solo domains
  (it's an OpenAI LLM), and for evaluator paths that call a judge (NL assertions) or dense
  retrieval (retail, banking). It is *not* needed to run a non-solo domain with a scripted
  offline user (`user_llm_args={"mock_response": "…"}`), nor for the solo `mock` domain.

## How it works

tau2's `Orchestrator` *drives the agent* (it asks the agent for its next action, executes
tools, invokes the user simulator, checks termination). hgym's harness *drives tool calls*.
The bridge (`mcp_server.py`) reconciles the two by **replacing only the agent** — mapped onto
hgym's env-as-center trio:

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
  episode, then runs this env's `finalize` hook (seal-before-verdict): tau2 agent-stop, the
  Orchestrator finalizes, and tau2's **evaluator** scores its final state — **exactly once**.
  The verdict is returned as the tool result and the episode ends there.

### finalize + verify

`done` is a `score` terminal, so hgym scores from **core-owned terminal evidence**, not marker
JSON. On the sealed episode the serve layer runs `finalize`, a tau2-owned atomic `finalize_once`
on the background Orchestrator: if tau2 already stopped autonomously (max-step, or the user
simulator ended the conversation) it returns the stashed verdict; otherwise it delivers `done`
once, waits, and runs `evaluate_simulation` once — never double-stopping the Orchestrator. The
evaluator's exception text (on failure) is a private diagnostic, never surfaced to the agent.
`_verify` then scores from `evidence.verdict`. Reaching the hgym horizon runs the same
`finalize` (source `horizon`), so a hit cap scores tau2's evaluator over the completed run
(**preserve_upstream_maxstep**), never an independent premature zero; an explicit `terminate`
(abort) is a no-score premature end.

---

tau2's `Orchestrator`, user simulator, domains/tools/tasks, and **evaluator** are reused
**verbatim** — only the agent is swapped and the hgym-side terminal wiring rides on the seal.
There are **zero hgym core changes**; the whole port is additive under
`src/hgym/envs/tau2/`. `import hgym` registers the `tau2_*` envs
**without importing tau2**, so the core stays lean and offline; tau2 is loaded only when a
tau2 env is constructed or served.

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

Scoring is tau2's own `evaluate_simulation`, run server-side over tau2's final state; hgym
never reimplements it. `_verify` surfaces it as episode feedback:

- **`reward`** — tau2's reward (the product of the applicable components in the task's
  `reward_basis`); `1.0` is a full pass.
- **`success`** — `reward >= 1.0`.
- **`db_match`** — did the final database state match the task's expected state.
- **`action_match_proportion`** — fraction of the task's expected agent actions that matched.

A premature end — the harness calls `terminate` (abort) instead of `done` — scores
`reward = 0.0`, `success = False`. An evaluator failure fails closed to `reward = 0.0` with an
`eval_error` flag (the exception text is a private diagnostic, never shown to the agent).

## Gotchas

- **Solo-mode task limits.** Some `mock` tasks start mid-conversation (an
  `initial_state.message_history` ending in an agent→user turn) and require a *user* reply —
  e.g. `mock` task 4 (`update_task_with_message_history`). In solo mode there is no user
  (`DummyUser`), so tau2 terminates immediately and the task scores **0.0**. Use a
  solo-solvable task (e.g. `mock` task 0 scores 1.0) or a non-solo domain for conversational
  tasks.
- **banking uses `bm25_grep`.** `tau2_banking_knowledge` is pinned to the offline `bm25_grep`
  retrieval variant (rank-bm25, no embeddings) so it constructs/serves offline. The benchmark
  default (`alltools`, dense OpenAI embeddings) needs a key at construction time and is a
  keyed follow-up.
- **NL-assertion reward is keyed.** `retail` and `banking_knowledge` have `NL_ASSERTION` in
  their `reward_basis`, so the *full* reward needs a judge LLM. Offline runs use
  `evaluation_type="env"` to score the deterministic DB component only.
- **Task splits.** `airline` / `retail` / `telecom` use tau2's declared `train`/`test` splits
  verbatim (no positional slicing, no leakage). `mock` and `banking_knowledge` declare no
  train/test holdout, so both splits return the full task set.
- **Completion is one step:** `done` is the score terminal — it seals the episode, runs tau2's
  evaluator, and ends the episode in a single call. A `terminate` after `done` is a no-op
  (the episode is already sealed); `terminate` *instead of* `done` is an abort (premature zero).
- **Offline vs keyed tests.** Pure-`verify` unit tests and the mock/served + non-solo
  (`mock_response` user) tests run offline in the suite; a keyed fidelity test — which
  replays gold agent actions through both this bridge and upstream tau2's `AgentGymEnv` and
  asserts identical scores — is skipped unless `OPENAI_API_KEY` is set.
