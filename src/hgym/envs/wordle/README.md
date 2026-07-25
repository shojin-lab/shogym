# `wordle_v1` — hgym's reference env-as-center environment

This is the reference implementation of the **env-as-center** design (RFC 008). An
environment does exactly three things, and Wordle shows each one in the smallest honest
form:

1. **describe** a task — publish a `TaskSpec` (rules + tool manifest + horizon).
2. **serve** its essential tools over MCP — here `guess` and the reserved `terminate`.
3. **verify** — score the *recorded trajectory* of tool calls with a pure function.

There is no agent loop, no observation stream, and no model inside the env. An external
harness (Claude Code, Codex, pi, Hermes, or a small in-process loop) drives the tools; the
env only describes, serves, and scores. The end-to-end demo lives at
[`examples/wordle/claude_code/`](../../../../examples/wordle/claude_code/README.md).

No extra dependencies — `wordle_v1` runs on core hgym.

## Running it

**Instantiate directly:**

```python
import hgym
env = hgym.make("wordle_v1")                       # train split
env_test = hgym.make("wordle_v1", config={"task_split": "test"})
```

**Serve it as a stdio MCP server** any harness can spawn (`--trace` optional):

```bash
hgym serve wordle_v1 --task 0 --trace ./hgym_logs/wordle.jsonl
```

**Drive it with Claude Code** (the reference external harness):

```bash
python examples/wordle/claude_code/run.py --task 0
```

Claude Code spawns `hgym serve` as its MCP server, plays the episode through
`describe`/`guess`/`terminate`, and writes the JSONL trace; hgym never sees its model,
prompt, or loop. See [`examples/wordle/claude_code/README.md`](../../../../examples/wordle/claude_code/README.md).

**Drive it in-process** with `hgym.evaluate` — hand it an async harness callable that
receives a FastMCP `Client` connected to the served env:

```python
import hgym

async def harness(client):
    await client.call_tool("guess", {"word": "crane"})
    await client.call_tool("terminate", {})

result = await hgym.evaluate("wordle_v1", harness=harness, task=0)
print(result.value("check_answer"), result.feedback)
```

## How it works

`WordleV1Env` is a `ToolUsingEnv`: it declares its MCP servers, its horizon
(`MAX_GUESSES = 6`), and its advisory templates, and the base class probes the tool
manifest once at construction.

### describe → TaskSpec

`describe()` returns a JSON-serializable `TaskSpec` containing:

- `instructions` — the durable task framing, rendered from
  `functions/guess_v1/example/system.minijinja` (the Wordle rules and what G/Y/X mean).
- `tools` — the essential-tool manifest: `guess` (provenance `env-mandatory`) and
  `terminate` (provenance `reserved`), each with its real JSON Schema.
- `reference_templates` — the advisory system/user templates a harness *may* render
  (hgym never injects them).
- `horizon` — `6`.

A harness reads this once to configure itself. When served, the same contract is also
exposed as the `describe` tool and the `hgym://task` MCP resource.

### Tools (served over MCP)

- **`guess(word: str)`** — backed by the in-process server in `mcp_server.py`. It scores
  `word` against the session's target and returns a dict with `valid`, `score` (a 5-char
  `G`/`Y`/`X` mask), `solved`, `remaining_guesses` (the budget *after* this attempt), and a
  human-readable `feedback` rendering. The server decrements the guess budget on **every**
  accepted call — including malformed ones — so a flood of bad guesses cannot bypass the
  6-guess cap, and it serializes the read-decrement-score path under a lock so concurrent
  calls can't race past the cap.
- **`terminate()`** — the reserved episode-completion tool every env serves
  (`hgym.shared.terminate_mcp`). Its result is a stub acknowledgement; the *call itself*,
  detected by name, is the terminal signal. No env may expose a tool named `terminate` (or
  `describe`).

`score_guess` (in `utils.py`) is the standard two-pass Wordle scorer: greens first, then
yellows against the remaining letter counts, so duplicate letters are handled correctly.

The env pushes per-episode state into the in-process server via `begin_session(session_id,
target)` when the episode starts, and drops it via `end_session(session_id)` on teardown.
State is keyed by session id, so one server safely backs many concurrent episodes.

### verify

`verify` is a pure function over the recorded trajectory. `_verify` scores the recorded
**`guess` argument** against the task answer — it does **not trust the tool result**. For
each recorded `guess` step, `_score_guess_step` pulls `step.arguments["word"]`, re-validates
it (must be a 5-letter alphabetic string), and re-runs `score_guess(word, target)` itself.
The tool's returned `score`/`solved`/`feedback` is untrusted diagnostic data.

This is a deliberate integrity property: a forged or malformed tool result can neither grant
credit nor crash scoring. A malformed `word` scores as an invalid guess rather than raising.
The only thing that earns credit is a real, correct `guess` argument in the trajectory.
Because the transport also strips any caller-supplied `_session_id` before the step is
recorded, the verifier reads only trustworthy trajectory data.

Feedback emitted:

- **Per step (inference):** `format_reward` — was the most recent `guess` well-formed?
- **On termination (episode):**
  - `check_answer` — `True` iff some recorded guess scored `GGGGG`.
  - `partial_credit` — `1.0` if solved, else `best_green / 5.0` (the most greens any single
    guess achieved).
  - `count_turns` — the number of `guess` steps consumed (capped at `MAX_GUESSES`).

Termination happens when `terminate` is called **or** the horizon is reached (step count
≥ 6), whichever comes first — gated env-side in the serving engine.

## Tasks

`WordleV1Default` (registered as `wordle_v1`) loads all 2,315 words and splits 80/20 by
index: the first 80% is `train`, the last 20% is `test`. It defaults to `train`; pass
`config={"task_split": "test"}` to `hgym.make` (or the `task_split` constructor arg) for
the held-out set. Task indices are relative to the chosen split.

## Scoring

Every served tool call appends one row to the JSONL trace
(`session_id, env_name, task_id, step, tool, feedback, terminated`). The feedback is stored
in the same wire form it rides out on each result's MCP `_meta` sidecar, so the trace and
the in-band signal never diverge. Dense per-step `format_reward` is recorded but not
surfaced in-band; the episode scores ride out only on the terminal result.

To read the score back, filter the trace to the run and take its terminal row:

```python
import hgym
result = hgym.result_from_trace("hgym_logs/wordle.jsonl", env="wordle_v1", task="0")
print(result.terminated, result.value("check_answer"), result.value("partial_credit"))
```

`result_from_trace` treats `env`/`task`/`session_id` as **filters** (not just labels): each
narrows the rows before the terminal row is chosen, so a shared, append-only trace can't let
another run supply a stale result. For a guaranteed 1:1 mapping, give each run its own trace
file. The in-process `evaluate` path and the external `hgym serve` path both converge on the
same `EvalResult` this way — hold `(env, task)` fixed, swap the harness, and the delta in the
trace is attributable to the harness.

## Layout

A source map for orientation:

| File | Role |
|---|---|
| `env_v1.py` | `WordleV1Env` (the env) + `WordleV1Default` (the registered `wordle_v1`, with the train/test split) and the authoritative verifier. |
| `mcp_server.py` | The in-process FastMCP server backing the `guess` tool; holds per-episode target + guess budget keyed by session id. |
| `utils.py` | `load_words`, `score_guess` (the G/Y/X scorer), `format_feedback`. |
| `functions/` | The advisory instruction templates (`guess_v1/example/*.minijinja`) and their variable schemas. |
| `data/words.txt` | 2,315 five-letter answers, one per line. |
