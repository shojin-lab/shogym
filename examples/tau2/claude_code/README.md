# Claude Code plays τ²-bench (through a served hgym env)

The end-to-end demo of the env-as-center design (RFC 008) on the tau2-bench port (issue
#31): hgym serves a tau2 domain as an MCP server, **Claude Code** drives it as an external
harness, and hgym scores the result off the trace using **tau2's own evaluator**. hgym
never sees Claude Code's model, prompt, or loop — only the tool calls and the verifier's
feedback.

```
  Claude Code  ──spawns──▶  hgym serve tau2_mock       (stdio MCP server)
  (the harness)             ├─ describe        → domain policy + your task + tools
      │                     ├─ create_task(…)  → tool result   (+ _meta feedback)
      │                     ├─ done()          → tau2 evaluator verdict
      └──tool calls────────▶└─ terminate()     → episode scored → ./hgym_logs/tau2_mock.jsonl
                                                                          │
                            hgym.result_from_trace(...) ◀────────────────┘
```

## The flow: `done` → `terminate`

tau2 defines task completion (its evaluator runs, `success`/`db_match`/`action_match`) and
hgym defines episode termination. The agent does both, in order:

1. Complete the task with the domain tools (and, on non-solo domains, `send_message` to talk
   to the user — its result is the user's reply).
2. Call **`done`** — this ends tau2's simulation and runs its evaluator; the verdict comes
   back **in the tool result**.
3. Call **`terminate`** — this ends the hgym episode; hgym's verifier reads the verdict off
   the recorded `done` step into the terminal feedback.

The agent learns this from the task instructions it reads via `describe`.

## Prerequisites

The project is pinned to **Python 3.12** (the pinned tau2 revision imports the stdlib
`audioop`, removed in 3.13), so setup is one command:

```bash
uv sync   # builds the 3.12 .venv with tau2 (it's in the dev group); `uv run` then just works
```

Confirm: `uv run python -c "import tau2; print('ok')"`. The only other requirement is the
[`claude`](https://www.anthropic.com/claude-code) CLI on your `PATH`, with credentials.

**tau2 data is handled for you:** tau2 doesn't ship its `data/`, so on the first run `run.py`
lazy-downloads the pinned tau2-bench data to `~/.cache/hgym/tau2-bench` (one-time). Set
`TAU2_DATA_DIR` to an existing checkout to skip the download. `run.py` also runs a
**preflight** that fails fast with the exact fix if tau2 can't be imported or the domain won't
build — rather than letting Claude connect to a crashed, toolless server and give up.

## Run it — the `mock` domain (no OpenAI key, no user-sim cost)

The default domain is `mock`, which tau2 runs **solo** (its `DummyUser`) — so the tau2 side
needs no OpenAI key and no user-simulator LLM. (The Claude harness itself still makes model
calls and needs Claude credentials, as any `claude -p` run does.)

```bash
uv run python examples/tau2/claude_code/run.py --task 0

# print Claude Code's turn-by-turn tool calls and reasoning as it plays:
uv run python examples/tau2/claude_code/run.py --task 0 --transcript

# pick the model / reasoning effort (defaults: claude-sonnet-5 / low):
uv run python examples/tau2/claude_code/run.py --task 0 --model opus --effort high
```

(`uv run` uses the pinned 3.12 `.venv`; a bare `python` only works if that interpreter has
the tau2 extra installed.)

The script writes a per-run `.mcp.json`, runs Claude Code, then prints the tau2 score
(`reward`, `success`, `db_match`, …) read back off the trace.

## Run it — a real domain (needs `OPENAI_API_KEY`, real cost)

The other domains (`airline`, `retail`, `telecom`, `banking_knowledge`) are **non-solo**:
tau2's user simulator is an LLM, so an episode calls the OpenAI API and **costs money**.

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/tau2/claude_code/run.py --domain telecom --task 0 --transcript
```

`telecom` scores offline (ACTION + ENV_ASSERTION), so only the user simulator needs the key.
`retail` / `banking_knowledge` additionally use an LLM judge for their NL-assertion reward.

## Drive it by hand

Using the checked-in [`.mcp.json`](./.mcp.json) (mock, task 0, trace at
`./hgym_logs/tau2_mock.jsonl`) — run from the repo root, so `uv run` (which the config uses to
launch the server) resolves this project's `.venv`:

```bash
claude -p "You are the tau2 agent. Call describe, complete the task, then done, then terminate." \
  --mcp-config examples/tau2/claude_code/.mcp.json \
  --strict-mcp-config \
  --allowedTools "mcp__tau2__*" \
  --disallowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,NotebookEdit,BashOutput,KillShell" \
  --permission-mode dontAsk

# then read the score hgym recorded (via tau2's evaluator):
uv run python -c "import hgym; print(hgym.result_from_trace('hgym_logs/tau2_mock.jsonl'))"
```

(The checked-in config launches the server via `uv run python -m hgym.cli serve` so it uses
the pinned 3.12 `.venv` — a bare `python` on `PATH` need not have hgym or the tau2 extra.)

Why these flags: `--strict-mcp-config` isolates the session to this one server;
`--allowedTools "mcp__tau2__*"` pre-approves the tau2 tools; `--permission-mode dontAsk` runs
non-interactively by **denying** anything not pre-allowed; and `--disallowedTools "Bash,…"`
explicitly removes the built-in tools (Bash/Read/…) so the agent can't take untraced
side-channel actions (read the tau2 data, shell out) — the score stays attributable to the
MCP tool surface alone. We do **not** use `--tools ""`: in current Claude Code that also
strips the MCP tools, leaving the agent with an empty toolset.

## Swapping the harness

Nothing here is Claude-specific. Point any MCP-speaking harness at the same `hgym serve
tau2_<domain>` server and the trace/score path is unchanged — that is the whole point of the
design. Hold `(env, task)` fixed, swap the harness, and the delta in `hgym_logs/*.jsonl` is
attributable to the harness.
