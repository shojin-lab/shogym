# Claude Code solves a Frontier-Bench task (through a served hgym env)

The end-to-end demo of the env-as-center design (RFC 008) on the Frontier-Bench port (issue
#44): hgym serves `frontier_bench` as an MCP server, **Claude Code** drives it as an external
harness, and hgym scores the run off the trace using **the task's own verifier over the
container end-state**. hgym never sees Claude Code's model, prompt, or loop — only the tool
calls and the verifier's 0/1 feedback.

```
  Claude Code  ──spawns──▶  hgym serve frontier_bench      (stdio MCP server)
  (the harness)             ├─ describe            → the task's instruction.md + tools
      │                     ├─ exec("…") / write_file/read_file → operate the container
      │                     ├─ done()  (score terminal) → seal + run the verifier → 0/1 reward
      └──tool calls────────▶└─ (terminate = abort: bail out with no score)
                                                          episode scored → ./hgym_logs/frontier_bench.jsonl
                                                                             │
                            hgym.result_from_trace(...) ◀───────────────────┘
```

## The flow: `exec`/`write_file` → `done`

The agent solves the task inside its Docker container, then ends the run with a single terminal
call:

1. Read the task with **`describe`** (inputs under `/app/inputs/`, the exact outputs to
   produce), then operate the container with **`exec`** (shell commands), **`read_file`**, and
   **`write_file`** — write the required outputs under `/app`.
2. Call **`done`** — the **`score` terminal**. This **seals and ends the episode**: hgym runs the
   env's `finalize` hook, which collects the task's declared output artifacts off the container's
   final state, runs the task's verifier in a **separate** container, and commits the recorded
   **0/1 reward** as the terminal result. `done` is one-shot — every later call (including
   `terminate`) is tombstoned, so make sure the outputs are complete before calling it.

`terminate` is the alternative **`abort`** terminal: a no-score bail-out for ending the episode
*without* grading (reward 0). A successful run ends with `done`, not `terminate`.

The score is the task verifier's own verdict over the container end-state — never the
transcript — computed server-side in `finalize` and handed to hgym as core-owned terminal
evidence, so the transcript cannot influence it.

## Prerequisites

The project is pinned to **Python 3.12**:

```bash
uv sync   # builds the 3.12 .venv; the frontier_bench extra is in the dev group, so `uv run` just works
```

Two other requirements:

- **Docker.** Frontier-Bench tasks are Docker-backed: the env builds+runs the task's
  `environment/Dockerfile` as the agent's container and the task's `tests/Dockerfile` as a
  separate verifier. `docker info` must succeed. `run.py` runs a **preflight** that fails loud
  with the exact fix if Docker is down or the env won't build — rather than letting Claude
  connect to a crashed, toolless server and give up.
- **The [`claude`](https://www.anthropic.com/claude-code) CLI** on your `PATH`, with
  credentials (the harness makes model calls, as any `claude -p` run does).

**No API key or data download** for the task itself — the task files are vendored and pinned to
upstream `v0.1.0`; the task image pulls a digest-pinned `python:3.11-slim` base at build time.

## Run it

One command, via the orchestration script (it writes a per-run `.mcp.json`, runs Claude Code,
then prints the score read back off the trace):

```bash
uv run python examples/frontier_bench/claude_code/run.py --task 0

# print Claude Code's turn-by-turn tool calls and reasoning as it works:
uv run python examples/frontier_bench/claude_code/run.py --task 0 --transcript

# pick the model / reasoning effort (defaults: claude-sonnet-5 / medium):
uv run python examples/frontier_bench/claude_code/run.py --task 0 --model opus --effort high
```

`--task` selects the vendored task (only `0` — `fin-saccr-rwa` — in this first slice).
`--effort` sets Claude Code's reasoning-effort level (`low`/`medium`/`high`/`xhigh`/`max`).
`--transcript` renders each `exec`/`write_file`/`done`/`terminate` call and the model's
reasoning; without it you just get the final result and the score (`reward`, `success`,
`verified`).

## Drive it by hand

Serve the env yourself and read the score — no orchestration script. Save a minimal `.mcp.json`
(this is what `run.py` generates — the server runs under this project's interpreter):

```json
{
  "mcpServers": {
    "frontier_bench": {
      "command": "python",
      "args": ["-m", "hgym.cli", "serve", "frontier_bench",
               "--task", "0", "--trace", "./hgym_logs/frontier_bench.jsonl"]
    }
  }
}
```

Then let Claude Code drive it:

```bash
claude -p "Solve the Frontier-Bench task. Call describe, operate the container with exec/read_file/write_file to produce the required outputs, then call done once to finish (it seals and scores the episode)." \
  --mcp-config ./.mcp.json \
  --strict-mcp-config \
  --allowedTools "mcp__frontier_bench__*" \
  --disallowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,NotebookEdit,BashOutput,KillShell" \
  --permission-mode dontAsk

# then read the score hgym recorded off the trace:
uv run python -c "import hgym; print(hgym.result_from_trace('hgym_logs/frontier_bench.jsonl'))"
```

Why these flags: `--strict-mcp-config` isolates the session to this one server;
`--allowedTools "mcp__frontier_bench__*"` pre-approves the env's tools (`exec`, `read_file`,
`write_file`, `done`, `describe`, `terminate`); `--permission-mode dontAsk` runs
non-interactively by **denying** anything not pre-allowed; and the `--disallowedTools` list
removes the built-in tools so the agent can't take untraced side-channel actions on the host —
its only shell is the served `exec` tool, which runs **inside the task container**. We do **not**
use `--tools ""`: in current Claude Code that also strips the MCP tools.

## Swapping the harness

Nothing here is Claude-specific. Point any MCP-speaking harness at the same `hgym serve
frontier_bench` server and the trace/score path is unchanged:

- **Codex:** add the `frontier_bench` server to its MCP config, same `hgym serve` command.
- **pi:** register `hgym serve` in its MCP registry; the reserved `terminate` tool maps to pi's
  native `execute() -> { terminate: true }`.
- **Hermes:** add it under `hermes tools` / its MCP server config.

Hold `(env, task)` fixed, swap the harness, and the delta in `hgym_logs/*.jsonl` is
attributable to the harness.
