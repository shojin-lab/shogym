# Claude Code plays YC-Bench (through a served hgym env)

The end-to-end demo of the env-as-center design (RFC 008) on the YC-Bench port (issue #32):
hgym serves `yc_bench` as an MCP server, **Claude Code** drives it as an external harness, and
hgym scores the run off the trace using **YC-Bench's own final sim state**. hgym never sees
Claude Code's model, prompt, or loop — only the tool calls and the verifier's feedback.

```
  Claude Code  ──spawns──▶  hgym serve yc_bench          (stdio MCP server)
  (the harness)             ├─ describe            → the rules + this run's seed + tools
      │                     ├─ run_command("yc-bench …") → the CLI's JSON result
      │                     ├─ submit()            → final funds / survival verdict
      └──tool calls────────▶└─ terminate()         → episode scored → ./hgym_logs/yc_bench.jsonl
                                                                             │
                            hgym.result_from_trace(...) ◀───────────────────┘
```

## The flow: `run_command` → `submit` → `terminate`

The agent operates a simulated AI startup for one year — starting with $200,000 — by issuing
YC-Bench CLI commands through the `run_command` tool, then ends the run in two steps:

1. Run the company with **`run_command`**: `yc-bench market browse`, `yc-bench task accept`,
   `yc-bench task assign`, `yc-bench task dispatch`, then `yc-bench sim resume` to advance the
   clock to the next event — repeat until the run ends (bankruptcy or the one-year horizon).
   Every command returns JSON in the tool result.
2. Call **`submit`** — this reads YC-Bench's final sim state (final funds, survival, task
   outcomes) and returns the verdict **in the tool result**.
3. Call **`terminate`** — this ends the hgym episode; hgym's verifier reads the verdict off the
   recorded `submit` step into the terminal feedback.

The agent learns this from the task instructions it reads via `describe`. Scoring credits the
final funds **only from a genuine terminal state** (the horizon, or bankruptcy) — a solvent,
pre-horizon `submit` (stopping early) scores a premature zero, so the agent can't bank the
starting cash without operating the company.

## Prerequisites

The project is pinned to **Python 3.12**, so setup is one command:

```bash
uv sync   # builds the 3.12 .venv with the yc_bench extra (it's in the dev group); `uv run` just works
```

Confirm: `uv run python -c "import yc_bench; print('ok')"`. The only other requirement is the
[`claude`](https://www.anthropic.com/claude-code) CLI on your `PATH`, with credentials.

**No data, no OpenAI key.** Unlike the tau2 example, YC-Bench needs nothing provisioned: it
generates its whole world deterministically from the task's seed and runs its sim **in
process**, so a served episode is fully offline. (The Claude harness itself still makes model
calls and needs Claude credentials, as any `claude -p` run does.) `run.py` runs a **preflight**
that fails loud with the exact fix if `yc_bench` can't be imported or the env won't build —
rather than letting Claude connect to a crashed, toolless server and give up.

## Run it

One command, via the orchestration script (it writes a per-run `.mcp.json`, runs Claude Code,
then prints the score read back off the trace):

```bash
uv run python examples/yc_bench/claude_code/run.py --task 0

# print Claude Code's turn-by-turn tool calls and reasoning as it plays:
uv run python examples/yc_bench/claude_code/run.py --task 0 --transcript

# pick the model / reasoning effort (defaults: claude-sonnet-5 / low):
uv run python examples/yc_bench/claude_code/run.py --task 0 --model opus --effort high
```

`--task` selects the world seed (index into the split; default `0`). `--effort` sets Claude
Code's reasoning-effort level (`low`/`medium`/`high`/`xhigh`/`max`), defaulting to `low` to keep
the example cheap. `--transcript` runs Claude Code with `--output-format stream-json --verbose`
and renders each `run_command`/`submit`/`terminate` call and the model's reasoning; without it
you just get the final result and the score (`reward`, `survived`, `success`, …).

(`uv run` uses the pinned 3.12 `.venv`; a bare `python` only works if that interpreter has the
`yc_bench` extra installed.)

## Drive it by hand

Serve the env yourself and read the score — no orchestration script. First serve `yc_bench`
(task 0, writing a JSONL trace):

Save a minimal `.mcp.json` (this is what `run.py` generates — the server runs under this
project's interpreter so it has the `yc_bench` extra):

```json
{
  "mcpServers": {
    "yc_bench": {
      "command": "python",
      "args": ["-m", "hgym.cli", "serve", "yc_bench",
               "--task", "0", "--trace", "./hgym_logs/yc_bench.jsonl"]
    }
  }
}
```

Then let Claude Code drive it:

```bash
claude -p "You are the YC-Bench CEO. Call describe, run the company with run_command (yc-bench …), then submit, then terminate." \
  --mcp-config ./.mcp.json \
  --strict-mcp-config \
  --allowedTools "mcp__yc_bench__*" \
  --disallowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,NotebookEdit,BashOutput,KillShell" \
  --permission-mode dontAsk

# then read the score hgym recorded off the trace:
uv run python -c "import hgym; print(hgym.result_from_trace('hgym_logs/yc_bench.jsonl'))"
```

Why these flags: `--strict-mcp-config` isolates the session to this one server;
`--allowedTools "mcp__yc_bench__*"` pre-approves the env's tools (`run_command`, `submit`,
`describe`, `terminate`); `--permission-mode dontAsk` runs non-interactively by **denying**
anything not pre-allowed; and `--disallowedTools "Bash,…,WebFetch,WebSearch,…"` explicitly
removes the built-in tools so the agent can't take untraced side-channel actions (shell out to
run `yc-bench` directly, read the sim DB, fetch the web) — the score stays attributable to the
MCP tool surface alone. We do **not** use `--tools ""`: in current Claude Code that also strips
the MCP tools, leaving the agent with an empty toolset.

## Swapping the harness

Nothing here is Claude-specific. Point any MCP-speaking harness at the same `hgym serve
yc_bench` server and the trace/score path is unchanged — that is the whole point of the design:

- **Codex:** add the `yc_bench` server to its MCP config, same `hgym serve` command.
- **pi:** register `hgym serve` in its MCP registry; the reserved `terminate` tool maps to pi's
  native `execute() -> { terminate: true }`.
- **Hermes:** add it under `hermes tools` / its MCP server config.

Hold `(env, task)` fixed, swap the harness, and the delta in `hgym_logs/*.jsonl` is
attributable to the harness.
