# Claude Code plays AutomationBench (through a served hgym env)

The end-to-end demo of the env-as-center design (RFC 008) on the AutomationBench port (issue #42):
hgym serves `automationbench` as an MCP server, **Claude Code** drives it as an external harness,
and hgym scores the run off the trace using **AutomationBench's own rubric**. hgym never sees
Claude Code's model, prompt, or loop — only the tool calls and the verifier's feedback.

```
  Claude Code  ──spawns──▶  hgym serve automationbench     (stdio MCP server)
  (the harness)             ├─ describe        → the task request + the api tools
      │                     ├─ api_search(…)   → matching endpoints (BM25 top-5)
      │                     ├─ api_fetch(…)     → reads / mutates the simulated WorldState
      └──tool calls────────▶└─ done()           → seals + scores the end-state → ./hgym_logs/automationbench.jsonl
                                                                             │
                            hgym.result_from_trace(...) ◀───────────────────┘
```

## The flow: `api_search` → `api_fetch` → `done`

The agent carries out a cross-application workflow over ~47 simulated SaaS apps, then ends it:

1. Discover and act with **`api_search`** + **`api_fetch`**: search for the right endpoint by
   API-native keywords, then call it by its full URL (`api_fetch` is the only tool that changes
   state; `base64_encode` prepares Gmail bodies).
2. Call **`done`** — the `score` terminal. Calling it atomically **seals** the episode, then hgym
   reruns AutomationBench's rubric (`partial_credit` / `task_completed_correctly`) over the sealed
   end-state in one step. There is no separate `terminate` step, and no `done`-then-fix loop.

The agent learns this from the task instructions it reads via `describe`. A premature end (an
explicit `terminate`, or never acting) scores a clean zero; running out of steps (the horizon)
scores whatever partial state the workspace is in.

## Prerequisites

The project is pinned to **Python 3.12**, so setup is one command:

```bash
uv sync   # builds the 3.12 .venv with the automationbench extra (it's in the dev group); `uv run` just works
```

The first run **provisions the pinned upstream source** into `~/.cache/hgym` (a one-time network
fetch of commit `a321764`); after that a served episode is fully offline. Set
`AUTOMATIONBENCH_SRC` to a local checkout for an air-gapped run. The only other requirement is the
[`claude`](https://www.anthropic.com/claude-code) CLI on your `PATH`, with credentials.

**No OpenAI/Zapier key.** The ~47-app world and the rubric run in-process, so scoring is fully
offline and deterministic. (The Claude harness itself still makes model calls and needs Claude
credentials, as any `claude -p` run does.) `run.py` runs a **preflight** that fails loud with the
exact fix if `datasets` can't be imported or the env won't build (including the first-run source
fetch) — rather than letting Claude connect to a crashed, toolless server and give up.

## Run it

One command, via the orchestration script (it writes a per-run `.mcp.json`, runs Claude Code,
then prints the score read back off the trace):

```bash
uv run python examples/automationbench/claude_code/run.py --task 0

# print Claude Code's turn-by-turn tool calls and reasoning as it plays:
uv run python examples/automationbench/claude_code/run.py --task 0 --transcript

# pick the model / reasoning effort (defaults: claude-sonnet-5 / low):
uv run python examples/automationbench/claude_code/run.py --task 0 --model opus --effort high
```

`--task` selects the task index into the default `public` domain set. `--effort` sets Claude
Code's reasoning-effort level (`low`/`medium`/`high`/`xhigh`/`max`), defaulting to `low` to keep
the example cheap. `--transcript` renders each `api_search`/`api_fetch`/`done`/`terminate` call
and the model's reasoning; without it you just get the final result and the score (`reward`,
`partial_credit`, `success`).

## Drive it by hand

Serve the env yourself and read the score — no orchestration script. Save a minimal `.mcp.json`
(this is what `run.py` generates — the server runs under this project's interpreter so it has the
`automationbench` extra):

```json
{
  "mcpServers": {
    "automationbench": {
      "command": "uv",
      "args": ["run", "python", "-m", "hgym.cli", "serve", "automationbench",
               "--task", "0", "--trace", "hgym_logs/automationbench.jsonl"]
    }
  }
}
```

Then let Claude Code drive it:

```bash
claude -p "You are a workflow automation agent. Call describe, carry out the task with api_search/api_fetch, then done (which seals and scores the episode)." \
  --mcp-config ./.mcp.json \
  --strict-mcp-config \
  --allowedTools "mcp__automationbench__*" \
  --disallowedTools "Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Task,TodoWrite,NotebookEdit,BashOutput,KillShell" \
  --permission-mode dontAsk

# then read the score hgym recorded off the trace:
uv run python -c "import hgym; print(hgym.result_from_trace('hgym_logs/automationbench.jsonl'))"
```

Why these flags: `--strict-mcp-config` isolates the session to this one server;
`--allowedTools "mcp__automationbench__*"` pre-approves the env's tools (`api_search`,
`api_fetch`, `base64_encode`, `done`, `describe`, `terminate`); `--permission-mode dontAsk` runs
non-interactively by **denying** anything not pre-allowed; and `--disallowedTools "Bash,…"`
removes the built-in tools so the agent can't take untraced side-channel actions — the score
stays attributable to the MCP tool surface alone. We do **not** use `--tools ""`: in current
Claude Code that also strips the MCP tools, leaving the agent with an empty toolset.

## Swapping the harness

Nothing here is Claude-specific. Point any MCP-speaking harness at the same `hgym serve
automationbench` server and the trace/score path is unchanged:

- **Codex:** add the `automationbench` server to its MCP config, same `hgym serve` command.
- **pi:** register `hgym serve` in its MCP registry; the reserved `terminate` tool maps to pi's
  native `execute() -> { terminate: true }`.
- **Hermes:** add it under `hermes tools` / its MCP server config.

Hold `(env, task)` fixed, swap the harness, and the delta in `hgym_logs/*.jsonl` is attributable
to the harness.
