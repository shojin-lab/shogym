# Claude Code plays Wordle (through a served hgym env)

The end-to-end demo of the env-as-center design (RFC 008): hgym serves Wordle as an MCP
server, **Claude Code** drives it as an external harness, and hgym scores the result off
the trace. hgym never sees Claude Code's model, prompt, or loop — only the tool calls and
the verifier's feedback.

```
  Claude Code  ──spawns──▶  hgym serve wordle_v1   (stdio MCP server)
  (the harness)             ├─ describe   → the task (rules + tools)
      │                     ├─ guess(word) → G/Y/X result  (+ _meta feedback)
      └──tool calls────────▶└─ terminate() → episode scored → ./hgym_logs/wordle.jsonl
                                                                      │
                            hgym.result_from_trace(...) ◀────────────┘
```

## Prerequisites

- `pip install -e .` in the repo root, so the `hgym` command is on your `PATH`
  (check: `hgym --help`).
- The [`claude`](https://www.anthropic.com/claude-code) CLI on your `PATH`, with an API
  key configured.

## Run it

One command, via the orchestration script (it writes a per-task `.mcp.json`, runs Claude
Code, then prints the score):

```bash
python examples/wordle/claude_code/run.py --task 0

# pick the model (default: claude-sonnet-5; any Claude Code id/alias works)
# and reasoning effort (default: low — Wordle needs little deliberation):
python examples/wordle/claude_code/run.py --task 0 --model opus --effort high

# print Claude Code's turn-by-turn tool calls and reasoning as it plays:
python examples/wordle/claude_code/run.py --task 0 --transcript
```

`--effort` sets Claude Code's reasoning-effort level (`low`/`medium`/`high`/`xhigh`/`max`);
it defaults to `low` here to keep the example fast and cheap. `--transcript` runs Claude
Code with `--output-format stream-json --verbose` and renders each `guess`/`terminate` call
and the model's reasoning; without it you just get the final result and the score.

Or drive it by hand to watch Claude Code work. Using the checked-in
[`.mcp.json`](./.mcp.json) (task 0, trace at `./hgym_logs/wordle.jsonl`):

```bash
claude -p "Play Wordle with the wordle MCP tools: call describe, then guess, then terminate." \
  --mcp-config examples/wordle/claude_code/.mcp.json \
  --allowedTools "mcp__wordle__describe,mcp__wordle__guess,mcp__wordle__terminate"

# then read the score hgym recorded:
python -c "import hgym; print(hgym.result_from_trace('hgym_logs/wordle.jsonl'))"
```

Example output:

```
EvalResult(env='wordle_v1', task='0', terminated=True,
           feedback=[{'name': 'check_answer', 'value': True, 'level': 'episode'},
                     {'name': 'partial_credit', 'value': 1.0, 'level': 'episode'},
                     {'name': 'count_turns', 'value': 3.0, 'level': 'episode'}],
           trace_path='hgym_logs/wordle.jsonl')
```

## Swapping the harness

Nothing here is Claude-specific. Point any MCP-speaking harness at the same server and the
trace/score path is unchanged — that is the whole point of the design:

- **Codex:** add the `wordle` server to its MCP config, same `hgym serve` command.
- **pi:** register `hgym serve` in its MCP registry; the reserved `terminate` tool maps to
  pi's native `execute() -> { terminate: true }`.
- **Hermes:** add it under `hermes tools` / its MCP server config.

Hold `(env, task)` fixed, swap the harness, and the delta in `hgym_logs/*.jsonl` is
attributable to the harness.
