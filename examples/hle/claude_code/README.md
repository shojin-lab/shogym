# Claude Code plays Humanity's Last Exam (through a served hgym env)

The end-to-end demo of the env-as-center design (RFC 008) on the HLE port (issue #33): hgym
serves one HLE question as an MCP server, **Claude Code** answers it as an external harness,
and hgym scores the answer with a **model-graded judge** running server-side. hgym never
sees Claude Code's model, prompt, or loop — only the `submit_answer` call and the verifier's
feedback.

```
  Claude Code  ──spawns──▶  hgym serve hle              (stdio MCP server)
  (the harness)             ├─ describe            → the question + tools
      │                     ├─ submit_answer(a,c)  → LLM judge grades it → verdict
      └──tool calls────────▶└─ terminate()         → episode scored → ./hgym_logs/hle.jsonl
                                                                          │
                            hgym.result_from_trace(...) ◀────────────────┘
```

## The flow: `submit_answer` → `terminate`

HLE is single-turn. The agent does two things, in order:

1. Call **`describe`** to read the question, reason from its own knowledge, then call
   **`submit_answer`** exactly once with its final `answer` and a `confidence` (0–100). The
   env grades it **server-side** — an exact-match fast path, then an LLM judge — and returns
   the verdict in the tool result.
2. Call **`terminate`** — this ends the hgym episode; hgym's verifier reads the verdict off
   the recorded `submit_answer` step into the terminal feedback (`correct` +
   `calibration_error`).

The agent learns this from the question it reads via `describe`.

## Prerequisites

- **Python 3.12 + the `hle` extra.** `uv sync` builds the 3.12 `.venv` with the extra (it's
  in the dev group), so `uv run` just works. Confirm: `uv run python -c "import datasets, openai; print('ok')"`.
- **`OPENAI_API_KEY`** — the model-graded judge calls an OpenAI model. `export OPENAI_API_KEY=sk-...`
- **Hugging Face access to `cais/hle`.** The dataset is **gated**: accept its terms at
  <https://huggingface.co/datasets/cais/hle> and authenticate (`huggingface-cli login`, or
  set `HF_TOKEN`). It downloads once to `~/.cache/hgym/hle` (honor `HF_HOME` or
  `HGYM_HLE_DATA_DIR` to relocate). Please do not redistribute the dataset.
- The [`claude`](https://www.anthropic.com/claude-code) CLI on your `PATH`, with credentials.

`run.py` runs a **preflight** that fails fast with the exact fix if the extra can't import,
the judge key is missing, or the gated dataset won't load — rather than letting Claude
connect to a crashed, toolless server and give up.

## Run it

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/hle/claude_code/run.py --task 0

# print Claude Code's turn-by-turn reasoning and tool calls as it plays:
uv run python examples/hle/claude_code/run.py --task 0 --transcript

# pick the model / reasoning effort (defaults: claude-sonnet-5 / high):
uv run python examples/hle/claude_code/run.py --task 0 --model opus --effort max
```

The script writes a per-run `.mcp.json`, runs Claude Code, then prints the score
(`correct`, `calibration_error`) read back off the trace.

## Drive it by hand

Using the checked-in [`.mcp.json`](./.mcp.json) (task 0, trace at `./hgym_logs/hle.jsonl`) —
run from the repo root, so `uv run` (which the config uses to launch the server) resolves
this project's `.venv`:

```bash
claude -p "Answer the HLE question: call describe, reason it out, then submit_answer, then terminate." \
  --mcp-config examples/hle/claude_code/.mcp.json \
  --strict-mcp-config \
  --allowedTools "mcp__hle__*" \
  --disallowedTools "WebFetch,WebSearch,Bash,Read,Write,Edit,Glob,Grep,Task,TodoWrite,NotebookEdit,BashOutput,KillShell" \
  --permission-mode dontAsk

# then read the score hgym recorded (via the LLM judge):
uv run python -c "import hgym; print(hgym.result_from_trace('hgym_logs/hle.jsonl'))"
```

Why these flags: `--strict-mcp-config` isolates the session to this one server;
`--allowedTools "mcp__hle__*"` pre-approves the hle tools; `--permission-mode dontAsk` runs
non-interactively by **denying** anything not pre-allowed; and `--disallowedTools "WebFetch,WebSearch,…"`
removes the built-ins so the agent can't **look the answer up** — HLE measures the model's
own reasoning, so the score stays attributable to the MCP tool surface alone. We do **not**
use `--tools ""`: in current Claude Code that also strips the MCP tools, leaving the agent
with an empty toolset.

## Swapping the harness

Nothing here is Claude-specific. Point any MCP-speaking harness at the same `hgym serve hle`
server and the trace/score path is unchanged — hold `(env, task)` fixed, swap the harness,
and the delta in `hgym_logs/hle.jsonl` is attributable to the harness.
