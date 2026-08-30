# Claude Code quickstart

Point the `claude` CLI you already use at **one shogym task, served over MCP**. The agent pulls
the work, plays it with the env's own tools, ends it with the tool that ends it, and pulls again
until the stream says it is done. The stream seals and scores the attempt itself, into a record
the agent never sees.

Three moves, and the whole quickstart is these three:

1. **One task, served.** `serve.py` publishes one MCP endpoint: `pull` plus the env's native
   tools, each wrapped so that a call names the attempt it belongs to.
2. **One variable swaps the env.** `ENV = "wordle_v1"` at the top of `serve.py`. That line
   is the entire migration to any other env in the catalogue.
3. **The stream keeps the score.** Sealing is server-side and the score stays in the stream's own
   durable history. The agent is not told it, and neither is anything else.

## Prerequisites

- The `claude` CLI on `PATH` (`claude --version`).
- Credentials for it: `ANTHROPIC_API_KEY`, or an OAuth session from `claude /login`.
- [uv](https://docs.astral.sh/uv/), for the pinned Python 3.12 venv. `uv sync` at the repo root
  installs shogym with every env extra. Outside this repo the install is `pip install shogym` and
  nothing else: the stream's history, replay and timers are Temporal's, and `temporalio` is a
  dependency of the package rather than an extra you have to know to ask for.
- Network, once. The first serve starts an embedded durable service and downloads its binary
  (about 130 MB) into `~/.cache/shogym/temporal/`, and there is nothing to configure: every serve
  after that reuses it. Set `SHOGYM_TEMPORAL_ADDRESS` to use a service you already run instead.
  On its first run `automationbench` also fetches its pinned upstream source into
  `~/.cache/shogym` once; after that it is fully offline and needs no key.

## Run it

```bash
cd examples/claude_code

# 1. install (from anywhere in the repo)
uv sync

# 2. play the task
#   --mcp-config .mcp.json      -> spawns serve.py (server key "shogym", so its tools are mcp__shogym__*)
#   --strict-mcp-config         -> only this config; your own MCP servers stay out of the run
#   --allowedTools mcp__shogym__* -> pre-approve the served tools so nothing stops to ask
#   --permission-mode dontAsk   -> never block on a permission prompt
#   --model / --effort           -> pinned and cheap for a first run
#   --output-format stream-json --verbose -> watch the tool calls go by
claude -p "$(cat PROMPT.txt)" \
    --mcp-config .mcp.json --strict-mcp-config \
    --allowedTools 'mcp__shogym__*' \
    --permission-mode dontAsk \
    --model sonnet --effort low \
    --output-format stream-json --verbose
```

Claude Code keeps its built-ins (Bash, Read, web) alongside the served tools, which is the
right default for a quickstart. For a run whose scores you want to defend, add a deny list so the
served tools are the only affordances (an agent with `Read` can find the env's task definitions
on disk):

```bash
--disallowedTools Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Agent,Task
```

Do NOT use `--tools ""` to strip built-ins: it strips the MCP tools too.

## One episode per launch

`serve.py` serves one env at one task and closes the queue before the agent can pull, so `done`
arrives as soon as that task has been sealed and paid out. A run of three tasks is three launches
of the harness command, one per task, and not one queue of three:

```bash
for task in 0 1 2; do SHOGYM_TASK=$task <the command above>; done
```

Each launch gets a fresh env, and each writes its own directory under `runs/`.

## The loop the agent runs

`PROMPT.txt` is the whole of it, and the shape on the wire is worth knowing before you read a
transcript:

```jsonc
// pull, which takes no arguments
{"protocol_version": 2, "kind": "task", "message_id": "...", "attempt_id": "9f3c...", "body": "..."}

// every env tool, wrapped: the attempt is the routing handle
{"attempt_id": "9f3c...", "arguments": {"word": "crane"}}

// the tool that ends the task, wrapped the same way, answered by the stream and not the env
{"protocol_version": 2, "kind": "seal_ack", "attempt_id": "9f3c...", "submission_digest": "..."}

// pull again
{"protocol_version": 2, "kind": "done", "message_id": "..."}
```

A `wait` record means nothing is ready yet, so pull again shortly. A `seal_reject` means the
terminal's own arguments were malformed; the task is still open, so the agent can correct them and
file again. There is no queue to inspect and no task index anywhere on the wire: a task record
carries an attempt id and a body, and has no field an index or a target could be written into.

## Swap the env

Either set it for one run, without touching a tracked file:

```bash
SHOGYM_ENV=wordle_v1 SHOGYM_TASK=1 <the command above>
```

or change the default, which is one line in `serve.py`:

```python
ENV = os.environ.get("SHOGYM_ENV") or "wordle_v1"   # "wordle_v1", "hle", "yc_bench", ...
```

`SHOGYM_ENV` wins when it is set, so the environment variable is the one to reach for while you are
trying envs out and the literal is the one to edit when you have picked.

Nothing else changes. Not `.mcp.json`, not the prompt, not the command above. `TASK = 0` is the
other constant, and the only thing to check when you swap: task index ranges differ per env, and
some envs need their extra installed and a key exported (see `src/shogym/envs/<env>/README.md`).
`wordle_v1` needs neither and is the cheapest place to start.

## The stream keeps the score

The stream seals the attempt, grades it server-side and records the outcome in its own durable
history. `runs/<env>-<task>-<stamp>/` holds that history, beside the blobs a presentation
referenced and a `generation.json` manifest saying which generation lived there, so the score is
read by asking the history for it:

```bash
shogym results runs/<env>-<task>-<stamp>
```

That prints one row per attempt, with what it filed and what it scored, and leaves the same rows
in the directory as `records.jsonl`. The file is a derived view, rebuilt every time it is asked
for, so the history stays the record and nothing reads the file back as authority.

What the agent is told is the acknowledgement and whatever payload the stream releases against the
attempt, which commits to what was filed and says nothing about how good it was.

Runs recorded by the retired v1 serving path are still readable offline, with
`shogym.serve.v1_runs.read_results` / `read_dispenses` / `reconcile` over their old directories.
Nothing in this quickstart writes those any more.

`SHOGYM_RUNS` moves the run directory somewhere the agent is not working, which is what a run
whose record you intend to defend wants:

```bash
SHOGYM_RUNS=~/somewhere-else/runs <the command above>
```

## Files

| File | What it is |
|---|---|
| `serve.py` | the MCP endpoint Claude Code spawns: one env, one task, served over stdio |
| `.mcp.json` | tells Claude Code how to spawn it, under the server key `shogym` |
| `PROMPT.txt` | the loop the agent runs: `pull`, work, end the task, `pull`, stop on `done` |
| `runs/` | one directory per launch (blobs + `generation.json`). Gitignored. |

Knobs worth knowing, all in `serve.py` and all settable for one launch from the environment:
`SHOGYM_ENV` names the env, `SHOGYM_TASK` names the task index, and `SHOGYM_RUNS` puts `runs/`
somewhere else.

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. [`codex/`](../codex/README.md), `pi/`, `hermes/` and
`prime_agent/` follow this one and demonstrate the same three moves in their own idiom.
