# Codex quickstart

Point the `codex` CLI you already use at **one shogym task, served over MCP**. The agent pulls
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

- The `codex` CLI on `PATH` (`codex --version`; written against `codex-cli 0.145.0`).
- Credentials for it: a ChatGPT sign-in (`codex login`), or `OPENAI_API_KEY` exported for an
  API-billed run. The model below works with either; `codex debug models` lists what your
  account can reach, and a few of them are sign-in only (`supported_in_api: false`).
- Nothing else. The command below declares the server inline, so it needs no trusted project
  and writes nothing to your Codex config. (`.codex/config.toml` here holds the same server for
  when you would rather not repeat the flags; it loads only for a trusted project, and it fails
  silently when that is not the case. See "How the server gets attached".)
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
cd examples/codex

# 1. install (from anywhere in the repo)
uv sync

# 2. play the task
#   exec ... -                  -> non-interactive, and `-` reads the prompt from stdin
#   --json                      -> newline-delimited events; watch the tool calls go by
#   -m / model_reasoning_effort -> pinned and cheap for a first run
#   --sandbox read-only         -> exec's default, spelled out
#   -c mcp_servers.shogym.*       -> the server, declared inline. Needs no trusted project and
#                                  writes nothing anywhere. The same server is checked in at
#                                  .codex/config.toml if you would rather not repeat the flags
#                                  (see "How the server gets attached")
codex exec --json \
    -m gpt-5.6-terra \
    -c model_reasoning_effort="low" \
    --sandbox read-only \
    -c 'mcp_servers.shogym.command="uv"' \
    -c 'mcp_servers.shogym.args=["run","python","serve.py"]' \
    -c 'mcp_servers.shogym.default_tools_approval_mode="approve"' \
    -c 'mcp_servers.shogym.startup_timeout_sec=180' \
    -c 'mcp_servers.shogym.tool_timeout_sec=900' \
    - < PROMPT.txt
```

## One episode per launch

`serve.py` serves one env at one task and closes the queue before the agent can pull, so `done`
arrives as soon as that task has been sealed and paid out. A run of three tasks is three launches
of `codex exec`, one per task, and not one queue of three:

```bash
for task in 0 1 2; do SHOGYM_TASK=$task <the command above>; done
```

Each launch gets a fresh env, and each writes its own directory under `runs/`.

## The loop the agent runs

`PROMPT.txt` is the whole of it, and the shape on the wire is worth knowing before you read a
`--json` event log:

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

Nothing else changes. Not `.codex/config.toml`, not the prompt, not the command above. `TASK = 0`
is the other constant, and the only thing to check when you swap: task index ranges differ per
env, and some envs need their extra installed and a key exported (see
`src/shogym/envs/<env>/README.md`). `wordle_v1` needs neither and is the cheapest place to start.

## How the server gets attached

[Run it](#run-it) declares the server inline, with `-c mcp_servers.shogym.*`. That is the default
here because it needs nothing set up: no trusted project, no file, nothing written anywhere, and
it works the same on a fresh clone and in CI.

The same server is also checked in at `.codex/config.toml`, for when you would rather not repeat
five flags. Codex layers any `.codex/config.toml` it finds between your working directory and the
repo root on top of your user config, **but only for a project you have trusted**:

```toml
[mcp_servers.shogym]
command = "uv"
args = ["run", "python", "serve.py"]
default_tools_approval_mode = "approve"
```

So the config is in force exactly when you run `codex` from this directory, which is also what
makes the relative `serve.py` resolve, and `~/.codex/config.toml` is never written to: your own
servers, model and plugins are untouched.

Two things in that file are worth knowing before you write your own:

- **`default_tools_approval_mode = "approve"`** pre-approves the served tools. Codex asks
  before running a tool it cannot see is read-only, and `codex exec` has nobody to ask, so
  without this line every call comes back `user cancelled MCP tool call` and the agent concludes
  the server is broken. `"approve"` grants them; `"auto"` is *not* the same thing (it decides per
  tool, and an unannotated tool decides to ask).
- **The timeouts.** Codex allows a server 10s to start and 60s per tool call. Startup builds the
  env and starts the durable service before the server answers, and the call that ends a task
  seals and grades it before it answers, so the file raises both.

**Trust is the catch, and it fails silently.** An untrusted project's `.codex/config.toml` is
skipped with no error and no warning: the run proceeds, the served tools are simply absent, and
the agent reports that it cannot find them. `codex mcp list` from this directory is the check:
`shogym` in the output means the file loaded.

Trust has to be *persisted*; there is no flag for it. `-c projects."<path>".trust_level="trusted"`
does **not** work, because trust is resolved before `-c` overrides apply. Either run `codex`
(interactive, not `exec`) here once and accept the prompt, or add the entry to
`~/.codex/config.toml` yourself:

```toml
[projects."/absolute/path/to/examples/codex"]
trust_level = "trusted"
```

`codex exec` never prompts, so anyone who only ever runs the non-interactive command is never
asked and never told.

There is a third route, `codex mcp add shogym -- uv run python serve.py`, and it is the one to
avoid here: it writes the server into `~/.codex/config.toml`, where it stays and follows you into
every other repo. `codex mcp remove shogym` undoes it.

`shogym` is the server name in all three, and Codex namespaces the server's tools under it: in the
`--json` stream each call shows up as `{"server": "shogym", "tool": "pull"}`.

## The stream keeps the score

The stream seals the attempt, grades it server-side and records the outcome in its own durable
history. Nothing surfaces that score where you can read it: a live generation reports states and
counts rather than scores, and `runs/<env>-<task>-<stamp>/` holds the blobs a presentation
referenced plus a `generation.json` manifest saying which generation lived there. A reader that
reports the score is not part of this protocol yet, so this quickstart does not ship one and you
should not infer a number from the run directory.

Runs recorded by the retired v1 serving path are still readable offline, with
`shogym.serve.v1_runs.read_results` / `read_dispenses` / `reconcile` over their old directories.
Nothing in this quickstart writes those any more.

Then take away the affordances the score was not meant to include. Codex keeps its shell, its
subagents and a cached web search alongside the served tools, and `--sandbox read-only` does
not change that: read-only still reads. An agent one guess from losing a Wordle will go looking
for a word list on disk, find the run directory and read it. Each of those is one flag:

```bash
--disable shell_tool          # no shell, so no reading the env's task definitions off disk
--disable multi_agent         # no spawning subagents to do it instead
-c web_search="disabled"      # the default is "cached", which is not off
```

Re-run the same task with `--disable shell_tool` and the event log holds nothing but MCP tool
calls, which is the shape an evaluation wants: the served tools are the only affordance.

## Files

| File | What it is |
|---|---|
| `serve.py` | the MCP endpoint Codex spawns: one env, one task, served over stdio |
| `.codex/config.toml` | the project layer that declares the server, under the key `shogym` |
| `PROMPT.txt` | the loop the agent runs: `pull`, work, end the task, `pull`, stop on `done`. Piped in on stdin |
| `runs/` | one directory per launch (blobs + `generation.json`). Gitignored. |

Knobs worth knowing, both in `serve.py` and both settable for one launch from the environment:
`SHOGYM_ENV` names the env and `SHOGYM_TASK` names the task index.

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. [`claude_code/`](../claude_code/README.md) is the
reference implementation, and `pi/`, `hermes/` and `prime_agent/` demonstrate the same three moves
in their own idiom.
