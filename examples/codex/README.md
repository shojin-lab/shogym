# Codex quickstart

Point the `codex` CLI you already use at a **stream of shogym tasks**. The agent pulls a task,
plays it with the env's own tools, pulls the next one, and stops when the queue is empty. The
server scores every task as it ends; you read the scores back afterwards, out of a durable record
the agent never sees.

Three moves, and the whole quickstart is these three:

1. **A stream of tasks.** `serve.py` publishes one MCP endpoint for a whole queue: `get_task`
   plus the env's native tools, routed to whichever task is live.
2. **One variable swaps the env.** `ENV = "automationbench"` at the top of `serve.py`. That line
   is the entire migration to any other env in the catalogue.
3. **The server keeps the score.** Every task is scored server-side into a durable record that
   `results.py` reads back. The agent hears its score as it goes (the practice default); the
   record is what you trust.

## Prerequisites

- The `codex` CLI on `PATH` (`codex --version`; written against `codex-cli 0.145.0`).
- Credentials for it: a ChatGPT sign-in (`codex login`), or `OPENAI_API_KEY` exported for an
  API-billed run. The model below works with either; `codex debug models` lists what your
  account can reach, and a few of them are sign-in only (`supported_in_api: false`).
- Nothing else. The command below declares the stream inline, so it needs no trusted project
  and writes nothing to your Codex config. (`.codex/config.toml` here holds the same server for
  when you would rather not repeat the flags; it loads only for a trusted project, and it fails
  silently when that is not the case. See "How the server gets attached".)
- [uv](https://docs.astral.sh/uv/), for the pinned Python 3.12 venv. `uv sync` at the repo root
  installs shogym with every env extra (the default dev group), which is what the default env
  below needs. On its first run `automationbench` also fetches its pinned upstream source into
  `~/.cache/shogym` once; after that it is fully offline and needs no key.

## Run it

```bash
cd examples/codex

# 1. install (from anywhere in the repo)
uv sync

# 2. play the stream
#   exec ... -                  -> non-interactive, and `-` reads the prompt from stdin
#   --json                      -> newline-delimited events; watch the tool calls go by
#   -m / model_reasoning_effort -> pinned and cheap for a first run
#   --sandbox read-only         -> exec's default, spelled out
#   -c mcp_servers.shogym.*       -> the stream, declared inline. Needs no trusted project and
#                                  writes nothing anywhere. The same server is checked in at
#                                  .codex/config.toml if you would rather not repeat the flags
#                                  -- see "How the server gets attached"
codex exec --json \
    -m gpt-5.6-terra \
    -c model_reasoning_effort="low" \
    --sandbox read-only \
    -c 'mcp_servers.shogym.command="uv"' \
    -c 'mcp_servers.shogym.args=["run","python","serve.py"]' \
    -c 'mcp_servers.shogym.default_tools_approval_mode="approve"' \
    -c 'mcp_servers.shogym.startup_timeout_sec=60' \
    -c 'mcp_servers.shogym.tool_timeout_sec=900' \
    - < PROMPT.txt

# 3. read the scores
uv run python results.py
```

## Swap the env

Either set it for one run, without touching a tracked file:

```bash
SHOGYM_ENV=wordle_v1 SHOGYM_TASKS=0,1 <the command above>
```

or change the default, which is one line in `serve.py`:

```python
ENV = os.environ.get("SHOGYM_ENV") or "automationbench"   # "wordle_v1", "hle", "yc_bench", ...
```

`SHOGYM_ENV` wins when it is set, so the environment variable is the one to reach for while you are
trying envs out and the literal is the one to edit when you have picked.

Nothing else changes. Not `.codex/config.toml`, not the prompt, not `results.py`, not the command
above. `TASKS = [0, 1, 2]` is the other constant, and the only thing to check when you swap: task
index ranges differ per env, and some envs need their extra installed and a key exported (see
`src/shogym/envs/<env>/README.md`). `wordle_v1` needs neither and is the cheapest place to start.

## Read the results

A real run of this quickstart, with `ENV = "wordle_v1"` and `TASKS = [0, 1]`:

```
runs/wordle_v1-20260806T053845Z  (2 tasks)

  #1   wordle_v1[0]  sealed         reward=0.6  success=None
  #2   wordle_v1[1]  aborted        reward=1.0  success=None

  scored   2/2
  reward   mean 0.800
```

One row per dispensed task, and the columns are the record's own fields:

- **`closure`** says how the task ended: `sealed` (the agent called the env's score terminal, or
  spent its budget), `aborted` (the agent called `terminate`), `drained` (the stream forced the
  terminal because the agent moved on or the run ended), `timeout`, `finalize_error`, or
  `broker_abort` (dispensed and never sealed, i.e. the server was killed holding it).
- **`reward`** and **`success`** are whatever the env published at episode level under those
  names. `None` means the env published no such field, not zero, and that is exactly what the
  run above shows: wordle reports its verdict as `partial_credit` and `check_answer`, so
  `success` is empty while `reward` is real. `results.py --verbose` prints every value the env
  published, verbatim.
- The last three closures carry **no score at all**, so an infrastructure failure can never be
  averaged in as a zero. `results.py` reports `scored N/M` for exactly that reason.

The rows are JSONL on disk under `runs/<env>-<stamp>/`, so any reader will do:

```bash
uv run python -c "
from shogym.serve.stream import read_results
for r in read_results('runs/wordle_v1-<stamp>'):
    print(r.position, r.env, r.task_idx, r.closure, r.score and r.score.reward)"
```

`results.py` adds one thing over `read_results`: it also calls `reconcile()`, which pairs
`dispenses.jsonl` against `results.jsonl` and reports any task that went out and never came back
as a `broker_abort`. A clean run has none. A killed server mid-run has one.

## How the server gets attached

[Run it](#run-it) declares the server inline, with `-c mcp_servers.shogym.*`. That is the default
here because it needs nothing set up: no trusted project, no file, nothing written anywhere, and
it works the same on a fresh clone and in CI.

The same server is also checked in at `.codex/config.toml`, for when you would rather not repeat
five flags. Codex layers any `.codex/config.toml` it finds between your working directory and the
repo root on top of your user config -- **but only for a project you have trusted**:

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

- **`default_tools_approval_mode = "approve"`** pre-approves the stream's tools. Codex asks
  before running a tool it cannot see is read-only, and `codex exec` has nobody to ask, so
  without this line every call comes back `user cancelled MCP tool call` and the agent concludes
  the stream is broken. `"approve"` grants them; `"auto"` is *not* the same thing (it decides per
  tool, and an unannotated tool decides to ask).
- **The timeouts.** Codex allows a server 10s to start and 60s per tool call. A cold venv and an
  env that fetches its upstream source on the first `get_task` both blow through those, so the
  file raises them.

**Trust is the catch, and it fails silently.** An untrusted project's `.codex/config.toml` is
skipped with no error and no warning: the run proceeds, the stream's tools are simply absent, and
the agent reports that it cannot find them. `codex mcp list` from this directory is the check --
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
`--json` stream each call shows up as `{"server": "shogym", "tool": "get_task"}`.

## The server keeps the score

Scoring is server-side and the durable record is the authority. By default this quickstart uses
`feedback=Immediate()`: ending a task returns the env's own published verdict, which is the
useful setting for iterating on an agent, and every row is stamped `feedback_regime="immediate"`
so the records say what regime produced them.

For scores you intend to defend, construct `EvalStream` instead of `TaskStream` in `serve.py`.
It refuses any feedback policy at construction, answers every task ending with one fixed
acknowledgement, stamps rows `feedback_regime="never"`, and refuses to resume a directory whose
rows were produced under any other regime. The agent is never told how it did; the harness
cannot grade itself.

Then take away the affordances the score was not meant to include. Codex keeps its shell, its
subagents and a cached web search alongside the stream's tools, and `--sandbox read-only` does
not change that: read-only still reads. In the run printed above, one guess from losing a Wordle,
the agent went looking for a word list on disk, found the run directory and read it. Each of
those is one flag:

```bash
--disable shell_tool          # no shell, so no reading the env's task definitions off disk
--disable multi_agent         # no spawning subagents to do it instead
-c web_search="disabled"      # the default is "cached", which is not off
```

Re-run the same stream with `--disable shell_tool` and the event log holds nothing but MCP tool
calls, which is the shape an evaluation wants: the served tools are the only affordance.

Concurrency is available too: `max_in_flight=N` serves several tasks at once, each named by a
lease (above 1, the served tools gain a `lease` argument).

## Files

| File | What it is |
|---|---|
| `serve.py` | the MCP endpoint Codex spawns: builds the `TaskStream`, serves it over stdio |
| `.codex/config.toml` | the project layer that declares the server, under the key `shogym` |
| `PROMPT.txt` | the loop the agent runs: `get_task`, play, end, repeat. Piped in on stdin |
| `results.py` | reads the durable rows back out after the run |
| `runs/` | one directory per run (`results.jsonl` + `dispenses.jsonl`). Gitignored. |

Knobs worth knowing, all in `serve.py`: `feedback=` (the `Immediate()` default above;
`Never()` or `EvalStream` for evaluation), `deadline=` bounds each task in seconds (an expired
task is recorded unscored), `max_in_flight=` serves several tasks concurrently, and
`resume=True` continues an interrupted run's directory instead of refusing it.

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. [`claude_code/`](../claude_code/README.md) is the
reference implementation, and `pi/` and `hermes/` demonstrate the same three moves in their own
idiom.
