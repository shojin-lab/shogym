# Claude Code quickstart

Point the `claude` CLI you already use at a **stream of hgym tasks**. The agent pulls a task,
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

- The `claude` CLI on `PATH` (`claude --version`).
- Credentials for it: `ANTHROPIC_API_KEY`, or an OAuth session from `claude /login`.
- [uv](https://docs.astral.sh/uv/), for the pinned Python 3.12 venv. `uv sync` at the repo root
  installs hgym with every env extra (the default dev group), which is what the default env
  below needs. On its first run `automationbench` also fetches its pinned upstream source into
  `~/.cache/hgym` once; after that it is fully offline and needs no key.

## Run it

```bash
cd examples/quickstarts/claude_code

# 1. install (from anywhere in the repo)
uv sync

# 2. play the stream
#   --mcp-config .mcp.json      -> spawns serve.py (server key "hgym", so its tools are mcp__hgym__*)
#   --strict-mcp-config         -> only this config; your own MCP servers stay out of the run
#   --allowedTools mcp__hgym__* -> pre-approve the stream's tools so nothing stops to ask
#   --permission-mode dontAsk   -> never block on a permission prompt
#   --model / --effort           -> pinned and cheap for a first run
#   --output-format stream-json --verbose -> watch the tool calls go by
claude -p "$(cat PROMPT.txt)" \
    --mcp-config .mcp.json --strict-mcp-config \
    --allowedTools 'mcp__hgym__*' \
    --permission-mode dontAsk \
    --model sonnet --effort low \
    --output-format stream-json --verbose

# 3. read the scores
uv run python results.py
```

Claude Code keeps its built-ins (Bash, Read, web) alongside the stream's tools, which is the
right default for a quickstart. For a run whose scores you want to defend, add a deny list so the
served tools are the only affordances (an agent with `Read` can find the env's task definitions
on disk):

```bash
--disallowedTools Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Agent,Task
```

Do NOT use `--tools ""` to strip built-ins: it strips the MCP tools too.


## Swap the env

Either set it for one run, without touching a tracked file:

```bash
HGYM_ENV=wordle_v1 HGYM_TASKS=0,1 <the command above>
```

or change the default, which is one line in `serve.py`:

```python
ENV = os.environ.get("HGYM_ENV") or "automationbench"   # "wordle_v1", "hle", "yc_bench", ...
```

`HGYM_ENV` wins when it is set, so the environment variable is the one to reach for while you are
trying envs out and the literal is the one to edit when you have picked.

Nothing else changes. Not `.mcp.json`, not the prompt, not `results.py`, not the command above.
`TASKS = [0, 1, 2]` is the other constant, and the only thing to check when you swap: task index
ranges differ per env, and some envs need their extra installed and a key exported (see
`src/hgym/envs/<env>/README.md`). `wordle_v1` needs neither and is the cheapest place to start.

## Read the results

```
runs/automationbench-<stamp>  (3 tasks)

  #1   automationbench[0]  sealed        reward=1.0  success=True
  #2   automationbench[1]  sealed        reward=0.0  success=False
  #3   automationbench[2]  broker_abort  no score

  scored   2/3
  reward   mean 0.500
```

One row per dispensed task, and the columns are the record's own fields:

- **`closure`** says how the task ended: `sealed` (the agent called the env's score terminal, or
  spent its budget), `aborted` (the agent called `terminate`), `drained` (the stream forced the
  terminal because the agent moved on or the run ended), `timeout`, `finalize_error`, or
  `broker_abort` (dispensed and never sealed, i.e. the server was killed holding it).
- **`reward`** and **`success`** are whatever the env published at episode level under those
  names. `None` means the env published no such field, not zero; some envs report their verdict
  under other names (`partial_credit`, `check_answer`). `results.py --verbose` prints every value
  the env published, verbatim.
- The last three closures carry **no score at all**, so an infrastructure failure can never be
  averaged in as a zero. `results.py` reports `scored N/M` for exactly that reason.

The rows are JSONL on disk under `runs/<env>-<stamp>/`, so any reader will do:

```bash
uv run python -c "
from hgym.serve.stream import read_results
for r in read_results('runs/automationbench-<stamp>'):
    print(r.position, r.env, r.task_idx, r.closure, r.score and r.score.reward)"
```

`results.py` adds one thing over `read_results`: it also calls `reconcile()`, which pairs
`dispenses.jsonl` against `results.jsonl` and reports any task that went out and never came back
as a `broker_abort`. A clean run has none. A `docker rm -f` mid-run has one.

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

Concurrency is available too: `max_in_flight=N` serves several tasks at once, each named by a
lease (above 1, the served tools gain a `lease` argument).


## Files

| File | What it is |
|---|---|
| `serve.py` | the MCP endpoint Claude Code spawns: builds the `TaskStream`, serves it over stdio |
| `.mcp.json` | tells Claude Code how to spawn it, under the server key `hgym` |
| `PROMPT.txt` | the loop the agent runs: `get_task`, play, end, repeat |
| `results.py` | reads the durable rows back out after the run |
| `runs/` | one directory per run (`results.jsonl` + `dispenses.jsonl`). Gitignored. |

Knobs worth knowing, all in `serve.py`: `feedback=` (the `Immediate()` default above;
`Never()` or `EvalStream` for evaluation), `deadline=` bounds each task in seconds (an expired
task is recorded unscored), `max_in_flight=` serves several tasks concurrently, and
`resume=True` continues an interrupted run's directory instead of refusing it.

## The other quickstarts

`examples/quickstarts/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. `codex/`, `pi/` and `hermes/` follow this one and
demonstrate the same three moves in their own idiom.
