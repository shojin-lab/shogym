# Pi quickstart

Point the `pi` CLI you already use at a **stream of shogym tasks**. The agent pulls a task, plays it
with the env's own tools, pulls the next one, and stops when the queue is empty. The server scores
every task as it ends; you read the scores back afterwards, out of a durable record the agent
never sees.

Three moves, and the whole quickstart is these three:

1. **A stream of tasks.** `serve.py` publishes one MCP endpoint for a whole queue: `get_task`
   plus the env's native tools, routed to whichever task is live.
2. **One variable swaps the env.** `ENV = "automationbench"` at the top of `serve.py`. That line
   is the entire migration to any other env in the catalogue.
3. **The server keeps the score.** Every task is scored server-side into a durable record that
   `results.py` reads back. The agent hears its score as it goes (the practice default); the
   record is what you trust.

## Prerequisites

- The `pi` CLI on `PATH` (`pi --version`):
  `npm install -g --ignore-scripts @earendil-works/pi-coding-agent`.
- **A bridge, because Pi ships no MCP client.** Pi's docs put it flatly: it "intentionally does
  not include built-in MCP, sub-agents, permission popups, plan mode, to-dos, or background
  bash", and you add those as extensions or packages. This quickstart adds `pi-mcp-extension`,
  pinned in `.pi/settings.json`. Read
  [The bridge is a third-party dependency](#the-bridge-is-a-third-party-dependency) before you install it.
- Credentials, and a provider named explicitly. `pi` defaults to `--provider google`, so every
  command below passes `--provider`. `OPENAI_API_KEY` with `--provider openai` is what the
  sample command uses; `ANTHROPIC_API_KEY` with `--provider anthropic` reads the same.
  `pi --list-models` prints exactly what your credentials unlock. An Anthropic **subscription**
  is not a substitute for a key here: a `CLAUDE_CODE_OAUTH_TOKEN` is minted for Anthropic's own
  client and is refused when a third-party harness presents it, and Pi's own Claude Pro/Max
  login bills per token out of extra usage rather than against plan limits. Either way this run
  spends API money, not subscription allowance.
- [uv](https://docs.astral.sh/uv/), for the pinned Python 3.12 venv. `uv sync` at the repo root
  installs shogym with every env extra (the default dev group), which is what the default env
  below needs. On its first run `automationbench` also fetches its pinned upstream source into
  `~/.cache/shogym` once; after that it is fully offline and needs no key.

## Run it

```bash
cd examples/quickstarts/pi

# 1. install (from anywhere in the repo)
uv sync

# 2. install the bridge, project-scoped, at the version .pi/settings.json pins
#    -l keeps it under ./.pi/ so nothing in ~/.pi changes; -a trusts this project's .pi/
pi install npm:pi-mcp-extension@1.5.0 -l -a

# 3. play the stream
#   .pi/mcp.json is found by cwd -> no flag; it spawns serve.py under the server key "shogym",
#                                   so the stream's tools are mcp_shogym_*
#   -p                           -> non-interactive: run the prompt to completion and exit
#   --approve                    -> trust this project's .pi/ for the run (it is code, so Pi asks)
#   --provider / --model         -> not optional: Pi's default provider is google
#   --thinking low               -> cheap for a first run
#   --mode json                  -> watch the tool calls go by (drop it for just the final text)
pi -p "$(cat PROMPT.txt)" \
    --approve \
    --provider openai --model gpt-5.6-terra \
    --thinking low \
    --mode json

# 4. read the scores
uv run python results.py
```

Pi keeps its built-ins (bash, read, edit, write) alongside the stream's tools, which is the right
default for a quickstart. For a run whose scores you want to defend, drop them and keep the
bridge's:

```bash
--no-builtin-tools
```

That flag is the whole answer here: it disables the built-ins and leaves extension-registered
tools enabled, so `mcp_shogym_*` is the entire affordance set and an agent with `read` cannot go
find the env's task definitions on disk. Do NOT reach for `--no-tools` instead: that one strips
everything, the bridge's tools included.

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

Nothing else changes. Not `.pi/mcp.json`, not the prompt, not `results.py`, not the command above.
`TASKS = [0, 1, 2]` is the other constant, and the only thing to check when you swap: task index
ranges differ per env, and some envs need their extra installed and a key exported (see
`src/shogym/envs/<env>/README.md`). `wordle_v1` needs neither and is the cheapest place to start.

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
from shogym.serve.stream import read_results
for r in read_results('runs/automationbench-<stamp>'):
    print(r.position, r.env, r.task_idx, r.closure, r.score and r.score.reward)"
```

`results.py` adds one thing over `read_results`: it also calls `reconcile()`, which pairs
`dispenses.jsonl` against `results.jsonl` and reports any task that went out and never came back
as a `broker_abort`. A clean run has none. A `docker rm -f` mid-run has one.

## The bridge is a third-party dependency

Plainly, because it is the one thing this quickstart asks of you that the other quickstarts do
not: `pi-mcp-extension` is an npm package written by a single maintainer (`irahardianto`, MIT,
[source](https://github.com/irahardianto/pi-mcp-extension)), and installing it runs their code
inside your agent for the whole session. It is roughly 2,200 lines of TypeScript over 7 files and
brings 94 transitive packages with it. `.pi/settings.json` pins it to an exact version rather
than a range, and Pi re-checks the installed version against that pin every time it starts, so
moving to a new release is an edit to a checked-in file rather than something that happens to
you. `npm pack pi-mcp-extension@1.5.0` and read it before you install it.

If you will not take that dependency, Pi's own extension API is the documented alternative. An
in-repo extension under `.pi/extensions/` can open the stdio connection to `serve.py` itself and
call `pi.registerTool()` once per tool the stream publishes, which is the whole of what the
bridge does for this quickstart: one local stdio server, no OAuth, no reconnection schedule, no
paginated discovery, no health checks. Same wire, code you own and review, and you write and
maintain it. The bridge is the shortcut, not the only path.

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
| `serve.py` | the MCP endpoint the bridge spawns: builds the `TaskStream`, serves it over stdio |
| `.pi/mcp.json` | tells the bridge how to spawn it, under the server key `shogym` (hence `mcp_shogym_*`) |
| `.pi/settings.json` | the bridge itself, pinned, project-scoped so `~/.pi` is untouched |
| `PROMPT.txt` | the loop the agent runs: `get_task`, play, end, repeat |
| `results.py` | reads the durable rows back out after the run |
| `runs/` | one directory per run (`results.jsonl` + `dispenses.jsonl`). Gitignored. |

Knobs worth knowing, all in `serve.py`: `feedback=` (the `Immediate()` default above;
`Never()` or `EvalStream` for evaluation), `deadline=` bounds each task in seconds (an expired
task is recorded unscored), `max_in_flight=` serves several tasks concurrently, and
`resume=True` continues an interrupted run's directory instead of refusing it.

Two knobs live on the bridge instead, in `.pi/mcp.json`: `"lifecycle": "eager"` starts the server
at session start, which is what a `-p` run needs (the bridge's own default is `"lazy"`, i.e. wait
for `/mcp:start` in an interactive session), and `"requestTimeoutMs"` bounds a single tool call.

## The other quickstarts

`examples/quickstarts/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. `claude_code/` is the reference implementation; `codex/`
and `hermes/` demonstrate the same three moves in their own idiom.
