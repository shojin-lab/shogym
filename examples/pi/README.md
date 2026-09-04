# Pi quickstart

Point the `pi` CLI you already use at **one shogym task, served over MCP**. The agent pulls the
work, plays it with the env's own tools, ends it with the tool that ends it, and pulls again until
the stream says it is done. The stream seals and scores the attempt itself, and tells the agent
what it scored.

Three moves, and the whole quickstart is these three:

1. **One task, served.** `serve.py` publishes one MCP endpoint: `pull` plus the env's native
   tools, each wrapped so that a call names the attempt it belongs to.
2. **One variable swaps the env.** `ENV = "wordle_v1"` at the top of `serve.py`. That line
   is the entire migration to any other env in the catalogue.
3. **The stream keeps the record and the agent is told its score.** Sealing is server-side and the
   authoritative record stays in the stream's own durable history. The payload released against an
   attempt reports the score honestly by default; concealing it is an experiment arm a run
   registers.

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
cd examples/pi

# 1. install (from anywhere in the repo)
uv sync

# 2. install the bridge, project-scoped, at the version .pi/settings.json pins
#    -l keeps it under ./.pi/ so nothing in ~/.pi changes; -a trusts this project's .pi/
pi install npm:pi-mcp-extension@1.5.0 -l -a

# 3. play the task
#   .pi/mcp.json is found by cwd -> no flag; it spawns serve.py under the server key "shogym",
#                                   so the served tools are mcp_shogym_*
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
```

Pi keeps its built-ins (bash, read, edit, write) alongside the served tools, which is the right
default for a quickstart. For a run whose scores you want to defend, drop them and keep the
bridge's:

```bash
--no-builtin-tools
```

That flag is the whole answer here: it disables the built-ins and leaves extension-registered
tools enabled, so `mcp_shogym_*` is the entire affordance set and an agent with `read` cannot go
find the env's task definitions on disk. Do NOT reach for `--no-tools` instead: that one strips
everything, the bridge's tools included.

## One episode per launch

`serve.py` serves one env at one task and closes the queue before the agent can pull, so `done`
arrives as soon as that task has been sealed and paid out. A run of three tasks is three launches
of the `pi` command, one per task, and not one queue of three:

```bash
for task in 0 1 2; do SHOGYM_TASK=$task <the command above>; done
```

Each launch gets a fresh env, and each writes its own directory under `runs/`.

## The loop the agent runs

`PROMPT.txt` is the whole of it. The bridge renames every served tool `mcp_shogym_<tool>`, so the
prompt asks for `mcp_shogym_pull`; the wire names below are what `serve.py` publishes underneath:

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
carries an attempt id and a body, and has no field an index or a target could be written into. A
run that declares a step budget serves `budget` on every task as well, one number for the whole
run; this quickstart declares none.

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

Nothing else changes. Not `.pi/mcp.json`, not the prompt, not the command above. `TASK = 0` is the
other constant, and the only thing to check when you swap: task index ranges differ per env, and
some envs need their extra installed and a key exported (see `src/shogym/envs/<env>/README.md`).
`wordle_v1` needs neither and is the cheapest place to start.

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
call `pi.registerTool()` once per tool the server publishes, which is the whole of what the
bridge does for this quickstart: one local stdio server, no OAuth, no reconnection schedule, no
paginated discovery, no health checks. Same wire, code you own and review, and you write and
maintain it. The bridge is the shortcut, not the only path.

## The stream keeps the record

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

Runs recorded by the retired v1 serving path are still readable offline, with
`shogym.serve.v1_runs.read_results` / `read_dispenses` / `reconcile` over their old directories.
Nothing in this quickstart writes those any more.

## Files

| File | What it is |
|---|---|
| `serve.py` | the MCP endpoint the bridge spawns: one env, one task, served over stdio |
| `.pi/mcp.json` | tells the bridge how to spawn it, under the server key `shogym` (hence `mcp_shogym_*`) |
| `.pi/settings.json` | the bridge itself, pinned, project-scoped so `~/.pi` is untouched |
| `PROMPT.txt` | the loop the agent runs: `pull`, work, end the task, `pull`, stop on `done` |
| `runs/` | one directory per launch (blobs + `generation.json`). Gitignored. |

Knobs worth knowing, both in `serve.py` and both settable for one launch from the environment:
`SHOGYM_ENV` names the env and `SHOGYM_TASK` names the task index.

Two knobs live on the bridge instead, in `.pi/mcp.json`: `"lifecycle": "eager"` starts the server
at session start, which is what a `-p` run needs (the bridge's own default is `"lazy"`, i.e. wait
for `/mcp:start` in an interactive session), and `"requestTimeoutMs"` bounds a single tool call.
Both matter more than they used to: the server builds the env and starts the durable service
before it answers, and the call that ends a task seals and grades it before it answers.

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. `claude_code/` is the reference implementation; `codex/`,
`hermes/` and `prime_agent/` demonstrate the same three moves in their own idiom.
