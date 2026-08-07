# Prime Agent quickstart

Point the `prime-agent` CLI you already use at a **stream of shogym tasks**. The agent pulls a
task, plays it with the env's own tools, pulls the next one, and stops when the queue is empty.
The server scores every task as it ends; you read the scores back afterwards, out of a durable
record the agent never sees.

Three moves, and the whole quickstart is these three:

1. **A stream of tasks.** `serve.py` publishes one MCP endpoint for a whole queue: `get_task`
   plus the env's native tools, routed to whichever task is live.
2. **One variable swaps the env.** `ENV = "automationbench"` at the top of `serve.py`. That line
   is the entire migration to any other env in the catalogue.
3. **The server keeps the score.** Every task is scored server-side into a durable record that
   `results.py` reads back. The agent hears its score as it goes (the practice default); the
   record is what you trust.

## Prerequisites

- The `prime-agent` CLI on `PATH` (`prime-agent --version`):
  `curl -fsSL https://app.primeintellect.ai/prime-agent/install.sh | sh`. Do **not** reach for
  npm here: Prime Agent's source tree still carries Pi's workspace identity (`package.json` is
  `@earendil-works/pi-coding-agent`, and its `bin` is `pi`), and their quickstart says outright
  that "the inherited npm workspace identifiers in the source tree are not the public install
  path". `npm i -g @earendil-works/pi-coding-agent` is what `examples/pi/` installs,
  and it gets you Pi, which has no MCP client at all.
- Credentials, and a provider named explicitly: `--provider anthropic` with `ANTHROPIC_API_KEY`,
  `--provider openai` with `OPENAI_API_KEY`, and so on. `/login` stores a subscription or an API
  key in `~/.prime/agent/auth.json` instead. `prime-agent model list` prints what your
  credentials unlock.
- [uv](https://docs.astral.sh/uv/), for the pinned Python 3.12 venv. `uv sync` at the repo root
  installs shogym with every env extra (the default dev group), which is what the default env
  below needs. On its first run `automationbench` also fetches its pinned upstream source into
  `~/.cache/shogym` once; after that it is fully offline and needs no key.
- Room for Prime Agent's own kernel: the first session bootstraps `~/.prime/agent/kernel-venv`
  (uv, Python 3.11, `ipykernel`, `prime-agent-runtime`, and a dozen default packages) and needs
  the network once to do it. If you have `PRIME_AGENT_KERNEL_PYTHON` set, unset it for this
  quickstart: with a custom kernel Python, Prime Agent installs nothing, and the skill below is
  disabled with a warning instead of imported.

## Run it

Two shells, because nothing here spawns the server for you.

```bash
# --- shell 1: the stream. Leave it running. ---
cd examples/prime_agent
uv sync                                # install (from anywhere in the repo)
uv run python serve.py                 # serves 127.0.0.1:8973/mcp until you stop it
```

```bash
# --- shell 2: the agent, from THIS directory (settings and skills are cwd-scoped) ---
cd examples/prime_agent
export SHOGYM_MCP_TOKEN=local            # any non-empty value; serve.py never reads it

#   -p                    -> print mode: run the prompt to completion and exit
#   --mode json           -> stream events as they happen. Without it -p prints only the final
#                            text, and a three-task run is several silent minutes
#   --provider / --model  -> not optional: name the provider you have credentials for.
#                            --model takes a pattern; `prime-agent model list` prints yours
#   --thinking low        -> cheap for a first run
prime-agent -p --mode json "$(cat PROMPT.txt)" \
    --provider openai --model gpt-5.6-terra \
    --thinking low

# then read the scores
uv run python results.py
```

`/mcp` inside an interactive session lists the integration and whether it is connected, which is
the cheapest check that the wiring took: with `SHOGYM_MCP_TOKEN` exported it reports `shogym` as
enabled, without it as disabled. `prime-agent --verbose` lists the skills it loaded at startup.

Two things about that command are easy to get wrong:

- **Run it from this directory.** `.prime/agent/settings.json` and `.prime/agent/skills/` are
  project scope, read from the current directory, so `~/.prime/agent/` is never written to and
  your own servers, skills and settings are untouched. Launch from elsewhere and neither file is
  in scope, and the run looks like the skill was never written.
- **Export the token in the shell that launches the agent.** The kernel is spawned with the
  host's environment (`{...process.env, ...}`), so it inherits whatever `prime-agent` had. A
  token exported only in `serve.py`'s shell reaches nothing that needs it.

If the agent stops after one task, `--autonomous` is the host policy for unattended
continuations (`--autonomous-max-continuations`, default 3; `--autonomous-max-turns`, default
12). The prompt asks it to drive itself; the flag is what keeps the host asking.

There is no way to fence the affordances here, and it is worth being blunt about that. The
other quickstarts end this section with a deny list (`--disallowedTools`, `-t shogym`,
`--no-builtin-tools`) so that the served tools are the only thing the agent can reach. Prime
Agent has one built-in tool and the stream lives *inside* it: `--no-builtin-tools` and
`--no-tools` both remove `ipython`, which removes the kernel, which removes the stream. So the
same kernel that plays the task can read the env's task definitions off disk, and no flag
separates the two. For scores you intend to defend, run this in a container that does not have
shogym's source in it.

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

Nothing else changes. Not the settings, not the skill, not the prompt, not `results.py`, not the
command above. `TASKS = [0, 1, 2]` is the other constant, and the only thing to check when you
swap: task index ranges differ per env, and some envs need their extra installed and a key
exported (see `src/shogym/envs/<env>/README.md`; the key reaches this server because you started
it). `wordle_v1` needs neither and is the cheapest place to start.

`PORT` is the third constant, and the one thing two other files name: `.prime/agent/settings.json`
and the skill's `url`. Change it in one place and the run fails to connect; the quickstart's
tests assert all three agree.

## Read the results

Real rows, from the `wordle_v1` stream the verification above played (its directory shown
here as `serve.py` would have named it):

```
runs/wordle_v1-<stamp>  (2 tasks)

  #1   wordle_v1[0]  aborted        reward=0.2  success=None
  #2   wordle_v1[1]  aborted        reward=0.0  success=None

  scored   2/2
  reward   mean 0.100
```

One row per dispensed task, and the columns are the record's own fields:

- **`closure`** says how the task ended: `sealed` (the agent called the env's score terminal, or
  spent its budget), `aborted` (the agent called `terminate`), `drained` (the stream forced the
  terminal because the agent moved on or the run ended), `timeout`, `finalize_error`, or
  `broker_abort` (dispensed and never sealed, i.e. the server was killed holding it).
- **`reward`** and **`success`** are whatever the env published at episode level under those
  names. `None` means the env published no such field, not zero; some envs report their verdict
  under other names (`partial_credit`, `check_answer`, which is what `wordle_v1` does above).
  `results.py --verbose` prints every value the env published, verbatim.
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
as a `broker_abort`. A clean run has none, and so does a `Ctrl-C`, which still drains. A
`kill -9` on `serve.py` mid-task has one.

## The stream is not a tool set

This is the one thing that makes this quickstart different from the other four, and it is worth
reading before you run anything. From Prime Agent's own docs:

> Consistent with Prime Agent's single-tool design, MCP integrations are **not** exposed as new
> agent tools. Each integration is a Python-backed skill that the model imports and calls from
> the IPython kernel.

The model has exactly one built-in tool, `ipython`. An MCP server is reached by writing Python
inside that kernel (`await shogym_stream.get_task()`), and the MCP client is the `mcp` SDK
running kernel-side, not a host-managed tool bridge. Three consequences, all load-bearing:

**HTTP only, and you start it.** The host skips every `mcpServers` entry that is not HTTP
(`if (config.type !== "http") continue; // stdio servers self-manage in Python`, in
`mcp-manager.ts`), and the kernel-side client connects over streamable HTTP. So there is nobody
to spawn a stdio `serve.py`, and a stdio entry is *ignored* rather than rejected: you get an
integration that quietly is not there. `serve.py` therefore serves HTTP and runs in your shell.
Which also solves the key problem the `hermes/` quickstart has a section about: an env that
needs `OPENAI_API_KEY` gets your environment, because it is your process.

**A bearer token, even though nothing checks it.** `McpIntegration._open_session` resolves a
token before every connection and raises `NotEnabled` when there is none; there is no
unauthenticated branch through it. The two ways to have one are browser OAuth (which needs the
server to support OAuth 2.1 dynamic client registration, which a local script does not) and a
static `bearerTokenEnvVar`. So this quickstart declares `SHOGYM_MCP_TOKEN`, you export any non-empty
value, the kernel sends it as `Authorization: Bearer ...`, and `serve.py` never looks at it. One
trap comes with it: `NotEnabled`'s message says to run `/mcp login shogym`, which for a
bearer-only server reports "Unknown MCP integration". `SKILL.md` contradicts that message on
purpose, because the model will otherwise relay it.

**A skill package, not a config entry.** `.prime/agent/skills/shogym-stream/` is a Python package
with a `SKILL.md`; Prime Agent installs it editable into the kernel venv at session start and
exposes it as `shogym_stream`. It is about forty lines and the class body is three attributes.
Unlike the built-in Linear/Notion integrations it is not auth-gated: a skill you drop in a skills
directory loads whether or not credentials exist, and only fails at call time.

One more thing to expect at the keyboard: every tool answers with a **JSON string**, not a dict.
shogym's tools return text content, and the integration's result parser hands text back verbatim.
`PROMPT.txt` and `SKILL.md` both say to `json.loads` it.

### What was checked, and how

The attachment above was run, not inferred: the skill's own client code (Prime Agent's real
`rlm.McpIntegration`, installed from `prime-agent-runtime`) was pointed at a live `serve.py` on
`wordle_v1`, and it discovered the tools, played two tasks and got its verdict back:

```
NO-TOKEN: NotEnabled -> The 'shogym' integration is not enabled: no credentials found. ...
TOKEN: tools = ['get_task', 'guess', 'queue_info', 'terminate']
get_task: type=str  ->  keys=['budget', 'env', 'instructions', 'tools']
terminate: type=str ->  keys=['content', 'feedback', 'hint', 'terminated']
    feedback = [{'name': 'check_answer', 'value': False, 'level': 'episode'}, ...]
```

That covers both branches the client takes: the URL resolved from the host's `mcp.config` (the
settings entry below) and the fallback to the skill's own `url` when the host has no entry.
What it does **not** cover is the host half, Prime Agent installing this package into its
kernel venv and importing it, which was read out of `bootstrap.ts` and `skills.md` rather than
run. If the model reports that `shogym_stream` does not exist, that seam is where to look:
`/reload` rediscovers skill metadata, but a *new* Python-backed skill needs a fresh session so
kernel setup can install it.

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
| `serve.py` | the MCP endpoint **you** run: builds the `TaskStream`, serves it over streamable HTTP |
| `.prime/agent/settings.json` | the `mcpServers` entry: HTTP, the URL, and the name of the token env var |
| `.prime/agent/skills/shogym-stream/` | the integration: `SKILL.md`, `pyproject.toml`, and the `McpIntegration` subclass the kernel imports as `shogym_stream` |
| `PROMPT.txt` | the loop the agent runs: `get_task`, play, end, repeat |
| `results.py` | reads the durable rows back out after the run |
| `runs/` | one directory per run (`results.jsonl` + `dispenses.jsonl`). Gitignored. |

The settings entry and the skill overlap on purpose, and it is fair to ask what each buys. The
skill can connect on its own: when the host has no entry for the server, `mcp.config` fails and
the client falls back to the `url` on the class. The entry is what makes the host's answer
authoritative instead, so the URL can move without editing the package, and what puts `shogym`
in `/mcp` with a connection status. Delete it and the run still works; delete the skill and
there is nothing to import.

Knobs worth knowing, all in `serve.py`: `feedback=` (the `Immediate()` default above;
`Never()` or `EvalStream` for evaluation), `deadline=` bounds each task in seconds (an expired
task is recorded unscored), `max_in_flight=` serves several tasks concurrently, and
`resume=True` continues an interrupted run's directory instead of refusing it.

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. `claude_code/` is the reference implementation;
`codex/`, `pi/` and `hermes/` demonstrate the same three moves in their own idiom.
