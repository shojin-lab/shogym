# Prime Agent quickstart

Point the `prime-agent` CLI you already use at **one shogym task, served over MCP**. The agent
pulls the work, plays it with the env's own tools, ends it with the tool that ends it, and pulls
again until the stream says it is done. The stream seals and scores the attempt itself, into a
record the agent never sees.

Three moves, and the whole quickstart is these three:

1. **One task, served.** `serve.py` publishes one MCP endpoint: `pull` plus the env's native
   tools, each wrapped so that a call names the attempt it belongs to.
2. **One variable swaps the env.** `ENV = "wordle_v1"` at the top of `serve.py`. That line
   is the entire migration to any other env in the catalogue.
3. **The stream keeps the score.** Sealing is server-side and the score stays in the stream's own
   durable history. The agent is not told it, and neither is anything else.

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
  installs shogym with every env extra. Outside this repo the install is `pip install shogym` and
  nothing else: the stream's history, replay and timers are Temporal's, and `temporalio` is a
  dependency of the package rather than an extra you have to know to ask for.
- Network, once. The first serve starts an embedded durable service and downloads its binary
  (about 130 MB) into `~/.cache/shogym/temporal/`, and there is nothing to configure: every serve
  after that reuses it. Set `SHOGYM_TEMPORAL_ADDRESS` to use a service you already run instead.
  On its first run `automationbench` also fetches its pinned upstream source into
  `~/.cache/shogym` once; after that it is fully offline and needs no key.
- Room for Prime Agent's own kernel: the first session bootstraps `~/.prime/agent/kernel-venv`
  (uv, Python 3.11, `ipykernel`, `prime-agent-runtime`, and a dozen default packages) and needs
  the network once to do it. If you have `PRIME_AGENT_KERNEL_PYTHON` set, unset it for this
  quickstart: with a custom kernel Python, Prime Agent installs nothing, and the skill below is
  disabled with a warning instead of imported.

## Run it

Two shells, because nothing here spawns the server for you.

```bash
# --- shell 1: the server. Leave it running. ---
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
#                            text, and a task is several silent minutes
#   --provider / --model  -> not optional: name the provider you have credentials for.
#                            --model takes a pattern; `prime-agent model list` prints yours
#   --thinking low        -> cheap for a first run
prime-agent -p --mode json "$(cat PROMPT.txt)" \
    --provider openai --model gpt-5.6-terra \
    --thinking low
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

If the agent stops before it has pulled a `done`, `--autonomous` is the host policy for unattended
continuations (`--autonomous-max-continuations`, default 3; `--autonomous-max-turns`, default
12). The prompt asks it to drive itself; the flag is what keeps the host asking.

There is no way to fence the affordances here. The other quickstarts end this section with a
deny list (`--disallowedTools`, `-t shogym`, `--no-builtin-tools`) so that the served tools are
the only thing the agent can reach. Prime Agent has one built-in tool and the server lives
*inside* it: `--no-builtin-tools` and `--no-tools` both remove `ipython`, which removes the
kernel, which removes the connection. So the same kernel that plays the task can read the env's
task definitions off disk, and no flag separates the two. For scores you intend to defend, run
this in a container that does not have shogym's source in it.

## One episode per launch

`serve.py` serves one env at one task and closes the queue before the agent can pull, so `done`
arrives as soon as that task has been sealed and paid out. A run of three tasks is three of these
pairs of shells, one per task, and not one queue of three: stop the server, set `SHOGYM_TASK`,
start it again.

```bash
SHOGYM_TASK=1 uv run python serve.py
```

Each launch gets a fresh env, and each writes its own directory under `runs/`.

## The loop the agent runs

`PROMPT.txt` and `SKILL.md` are the whole of it. Every call is Python inside the kernel, and every
answer is a JSON string. With `ENV = "wordle_v1"`, whose tools are `guess` and `terminate`:

```python
import json

record = json.loads(await shogym_stream.pull())          # {"kind": "task", "attempt_id": ..., "body": ...}
attempt = record["attempt_id"]

played = json.loads(await shogym_stream.guess(attempt_id=attempt, arguments={"word": "crane"}))
ack = json.loads(await shogym_stream.terminate(attempt_id=attempt, arguments={}))  # {"kind": "seal_ack", ...}

json.loads(await shogym_stream.pull())                    # eventually {"kind": "done", ...}
```

A `wait` record means nothing is ready yet, so pull again shortly. A `seal_reject` means the
terminal's own arguments were malformed; the task is still open, so the agent can correct them and
file again. There is no queue to inspect and no task index anywhere on the wire: a task record
carries an attempt id and a body, and has no field an index or a target could be written into.

## Swap the env

Either set it for one run, without touching a tracked file:

```bash
SHOGYM_ENV=wordle_v1 SHOGYM_TASK=1 uv run python serve.py
```

or change the default, which is one line in `serve.py`:

```python
ENV = os.environ.get("SHOGYM_ENV") or "wordle_v1"   # "wordle_v1", "hle", "yc_bench", ...
```

`SHOGYM_ENV` wins when it is set, so the environment variable is the one to reach for while you are
trying envs out and the literal is the one to edit when you have picked.

Nothing else changes. Not the settings, not the skill, not the prompt, not the command above.
`TASK = 0` is the other constant, and the only thing to check when you swap: task index ranges
differ per env, and some envs need their extra installed and a key exported (see
`src/shogym/envs/<env>/README.md`; the key reaches this server because you started it).
`wordle_v1` needs neither and is the cheapest place to start.

`PORT` is the third constant, and the one thing two other files name: `.prime/agent/settings.json`
and the skill's `url`. Change it in one place and the run fails to connect; the quickstart's
tests assert all three agree.

## The server is not a tool set

This is the one thing that makes this quickstart different from the other four, and it is worth
reading before you run anything. From Prime Agent's own docs:

> Consistent with Prime Agent's single-tool design, MCP integrations are **not** exposed as new
> agent tools. Each integration is a Python-backed skill that the model imports and calls from
> the IPython kernel.

The model has exactly one built-in tool, `ipython`. An MCP server is reached by writing Python
inside that kernel (`await shogym_stream.pull()`), and the MCP client is the `mcp` SDK running
kernel-side, not a host-managed tool bridge. Three consequences, all load-bearing:

**HTTP only, and you start it.** The host skips every `mcpServers` entry that is not HTTP
(`if (config.type !== "http") continue; // stdio servers self-manage in Python`, in
`mcp-manager.ts`), and the kernel-side client connects over streamable HTTP. So there is nobody
to spawn a stdio `serve.py`, and a stdio entry is *ignored* rather than rejected: you get an
integration that quietly is not there. Protocol v2 ships one serving entrypoint, `run_stdio_v2`,
and it speaks stdio, so this `serve.py` is that function with the transport swapped: the same
episode, the same worker, the same generation, over streamable HTTP. It is the only quickstart
carrying that duplication, and a change to the generation's lifecycle has to be made here too.
Running in your own shell also solves the key problem the `hermes/` quickstart has a section
about: an env that needs `OPENAI_API_KEY` gets your environment, because it is your process.

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

The skill's own client code (Prime Agent's real `rlm.McpIntegration`, installed from
`prime-agent-runtime`) was pointed at a live `serve.py` and made to discover the tools, play
tasks and read what came back. That run predates protocol v2, so the tool names it discovered
were the retired serving path's; what it establishes is the client half, which is unchanged:
`NotEnabled` without a token, a working connection with one, the URL resolved from the host's
`mcp.config` and the fallback to the skill's own `url` when the host has no entry, and every
result arriving as a `str` to be parsed.

What it does **not** cover is the host half, Prime Agent installing this package into its kernel
venv and importing it, which was read out of `bootstrap.ts` and `skills.md` rather than run. If
the model reports that `shogym_stream` does not exist, that seam is where to look: `/reload`
rediscovers skill metadata, but a *new* Python-backed skill needs a fresh session so kernel setup
can install it.

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

## Files

| File | What it is |
|---|---|
| `serve.py` | the MCP endpoint **you** run: one env, one task, served over streamable HTTP |
| `.prime/agent/settings.json` | the `mcpServers` entry: HTTP, the URL, and the name of the token env var |
| `.prime/agent/skills/shogym-stream/` | the integration: `SKILL.md`, `pyproject.toml`, and the `McpIntegration` subclass the kernel imports as `shogym_stream` |
| `PROMPT.txt` | the loop the agent runs: `pull`, work, end the task, `pull`, stop on `done` |
| `runs/` | one directory per launch (blobs + `generation.json`). Gitignored. |

The settings entry and the skill overlap on purpose, and it is fair to ask what each buys. The
skill can connect on its own: when the host has no entry for the server, `mcp.config` fails and
the client falls back to the `url` on the class. The entry is what makes the host's answer
authoritative instead, so the URL can move without editing the package, and what puts `shogym`
in `/mcp` with a connection status. Delete it and the run still works; delete the skill and
there is nothing to import.

Knobs worth knowing, all in `serve.py` and all settable for one launch from the environment:
`SHOGYM_ENV` names the env, `SHOGYM_TASK` names the task index, and `SHOGYM_PORT` moves the
endpoint (which then has to move in the settings entry and the skill too).

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. `claude_code/` is the reference implementation;
`codex/`, `pi/` and `hermes/` demonstrate the same three moves in their own idiom.
