# Hermes quickstart

Point the `hermes` CLI you already use at **one shogym task, served over MCP**. The agent pulls
the work, plays it with the env's own tools, ends it with the tool that ends it, and pulls again
until the stream says it is done. The stream seals and scores the attempt itself, into a record
the agent never sees.

Three moves, and the whole quickstart is these three:

1. **One task, served.** `serve.py` publishes one MCP endpoint: `pull` plus the env's native
   tools, each wrapped so that a call names the attempt it belongs to.
2. **One variable swaps the env.** `ENV = "automationbench"` at the top of `serve.py`. That line
   is the entire migration to any other env in the catalogue.
3. **The stream keeps the score.** Sealing is server-side and the score stays in the stream's own
   durable history. The agent is not told it, and neither is anything else.

## Prerequisites

- The `hermes` CLI on `PATH` (`hermes --version`): see [Installing Hermes](#installing-hermes)
  below, which is less obvious than it looks.
- Credentials for it. See [Providers](#providers) below, which has one trap worth reading.
- [uv](https://docs.astral.sh/uv/), for the pinned Python 3.12 venv. `uv sync` at the repo root
  installs shogym with every env extra and with the `durable` extra, which is what serving needs
  now: the stream's history, replay and timers are Temporal's. Outside this repo that extra is
  `pip install "shogym[durable]"` (or `uv sync --extra durable`); `import shogym` still works
  without it, and only serving does not.
- Network, once. The first serve starts an embedded durable service and downloads its binary
  into `~/.cache/shogym/temporal/`; set `SHOGYM_TEMPORAL_ADDRESS` to use a service you already
  run instead. On its first run `automationbench` also fetches its pinned upstream source into
  `~/.cache/shogym` once; after that it is fully offline and needs no key.

### Installing Hermes

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Or from a clone, if you would rather not pipe a script to a shell. Put the venv **outside** the
checkout: upstream warns that a venv inside the directory the agent operates from can be wiped by
a relative-path command the agent runs against its own source, destroying the runtime mid-session.

```bash
git clone https://github.com/NousResearch/hermes-agent && cd hermes-agent
uv venv ~/.hermes/venvs/hermes --python 3.12
source ~/.hermes/venvs/hermes/bin/activate
uv pip install -e ".[mcp]"
```

**`[mcp]` is not optional here.** Without it Hermes has no MCP client at all and this quickstart
has nothing to connect to.

Two things that will otherwise cost you an afternoon:

- **`pip install hermes-agent` fails by design.** `setup.py` raises on `bdist_wheel` and `sdist`
  ("Building wheels or sdists for hermes-agent is not supported"), so `uv build` fails too. The
  supported paths are the shell installer, the Docker image, Nix, and an editable source install.
- **The pin inside that extra is load-bearing**, not routine hygiene: `mcp==1.28.1`. Hermes probes
  for `mcp.client.streamable_http.streamablehttp_client` at import and silently sets
  `_MCP_HTTP_AVAILABLE = False` when it is missing, so an unpinned `mcp` 2.0.0 leaves stdio working
  while the HTTP transport quietly disappears.

## Run it

```bash
# 1. run from this directory: the server's `serve.py` path is relative, and a stdio MCP
#    server inherits the cwd of the `hermes` process (its config takes no `cwd` key)
cd examples/hermes

# 2. give the quickstart its own Hermes home, so your real ~/.hermes is untouched
#    (see "An isolated HERMES_HOME" below for why this is not optional)
export HERMES_HOME="$PWD/.hermes"
mkdir -p "$HERMES_HOME"
cp config.yaml "$HERMES_HOME/config.yaml"

# 3. install (from anywhere in the repo)
uv sync

# 4. check the wiring: connects, lists tools, disconnects. No model, no spend, no run recorded
hermes mcp test shogym

# 5. play the task
#   -z                    -> one-shot: run the prompt to completion, print only the final text
#   --provider openai-api -> the direct OpenAI API (NOT `openai`; see Providers)
#   --usage-file          -> a JSON token/cost report written even if the run fails
hermes -z "$(cat PROMPT.txt)" \
    --provider openai-api --model gpt-5.6-terra \
    --reasoning low \
    --usage-file usage.json
```

`hermes mcp test shogym` is the cheapest thing in this directory and worth running first. It
spawns `serve.py`, completes the MCP handshake and prints the tool list, which is exactly what
the agent will be handed. With `ENV = "automationbench"` the shape is:

```
  Testing 'shogym'...
  Transport: stdio → uv
  Auth: none
  ✓ Connected
  ✓ Tools discovered: 5

    pull                                 Ask the stream for your next message. Takes no argum...
    api_search                           Search available API endpoints by keyword (BM25 over...
    api_fetch                            Call an API endpoint by its full URL, routing to the...
    base64_encode                        Encode text to base64url, the format Gmail API body ...
    done                                 Finish the task: end the episode and score the final...
```

Expect that connect to take seconds rather than milliseconds: it builds the env and starts the
durable service before the server answers, which is what `connect_timeout: 180.0` in `config.yaml`
is for. The env's reserved abort is not in the list because the stream serves exactly one tool
that can end an attempt, and for this env that is `done`.

Hermes keeps its own toolsets (terminal, file, web, memory, and the rest) alongside the served
tools, which is the right default for a quickstart. For a run whose scores you want to defend,
hand it only what it needs: an agent with the `file` toolset can find the env's task definitions
on disk. Hermes makes that an allowlist rather than a deny list, because **each MCP server is
itself a toolset**, registered as `mcp-<server>` with the bare server name as an alias:

```bash
hermes -z "$(cat PROMPT.txt)" -t shogym        # the served tools, and nothing else
```

`-t/--toolsets` replaces the enabled set outright, so naming only `shogym` turns every built-in
toolset off. What survives is this server plus Hermes's own tool-calling surface:

```
tool_search  tool_describe  tool_call  parallel
mcp__shogym__pull         mcp__shogym__api_search    mcp__shogym__api_fetch
mcp__shogym__base64_encode   mcp__shogym__done
mcp__shogym__get_prompt   mcp__shogym__list_prompts  mcp__shogym__list_resources  mcp__shogym__read_resource
```

`mcp__<server>__<tool>` is the wire name, so the server key in `config.yaml` is what the model
sees. The four `get_prompt`/`list_*`/`read_resource` entries are the MCP protocol surface Hermes
registers per server; shogym publishes neither prompts nor resources, so they return nothing.
`hermes tools disable <toolset>` makes the same choice persistent in this `HERMES_HOME` instead of
per-invocation.

## One episode per launch

`serve.py` serves one env at one task and closes the queue before the agent can pull, so `done`
arrives as soon as that task has been sealed and paid out. A run of three tasks is three launches
of the `hermes` command, one per task, and not one queue of three:

```bash
for task in 0 1 2; do SHOGYM_TASK=$task hermes -z "$(cat PROMPT.txt)" \
    --provider openai-api --model gpt-5.6-terra --reasoning low; done
```

Each launch gets a fresh env, and each writes its own directory under `runs/`.

## The loop the agent runs

`PROMPT.txt` is the whole of it, and the shape on the wire is worth knowing before you read a
session log:

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
ENV = os.environ.get("SHOGYM_ENV") or "automationbench"   # "wordle_v1", "hle", "yc_bench", ...
```

`SHOGYM_ENV` wins when it is set, so the environment variable is the one to reach for while you are
trying envs out and the literal is the one to edit when you have picked.

Nothing else changes. Not `config.yaml`, not the prompt, not the command above. `TASK = 0` is the
other constant, and the only thing to check when you swap: task index ranges differ per env, and
some envs need their extra installed and a key exported (see `src/shogym/envs/<env>/README.md`,
and the `env:` block below for how a key reaches a stdio server). `wordle_v1` needs neither and is
the cheapest place to start.

## An isolated HERMES_HOME

Hermes has no project-local MCP file. `mcp_servers` lives in the single `config.yaml` under
`$HERMES_HOME` (default `~/.hermes`), alongside your sessions, memories, skills and auth. Adding
a quickstart server there would edit your real setup, and removing it afterwards is on you.

So give the quickstart its own home. `HERMES_HOME` is read before anything else, and the
directory is created on demand. Those are steps 1 and 2 of [Run it](#run-it) above. The home is
throwaway and gitignored.

That home starts as one file. Hermes scaffolds the rest on first use (`sessions/`, `logs/`,
`skills/`, `memories/`, `state.db`, a default `SOUL.md`), and all of it is this quickstart's, so
none of your skills, memories or MCP servers are in the run and none of the run's leftovers are
in yours. `rm -rf .hermes` is the uninstall. Unset `HERMES_HOME` and your usual Hermes is back,
untouched.

The checked-in `config.yaml` is the whole configuration and is deliberately short:

```yaml
mcp_servers:
  shogym:
    command: uv
    args: ["run", "python", "serve.py"]
    connect_timeout: 180.0
    enabled: true
```

That is the whole file, and Hermes accepts it as-is: everything else in a Hermes config has a
default. `hermes mcp add shogym --command uv --args run python serve.py --connect-timeout 180`
writes the same block (plus a `_config_version:` line and a commented template of every other
setting), and `hermes mcp list` / `hermes mcp test` / `hermes mcp remove shogym` manage it. Note
that `mcp add` is discovery-first (it connects, lists the tools, and asks which to enable), so
it wants a TTY; copying the file is the headless path. Two things about the block are
load-bearing:

- **The paths are relative, so run `hermes` from this directory.** A stdio MCP server inherits
  the cwd of the `hermes` process and its config takes no `cwd` key. Running Hermes from
  elsewhere means absolute paths in `command`/`args`.
- **Stdio servers get a filtered environment.** Hermes passes a subprocess only `PATH`, `HOME`,
  `USER`, `LANG`, `LC_ALL`, `TERM`, `SHELL`, `TMPDIR` and `XDG_*`, so that an MCP server you
  installed cannot read your keys. Good default, and it means an env that needs a key (`hle`'s
  judge, `browsecomp_plus`) will not see your exported one. Name it in the server's own block:

  ```yaml
      env:
        OPENAI_API_KEY: "${OPENAI_API_KEY}"     # ${VAR} resolves from your env or $HERMES_HOME/.env
  ```

  That is the whole of the answer now. Protocol v2 ships one serving entrypoint and it speaks
  stdio, so the HTTP variant of `serve.py` this quickstart used to offer, which ran in your own
  shell with your own environment, is gone.

## Providers

One trap, and it is silent: **bare `--provider openai` is an alias for OpenRouter**
(`ALIASES = {"openai": "openrouter", ...}` in Hermes's provider table, "route through
aggregator"). It does not fail; it bills a different account through a different endpoint. The
direct OpenAI API is `--provider openai-api`, which resolves to `https://api.openai.com/v1` and
reads `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=sk-...
hermes -z "$(cat PROMPT.txt)" --provider openai-api --model gpt-5.6-terra
```

Native Anthropic is `--provider anthropic` with `ANTHROPIC_API_KEY` (`--provider claude` aliases
to it). Hermes will also read `CLAUDE_CODE_OAUTH_TOKEN` as a fallback, but a Claude subscription
OAuth token is **not** a substitute for an API key: Anthropic restricts subscription credentials
to its own products, and the refusal comes from the API, not from Hermes. `hermes mcp test`
needs no provider at all, which is why it is step 4 above.

## The stream keeps the score

The stream seals the attempt, grades it server-side and records the outcome in its own durable
history. Nothing surfaces that score where you can read it: a live generation reports states and
counts rather than scores, and `runs/<env>-<task>-<stamp>/` holds the blobs a presentation
referenced plus a `generation.json` manifest saying which generation lived there. A reader that
reports the score is not part of this protocol yet, so this quickstart does not ship one and you
should not infer a number from the run directory.

This is the part worth keeping when the harness sounds confident. A Hermes run of the retired
serving path ended with *"Stream exhausted. I completed 2 tasks."* while the record showed one
task played and one pulled and abandoned. Both statements were sincere; only one of them was
written by something other than the agent.

Runs recorded by that retired v1 path are still readable offline, with
`shogym.serve.v1_runs.read_results` / `read_dispenses` / `reconcile` over their old directories.
Nothing in this quickstart writes those any more.

## Files

| File | What it is |
|---|---|
| `serve.py` | the MCP endpoint Hermes spawns: one env, one task, served over stdio |
| `config.yaml` | the `mcp_servers` block, copied into an isolated `$HERMES_HOME`, server key `shogym` |
| `PROMPT.txt` | the loop the agent runs: `pull`, work, end the task, `pull`, stop on `done` |
| `.hermes/` | the throwaway Hermes home this quickstart creates. Gitignored. |
| `runs/` | one directory per launch (blobs + `generation.json`). Gitignored. |

Knobs worth knowing, both in `serve.py` and both settable for one launch from the environment:
`SHOGYM_ENV` names the env and `SHOGYM_TASK` names the task index.

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. `claude_code/` is the reference implementation; this one
and `codex/`, `pi/` and `prime_agent/` demonstrate the same three moves in their own idiom.
