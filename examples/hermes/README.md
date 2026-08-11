# Hermes quickstart

Point the `hermes` CLI you already use at a **stream of shogym tasks**. The agent pulls a task,
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

- The `hermes` CLI on `PATH` (`hermes --version`): see [Installing Hermes](#installing-hermes)
  below, which is less obvious than it looks.
- Credentials for it. See [Providers](#providers) below, which has one trap worth reading.
- [uv](https://docs.astral.sh/uv/), for the pinned Python 3.12 venv. `uv sync` at the repo root
  installs shogym with every env extra (the default dev group), which is what the default env
  below needs. On its first run `automationbench` also fetches its pinned upstream source into
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
  while the HTTP transport quietly disappears. Symptom: the stdio server below connects and the
  `url:` variant does not.

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

# 4. check the wiring -- connects, lists tools, disconnects. No model, no spend, no run recorded
hermes mcp test shogym

# 5. play the stream
#   -z                    -> one-shot: run the prompt to completion, print only the final text
#   --provider openai-api -> the direct OpenAI API (NOT `openai`; see Providers)
#   --usage-file          -> a JSON token/cost report written even if the run fails
hermes -z "$(cat PROMPT.txt)" \
    --provider openai-api --model gpt-5.6-terra \
    --reasoning low \
    --usage-file usage.json

# 6. read the scores
uv run python results.py
```

`hermes mcp test shogym` is the cheapest thing in this directory and worth running first. It
spawns `serve.py`, completes the MCP handshake and prints the tool list, which is exactly what
the agent will be handed:

```
  Testing 'shogym'...
  Transport: stdio → uv
  Auth: none
  ✓ Connected (12642ms)
  ✓ Tools discovered: 7

    get_task                             Takes the next task off the queue and starts it...
    queue_info                           Reports ``{remaining, consumed, in_flight}`` for the ta...
    terminate                            End the current episode...
    api_search                           Search available API endpoints by keyword (BM25 over en...
    api_fetch                            Call an API endpoint by its full URL, routing to the ap...
    base64_encode                        Encode text to base64url — the format Gmail API body fi...
    done                                 Finish the task: end the episode and score the final wo...
```

(Real output, `ENV = "automationbench"`. The first connect builds the env, hence the seconds;
`connect_timeout: 90.0` in `config.yaml` exists for exactly that.)

Hermes keeps its own toolsets (terminal, file, web, memory, and the rest) alongside the
stream's tools,
which is the right default for a quickstart. For a run whose scores you want to defend, hand it
only what it needs: an agent with the `file` toolset can find the env's task definitions on
disk. Hermes makes that an allowlist rather than a deny list, because **each MCP server is
itself a toolset**, registered as `mcp-<server>` with the bare server name as an alias:

```bash
hermes -z "$(cat PROMPT.txt)" -t shogym        # the stream, and nothing else
```

`-t/--toolsets` replaces the enabled set outright, so naming only `shogym` turns every built-in
toolset off. What survives is the stream plus Hermes's own tool-calling surface, verified by
asking a run under `-t shogym` to list its tools:

```
tool_search  tool_describe  tool_call  parallel
mcp__shogym__get_task     mcp__shogym__queue_info    mcp__shogym__terminate
mcp__shogym__api_search   mcp__shogym__api_fetch     mcp__shogym__base64_encode   mcp__shogym__done
mcp__shogym__get_prompt   mcp__shogym__list_prompts  mcp__shogym__list_resources  mcp__shogym__read_resource
```

(Real output, reflowed; `ENV = "automationbench"`.) `mcp__<server>__<tool>` is the wire name, so
the server key in `config.yaml` is what the model sees. The four `get_prompt`/`list_*`/
`read_resource` entries are the MCP protocol surface Hermes registers per server; shogym publishes
neither prompts nor resources, so they return nothing. `hermes tools disable <toolset>` makes the
same choice persistent in this `HERMES_HOME` instead of per-invocation.

### Over HTTP instead

Hermes speaks streamable HTTP natively, and it is the easier transport when the env needs a
secret or the agent runs somewhere else. Start the server yourself, in your own shell, with your
own environment:

```bash
uv run python serve.py http 8973      # 127.0.0.1:8973/mcp
```

and point the same server key at it. No `command`, no `args`, no `env` block:

```yaml
mcp_servers:
  shogym:
    url: http://127.0.0.1:8973/mcp
    connect_timeout: 60.0
    enabled: true
```

`hermes mcp add shogym --url http://127.0.0.1:8973/mcp` is the `mcp add` form of the same entry.
Hermes also supports `transport: sse` for SSE servers; shogym serves streamable HTTP, so leave it
off.

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

Nothing else changes. Not `config.yaml`, not the prompt, not `results.py`, not the command above.
`TASKS = [0, 1, 2]` is the other constant, and the only thing to check when you swap: task index
ranges differ per env, and some envs need their extra installed and a key exported (see
`src/shogym/envs/<env>/README.md`, and the `env:` block above for how a key reaches a stdio
server). `wordle_v1` needs neither and is the cheapest place to start.

## Read the results

Real output from this quickstart's own smoke run. `ENV = "wordle_v1"`, `TASKS = [0, 1]`,
`--provider openai-api --model gpt-5.4-mini --reasoning low`. `--usage-file` reported 11 API
calls, 21.4k input and 5.2k output tokens for the whole run:

```
runs/wordle_v1-20260806T053309Z  (2 tasks)

  #1   wordle_v1[0]  sealed         reward=1.0  success=None
  #2   wordle_v1[1]  drained        reward=0.0  success=None

  scored   2/2
  reward   mean 0.500
```

Hermes's own final message on that run was *"Stream exhausted. I completed 2 tasks."* The record
says one sealed task and one drained with `count_turns = 0.0`: the agent pulled the second task
and never played it. Both statements are sincere; only one of them was scored by something other
than the agent. That gap is the whole reason the scoring lives in the server.

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
as a `broker_abort`. A clean run has none. A `docker rm -f` mid-run has one.

## An isolated HERMES_HOME

Hermes has no project-local MCP file. `mcp_servers` lives in the single `config.yaml` under
`$HERMES_HOME` (default `~/.hermes`), alongside your sessions, memories, skills and auth. Adding
a quickstart server there would edit your real setup, and removing it afterwards is on you.

So give the quickstart its own home. `HERMES_HOME` is read before anything else, and the
directory is created on demand:

Those are steps 1 and 2 of [Run it](#run-it) above. The home is throwaway and gitignored.

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
    connect_timeout: 90.0
    enabled: true
```

That is the whole file, and Hermes accepts it as-is: everything else in a Hermes config has a
default. `hermes mcp add shogym --command uv --args run python serve.py --connect-timeout 90`
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

  Or use the HTTP transport below, where the server runs in your shell with your environment.

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
needs no provider at all, which is why it is step 2 above.

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
| `serve.py` | the MCP endpoint Hermes spawns: builds the `TaskStream`, serves it over stdio (or HTTP) |
| `config.yaml` | the `mcp_servers` block, copied into an isolated `$HERMES_HOME`, server key `shogym` |
| `PROMPT.txt` | the loop the agent runs: `get_task`, play, end, repeat |
| `results.py` | reads the durable rows back out after the run |
| `.hermes/` | the throwaway Hermes home this quickstart creates. Gitignored. |
| `runs/` | one directory per run (`results.jsonl` + `dispenses.jsonl`). Gitignored. |

Knobs worth knowing, all in `serve.py`: `feedback=` (the `Immediate()` default above;
`Never()` or `EvalStream` for evaluation), `deadline=` bounds each task in seconds (an expired
task is recorded unscored), `max_in_flight=` serves several tasks concurrently, and
`resume=True` continues an interrupted run's directory instead of refusing it.

## The other quickstarts

`examples/` holds one directory per harness, each idiomatic to that harness rather
than squeezed into a shared abstraction. `claude_code/` is the reference implementation; this one
and `codex/` and `pi/` demonstrate the same three moves in their own idiom.
