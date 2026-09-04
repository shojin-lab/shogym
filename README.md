# shōgym

A gym for running agents on tasks: bring any harness.

An environment describes a task, serves its tools over MCP, and verifies the recorded trajectory
with a pure function. A harness (Claude Code, Codex, Pi, Hermes, Prime Agent, or a plain loop)
drives the tools; shōgym never owns the model, the prompt, or the agent loop. The install is one
line: `pip install shogym` plus an API key for your harness gets you the envs, an in-process
episode, and a served task, with nothing you have to run yourself.

It is a substrate for questions about agents on tasks. How do agents improve themselves? What
does adding a tool change? Does an agent's self-report match an honest score?

An environment (`Env`) is a task loader, an initial observation, a pure verifier, and a set of
MCP servers. Tools are [MCP](https://modelcontextprotocol.io) servers
(in-process, stdio, or HTTP); episodes are session-keyed for safe concurrent rollouts;
termination is a reserved `terminate` tool or the horizon; verification is a pure function over
the recorded trajectory.

## Quickstart

Start with a harness. [`examples/`](examples/) holds one directory per harness, each idiomatic to
that harness rather than squeezed into a shared abstraction. Every quickstart does the same three
things: serve **one task** over one endpoint, swap the env with **one variable**, and leave the
scoring to the stream, which seals and grades each attempt itself (the harness never grades
itself).

- **[`claude_code/`](examples/claude_code/README.md)**: the reference implementation. One
  `--mcp-config` flag points the [`claude` CLI](https://claude.com/claude-code) at the served
  task.
- **[`pi/`](examples/pi/README.md)**: [Pi](https://github.com/earendil-works/pi) ships no MCP
  client, so this one adds a bridge extension, pinned to an exact version and scoped to the
  project's `.pi/`.
- **[`hermes/`](examples/hermes/README.md)**:
  [Hermes](https://hermes-agent.nousresearch.com) speaks MCP natively (with its `[mcp]` extra),
  configured by one `config.yaml` under an isolated `HERMES_HOME`.
- **[`codex/`](examples/codex/README.md)**: [Codex](https://github.com/openai/codex) runs
  `codex exec` with the server declared inline on the command line, or read from the
  project-scoped `.codex/config.toml` checked in here.
- **[`prime_agent/`](examples/prime_agent/README.md)**:
  [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) takes MCP as a Python-backed
  kernel skill, and `serve.py` runs over HTTP in a second shell.

Each one defaults to `wordle_v1` at task `0` (the one env that needs no extra; `SHOGYM_ENV=automationbench` needs `pip install "shogym[automationbench]"`), and one launch is one episode: the stream
serves a single env at a single task, so three tasks are three launches. Two variables name the
env and the task for one run, without editing a tracked file:

```bash
SHOGYM_ENV=tau2_banking_knowledge SHOGYM_TASK=1 <the quickstart's command>
```

That one needs the `tau2` extra, tau2's data, and `OPENAI_API_KEY` for the user simulator, and it
has 97 tasks (`0` to `96`). `SHOGYM_ENV` wins when it is set, so reach for the variable while you
are trying envs out and edit the `ENV` literal at the top of `serve.py` once you have picked. Any
quickstart runs any env in the catalogue below, and task index ranges differ per env. `wordle_v1`
needs no extra and no key, and is the cheapest place to start.

shōgym is pinned to **Python 3.12** (`requires-python = ">=3.12,<3.13"`, with a committed
`.python-version`) because the tau2-bench port needs it. With [uv](https://docs.astral.sh/uv/),
`uv sync` builds the 3.12 venv and `uv run …` runs against it. Serving needs no extra and no
second install: the stream's history, replay and timers are Temporal's, and `temporalio` is a
dependency of the package. The first serve downloads a dev-server binary (about 130 MB) into
`~/.cache/shogym/temporal` and starts it; there is nothing to configure, nothing to run
yourself, and every serve after that reuses the binary. Set `SHOGYM_TEMPORAL_ADDRESS` to point
at a server you already run instead, and then nothing is downloaded or started at all.

## Environments

Each env **describes** a task, **serves** its tools over MCP, and **verifies** a recorded
trajectory; an external harness drives it. The [`src/shogym/envs/`](src/shogym/envs/README.md)
README covers the model and the shared env-README template.

- **[`wordle_v1`](src/shogym/envs/wordle/README.md)**: the reference environment. Wordle with a
  `guess` tool and a pure trajectory verifier. No extra deps.
- **[tau2-bench](src/shogym/envs/tau2/README.md)**: a port of
  [τ²-bench](https://github.com/sierra-research/tau2-bench). Tool-using customer-service agents
  (`tau2_mock`, `tau2_airline`, `tau2_retail`, `tau2_telecom`, `tau2_banking_knowledge`), scored
  by tau2's own evaluator. Needs the `tau2` extra + data.
- **[`yc_bench`](src/shogym/envs/yc_bench/README.md)**: a port of
  [YC-Bench](https://github.com/collinear-ai/yc-bench). Operate a simulated AI startup for a year
  via a single `run_command` tool, scored on survival, funds, and tasks. In-process sim, no data
  or key; a seed reproduces the business attributes, not the `uuid4` row ids. Needs the
  `yc_bench` extra.
- **[`hle`](src/shogym/envs/hle/README.md)**: a port of
  [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle). A single-turn, expert-level
  question answered via one `submit_answer` tool and graded server-side (exact-match fast path,
  then an OpenAI model judge). shōgym's first model-graded verifier. Needs the `hle` extra,
  `OPENAI_API_KEY`, and gated `cais/hle` access.
- **[`browsecomp_plus`](src/shogym/envs/browsecomp_plus/README.md)**: a port of
  [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus). Answer reasoning-heavy queries
  against a fixed ~100K-doc corpus via `search` / `get_document` / `submit_answer`, graded by an
  LLM judge plus deterministic retrieval-recall and citation metrics. Needs the
  `browsecomp_plus` extra, `OPENAI_API_KEY`, Java 21, and Hugging Face network access (the
  datasets are public).
- **[`automationbench`](src/shogym/envs/automationbench/README.md)**: a port of
  [AutomationBench](https://github.com/zapier/AutomationBench). Carry out a cross-application
  business workflow over a fully simulated world of ~47 SaaS apps via an `api` tool surface,
  scored end-state-only by a pure rubric. Offline and keyless, and the rubric is deterministic
  over a given end-state. Needs the `automationbench` extra.
- **[`frontier_bench`](src/shogym/envs/frontier_bench/README.md)**: a port of
  [Frontier-Bench](https://github.com/harbor-framework/frontier-bench). Operate a per-task Docker
  container through a shell (`exec` / `read_file` / `write_file` / `done`), scored by the task's
  own verifier over the container end-state. A CPU-only, single-container slice (5 tasks). Needs
  the `frontier_bench` extra and a local Docker daemon (no key or data download).

## The task server

Every quickstart is a thin wrapper around one command. It serves one env at one task over stdio
MCP, and the generation behind it is durable: the stream owns the queue, the attempt, the seal and
the score, and this process is only its transport.

```bash
shogym serve wordle_v1 --task 17 --run-dir runs/wordle-0001 --trace ./shogym_logs/run.jsonl
```

Two kinds of tool reach the model. `pull` takes no arguments and answers with exactly one JSON
record, which is the whole of what the agent is told:

```jsonc
{"protocol_version": 2, "kind": "task",    "message_id": "...", "attempt_id": "9f3c...", "body": "..."}
{"protocol_version": 2, "kind": "payload", "message_id": "...", "attempt_id": "9f3c...", "body": "..."}
{"protocol_version": 2, "kind": "wait",    "message_id": "...", "retry_after_ms": 500}
{"protocol_version": 2, "kind": "done",    "message_id": "..."}
```

Every environment tool is the second kind, wrapped so that a call names the attempt it belongs to
and no native argument can collide with a protocol field:

```jsonc
{"attempt_id": "9f3c...", "arguments": {"word": "crane"}}
```

The env's terminal tool is wrapped the same way and then intercepted: it never reaches the env
from the transport, it becomes the stream's terminal request, and it answers with the
acknowledgement the stream minted (or a refusal, which leaves the attempt open to file again). So
the loop is `pull`, work, end the task, `pull`, and stop on `done`. The payload that arrives after
an acknowledgement is honest by default: it carries the score the seal committed and the numbers
the env published beside it, so an agent under the defaults can tell how it did. Concealing that,
or replacing it with something else, is a payload policy an experiment registers, and the run's own
records name the policy every attempt was served under. There is no task index
anywhere on the wire: a task record carries an attempt id and a body, and has no field an index or
a target could be written into. A run may also declare a step budget, and then every task it
serves carries `budget` as well: one number for the whole run, the number of environment tool
calls the attempt gets. A run that declares none serves the task record exactly as it is shown
above.

`--run-dir` is where the generation keeps the blobs its presentations reference, the manifest a
later owner would resume it from, and the embedded service's own database. All three belong to the
run, which is what lets two `shogym serve` processes run side by side on one machine: they share
the downloaded binary and share no state. Without `--run-dir` the database is a temporary file this
process owns and deletes when it exits, and there is nothing to resume, by design: a run you might
want to take over later is a run you gave a directory. The grade is in there too. Sealing grades
server-side, and the outcome stays in the stream's own durable history, which is the database this
directory holds: `shogym results runs/wordle-0001` brings that history back up, prints one row per
attempt, and leaves the same rows in the directory as `records.jsonl`. That file is a derived view,
rebuilt from the history every time it is asked for, so the history stays the record and the file
is never the authority. A run whose score you intend to defend therefore gets a directory the agent
is not working in, and the deny list its quickstart names: an agent that can read the directory can
read what the seal recorded before the stream presents it. Directories written by the retired v1
serving path stay readable offline, through `shogym.serve.v1_runs.read_results` / `read_dispenses`
/ `reconcile`.

## One episode, no harness

The quickstarts serve a task to somebody else's agent. A single episode is smaller: drive one
in-process and read the terminal feedback, with no MCP transport and nothing durable underneath.

```python
import shogym

# `harness` is an async callable given a FastMCP client connected to the served env.
result = await shogym.evaluate("wordle_v1", task=17, harness=my_harness)
print(result.value("check_answer"))
```

## Community

Questions, results, and replications: [the shōjin Discord](https://discord.gg/cRmZYt5smz).

## License

Apache-2.0. Portions derived from [llmgym](https://github.com/tensorzero/llmgym)
(© TensorZero, Apache-2.0): see NOTICE.
