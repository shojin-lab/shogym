# shōgym

A gym for running agents on tasks: bring any harness.

An environment describes a task, serves its tools over MCP, and verifies the recorded trajectory
with a pure function. A harness (Claude Code, Codex, Pi, Hermes, Prime Agent, or a plain loop)
drives the tools; shōgym never owns the model, the prompt, or the agent loop. The core is
zero-infrastructure: `pip install shogym`, point a harness at a served env, get a trace and a
score.

It is a substrate for questions about agents on tasks. How do agents improve themselves? What
does adding a tool change? Does an agent's self-report match an honest score?

Under the hood, an environment (`Env`) is a task loader, an initial observation, a pure verifier,
and a set of MCP servers. Tools are [MCP](https://modelcontextprotocol.io) servers
(in-process, stdio, or HTTP); episodes are session-keyed for safe concurrent rollouts;
termination is a reserved `terminate` tool or the horizon; verification is a pure function over
the recorded trajectory.

## Quickstart

Start with a harness. [`examples/`](examples/) holds one directory per harness, each idiomatic to
that harness rather than squeezed into a shared abstraction. Every quickstart does the same three
things: serve a **stream** of tasks over one endpoint, swap the env with **one variable**, and
read the scores back out of the server's own durable rows (the harness never grades itself).

- **[`claude_code/`](examples/claude_code/README.md)**: the reference implementation. One
  `--mcp-config` flag points the [`claude` CLI](https://claude.com/claude-code) at a queue of
  tasks; `results.py` reads the scores back.
- **[`pi/`](examples/pi/README.md)**: [Pi](https://github.com/earendil-works/pi) ships no MCP
  client, so this one adds a bridge extension, pinned to an exact version and scoped to the
  project's `.pi/`.
- **[`hermes/`](examples/hermes/README.md)**:
  [Hermes](https://hermes-agent.nousresearch.com) speaks MCP natively (with its `[mcp]` extra),
  configured by one `config.yaml` under an isolated `HERMES_HOME`.
- **[`codex/`](examples/codex/README.md)**: [Codex](https://github.com/openai/codex) runs
  `codex exec` with the stream declared inline on the command line, or read from the
  project-scoped `.codex/config.toml` checked in here.
- **[`prime_agent/`](examples/prime_agent/README.md)**:
  [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) takes MCP as a Python-backed
  kernel skill, and `serve.py` runs over HTTP in a second shell.

Each one defaults to `automationbench` over tasks `[0, 1, 2]`. One variable swaps the env for a
run, without editing a tracked file:

```bash
SHOGYM_ENV=tau2_banking_knowledge SHOGYM_TASKS=0,1 <the quickstart's command>
```

That one needs the `tau2` extra, tau2's data, and `OPENAI_API_KEY` for the user simulator, and it
has 97 tasks (`0` to `96`). `SHOGYM_ENV` wins when it is set, so reach for the variable while you
are trying envs out and edit the `ENV` literal at the top of `serve.py` once you have picked. Any
quickstart runs any env in the catalogue below, and task index ranges differ per env. `wordle_v1`
needs no extra and no key, and is the cheapest place to start.

shōgym is pinned to **Python 3.12** (`requires-python = ">=3.12,<3.13"`, with a committed
`.python-version`) because the tau2-bench port needs it. With [uv](https://docs.astral.sh/uv/),
`uv sync` builds the 3.12 venv and `uv run …` runs against it.

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
  via a single `run_command` tool, scored on survival, funds, and tasks. Deterministic in-process
  sim (no data or key). Needs the `yc_bench` extra.
- **[`hle`](src/shogym/envs/hle/README.md)**: a port of
  [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle). A single-turn, expert-level
  question answered via one `submit_answer` tool and graded server-side (exact-match fast path,
  then an OpenAI model judge). shōgym's first model-graded verifier. Needs the `hle` extra,
  `OPENAI_API_KEY`, and gated `cais/hle` access.
- **[`browsecomp_plus`](src/shogym/envs/browsecomp_plus/README.md)**: a port of
  [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus). Answer reasoning-heavy queries
  against a fixed ~100K-doc corpus via `search` / `get_document` / `submit_answer`, graded by an
  LLM judge plus deterministic retrieval-recall and citation metrics. Needs the
  `browsecomp_plus` extra, `OPENAI_API_KEY`, Java 21, and gated dataset access.
- **[`automationbench`](src/shogym/envs/automationbench/README.md)**: a port of
  [AutomationBench](https://github.com/zapier/AutomationBench). Carry out a cross-application
  business workflow over a fully simulated world of ~47 SaaS apps via an `api` tool surface,
  scored end-state-only by a pure rubric. Deterministic and offline (no key). Needs the
  `automationbench` extra.
- **[`appworld`](src/shogym/envs/appworld/README.md)**: a port of
  [AppWorld](https://github.com/StonyBrookNLP/appworld). Carry out a natural-language instruction
  across nine simulated apps by writing Python against their APIs, with one authored paragraph
  appended that asks for a filing log whose values a house convention computes from the world's
  own data and never states. Ships a matched pair of payloads for `Information` / `Placebo` runs.
  Needs no key; provisions its own interpreter (upstream pins `pydantic<2`) and a pinned data
  bundle.
- **[`frontier_bench`](src/shogym/envs/frontier_bench/README.md)**: a port of
  [Frontier-Bench](https://github.com/harbor-framework/frontier-bench). Operate a per-task Docker
  container through a shell (`exec` / `read_file` / `write_file` / `done`), scored by the task's
  own verifier over the container end-state. A CPU-only, single-container slice (5 tasks). Needs
  the `frontier_bench` extra and a local Docker daemon (no key or data download).

## The task server

Every quickstart is a thin wrapper around one object. A `TaskStream` publishes a whole queue over
a single MCP endpoint: it hands out tasks one at a time through `get_task`, routes the env's own
tools to whichever task is live, and seals and scores each one server-side.

```python
import asyncio
from pathlib import Path

import shogym
from shogym.serve.stream import Immediate, TaskRef, TaskStream, build_stream_server


async def main() -> None:
    stream = TaskStream(
        shogym.make,  # a factory, so each task gets a fresh env
        [TaskRef("tau2_banking_knowledge", i) for i in (0, 1, 2)],
        # Fresh per run: a stream refuses a directory another run recorded into.
        prov_dir=Path("runs/banking-0001"),
        feedback=Immediate(),  # ending a task returns the env's own verdict
    )
    async with stream:
        await build_stream_server(stream, name="shogym").run_async(transport="stdio")


asyncio.run(main())
```

Every dispensed task lands exactly one durable row under `prov_dir`, which
`shogym.serve.stream.read_results` reads back after the run.

For scores you intend to defend, construct `EvalStream` instead. It pins `feedback=Never()` and
refuses the argument outright, so a terminating call answers with the same fixed payload for every
env, task and outcome. It stamps `feedback_regime="never"` on every row it writes, and refuses to
resume a directory whose rows were recorded under any other regime. The agent is never told how it
did, so the harness cannot grade itself.

## One episode, no harness

The quickstarts serve a stream of tasks. A single episode is smaller: `shogym serve` publishes
one task over MCP for any client to spawn and drive, and scores it off a local JSONL trace.

```bash
shogym serve wordle_v1 --task 17 --trace ./shogym_logs/run.jsonl
```

Or drive one in-process and read the terminal feedback:

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
