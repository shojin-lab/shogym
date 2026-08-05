# hgym

A minimal gym for running agents on tasks — bring any harness.

An environment is small and honest: it **describes** a task, **serves** its tools over MCP, and **verifies** the recorded trajectory with a pure function. An external harness — Claude Code, Codex, pi, Hermes, or a plain loop — drives the tools; hgym never owns the model, the prompt, or the agent loop. The core is zero-infrastructure: `pip install hgym`, point a harness at a served env, get a local JSONL trace and a score.

It's a neutral substrate for asking questions about agents on tasks. One example: *how do agents improve themselves — and what do existing agents actually do to get better?* But the same minimal interface serves others just as well — comparing harnesses on a task, measuring what adding a tool changes, or checking whether an agent's self-report matches an honest score. The [examples](examples/) show it in use.

Under the hood, environments are `ToolUsingEnv`s: a task loader, an initial observation, a
pure verifier, and a set of MCP servers. Tools are [MCP](https://modelcontextprotocol.io)
servers (in-process, stdio, or HTTP); episodes are session-keyed for safe concurrent
rollouts; termination is a reserved `terminate` tool or the horizon; verification is a
pure function over the recorded trajectory. A thin, **provider-neutral** model-client seam
speaks the OpenAI-compatible wire schema, so any provider, proxy, or local server works.

## Status

Pre-alpha. The core engine is ported and under active development. The design —
[RFC 008: the environment is the center of gravity](https://github.com/anndvision/hgym/wiki/RFC-008-Environment-as-Center-of-Gravity)
(with the background lit reviews) — lives in the [wiki](https://github.com/anndvision/hgym/wiki).
Not yet ready for use.

## Quickstart

Serve an environment over MCP for any harness to spawn and drive; it scores the result off
a local JSONL trace:

```bash
hgym serve wordle_v1 --task 17 --trace ./hgym_logs/run.jsonl
```

Or evaluate a harness in-process and read the terminal feedback:

```python
import hgym

# `harness` is an async callable given a FastMCP client connected to the served env.
result = await hgym.evaluate("wordle_v1", task=17, harness=my_harness)
print(result.value("check_answer"))
```

hgym is pinned to **Python 3.12** (`requires-python = ">=3.12,<3.13"`, with a committed
`.python-version`) — the tau2-bench port needs it. With [uv](https://docs.astral.sh/uv/),
`uv sync` builds the 3.12 venv (including the `tau2` extra) and `uv run …` just works.

## Environments

Each env **describes** a task, **serves** its tools over MCP, and **verifies** a recorded
trajectory; an external harness drives it (see
[RFC 008](https://github.com/anndvision/hgym/wiki/RFC-008-Environment-as-Center-of-Gravity)).
The [`src/hgym/envs/`](src/hgym/envs/README.md) README covers the model and the shared
env-README template.

- **[`wordle_v1`](src/hgym/envs/wordle/README.md)** — the reference env-as-center
  environment; Wordle with a `guess` tool and a pure trajectory verifier. No extra deps.
- **[tau2-bench](src/hgym/envs/tau2/README.md)** — a faithful port of
  [τ²-bench](https://github.com/sierra-research/tau2-bench): tool-using customer-service
  agents (`tau2_mock`, `tau2_airline`, `tau2_retail`, `tau2_telecom`,
  `tau2_banking_knowledge`), scored by tau2's own evaluator. Needs the `tau2` extra + data.
- **[`yc_bench`](src/hgym/envs/yc_bench/README.md)** — a faithful port of
  [YC-Bench](https://github.com/collinear-ai/yc-bench): operate a simulated AI startup for a
  year via a single `run_command` tool, scored on survival, funds, and tasks. Deterministic
  in-process sim (no data or key). Needs the `yc_bench` extra.
- **[`hle`](src/hgym/envs/hle/README.md)** — a faithful port of
  [Humanity's Last Exam](https://huggingface.co/datasets/cais/hle): a single-turn,
  expert-level question answered via one `submit_answer` tool and graded server-side
  (exact-match fast path, then an OpenAI model judge). hgym's first model-graded verifier.
  Needs the `hle` extra, `OPENAI_API_KEY`, and gated `cais/hle` access.
- **[`browsecomp_plus`](src/hgym/envs/browsecomp_plus/README.md)** — a faithful port of
  [BrowseComp-Plus](https://github.com/texttron/BrowseComp-Plus): answer reasoning-heavy queries
  against a fixed ~100K-doc corpus via `search` / `get_document` / `submit_answer`, graded by an
  LLM judge plus deterministic retrieval-recall and citation metrics. Needs the
  `browsecomp_plus` extra, `OPENAI_API_KEY`, Java 21, and gated dataset access.
- **[`automationbench`](src/hgym/envs/automationbench/README.md)** — a faithful port of
  [AutomationBench](https://github.com/zapier/AutomationBench): carry out a cross-application
  business workflow over a fully simulated world of ~47 SaaS apps via an `api` tool surface,
  scored end-state-only by a pure rubric. Deterministic and offline (no key). Needs the
  `automationbench` extra.
- **[`frontier_bench`](src/hgym/envs/frontier_bench/README.md)** — a faithful port of
  [Frontier-Bench](https://github.com/harbor-framework/frontier-bench): operate a per-task Docker
  container through a shell (`exec` / `read_file` / `write_file` / `done`), scored by the task's
  own verifier over the container end-state. A CPU-only, single-container slice (5 tasks). Needs
  the `frontier_bench` extra and a local Docker daemon (no key or data download).

## Quickstarts

One directory per harness, each idiomatic to that harness. Every quickstart does the same three
things: serve a **stream** of tasks over one endpoint, swap the env with **one variable**, and
read the scores back out of the server's own durable rows (the harness never grades itself).

- **[`examples/quickstarts/claude_code/`](examples/quickstarts/claude_code/README.md)**: the
  reference implementation. Point the `claude` CLI at a queue of tasks and read the results.

Swapping the environment is one variable at the top of `serve.py`, so any quickstart runs any
env in the catalogue above.

## License

Apache-2.0. Portions derived from [llmgym](https://github.com/tensorzero/llmgym)
(© TensorZero, Apache-2.0) — see NOTICE.
