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

# `harness` is an async callable given a FastMCP client connected to the served env;
# see examples/openai_harness.py for a runnable one.
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

Runnable end-to-end demos (Claude Code drives a served env; hgym scores off the trace):

- **[`examples/wordle/claude_code/`](examples/wordle/claude_code/README.md)** — Claude Code
  plays Wordle through a served hgym env.
- **[`examples/tau2/claude_code/`](examples/tau2/claude_code/README.md)** — Claude Code plays
  a tau2 domain; hgym scores it with tau2's evaluator.
- **[`examples/yc_bench/claude_code/`](examples/yc_bench/claude_code/README.md)** — Claude Code
  operates the YC-Bench startup sim; hgym scores the run off the trace.
- **[`examples/hle/claude_code/`](examples/hle/claude_code/README.md)** — Claude Code answers an
  HLE question through a served hgym env; hgym grades it server-side.

## License

Apache-2.0. Portions derived from [llmgym](https://github.com/tensorzero/llmgym)
(© TensorZero, Apache-2.0) — see NOTICE.
