# hgym — Harness Gym

**MCP-native environments that any agent harness can step.**

`hgym` is a fresh start of [LLM Gym](https://github.com/tensorzero/llmgym), rebuilt
around one idea: the **environment** is the fixed thing, and the **harness** — the agent
loop that drives it — is external and swappable. An hgym environment does exactly three
things:

- **describe** — publish a `TaskSpec`: the task framing, the essential-tool manifest,
  reference templates, and schemas (`env.describe()`).
- **serve** — expose its essential tools as [MCP](https://modelcontextprotocol.io)
  servers (in-process, stdio, or HTTP), including the reserved `terminate` tool. A tool
  call *is* a step; the tool handler is where env state mutates.
- **verify** — a pure function over the recorded trajectory that returns feedback.

Because the interface every real agent harness shares is *tools + a task + a terminate
convention* — not a `step(action)` method — an hgym env plugs into
[Claude Code](https://www.anthropic.com/claude-code), Codex,
[pi](https://github.com/earendil-works/pi),
[Hermes](https://github.com/NousResearch/hermes-agent), or a thirty-line example loop,
with no per-harness glue. Feedback rides back over the same MCP wire (a `_meta` sidecar)
and is always also written to a local JSONL trace.

Design goals:

- **Zero infrastructure.** `pip install hgym`, serve an env, point a harness at it.
  No databases, no gateway servers. Traces are local JSONL files.
- **Harness-agnostic.** Hold `(env, task)` fixed, swap the harness, and the delta is
  attributable to the harness — an apples-to-apples comparison of real agent loops.
- **The env is the policy-free substrate.** Instructions, context, model, retries, and
  sandboxing all belong to the external harness; hgym owns the task, the tools, and the
  verifier, and nothing else.

## Status

Pre-alpha, under active development toward a v0 (Wordle + a Claude Code example). Design
documents — [the surfaces of an agent harness](https://github.com/anndvision/hgym/wiki/The-Surfaces-of-an-Agent-Harness),
[the roadmap](https://github.com/anndvision/hgym/wiki/Roadmap), and the
[RFCs](https://github.com/anndvision/hgym/wiki/Surface-RFCs) (including RFC 008, the
env-as-center design) — live in the [wiki](https://github.com/anndvision/hgym/wiki).
Not yet ready for use.

## Quickstart (target API)

Describe an environment's task contract today:

```python
import hgym

spec = hgym.make("wordle_v1").describe()
print(spec.instructions)          # the task framing to hand a harness
print([t.name for t in spec.tools])   # ['terminate', 'guess']
```

Serving an env and driving it with an external harness lands across the v0 stack:

```python
# target API (in progress)
feedback = hgym.evaluate(
    env="wordle_v1", task="17",
    harness=["claude", "-p", "--mcp-config", "hgym://task"],
)
```

## License

Apache-2.0. Portions derived from [llmgym](https://github.com/tensorzero/llmgym)
(© TensorZero, Apache-2.0) — see NOTICE.
