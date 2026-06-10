# hgym — Harness Gym

**Environments for agents whose policy boundary is the tool surface.**

`hgym` is a fresh start of [LLM Gym](https://github.com/tensorzero/llmgym), rebuilt
around one idea: the *tool surface* — the set of tools an agent can call — is the
program. Same environment, same model: `{search, compose, terminate}` is one research
program; `{search, compose, draft, critique, revise, terminate}` is draft-and-revise;
`{answer, terminate}` is one-shot. Surfaces compose as config diffs, not Python forks,
which makes them a target an optimizer can range over.

Environments are `ToolUsingEnv`s: a task loader, an initial observation, a pure
verifier, and a set of MCP servers. Tools are [MCP](https://modelcontextprotocol.io)
servers (in-process, stdio, or HTTP); episodes are session-keyed for safe concurrent
rollouts; termination is a reserved `terminate` tool or the horizon; verification is a
pure function over the recorded trajectory.

Design goals:

- **Zero infrastructure.** `pip install hgym`, export an API key, run an episode.
  No Docker, no databases, no gateway servers. Traces are local JSONL files.
- **Provider-neutral.** A thin model-client seam speaking the OpenAI-compatible wire
  schema; bring any provider, proxy, or local server.
- **The optimizer workflow is first-class.** Export an environment's harness as an
  editable directory (templates, model, tool manifest), let a human or an agent edit
  it, re-run, score.

## Status

Pre-alpha. The core engine is ported and under active development; see
[the spec](https://github.com/anndvision/hgym) and roadmap. Not yet ready for use.

## Quickstart (target API)

```python
import hgym

rollouts = await hgym.run_episodes(
    env_name="wordle_v1",
    model="openai/gpt-5.2-mini",
    num_tasks=50,
)
```

## License

Apache-2.0. Portions derived from [llmgym](https://github.com/tensorzero/llmgym)
(© TensorZero, Apache-2.0) — see NOTICE.
