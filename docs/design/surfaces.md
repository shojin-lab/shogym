# The surfaces of an agent harness

*Design north star for hgym. A taxonomy of the harness, partitioned so that every
part is independently editable and independently measurable.*

## Definitions

An **agent** is a model running tools in a loop toward a goal ([Willison][w1]). The
**harness** is everything in that sentence except the model ([Trivedy][t1]: "Agent =
Model + Harness. If you're not the model, you're the harness").

A **surface** is a partition of the harness where an engineer — or an optimizer —
can intervene, and whose contribution to agent behavior can be attributed by holding
the other surfaces fixed. The taxonomy's design constraint: each surface must be
**(a)** an editable artifact (a file you can diff), **(b)** an attributable cause (a
thing an ablation can isolate), and **(c)** a meaningful unit of decay (see the
model-relativity principle below).

## The seven surfaces

| # | Surface | Controls | Concrete artifacts |
|---|---|---|---|
| 1 | **Instruction** | What the model is told | System/user templates, skills, AGENTS.md-style files, tool descriptions |
| 2 | **Tool** | What the model can do (the action space) | Tool/MCP registry, code execution, subagent-spawn-as-tool, `terminate` |
| 3 | **Context** | What the model sees each step | Window policy: compaction vs resets vs structured handoffs; memory files; filesystem offloading; retrieval |
| 4 | **Control** | The shape of the loop(s) | Loop + stopping conditions, retries, hooks/middleware, multi-agent topology (planner/generator/evaluator), routing |
| 5 | **Environment** | Where actions land; the blast radius | Sandbox/container, filesystem+git as durable state, network, credentials, budgets, permission posture |
| 6 | **Verification** | The feedback signal closing the loop | Test suites, verifiers, evaluator agents, rubrics, success criteria |
| 7 | **Observability** | What the engineer/optimizer sees | Traces, logs, cost metering, run ledgers |

Note: tool *descriptions* belong to the instruction surface even though tools belong
to the tool surface — adding a tool changes the action space; rewording its
description changes the policy over an unchanged action space. Attribution requires
keeping these apart.

## Structure

**Surfaces 1–3 shape the policy; 4–6 shape the system; 7 shapes the next harness.**
Instruction, tool, and context surfaces determine what the model emits at each step.
Control, environment, and verification determine what those emissions become in the
world. Observability changes nothing in-episode — it exists so traces can become
edits; it is the substrate of automated harness optimization.

**Inner vs outer harness.** Surfaces 1–3 and 5–6 exist *per loop*. Surface 4 is
where one loop becomes many. The apparent dispute between minimal definitions
([Willison][w1]: a loop executing tools) and maximal ones ([Anthropic][a1]: societies
of planners, generators, and evaluators) is a level distinction, not a disagreement:
an outer harness is control-surface composition of inner harnesses.

**Model-relativity and decay.** "Every component in a harness encodes an assumption
about what the model can't do on its own" ([Anthropic][a1]). Surfaces therefore decay
at different rates as models improve: context-surface machinery visibly decayed
between successive frontier models (the "context anxiety" pathology), while
verification machinery is *appreciating* (self-evaluation bias persists). A taxonomy
of surfaces is also a taxonomy of decay rates — and harness attribution is the
measurement program that tells you which scaffolding to delete this year.

## hgym's stance

hgym makes each surface a **config artifact in the exported harness directory** —
diffable, validatable, and rangeable by a search procedure ("the unit of edit is the
lever"). The tool surface is the policy boundary (env-mandatory pool ∪ extras pool);
verification is a pure function over the recorded trajectory; the roadmap for
opening the remaining surfaces is in [roadmap.md](roadmap.md).

## Sources

- [t1]: V. Trivedy, "Agent = Model + Harness" (X, 2026); the component inventory is
  elaborated in LangChain's "The Anatomy of an Agent Harness."
- A. Osmani, ["Agent Harness Engineering"](https://addyosmani.com/blog/agent-harness-engineering/) —
  the ratchet principle, tool discipline, hooks doctrine.
- [w1]: S. Willison, ["I think 'agent' may finally have a widely enough agreed upon definition"](https://simonwillison.net/2025/Sep/18/agents/)
  and ["Designing agentic loops"](https://simonwillison.net/2025/Sep/30/designing-agentic-loops/) —
  the minimal definition; the environment trio (sandboxing, credentials, permission posture).
- [a1]: Anthropic, ["Effective harness design for long-running agents"](https://www.anthropic.com/engineering/harness-design-long-running-apps) —
  context strategies, generation/evaluation separation, pathologies, model-relativity.
- Lee, Nair, Zhang, Lee, Khattab, Finn, ["Meta-Harness: End-to-End Optimization of Model Harnesses"](https://arxiv.org/abs/2603.28052) —
  end-to-end harness search; the complementary pole to per-surface attribution.

[t1]: https://x.com/Vtrivedy10/article/2031408954517971368
[w1]: https://simonwillison.net/2025/Sep/18/agents/
[a1]: https://www.anthropic.com/engineering/harness-design-long-running-apps
