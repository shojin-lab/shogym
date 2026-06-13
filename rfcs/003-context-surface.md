# RFC 003: The context surface

- **Status:** Draft (proposed 2026-06-13)
- **Depends on:** RFC 000 (esp. §7.1, the thesis revision); couples with RFC 007
  (persistence of derived state)
- **Locus:** **rollout** — a transform over the immutable trajectory, harness-configured
- **State today:** unbuilt, and currently mis-located. The env emits the full trajectory
  as the view (`tool_using_env.py:429`); the agent sends all of it (`agent.py:47`). The
  "view" is hard-wired to the identity.

---

## 1. What it is, and the thesis revision

The context surface is **what the model sees each step**, as a function of everything that
has happened. RFC 000 §7.1 establishes the necessary reframing:

- **Trajectory** = the full, immutable record the env accumulates and `_verify` and the logs
  read. Ground truth. Never altered.
- **View** = a transform over the trajectory, computed at the rollout, deciding what the
  *model* sees this step. Today: `view = trajectory` (identity). Optimizing the context
  surface means making the view function configurable.

The env keeps producing the full trajectory; a new rollout step computes
`view = context_strategy(trajectory)` before `agent.act`. Verification and logging stay on
ground truth; only the model's window changes. This is the cleanest separation because it
preserves the measurement (full trajectory) while opening the policy input (the view).

**Why it matters empirically:** context rot is universal (every one of 18 frontier models
tested by Chroma degrades as input grows, often well before the window fills; Anthropic
frames context as a depletable attention budget). The view function is therefore not a
nicety; it is a first-class performance lever, and one that interacts strongly with the model
(Cognition found defensive resets that helped Sonnet 4.5 became "dead weight" on Opus 4.5).
That model-dependence is exactly why it must be an *optimizable* surface, not a fixed default.

## 2. The single design constraint that shapes everything: replay determinism

hgym is an attribution library. Every result must be reproducible and every surface edit
attributable. That puts a constraint on context strategies that a normal agent framework does
not face: **the view must be reconstructable from what we persist.** Strategies fall into
three classes by how hard that is (this is the load-bearing taxonomy):

| Class | Strategies | Replay cost | What observability must persist |
|---|---|---|---|
| **pure** | `full`, `window`, deterministic `evict`, truncation in `offload_tools` | free — `view = f(trajectory)` recomputes exactly | nothing beyond the trajectory |
| **event** | `compact`, `reset_on_threshold` | the summary/handoff is a *model output*, not a function of inputs | the generated artifact + the boundary index (RFC 007) |
| **stateful** | `retrieve`, `memory` | the store is mutable, side-effecting | a snapshot of the store per step + the embed-model version/seed |

This classification dictates the staging (§5) and the observability contract (RFC 007): pure
strategies are clean attribution out of the box; event and stateful strategies require RFC 007
to persist derived state or replay diverges silently — the worst failure for an eval library.

## 3. The strategy menu (composable layers)

A context config picks one **base view policy**, plus an optional **tool-result modifier**,
plus an optional **memory store**. The rollout composes them in order: tool-result handling
first (it shrinks the rawest, largest content), then the base policy, then memory injection.

| Strategy | Class | Mechanism | Key params |
|---|---|---|---|
| `full` | pure | render the whole trajectory (the control / baseline) | — |
| `window` | pure | keep system + last N turns or K tokens, turn-boundary safe | `max_turns`, `max_tokens`, `pin_system`, `pin_task_spec` |
| `offload_tools` | pure (+disk) | truncate/clear/offload large tool results to file handles + previews | `tool_result_max_tokens` (~20k), `preview_lines` (~10), `per_tool_char_cap` (~50k), `clear_after_turns` |
| `evict` | pure if scorer is deterministic | score-based selective inclusion (recency / lru / relevance / dependency-graph), with pinning | `policy`, `evict_trigger_tokens`, `pin_set`, weights |
| `compact` | event | summarize older prefix at a threshold; keep recent tail + files | `trigger_fraction` (~0.8), `keep_recent_turns`, `preserve_schema`, `summary_prompt`, `recursive` |
| `reset_on_threshold` | event | clear and restart from a structured handoff artifact + durable files | `reset_on`, `handoff_schema`, `handoff_format` (json), `durable_state_paths` |
| `retrieve` | stateful | keep handles; pull top-K relevant past content on demand | `mode` (jit/prefetch/hybrid), `top_k`, `embed_model`, `index_backend` |
| `memory` | stateful | tiered self-managed memory (core/recall/archival) via tools | `backend`, `core_memory_tokens`, `write_policy`, `tiers` |

Empirically grounded defaults worth baking in: turn-boundary windowing (token windowing
orphans tool calls and most APIs 400 on that); `offload_tools` almost always on (tool spew is
the dominant filler, and offloading is the largest cheap win — DeepAgents offloads >20k-token
results to a path + 10-line preview); `preserve_schema` for `compact` should be explicit
(open goals, decisions+rationale, file/env state, open errors, next step) because the failure
mode (ACE's "context collapse": 18k tokens → 122 tokens, accuracy 66.7% → 57.1% after one
monolithic rewrite) comes precisely from unstructured summarization.

## 4. The artifact

```toml
# context.toml — absent means `full` (the default and the eval control)
base = "window"                 # full | window | evict | compact | reset_on_threshold
class = "pure"                  # pure | event | stateful — tells observability what to persist

[window]
unit = "turns"
max_turns = 12
pin_system = true
pin_task_spec = true

[offload_tools]                 # a modifier; composes with any base
tool_result_max_tokens = 20000
preview_lines = 10
recoverable = true
```

The `class` field is mandatory and is the contract with RFC 007: it declares what must be
checkpointed for replay. A mismatch (declaring `pure` for a `compact` base) is a load-time
error.

## 5. Staging: ship pure first

The determinism taxonomy (§2) gives a clean build order:

- **M2 (first cut): pure strategies only** — `full`, `window`, `evict` (deterministic
  scorers, e.g. the dependency-graph policy that needs no LLM), and `offload_tools`
  truncation. These replay for free, attribute cleanly, and cover the highest-value cheap
  wins (windowing + tool-result offload). The first context-attribution experiment
  ("does a window beat full on the envs where tools didn't help?") runs entirely on these.
- **M3: event strategies** — `compact`, `reset_on_threshold`. These need RFC 007 to persist
  the generated artifact + boundary. They unlock long-horizon work but carry the
  context-collapse and lost-state risks, so they want the structured `preserve_schema` and
  measurement before they are trusted.
- **M3+: stateful strategies** — `retrieve`, `memory`. These need store snapshotting and an
  embed-model version pin; they are the most powerful and the least reproducible, so they
  come last and behind the strongest persistence guarantees.

## 6. Where the transform attaches (resolving RFC 000 §9.2)

**Decision: the runner applies it.** `view = context_strategy(trajectory)` is computed in
`run_episode` between `env.step` (which returns the full-trajectory observation) and
`agent.act` (which receives a view-substituted observation). Rationale:

- The agent stays thin and swappable (RFC 000 §2.2); it should not each re-implement context
  management.
- `agent_builder` users (deployed configs, shared gateways) get context management for free,
  which is exactly the population that most needs it and least wants to reimplement it.
- The runner already owns the loop and the place where trajectory becomes the next input; the
  transform is one line there.

The agent still receives an `Observation`; the runner just hands it one whose `messages` are
the view rather than the raw trajectory. The `StepData`/logged trajectory remains full.

## 7. Attribution nuance: strategies that call models

`compact` and `memory` make their own model calls (a summarizer, a memory curator). Those
calls (a) cost tokens that must be attributed to the context surface, not the agent, and
(b) contain a sub-prompt (`summary_prompt`) that is itself instruction-like. Keep the
sub-prompt in `context.toml` (it is part of the strategy, and moving it to `instruction/`
would split one concern across two files, defeating per-file attribution). RFC 007 attributes
the summarizer's token cost to the context surface via the call's provenance tag.

## 8. Risks / where this might be wrong

- **The pure/event/stateful line is the whole bet.** If most of the value turns out to be in
  `compact`/`memory` (event/stateful), then the "ship pure first" staging delivers little
  early, and the hard persistence work cannot be deferred. Mitigation: the first
  context-attribution experiment is on pure strategies precisely to test whether windowing
  alone moves the metric; if it does not, we learn that early and reprioritize.
- **Runner-applied context couples the runner to the harness format.** The runner must learn
  to read `context.toml`. Acceptable (it already reads `extra_toolset`), but it grows the
  rollout's responsibilities. The alternative (agent-wrapper) was rejected for the
  free-for-agent_builder property; revisit if the runner gets too heavy.
- **`memory`/`retrieve` may not be reproducible enough to be a research surface at all.** If
  store snapshotting proves too costly or non-deterministic (embed-model drift), these may be
  "supported but not attributable" — usable for capability demos, excluded from controlled
  ablations. Honest possibility; decide when M3+ lands.

## Sources

Anthropic, effective context engineering / long-running harnesses; ACE (arXiv:2510.04618,
context collapse); Chroma context rot; Liu et al. lost-in-the-middle; Cognition (Devin on
Sonnet 4.5 vs Opus 4.5); LangChain DeepAgents context management; Arize/Letta tool-result
offload; Beyond Compaction / structured eviction (arXiv:2606.11213). Full URLs in the
research transcript appended to this RFC stack.
