# RFC 004: The control surface

- **Status:** Draft (proposed 2026-06-13)
- **Depends on:** RFC 000 (esp. §7.2, topology collapses into tools); couples with
  RFC 002 (subagents-as-tools) and RFC 005 (hook process isolation)
- **Locus:** **rollout** (the env contributes only the horizon constraint and the
  `terminate` convention)
- **State today:** the loop is four lines with no extension points
  (`runner.py:62`); termination is `terminate`-tool-or-horizon (`tool_using_env.py:407`);
  `available_tools(state)` was reserved in the roadmap but is unbuilt.

---

## 1. What it is, and the hard constraint

The control surface is **the shape of the loop**: stopping conditions, retries/repair, hooks
around tool calls and turns, and phase-gating of the tool set. The base loop is fixed (model
emits action → tools dispatch → results append → repeat). The surface is the *extension
points* on that loop.

The hard constraint, learned from every production harness surveyed (Claude Code, LangChain
middleware, OpenAI Agents SDK): **make the loop configurable without shipping a control-flow
language.** The failure mode is a workflow DSL — conditionals, loops, edges in config — which
becomes a Turing tarpit nobody can read, diff, or attribute. The discipline that prevents it:
**config owns the axes of the loop; Python owns the bodies.** Config sets scalars and selects
named behaviors and points at user functions; it never expresses control flow itself.

## 2. The observer / transformer / gate taxonomy (the rule that prevents the tarpit)

Every extension point is exactly one of three kinds, and **may return only the type its kind
permits**. This single rule is what stops hooks from quietly becoming a workflow engine; it is
why OpenAI keeps lifecycle hooks observer-only and routes all blocking through guardrails.

- **OBSERVER** — logging, metrics, side effects. Return ignored. (`on_step`, `on_tool_start`,
  `on_episode_end`.)
- **TRANSFORMER** — may rewrite one message / tool call / result, returning a same-typed value.
  (`before_tool(call) → call`, `after_tool(result) → result`.)
- **GATE** — may block / deny / redirect / continue, returning an allow-deny-continue verdict.
  (`should_continue(state) → bool`, guardrails, phase-gating.)

A hook declared `observer` whose function returns a rewritten value is a load-time error. This
is the boundary that keeps the surface declarative.

## 3. The five knobs (80% of the value, no DSL)

Grounded in the convergent design of Claude Code hooks, LangChain built-in middleware, and the
OpenAI Agents SDK:

### 3.1 Stopping conditions (GATE)
`max_turns` (hard horizon; may only *tighten* the env's horizon, never loosen it — RFC 000 §2),
the reserved `terminate` tool, and an optional `stop_when(state) -> bool` predicate. The env
owns the horizon as a constraint; the harness may tighten it and add a predicate. This is the
single most important knob and is pure config.

### 3.2 Retries / repair (GATE + loop)
Per-tool `retries = {max, backoff, retry_on: [exception classes]}`, with the standard
trichotomy: **transient** (retry with backoff), **invalid-input** (do not retry; the error is
fed back and the model re-emits), **fatal** (abort to the orchestrator). Crucially, *repair
needs no new primitive*: hgym's base loop already appends a failed tool's error as a result and
continues (`tool_using_env.py:479`), which is exactly "feed the error back and let the model
try again." The only thing to add is the retry cap and the retryable-exception classification.
Mirrors LangChain `ToolRetryMiddleware` + `ToolCallLimitMiddleware`.

### 3.3 Phase-gating (GATE on the action space)
One predicate `available_tools(state) -> subset`, applied by **pre-filtering the advertised
tool list** before the model sees it (strictly better than post-hoc rejection: the model never
proposes an illegal action). Captures "must `plan` before `act`," per-state allowed-tools, and
most of routing. State is a small named enum plus the transcript; a transition rule advances it
(e.g. a `transition` tool or a predicate on the trajectory). This is the highest expressiveness
per unit of config, and the upper bound of what stays declarative (statewright's JSON FSM is the
reference). Anything requiring procedural next-state computation is a custom env in Python, not
config.

### 3.4 Named lifecycle hooks (typed per §2)
A small fixed set, each pointing at a user (optimizer-authored) function in the harness dir, each
tagged with its kind:
- `on_step`, `on_tool_start`, `on_tool_end`, `on_episode_end` — **OBSERVER**.
- `before_tool(call) -> call | DENY` — **TRANSFORMER + GATE** (Claude Code PreToolUse).
- `after_tool(result) -> result` — **TRANSFORMER** (Claude Code PostToolUse; the natural place
  for a self-correction-feedback injection).
- `should_continue(state) -> bool` — **GATE** (Claude Code Stop).

### 3.5 Guardrails (GATE)
`input_guardrails` / `output_guardrails` returning a tripwire boolean, run before/after the loop
(OpenAI's model). Keeps blocking out of the observer hooks, where it does not belong.

## 4. The artifact

```toml
# control.toml — absent means: terminate-or-horizon, no hooks, no gating (the default)

[stopping]
max_turns = 12                    # may only tighten the env horizon
stop_when = "harness.control.stop:done"   # optional predicate (module:fn)

[retries.run_python]              # per-tool
max = 2
backoff = "exponential"
retry_on = ["TimeoutError", "ConnectionError"]

[gating]
predicate = "harness.control.gating:available_tools"   # state -> subset
states = ["plan", "act"]          # the named enum

[hooks]
after_tool = "harness.control.hooks:inject_lint_feedback"   # TRANSFORMER
on_step     = "harness.control.hooks:log_step"               # OBSERVER
```

Hook/predicate values are `module:function` pointers into the harness directory. The function
bodies are Python the optimizer writes; the config only names them and their kind. This is the
declarative/Python line in concrete form.

## 5. What does NOT go here

- **Multi-agent topology.** A subagent is a tool whose body runs an inner episode (RFC 000
  §7.2, RFC 002 §4). Planner/generator/evaluator are three tools orchestrated by the outer loop
  plus phase-gating, not a separate control construct. There is no topology config.
- **Handoffs that replace the active agent mid-run** (swap system prompt + tool set + identity).
  This is the one control pattern that does not collapse into tools, because it mutates the
  loop's configuration rather than returning a value. It is explicitly **out of scope for
  declarative config**; if needed, it is a custom agent in Python. Keeping it out is the
  difference between a config surface and a workflow engine.
- **Any conditional/loop/edge syntax.** The moment a user needs `if X then go to Y else loop`,
  that is a graph, and it belongs in a Python env/agent, not `control.toml`.

## 6. The critical risk: hooks are optimizer-authored code in the trusted loop

Hooks and the gating predicate are Python functions the *optimizer* may write, running in the
rollout process — which can reach the env and its ground truth. A `before_tool` transformer that
reads the env's gold answer and injects it into the tool call, or an `after_tool` hook that
rewrites a result to smuggle the answer to the model, is total Goodhart, and worse than the
tool-process hole (RFC 002 §6.1) because hooks run *in-process by design*.

**This is the most dangerous surface and needs the strongest guardrail.** Proposal (RFC 005
owns enforcement): hooks receive a **restricted view** — the trajectory, the current tool
call/result, and harness config — and provably *not* the env instance, the verifier, the gold
label, or the task record. The hook API passes data, not the `Env` object. A hook that needs
env internals is, by definition, changing the task, not the harness, and must be rejected. Until
this restricted-view boundary exists, optimizer-authored hooks should be disabled in
attribution runs (allowed only for trusted, human-authored harnesses). This gate is a
precondition for opening the control surface to an optimizer at all.

## 7. Prior art

Claude Code hooks (the cleanest gate/transform/observer taxonomy; PreToolUse gate+transform,
PostToolUse transform, Stop continuation-gate, and the deliberate asymmetry that
UserPromptSubmit can inject/reject but not silently rewrite intent); LangChain agent middleware
(named built-ins for retry/limits/summarization/HITL/fallback + one `wrap_*` escape hatch — the
exact "named knobs + one Python hatch" boundary); OpenAI Agents SDK (observer-only lifecycle
hooks + guardrails-for-blocking discipline; `max_turns` horizon; agent-as-tool); statewright
(declarative JSON FSM tool-gating via pre-filtering); self-healing orchestrator literature (the
transient/invalid-input/fatal failure trichotomy). Full URLs in the research transcript.

## 8. Open questions

1. **Hook restricted-view API (§6).** Exactly what does a hook receive? Leaning:
   `(trajectory, current_call_or_result, harness_config) -> typed return`, with no `Env` handle.
   The precise type is the most important detail in this RFC.
2. **Is `stop_when` worth it over `terminate` + `max_turns`?** The `terminate` tool already lets
   the agent self-stop; `stop_when` lets the *harness* stop on a trajectory predicate (e.g.
   "stop once `submit_answer` was called"). Probably yes for envs where the agent over-runs, but
   it is the most DSL-adjacent knob; watch it.
3. **Phase state representation.** A named enum + transition rule (statewright) vs a free predicate
   over the trajectory. The enum is more readable and more attributable (the state is loggable);
   the free predicate is more expressive but slides toward a DSL. Leaning enum.
4. **Do retries belong here or in execution (RFC 005)?** Retries are loop behavior (control) but
   per-tool caps feel execution-policy-ish. Leaning control for repair semantics, execution for
   hard budget caps; the line is fuzzy.
