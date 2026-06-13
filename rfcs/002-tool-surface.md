# RFC 002: The tool surface

- **Status:** Draft (proposed 2026-06-13)
- **Depends on:** RFC 000; couples with RFC 005 (execution)
- **Locus:** mandatory tools = environment (fixed); extras = rollout (optimizable);
  the surface the model sees = their union, composed at construction time
- **State today:** built. `ToolUsingEnv` merges terminate + `mcp_servers` +
  `extra_toolset` (`tool_using_env.py:145-209`); extras load from
  `hgym_extras.toml` (`mcp/config.py`). This RFC formalizes the optimization story
  and flags two real risks.

---

## 1. What it is

The tool surface is **what the model can do**: the set of MCP tools advertised each step. It
is hgym's thesis surface — "the tool surface is the policy boundary." Same env, same model:
`{search, answer, terminate}` is one program, `{search, draft, critique, revise, terminate}`
is another. Optimizing the tool surface is editing the program's action space.

## 2. Locus: mandatory fixed, extras optimizable

From RFC 000 §3 corollary 1, the surface is two pools:

- **Mandatory (env-owned, fixed):** `ToolUsingEnv.mcp_servers` — the servers the verifier
  needs evidence from (`guess`, `submit_answer`, the simulated user), plus the reserved
  `terminate`. The optimizer may **never** remove, add to, or reword these: they are the
  task's interface and the verifier reads their effects. Removing `submit_answer` would make
  the task unscorable; that is not an optimization, it is sabotage.
- **Extras (rollout-owned, optimizable):** the `extra_toolset`, threaded through
  `make(..., extra_toolset=...)`. This is the lever: `think`, `plan`, a scratchpad,
  `run_python`, `web_fetch`, or a bespoke server the optimizer wrote for this task.

The union is computed once at `__init__` with conflict detection
(`ToolNameConflictError`), attributed to the right side of the boundary so the error names
where to look. That conflict-at-construction behavior is exactly right and stays.

## 3. The artifact

Rename `hgym_extras.toml` → `tools.toml` (RFC 000 §5: one consistent per-surface naming
scheme). Content unchanged in spirit:

```toml
# tools.toml — the extras pool (the optimizable tool surface)
[[mcp_servers]]
name = "think"
transport = "in_process"
module = "harness.tools.think_mcp"        # a module the optimizer wrote, in the harness dir

[[mcp_servers]]
name = "run_python"
transport = "stdio"
command = ["uv", "run", "python", "-m", "harness.tools.py_sandbox"]

[[mcp_servers]]
name = "web_fetch"
transport = "streamable_http"
url = "https://fetch.internal/mcp"
```

The optimizer engages the tool surface by (a) creating `tools.toml` and (b), for tools it
authors, writing the MCP server file alongside it (e.g. `harness/tools/think_mcp.py`). The
"unit of edit" for this surface is *a server spec plus possibly a server implementation*.
This is the optimizer-workflow's keystone: an optimizing agent can literally write a new
tool and add it to the surface, then re-run.

## 4. Subagents are tools (topology collapses here)

Per RFC 000 §7.2, a subagent is a tool whose body runs an inner episode against an inner
harness. So "multi-agent topology" is not a separate surface; it is the tool surface used
recursively. Concretely: an extras MCP server whose `call_tool` implementation runs
`run_episode(inner_env, harness="harness/tools/critic_harness/")` and returns the inner
result. The inner harness is a **nested harness directory**, optimizable by the same rules,
attributable by the same per-surface hashing (the inner harness's hash is part of the outer
tool surface's hash). No new ontology: planner/generator/evaluator splits are just three
tools whose bodies are inner episodes. This is the single biggest simplification the locus
analysis buys.

## 5. Execution rides with the tool (the coupling with RFC 005)

An MCP server delivers both an **interface** (the tools it advertises) and an **execution
context** (where its code runs: in-process, a subprocess, a remote container). The
`MCPServerSpec` already carries this: `transport` (the sandbox boundary), `env`
(environment variables), `headers`, `command`/`url`. So editing the tool surface inherently
brings execution context along; you cannot add a tool without choosing where it runs.

RFC 000 §7.3 names this honestly: the "execution surface" (RFC 005) is only cleanly
separable into *run-level policy* (budgets, egress, permissions — rollout-owned, its own
artifact). The *per-tool* execution substrate (transport/sandbox) stays here, in the tool
spec, because it is inseparable from the tool. RFC 005 picks up the run-level half.

## 6. Two real risks (the critical part)

### 6.1 Goodhart via tool access to ground truth

An optimizer-authored tool runs code. If an `in_process` extra shares the env's Python
process, nothing structural stops it from importing env internals and reading the answer,
the task's gold label, or the verifier's expectations — then "solving" the task by leaking
the answer to the model. That is total Goodhart, and the v1 core does not currently prevent
it (in-process FastMCP servers share the process). This is the tool-surface analogue of
Harness-Bench's "grade-time execution of agent-controlled code" anti-pattern
(`lit-reviews/harness-bench-code-review.md` §4.7).

**Mitigations (RFC 005 owns the policy, but the threat is named here):**
- The env's ground truth (gold answers, the verifier, the task record) must not be reachable
  from an extras tool's process. For `stdio`/`streamable_http` extras this is natural (separate
  process/container, no shared memory). For `in_process` extras it is *not* guaranteed.
- Proposal: **in-process extras are a trust boundary.** Either (a) restrict `in_process` to
  env-mandatory tools and require extras to be `stdio`+ (process isolation by default), or (b)
  keep `in_process` extras but document them as "trusted, non-adversarial use only" and forbid
  them in any optimizer-driven run. Leaning (a): the optimizer-authored surface should never run
  in the env's process. This is a real change to the default and is the most important security
  decision in the surface program.

### 6.2 The advertised-surface duplication contract is gateway-fragile

`tool_using_env.py:84-101` documents a subtle contract: tools are advertised both statically
(`function.tools_available`) and dynamically (`Observation.tools`), and a deployed config that
*also* declares the tools statically will 400 on duplicates. The RFC-001-era fix ("deployed
configs must declare a bare function, supply tools via the observation") works but is a
landmine for users wiring hgym to an external gateway. The "gateway-agnostic alternative"
noted in the docstring (tools_available = mandatory-only; carry runtime extras as an explicit
observation field) should be adopted before the tool surface is heavily optimized, because
optimizer-added extras are exactly the tools that would trip it. Flagged as a precondition,
not part of this RFC's core.

## 7. Prior art

- **MCP** as the tool standard; no other gym is MCP-native (lit-reviews/hgym-landscape §(c)).
- **Anthropic, "Writing effective tools for agents"** (tools as contracts; ten focused tools
  beat fifty).
- **Harness-Bench** (`lit-reviews/harness-bench-code-review.md`): the tool/execution coupling
  and the grade-time-execution hole are both lessons taken from its review.

## 8. Alternatives considered

- **Let the optimizer edit mandatory tools.** Rejected: that edits the task interface; the
  verifier's evidence comes from mandatory tools, so editing them is editing the measurement.
- **A flat tool allowlist instead of MCP specs.** Rejected: loses execution context (transport
  /sandbox) and the ability for the optimizer to *author* a tool, which is the whole point.
- **Topology as its own surface.** Rejected per §4: it is tools-recursively; a separate surface
  would duplicate the harness machinery.

## 9. Open questions

1. **in-process extras: ban or allow-with-warning (§6.1)?** The most consequential call. Leaning
   ban for optimizer runs.
2. **Per-tool budgets** (this tool may be called ≤N times / cost ≤$X): tool-surface or
   execution-surface (RFC 005)? Leaning execution, but per-tool caps live closest to the tool
   spec. Possible `tools.toml` per-entry `limits`.
3. **Tool *description* edits on extras** are instruction content (RFC 001 §4) but live in
   `tools.toml`/the server file. Attribution: a description-only change to an extra is arguably
   an instruction edit, not a tool-surface edit. Does the per-surface hash double-count? Resolve
   in RFC 007 (hash the *spec* for tool-surface, the *description text* for instruction, so a
   description-only change moves only the instruction hash).
