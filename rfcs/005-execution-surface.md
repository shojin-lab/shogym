# RFC 005: The execution surface (renamed from "environment")

- **Status:** Draft (proposed 2026-06-13)
- **Depends on:** RFC 000 (esp. §7.3); enforces RFC 002 §6.1 and RFC 004 §6
- **Locus:** **rollout** (run-level policy); the per-tool substrate rides with each tool (RFC 002)
- **State today:** partial. `MCPServerSpec.{transport, env, headers, command, url}` already
  carries per-tool execution context; there is no run-level policy layer (no budgets,
  permissions, or isolation enforcement).

---

## 1. Rename, and why

The wiki's fifth surface was "environment," which collides with the `Env` object and confuses
"the task" with "where tool code runs." This RFC renames it **execution** and scopes it to its
one cleanly-optimizable part. (RFC 000 §7.3.)

## 2. The surface is two things; only one is optimizable here

- **Per-tool execution substrate** (transport, sandbox, env vars, credentials for a given MCP
  server). This is *inseparable from the tool* — an MCP server delivers interface and execution
  together — so it lives in `tools.toml` per RFC 002 §5, not here. Adding a tool inherently
  chooses where it runs.
- **Run-level policy** (budgets, permissions, egress, isolation defaults). This *is* a clean
  rollout attribute: it applies across all tools and the whole episode/run, independent of any
  one tool. This is the execution surface's optimizable artifact.

So the execution surface is the smallest of the seven: it is *run-level policy plus the
enforcement of the isolation guarantees the other RFCs depend on.*

## 3. What it controls

### 3.1 Budgets (the most important optimizable knob)
Caps the optimizer (and the agent) must live within: cost ($), tokens, wall-clock, and tool-call
counts, at run / episode / per-tool granularity. Budgets matter for two reasons: (a) they bound
the optimizer-workflow's spend (Willison's $5-Fly.io-org discipline), and (b) "how good can the
harness get *under a fixed budget*" is itself a research axis — a harness that wins only by
spending 10x is a different finding than one that wins at parity. Budgets make cost a
first-class, attributable dimension rather than an afterthought.

### 3.2 Permissions / egress
Network egress (allow/deny/allowlist), filesystem scope, and the permission posture for tool
execution ("YOLO mode" only inside a sandbox without network, per Willison/Anthropic). The
default is deny-by-default egress for optimizer-authored tools; opening it is an explicit,
recorded choice.

### 3.3 Isolation enforcement (this surface's special duty)
The Goodhart guards that RFC 002 §6.1 (tool processes) and RFC 004 §6 (hook restricted-view)
*name* are *enforced here*. The execution surface owns:
- **Ground-truth unreachability:** no optimizer-authored tool process and no hook may reach the
  env instance, the verifier, the gold label, or the task record. For `stdio`/`http` tools this
  is process/container isolation; for hooks it is the restricted-view API (data, not the `Env`).
- **The in-process-extras decision (RFC 002 §6.1 open q1):** the proposal here is to *enforce*
  "optimizer-authored extras run process-isolated (`stdio`+), never `in_process`" as run policy.
  `in_process` stays available for trusted, human-authored, env-mandatory servers.

This is why execution, though the smallest surface, is not optional: it is where "the optimizer
cannot cheat" stops being a principle and becomes a runtime guarantee.

## 4. The artifact

```toml
# execution.toml — absent means: default budgets off, egress denied, extras process-isolated

[budget]
max_cost_usd = 5.0           # per run
max_tokens = 2_000_000
max_wall_seconds = 1800
max_tool_calls = 500

[permissions]
egress = "deny"              # deny | allow | ["allowlisted.host"]
filesystem = "harness/workspace"   # scope tool fs access

[isolation]
extras_transport = "stdio"  # enforce: optimizer extras are process-isolated, never in_process
```

## 5. Is this really separable from "tool" and from `harness.toml` limits?

Honest answer (RFC 000 §9.3): the run-level policy *is* separable (it spans all tools), but it
is thin. Two outcomes are possible and we should let the implementation decide:
- If budgets + permissions + isolation justify their own file (likely, because isolation is
  security-critical and should be explicit and diffable, not buried), keep `execution.toml`.
- If it collapses to three scalars, fold them into `harness.toml [limits]` and let the per-tool
  substrate (tools.toml) carry the rest — and the seven surfaces become six. The
  isolation-enforcement duty (§3.3) is the strongest argument for keeping it distinct: security
  policy deserves its own reviewed, hashed artifact.

Leaning: keep it, because the isolation guarantees are too important to inline.

## 6. Prior art

Willison, "Designing agentic loops" (sandboxing, credentials, tight budget caps, YOLO-only-in-a-
container); Anthropic (dangerously-skip-permissions only without network); OSWorld/containerized
envs (the heavyweight execution substrate hgym fronts via transports); Harness-Bench's
episode-scoped stdlib proxy and its grade-time-execution hole (both lessons:
`lit-reviews/harness-bench-code-review.md` §3.4, §4.7).

## 7. Open questions

1. **Keep `execution.toml` or fold into `harness.toml` (§5)?** Leaning keep, for the isolation
   artifact. Decide when the run-level policy's real size is known.
2. **Budget attribution granularity.** Per-tool budgets (RFC 002 §9.2) vs per-run only. Per-tool
   is more expressive but pushes budget config toward the tool spec; per-run is simpler. Possibly
   both, with per-tool caps in `tools.toml` and aggregate caps here.
3. **Enforcement mechanism for ground-truth unreachability.** Process isolation handles tool
   processes; the hook restricted-view (RFC 004 §6) handles hooks; but `in_process` *mandatory*
   tools still share the env process — is that a hole? (No, because mandatory tools are env-
   authored and trusted; only optimizer-authored code is the threat.) Confirm the trust model is
   exactly "env-authored = trusted, optimizer-authored = isolated."
