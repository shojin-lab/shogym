# Roadmap: opening every surface to optimization

*The long-term plan. Companion to [surfaces.md](surfaces.md).*

## The principle

hgym's goal is to make every harness surface **optimizable in the cleanest, most
user-friendly way possible**. Concretely, "open" means a surface satisfies all four:

1. **Editable** — it is a file (or a few lines) in the exported `harness/` directory:
   human-diffable, machine-patchable, no Python forks required.
2. **Validated** — `load_harness` rejects ill-formed edits at load time with
   actionable errors, never mid-episode.
3. **Attributable** — the trace schema records which surface configuration produced
   each episode (harness hash + per-surface sub-hashes), so ablations are queries,
   not bookkeeping.
4. **Safe by default** — an unedited harness runs correctly; every opened knob has a
   default that a new user never has to think about.

The unit of edit is the lever. An optimizer (human, Claude Code, or program) should
be able to range over any surface by editing files and re-running — nothing more.

## Surface-by-surface plan

| Surface | Status | The artifact | Opens in |
|---|---|---|---|
| 1. Instruction | **Open at M1** | `harness/templates/*.minijinja`, `harness.toml` (model, params) | M1 (`export_harness`/`load_harness`) |
| 2. Tool | **Open (v1 core)** | `harness/extras.toml` (`[[mcp_servers]]`), optimizer-authored FastMCP server files | shipped in the ported core; runner wiring at M1 |
| 3. Context | Designed (RFC 002) | `[context]` in `harness.toml`: v0 = a truncation knob (`on_tool_result` max bytes); later: named `ContextStrategy` (compaction, resets, artifact handoffs) | v0 knob: M2 · strategies: M3 |
| 4. Control | Partially open | `horizon` + reserved `terminate` (open now); `available_tools(state)` phase gating (API reserved); hooks/middleware (RFC 003) as a `[hooks]` config table | gating: M2 · hooks: M3 |
| 5. Environment | Partially open | model/provider via `harness.toml` (M1); per-run budgets (M2); sandbox/transport spec for `streamable_http` envs (sandbox tier) | budgets: M2 · sandbox: M3+ |
| 6. Verification | **Deliberately closed** | Env-owned pure `_verify`; not exported into `harness/` | see Goodhart stance below |
| 7. Observability | Substrate, always on | JSONL trace schema (versioned, with per-surface hashes), `analysis` dataframes, optional OTel | M1; never an optimization target |

## Design stances

**Verification is the measuring stick, not a lever.** If the optimizer can edit the
verifier, every result is Goodharted by construction. `_verify` stays env-owned and
pure. The one principled exception, later: *judge configuration* (rubric text,
judge model) may become an exportable artifact — but always attributed and reported
separately, never silently co-optimized with the surfaces being measured.

**Open surfaces one at a time, with attribution.** Each newly opened surface ships
with (a) its harness artifact, (b) trace-schema support for isolating it, and (c) an
ablation recipe in docs — the experiment that measures the surface's contribution on
a reference env. Opening a surface without the measurement is how harnesses bloat
(every component is an assumption about model limitations — assumptions need expiry
dates).

**User-friendliness ratchet.** Every milestone keeps the two-minute quickstart true:
`pip install hgym`, one API key, no infra. A surface that can't be opened without a
server or a database isn't ready to open.

**Subagents are tools.** When inner-agent tools land, an inner agent's harness is a
nested `harness/` directory — the same artifacts, recursively. No new ontology.

## Sequencing (tied to releases)

- **M1 (now):** instruction surface opens; trace schema lands with per-surface
  hashes and the synthesis-consumer fields; tool surface gets runner wiring.
  → First attribution experiment: prompt surface vs tool surface (blog post, July 1).
- **M2:** context v0 (truncation knob), phase gating, budgets.
  → Attribution experiment: does the context knob matter where tools didn't?
- **M3:** named context strategies, hooks/middleware, sandbox-tier environments.
  → The full inner harness is config; outer-harness (multi-loop) design begins.
- **Beyond:** outer-harness topologies (planner/evaluator splits) as config; the
  optimizer-workflow matured into a benchmark of *optimizers* (which agent improves
  harnesses best, per surface, per env class).

## Non-goals

- A gateway, a database, or any always-on service.
- Optimizing model weights (post-training); hgym measures and edits harnesses.
- A universal agent framework — hgym is the measurement substrate and the editable
  harness format; bring your own agent.
