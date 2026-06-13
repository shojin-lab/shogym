# hgym surface RFCs

Draft RFCs for making every harness surface optimizable, cleanly. Proposed 2026-06-13.
**Status: all Draft, for Andrew's review.** Start with RFC 000; it is the spine the other
seven assume.

| RFC | Surface | One-line thesis | State today |
|---|---|---|---|
| [000](000-surfaces-and-the-optimizability-rule.md) | **foundation** | optimizable ⟺ rollout/agent attribute; fixed ⟺ env/verifier attribute | — |
| [001](001-instruction-surface.md) | instruction | shape is the env's, content is the harness's | partial |
| [002](002-tool-surface.md) | tool | mandatory = env (fixed), extras = harness (optimizable); subagents are tools | built |
| [003](003-context-surface.md) | context | a rollout transform over the immutable trajectory; replay class drives everything | **unbuilt + mis-located today** |
| [004](004-control-surface.md) | control | five declarative knobs, observer/transformer/gate typing, no DSL | unbuilt |
| [005](005-execution-surface.md) | execution (was "environment") | run-level policy + the isolation guarantees the others depend on | partial |
| [006](006-verification-surface.md) | verification | deliberately closed; self-verification is a tool, not the verifier | built |
| [007](007-observability-surface.md) | observability | the substrate; per-surface hashing makes attribution a query; = the M1 trace schema | unbuilt |

## The one rule everything hangs on

Classify every part by its **locus** — environment (the task), agent (the policy), or rollout
(the loop). Then: **a surface is optimizable exactly when it is a rollout or agent attribute;
it is fixed when it is an environment or verifier attribute.** The **harness** is the editable
projection of the rollout-and-agent config, holding env and verifier fixed. That is what
`export_harness` exports and what an optimizer edits.

## Cross-cutting decisions (made across the set)

1. **One file per surface, absent by default** (RFC 000 §5). Diffs self-classify; a surface's
   hash is its file's hash; the directory's size equals the number of surfaces the optimizer
   has touched. This is the config-dir readability answer.
2. **The trust model: env-authored = trusted, optimizer-authored = isolated.** The Goodhart
   thread runs through tool (002 §6), control (004 §6), and execution (005 §3.3): no
   optimizer-authored tool process or hook may reach the env's ground truth or verifier.
   Execution enforces it. Until that boundary exists, optimizer-authored tools/hooks are
   disabled in attribution runs. **This is the most important safety decision in the program.**
3. **Replay determinism is a first-class constraint** because hgym is an attribution library,
   not just an agent framework (RFC 003 §2, RFC 007 §4). Context strategies are typed
   pure/event/stateful; the type dictates what observability must persist. Pure ships first.
4. **Per-surface hashing** (RFC 007 §3) makes the central scientific claim ("which surface
   caused the delta") a `group by`, not bookkeeping.
5. **The model is the substrate, not a surface** (RFC 001 §6). It lives in `harness.toml`;
   report model-swaps and surface-edits on separate axes.

## Thesis revisions to the current wiki design docs

- **(Significant) Context is mis-located.** The env currently owns the trajectory and emits it
  whole as the model's view; the "view" is hard-wired to the identity. RFC 003 separates the
  immutable **trajectory** (env, verified, logged) from the **view** (a rollout transform,
  harness-configured). This is the deepest change and touches the env/agent boundary.
- **(Clarifying) Multi-agent topology is not a surface** — a subagent is a tool whose body runs
  an inner episode against a nested harness (RFC 000 §7.2, RFC 002 §4). Control shrinks to
  loop + hooks.
- **(Clarifying) "Environment" → "execution"**, scoped to run-level policy; the per-tool
  substrate rides with the tool (RFC 000 §7.3, RFC 005). Note: RFC 000's tables still say
  `environment.toml` in places; RFC 005 is the authority on the rename. Align on a final pass.

## Open tensions I could not fully resolve (your calls)

- **Could the seven be six?** Execution may fold into `harness.toml` limits if its run-level
  policy turns out thin; the isolation-artifact duty is the main reason to keep it (RFC 005 §5).
- **Could the seven be the wrong grain entirely?** The locus analysis suggests the deepest cut
  is three (env/rollout/agent), with "surfaces" a finer user-facing slicing. Treat the loci as
  the architecture and the surfaces as the editable artifacts; where a surface fails to map to
  one locus (tool and execution both straddle env/rollout), prefer the locus (RFC 000 §9.6).
- **Verification stays closed — is that right for *your* program?** I argued yes (a ruler the
  optimizer edits measures nothing), with judge-robustness as a separate experiment *mode*, not
  a harness surface (RFC 006 §4). If you want the optimizer to ever touch judges, that is the
  thesis to revisit first.
- **Hooks are the most dangerous surface** (optimizer code in the trusted loop). The
  restricted-view API (RFC 004 §6) is the gating precondition and its exact type is unspecified.

## Suggested reading order

000 (foundation) → 002 + 001 (the built/partial surfaces, to ground the model) →
003 (the big revision) → 004 → 005 → 006 + 007 (the two "special" surfaces). The index above
links each.

## Provenance

Grounded in the ported v1 code (`tool_using_env.py`, `runner.py`, `mcp/`, `types/`), the wiki
design docs (surfaces + roadmap), the Harness-Bench code review
(`lit-reviews/harness-bench-code-review.md`), and two June-2026 research passes (context-management
strategies; control/hooks/middleware patterns) whose sources are cited inline in RFCs 003-004.
