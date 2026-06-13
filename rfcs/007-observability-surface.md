# RFC 007: The observability surface (the substrate; the trace schema)

- **Status:** Draft (proposed 2026-06-13)
- **Depends on:** RFC 000; consumes the per-surface boundaries from RFCs 001-006
- **Locus:** substrate — always on, emitted by the rollout, **never optimized**
- **State today:** unbuilt. No trace store, no `export_harness`/`load_harness`, no per-surface
  hashing. This RFC doubles as the **M1 trace-schema design** that P1 and P2 depend on.

---

## 1. The stance, and why it is not an optimization target

Observability is **what the engineer and the optimizer see**: traces, costs, the run ledger. It
is the substrate that makes the other six surfaces *attributable*. It is deliberately not
optimizable, for the same reason the verifier is not (RFC 006): a record the optimizer can
configure is no longer a neutral record — you cannot optimize the ruler and then trust the
measurement. The runner emits observability; nothing in `harness/` configures it.

Its job is singular and load-bearing: **make "which surface caused the delta" a query.**

## 2. The trace schema (one JSONL record per step, one per episode)

Zero-infra (RFC 000, the project's non-negotiable): append-only JSONL under `./hgym_logs/`, one
file per run, plus a SQLite run-ledger for mutable state. Each **step** record:

```jsonc
{
  "run_id": "...", "episode_id": "...", "step": 3,
  "env": "tau2_bench_telecom_v1", "task_idx": 17, "seed": 4,
  "model": "openai/gpt-5.4-nano",
  "harness_hash": "h:ab12…",                 // hash of the whole harness dir
  "surface_hashes": {                         // per-surface sub-hashes (§3)
    "instruction": "i:9f…", "tool": "t:00…(default)",
    "context": "c:default", "control": "k:default",
    "execution": "x:default", "model": "m:gpt-5.4-nano",
    "verifier": "v:env@1.2"                   // recorded for provenance, never optimized
  },
  "view": {                                   // what the model actually saw (RFC 003)
    "context_class": "pure|event|stateful",
    "derived_state_ref": null                 // pointer to persisted summary/handoff/snapshot
  },
  "action": [ … ],                            // the agent's emitted action
  "tool_calls": [ { "name": "respond_to_user", "args_hash": "…", "provenance": "env-mandatory" } ],
  "tool_results": [ … ],                      // _session_id stripped (RFC 002)
  "feedback": { "scalar": 0.0, "checks": [ {"id":"db_match","weight":0.5,"passed":false} ] },
  "cost": { "tokens_in": 1840, "tokens_out": 210, "usd": 0.0007,
            "attributed_to": "agent" },       // or "context" for a summarizer call (RFC 003 §7)
  "timing": { "inference_s": 1.2, "tool_s": 0.4 }
}
```

Episode records carry the terminal feedback, the total cost, the termination reason
(`terminate` / horizon / guardrail), and the full surface-hash set.

## 3. Per-surface hashing (resolving RFC 002 §9.3)

The central mechanism. Each surface hash covers exactly that surface's editable artifact, so a
diff to one surface moves one hash, and an ablation is `group by surface_hashes.<surface>`:

| Surface | Hash covers | Notes |
|---|---|---|
| instruction | template + skill text + **extras tool descriptions** | description-only edits move *this* hash, not tool |
| tool | the extras **server specs** (name/transport/module), not their descriptions | a spec change (added tool) moves this |
| context | `context.toml` + the `class` (pure/event/stateful) | derived-state pointer recorded in `view` |
| control | `control.toml` + the **source of referenced hook/predicate functions** | a hook body edit changes behavior, so hash its source |
| execution | `execution.toml` | budgets/permissions/isolation policy |
| model | the model string | the substrate axis (RFC 001 §6) |
| verifier | env + verifier version | **fixed**; recorded for provenance and to detect accidental drift, never an optimization axis |

The instruction/tool split on tool *descriptions* (a description is instruction content per
RFC 001 §4, even though it lives near the tool) is what makes the two hashes clean: hash the
spec for tool, the text for instruction. A pure rewording of an extra's description is correctly
attributed to instruction.

## 4. Replay/persistence contract (consumes RFC 003 §2)

Observability must persist exactly what each context-strategy class needs to replay (RFC 003):
- **pure** strategies: nothing beyond the trajectory; `derived_state_ref` is null.
- **event** strategies (`compact`, `reset`): persist the generated summary/handoff artifact and
  the boundary index; `derived_state_ref` points at it.
- **stateful** strategies (`retrieve`, `memory`): snapshot the store per step and pin the
  embed-model version; `derived_state_ref` points at the snapshot.

A run whose `context.class` is event/stateful but whose records lack `derived_state_ref` is a
broken trace (replay would diverge) and must fail loudly. This is the contract that keeps
non-pure context strategies attributable.

## 5. Analysis layer and optional exporters

- **Dataframes** (`hgym.analysis`): `runs_df`, `episodes_df`, `steps_df`, `feedback_df`,
  `checks_df` — pandas readers over the JSONL, plus optional in-place DuckDB SQL. The attribution
  query ("Δ success by `surface_hashes.tool`, holding all else") is one `group by`.
- **Optional OTel exporter** (`[otel]` extra): emit GenAI-semantic-convention spans for users who
  want Langfuse/Phoenix. Pinned to a convention version (still experimental in 2026). Core never
  imports it.
- **Synthesis-consumer fields (P2):** the schema deliberately records function/template metadata
  and the harness hash so the dataset-synthesis skill (P2) can read a harness + its real traces
  and generate plausible synthetic episodes in the same format. Designing the schema with P2 as a
  named consumer now is the cheap-insurance decision the research plan flagged.

## 6. Why always-on and never configured

If the optimizer could turn observability off or down (sample less, drop checks, coarsen cost),
it could hide its own gaming and make ablations incomparable. So observability is emitted
unconditionally by the runner at a fixed schema version; the only "config" is the log directory
path, which is a deployment detail, not a harness surface. The schema is versioned so it can
evolve, but never per-run by the optimizer.

## 7. Open questions

1. **Hashing hook source (§3, control row).** Hashing the *source* of referenced functions ties
   the control hash to code, not just config. Right (behavior lives in the code), but it means a
   whitespace edit to a hook moves the hash. Acceptable; note it.
2. **Snapshot cost for stateful context (§4).** Per-step store snapshots could dominate trace
   size for `memory`/`retrieve`. May need delta-snapshots or a "this run is capability-demo, not
   attribution-grade" flag. Ties to RFC 003 §8's "supported but not attributable" possibility.
3. **Check schema (§2 feedback.checks).** Depends on RFC 006 §7 making `checks` first-class in
   `FeedbackCollection`. If it stays env-convention, `checks_df` is best-effort. Strongly prefer
   first-class.
4. **Run ledger contents.** What belongs in the SQLite ledger (mutable: experiment registry,
   optimizer generations, surface-hash → file mapping) vs the immutable JSONL. Draft: JSONL is
   the immutable per-step record; SQLite indexes runs and maps hashes to harness snapshots.
