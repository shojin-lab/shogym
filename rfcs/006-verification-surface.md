# RFC 006: The verification surface (why it stays closed)

- **Status:** Draft (proposed 2026-06-13)
- **Depends on:** RFC 000; RFC 002 §6.1 (process isolation)
- **Locus:** environment / verifier — **fixed by design, not optimizable**
- **State today:** built and correct. `_verify(trajectory, task, *, terminated)` is a
  pure function over the recorded trajectory, env-owned (`tool_using_env.py:245`).

---

## 1. The stance

Verification is the one surface hgym deliberately does **not** open to the optimizer. This
RFC defends that as a positive design choice, draws the one bright line people get wrong
(self-verification tools vs the verifier), and specifies the narrow research exception so it
cannot leak into the optimization loop.

**The rule:** the optimizer may never edit `_verify`, the metrics, or any judge/rubric the
env uses to score. If it could, every result would be Goodharted by construction — the
optimizer would "improve" the score by lowering the bar, and the number would mean nothing.
The verifier is the measuring stick; you do not let the thing being measured hold the ruler.

## 2. Why closed is correct, not a limitation

- **Attribution requires a fixed referent.** RFC 000's whole program is "which surface caused
  the delta." A delta is only meaningful against an unchanging verifier. A moving verifier
  makes every comparison incoherent.
- **Optimizer-proof scoring is the product.** hgym's value over end-to-end harness search
  (Meta-Harness) and over Harness-Bench is *trustworthy* numbers. The verifier being outside
  the optimizer's reach is what makes them trustworthy.
- **Purity is already the design.** `_verify` is a pure function over the trajectory; it has no
  hidden state and no dependence on the harness. Keeping it env-owned costs nothing and is
  already true in the code.

## 3. The bright line: self-verification is a TOOL, not the verifier

The error everyone makes (Harness-Bench among them,
`lit-reviews/harness-bench-code-review.md` §4.7) is conflating two different things:

- **Agent-callable check tools** — `run_tests`, `lint`, `typecheck`, `compile`, a calculator
  the agent calls mid-trajectory to check its own work. These are the **tool surface**
  (RFC 002), fully optimizable. An optimizer adding a `run_tests` tool so the agent can
  self-correct is a tool-surface edit and a legitimate, encouraged move.
- **The verifier** — `_verify`, which runs *after the agent exits*, scores the episode, and
  the agent can neither see nor call. This is the verification surface, closed.

The distinction is mechanical: anything the agent can invoke during the episode is a tool
(open); the post-hoc scorer the agent cannot touch is the verifier (closed). An env author
keeps them physically separate (the verifier runs in the env's trusted context, after
teardown), and per RFC 002 §6.1 the env's verifier and ground truth must not be reachable
from any optimizer-authored tool process.

## 4. Judges and rubrics: env-owned, with a fenced research exception

Some envs verify with an LLM judge against a rubric (tau2's reward bases, arena-hard,
healthbench). The judge model and rubric are **part of the verifier**, hence env-owned and
closed. The optimizer never edits the rubric to make its outputs score better.

There is one legitimate reason to vary a judge: **research on judge robustness** (does the
ranking hold under a different judge model or rubric phrasing?). That is a *separate
experiment mode*, not a harness surface, and the boundary must be unambiguous:

- In **optimization mode** (the default), the judge config is frozen and lives with the env.
  The harness directory has no judge artifact. There is no path by which an optimizer edit
  reaches the judge.
- In **judge-robustness mode** (an explicit, separate experiment the researcher runs), the
  judge config is swept *by the researcher*, holding the harness fixed, and every result is
  reported per-judge. This is the mirror image of surface attribution: vary the measurement,
  hold the harness, to characterize the measurement. It is never co-run with surface
  optimization, because varying ruler and surface together confounds both.

Making this a *mode*, not a config file in `harness/`, is what keeps it from leaking. If a
judge config ever appears in the harness directory, that is a bug.

## 5. Recommended (env-side) verifier quality patterns

Closed does not mean crude. Two patterns from the Harness-Bench review are worth adopting on
the *env-authoring* side (they improve attribution without opening the surface):

- **Weighted named checks** (`harness-bench-code-review.md` §3.1): `_verify` returns a list of
  `{id, weight, passed, detail}` alongside the scalar, recorded in the trace. Then attribution
  sharpens from task-level to *check-level*: a per-check pass-rate diff across surface hashes
  tells you which surface edit fixed which sub-criterion. This is an env-side enrichment of the
  verifier output, fully compatible with the surface staying closed, and it makes RFC 007's
  attribution far more powerful. Strongly recommended.
- **Verifier self-test** (`harness-bench-code-review.md` §3.8): ship each env's `_verify` with a
  tiny fixture proving it scores known-good and known-bad trajectories correctly, so a broken
  verifier (the "8 of 28 silently zero-weight" failure) is caught at CI, not in results.

## 6. The honest limit

A fixed verifier can still be gamed: a capable agent may find a spurious trajectory that
satisfies `_verify` without solving the task (reward hacking). hgym does not prevent this, and
should not pretend to. But note: *measuring where and how the agent games a fixed metric is
itself a finding*, and a valuable one for the harness-attribution program (which surfaces make
gaming more or less likely?). The defenses are (a) verifier quality, the env author's job, and
(b) reporting trajectories, not just scores, so gaming is visible. An optimizer-editable
verifier would hide gaming; a fixed one surfaces it.

## 7. Open questions

1. **Check-level output (§5) in the type system.** Should `FeedbackCollection` gain a typed
   `checks` field so weighted named checks are first-class (and RFC 007 can hash/group on them),
   or stay env-convention? Leaning first-class — it is the single highest-leverage attribution
   enrichment.
2. **Where exactly does the verifier run relative to optimizer-authored tool processes?** RFC
   002 §6.1 says ground truth must be unreachable from extras; this RFC's §3 says the verifier is
   trusted-context, post-teardown. The two must be specified together so there is provably no path
   from an extras process to the verifier or gold data.
