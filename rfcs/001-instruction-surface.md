# RFC 001: The instruction surface

- **Status:** Draft (proposed 2026-06-13)
- **Depends on:** RFC 000
- **Locus:** content = agent/rollout (optimizable); shape = environment (fixed)
- **State today:** partially built. `FunctionConfigChat` carries
  `example_*_template` and schema pointers; templates render via minijinja. No
  `export_harness`/`load_harness` yet.

---

## 1. What it is

The instruction surface is **what the model is told**: the system/user/assistant
templates, any skill files, and the descriptions of the tools it is offered. It is the
most-optimized surface in the wild (DSPy/GEPA optimize exactly this) and the first hgym
should make first-class.

## 2. Locus: shape is the env's, content is the harness's

`FunctionConfigChat` (`types/config.py:195`) already separates the two halves, and the
split is the key to keeping this surface clean:

- **Shape (fixed, env-owned):** `system_schema`, `user_schema`, `assistant_schema` —
  pointers to pydantic models that define *what variables a template may reference*. The
  env declares these because they are part of the task interface (the env feeds the
  template its arguments). The optimizer may not add a schema slot the env never declared;
  doing so would be asking the env for data it does not produce.
- **Content (optimizable, harness-owned):** the rendered template text itself. The
  optimizer rewrites `system.minijinja` freely, subject only to "reference only declared
  variables." This is the lever.

This is the general pattern from RFC 000 §3 corollary 2: the env declares an interface, the
harness fills it. It is also the safety property: a content-only edit can never make the env
produce different observations, only change how the model reads the same observations.

## 3. The artifact: just the system template (revised per review)

Since every function is tool-calling (JSON-output `FunctionConfigJson` is being removed,
issue #10), **there are no user or assistant templates in the optimizable surface.** The
reasoning:

- **Assistant content is model output**, generated at inference (tool calls), never
  template-rendered. There is nothing to template.
- **User content is the task instance** — the initial task presentation and the tool results
  — built by the env's `_initial_observations` and the tool loop. That is env-owned (changing
  how the task is presented changes the task), not the optimizer's to edit.
- **System content is the policy** — "you are an agent that…, here is how to approach this."
  This is the optimizable instruction.

So the instruction surface collapses to one file:

```
harness/
└── instruction/
    ├── system.minijinja        # the policy: the only optimizable template
    └── skills/                 # optional instruction modules; see §5 (representation pending)
```

- `load_harness` validates that `system.minijinja` references only variables in the function's
  `system_schema` (dry-render against the schema's field set; an undeclared `{{ foo }}` is a
  load-time error, never a mid-episode one — the Harness-Bench anti-pattern we avoid).
- If the env declares no `system_schema` (a static system prompt), `system.minijinja` is plain
  text. Most envs will be here.
- The inference config (model + params) lives in `harness.toml [inference]` (§6), not here.

Consequence for the code: `FunctionConfigChat`'s `user_schema`/`assistant_schema` and the
`example_user_template`/`example_assistant_template` fields become vestigial for v1 envs and
can be dropped alongside issue #10's `FunctionConfigJson` removal. Worth a cleanup ticket.

## 4. Tool descriptions are instruction content, but stay with their tool

A tool's `description` (`ToolConfig.description`) is prompt content — the model reads it to
decide when to call the tool — so it is instruction-surface in nature. But it is delivered
with the tool, which creates an ownership question. The clean rule:

- **Mandatory tools' descriptions are env-owned and fixed.** They are part of the task
  interface the verifier expects; the optimizer does not edit them.
- **Extras' descriptions are optimizer-owned.** When the optimizer authors an extra MCP
  server (RFC 002), it writes that tool's description as part of authoring it.
- If the optimizer wants the model to understand a *mandatory* tool better, it says so in
  the **system prompt** (instruction content it owns), not by editing the env's tool. This
  keeps the env interface immutable while still letting the optimizer shape understanding.

So the instruction surface's editable text is: the templates + skills + extras-tool
descriptions. Mandatory-tool descriptions are read-only env interface.

## 5. Skills: adopt the SKILL.md open standard (resolved by research, 2026-06-13)

The open question was whether skills are Claude-Code-specific and whether "an injectable
optional fragment of the system prompt" faithfully represents them. Research settled both:

- **Skills are no longer Claude-specific.** `SKILL.md` is a cross-vendor **open standard**
  (agentskills.io), adopted by Claude Code, OpenAI Codex, Google Gemini CLI / Antigravity,
  Cursor, Copilot, and ~30 others. Every one converges on the same shape: a folder named for
  the skill, a `SKILL.md` with `name` + `description` frontmatter and a Markdown body, optional
  `references/`/`assets/`/`scripts/`, loaded by **progressive disclosure** (the name+description
  always in context for routing; the body loaded on match; resources/scripts on demand).
- **"Injectable system-prompt fragment" is faithful only for the trivial case** (a small
  reference skill with no resources). It loses the three things that make skills more than
  snippets, all of which are in the standard, not Claude-only: (1) progressive disclosure (only
  ~name+description in context until matched, budgeted to ~1-2% of the window), (2) linked
  files/executable scripts (a script runs and only its *output* enters context), (3) per-skill
  `allowed-tools` (experimental in the standard).

**Decision: adopt the SKILL.md standard verbatim as hgym's on-disk skills format.** It is
simultaneously the most faithful to Claude Code and the most portable (it is literally the
shared format), so it is the agent-agnostic answer. The instruction surface becomes two tiers:

```
harness/
└── instruction/
    ├── system.minijinja            # the system template (the policy)
    └── skills/                     # the Agent Skills standard, verbatim
        └── failure-analysis/
            ├── SKILL.md            # name + description frontmatter (always loaded) + body (on demand)
            ├── references/         # optional, lazy
            └── scripts/            # optional, executed; output-only into context
```

Notes:

- This is *not* a return of the harness manifest (RFC 000 §5 P3): the per-skill `SKILL.md`
  frontmatter is the skill's *own* descriptor (the always-loaded routing layer the standard
  requires), not a harness-level index of surfaces. File-presence still governs which surfaces
  are engaged; SKILL.md frontmatter governs which skill the model loads when. No conflict.
- **Graceful degradation:** for an agent (or a v1 reference agent) that cannot do on-demand
  loading, the fallback is deterministic: concatenate each skill's `SKILL.md` body (description
  order) into the system prompt. A skill with no scripts/resources degrades losslessly; the
  only loss is context-economy and script-execution-without-context. So "injectable fragment"
  becomes the *fallback path*, not the primary representation — which is exactly the right
  relationship.
- **Keep only the portable frontmatter subset** in the canonical descriptor (`name`,
  `description`, optional `license`/`compatibility`/`metadata`); treat `allowed-tools` as an
  experimental hint; tuck any Claude-only knobs (`context: fork`, `model`, `hooks`) under
  `metadata` so non-Claude targets ignore them.
- **Why this matters for P3 (self-improvement):** the optimizer mines its own traces into
  skills. Each skill being its own diffable folder with a description gives per-skill
  attribution (RFC 007) for free, and the standard's progressive disclosure is exactly the
  "don't bloat the prompt with every learned lesson" property P3's bloat-guard needs (cf. ACE,
  Cursor's rolled-back auto-memories in `lit-reviews/self-optimization.md`).

Open sub-question: does a skill's *content* attribute to the instruction surface as one hash,
or does each skill get its own sub-hash (so "which skill helped" is a query)? Leaning per-skill
sub-hash, given P3. RFC 007 decides.

## 6. The inference surface (model + inference params) (revised per review)

The model is its own optimizable surface, not a non-surface "substrate." The **inference
surface** is the model id *plus the inference-API request config that travels with it*:
everything a gateway's completion call takes besides the messages and the tools.

```toml
# harness.toml — the inference surface gets a section here, NOT its own file
[inference]
model = "openai/gpt-5.4-nano"
temperature = 0.7
top_p = 1.0
max_tokens = 2048
# reasoning_effort = "medium"   # reasoning models
# stop = ["</done>"]
```

Rationale for treating these together: they are exactly the non-messages, non-tools fields of
the `ModelClient.CompletionRequest` (the M1 inference seam). A gateway abstracts precisely
this set, so the inference surface == "what you'd send a gateway besides messages and tools."
Grouping them is more honest than carving out the model id alone.

It lives in `harness.toml` (no own file: a handful of scalars does not warrant one — RFC 000
§5). It is optimizable: the optimizer may tune `temperature`, raise `reasoning_effort`, or
swap `model`. **Still report model swaps on a distinct axis from in-model param/prompt edits**
(RFC 007): a model swap is a categorically larger lever, and conflating it with prompt tuning
gives "the new prompt is better" when the model changed underneath. One caveat: `seed`, if
present, is pinned by the experiment runner for reproducibility, not optimized — it is a
rollout/observability concern, not an inference-surface knob.

## 7. Prior art

- **DSPy optimizers (MIPROv2, SIMBA, GEPA).** The state of the art in instruction
  optimization; they search instruction text against a metric. In hgym they become *pluggable
  single-surface optimizers*: point GEPA at `instruction/` and hold every other surface fixed,
  and you have a clean instruction-surface ablation. hgym does not reimplement them; it gives
  them a typed surface to act on and an attribution story they lack.
- **Anthropic, "Writing effective tools for agents"** (tool descriptions as contracts).
- **GEPA (arXiv:2507.19457):** reflective prompt evolution from traces — the natural P1
  baseline for "prompt-only" optimization.

## 8. Alternatives considered

- **Templates inline in `harness.toml` as strings.** Rejected: minijinja templates are
  multi-line and want their own files for readability and clean diffs (per-file attribution,
  RFC 000 §5 P1). TOML multi-line strings are a readability tax on exactly the surface that
  is edited most.
- **Let the optimizer add schema slots.** Rejected: that is editing the env interface, not
  the instruction. If a template needs data the env does not produce, that is a task-design
  change, not an optimization.

## 9. Risks / where this might be wrong

- **The shape/content line can blur.** A template that conditionally references a variable
  only present in some tasks looks like a content edit but depends on env behavior. Mitigation:
  the schema is the contract; validate against it, and if a template needs richer conditioning,
  that pressure is a signal the env's schema should expose the condition (an env change,
  reviewed, not an optimizer move).
- **Skills may want to be their own surface.** If skills accrue retrieval, versioning, and
  per-skill metrics, "instruction" may be too coarse and skills graduate to their own RFC. Flag
  for revisit after P3.
