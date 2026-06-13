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

## 3. The artifact

```
harness/
└── instruction/
    ├── system.minijinja        # required if the function has a system template
    ├── user.minijinja          # iff user_schema is declared
    ├── assistant.minijinja     # iff assistant_schema is declared
    └── skills/                 # optional; see §5
        ├── failure_analysis.md
        └── ...
```

- One template file per declared role. `load_harness` validates that every template
  references only variables in its declared schema (minijinja can be rendered against the
  schema's field set in a dry run; an undeclared `{{ foo }}` is a load-time error, never a
  mid-episode one — the Harness-Bench anti-pattern we are avoiding).
- The model line lives in `harness.toml` (`model = "..."`), not here: the model is the
  substrate (§6), templates are the instruction.

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

## 5. Skills (open question, leaning yes)

"Skills" (Claude Code's SKILL.md pattern) are instruction content the model loads on demand
rather than carrying in every prompt: progressive disclosure. They matter disproportionately
for the self-improvement program (P3), where the optimizer mines its own traces into skills.

Proposal: `instruction/skills/*.md`, each a named, on-demand instruction module, surfaced to
the model through whatever mechanism the agent supports (a `load_skill` tool, or
context-strategy injection). Whether skills are "just more instruction templates" or deserve
their own structure (frontmatter: when-to-use, token budget) is the open question. Leaning
toward light structure (frontmatter + body) because the self-improvement loop needs to
reason about *which* skill helped (per-skill attribution), which wants each skill to be its
own diffable file with metadata. Deferred to a focused follow-up once P1 lands.

## 6. The model is the substrate, not the instruction

`model = "openai/gpt-5.4-nano"` in `harness.toml`. Swapping the model is optimizable and
belongs to the harness, but it is *not* the instruction surface and not any surface (RFC 000
§3 corollary 7). Report two axes separately: surfaces-at-fixed-model, and
model-swaps-at-fixed-surfaces. Conflating them is how you get "the new prompt is better"
when actually the model changed underneath.

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
