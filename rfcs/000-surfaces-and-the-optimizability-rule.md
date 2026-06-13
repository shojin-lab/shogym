# RFC 000: Surfaces, loci, and the optimizability rule

- **Status:** Draft (proposed 2026-06-13)
- **Supersedes/revises:** parts of the wiki `Roadmap` and `The Surfaces of an Agent Harness`
- **Depends on:** the ported v1 core (`ToolUsingEnv`, `runner.py`, `mcp/`, the harness
  export format, which is still unbuilt)
- **Companion RFCs:** 001 Instruction, 002 Tool, 003 Context, 004 Control,
  005 Environment, 006 Verification, 007 Observability

> This RFC is the spine. The seven surface RFCs each assume the vocabulary and the
> optimizability rule defined here. Read this first. It also revises three theses from
> the current wiki design docs; those revisions are called out explicitly in §7.

---

## 1. Summary

The wiki taxonomy names seven harness **surfaces** (instruction, tool, context, control,
environment, verification, observability) and asserts each should become "optimizable."
That framing is necessary but not sufficient: it says *what* the parts are, not *where
they live* or *who is allowed to change them*. This RFC supplies the missing half.

The core move: classify everything by its **locus** — is it an attribute of the
**environment** (the task), the **agent** (the policy), or the **rollout** (the loop that
connects them)? Grounding that classification in the actual code yields a single clean
rule for what may be optimized, resolves several ambiguities the seven-surface picture
left open, and dictates a config-directory design that stays readable as all surfaces
open up.

**The optimizability rule (one sentence):** a surface is optimizable exactly when it is an
attribute of the **rollout or the agent**; it is fixed when it is an attribute of the
**environment** (which defines the problem) or the **verifier** (which measures it).

The **harness** is then defined precisely, not vibily: *the harness is the editable
projection of the rollout-and-agent configuration, holding the environment and verifier
fixed.* Optimizing a harness means editing that projection. This is what `export_harness`
should export and `load_harness` should load.

---

## 2. The three loci, grounded in the code

The v1 core already has three distinct objects. Naming them precisely is the whole game.

### 2.1 Environment (`ToolUsingEnv`) — the task

`src/hgym/envs/tool_using_env.py`. An environment owns:

- the **task** (`_load_task`, `_initial_observations`),
- the **mandatory tool surface** (`mcp_servers` class field: the servers the verifier
  needs evidence from — `guess`, `submit_answer`, the simulated counterparty),
- the **verifier** (`_verify`, a pure function over the recorded trajectory),
- the **horizon** (a terminal constraint),
- the **function shape** (`function: FunctionConfigChat` — the schemas and which
  templates exist, i.e. the *interface*, not the prompt *content*),
- and, today, the **trajectory** itself (`self._trajectory`), which it accumulates and
  emits whole in every `Observation.messages` (`tool_using_env.py:429`). §7 argues this
  last responsibility is mis-placed.

The environment defines the problem. Changing any of the above changes *what is being
asked*, so none of it is optimizable: an optimizer that may edit the task or the verifier
is not solving the task, it is redefining it (Goodhart in the limit). The environment is
the fixed point against which everything else is measured.

### 2.2 Agent (`OpenAIAgent`/`LLMAgent`) — the policy

`src/hgym/agents/openai/agent.py`. Deliberately thin: it holds the **model**
(`model_name`), reads an `Observation`, translates it to a provider call
(`parse_observation`), and returns an `Action`. It reads `obs.tools` dynamically and
forces `parallel_tool_calls=False`.

The agent is the **model plus the translation to the wire**. It is swappable (the runner
takes an `agent_cls` or an `agent_builder`). The model it wraps is the one thing the seven
surfaces do *not* cover (Agent = Model + Harness): the model is the **optimizable
substrate**, not a surface. See §6.

### 2.3 Rollout (`runner.py`) — the loop

`src/hgym/runner.py:62`. The entire loop is four lines:

```python
action = await agent.act(obs)
step_data = await env.step(action)
...
obs = step_data.observation
```

There are **no hooks, no context transform, no retry policy, no routing**. The rollout
today owns only: task selection, concurrency (`max_concurrent`, semaphores), the
`extra_toolset` it threads into `make()`, and the `agent_builder`. It is an almost-empty
seam.

This emptiness is the opportunity. Everything that is "how the loop runs, as opposed to
what the task is or what the model is" belongs here, and almost none of it exists yet.
Context transforms, control hooks, retry/repair, run-level budgets and permissions: all of
it attaches at the rollout, which means it can be added *without touching the environments*
— the precondition for clean attribution.

### 2.4 The picture

```
        ENVIRONMENT                 ROLLOUT                 AGENT
        (the task)                  (the loop)              (the policy)
   ┌──────────────────┐      ┌────────────────────┐    ┌──────────────┐
   │ task loader      │      │ loop / stopping    │    │ model        │
   │ mandatory tools  │ ───▶ │ context transform  │──▶ │ obs → wire   │
   │ verifier (pure)  │ ◀─── │ control hooks      │ ◀──│ action       │
   │ horizon          │      │ retry / repair     │    └──────────────┘
   │ function shape   │      │ extras tools (∪)   │
   │ [trajectory]*    │      │ run policy (budget)│
   └──────────────────┘      └────────────────────┘
        FIXED                  OPTIMIZABLE (the harness)   SUBSTRATE: model
   (* trajectory ownership moves out of the env per §7.1)
```

---

## 3. The optimizability rule and its corollaries

**Rule.** Optimizable ⟺ attribute of the rollout or the agent. Fixed ⟺ attribute of the
environment or the verifier.

Corollaries that fall straight out, several of which the seven-surface framing left
ambiguous:

1. **The tool surface is split, not unitary.** Mandatory tools are an environment
   attribute (fixed); extras are a rollout attribute (optimizable). The *surface the model
   sees* is their union, composed at rollout time. So "optimize the tool surface" means
   precisely "edit the extras," never "remove a mandatory tool." (RFC 002.)
2. **Instruction content is agent/rollout; instruction shape is environment.** The env
   declares which templates exist and their schemas (the interface); the *content* of
   those templates is the optimizable artifact. The optimizer rewrites `system.minijinja`;
   it does not invent new schema slots the env never declared. (RFC 001.)
3. **Context is a rollout transform, not an environment output.** (Thesis revision, §7.1.)
4. **Control is a rollout attribute.** The env contributes only the horizon (a constraint)
   and the `terminate` convention; the loop's shape, hooks, and retries are the rollout's.
   (RFC 004.)
5. **Verification is fixed and is the measuring stick.** The optimizer may never edit
   `_verify`, nor any judge/rubric the env uses. "Self-verification" the agent calls
   mid-trajectory (`run_tests`) is a *tool-surface* move, not a verification-surface move;
   the two must not be conflated. (RFC 006.)
6. **Observability is the substrate, not a target.** It is always on and never optimized;
   its job is to make the other six *attributable* (per-surface hashes in the trace).
   (RFC 007.)
7. **The model is the substrate the surfaces wrap.** Swapping it is allowed and lives in
   the harness (`model = ...`), but it is not one of the seven surfaces; report
   model-holding-surfaces-fixed and surfaces-holding-model-fixed as distinct axes.

---

## 4. Where the seven surfaces actually live

| Surface | Primary locus | Optimizable? | Editable artifact (harness) | RFC |
|---|---|---|---|---|
| Instruction | agent/rollout (content); env (shape) | yes (content) | `instruction/*.minijinja` | 001 |
| Tool | env (mandatory) + rollout (extras) | yes (extras only) | `tools.toml` | 002 |
| Context | **rollout** (transform over trajectory) | yes | `context.toml` | 003 |
| Control | rollout (env contributes horizon + `terminate`) | yes | `control.toml` | 004 |
| Environment | rollout (run policy) + tool exec substrate | yes (policy) | `environment.toml` | 005 |
| Verification | **env / verifier** | **no (by design)** | — (env-owned) | 006 |
| Observability | substrate | no (always on) | — (emitted, not configured) | 007 |
| *(Model)* | agent | yes (substrate, not a surface) | `harness.toml: model` | 001 §6 |

The two "no" rows are not omissions; they are load-bearing. A measurement instrument the
optimizer can edit measures nothing; a substrate that is itself optimized stops being a
neutral record. RFCs 006 and 007 defend these as positive design choices.

---

## 5. The harness directory: one file per surface, absent by default

The recurring design pressure is *readability for the optimizing agent and the human*. The
harness directory is what a coding agent reads, edits, and diffs. Three principles keep it
manageable as all surfaces open:

### P1. One file per surface, so diffs self-classify

Each optimizable surface gets exactly one artifact. A diff that touches only `context.toml`
is, by construction, a context-surface edit. This makes **attribution a file-level fact**:
the per-surface hash (RFC 007) is just the hash of that surface's file(s). No parsing of
"which lines changed which concern." Compare Harness-Bench, which entangled grader, runtime,
and config (see `lit-reviews/harness-bench-code-review.md` §4.2): hgym's separation is the
antidote.

### P2. Surfaces are absent by default; engaging one means creating its file

`export_harness` writes only `harness.toml` (model + manifest) and `instruction/` for a
baseline env. There is no `context.toml`, `control.toml`, or `environment.toml` until the
optimizer decides to engage that surface. **The size of the directory equals the number of
surfaces the optimizer has touched.** A baseline harness is two or three items; a harness
that has only tuned prompts and added one tool stays tiny. This is the single most
important readability lever: the optimizer (and the reader) never wade through inert config
for surfaces nobody touched. A missing file means "this surface is at its default," which is
also the cleanest possible semantics for attribution (no edit, no contribution).

### P3. The manifest is the index, the files are the content

`harness.toml` carries the model, run limits, and a short manifest listing which surface
files are active. Reading `harness.toml` tells you the whole shape of the harness in one
screen; the per-surface files hold the detail. This mirrors the "Home page + linked pages"
structure that already works for the wiki.

### Proposed layout (baseline → fully engaged)

```
# Baseline (what export_harness writes for a fresh env):
harness/
├── harness.toml            # model, params, limits, manifest
└── instruction/
    └── system.minijinja    # (+ user/assistant templates if the function declares them)

# After an optimizer has engaged several surfaces:
harness/
├── harness.toml            # model + manifest now lists the active surface files
├── instruction/
│   ├── system.minijinja
│   └── skills/             # optional skill files (RFC 001)
├── tools.toml              # extras MCP servers (RFC 002); replaces hgym_extras.toml
├── context.toml            # context strategy (RFC 003)
├── control.toml            # hooks, retries, gating (RFC 004)
└── environment.toml        # budgets, permissions, sandbox policy (RFC 005)
```

`harness.toml` sketch:

```toml
model = "openai/gpt-5.4-nano"        # the substrate (RFC 001 §6)
[limits]
horizon = 12                          # may only tighten the env's horizon, never loosen it
[manifest]                            # which surfaces are engaged; absent file = default
instruction = "instruction/"
tools       = "tools.toml"
context     = "context.toml"
# control / environment omitted here => at their defaults
```

Notes:

- `tools.toml` **replaces** the current `hgym_extras.toml` name (`mcp/config.py`). One
  consistent per-surface naming scheme beats a special-cased filename.
- The manifest is redundant with file presence (P2 already encodes "engaged = file
  exists"); it exists so a reader/optimizer gets the shape from one file, and so
  `load_harness` can fail loudly on a referenced-but-missing file. If the redundancy
  proves annoying, the manifest can be dropped in favour of pure file-presence; that is an
  open question (§9).

---

## 6. Attribution mechanics (what RFC 007 must provide)

The whole program is *which surface caused the delta*. With P1 (one file per surface),
attribution is mechanical:

- Each trace records a **harness hash** plus a **per-surface sub-hash** (the hash of that
  surface's file(s), or a sentinel for "default/absent").
- A controlled experiment varies one surface file between arms, holding the others (and the
  env, verifier, and model) fixed.
- The per-surface hash in the trace identifies which arm produced each episode; the
  ablation is a `group by surface_hash` over the JSONL, not bookkeeping.

This is why P1 is not just tidiness: it makes the central scientific claim a query.

---

## 7. Thesis revisions

The current wiki docs are mostly right. Three claims need revising; one is significant.

### 7.1 (Significant) Context management is mis-located in the environment

The wiki `Roadmap` lists context as "RFC 002 territory, a truncation knob in
`harness.toml`." But the code shows the env *owns the trajectory* and emits it whole as the
model's view (`tool_using_env.py:429`, `messages=list(self._trajectory)`); the agent sends
all of it (`agent.py:47`). So today the env, not the harness, decides what the model sees,
and the "view" is hard-wired to "everything."

**Revision:** separate the **trajectory** (environment-recorded, immutable, what `_verify`
and the logs see — ground truth) from the **view** (a rollout-level transform over the
trajectory, harness-configured, what the agent sees). Context management is the view
function; today it is the identity. Making it optimizable means inserting a transform at
the rollout, between `env.step`'s observation and `agent.act`. The env keeps producing the
full trajectory; the rollout projects it. This keeps verification and logging on ground
truth while the *model's window* becomes an optimizable surface. RFC 003 specifies it.

Consequence: `Observation.messages` should be understood as "the trajectory so far," and a
new rollout step computes `view = context_strategy(trajectory)` before the agent sees it.
Either the runner applies it (preferred: agent stays thin) or a thin agent-wrapper does.

### 7.2 (Clarifying) Multi-agent topology collapses into the tool surface

The wiki lists "multi-agent topology (planner/generator/evaluator)" under the control
surface. RFC 001's own observation is cleaner: a subagent is a tool whose body runs an
inner episode (an inner harness). So topology is not a separate control concern; it is the
**tool surface applied recursively**. This shrinks the control surface to "loop policy +
hooks" (RFC 004) and means a subagent's own harness is a nested harness directory (RFC 002
§ subagents). One ontology, used recursively, instead of two.

### 7.3 (Clarifying) "Environment surface" is two things wearing one name, and neither is the env

The wiki's fifth surface, "Environment" (sandbox, credentials, budgets, permissions),
collides nominally with the `Env` object and bundles two different things: (a) the **tool
execution substrate** (transport/sandbox per MCP server — already partly in
`MCPServerSpec.{transport,env,headers}`), and (b) **run-level policy** (budgets, egress,
permission posture). (b) is a clean rollout attribute and is the optimizable part; (a)
rides along with each tool (RFC 002) because an MCP server delivers interface and execution
together. RFC 005 proposes renaming this surface **"execution"** to end the collision with
`Env`, and scoping the optimizable artifact to run-level policy.

---

## 8. Alternatives considered

- **One monolithic `harness.toml` with `[instruction]`, `[context]`, … sections.** Rejected:
  violates P1 (a diff can touch two surfaces in one file, so attribution needs line-level
  parsing) and grows into a wall of inert config as surfaces open (violates P2). The
  per-file split is what makes attribution a file-hash.
- **A `surfaces/` subdirectory.** Cosmetic; adds a level of nesting for no attribution or
  readability gain over flat per-surface files at the harness root.
- **Optimize the environment/verifier too (drop the fixed/optimizable distinction).** This
  is end-to-end harness search (Meta-Harness, arXiv:2603.28052). It is a valid and powerful
  mode, and hgym can support it as the special case "unlock all surfaces at once." But as the
  *default*, it destroys attribution (you cannot say which change caused the gain) and invites
  verifier-gaming. hgym's contribution is the controlled, one-surface-at-a-time regime;
  end-to-end is recoverable on top, not the foundation.
- **Make the model just another surface.** Rejected for clarity: the model is what the
  surfaces wrap (Agent = Model + Harness). Folding it in muddies the central metaphor and
  the attribution story (model swaps and surface edits have very different cost/▵ profiles
  and should be reported on separate axes). It still lives in `harness.toml` for convenience.

---

## 9. Open questions / where this might be wrong

1. **Manifest vs pure file-presence (§5 P3).** Is the `[manifest]` worth its redundancy, or
   should "engaged = file exists" be the sole signal? Leaning toward a minimal manifest for
   the one-screen overview and loud load-time errors, but it is a real call.
2. **Where the context transform attaches (§7.1).** Runner-applied (agent stays thin, but
   the runner grows a responsibility) vs agent-wrapper (composes, but every agent must opt
   in). RFC 003 picks one; the choice affects how `agent_builder` users get context for free.
3. **Is "execution" (§7.3) really separable from "tool"?** Because an MCP server delivers
   interface + execution together, the run-level policy (budgets/permissions) is the only
   cleanly separable part. If that part turns out thin, the execution surface may fold into
   tool + a couple of `harness.toml` limits, and the seven become six. RFC 005 tests this.
4. **Does the instruction surface need first-class "skills"?** (RFC 001.) Skills are
   instruction content the model loads on demand; they may deserve structure beyond flat
   templates, especially for the self-improvement program (P3), where the optimizer mines
   traces into skills. Possibly an instruction sub-artifact, possibly its own thing.
5. **The fixed/optimizable line for judges.** If an env verifies with an LLM judge, the
   judge config is env-owned and closed (RFC 006). But research *on judge robustness* wants
   to vary it. That is a separate experiment mode, not a harness surface; RFC 006 must make
   the mode boundary unambiguous so it never leaks into the optimization loop.
6. **Could the whole seven-surface taxonomy be wrong-grained?** The locus analysis suggests
   the deepest cut is actually three (env / rollout / agent), and "surfaces" are a
   finer-grained, user-facing slicing of the rollout-and-agent locus. That is a feature (the
   surfaces are the editable artifacts; the loci are the architecture), but if a surface
   ever fails to map to a single locus cleanly, prefer the locus. Tool and Execution already
   strain this (both span env and rollout); watch for more.
