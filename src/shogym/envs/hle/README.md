# `hle` — Humanity's Last Exam, shogym's first model-graded env

A faithful shogym port of [**Humanity's Last Exam**](https://agi.safe.ai/) (HLE) — 2,500
expert, frontier-difficulty, closed-ended academic questions from the Center for AI Safety &
Scale AI. HLE has essentially no tool surface: it is single-turn Q&A. Its value here is (a)
coverage of a reasoning/QA task and (b) exercising shogym's **verification surface** with a real
**LLM judge** — a model-graded verifier that nothing else in shogym uses.

Like every shogym env this **describes** a task, **serves** its tools over MCP, and **verifies**
a recorded trajectory while an external harness drives the tools — see
[`../README.md`](../README.md). Here the single tool grades server-side, so the verifier stays
a pure function over the sealed episode's evidence. The runnable demo is
[`examples/`](../../../../examples/).

## Running it

> Requires **Python 3.12 + the `hle` extra**, an OpenAI key for the judge, and Hugging Face
> access to the gated `cais/hle` dataset — see [Requirements](#requirements). The Claude Code
> example handles the wiring.

### Construct + serve

```python
import shogym

env = shogym.make("hle")                                  # train split; loads cais/hle (gated)
spec = env.describe("0")                                 # task 0: the question + tool manifest
```

Serve it as a stdio MCP server any harness can drive:

```bash
export OPENAI_API_KEY=sk-...                             # the judge is an OpenAI model
uv run python -m shogym.cli serve hle --task 0 --trace ./shogym_logs/hle.jsonl
```

The harness reads the question via `describe` and calls **`submit_answer`** exactly once: that
call is the **score terminal** — it validates the args, atomically **seals** the episode,
grades it in `finalize`, and **ends** the episode in one step (no separate `terminate`; a
`terminate` or a repeat `submit_answer` afterward is tombstoned). See the shared
[terminal lifecycle](../README.md#terminal-lifecycle-seal-terminal-score-terminal-abort). shogym
reads the verdict off the trace via `shogym.result_from_trace(...)`.

**Config** (via `shogym.make(name, config)` / `env_config`): `task_split` (`"train"`/`"test"`),
`tasks` (an explicit task list — bypasses the dataset download, used by the offline tests),
`judge` (an injected [`Judge`](judge.py), a scripted judge for offline runs),
`judge_model` / `judge_base_url` (the default judge's model + endpoint), and `judge_kwargs`
(sampling fields for the default judge's chat-completions request, e.g.
`{"reasoning_effort": "low"}`, sent verbatim and omitted entirely when unset). `judge_kwargs`
takes sampling and nothing else (`reasoning_effort`, `temperature`, `top_p`, `seed`,
`frequency_penalty`, `presence_penalty`): the judge owns what it asks and the shape of the reply
it parses, so every other name is refused when the episode starts. It is an allowlist because
the failure it prevents is silent, and a name nobody thought to exclude, a legacy one or a new
SDK one, would otherwise arrive already permitted.

### Quickstart

Any quickstart under [`examples/`](../../../../examples/) serves this env: one MCP endpoint
hands out a queue of tasks and scores each one server-side. Point it here with the single
variable at the top of its `serve.py`:

```python
ENV = "hle"
```

The default judge is model-graded, so this needs `OPENAI_API_KEY` set in the environment
the server runs in.

## Requirements

The Python pin and the `uv sync` / `pip install` / `import shogym` mechanics are the shared
[requirements boilerplate](../README.md#requirements-boilerplate). The `hle` extra pulls
`datasets` (loads `cais/hle`) + `openai` (the default judge client). On top of that:

- **`OPENAI_API_KEY`.** Grading (in the env's `finalize`) uses an OpenAI LLM judge. With the
  default judge, an episode **fails fast at startup** — a clear, actionable error raised before
  any tool runs — if no key is set, so a keyless run never silently scores everything wrong. Opt
  out by injecting a scripted `judge` (offline tests do this, need no key) or by pointing
  `judge_base_url` at a keyless OpenAI-compatible endpoint.
- **Hugging Face access to `cais/hle`.** The dataset is **gated** and must not be
  redistributed, so shogym never ships it. Accept its terms at
  <https://huggingface.co/datasets/cais/hle> and authenticate (`huggingface-cli login`, or set
  `HF_TOKEN`); it downloads once to `~/.cache/shogym/hle` (honor `HF_HOME` or `SHOGYM_HLE_DATA_DIR`
  to relocate). Offline tests inject their own tasks and need neither the download nor a token.

## How it works

### describe → TaskSpec

`env.describe(task_id)` publishes the task contract the harness reads: the **question** (in
`instructions`) and the **tool manifest** — `submit_answer` (provenance `env-mandatory`,
**`terminal_kind: score`**) and the reserved `terminate` (`terminal_kind: abort`). There is no
observation stream and no other tool.

### Tools (served over MCP)

- **`submit_answer(answer: str, confidence: int = 100)`** — the **score terminal**. Calling it
  validates the args against the advertised schema, atomically **seals** the episode, grades it
  in `finalize`, and **ends** the episode — so a verdict only ever exists for an already-sealed,
  un-continuable episode (an agent cannot grade, read the verdict, then revise). Grading in the
  `finalize` hook is:
  - an **exact-match fast path** first (normalize + compare, offline and free); on a match the
    LLM judge is not called;
  - otherwise the session's **LLM judge** (`judge.py`) grades it, using HLE's own judge prompt.
  The judge is **injectable** (`config={"judge": …}`): the registered env defaults to
  `OpenAIJudge`; offline tests inject a scripted judge, mirroring how the tau2 port injects a
  scripted user simulator. A judge failure **fails closed** — the answer scores incorrect and
  the verdict is flagged `judge_error` — rather than crashing the episode. The **result
  returned to the agent is sanitized**: only the public-safe `correct` (bool) and `judge_error`
  (bool). The judge's reasoning / extracted answer and any exception text are answer oracles —
  they never reach the agent; they live only in the private, off-trace diagnostic.
- **`terminate()`** — the reserved abort tool every env serves (`terminal_kind: abort`). Since
  `submit_answer` already ends the episode, a harness does **not** call `terminate` after
  submitting (it would be tombstoned); `terminate` before submitting ends the episode with no
  submission (`correct = False`).

The env holds per-episode state — the question, its gold answer, and the judge — keyed by
session id, so one env instance safely backs many concurrent episodes; `finalize` reads it to
grade the sealed submission.

### finalize + verify

`submit_answer` is a `score` terminal, so on the sealed episode the env's `finalize` hook runs
the judge (or the exact-match fast path) and commits **core-owned terminal evidence** — not
marker JSON off the trajectory. The pure `verify` then scores from that evidence:
`score_evidence` reads the authoritative, seal-protected `correct` from the evidence verdict and
the **confidence** from the validated submit arguments. A fail-closed grade (any grading-infra
failure) is labelled `judge_error`; a terminal with no submission scores just `correct = False`.

Feedback emitted on termination (episode-level):

- **`correct`** — did the judge (or the exact-match fast path) accept the answer.
- **`confidence`** — the submitted 0–100 confidence as a 0–1 fraction.
- **`calibration_error`** — `|confidence − correct|`, the per-episode Brier-style gap between
  the stated confidence and the outcome (0 is perfectly calibrated). A terminal with no graded
  submission emits only `correct = False` (there is no confidence to calibrate).
- **`judge_error`** — set (`True`) only when the grade fail-closed on a grading-infra failure
  (a judge exception, or a serve-layer finalize deadline/crash), so an analyst can filter those
  out of the genuine zeros.
- **`judge_model`** (and **`judge_effort`**, when `judge_kwargs` set a `reasoning_effort`):
  which model graded, so a score read back off the trace says what produced it. It is the model
  the response reported, not the id that was requested, since an alias, a router, or a
  `judge_base_url` endpoint can answer as something else; the configured id stands in only when
  the judge failed before there was a response. Emitted only
  when the env built the judge itself: an exact-match episode was read by no model, and an
  injected `judge` is the caller's to describe, so both stay silent rather than guess.

Termination happens when the `submit_answer` score terminal seals the episode, when `terminate`
is called, **or** when the horizon (1) is reached — whichever comes first. Reaching the horizon
with no submission scores `correct = False` (the `zero_unsubmitted` policy).

## Tasks

`hle` loads the text-only questions from `cais/hle` (its single `test` split of 2,500,
filtered to text-only for now — multimodal is a follow-up) and slices them **positionally
80/20** into `train` / `test`, like wordle. It defaults to `train`; pass
`config={"task_split": "test"}` for the held-out set. Task indices are relative to the chosen
split.

## Scoring

The episode scores ride out on the terminal result's `_meta` sidecar and the terminal trace
row. Read the score back with:

```python
import shogym
result = shogym.result_from_trace("shogym_logs/hle.jsonl", env="hle", task="0")
print(result.terminated, result.value("correct"), result.value("calibration_error"))
```

`result_from_trace` treats `env` / `task` / `session_id` as **filters** — see
[Reading a score back](../README.md#reading-a-score-back-result_from_trace) for the shared
semantics (give each run its own trace file for a guaranteed 1:1 mapping).

## Fidelity & deviations

- **Grading is HLE's own.** The judge uses HLE's own judge prompt; the registered env defaults
  to `OpenAIJudge` (overridable via `judge_model` / `judge_kwargs` / `judge_base_url`, or a
  fully injected `judge`). The exact-match fast path is a free, offline pre-check that never
  changes a correct verdict.
- **The default judge model is a scoring decision.** It is `gpt-5.6-luna`, at that model's own
  default reasoning effort, chosen on grading quality measured against the previous default
  (issue #122). Changing it changes measured accuracy, which is why every model-graded episode
  now records the model that graded it.
- **Text-only for now.** Questions carrying an image are filtered out; multimodal is a
  follow-up.
- **Judge fail-closed.** A grading-infra failure scores `correct = False` with `judge_error =
  True` rather than crashing — so an infra failure is distinguishable from a genuine wrong
  answer, not silently counted as one.

## Gotchas

- **The judge is model-graded and keyed.** `submit_answer` calls an OpenAI model unless the
  answer hits the exact-match fast path. With the default judge, starting an episode without
  `OPENAI_API_KEY` raises early (before any tool runs) rather than letting the run score
  everything incorrect — inject a scripted `judge`, or set `judge_base_url` for a keyless local
  endpoint, to opt out.
- **The dataset is gated.** `cais/hle` needs accepted terms + HF auth; constructing the
  registered env downloads it (once). Inject `tasks` to construct the env without the download.
- **Look-ups defeat the point.** HLE measures the model's own reasoning — a harness must deny
  web tools (`--disallowedTools "WebFetch,WebSearch,…"`), as the example does.
- **Single-turn, single terminal action.** `submit_answer` is the score terminal: submitting
  seals + grades + ends the episode in one step (horizon 1), so there is no second submission
  and no separate `terminate` to call afterward.
- **Offline vs keyed tests.** Follows the shared
  [offline-vs-keyed split](../README.md#tests-offline-vs-keyed): the pure verifier + judge-helper
  tests and the served scripted-judge tests run offline; the keyed fidelity test (the real
  `OpenAIJudge` grading a served episode) is skipped unless `OPENAI_API_KEY` is set.

## Layout

A source map for orientation:

| File | Role |
|---|---|
| `env_v1.py` | The registered `hle` env: `describe` (question + manifest), the dataset load + 80/20 split, the `finalize` hook (exact-match fast path → LLM judge on the sealed submission), and the pure `score_evidence` verifier. |
| `mcp_server.py` | The in-process MCP server backing `submit_answer` (the score terminal, sealed before any verdict) + the reserved `terminate`. |
| `judge.py` | The `Judge` seam: `OpenAIJudge` (default) + the scripted judge used by offline tests, plus HLE's own judge prompt. |
| `data.py` | Loads the gated `cais/hle` dataset, filters to text-only, and caches under `~/.cache/shogym/hle`. |
</content>
