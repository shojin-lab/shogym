# `hle` — Humanity's Last Exam, hgym's first model-graded env

An hgym port of [**Humanity's Last Exam**](https://agi.safe.ai/) (HLE) — 2,500 expert,
frontier-difficulty, closed-ended academic questions from the Center for AI Safety & Scale
AI. HLE has essentially no tool surface: it is single-turn Q&A. Its value here is (a)
coverage of a reasoning/QA task and (b) exercising hgym's **verification surface** with a
real **LLM judge** — a model-graded verifier that nothing else in hgym uses.

An hgym env **describes** a task, **serves** its tools over MCP, and **verifies** a recorded
trajectory — an external harness (Claude Code, Codex, …) drives the tools. Here the single
tool grades server-side, so the verifier stays a pure function over the trajectory.

## Running it

> Requires **Python 3.12 + the `hle` extra**, an OpenAI key for the judge, and Hugging Face
> access to the gated `cais/hle` dataset — see [Requirements](#requirements). The Claude Code
> example handles the wiring.

### Construct + serve

```python
import hgym

env = hgym.make("hle")                                  # train split; loads cais/hle (gated)
spec = env.describe("0")                                 # task 0: the question + tool manifest
```

Serve it as a stdio MCP server any harness can drive:

```bash
export OPENAI_API_KEY=sk-...                             # the judge is an OpenAI model
uv run python -m hgym.cli serve hle --task 0 --trace ./hgym_logs/hle.jsonl
```

The harness reads the question via `describe`, calls **`submit_answer`** (the env grades it
server-side), then **`terminate`**; hgym reads the verdict off the trace via
`hgym.result_from_trace(...)`.

**Config** (via `hgym.make(name, config)` / `env_config`): `task_split` (`"train"`/`"test"`),
`tasks` (an explicit task list — bypasses the dataset download, used by the offline tests),
`judge` (an injected [`Judge`](judge.py) — a scripted judge for offline runs), and
`judge_model` / `judge_base_url` (the default judge's model + endpoint).

### Claude Code example

The runnable end-to-end demo (Claude Code answers a served HLE question; hgym scores it with
the LLM judge) lives in [`examples/hle/claude_code/`](../../../../examples/hle/claude_code/):

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/hle/claude_code/run.py --task 0 --transcript
```

## Requirements

- **Python 3.12.** The project is pinned to 3.12 (`requires-python = ">=3.12,<3.13"`).
- **The `hle` extra.** `datasets` (loads `cais/hle`) + `openai` (the default judge client).
  `uv sync` installs it (it's in the dev group); `pip install 'hgym[hle]'` adds it to a plain
  install. `import hgym` registers `hle` **without** importing either — the core stays lean.
- **`OPENAI_API_KEY`.** The `submit_answer` handler grades with an OpenAI LLM judge. With the
  default judge, an episode **fails fast at startup** — a clear, actionable error raised before
  any tool runs — if no key is set, so a keyless run never silently scores everything wrong. Opt
  out by injecting a scripted `judge` (offline tests do this, need no key) or by pointing
  `judge_base_url` at a keyless OpenAI-compatible endpoint.
- **Hugging Face access to `cais/hle`.** The dataset is **gated** and must not be
  redistributed, so hgym never ships it. Accept its terms at
  <https://huggingface.co/datasets/cais/hle> and authenticate (`huggingface-cli login`, or set
  `HF_TOKEN`); it downloads once to `~/.cache/hgym/hle` (honor `HF_HOME` or `HGYM_HLE_DATA_DIR`
  to relocate). Offline tests inject their own tasks and need neither the download nor a token.

## How it works

### describe → TaskSpec

`env.describe(task_id)` publishes the task contract the harness reads: the **question** (in
`instructions`) and the **tool manifest** — `submit_answer` (provenance `env-mandatory`) and
the reserved `terminate`. There is no observation stream and no other tool.

### Tools (served over MCP)

- **`submit_answer(answer: str, confidence: int = 100)`** — backed by the in-process server in
  `mcp_server.py`. The handler grades server-side and returns a marked verdict dict
  (`hle_grade`, `correct`, `confidence`, `judged_by`, plus judge diagnostics):
  - an **exact-match fast path** first (normalize + compare, offline and free); on a match the
    LLM judge is not called;
  - otherwise the session's **LLM judge** (`judge.py`) grades it, using HLE's own judge prompt.
  The judge is **injectable** (`config={"judge": …}`): the registered env defaults to
  `OpenAIJudge`; offline tests inject a scripted judge, mirroring how the tau2 port injects a
  scripted user simulator. A judge exception scores the answer incorrect rather than crashing
  the episode.
- **`terminate()`** — the reserved episode-completion tool every env serves. Its call, detected
  by name, is the terminal signal.

The env pushes per-episode state — the question, its gold answer, and the judge — into the
in-process server via `begin_session(session_id, …)` on start and drops it via `end_session`
on teardown. State is keyed by session id, so one server safely backs many concurrent episodes.

### verify

`verify` is a **pure** function over the recorded trajectory. `score_trajectory` scans for the
terminal **`submit_answer`** step whose result carries the `hle_grade` marker and reads
`correct` from it; the **confidence** is read from the step **arguments** (the trustworthy
trajectory value), not the echoed result. Only a `submit_answer` step is trusted — a forged
marker on any other tool result grants no credit — and a missing / malformed / non-JSON grade
scores as incorrect rather than raising (the same untrusted-result discipline as wordle/tau2).

Feedback emitted on termination (episode-level):

- **`correct`** — did the judge (or the exact-match fast path) accept the answer.
- **`confidence`** — the submitted 0–100 confidence as a 0–1 fraction.
- **`calibration_error`** — `|confidence − correct|`, the per-episode Brier-style gap between
  the stated confidence and the outcome (0 is perfectly calibrated). A premature end with no
  graded submission emits only `correct = False` (there is no confidence to calibrate).

Termination happens when `terminate` is called **or** the horizon (2) is reached, whichever
comes first.

## Tasks

`hle` loads the text-only questions from `cais/hle` (its single `test` split of 2,500,
filtered to text-only for now — multimodal is a follow-up) and slices them **positionally
80/20** into `train` / `test`, like wordle. It defaults to `train`; pass
`config={"task_split": "test"}` for the held-out set. Task indices are relative to the chosen
split.

## Scoring

Every served tool call appends one row to the JSONL trace; the episode scores ride out on the
terminal result's `_meta` sidecar and the terminal trace row. Read the score back with:

```python
import hgym
result = hgym.result_from_trace("hgym_logs/hle.jsonl", env="hle", task="0")
print(result.terminated, result.value("correct"), result.value("calibration_error"))
```

`result_from_trace` treats `env`/`task`/`session_id` as **filters**, so a shared, append-only
trace can't let another run supply a stale result. For a guaranteed 1:1 mapping, give each run
its own trace file.

## Gotchas

- **The judge is model-graded and keyed.** `submit_answer` calls an OpenAI model unless the
  answer hits the exact-match fast path. With the default judge, starting an episode without
  `OPENAI_API_KEY` raises early (before any tool runs) rather than letting the run score
  everything incorrect — inject a scripted `judge`, or set `judge_base_url` for a keyless local
  endpoint, to opt out. A judge that fails *mid-run* (revoked key, rate-limit, network) still
  fail-closes to `correct = False`, but is flagged with `judge_error = True` in the episode
  feedback so an infra failure isn't counted as a genuine wrong answer.
- **The dataset is gated.** `cais/hle` needs accepted terms + HF auth; constructing the
  registered env downloads it (once). Inject `tasks` to construct the env without the download.
- **Text-only for now.** Questions carrying an image are filtered out; multimodal is a
  follow-up.
- **Look-ups defeat the point.** HLE measures the model's own reasoning — a harness must deny
  web tools (`--disallowedTools "WebFetch,WebSearch,…"`), as the example does.
- **Single-turn.** The horizon is 2 (`submit_answer` then `terminate`); the last graded
  submission before the cap is scored.
- **Offline vs keyed tests.** The pure verifier + judge-helper tests and the served
  scripted-judge tests run offline in the suite; a keyed fidelity test — the real `OpenAIJudge`
  grading a served episode — is skipped unless `OPENAI_API_KEY` is set.
