# `appworld` — AppWorld, with one paragraph appended that nobody answers for you

A faithful shogym port of [AppWorld](https://github.com/StonyBrookNLP/appworld): carry out a
natural-language instruction across nine simulated apps by writing Python against their APIs. On
top of the rented task, one authored paragraph asks for a filing log in a corner of the world no
scenario touches, whose values a house convention computes from the world's own data — and the
paragraph names none of the four choices that convention makes.

Env-as-center: describe, serve, verify. See [`../README.md`](../README.md) for the shared
machinery. A runnable end-to-end demo lives under
[`examples/claude_code/`](../../../../examples/claude_code/).

## Running it

### Construct + serve

```python
import shogym

env = shogym.make("appworld")           # config: pulse, report, horizon
print(env.num_tasks)                    # 318
print(env.describe("0").instructions)   # the world's guide, the task, then the paragraph
```

```python
from shogym.serve import ServedEpisode

episode = await ServedEpisode.open_env(env, env_name="appworld", task=0)
await episode.call("execute", {"code": 'print(apis.supervisor.show_profile())'})
await episode.call("submit", {})        # seals, then scores
await episode.close()
```

### Claude Code example

```bash
SHOGYM_ENV=appworld SHOGYM_TASKS=0 claude -p "$(cat examples/claude_code/PROMPT.txt)" \
    --mcp-config examples/claude_code/.mcp.json --strict-mcp-config \
    --allowedTools 'mcp__shogym__*' --permission-mode dontAsk
```

The paired feedback policies are what this env was built for. `Information()` answers a
terminating call with the receipt; `Placebo()` answers it with a digest of the agent's own
submission, the same size on the wire; `Never()` answers with no channel at all.

```python
from shogym.serve.stream import Information, Placebo, TaskRef, TaskStream

TaskStream(shogym.make, [TaskRef("appworld", 0)], prov_dir=..., feedback=Information())
```

## Requirements

Pin-and-install mechanics are shared; see [`../README.md`](../README.md). What is specific:

- **The `appworld` extra carries no packages, and cannot.** `appworld` 0.1.3.post1 pins
  `pydantic>=1.9,<2`; shogym's MCP layer needs `pydantic>=2.7`. No environment satisfies both. The
  port therefore builds an interpreter of its own — a virtual environment under
  `~/.cache/shogym/appworld/runtime-<version>/` holding the pinned release — and runs every world
  in a subprocess under it. `SHOGYM_CACHE` relocates it; `uv` is used when it is on `PATH` and
  `venv` + `pip` otherwise. This is provisioned when an `appworld` env is **constructed**, so
  `import shogym` stays offline.
- **The data bundle is fetched and checked.** ~33 MB, once, into
  `~/.cache/shogym/appworld/corpus-0.1.0/`. Upstream's own downloader verifies nothing; this one
  refuses a bundle whose size and sha256 are not the pinned pair. `APPWORLD_ROOT` pointing at a
  directory whose `data/tasks` exists is used as it stands, so a machine that already ran
  `appworld download data` needs no download.
- **No API key, no Docker, no network once provisioned.**
- The port's pure pieces — the backlog generator, the scorer, the three payload classes — need
  none of the above, and their tests run in the core offline suite.

## How it works

### describe → TaskSpec

Instructions are three parts in one order: the world's own conventions for driving it (the `apis`
object, the API documentation calls, how to log in as the supervisor), the task's shipped
instruction and whose accounts it is carried out with, and then the appended paragraph.

The paragraph is 1,389 bytes of pure ASCII and is **byte-identical on every task**, so an agent
reading its hundredth task reads the words it read on its first. It asks for one `Filing` task in
the seeded `Task Log` project, placed in a section, with a priority, a duration and a duration
unit; a `task-log` label with a colour, attached to it; and one line per waiting request in the
description, holding the request's reference and its band.

It then says, in as many words, that it does not say which recorded date a window starts from,
which days in the window are counted, which band a count on a printed figure takes, what an
undated request gets, or which section, priority, duration unit or label colour to use. Every one
of those is constructible from what the world shows. None is named.

### Tools (served over MCP)

| Tool | Terminal | What it does |
|---|---|---|
| `execute(code)` | `none` | Runs one block of Python in the world's shell and returns what it printed. The shell persists across calls; `apis` is already bound. |
| `submit()` | `score` | Ends the task. Seals first, then `finalize` scores. Takes no arguments and reveals nothing. |
| `terminate()` | `abort` | The reserved no-credit end. |

The base task's answer is recorded the way AppWorld records it, with
`apis.supervisor.complete_task(answer=...)` inside `execute`. `submit` is the seal.

### finalize + verify

`submit` seals the episode. `finalize` then reads the world's end state, collects the base task's
own checks, and scores the filing against the drawn key — in the **serving** process, never in the
world's. The worker's protocol has no field for the key and no comparison in it, so a world an
agent had complete control of still could not be made to say what the answer was.

`_verify` publishes the numbers plus the matched pair: `report` carries the receipt, `notice`
carries the digest, and both are always published whatever regime the run is serving, because the
env does not know the regime and an env that published only the one its run would reveal would
have made the record depend on the treatment.

## Tasks

318 tasks, index-addressable and stable, from AppWorld's `test_challenge` split. The split holds
417 tasks over 139 scenarios; the served roster is smaller for two committed reasons, both settled
before any episode and recorded in [`task_manifest.txt`](task_manifest.txt):

- **98 tasks have no admissible backlog.** A backlog is admitted only if the 64 conventions give
  64 different answer keys and moving any one choice on its own changes at least two requests.
  The generator redraws up to 140 times against a task's own reference date; a task no draw
  satisfies is not served, because a backlog that cannot separate two conventions cannot be
  graded against either of them.
- **1 task's own evaluation names a model the appended chore adds.** The chore adds
  `todoist.Task`, `todoist.Label` and `todoist.TaskLabelLink`, which go into every task's ignore
  list so a scenario's "nothing else changed" assertion is not failed by the port. Ignoring a
  model a scenario *asserts on* would turn a passing check into a failing one, so that task is not
  served either.

A task's world is the shipped one with 35 rows added: the `Task Log` project (its description
holds the working week and three closed dates and nothing else), a collaborator link to the
supervisor alone, four sections, 28 dated requests and one undated one. The rows go into the
task's **input** databases, so the runtime world and the evaluator's baseline see the same
backlog and the scenario's own score survives them.

Repeats are legal and a repeat is a repeat: the backlog, the key and the world's own generator
seed are all deterministic functions of the task identity.

## Scoring

| Metric | What it is |
|---|---|
| `ledger_fraction` | **the headline.** Correct bands over 29 requests, with a request the agent filed no line for counted incorrect. |
| `pinned_fraction` | **the control.** Correct stored slots over 4. Each reads its own slot alone, so a failure crosses off one option and says nothing about any other row: whatever the agent learns, this cannot pass `mean(1/c) = 77/240 = 0.3208` in expectation. A run in which it moves with the headline is a run whose headline is measuring something else. |
| `exercise_fraction` | How much of the ledger was attempted. A contrast carried by this is an agent learning a chore. |
| `parse_fraction` | How many written lines named exactly one request. Reported as a shape error, never graded as a lesson. |
| `assertion_fraction` | The base task's own checks, from AppWorld's evaluator, reported beside the headline and never summed with it. |
| `distinct_bands`, `filing_rows` | A degenerate submission and hedging by over-filing, both visible and neither scoreable. |
| `world_digest`, `rng_digest` | What the world became, and the state of the generator it draws from. |

```python
import shogym
result = shogym.result_from_trace("shogym_logs/run.jsonl", env="appworld", task="0")
print(result.terminated, result.value("ledger_fraction"))
```

Filter semantics are shared; see [`../README.md`](../README.md).

### The three payload classes

All three list every scored item in canonical order and **are the same length on the wire**, by
construction rather than by padding: every column is fixed width and every cell comes from a
closed ASCII vocabulary, so no world data reaches a payload and the corpus's own non-ASCII cells
cannot change the byte count.

- **the receipt** (`report`, under `Information`): `PASS` or `FAIL` per item and nothing else. No
  expected value, no rule statement, no per-item error count, and no naming of which choice an
  item turns on.
- **the digest** (`notice`, under `Placebo`): `sha256(task || check || observed)[:4]` per item. A
  function of the task identity and the agent's own submission, both of which are already in the
  agent's transcript, so it carries nothing about the key by construction.
- **the drawn receipt** (`report`, with `report="drawn"`): the receipt's shape with verdicts that
  were sampled rather than computed. The number of passing dated requests is drawn from
  [`pass_counts.txt`](pass_counts.txt), the roster's own distribution enumerated over every served
  task crossed with all 64 conventions, so it is not separable from a real receipt by its count
  either. It is **not** inert and **not** a control: it states false verdicts, and a reader that
  acts on them can finish worse than one handed nothing. The digest is the zero-information
  payload.

## Fidelity & deviations

**Pinned upstream:** `appworld` 0.1.3.post1, the PyPI release cut from commit
`66ad8099e12188ece0d3fe45e661dbc01880813b`. **Pinned data:** `data-0.1.0.bundle`, 34,280,074
bytes, sha256 `fd9f9608c2ec71ed0ac25c3633a738b9129a318a129e31230425b9188e508250`.

- **The world, the task text and the base score are upstream's, unmodified.** The convention, its
  option sets, the seeded backlog, the scorer of the filing and the three payload classes are this
  port's, entirely.
- **The `test_challenge` split is used as experience, not as a held-out test.** AppWorld's authors
  ask that the test splits not be used for teaching or prompt tuning. This port reports no
  AppWorld score: `assertion_fraction` rides along beside the headline as context for the appended
  chore. Anything published off it should disclose the repurposing rather than borrow AppWorld's
  held-out-test semantics.
- **Three models are added to every task's ignore list** when the base task's checks are
  collected, so the appended chore does not fail a scenario's changed-model assertion. A task
  whose own evaluation names one of them is not served (see [Tasks](#tasks)).
- **Assertion rows carry no expected-versus-actual diff.** Upstream's failure traces quote the
  values a check asserts on, which would put world data — and part of the task's answer — into a
  payload whose whole argument is a closed ASCII vocabulary. Rows report `ok` / `not ok`. The
  diff-carrying variant is not built.
- **The world runs in a subprocess under its own interpreter,** for the packaging conflict above
  and because AppWorld freezes the process clock and holds each app's database engine on a class
  attribute. That subprocess binds a loopback port and refuses every request without a token
  minted at spawn. AppWorld's own environment server publishes `evaluate`, `save_state` and
  `load_state` unauthenticated on every interface; this one does not.
- **The generator's global-RNG state is captured and restored around every episode,** and the seed
  it is started from names the task and nothing else — not the session, not the run, not the
  feedback regime. AppWorld saves databases and not generator state, and every `login` draws from
  the global generator, so a port that replayed only the databases would serve two worlds that
  agreed on their contents and disagreed on their next draw.
- **Not built:** the yoked payload (a donor's submission and the receipt computed on it), the
  foreign rendering of a verdict vector as prose and the parser that round-trips it, the fixed-size
  envelope that would pad those to one predeclared size, and any assignment, forking or analysis
  machinery. This is the environment and its three payload classes; the experiment that uses them
  is not in this repository.

## Gotchas

- **`env.num_tasks` is 318, not 417.** Task indices address the manifest, not the split.
- **The first construction is slow and online.** Building the runtime and fetching the corpus is a
  one-time cost; deriving a task's seeded world costs about a second the first time it is served
  and nothing after that. Set `APPWORLD_ROOT` to skip the download.
- **A different `pulse` is a different experiment.** It fixes the convention and the four stored
  slots for every task. Scores drawn under two pulses are not comparable.
- **`show_tasks` returns a dict**, not a list: seeded requests carry no section, so they arrive
  under `no_section_tasks`. `show_projects` defaults to `page_limit=5`.
- **The receipt names the task, and the draw is deterministic.** Both payloads print
  `task <task_id>` in their header, and the key is a pure function of that identity and the run's
  `pulse`. An agent that can read this source *and* knows the pulse can therefore compute the key
  from the header alone. Nothing the env can do closes that: a payload has to identify itself, and
  a draw that a reader cannot reproduce is a draw nobody can audit. Keeping the source and the
  pulse off the agent's filesystem is the harness's job. The same goes for the corpus: the ground
  truth ships in plaintext under `data/tasks/<id>/ground_truth/`.
- **The chore is deliberately not dodge-proof.** A forcing assertion would correlate the two score
  components by construction and turn a visible dodge into an invisible one, since an agent that
  writes one band on all 29 lines passes any forcing check anybody would write. `exercise_fraction`
  and `distinct_bands` are what make a dodge visible instead.

## Layout

| File | What it is |
|---|---|
| `env_v1.py` | the `Env`: describe, session lifecycle, `finalize`, `_verify` |
| `mcp_server.py` | the two served tools |
| `ledger.py` | the convention space, the backlog generator, and the gates a backlog must clear |
| `world.py` | the appended paragraph, the four stored slots, and the derived corpus |
| `scorer.py` | the drawn key, the per-item verdicts, and the fractions |
| `payload.py` | the receipt, the digest and the drawn receipt |
| `worker.py` | one world, in its own interpreter, behind a token-gated loopback port |
| `adapter.py` | the pins, the provisioning, the served roster, the worker client |
| `task_manifest.txt` | the 318 served tasks, settled before any episode |
| `pass_counts.txt` | the roster's own distribution of passing-request counts |
