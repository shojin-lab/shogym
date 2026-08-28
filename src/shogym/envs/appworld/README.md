# `appworld`: AppWorld, with one paragraph appended that nobody answers for you

A faithful shogym port of [AppWorld](https://github.com/StonyBrookNLP/appworld): carry out a
natural-language instruction across nine simulated apps by writing Python against their APIs. On
top of the rented task, one authored paragraph asks for a filing log in a corner of the world no
scenario touches, whose values a house convention computes from the world's own data, and the
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

TaskStream(
    shogym.make,
    [TaskRef("appworld", 0)],
    prov_dir=...,
    feedback=Information(),
    # Pass the fingerprint. It is what makes the rows of one record one measurement, and a resume
    # under a changed draw, payload class, block budget or corpus is refused rather than appended.
    identity=shogym.make("appworld").config_digest,
)
```

Passing `identity` is how a record defends itself. A directory that already names one refuses a
resume that names a different one, and refuses a caller that names none: the record has said what
produced its rows, and rows that decline to say make it unreadable as one run afterwards.

## Requirements

Pin-and-install mechanics are shared; see [`../README.md`](../README.md). What is specific:

- **The `appworld` extra carries no packages, and cannot.** `appworld` 0.1.3.post1 pins
  `pydantic>=1.9,<2`; shogym's MCP layer needs `pydantic>=2.7`. No environment satisfies both. The
  port therefore builds an interpreter of its own (a virtual environment under
  `~/.cache/shogym/appworld/runtime-<version>/` holding the pinned release) and runs every world
  in a subprocess under it. `SHOGYM_CACHE` relocates it; `uv` is used when it is on `PATH` and
  `venv` + `pip` otherwise. This is provisioned when an `appworld` env is **constructed**, so
  `import shogym` stays offline.
- **The data bundle is fetched and checked.** ~33 MB, once, into
  `~/.cache/shogym/appworld/corpus-0.1.0/`. Upstream's own downloader verifies nothing; this one
  refuses a bundle whose size and sha256 are not the pinned pair. `APPWORLD_ROOT` pointing at a
  directory whose `data/tasks` exists is used as it stands, so a machine that already ran
  `appworld download data` needs no download.
- **No API key, no Docker, no network once provisioned.**
- The port's pure pieces (the backlog generator, the scorer, the three payload classes) need
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
own checks, and scores the filing against the drawn key, in the **serving** process and never in
the world's. The worker's protocol has no field for the key and no comparison in it, so a world an
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

**One convention is drawn per scenario, not per task.** AppWorld numbers a scenario's
instantiations `_1`, `_2`, `_3`: the same template with different values, which is the sibling
relation this port rents rather than invents. A task and its sibling are therefore graded under
one rule, so a grade on the first is a grade about something the second still contains. Per task
it would be a rule that had already changed by the time the next task arrived, and the difference
between two arms would measure nothing. Two scenarios are two draws, so no single draw is the
whole experiment.

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
| `pinned_fraction` | **the control.** Correct stored slots over 4. Each reads its own slot alone, so a failure crosses off one option and says nothing about any other row. See [What the control caps](#what-the-control-caps) for the two numbers this involves, which are not the same number. |
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

### What the control caps

A stored slot is answered by picking one of `c` options, and a verdict on it says only whether the
pick was right. Two numbers follow and they are easy to confuse.

- **Ungraded compliance is `1/c` per slot**, so the four slots average
  `mean(1/c) = (1/4 + 1/5 + 1/3 + 1/2) / 4 = 77/240 = 0.3208`. That is what an arm with no
  feedback scores, and it is the level to check the placebo arm against.
- **After one verdict the ceiling is `2/c` per slot**: right one time in `c` and knowing it,
  wrong the rest and then picking from the `c - 1` survivors. The four slots average
  `mean(2/c) = 77/120 = 0.6417`.

So the cap on what a grade can be *worth* here is the difference, `mean(1/c) = 0.3208`, and
reaching it takes no skill: a competent reader and a brilliant one cross off the same option. That
is the whole point of keeping these four beside the headline. They are a negative control **only
where they sit at their cap**; a reader below it can climb toward it as experience accumulates, so
a moving pinned level is evidence of a broken fork only when the level was already there. Both
numbers are published beside the trend for that reason.

The ledger is not like this, and that is the difference the instrument rests on: a band is
*computed* from a request's own dates rather than picked off a list, so different conventions
collide on some requests and separate on others and there is a pattern to explain rather than a
bit to invert.

### What one receipt settles, and what it does not

A receipt is the vector of bits saying, per request, whether the drawn convention would have
written what the agent wrote. Conventions sharing a vector are conventions no reader can separate
however well it reads. Enumerated on the served roster, against the submission a measured panel of
readers actually produces:

| quantity | value |
|---|---|
| distinct verdict vectors, of 64 conventions | 47.8 |
| largest set of conventions sharing one vector | 7.7 |
| conventions still standing after one receipt | 2.88 |
| Bayes-action ceiling | 0.9530 |
| lookup floor (what the receipt's own labels concede) | 0.5901 |
| **headroom** | **0.3629** |

**The map is not injective and cannot be made so.** Some pairs of conventions agree on every
request the world can supply: when no window under the drawn anchor lands on a printed figure, the
boundary rule changes nothing anywhere, and no extra request can make it. Measured at 28, 34, 40
and 48 dated requests, no backlog at any length has a one-to-one map, and the mean posterior
asymptotes near 1.7 rather than reaching 1.

So the claim this instrument supports is **matching the drawn convention closely, not naming it**.
64 conventions down to about three is most of what there was to learn, and the headroom figure is
what a grade is worth on that scale. Anything published off this port should say "matches the
drawn convention more closely" and never "identifies the convention".

The admission gate is the one the enumeration justifies and no more: 64 distinct answer keys, and
every single-axis move changing at least two requests. Both are checked per backlog, and the
narrowing above is enumerated and asserted across the roster.

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
  agent's transcript, so it carries nothing about the key by construction. The base task's own
  checks render their observed value as `not determined`, always: a check asserts over models the
  agent touched through nine apps and has no value the agent wrote anywhere, and naming its
  outcome in the one column both classes share would make the inert payload move with the base
  task's result. A test flips every check and asserts no byte of the digest moves.
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
  values a check asserts on, which would put world data (and part of the task's answer) into a
  payload whose whole argument is a closed ASCII vocabulary. An assertion row's observed value is
  `not determined` in every payload class, and the outcome appears only in the graded receipt's
  verdict column. The diff-carrying variant is not built.
- **The world runs in a subprocess under its own interpreter,** for the packaging conflict above
  and because AppWorld freezes the process clock and holds each app's database engine on a class
  attribute. That subprocess binds a loopback port and refuses every request without a token
  minted at spawn. AppWorld's own environment server publishes `evaluate`, `save_state` and
  `load_state` unauthenticated on every interface; this one does not.
- **The base task is graded in a second process that never runs agent code.** The world an
  episode is served from is built with `load_ground_truth=False`, so the answers and the checks
  are not objects in the process an agent's code runs as and there is no evaluator there to call.
  At the seal the end state is flushed to disk and a short-lived grading process reads it.
- **The generator's global-RNG state is captured and restored around every episode,** and the seed
  it is started from names the task and nothing else: not the session, not the run, not the
  feedback regime. AppWorld saves databases and not generator state, and every `login` draws from
  the global generator, so a port that replayed only the databases would serve two worlds that
  agreed on their contents and disagreed on their next draw.
- **Deferred: the derived cache's identity does not fingerprint its source.** The cache name
  carries a digest of the backlog generator's constants, so changing a cut value or an option set
  derives a fresh corpus. It does not carry the source corpus's own identity or the derivation
  code's version, so pointing `APPWORLD_ROOT` at a different corpus, or changing how a task is
  derived without changing a ledger constant, would reuse the existing tree. The pinned bundle
  makes the first unlikely and neither is a silent wrong answer within one pinned run, so it is
  recorded here rather than fixed.
- **Not built:** the yoked payload (a donor's submission and the receipt computed on it), the
  foreign rendering of a verdict vector as prose and the parser that round-trips it, the fixed-size
  envelope that would pad those to one predeclared size, and any assignment, forking or analysis
  machinery. This is the environment and its three payload classes; the experiment that uses them
  is not in this repository.

### What the worker boundary is, and what it is not

The code an agent writes runs **as** the worker process, with that process's filesystem and
network, as the same user as the run. AppWorld's own `SafetyGuard` is not a boundary either: it
refuses `import sys` and lets `__import__("sys")` through, and it null-patches `os.walk` and
`os.listdir` while `io.open` reads whatever the user can read. **This port does not claim a
sandbox, and on one uid no filesystem arrangement is one.** What it claims is that the cheap
routes are closed and the expensive one is visible, each tested by running the probe:

| what | how |
|---|---|
| the worker's token and corpus root | passed on stdin, read once and closed, so `sys.argv` carries only the script and the subcommand |
| the serving process's environment | replaced with an allow-list, so an inherited provider key is not there to read |
| the answers, as objects | the world is built without ground truth, and there is no evaluator in the process |
| the answers, as files | the served tree contains no `ground_truth` and **no symlink into the corpus**: every file in it is a hard link or a copy, so no path in it names a directory the answers are a sibling of |
| the grader's tree | a directory with an unguessable name under a private parent, not a neighbour of the served root |
| one episode's grade | upstream's evaluator report is disabled, and an episode's whole output tree is named absolutely and lives outside every served corpus, so no episode holds another's end state, logs or verdicts |
| the corpus itself | served inputs are independent copies rather than hard links, so a write through the served pathname changes neither the corpus later episodes are derived from nor the baseline the grader diffs against |
| the next episode's inputs | each episode is served its own view: its task's world is copied per episode and removed with the episode, so a write through one episode's served pathname is not in the next one's starting inputs, or the other arm of its pair's |
| the drawn key | never sent to either process: the protocol has no field for one |

**There is no file-access audit, and a run must not be read as though there were one.** An earlier
revision of this port installed an audit hook to journal files opened from outside the served tree.
It recorded opens made by the worker process, and in isolation it recorded opens made inside an
embedded IPython shell, but it did not record a file read through a served `execute` call in a live
world. A control that works on some paths and not the one that matters is worse than none, because
an empty journal reads as "file access was audited and nothing was found" when it means nothing of
the kind. It has been removed rather than shipped with a caveat. Nothing here records what an
agent's code opens.

**The world is stopped before it is graded.** Sealing closes the tool surface and does not stop
work an earlier call left running, so the worker and everything it started (it owns a process
group) are terminated after the end state is read and before the evaluator runs. What is scored is
a snapshot on disk that nothing can still be writing to.

**The residual, stated exactly.** The private tree is hard to guess and not hard to read: it is the
same uid, so its 0700 mode stops other users and stops nothing else. The port's own source is
readable, and with it the draw's algorithm. The run's provenance directory retains true reports
even under `Placebo`. Anything else the user running the port can read is readable.

**What would close it** is an OS namespace in which none of that is mounted: the worker in a
container with only the served tree bound in. That is assessed and not built here, because it is
not configuration. The existing agent image mounts `/root`, `/work` and a read-only `/cfg` and
publishes no ports, so it needs a new image (the pinned interpreter baked in), a new argument
builder, a fixed container port published to host loopback and read back with `docker port`
(a container-loopback listener cannot be forwarded, and `--network none` and `-p` are mutually
exclusive), and container-aware teardown. Until it lands there is no mitigation, only the closed
routes above: a run whose scores must survive an adversary needs that container, and a run made
before it should be read knowing that nothing was watching.

## Gotchas

- **`env.num_tasks` is 318, not 417.** Task indices address the manifest, not the split.
- **Construction is slow, online, and blocking.** Building the runtime, fetching the corpus and
  copying it into the two derived views is a one-time cost; deriving a task's seeded world costs
  about a second the first time it is served and nothing after that. `TaskStream` builds envs off
  the event loop, so a queue is not held by it; a caller constructing an env directly on a loop it
  is also serving on will block that loop and should build it in a thread. Set `APPWORLD_ROOT` to
  skip the download.
- **A different `pulse` is a different experiment.** It fixes the convention and the four stored
  slots for every leg. Scores drawn under two pulses are not comparable, and neither the pulse nor
  the payload class appears anywhere else in a run's record, so every row carries a
  `config_digest`, which covers the draw, the payload class, the block budget, the pinned bundle,
  the interpreter, the served roster, the frozen count table, a scoring version, and what the
  corpus at `APPWORLD_ROOT` actually holds rather than what the pin says it should. Published at
  inference level: the record keeps it and no feedback policy
  reveals it, not even the one that reveals everything else. Reopening a provenance directory
  under a different pulse is then visible in the rows. Refusing such a resume belongs to the
  stream, which owns run identity; an env is handed a task and does not know which run it is
  being served into.
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
