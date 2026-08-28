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

`config_digest` covers the draw, the payload class, the block budget, every constant a payload is
generated from, the text the agent is given (the world guide, the tool guide and the appended
paragraph), the generator that decides the seeded backlog (its constants and the bytes of the
three modules that implement it), the corpus this run actually serves (all of it, including the
134 MB of shared base episodes read as input), the derivation layout, what the worker's
interpreter turned out to hold, and a hand-bumped scoring version that moves when how a score is
read moves. "What the interpreter holds" is a stated scope rather than a slogan: it is every
installed source and data byte under `site-packages` plus the base executable `bin/python`
resolves to. It is not the base interpreter's standard library, which belongs to the host rather
than to anything this port installs, and it is not the bytecode caches, which are left out because
the whole interpreter is compiled at provisioning with hash-based caches the import system checks
against each source's own hash, and workers write none back: every `.pyc` that can be consulted
therefore stands for a source the digest did read. An env serves the corpus it
read at construction for the whole of its life: the instructions, the supervisors and the dates
come from that one reading, so a corpus edited underneath a running env cannot put new authored
text behind an unchanged fingerprint. Pinning the text is not the whole of it, because a task's
databases and its ground truth are read when that task is first served rather than at
construction, so each unit of the corpus is checked against what the snapshot read before it is
derived, and a unit that moved is an episode that does not happen. A stream's `deadline` and
`max_in_flight` belong to a run's identity too and are not an env's to know, so the stream folds
them in itself: what a record is filed under is a structured identity whose members are the name
passed here, the digest each env in the queue published about itself, the deadline and the
capacity, compared member by member on every resume. The derived and grader caches are named by
the same source digest and carry a stamp inside them saying what they were built from, so a run
pointed at a second corpus cannot serve material derived from the first.

Passing `identity` is how a record defends itself. A directory that already names one refuses a
resume that names a different one, and refuses a caller that names none: the record has said what
produced its rows, and rows that decline to say make it unreadable as one run afterwards. A
directory recorded before identities existed is refused too, in the other direction: it cannot say
what produced its rows, so a named caller has to state that it knows
(`adopt_unidentified=True`) rather than have the silence read as consent. That assertion is
performed once and written into the directory, so the next ordinary resume under the same identity
needs no flag.

The name is also checked against the env, which is the one party that knows, and the env's own
answer is a member of the identity rather than something searched for inside that name. This env
declares the item it describes itself with (`identity_feedback_name = "config_digest"`) and
answers to it before any episode runs, so the stream reads the digest off the env at construction
and writes it into the ownership claim: a run killed between its claim and its first row still
leaves behind what its env said it was, and a resume against a changed corpus, draw or runtime is
refused before a task is spent. Every terminal this env produces publishes the same
`config_digest` at inference level, and a row whose env says something else than the identity
holds for it is refused before it can be scored. The first row of an env that publishes a digest
also binds the directory for that env, which is what covers an env that publishes one without
answering to it, and it works whether or not a caller named an identity at all.

Each row's `feedback_regime` is the arm the task was **assigned**, not the arm it was told through.
It has to be: the row is fsynced before the policy's answer is composed, because the answer is
composed from the recorded row. So a cancelled terminal, a task the stream ended itself and a
policy that could not answer all leave a scored row stamped `information` or `placebo` with nobody
told. That is the field an intention-to-treat estimate wants, and every assigned task has one. What
was actually delivered is a separate state in `exposures.jsonl` beside the results, joined by
lease: a revealing run writes one line per terminating call it answered, and a row with no line
there was never told. A run under `Never` writes no such log, because it opens no channel.

The absence of a line is load-bearing, so two things fail closed on it. A delivery whose line
cannot be written is not delivered: the terminating call is answered with the empty member every
other silence uses, and the stream stops. And a terminal that outran its deadline delivers
nothing even when the env's finalization eventually returns: the watchdog has already sealed that
task into an unscored `timeout` row, which the design counts as a failed delivery to be scored at
the floor and retried rather than as a dose, so the late answer carries the empty member and the
log stays silent about it.

## Requirements

Pin-and-install mechanics are shared; see [`../README.md`](../README.md). What is specific:

- **The `appworld` extra carries no packages, and cannot.** `appworld` 0.1.3.post1 pins
  `pydantic>=1.9,<2`; shogym's MCP layer needs `pydantic>=2.7`. No environment satisfies both. The
  port therefore builds an interpreter of its own (a virtual environment under
  `~/.cache/shogym/appworld/runtime-<version>-<sha>/` holding the pinned release) and runs every
  world in a subprocess under it. `SHOGYM_CACHE` relocates it; `uv` is used when it is on `PATH`
  and `venv` + `pip` otherwise. This is provisioned when an `appworld` env is **constructed**, so
  `import shogym` stays offline. Both pins are in that name and in a stamp inside it, and the
  installed distribution is checked to be the pinned release before the tree is published, so a
  build that resolved something else never gets served and a pin that moves builds a second
  interpreter rather than reusing the first. What the wheel cannot say is which commit it was cut
  from; that half of the pin names the runtime and is not verifiable against the artifact, and
  what the realized code actually is comes from hashing the installed bytes instead. The app
  sources the wheel ships packed are unpacked into that interpreter afterwards, its bytecode
  caches are rewritten as hash-based ones, and a second stamp written once both have exited zero
  is what says it is done: the runtime is already published by then, so an unpack interrupted part
  way through would otherwise leave a complete runtime with an incomplete package inside it.
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
- **The derived cache's name says everything that filled it.** Four things: the corpus it was
  derived from, the derivation layout, the generator (its constants and the bytes of `ledger.py`,
  `world.py` and `worker.py`), and the realized interpreter, which is the process that writes a
  task's seeded database log through upstream's own model layer. Each of them is also inside a
  stamp written into the cache, because a name cannot cover a tree that was edited, moved or
  restored under it. So pointing `APPWORLD_ROOT` at a second corpus, editing how a task is derived
  without touching a ledger constant, or reinstalling the runtime, all derive a fresh tree rather
  than reusing one an older combination seeded.
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
| the answers, as files | the served tree contains no `ground_truth`, and **no path in it leads to a directory the answers are a sibling of**: the task's own files are copies, and the shared base is named by a link into the derived tree, which has no answers in it by construction |
| the grader's tree | a directory with an unguessable name under a private parent, not a neighbour of the served root |
| one episode's grade | upstream's evaluator report is disabled, and an episode's whole output tree is named absolutely and lives outside every served corpus, so no episode holds another's end state, logs or verdicts |
| the corpus itself | served inputs are independent copies rather than hard links, so a write through the served pathname changes neither the corpus later episodes are derived from nor the baseline the grader diffs against |
| the next episode's inputs | each episode is served its own view: its task's world is copied per episode and removed with the episode, and everything else it can reach (the 134 MB of shared base, and the shared task cache the views are copied out of) is sealed read-only, so a write through one episode's served pathname is not in the next one's starting inputs, or the other arm of its pair's |
| the *names* the next episode resolves | a view names the shared entries by absolute path, so the directory holding those names is sealed read-only too and opened only under the derivation lock; without it, `base_dbs` could be renamed aside and something else put there under the same name, which every current and later view would follow |
| the drawn key | never sent to either process: the protocol has no field for one |

**There is no file-access audit, and a run must not be read as though there were one.** An earlier
revision of this port installed an audit hook to journal files opened from outside the served tree.
It recorded opens made by the worker process, and in isolation it recorded opens made inside an
embedded IPython shell, but it did not record a file read through a served `execute` call in a live
world. A control that works on some paths and not the one that matters is worse than none, because
an empty journal reads as "file access was audited and nothing was found" when it means nothing of
the kind. It has been removed rather than shipped with a caveat. Nothing here records what an
agent's code opens.

**Nothing the host needs to know comes from the process that runs the agent's code.** There is no
seal command, no quiesce command and no read command: the writer on the protocol's own socket is
reachable from inside the interpreter that executes agent-authored Python, so a reply saying "I
have flushed", "I have stopped" or "the filing says X" is a reply the episode could have written.
The host signals the worker's process group, confirms from the process table that the group is
empty, and grades what is on disk.

**What is on disk is the world as upstream last persisted it.** `AppWorld.execute` ends with its
own save into the episode's output tree and `initialize` writes one before any block runs. So an
episode that ran N blocks is graded on the state after block N, and one that ran none is graded on
its opening state. The generator digest is written to the same tree for the same reason a
lifecycle fact is not asked for: a diagnostic read out of a reply is a diagnostic the episode can
choose.

**What an empty process group proves, and what it does not.** It proves quiescence *at the moment
the snapshot is taken*: nothing of the episode's was running when the tree was copied, so the
filing, the digests and the evaluator all read one state. It does not prove that nothing was
written on the way down. The stop is SIGTERM, a short grace, then SIGKILL, and SIGTERM is
catchable, so a process that ran agent-authored code has that grace in which an exit path of its
own could still write. The grace is short for exactly this reason and it is not zero, so the
honest reading of a graded tree is "the state upstream persisted, plus anything the episode's own
shutdown persisted inside the grace", not "the state at the end of block N and nothing else".
Making the two the same needs a namespace that can be destroyed rather than a signal that can be
handled, which is
[shojin-lab/shogym#140](https://github.com/shojin-lab/shogym/pull/140).

**A stop the process table will not confirm ends the episode unscored.** Not reaped inside the
bound, or a process table that would not answer: each of those leaves a tree something may still
be writing to. The group is signalled *before* the leader is reaped, which is what makes the
number safe to use and what stops an episode buying an unscored row by killing its own world: a
pid is reserved until its parent reaps it and a group exists while any member does, so an
exited-but-unreaped leader is still holding the group and is stopped and confirmed like any other.
A leader something else reaped first leaves a number nothing may signal or enumerate. The
episode is closed as an infrastructure failure rather than scored on it, so the row is `unscored`
with a `finalize_error` closure and nothing in `observed` at all, which means neither feedback arm
has a payload to reveal. The same is true of an output tree the grader refuses.

**The grader is given a snapshot, not the tree the world wrote.** The grading process is pointed
at the root that holds the answers, so a link left under the output tree would resolve there. The
root itself is checked before it is resolved, and every entry under it must be a plain file or
directory; anything else refuses the episode rather than being skipped, because a grade over a
tree with an entry quietly dropped is a grade over a tree nobody submitted. The walk is bounded in
nodes, bytes, depth and elapsed time, and it stops on a flag the finalizer sets if its own await
is abandoned, because an episode wrote this tree and nothing else here would stop it deciding how
long finalization takes. It reads the filing, the databases' digest, the generator digest and the base
task's own evaluator, all from those same bytes, so what is scored and what is graded are one
state by construction rather than two observations that happened to agree.

**The residual, stated exactly.** The private tree is hard to guess and not hard to read: it is the
same uid, so its 0700 mode stops other users and stops nothing else. The read-only seal on the
shared base, the shared task cache and the directory holding their names is the same kind of thing: it refuses an ordinary write,
upstream never writes there, and the process that runs agent code owns those files and could put
the write bits back. Two ancestors above that directory stay writable as well, because the port
writes its cache stamp in one and provisions in the other, so the shared root's own name is
movable by a process willing to work a level up. The confirmed stop is likewise the process table's word rather than a
namespace: a process that left this group is outside what it can see. The
port's own source is readable, and with it the draw's algorithm. The run's provenance directory
retains true reports even under `Placebo`. Anything else the user running the port can read is
readable.

**What would close it** is an OS namespace in which none of that is mounted: the worker in a
container with only the served tree bound in, and the shared base bound in read-only rather than
merely sealed read-only ([shojin-lab/shogym#140](https://github.com/shojin-lab/shogym/pull/140)). That is assessed and not built here, because it is
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
  what the interpreter actually holds rather than what version it was asked for, the served
  roster, the frozen count table, a scoring version, and what the corpus at `APPWORLD_ROOT`
  actually holds rather than what the pin says it should. Published at
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
