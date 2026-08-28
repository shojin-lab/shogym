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

**That snippet is a development loop and not an arm of anything.** A `ServedEpisode` has no
feedback policy in it, so its terminal answers with every episode-level item the env published:
the numeric metrics, the receipt and the digest together. Use it to see what an episode produces.
An experiment runs the two arms below, where one policy decides what a terminal reveals and the
record says which arm each row was assigned.

### The paired arms

The paired feedback policies are what this env was built for. `Information()` answers a
terminating call with the receipt; `Placebo()` answers it with a digest of the agent's own
submission, the same size on the wire; `Never()` answers with no channel at all.

**Which arm is running is the server's configuration and never the agent's.** The agent process is
launched identically in both arms: the same argv, the same environment, the same working
directory. What differs between them is one file that only the MCP server reads. That is not
tidiness. Before the first task is dispensed the run directory holds `claim.json`, which names the
regime the run serves under, and afterwards `results.jsonl` holds every payload the env published
on every row, the receipt included, whichever arm the terminal was answered under. An agent that can read its own launch environment, or that directory, can read its
assignment before treatment and read the receipt the control arm exists to withhold. So the
configuration travels in the MCP config's `env` block, which reaches the server process and
nothing else; the records go somewhere the agent is not working; and the built-in tools that could
read either are taken away.

One directory holds the pair, and it is not one the agent is launched into:

```bash
repo=$(git rev-parse --show-toplevel)     # this line from inside your checkout; the rest is absolute
pair=~/appworld-pair
mkdir -p "$pair"

# The arm, and everything else a record is filed under. Written where the server reads it, on its
# way to `serve.py`, and nowhere the agent's own process can see. The paths are absolute because
# Claude Code resolves a relative command argument in an MCP config against the directory the
# agent was launched from rather than against the directory the config file lives in, so a
# relative `serve.py` is a server that starts from one directory and not from another.
arm() {
  cat > "$pair/mcp.json" <<JSON
{
  "mcpServers": {
    "shogym": {
      "command": "uv",
      "args": ["run", "--project", "$repo", "python", "$repo/examples/claude_code/serve.py"],
      "env": {
        "SHOGYM_ENV": "appworld",
        "SHOGYM_TASKS": "0",
        "SHOGYM_FEEDBACK": "$1",
        "SHOGYM_DEADLINE": "1800",
        "SHOGYM_IN_FLIGHT": "1",
        "SHOGYM_RUNS": "$pair/runs"
      }
    }
  }
}
JSON
}
```

Then the two launches, one per arm, and the second is a copy of the first:

```bash
arm information   # the treatment: the terminal carries the receipt, and nothing beside it
claude -p "$(cat "$repo/examples/claude_code/PROMPT.txt")" \
    --mcp-config "$pair/mcp.json" --strict-mcp-config \
    --allowedTools 'mcp__shogym__*' --permission-mode dontAsk \
    --disallowedTools Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Agent,Task

arm placebo       # the control: the same channel, the same shape, the inert digest in it
claude -p "$(cat "$repo/examples/claude_code/PROMPT.txt")" \
    --mcp-config "$pair/mcp.json" --strict-mcp-config \
    --allowedTools 'mcp__shogym__*' --permission-mode dontAsk \
    --disallowedTools Bash,Read,Write,Edit,Glob,Grep,WebFetch,WebSearch,Agent,Task
```

The deny list is part of the measurement here rather than hygiene. Claude Code keeps its built-in
tools alongside the served ones by default, which is the right default for the quickstart and the
wrong one for an arm of a pair: `Read` alone reaches the config that says which arm this is, and
the env's own task definitions on disk. Removing the built-ins is what separates the two arms
until the agent runs inside a container that cannot see any of it. Do not reach for `--tools ""`,
which strips the served tools too.

The arms run one after the other, because both name the same config path and only that file's
contents change between them. Running both at once takes a second `$pair` with a config of its
own, and then the config path is the one thing the two launches differ in, which is a name that
says nothing about the arm.

Each arm writes its own provenance directory under `$pair/runs`, named for its regime
(`appworld-information-<stamp>/`), because a directory that holds one arm's rows refuses the
other's. The deadline and the capacity are the same in both configs on purpose: a deadline decides
whether a slow episode is scored or timed out and a capacity decides what an agent may work on
next, so two runs that disagree on either are two measurements rather than two arms of one. Read
an arm back by naming its directory: `uv run --project "$repo" python
"$repo/examples/claude_code/results.py" "$pair/runs/appworld-information-<stamp>"`, and again for
the other one.

**`SHOGYM_FEEDBACK=immediate` is the practice path, not a third arm.** It is the default, it
hands back every episode-level item the env published (both payloads and the numeric grades), and
a run under it is for watching an agent work rather than for comparing anything.

The same policies from Python, for a runner that builds its own stream:

```python
from shogym.serve.stream import Information, Placebo, TaskRef, TaskStream

TaskStream(
    shogym.make,
    [TaskRef("appworld", 0)],
    prov_dir=...,
    feedback=Information(),
    # Pass the fingerprint, and whatever else about this run makes its rows one measurement. The
    # env's digest covers the draw, the payload class, the block budget, the corpus, the worker
    # image and the machine it was given; the deadline and the capacity are the stream's, and a
    # record whose rows ran under two of them is a record about two different opportunities. The
    # string is compared and never read, so what goes in it is the runner's to decide.
    identity=f"{shogym.make('appworld').config_digest}|deadline=600.0|flight=1",
)
```

Passing `identity` is how a record defends itself. A directory that already names one refuses a
resume that names a different one, and refuses a caller that names none: the record has said what
produced its rows, and rows that decline to say make it unreadable as one run afterwards.

The absence of a line is load-bearing, so two things fail closed on it. A delivery whose line
cannot be written is not delivered: the terminating call is answered with the empty member every
other silence uses, and the stream stops. And a terminal that outran its deadline delivers
nothing even when the env's finalization eventually returns: the watchdog has already sealed that
task into an unscored `timeout` row, which the design counts as a failed delivery to be scored at
the floor and retried rather than as a dose, so the late answer carries the empty member and the
log stays silent about it.

## Requirements

Pin-and-install mechanics are shared; see [`../README.md`](../README.md). What is specific:

- **Docker, and no fallback.** Every world runs in a container, because the code an agent writes
  runs *as* the worker and a worker on the host runs it as the user running the run. A machine
  with no reachable daemon is refused when an `appworld` env is **constructed**, with the reason.
- **The `appworld` extra carries no packages, and cannot.** `appworld` 0.1.3.post1 pins
  `pydantic>=1.9,<2`; shogym's MCP layer needs `pydantic>=2.7`. No environment satisfies both, so
  upstream is not installed beside shogym; it is not installed on the host at all. It lives in an
  image (`worker.Dockerfile`), built on first use and tagged with a digest over the Dockerfile and
  the worker together. The build takes about a minute cold and resolves natively on both
  architectures this port has run on; `SHOGYM_APPWORLD_PLATFORM` forces one if a machine needs it.
  This is provisioned when an env is **constructed**, so `import shogym` stays offline.
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

`submit` seals the episode. `finalize` then stops its container, reads the end state upstream had
already written, collects the base task's own checks, and scores the filing against the drawn key
in the **serving** process, never in the world's. The worker's protocol has no field for the key and no comparison in it, so a world an
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
| `reward` | `ledger_fraction` again, under the name a durable row's summary is read from. |

```python
import shogym
result = shogym.result_from_trace("shogym_logs/run.jsonl", env="appworld", task="0")
print(result.terminated, result.value("ledger_fraction"))
```

`reward` is an alias and not a second number. A `TaskStream` row's `score.reward` is filled from
`reward` or `partial_credit` and from nothing else, so a port whose headline is called anything
else records rows that are scored and summarised as nothing: every shipped `results.py` counted
non-null rewards and reported `scored 0/N` for a complete run of this env. Publishing the headline
under both names fixes the summary without renaming the metric the scorer, this page and the
analysis all use. It changes no wire: `Information` and `Placebo` each reveal the one channel they
are named for, and neither of them is this.

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
- **the digest** (`notice`, under `Placebo`): `sha256(task || check || observed)[:4]` per item.
  Computed on the server, from the identity of the task the agent is working and from the agent's
  own submission. The identity is the env's own material, not something the serve layer showed the
  agent (a dispensed task deliberately carries nothing that names it), and it is the same value in
  both arms of a pair, so it cannot tell one arm from the other. The key is not an input to it at
  all. That is the whole of the claim: the identifier is not secret, and it is printed in the
  header of all three classes, which is a constant of the episode rather than a function of the
  key. The base task's own checks render their observed value as `not determined`, always. A check
  asserts over models the agent touched through nine apps and has no value the agent wrote
  anywhere, and naming its outcome in the one column both classes share would make the inert
  payload move with the base task's result. A test flips every check and asserts no byte of the
  digest moves, and a second renders two tasks from one submission and reads what moves: the
  header cell that prints the task, the digest column, and nothing else.
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
- **The world runs in a container of its own,** for the packaging conflict above, because
  AppWorld freezes the process clock and holds each app's database engine on a class attribute,
  and because the code an agent writes runs as that process. It has one task's tree read-only, its
  own output directory writable, and no network; it talks to its parent over the pipe pair the
  parent made and over nothing else. AppWorld's own environment server publishes `evaluate`,
  `save_state` and `load_state` unauthenticated on every interface; there is no interface here.
- **The base task is graded in a second process that never runs agent code.** The world an
  episode is served from is built with `load_ground_truth=False`, so the answers and the checks
  are not objects in the process an agent's code runs as and there is no evaluator there to call.
  Upstream persists the world at the end of every block, so the seal has nothing to flush: the
  host stops the container and a short-lived grading container reads what is there.
- **The generator's global-RNG state is captured and restored around every episode,** and the seed
  it is started from names the task and nothing else: not the session, not the run, not the
  feedback regime. AppWorld saves databases and not generator state, and every `login` draws from
  the global generator, so a port that replayed only the databases would serve two worlds that
  agreed on their contents and disagreed on their next draw.
- **The derived cache's name says everything that filled it.** Four things: the corpus it was
  derived from, the derivation layout, the generator, and the realized interpreter, which is the
  process that writes a task's seeded database log through upstream's own model layer. "The
  generator" is its constants plus the source bytes of every module reachable from the three that
  generate a world (`env_v1`, `world`, `worker`), walked through the imports rather than kept as a
  list somebody has to remember to extend: the list version missed the helpers that pick a task's
  backlog seed and its world seed, both of which live in `env_v1`. The price of the walk is that
  an edit anywhere in the port, a comment included, derives the corpus again. Each of the four is
  also inside a stamp written into the cache, because a name cannot cover a tree that was edited,
  moved or restored under it. So pointing `APPWORLD_ROOT` at a second corpus, editing how a task
  is derived without touching a ledger constant, or reinstalling the runtime, all derive a fresh
  tree rather than reusing one an older combination seeded.
- **A derived task is reused only when its derivation finished.** A completion marker is the last
  thing written into the staging tree, so it arrives under the task's name in the same rename that
  publishes it: there is no ordering in which a reader sees the marker over a half-built task. It
  says the derivation finished and not that the bytes are still the ones that were written, which
  on a development host is the operator's own filesystem to answer for.
- **Not built:** the yoked payload (a donor's submission and the receipt computed on it), the
  foreign rendering of a verdict vector as prose and the parser that round-trips it, the fixed-size
  envelope that would pad those to one predeclared size, and any assignment, forking or analysis
  machinery. This is the environment and its three payload classes; the experiment that uses them
  is not in this repository.

### What the worker boundary is

The code an agent writes runs **as** the worker process, and until this revision that process was
the run's own user: everything the user could read, agent-authored code could read. The port said
so in those words, because on one uid no filesystem arrangement is a boundary. The worker runs in
a container now, so the boundary is the machine's rather than the arrangement's.

**What one episode's world can see, and it is the whole list:**

| mounted | how |
|---|---|
| the corpus's shared parts (`base_dbs`, `api_docs`, `datasets`, `version.txt`) | read-only |
| **one** task's world, from this episode's own view of the corpus | read-only, mounted by name, so `data/tasks` inside the container holds exactly one directory |
| **one** episode's output tree, at `/outputs` | the only writable mount there is, and bounded before it is graded |

**What it cannot see, and the word is *absent* rather than refused:** the run's provenance
directory, the grader's tree and its private parent, the downloaded corpus, this port's cache, the
repository, every other task's derived tree, every other episode's output tree, and the user's
home. None of them is a file the world may not open. None of them is a path.

**And no network.** The container is started with `--network none`, which is what inverting the
transport bought: the worker used to answer on a loopback port, and a container-loopback listener
cannot be forwarded, a published port is not loopback-only, and `--network none` and `-p` are
mutually exclusive. It speaks length-prefixed JSON frames on the pipe pair its parent made
instead, so there is no port to find and no token to need: a pipe has one writer. A frame that is
not an object carrying an answer is a framing failure like any other, fatal to that worker: valid
JSON of the wrong shape used to reach the caller and come back out as an attribute error. Descriptors 0
and 1 are duplicated and then pointed at `/dev/null` and standard error, so a library that prints
on import cannot corrupt a frame and agent code reading `sys.stdin` cannot eat a command.

The arrangement the boundary replaced is all still there, and is now belt to its braces rather
than the whole defence. The answers are absent from the served tree and are not objects in the
process; nothing in the served tree is a symlink; served inputs are independent copies, so a
write through a served pathname changes neither the corpus later episodes are derived from nor
the baseline the grader diffs against, and each episode is served a view of its own so a write
through one is not in the next one's inputs (under the container there is no write to make: the
served tree is mounted read-only); upstream's evaluator report is off and an episode's output tree
is its own, so no episode holds another's end state, logs or verdicts; the grader's tree has
an unguessable name; and the drawn key reaches neither process, because the protocol has no field
for one.

**Nothing the host needs to know comes from the process that runs the agent's code.** There is no
seal command and no quiesce command: the writer on the protocol's own pipe is reachable from
inside the interpreter that executes agent-authored Python, so a reply saying "I have flushed" or
"I have stopped" is a reply the episode could have written. The host stops the container, asks the
daemon to confirm it, and grades what is on disk.

**What is on disk is the world at the end of the last block, because upstream puts it there.**
`AppWorld.execute` ends with its own save into the episode's output tree and `initialize` writes
one before any block runs. So an episode that ran N blocks is graded on the state after block N,
and one that ran none is graded on its opening state. Work an agent's thread does after its last
block is lost rather than scored, which is the same rule the block budget already states, and the
generator digest is written to the same tree for the same reason: a diagnostic read out of a reply
is a diagnostic the episode can choose.

**A removal the daemon will not confirm ends the episode.** `docker inspect` exiting nonzero is
not the same fact as the container being gone: every daemon outage, unreachable context and
timeout exits nonzero too. Not-found is read from the daemon's own wording, presence is read from
success, and anything else is unknown, which fails closed. Teardown's own removal is the other
contract and never raises, because a teardown that raises abandons the handles it was there to
release; what it could not remove belongs to the sweep.

**Docker's own proxy profile is emptied rather than left alone.** The client adds
`HTTP_PROXY` and its variants to every container it creates, from whatever profile is configured,
and a proxy URL can carry credentials or an internal host name. This port never passed them, which
is how they were missed; they are now passed empty, which is what overrides an injection.

**Two boundaries are bounded because they are the host's, not the container's.** The reader's frame
buffer is a host allocation that the container's memory limit does not reach, so a header that is
not a length and a body larger than any block's output are refused before anything is read, and
either is fatal to that worker. And `/outputs` is a host bind that Docker cannot put a quota on, so
what an episode wrote there is bounded at the boundary instead: bytes, files, depth and the wall
clock the snapshot may take, with a tree past any of them refusing the episode rather than being
copied. An episode can still fill a disk while it runs; bounding that needs a filesystem quota this
port does not have.

**`--cpus` is a ceiling, not a reservation.** It is a CFS bound on how much a container may use,
not a set of processors reserved for it, so two arms on one host share the same cpus and on a
loaded machine each still slows the other. What the quota buys is that neither can take more than
its share, which is the absence of starvation rather than independence. Arms that must not
influence each other at all need disjoint cpusets or separate hosts.

**The cache path is part of the surface, so it has to be arm-neutral.** A world can read the
host-side source of its own mounts, and those are rooted at `SHOGYM_CACHE`. Two arms configured
with cache roots named for their policy or their draw would therefore be able to read their own
treatment off a mount table. Run both arms against one cache root, which is the default, or name
it after nothing.

**A container nobody could remove is written down.** The sweep skips containers whose parent is
alive, which is right for the ordinary case and wrong for a long-lived process that failed to
remove one and has no later chance to try. A removal that could not be confirmed records the name
where the sweep also looks, as an append rather than a rewrite: the sweep used to read the whole
ledger and publish a new one, which dropped a name appended in between.

**A terminal may overtake a block, and finalization waits before it stops anything.** The serve
layer lets a terminal jump the queue on purpose, so a submit can arrive while upstream is inside
the save it ends every block with; removing the container then leaves a tree that is stable and
partial. The accepted call is waited for, bounded, and a world that will not settle is refused
rather than stopped underneath it.

**And the wait is not the only thing standing there, because the reply that ends a call cannot be
authenticated.** Upstream disables its own guard before that save, so a thread an earlier block
left running can put a well-formed completion on the protocol's writer while the real save is in
progress. What the host does about it is not believe it: the snapshot is checked for a whole set
of database logs, none of them cut off mid-write, and an episode whose tree is half a save is
refused. Two things follow: a forged reply earns no block, because the budget is spent when a
request goes out, and moves no grade, because what is graded is the tree and the tree is checked.

**What this port says about an episode is kept beside the episode's tree, never inside it.** The
output tree is mounted writable, so an owner marker in there is cleanup authority its own subject
could rewrite, and a completion record in there is a claim about a save made by the thing that was
saving. Both live in a control directory no mount set names.

**A save is checked against a record written after it finished.** Upstream clears the database
directory and writes the logs one after another, one record per line, so an interruption leaves
every expected filename and a perfect tail with a suffix of state missing: no property of the
bytes tells the two apart. What tells them apart is a length recorded once the save returned,
beside the number of the block the host asked for, and the host compares that number against the
one it sent. A save that never finished leaves the record of the block before it. That record is
written inside the interpreter that runs agent code, so it establishes completeness against an
interruption rather than against an adversary; against an adversary the guarantee is the one that
was always there, which is that an episode already controls its own tree and nothing it writes
there improves its grade.

**A per-episode tree is removed when its owner is gone, never because it is old.** Every one
carries the pid and the birth stamp of the process that made it, and a sweep asks whether that
process is still there, which is the same question and the same evidence the container sweep
asks. An episode can legitimately run for hours; a directory nobody has touched is not evidence
that anybody left.

**The resolver a world reads says nothing about the host.** Docker writes one from the host's or
the daemon's configuration even under `--network none`, naming a nameserver and saying where it
came from. There is nothing to resolve in here, so a fixed file is mounted over it.

**The grader is given a snapshot, not the tree the world wrote.** The grading container mounts the
answers, so a symlink left under the output tree would resolve inside *its* namespace. Every entry
is checked to be a plain file or directory resolving inside the tree, and anything else refuses the
episode rather than being skipped; what is copied is a tree of regular files with no link in it.

**Two more containers, and neither runs a line an agent wrote.** Seeding writes one task's
database log into a staging directory, and grading reads the snapshot: the filing, the databases'
digest, the generator digest and the base task's own evaluator against the answers. Both are short-lived,
both are the same image, and grading is the only place ground truth is loaded at all.

**The transport carries an identifier on every frame.** An ordered pipe is not HTTP: a command
that timed out is still running and its answer still arrives, into the stream the next caller is
reading. An answer whose identifier is not the one a call sent is discarded, and a call that
stopped waiting poisons its worker outright, because a world with a command still running in it
is not a world worth reusing.

**The image.** Digest-pinned base (`python:3.12-slim-bookworm`), `appworld` version-pinned to the
release this port reproduces, the app sources the wheel ships packed unpacked at build time, and
everything byte-compiled so a read-only container is not recompiling on every episode. Tagged with
a digest over the Dockerfile and the worker together, so an edit to either builds a new image
rather than reusing one built under the old text. **The corpus is deliberately not baked in**: it
carries every task's `ground_truth` beside every task's `specs.json`, so an image holding it would
put the answers inside the container that runs the agent's code.

`appworld` is no longer installed on the host at all. There is no host fallback and no flag for
one: a worker on the host runs agent-authored code as the user running the run, which is the
arrangement this exists to end. A machine with no Docker daemon is refused when an env is
**constructed**, with the reason, rather than at the first `execute`.

**There is no file-access audit, and a run must not be read as though there were one.** An earlier
revision installed an audit hook to journal files opened from outside the served tree. It recorded
opens made by the worker process, and in isolation it recorded opens made inside an embedded
IPython shell, but it did not record a file read through a served `execute` call in a live world.
A control that works on some paths and not the one that matters is worse than none. It was removed
rather than caveated, and the container makes the property it was proxying for something to
enforce rather than something to observe.

**What a run is filed under.** The fingerprint carries the draw, the payload class, the block
budget, what the corpus actually holds (every file the world reads, not only the tasks), the image
the world ran in as the daemon has it, the cpu and memory that image was given, and how a score is
read. Both derived caches are named for the corpus they were built from, so a tree warmed under one
`APPWORLD_ROOT` cannot be served under another.

**What it costs.** Measured on darwin/arm64 against the host worker it replaces, same task and
same corpus: per-episode startup 1.87s to 2.32s, one `execute` round trip 1.8ms to 6.1ms on an
empty block and 3.9ms to 11.1ms on one that makes an API call, grading 1.43s to 1.80s, and a whole
served episode 8.72s to 8.99s. The image builds in about half a minute and is 95 MB.

**What the world can learn about the host is names, not contents.** `/proc/self/mountinfo` is
readable from inside and names the host-side source of every bind: the cache root, the private
home and its tag, and this episode's own output directory. None is openable from there, none is
the Docker socket, and none is a key, a pulse or a grade; and none of them differs between two
arms of a pair, so a world cannot read its treatment off its own filesystem. The container's short
id is kept out of the environment by giving every container the same hostname. The claim is about
contents, and the port does not claim its paths are secret.

**A container whose parent died is swept, not hoped about.** Every container carries the pid that
started it and this machine's boot, and construction removes the labelled ones whose parent is
gone. The case is a run that dies while a world is wedged in a command: the worker learns its
parent has gone only from end-of-file on its next read, which it never reaches, so it never exits
and `--rm` never fires.

**The residual, stated exactly.** The daemon is trusted: a container is a boundary against the
code inside it and not against whoever can talk to Docker, and this port's own parent process can
do both. Episodes are isolated from each other and from the run by the mount set, not by separate
kernels. The port's own source is readable to anyone with the repository, and with it the draw's
algorithm; what has changed is that the world's process is not one of them. And the run's
provenance directory still retains true reports under `Placebo`, which is the runner's business
rather than the env's, but it is no longer readable from inside an episode.

## Gotchas

- **`env.num_tasks` is 318, not 417.** Task indices address the manifest, not the split.
- **Construction is slow, online, and blocking.** Building the runtime, fetching the corpus and
  copying it into the two derived views is a one-time cost; deriving a task's seeded world costs
  about a second the first time it is served and nothing after that. `TaskStream` and
  `ServedEpisode.start` build each task's env off the event loop when they are told the factory
  is safe there (`off_loop_factory=True`, which this env is), so a served queue is not held by
  per-task construction. **The first construction is still the caller's.** `TaskStream.__init__`
  is synchronous and builds one env per name for the published manifest, which for this env is
  the cold provisioning call, so a caller on a loop it is also serving on must build the stream
  off that loop: `await asyncio.to_thread(TaskStream, ...)`. The same goes for a caller that
  calls `make()` itself. Set `APPWORLD_ROOT` to skip the download.
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
| `worker.py` | one world, in a container of its own, answering frames on stdin |
| `container.py` | the image, the mount set, and the flags a world is run under |
| `worker.Dockerfile` | the image: the pinned interpreter and release, and not the corpus |
| `adapter.py` | the pins, the provisioning, the served roster, the worker client |
| `task_manifest.txt` | the 318 served tasks, settled before any episode |
| `pass_counts.txt` | the roster's own distribution of passing-request counts |
