# Changelog

What changed between shogym releases, newest first. Each heading is a git tag; the tags and
their release notes are on [the releases page](https://github.com/shojin-lab/shogym/releases).

Versions before 1.0 keep the usual pre-1.0 promise, which is a modest one: a minor bump can
change behavior a consumer depends on, and this file exists to say which behavior and in which
direction.

## Unreleased

### `serve`: the version one serving contract is retired, and protocol v2 is the only path

`shogym.serve.stream` is gone, and with it `TaskStream`, `EvalStream`, `TaskRef`,
`build_stream_server`, the `get_task` and `queue_info` tools, feedback riding a terminal
response, and the feedback policies `Never`, `Immediate`, `Information` and `Placebo`. Every
consumer of any of those has to move. **This removes a public API.**

`shogym serve` now serves one env at one task as a durable protocol v2 stream, and the
`--protocol` flag is gone because there is nothing left to select. The model asks for its work
with `pull`, which takes no arguments and answers with exactly one JSON record (`task`,
`payload`, `wait` or `done`); every environment tool is wrapped as
`{"attempt_id": ..., "arguments": {...}}`; and the env's terminal call is intercepted into one
seal transaction that answers with `seal_ack`. Serving needs the `durable` extra now.
`import shogym` still needs nothing, and `shogym.evaluate` in process is unchanged.

Three things the retired path did are not in the protocol that replaced it. It served a queue of
tasks over one endpoint, and a v2 generation serves one episode. It reported a score offline, and
v2 keeps the score in the stream's own history where no reader reaches it. It served two matched
feedback arms, and the payload families that would replace them belong to the experiment layer
rather than to this package.

Run directories the retired path wrote stay readable. `shogym.serve.v1_runs` reads them, with the
same `read_dispenses`, `read_results` and `reconcile` and the same row shape, and it does nothing
else: there is no writer, and a protocol v2 resume refuses such a directory before it claims
anything.

`ServedEpisode` gains `ends_on_horizon`. It defaults to True, which is what a caller driving an
episode directly has always had: spending the env's step budget seals the episode and grades it.
A caller that ends the episode itself passes False, and the durable stream does, because an
episode that sealed and graded itself when a budget ran out would end an attempt the stream still
held open and record nothing about it where the stream could read it.

### `appworld`: withdrawn

The `appworld` environment is gone, and with it the port, the loopback worker each world ran in,
the empty `appworld` extra and the port's tests. `shogym.registered_envs()` no longer lists it and
`shogym.make("appworld")` raises the unknown-environment error.

The port never paid for its upkeep. It provisioned an interpreter of its own because upstream
cannot be installed beside shogym, downloaded a pinned corpus, and ran every world in a
subprocess, which made it the slowest and least reliable part of the suite. No planned experiment
in this repository uses it, so it is withdrawn rather than carried through a deprecation cycle. An
inverse of the removal commit restores the tracked tree; local caches are outside version control.

### `automationbench`: a task that needs a spreadsheet can list the spreadsheets

A task allowed to read Google Sheets is now also allowed to read Google Drive. **This changes
measured automationbench scores.**

Google Sheets publishes no list endpoint: all four of its read routes take the spreadsheet id as a
path segment. The one endpoint anywhere in the simulated world that enumerates spreadsheets is
Google Drive's file listing, which returns each spreadsheet as a Drive file. Which services a task
subscribes to is decided per task from what it seeds, asserts on, and grants a tool for, and a task
that hands the agent a spreadsheet without mentioning Drive in any of those three places locked the
only door to its own data: the Drive call answered 401, and the id appeared in no other response.

An enumeration of the shipped 600-task pool crawled every read route each task's world subscribes
to and found 106 tasks requiring at least one id no served endpoint could return. 105 of them are
this: a spreadsheet whose opaque author id (`ss_payroll_5104`, `ss_trext`) the request text never
names. 99 of the 100 hr tasks are in the class, against 0 of sales, operations and finance, because
the hr family was written to a template that grants Sheets tools and never a Drive one. What those
tasks measured was id-guessing luck. In one archived 200-task run they took 36 calls on average
against 23 for the rest, and 41% of their GET calls came back not-found, nearly all of them
consecutive guesses at an id.

Every seeded spreadsheet in the pool now comes back from Drive's listing, and the spreadsheet
metadata endpoint the id opens names the worksheets inside it, so the whole chain a task depends on
is reachable. Expect scores to rise on those 105 tasks and to be unchanged elsewhere; a run served
before this change and one served after are not comparable on them.
`shogym.envs.automationbench.undiscoverable.PREVIOUSLY_UNDISCOVERABLE` records the 106 pool indices
so an earlier run can be stratified rather than read as uniform agent weakness.

`adapter.allowed_services_for_task` is the service set a task is served under, and `build_world`
uses it. `adapter.compute_allowed_services` still answers exactly what upstream's runner answers.

### `automationbench`: the rubric scores the live world, not a rebuilt copy of it

Scoring used to serialize the session's `WorldState` and re-validate it back into a model before
running the rubric. It now runs the rubric against the live object the served tools mutated, which
is what upstream's own runner does. **This changes measured automationbench scores.**

The round trip was not a no-op in either direction. It could raise, which left the episode
unscored: the tools mutate the model in place and pydantic validates on construction rather than
on assignment, so a live world legitimately holds values re-validation rejects. Two shapes of that
occur in the shipped pool. A `linkedin` company record's size field validates under one name and
serializes under another, and the containing model forbids unknown keys, so **any** world holding
a company could never be rebuilt. Six of the 600 public tasks seed one, and they were
unscoreable for every agent on every run regardless of what the agent did. Separately, several endpoints
assign a request value straight into a field narrower than `str`, so an accepted, echoed-back tool
call could leave the world unrebuildable; 273 of 600 tasks reach at least one such endpoint.

It could also lose evidence without raising, which is the part that silently moved numbers.
Whether a spreadsheet row was *written to* is recorded by the tool layer outside the model's
declared fields, so it did not survive serialization. 24 of the 600 tasks carry an assertion
scored purely off that record. On those, a `google_sheets_row_not_updated` guard the agent had
broken read as intact and was credited, and a `google_sheets_row_updated` the agent had earned was
denied.

Measured by replaying 492 archived episodes against both scoring paths: 476 score identically, 9
were previously unscoreable and now score, and **7 change, all downward**, all of them the guard
case above. Expect a small downward correction concentrated on tasks with those assertions, and
no change at all elsewhere.

`adapter.score_state` now takes the live `WorldState`. It still accepts a mapping, so callers
written against the older signature keep working, but that path re-validates and is therefore
lossy in both ways above; its first parameter is renamed `world_dump` -> `world`, which matters
only to a caller passing it by keyword.

### `automationbench`: a fail-closed finalize publishes no score

An episode whose terminal transaction failed closed published `reward` / `partial_credit` /
`success` defaulted to zero beside its `finalize_error` flag. The row's own `score` was already
`None` for that closure, so the defaults only contradicted it, and anything reading the feedback
by name, including an agent under an immediate-feedback regime, could not tell the fabricated zero
from a scored one. Those names are now omitted entirely; the flag is published alone. Row `score`
is unchanged, since a failed transaction was never a scored closure.

### `serve`: a fail-closed row names the failure

The row's diagnostic read "the terminal transaction failed closed; the env published no verdict"
for every cause alike, so acting on one meant reproducing a failure the harness had already
caught. It now appends the failure's type and, for a failure that reports structured errors, how
many there were and which kinds: `(ValidationError: 17 errors; extra_forbidden)`. Both fail-closed
boundaries carry it, the evaluator's and the verifier's.

Everything published there is a count or a term from the validator's own fixed vocabulary.
Nothing that could have come from the data being validated is included, because for an env whose
state is what is being graded that data can be the answer. That rules out the message, which
renders the offending values, and it also rules out the field locations: a reported location
descends into the input rather than the schema, so a failure inside a mapping contributes that
mapping's keys and a rejected unknown key contributes the key itself. The full diagnostic,
locations and values included, is still written to the private durable record. The summary travels
on a harness-side channel, so the agent-facing terminal payload and the public trace are
unchanged.

## 0.1.0

### `hle`: the default judge is now `gpt-5.6-luna`

It was `gpt-5.4-nano`. **This changes measured HLE accuracy for anyone who upgrades without
pinning a judge.** The direction, stated no more strongly than it was measured: the two judges
disagreed on 19 of 367 probe cases, 18 of those disagreements had a knowable right answer, and
luna was right in all 18, every time because nano had returned a false negative. That predicts an
upward and uneven shift rather than a constant offset. Predicted, not promised: the probe's
candidate answers were constructed to exercise the judge rather than sampled from a consumer's
run, 18 is a small number of cases, and a workload whose items differ from the probe's can move by
a different amount, or in a way this measurement does not cover. The part that is not in doubt is
that the number moves: an unpinned run is graded by a different model than it was on 0.0.1.

The measurement is in [#122](https://github.com/shojin-lab/shogym/issues/122): 1,746 real calls
through this repo's own judge path, 873 per model, real prompt, real client, real
`parse_judge_response`.

| | `gpt-5.6-luna` | `gpt-5.4-nano` |
|---|---|---|
| correct verdicts vs constructed expectation | 307 / 307 | 289 / 307 |
| unparseable verdicts | 0 / 873 | 0 / 873 |
| verdict flips across repeat draws | 2 / 307 | 16 / 307 |
| median latency at concurrency 8 | 1.09s | 1.10s |

Every nano error was the same mode, multiple-choice matching. Gold `F` against a candidate
reading "The answer is f." was judged incorrect on the letter case. Gold `M`, whose option text
is "15", against a candidate reading "15" was judged incorrect on letter versus option text. Nano
was also unstable across repeat draws on roughly 10% of the probe's multiple-choice items, which
is the reason to expect an uneven shift rather than a fixed one: how much a run moves depends on
how many items of that shape it happens to contain.

Price and latency are a wash. Input costs the same per token, output is 4% cheaper, and median
latency matched at concurrency 8, so nothing here is a cost or speed change.

Numbers taken before and after this release are not comparable unless they say which judge
produced them. Pinning shogym by commit is unaffected until the pin moves, and the old behavior
is one config key away:

```python
shogym.make("hle", config={"judge_model": "gpt-5.4-nano"})
```

### `hle`: the judge's request is configurable, and the score names what graded it

`judge_kwargs` threads from the env config into the judge's chat-completions call, so a grading
setting is reachable at all:

```python
shogym.make("hle", config={"judge_kwargs": {"reasoning_effort": "low"}})
```

It takes sampling settings and nothing else. The whole allowlist is `reasoning_effort`,
`temperature`, `top_p`, `seed`, `frequency_penalty`, `presence_penalty`; every other name is
refused at construction, with an error that says what is allowed. Default-deny is the
load-bearing part, because the dangerous fields fail silently at grading time: a reply the parser
cannot read is recorded as a genuine wrong answer with no `judge_error`, so a judge that returns
nothing parseable quietly depresses the benchmark instead of announcing itself. Unset means
absent, so a judge configured with no kwargs makes exactly the request it made before this
parameter existed, which keeps `judge_base_url` working against servers that reject fields they
do not implement.

A model-graded episode now emits `judge_model`, plus `judge_effort` when one was configured. The
id published is the model that answered rather than the one that was asked for, since an alias, a
router, or a `judge_base_url` endpoint can reply as something else; the configured id stands in
only when the judge failed before there was a response to read. Two cases stay silent rather than
guess: the exact-match fast path, where no model read the answer, and an injected `judge=`, which
only the caller can describe.

### `orca_bench`: withdrawn from this release

The port's phase 1 is the offline half, and its compose backend, the part that makes an episode
runnable, is phase 2 and still open as
[#117](https://github.com/shojin-lab/shogym/issues/117). A registered env that cannot run an
episode is a broken promise in a versioned release, so `orca_bench` is not registered, not
advertised, and has no extra in 0.1.0. Phase 1 returns verbatim as the first PR of the phase 2
stack.

### `serve`: the stream says what a pull costs, and a failed seal always lands a row

- `get_task`'s advertised description is now built from the endpoint's own `max_in_flight`
  rather than being one fixed docstring. It names the number of slots the agent may hold, says a
  pull below the limit displaces nothing, and says a pull at the limit forfeits the oldest live
  task, which the stream seals and scores as a loss. `queue_info` reports the same limit. The old
  text promised that a dispensed task was "still yours to finish", which is false of every pull
  an agent can make at `max_in_flight = 1`.
- A seal that fails now lands the truest row it can compose and hands the claim back, so a retry
  retries the durable append rather than composing a second row. Before this, a seal that failed
  before composing anything left the retry to compose afresh over an episode the first attempt
  had already force-terminated, which landed the task under a scored closure the agent never
  earned.

## 0.0.1

The first public release: eleven registered environments and five harness quickstarts. Its notes
were written into the GitHub release rather than into the tree, so they are not restated here.
