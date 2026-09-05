# Changelog

What changed between shogym releases, newest first. Each heading is a git tag; the tags and
their release notes are on [the releases page](https://github.com/shojin-lab/shogym/releases).

Versions before 1.0 keep the usual pre-1.0 promise, which is a modest one: a minor bump can
change behavior a consumer depends on, and this file exists to say which behavior and in which
direction.

## Unreleased

### `serve`: a generation may tell the agent how much work is left

`stream_start` takes `info`, and a generation that declares it serves an `info` tool beside `pull`.
The tool takes no arguments. Its answer is a wire record of a new kind, `info`, carrying
`consumed`, how many attempts have been handed out so far; `in_flight`, how many of those have not
ended yet; and `remaining`, how many are still to be handed out.

The three are not a partition. `in_flight` is inside `consumed` rather than beside it, and neither
counts an attempt the generation ended before ever handing it out, which a controller finalizing a
planned attempt and the cascade that floors one whose gate can no longer open each produce. With
the roster `N` and those never-handed-out endings `U`, the invariants are `remaining + consumed +
U == N` and `in_flight <= consumed`. So `remaining` plus `consumed` is the roster wherever `U` is
zero and short of it by `U` everywhere else, while the three together sum to `N - U + in_flight`,
which can fall below the roster, meet it, or pass it, so it is not in general the roster.
`remaining` reaching zero does not mean the generation is done either: a payload can still be owed
and an attempt can still be live. The kernel defines the three once, where it computes them, and
every other statement of them says what that one says.

The answer is the stream's rather than the transport's. The counts are read inside the same
transition that mints the record, the record is reserved for the request that asked for it so a
retry gets the same bytes rather than a new reading, and it is delivered under the same attestation
a pull's answer is, so what the agent was told is among the presentations a run's own records list.
The call enters the transport's admission with every other one, so it is answered while the
generation is open and nothing is owed: it cannot overtake an owed pull or acknowledgement, a pull
cannot overtake it, it is refused while a filing prepared by a replaced owner is still to be
finished, and it is refused with `closed_stream` once the generation is done.

Where the tool is declared it is named in the served manifest, with the exact words and the closed
empty schema it is registered under, and the declaration is folded into the configuration hash a
resume is held to. **A generation composed without it serves what it always served: the same tool
set, the same control tool description, the same bytes and the same configuration hash**, so every
recorded history replays and every open stream resumes. No quickstart declares one.

`info` joins `pull` as a name an environment served under protocol v2 may not give a tool of its
own, whatever the generation declares. An environment that advertises a tool called `info` is now
refused at construction rather than served.

### `serve`: a generation may serve several tasks at once

`stream_start` takes `capacity`, how many tasks the generation lets the agent hold at once, and it
defaults to one. It is held to the rule a count is held to, an exact integer of at least one, where
the generation is composed and again where the kernel starts it. The gateway now holds one world
per live attempt and runs every environment call in the world of the attempt that call names, so an
agent holding several tasks works each of them where it started. A call naming an attempt this
transport holds no world for is refused with `invalid_attempt` before the stream is asked for a
step, rather than run in whichever world was opened last, and a terminal filing for such an attempt
is refused the same way rather than sent: filing one would end that attempt on an environment that
read no world.

A world is let go of when its attempt's seal is acknowledged, when the generation ends the attempt
itself on a deadline or a step cap, or when the transport stops, and the pairing of attempt to
world is dropped as the world closes. The call that finds an attempt out of budget closes that
attempt's world before it answers. So does a call whose attempt's deadline fell due while it was
inside that attempt's world, which is where a generation makes the ending it had to hold back, and
so does the call that comes back for an observation a lost answer or a failed graded-horizon filing
left owed: what the generation ended while a result was outstanding is read out of the same
question that result is handed over through, and a call that failed in the world is held to the
same rule as one that came back with an observation. Each of those calls makes one read after its
release, so an ending the generation has not applied by the time of that read is retired by the
next call or by the stop, which is what happens to every ending made while no call is in a world.
Retiring ended worlds and stopping both try every world they were asked to close even when one of
them refuses, keep the ones that failed, raise all of the failures together, and can be asked again
to finish them.

`StreamGateway` takes `world_attempt`. A transport replacing another mid attempt is handed the
world that attempt is working in, and this is where it says which attempt that is: the pairing goes
into the route an environment's own terminal resolves as well as into this transport's own map, so
the world an ordinary call reaches and the world a seal captures are the same world. A handed world
belongs to that attempt alone, no other attempt's call or filing can reach it, and no task starting
here inherits it. A world stops being routed when the owner that recorded it lets it go, so a
predecessor clearing up after the world it was holding does not unpair the one a replacement
restored. Without the name the episode is the seed of a generation this transport is starting, and
only the first task presented here may claim it.

The `pull` tool's description says what a capacity above one allows: how many tasks may be held,
that while that many are held a pull cannot return another task and answers a wait when nothing
else is ready, and that the agent ends a task by calling its terminal with that task's
`attempt_id`.

**A generation composed without a capacity serves what it always served: the same description, the
same configuration hash, one task at a time, and the same calls in the same order**, so every
recorded history replays and every open stream resumes. No quickstart declares one.

### `serve`: a generation may hand the agent its step budget

`stream_start` takes `budget`, and a generation that declares one serves `budget` on every task
record it offers: the number of ordinary environment actions the attempt may take, which is the
step cap this transport enforces. The declaration must be a number a task record could carry and it
is refused unless it equals the environment's own horizon, where a generation is composed, where
`open_gateway` opens a composed one over an episode, and where a `StreamGateway` is handed the
composition it is serving, so a transport that took a generation over cannot enforce one number
while its records carry another. Where a budget is declared, one sentence about it is appended to
the `pull` tool's description.

`budget` is the one key on the v2 wire a record may leave out. **A generation that declares no
budget serves the task record it served before, byte for byte, and hashes to the same
configuration**, so every recorded history replays and every open stream resumes. No quickstart
declares one.

### `serve`: a run says what it committed to deliver

A generation now keeps a record of every message it committed to deliver: the kind, the attempt
it belonged to, its place in the order, and the digest of the exact bytes. `presented_messages`
answers with those, `generation_records` answers with them and the attempt rows from one handler
call, and `read_records` returns them on `RunRecords.presentations`. A row is what the generation
accepted and not what any transport handed over, so a harness that keeps the model's transcript
reconciles the two to establish either delivery or consumption. `records.jsonl` is unchanged.

`open_gateway` and `StreamGateway` take an optional `on_refusal` sink, called with the new count
inside the call that issues a refusal. A refusal advances no protocol state, so the count is a
harness-side audit surface and this is the only moment a transport that is killed rather than
stopped is certain to reach.

### `serve`: an episode's finalization records go in the run directory

`ServedEpisode.start` and `ServedEpisode.open_env` take `run_directory`, and it decides where the
episode's durable finalization records live. The precedence is the run directory first
(`<run_dir>/finalizations`), then the directory holding the trace, then the unchanged shared
fallback under `~/.cache/shogym/sessions`. Every root is still shared across the sessions of
whatever it scopes, so the startup pass that resolves a record a crashed session left mid-finalize
scans the store this precedence selected. **A caller that passes a run directory and a trace path
gets its records in the run rather than beside the trace.**

`shogym serve --run-dir` now designates that store as well as the stream's blobs, manifest and
history, because it passes the directory to the episode it serves. Protocol v2 itself releases
every world with `close(finalize=False)`, so a v2 run currently produces no episode finalization
record of its own; what the run directory gains is where such a record goes and where recovery
looks for one, not a new record.

The Prime Agent example, which is `run_stdio_v2` with the transport swapped, is corrected to match
it: it passes the run directory to its episode, and it releases the world through `gateway.aclose()`
rather than closing the episode with the default `finalize=True`, which claimed an abort verdict for
an attempt the generation had already answered for.

### `serve`: a row whose seal could not go on says what stopped it

An accepted terminal runs the seal and the grade, and the renderer as well when the attempt
carries a payload obligation, and either way its batch can end the attempt wrote one word on the
row and nothing else: `seal_failed` for an Activity that failed for good, and `seal_unusable` or
`seal_renderer_mismatch` for one that answered with a result the seal would not commit, each with
a score of zero. A grader that was down, a grader that refused the task, a
Worker with no renderer for the declared policy and a score returned for another seal all read
alike, so the only way to explain a hole in a run was to open the raw durable history.

`AttemptRecord` now carries five more fields, and `records.jsonl` carries them with it:
`failure_activity` and `failure_activity_id` name the step the service gave up on, `failure_kind`
is the semantic type it failed as (the environment's own type for a deliberate refusal, the
exception's class for an ordinary failure, and which timeout for one that overran),
`failure_message` is what it said, flattened to one line and cut to 512 bytes with the cut marked
inside the value, and `failure_retry_state` is why the retries stopped. All five are `None` on
every row whose seal did not end it, and all five are read out of what the history already
recorded, so a replay produces the same words and no run writes anything new.

A result the seal could not vouch for fills two of the five. Its Activity succeeded, so no step
is named and there is no retry state to report: `failure_kind` and `failure_message` are the
semantic type of the check that refused the answer and what it found, for example
`UnusableActivityResult`, and `IncompleteCandidateBundle` or `RendererDescriptorMismatch` for the
checks that have their own.

The twelve-column table `shogym results` prints is unchanged, and so is what a model is handed
when its filing fails: a message an environment raised with can name what it was grading, so this
detail travels on the harness-only Queries, `attempt_records` and the `generation_records` the
reader uses, and in the run's own records and nowhere else.

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

### `automationbench`: the pin moves to upstream 1.0.6

`UPSTREAM_SHA` is now `6d21054`, the commit upstream's changelog calls the 1.0.6 release, in place
of `a321764`. **This changes measured automationbench scores and every configuration digest**, so a
run served at one pin and a run served at the other are not comparable, and a resumed generation
across the move is refused rather than scored against a rule nobody drew for it.

The release is titled "Bug fixes" and is mostly a fairness pass over the tasks. 110 files of the
package changed, 5243 lines added and 1055 removed. What a served episode sees:

- **Google Drive is seeded beside Google Sheets on 105 tasks**, 99 in hr and 6 in marketing. Sheets
  publishes no list endpoint, all four of its read routes take the spreadsheet id as a path
  segment, and Drive's file listing is the only enumeration of spreadsheets the world has. Those
  tasks handed the agent a spreadsheet whose opaque author id the request text never named and
  answered the Drive call with a 401, so the id could only be guessed. Every seeded spreadsheet in
  the pool is now listable, and the metadata endpoint the id opens names the worksheets inside it.
- **96 tasks have other initial-state changes after ignoring `google_drive`.** 20 of those overlap
  the 105 Drive rows, so **181 unique tasks have any seed change at all**.
- **68 tasks changed their assertions**, and 3 changed their granted tool list.
- **105 user requests were reworded**, and the shared system prompt gained an exception to its
  "do not narrate exclusions" instruction for workflows that require an exclusion notice.
- **15 rubric assertion modules changed, adding 18 registered assertion handlers**, so what a given
  end state scores can differ even where the task did not change. No assertion module was added or
  removed.
- **Every schema model now validates on assignment**, so a write outside a field's enum is refused
  at the endpoint with a 422 instead of landing in the world. It does not reach an in-place
  mutation of a container a field already holds, so this does not make a served world safe to
  rebuild. See the amendment to the entry below.
- The 600 rows and their `example_id` values are unchanged, and so is the count per domain.

The task name moved off the dataset row into `info["task_name"]`. **Every 1.0.6 dataset omits the
top-level `task` key**, the combined `public` alias included, and every shipped row carries the new
one. `env_v1.task_name` reads the new location, falling back to the old column and then to a
default; the fallbacks are for legacy rows and for the rows a caller injects through the `tasks`
config, which is a supported path that predates the move. The per-task `name` and the configuration
digest's per-row `task` both go through it.

An enumeration of the pool at the old pin found 106 tasks requiring at least one resource id no
served endpoint could return. `shogym.envs.automationbench.undiscoverable.PREVIOUSLY_UNDISCOVERABLE`
records those indices so runs made at that pin can be stratified by the class rather than read as
uniform agent weakness. It describes the old pin only and nothing in the env reads it.

### `automationbench`: the Jira project search answers from the world

The search now also reports the Jira projects the task seeded. **This changes a served response, so
what an agent does on the one pool task that seeds a project is not comparable across the change,
and no task with Jira connected is comparable across the pin move that makes the route reachable at
all.**

`JiraState.projects` is a declared field that nothing reads. The project search is the only Jira
lookup on the served surface, and upstream's handler answers out of the recorded project *actions*,
so a project seeded in that collection was invisible to lookup. On the one task in the shipped pool
that seeds one, the search answered `{"values": [], "total": 0, "isLast": true}` for a world holding
the project key `SUP` under the name "Support Issues", and the key appears in no email, no message
and not in the request text. An agent could not learn it, so a request naming `SUP` was one it
brought rather than one it found.

That task is still completable without the key, because the create route accepts an omitted project
and `find_actions` treats a filter the record does not carry as a match, so the assertions naming
the project pass on issues filed with no project at all. What was missing was the lookup, not the
route to a passing score.

`adapter.api_fetch` completes that one answer on the way out. A project matches the way Jira's own
search matches, on a case-insensitive literal in its key or name, the literal being the request's
`query` and an empty one matching everything; a project the route already returned is left alone,
identified case-insensitively too so one key is never reported twice in two spellings, and a response
that is not a search result, a 401 from the service gate for instance, is passed through untouched.
The completion identifies the route the way the router identifies it, by an Atlassian host and the
whole parsed and percent-decoded path, or by the internal path when the URL carries no host at all.
A path that merely ends in those segments belongs to whichever service the router handed it to.
Every other response is still the upstream router's own. The seven pool tasks that record their
projects in the action log get the same bytes as before on that route, because the handler already
found those.

The reachable lookup surface is wider than that one task, though. `api_search` is ungated, so every
task can discover the project search, and 28 of the 600 pool tasks have Jira connected. At the pin
this entry was written against, the URL that search advertised for the route repeated `/rest/api/3`
and reached no handler, so the discover-then-call sequence the served instructions describe answered
404 on all 28. Upstream corrected the advertised URL in its 1.0.6 release, and the entry above moves
the pin there, so the eight tasks that hold Jira projects now answer with them: the seeded one by
way of the completion above, and the other twenty answer an empty search rather than a 404.

### `automationbench`: the rubric scores the live world, not a rebuilt copy of it

Scoring used to serialize the session's `WorldState` and re-validate it back into a model before
running the rubric. It now runs the rubric against the live object the served tools mutated, which
is what upstream's own runner does. **This changes measured automationbench scores.**

The round trip was not a no-op in either direction. It could raise, which left the episode
unscored: the tools mutate the model in place, and at the pin this landed on, pydantic validated on
construction rather than on assignment, so a live world legitimately held values re-validation
rejects. Two shapes of that occurred in the shipped pool. A `linkedin` company record's size field
validates under one name and serializes under another, and the containing model forbids unknown
keys, so **any** world holding a company could never be rebuilt. Six of the 600 public tasks seed
one, and they were
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
lossy; its first parameter is renamed `world_dump` -> `world`, which matters only to a caller
passing it by keyword.

**Amended for the pin move above.** Everything to here describes the old pin `a321764`, which is
what it was written against, and the replay numbers are not restated for the new one. Upstream
1.0.6 closes the two rebuild failures named above, and only those: it lets LinkedIn's aliased
fields populate by name, so all 600 seeded worlds survive a dump and a rebuild, and it sets
`validate_assignment=True` on every schema model, so a write outside a field's enum is refused at
the endpoint with a 422 rather than landing in a world that cannot be rebuilt.

That is not every way a tool can reach one. Validating on assignment does not see an in-place
mutation of the container an attribute already holds, and the tools reach several of those: the
Gmail label endpoint appends each requested label straight onto a `list[str]`, so a non-string
label is accepted, echoed back, and leaves a live world its own dump cannot revalidate. So the
mapping compatibility path above is still lossy in both directions, and both are why scoring reads
the live object rather than a copy.

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
