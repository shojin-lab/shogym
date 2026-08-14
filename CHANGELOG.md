# Changelog

What changed between shogym releases, newest first. Each heading is a git tag; the tags and
their release notes are on [the releases page](https://github.com/shojin-lab/shogym/releases).

Versions before 1.0 keep the usual pre-1.0 promise, which is a modest one: a minor bump can
change behavior a consumer depends on, and this file exists to say which behavior and in which
direction.

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
