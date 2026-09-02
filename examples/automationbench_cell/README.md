# The AutomationBench cell

One Claude Code session works a roster of AutomationBench tasks, served as a single protocol v2
generation. It is a rerun of a cell that has already been recorded: same benchmark, same tasks in
the same order, same model at the same effort, same launch flags, and the same regime, so that the
serving contract is the only thing that differs between the two.

What differs is the whole of what this cell is for. The agent asks for work with `pull` where the
earlier cell's agent asked with `get_task`, every call names the attempt it belongs to, the tool
that ends a task is intercepted and sealed by the stream rather than by the harness, and the score
reaches the agent as a payload the stream released against the attempt rather than as a field on
the result of the call that ended it. The question is whether any of that changes how the agent
behaves.

One thing that is not a difference is what becomes of a task the agent never finishes. That
harness graded a run which spent its whole step budget on the partial state it left, and this one
does too: the env declares its horizon a graded ending, so the stream files the terminal for the
attempt as its last call commits and the score reaches the agent as the payload after it. The
table names the horizon where it filed, which keeps a task the agent ended and a task that ran out
of calls two numbers rather than one.

A fifth difference comes with the first, and it is a loss rather than a change. That harness
answered `get_task` with the number of tasks still queued and with the step budget for the one it
was handing over, and it offered a `queue_info` tool reporting the same counts on demand. The
recorded agent used them: its own progress summaries quote how many tasks were left. A `pull`
carries neither number and this protocol has no queue to inspect, so the agent here works the
roster without knowing where in it it is.

## The command

```bash
# the whole roster, the regime the earlier cell ran, the model and effort it ran
uv run python -m examples.automationbench_cell.cell run --tasks cell-one

# a shorter rerun: the same tasks in the same order, fewer of them
uv run python -m examples.automationbench_cell.cell run --tasks cell-one:20

# what it scored
uv run python -m examples.automationbench_cell.cell table <run directory>
```

`--tasks` also takes a plain list or range (`0-19`, `4,0,2`) for a roster of your own. `--model`
and `--effort` default to the earlier cell's `claude-opus-5` and `xhigh`, and both are overridable
for a smoke run, which is the same override that harness took for the same reason. `--runs` moves
the run directories.

## What the launch is pinned to

`claude` must be on `PATH` with a credential in the environment: `CLAUDE_CODE_OAUTH_TOKEN`, or an
`ANTHROPIC_API_KEY`. The agent gets a working directory of its own holding the one-line
`CLAUDE.md` the earlier cell's agent found in its own, and a fresh Claude Code home beside it.

The command line is the visible half of a launch, and the rest is pinned too, because a difference
in any of it is a difference in the agent rather than in the serving.

- The CLI build is resolved before anything is spawned and refused when it is not the `2.1.220`
  the recorded run reported. `--allow-cli-drift` runs anyway and has the difference recorded.
- The environment is built from an allowlist rather than inherited, so an operator's shell reaches
  neither the agent nor the server it spawns. Two variables this repo reads are why that is more
  than tidiness: `AUTOMATIONBENCH_SRC` and `SHOGYM_CACHE` would replace the task source under a
  run whose record still described the standard cell.
- The tool surface exists only once the agent has started, so it is read off the transcript's
  first line afterwards and compared with the surface the recorded run reported: the build, the
  tools, the subagents and the skills. What differs is printed and kept.

`run.json` holds the whole effective launch: the argv, the working directory, the environment as
the process was handed it with the credential named rather than copied, a digest of every
directory the agent started from, the build that ran, and the first line that build wrote.

The working directory's path cannot be pinned at all. That cell's agent worked in `/work` inside a
container and this one works under the run directory, so the path it sees is not the path that one
saw.

## The regime is pinned, and why

The earlier cell told the agent its score the moment each task ended: the score, the success flag
and the environment's own numbers were attached to the result of the tool call that ended the
task, on every task, by a code path with no branch that could withhold one. Its records show that
happening for every sealed task in the run.

So this cell defaults to `--schedule immediate`, which is this protocol's equivalent: the honest
payload released at the seal, carrying the score the seal committed and the numbers the
environment published beside it. **The comparison is only worth reading while the two regimes
match.** A rerun served under `--schedule never` would differ from the cell it is being compared
with in what the agent was told as well as in how it was served, and no result from it could say
which of the two the difference came from.

`never` is selectable because a run that deliberately measures the other regime is a run somebody
means to do. It creates no payload obligation anywhere, and the records say so: every row reads as
a position this generation was never going to deliver against, which is a different fact from a
payload that was owed and never arrived.

Both are ordinary runs under the platform's own policy. Concealing a score that was released is an
experiment arm a run registers, and this cell registers none.

## What a run leaves behind

```
cell-<schedule>-<stamp>-<token>/
  grades/        the durable history, the sealed grades, the blobs, cell.json and refusals.json
  self/          the directory the agent worked in, seeded with CLAUDE.md
  home/          the Claude Code home it was given
  cfg/.mcp.json  the config Claude Code spawned the server from
  stream.jsonl   the agent's own transcript
  run.json       the roster and schedule, and the launch as it resolved
```

`grades/cell.json` is the join. Nothing on the wire names a task and the history is keyed by
attempt, so the only party that knows which benchmark task an attempt was is the one that composed
the roster, and it writes that down once. `table` reads all three: the rows the history answers
with, then a row per roster position carrying the score, how the attempt ended, what was delivered
against it, what the agent's own transcript shows arriving, and how many tool calls went into it.

What was committed and what was received are two of those columns rather than one. The history
commits to the exact bytes of every message, which is what a generation can attest and is not
proof that anything crossed the transport, let alone that a model read it. Whether the bytes
arrived is written in the harness transcript instead. So `payload` reports what was committed for
the attempt and `seen` reports what the transcript holds.

That second column is a comparison of bytes. The history answers with every message it committed,
in order, and with the digest of what each one was; the transcript says which served call each
result answered and what came back in it; and `table` walks the two together. A result carrying
the right identifier under other bytes is a message the model did not receive, and so is a message
that never arrived, and the two are named apart as `mismatched` and `missing`. A read that finds
any difference prints it and exits nonzero, because an episode whose own transcript does not hold
what was committed is one whose analysis would be about feedback that may never have arrived.

Refusals are read the same way. A refusal advances no protocol state, so it exists only as text
the model saw, and `grades/refusals.json` is the count the server kept to check that reading
against. The server writes that count inside the call that issues each refusal, before the error
goes back, because a server is taken away rather than asked to stop and nothing runs on the way
out of that; anything that sampled the count instead would leave a stale number that reads exactly
like a good one. So a run that has no count is a run whose server never served, and for a launch
that calls the run finished that is a disagreement like any other: the check exists to catch a
refusal the server sent and the model never saw, and no count means nothing to catch it with. A
launch that recorded an incomplete run is the one place the absence is reported rather than
counted, and it is reported rather than passed over.

`grades/` is a sibling of the directory the agent works in rather than a child of it. That is
weaker than the mount boundary the earlier cell had, and it is what a launch on the host can
offer: an agent running under `bypassPermissions` can read the filesystem it is running on.

## What this is not

The earlier cell ran its agent in a container with the task server outside it, reachable over the
network. This serves over stdio, as the quickstarts do, which means the server is a child of the
agent's own process tree and cannot be on the far side of a container boundary. Restoring that
isolation needs a network transport for the gateway, which this does not add.

There is also no resume. The earlier cell needed three passes to work 223 of its 480 tasks, so a
full roster here will want one too, and a generation left unfinished is left unfinished.
