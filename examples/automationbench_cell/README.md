# The AutomationBench cell

One Claude Code session works a roster of AutomationBench tasks, served as a single protocol v2
generation. It is a rerun of a cell that has already been recorded: same benchmark, same tasks in
the same order, same model at the same effort, same launch flags, same regime, and the same two
containers with the same boundary between them, so that the serving contract is the only thing
that differs between the two.

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
answered `get_task` with the step budget of the task it was handing over, and it offered a
`queue_info` tool that answered, on request, how many tasks were still queued. The recorded agent
used both: it paced each task against the budget, and it called `queue_info` and quoted the count
in its own progress notes. A `pull` carries no budget, this protocol has no queue to inspect, and
the task record is built so that no position could be written into it, so the agent here works
the roster without knowing where in it it is.

## The command

Docker must be running, and the environment must hold a credential for the agent:
`CLAUDE_CODE_OAUTH_TOKEN`, or an `ANTHROPIC_API_KEY`. The agent's Claude Code home is fresh, so
nothing else authenticates it, and a launch without one refuses before it builds or serves
anything. The probe needs no credential.

```bash
# the whole roster, the regime the earlier cell ran, the model and effort it ran
uv run python -m examples.automationbench_cell.cell run --tasks cell-one

# a shorter rerun: the same tasks in the same order, fewer of them
uv run python -m examples.automationbench_cell.cell run --tasks cell-one:20

# what it scored
uv run python -m examples.automationbench_cell.cell table <run directory>

# whether the boundary holds, asked from inside a container started as the agent's is
uv run python -m examples.automationbench_cell.cell probe
```

`--tasks` also takes a plain list or range (`0-19`, `4,0,2`) for a roster of your own. `--model`
and `--effort` default to the earlier cell's `claude-opus-5` and `xhigh`, and both are overridable
for a smoke run, which is the same override that harness took for the same reason. `--runs` moves
the run directories.

## The two domains

The run is two containers on a private network the run makes for itself, and the boundary between
them is what makes this a rerun rather than a description of one.

The **agent's container** holds three directories and nothing else on this host: the directory it
works in at `/work`, the Claude Code home its memory and skills land in at `/root/.claude`, and,
read only and outside both, the one file naming the endpoint. It has no run directory, no
repository and no benchmark cache. So the roster, the durable history, the task answers, the
scoring assertions and every sealed grade are not files this agent could open, whatever it is
permitted to do, and it runs under `bypassPermissions` exactly as the earlier cell's agent did.

The **server's container** holds all four and publishes one thing, the gateway's MCP endpoint, on
the private network and on no port published to the host. The durable service the history lives in
starts inside that container and binds that container's own loopback, so a network the two share
still carries nothing but the endpoint. The agent keeps general internet egress, because that cell
did: its rollout arm ran with the agent's own tools in place, web included.

It has the benchmark source mounted read only, because the task definitions, the answers and the
scoring assertions are not a run's to write. A cache nobody has filled yet is therefore filled
before the server starts, by a one-shot container from the same image holding that one directory
and nothing else of the run, so a first launch on a clean host, or with a `--cache` of your own,
pays for the pinned source once.

`probe` is that claim as a measurement rather than as a paragraph. It stands both domains up over a
real generation, with a real roster written and a real history open, then starts the agent's own
container, with the agent's image, mounts, network, working directory and environment, and asks it
what it can reach: whether the run directory, the grades, the repository or the cache have any path
there, whether a roster, a history or a run record exists anywhere on that filesystem, what the
mount table holds, what `/cfg` holds and whether it can be written, which variables the container's
own process was handed, whether the PID namespace is its own and the container unprivileged and
seccomp filtered, whether the network namespace is its own and none of the server's loopback
addresses answer in it, whether a container runtime socket or a docker API is reachable, whether an
instance metadata service answers, what the server's name resolves to, what the gateway answers to
a plain read of `cell.json` or `stream.sqlite`, and what the server is listening on. It exits
non-zero if any of it is reachable, and a check it could not carry out counts as one that failed.

What it establishes is that this run's roster, history and grades are on the far side of the
boundary. It is not a claim of isolation, and it reports the retained egress as what it is.

## What the launch is pinned to

Docker must be running, with a credential in the environment: `CLAUDE_CODE_OAUTH_TOKEN`, or an
`ANTHROPIC_API_KEY`. Both images are built on first use. The agent gets a working directory of its
own holding the one-line `CLAUDE.md` the earlier cell's agent found in its own, and a fresh Claude
Code home beside it.

The command line is the visible half of a launch, and the rest is pinned too, because a difference
in any of it is a difference in the agent rather than in the serving.

- The CLI build is the `2.1.220` the recorded run reported, installed at that version into the
  agent's image and asked for its version before anything starts. `--allow-cli-drift` runs on
  another build and has the difference recorded rather than refused.
- The agent's image is pinned by what it is built from, not by its tag: the base by digest, the
  package and registry the CLI comes from, the version, the moment of the Debian archive its OS
  packages are installed from with the exact version of each, and a digest of the recipe. Those
  are checked before anything starts, and `--allow-image-drift` records the difference rather than
  refusing it. Both images carry a digest of their own build inputs as a label, so an image built
  from an earlier checkout is rebuilt rather than reused under the name of this one, which is what
  keeps the server's copy of the gateway and the grader the copy beside this README.
- The server's image installs from this repository's `uv.lock`, exported to a pinned requirements
  file with a hash for every distribution, and the lock is one of the inputs its label is a digest
  of. The ranges in `pyproject.toml` say which versions are admissible and the lock says which ones
  were chosen: a build resolving the ranges live gave the run a gateway two major versions above
  the locked one under a label that said nothing had moved. It runs this package from the source it
  copied rather than building a wheel of it here, which would ask the index for a build backend at
  whatever version it was serving that day, outside both the hashed requirements and the label.
- Each build is handed an archive of exactly the files its identity names, so what the image can
  hold is what the digest was taken over. What this repository generates is in neither: a run
  directory is where a probe and a launch write by default, so the tree the server's image copies
  holds one as soon as the first command above has been run, and an image built after that used to
  carry the last run's transcripts and grades under a label naming the source beside it.
- Neither container inherits anything. The agent is given this cell's two settings and, by name,
  whichever credential authenticates the run; the server is given the roster, the domain, the
  schedule and the paths its own mounts are at. Two variables this repo reads are why that is more
  than tidiness: `AUTOMATIONBENCH_SRC` and `SHOGYM_CACHE` would replace the task source under a
  run whose record still described the standard cell.
- The credential is handed to docker by name rather than by value, so neither the run's record nor
  the host's process table ever holds what it was worth.
- The tool surface exists only once the agent has started, so it is read off the transcript's
  first line afterwards and compared with the surface the recorded run reported: the build, the
  tools, the subagents and the skills. What differs is printed and kept.

`run.json` holds the whole effective launch: the command, the topology with the mount list of each
container, what each image was built from, the environment each was handed with the credential
named rather than copied, what the server's container was listening on, a digest of every directory
the agent started from, the build that ran, and the first line that build wrote.

It ends by saying whether the launch is a run at all. A CLI that never reaches the endpoint prints
its opening line and exits nought, which is how a trailing slash once produced a cell of no tasks
reported as a success, so a launch is complete only if the agent's transcript holds a `pull` that
came back with a task, the gateway's log holds a request it answered at `/mcp`, and both
containers came down afterwards. A `pull` on its own is not enough: it is a call the model wrote,
and a call the transport refused or redirected leaves the same line behind as one that was served.
Anything else is
recorded as `incomplete` with the reason and returns non-zero. Ordinary termination is handled
rather than fatal: `SIGTERM` and `SIGINT` stop both containers by name, take the network down, and
finish the record before the launcher exits. `SIGKILL` and a host that loses power can do neither,
and leave containers a later run has to clear by hand.

The working directory is `/work`, which is the path that cell's agent worked in.

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
  cfg/.mcp.json  the endpoint the agent was told to connect to
  stream.jsonl   the agent's own transcript
  server.log     what the server said while it served
  run.json       the roster and schedule, the topology, and the launch as it resolved
```

Only three of those were ever inside the agent's container, and `grades/` is not one of them.

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

## What this is not

There is no resume. The earlier cell needed three passes to work 223 of its 480 tasks, so a full
roster here will want one too, and a generation left unfinished is left unfinished.

The boundary is a mount list and a network, and neither is a claim about what the agent's own
egress reaches. That cell's agent had the internet and so does this one, so a run's integrity
rests on the answers being on the far side of the endpoint, not on where the agent could otherwise
go.

The agent's image is not that cell's image. That run recorded the CLI build and nothing about the
container it ran in, and the image itself is gone, so the shell, the Node, the OS package set and
whatever else the model reached through Bash cannot be shown to be the same. What is pinned here
is what this cell builds, from today's base digest, so every rerun after this one is comparable
with this one.
