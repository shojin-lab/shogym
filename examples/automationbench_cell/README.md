# The AutomationBench cell

One Claude Code session works a roster of AutomationBench tasks. It asks the server for a task,
works it, ends it, and the score for that task comes back when it asks for the next one. The
session runs in one container, the tasks and the answers and the scores are in another, and the
only thing between them is the endpoint the agent connects to.

Docker must be running. `run` also needs a `CLAUDE_CODE_OAUTH_TOKEN` in the environment: the agent
gets a fresh Claude Code home, so nothing else authenticates it, and a launch without the token
stops before it builds anything. Both images are built on first use, which takes a few minutes
once.

## The commands

From the repository root:

```bash
# ask, from inside a container started as the agent's is, what it can reach
uv run python -m examples.automationbench_cell.cell probe

# the default roster: 200 tasks, one session, claude-opus-5 at xhigh effort
uv run python -m examples.automationbench_cell.cell run

# what it scored. The run prints the directory it made as it starts; this picks the newest one
RUN_DIR=$(ls -dt examples/automationbench_cell/runs/cell-* | head -1)
uv run python -m examples.automationbench_cell.cell table "$RUN_DIR"
```

`--tasks cell-one:20` runs the first twenty of the roster, and `--tasks 0-19` or `--tasks 4,0,2`
runs one of your own. `--model` and `--effort` are overridable for a cheap smoke run, `--runs`
moves where run directories go, and `--schedule never` runs a session that is told nothing about
what it scored. `probe` needs no credential, and its container is then a run's with that one
variable missing.

## What a run leaves behind

```
cell-<schedule>-<stamp>-<token>/
  grades/              the history, the sealed grades, the blobs, cell.json and refusals.json
  self/                the directory the agent worked in
  home/                the Claude Code home it was given
  cfg/claude.mcp.json  the endpoint it was told to connect to
  stream.jsonl         the agent's own transcript
  server.log           what the server said while it served
  run.json             the launch as it resolved, and whether it is a run at all
```

Only `self/`, `home/` and `cfg/` were ever inside the agent's container. `grades/cell.json` says
which benchmark task each attempt was, and `table` reads the scores back through it.
