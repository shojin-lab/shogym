# `orca_bench`: ORCA-bench, root-cause analysis over a recorded telemetry stack

A faithful shogym port of [**ORCA-bench**](https://github.com/ORCA-bench/ORCA-bench): 755 SRE
tasks over 77 incidents in a microservice demo application. A task hands the agent a user-visible
complaint ("users cannot complete their purchase") and the time it was reported, and asks for a
structured incident RCA report. The evidence is a real observability stack (metrics, traces and
logs, behind Grafana) replaying a frozen snapshot; the report is graded by an LLM judge that ships
inside every task, against one rubric per plausible root cause.

Like every shogym env this **describes** a task, **serves** its tools over MCP, and **verifies** a
recorded trajectory while an external harness drives the tools. See
[`../README.md`](../README.md). The runnable demos live in
[`examples/`](../../../../examples/).

> **This port lands in two phases (issue #77). Phase 1 is what is here**: the dataset loader, the
> task index, the redacted `describe()`, the judge preflight, and the judge's verdict parsing,
> all offline. **Phase 2 adds the compose backend** that brings a task's 28-service stack up; the
> calls it needed are settled (see [Decisions](#decisions)), so what remains there is
> implementation. Until it lands, constructing and describing tasks works and *serving* an episode
> raises `BackendUnavailableError` with a message that says why.

## Running it

> Needs **Python 3.12**. Phase 1 needs no key and no Docker; it downloads the dataset (~192 MB)
> on first construction. Serving an episode (phase 2) will additionally need a Docker daemon and
> ~133 GB of disk. See [Requirements](#requirements).

### Construct + serve

```python
import shogym
from shogym.envs.orca_bench import tasks

env = shogym.make("orca_bench")          # downloads the pinned dataset once into ~/.cache/shogym
spec = env.describe("0")                 # task 0: its instruction.md + the tool manifest

# Slice the 755 tasks by the labels the benchmark publishes numbers for.
hard = tasks.select(env.refs, difficulty="hard", is_control=False)
by_snapshot = tasks.group_by_snapshot(env.refs)   # 125 groups: the phase-2 staging unit
```

Serving is the phase-2 path. Once the backend lands it is the usual one:

```bash
export OPENAI_API_KEY=sk-...             # the task's own verifier is an LLM judge
uv run python -m shogym.cli serve orca_bench --task 0 --trace ./shogym_logs/orca.jsonl
```

The harness reads the instruction via `describe`, investigates through `exec` / `read_file` /
`write_file`, writes its report to `/app/report.md`, and calls **`submit_report`**, the **score
terminal**: it seals the episode, `finalize` runs the task's own verifier over the report, and the
episode ends. See the shared
[terminal lifecycle](../README.md#terminal-lifecycle-seal-terminal-score-terminal-abort). shogym
reads the verdict off the trace via `shogym.result_from_trace(...)`.

**Config** (via `shogym.make(name, config)` / `env_config`): `task` (the default task, by name or
index), `dataset_dir` (an already-extracted dataset, which bypasses the download and is what the
offline tests use), `tasks` (an explicit list of `OrcaTaskRef`, e.g. one difficulty tier or one
snapshot's group), `judge_model` / `judge_effort` / `judge_base_url`, and `max_steps`.

### Claude Code example

Any quickstart under [`examples/`](../../../../examples/) serves this env once phase 2 lands.
Point it here with the single variable at the top of its `serve.py`:

```python
ENV = "orca_bench"
```

## Requirements

The Python pin and the `uv sync` / `pip install` / `import shogym` mechanics are the shared
[requirements boilerplate](../README.md#requirements-boilerplate). The `orca_bench` extra carries
no Python packages: everything the port needs is either stdlib or already a core dependency, and
the LLM judge ships inside every task. On top of that:

- **The dataset, downloaded on demand.** `orca-bench/orca-bench` (**CC BY 4.0**), revision 2, 755
  tasks, ~192 MB. shogym vendors none of it: the first construction fetches the pinned revision
  from the Harbor hub over plain HTTP into `~/.cache/shogym/orca_bench/<revision hash>/` and every
  later one reads the cache. `SHOGYM_ORCA_BENCH_DATA_DIR` relocates it; `SHOGYM_CACHE` relocates
  the shared cache root. No account, no token, no `harbor` CLI.
- **`OPENAI_API_KEY`, to grade.** The task's verifier is an LLM judge. The key is checked when an
  episode is *served*, not when the env is constructed, so `shogym.make`, `describe`, and the tool
  manifest stay keyless and offline. `judge_base_url` (or `OPENAI_BASE_URL` in the environment)
  points the judge at an OpenAI-compatible endpoint instead, and such an endpoint may be keyless:
  the port then hands the verifier a placeholder credential, because the verifier constructs its
  SDK client unconditionally and the SDK refuses a missing or empty key before it ever reaches the
  endpoint. With no endpoint named there is nowhere keyless to point at, so the key is required.
- **Docker + ~133 GB of disk, to serve (phase 2).** The environment image is ~86.8 GB and carries
  a ~46 GB telemetry snapshot cache. Measured on one host: 4.10 GiB RSS for the running stack, 3.4
  GB of disk per trial, ~135 s warm start, ~8 s teardown.
- **CI runs the offline half only, and always will.** A hosted runner has ~14 GB of disk, so the
  live path cannot run there at any budget, and no longer timeout changes that. Everything phase 1
  ships (loading, indexing, `describe`, the preflight, verdict parsing) is tested offline against
  synthetic fixtures and runs in the core suite; one `network`-marked test downloads a single real
  task (~30 KB) to exercise provisioning against real bytes and is deselected by `-m "not
  network"`.

## How it works

### describe → TaskSpec

`describe(task_id)` publishes the task's upstream `instruction.md` verbatim, plus a two-line
footer naming the task's public id and the dataset pin. **Nothing else**, and that is the port's
load-bearing correctness property rather than a style choice. Each task's `task.toml`
`[metadata]` carries the answer: the feature flag that caused the incident, every root-cause
event and its timestamp, and (for a control task) the quiet window that says there is no incident
to find. Upstream is safe by accident, because Harbor never mounts `task.toml` into the agent's
container; a shogym env has no such accident to rely on.

So the footer carries no label from `[metadata]` at all, not even the ones that are not answers:
`section` alone would give away that a task is a **control**, and the snapshot id timestamps the
window the incident sits in. `tests/envs/test_orca_bench_describe.py` asserts the instruction is
published verbatim, that no ground-truth string appears anywhere in the serialized `TaskSpec`, and
that the footer contains no metadata value. It also includes the mutation check that a
`describe()` which did leak is caught.

Tasks are selectable by **position** in the env's task list (`0..754` for the whole dataset, in
name-sorted order) or by **name**. An unknown name or an out-of-range position raises rather than
silently serving a different task under the requested id, and a task list containing the same name
twice is refused at construction: identities are a set, and a duplicate would answer to the other
copy's id.

A numeric task id is always a position in *this env's* list, which is a slice whenever `tasks=` is
given. `OrcaTaskRef.dataset_index` is the position in the full 755 and is deliberately named
apart: it is provenance (it rides along in the loaded task as `dataset_index`), never a selector.
Interchanging the two is silent when the wrong number happens to be in range, so the round trip
the serve layer performs, load a task and then describe the id it reported, is pinned by a test
over a non-trivial slice.

### Tools (served over MCP)

| Tool | What it does | Terminal |
|---|---|---|
| `exec(command)` | Run a shell command in the task container (the telemetry answers on the Grafana HTTP API named in the instruction). | `none` |
| `read_file(path)` / `write_file(path, content)` | File I/O in the container. The graded artifact is `/app/report.md`. | `none` |
| `submit_report()` | Seal the episode and grade the report. | **`score`** |
| `terminate()` | The reserved no-credit abort. | `abort` |

The tool *schemas* list with no backend, so `describe` and the manifest probe are offline, which
is what lets phase 1 register and describe this env with no stack at all. `begin_session` is where
the backend is constructed, and where phase 1 stops.

### finalize + verify

`submit_report` seals the episode; `finalize` then runs the task's own verifier over
`/app/report.md` and parses the two files it writes (`reward.json`, `reward-details.json`) into a
verdict. The judge's prompt, reasoning and rubrics are answer oracles and stay in the private
diagnostic; the public verdict carries only the published numbers. `verify` is pure over that
evidence.

**A failed submission is a failure, not an excluded attempt.** Upstream's verifier reads the
agent's `/app/report.md` inside the same blanket `except` that wraps the judge call, so an absent,
unreadable or oversized report produces exactly the payload a dead judge model does. Since judge
errors are filtered out of results, an agent that never wrote a report would convert a failed
attempt into no attempt at all, and would score better for it. So the class is decided by
**cause**, before the verifier runs: anything wrong with the agent's own artifact is an ordinary
verified zero carrying `submission_error`, and only grading infrastructure (the judge model, the
endpoint, the key, a crash inside the judge) is a `judge_error`. An **empty** report stays valid,
because that is the required answer for a control task, and a report over
`judge.MAX_REPORT_BYTES` (256 KiB, against real reports of 2 to 5 KB and a ~70 KB judge prompt) is
a failed submission too, so the same exclusion is not reachable from the other end.

**The bytes that were sealed are the bytes that get graded.** Sealing stops the agent's tool
calls, not processes it already started in its container, so checking the report and then letting
the verifier reopen the same agent-writable path grades whatever is there the second time: a
watcher can delete the report in between and recreate the exclusion, or swap it so the score
belongs to something that did not exist at the seal. The report is therefore **captured once**, at
the seal, through a single descriptor (symlinks refused rather than resolved, and the open is
non-blocking so a named pipe or device at that path is refused instead of stranding the terminal
call until a writer appears), validated against those bytes, and held where no agent process can
reach it; the verifier is pointed at the capture and never at the live path. The backend protocol states this as a requirement, since phase 2 is
where the container work happens.

**A judge failure is an explicit grade, never a silent zero.** Upstream's verifier writes
`{"reward": 0.0}` when the judge itself raises, byte-identical to what it writes for a wrong
report, and easy to hit, because the shipped default judge model (`openai-gpt-5.4`) was retired
and 400s on every call. This port refuses to inherit that: a verdict counts as a grade only when
`reward.json` carries `rca_accuracy` (the key the verifier emits exactly when at least one rubric
was scored); everything else grades as `judge_error=True`, which `verify` emits as its own
feedback field so grading-infra failures can be filtered out instead of averaged in. The
preflight is the other half: the retired model is refused **by name** at construction, and a
missing key at session start.

## Tasks

755 tasks of the pinned revision, indexed in name order (stable across hosts and re-downloads):

| Label | Values |
|---|---|
| `difficulty` | `easy` 245, `medium` 264, `hard` 246 |
| `section` | `exact` 208, `exact_range` 215, `broad_range_time_of_day` 146, `broad_range_day` 48, `control` 138 |
| `is_control` | 138 tasks with no incident to find |
| `snapshot` | 125 distinct telemetry snapshots; the largest group is 95 tasks |

`difficulty` is how vaguely the incident is specified; `section` is how the report time is phrased.
`is_control` is read from upstream's own definition (a task with no `events`), which agrees with
`section == "control"` on every task of this revision. `snapshot` is the phase-2 staging key:
`tasks.group_by_snapshot` gives the groups a runner should walk so each snapshot is staged once.

## Scoring

`verify` emits, per episode:

| Feedback | What it is |
|---|---|
| `reward` | The judge's rubric score, normalized to `[0, 1]` (the mean per-rubric score / 3). This is what the leaderboard averages. A payload outside that range, or not finite, is a judge error rather than a score. |
| `success` | The per-task binary (see below). |
| `rca_accuracy` | Did the report name **every** listed plausible root cause (strict all-causes)? The benchmark's headline metric. |
| `hallucinate_any` | Did the report name a cause matching **none** of them? Omitted where the benchmark leaves it undefined. |
| `verified` | False for an abort, or when the grade is a judge error. |
| `judge_error` | Present and `True` only when the verdict is not a grade at all. |
| `submission_error` | Present when the agent's own report could not be graded (absent, not a regular file, unreadable, or over the size the judge can be given). A graded zero, not an exclusion. |
| `judge_model` / `judge_effort` | Which judge produced the score: what `reward-details.json` recorded as having run, falling back to the configuration when the judge failed before recording anything. |
| `judge_endpoint` | `default`, or `custom:<digest>` for a configured endpoint. The URL itself is never published: a private endpoint can name internal infrastructure, and a digest is enough to tell two runs apart. |

That table is the whole published set, and the control label is not a row in it: it decides how
`success` is derived and then stays in the private diagnostic, which the serve layer drops before
anything reaches the caller or the trace.

**What redaction does and does not promise.** Before and during an attempt, nothing from
`[metadata]` reaches the agent: that is the `describe()` contract above, it is the one that
protects the attempt, and it is intact. **After** an attempt has been graded, the published
metrics can disclose whether the task was a control, and this port accepts that rather than
distorting them:

- `success` is derived class-dependently, because the benchmark's own metrics are (see below), so
  a passing control shows `success` true beside `rca_accuracy` false, a combination an incident
  task cannot produce.
- `hallucinate_any` is undefined on a control task, so upstream's verifier omits it, and its
  presence says the task had an incident.

Both are the same phenomenon: publishing upstream's per-task numbers publishes facts about the
task's class. The alternative is a class-independent `success`, which would diverge from what the
leaderboard means by these metrics, or invented values for undefined ones, which would misreport
them. Neither is worth buying a property that only holds until the first graded attempt of a task
anyway. A run that cares (an agent evaluated repeatedly on the same tasks with its own scores fed
back) should not feed graded verdicts back to the agent, which is a harness decision rather than
an env one.

All three are resolved **once, when the episode equips its verifier**, and carried through
grading. An endpoint that comes from `OPENAI_BASE_URL` is ambient state a process is free to
change while a long episode runs, so re-reading it at grading time would record an endpoint that
never scored the report. One reading, too, not one per derivation: the episode takes a frozen
snapshot of the judge's environment variables and the preflight decision, the environment the
verifier is handed and the audited endpoint id all come from that, since three reads of a mutable
mapping can disagree and the disagreement is invisible afterwards.

The judge provenance rides on the verdict and the feedback, not only in the server-side
diagnostic, because changing the judge model or its effort **changes the scoring function**. A
bare number cannot be compared against another run's, or re-read a year later when a model id has
started meaning something else, so every score says what scored it.

Read them back off the trace with `shogym.result_from_trace(...)`. See the shared
[read-back semantics](../README.md#reading-a-score-back-result_from_trace).

**`success` is `rca_accuracy` for an incident task, and a full reward for a control task**
(signed off; see [Decisions](#decisions)). The split is forced: on a control task the verifier synthesizes a single
`(no_incident)` rubric whose `feature_flag_match` is hard-coded `False`, so `rca_accuracy` is 0 on
all 138 controls however well the agent did. Reading `success` off it alone would score every
control as a failure. Upstream never mixes them either: the published metrics restrict
`rca_accuracy` and `hallucinate_any` to incident tasks.

## Fidelity & deviations

- **Pinned**: dataset `orca-bench/orca-bench` **revision 2**
  (`1ef729757d4974ffe4e835d541c601f957975edf8c93ef02eec97e26d3069b93`), and the environment image
  `orcabench/sre-otel-snapshot` by **digest**
  (`sha256:19c8c097ec10be561d6fd49c9b0fff0c6188b583bcb41ec1c5945d7f5fdbd671`), not by its mutable
  tag.
- **No upstream source is fetched.** The judge (`tests/check_prediction.py`) ships inside every
  task and is byte-identical across all 755, so unlike the tau2 / yc_bench / automationbench ports
  this one provisions no upstream Python at all.
- **The image is treated as the artifact, the repo as documentation.** The upstream repo cannot
  rebuild the published image: six files its setup references are absent, and the published
  entrypoint diverges from the repo's template by one load-bearing line. Hence the digest pin.
- **The judge model is overridden.** Upstream's default (`openai-gpt-5.4`) no longer exists; the
  port defaults to a current model at upstream's own default effort (`high`) and refuses the
  retired one by name. A different judge model is a scoring change, not a cost tweak, so it is
  recorded here rather than left to the environment.
- **`describe` publishes strictly less than upstream's task directory contains**, see
  [describe → TaskSpec](#describe--taskspec). This is a deliberate divergence from a benchmark
  whose safety here depended on a runner's incidental behavior.
- **The benchmark has already silently degraded, and this port does not yet fix it.**
  `jaeger-config-snapshot.yml` pins `max_span_age: 2160h` (90 days) and the snapshot data is from
  2026-04-19..23, so it aged out of Jaeger's lookback around 2026-07-22. Live, `GET /api/services`
  now returns empty: an agent's natural first move sees a system with no services, while
  explicit-window queries still work. It worsens daily, and **any run today is not comparable to
  the paper's numbers**, which predate the expiry. Phase 2 pins the Jaeger query service's clock
  to restore them; see [Decisions](#decisions).

## Decisions

The five calls this port needed are settled (owner, issue #77 and this PR's thread). They are
recorded here because each one changes what a number from this env means.

1. **The `max_span_age` expiry: pinned clock.** The snapshot data is from 2026-04-19..23 and aged
   out of Jaeger's 90-day lookback around 2026-07-22, so a live stack answers `GET /api/services`
   with an empty list and no run is comparable to the paper's numbers. Phase 2 pins fake time
   inside the Jaeger query service with `libfaketime`, scoped to that service alone, so the stack
   behaves as it did during the paper's runs. The artifact is not edited, and the pinned time and
   mechanism are recorded in run provenance. A lookback patch was rejected: it re-expires once the
   per-day index names exceed OpenSearch's 4096-byte URL limit, around day 135.
2. **No service trimming.** 21 of the 28 services produce no telemetry in replay and dropping them
   would save ~4 GB of RSS, but the agent's world stays identical to the paper's. Fidelity first.
3. **No image re-derivation.** A smaller image would cut the ~133 GB floor, but it cannot be
   rebuilt from the repo as published and re-deriving one breaks digest comparability. Phase 2
   pulls the pinned digest.
4. **Scoring: `success` is `rca_accuracy` on incident tasks and a full reward on controls**, for
   the reason given under [Scoring](#scoring). The alternative (publish `reward` alone and leave
   `success` undefined) was rejected: it gives no per-task pass/fail at all. The class disclosure
   this implies is accepted and documented under Scoring rather than scored around.
5. **CI runs the offline half only.** A hosted runner has ~14 GB of disk against this env's ~133
   GB floor, so the live path cannot run there at any budget.

What is left for phase 2 is implementation, not deliberation: the compose backend, the
named-volume staging of the snapshot cache, and the clock pin above.

## Gotchas

- **A cold cache costs one download; a partial one is never mistaken for a warm one.** The loader
  treats the dataset as present only when all 755 task directories are **complete**, so an
  interrupted first run resumes rather than silently indexing a subset. Each task directory is
  published by a single atomic rename, and concurrent provisioners are serialized by the same
  `flock` policy the upstream-source provisioner uses.
- **A complete published task is a winner, never an obstacle.** That `flock` policy deliberately
  degrades to no inter-process exclusion on a filesystem that cannot provide it, promising only
  that concurrent cold starts stay redundant-but-correct. Two provisioners can therefore both be
  publishing the same task, so the loser accepts a destination it finds complete and discards its
  own download (both came from the same pinned archive) instead of replacing a tree a reader may
  already be holding. The destructive repair is reserved for a destination observed incomplete,
  and it displaces that tree by rename rather than deleting it in place.
  Deciding that and acting on it are **one step**, under a per-task `os.mkdir` lock that is atomic
  on every filesystem, including the ones `flock` has already given up on. Without it the same
  hazard returns one level down: two repairs of the same damaged tree both decide to replace it,
  and whichever acts second is acting on something it learned before its peer's tree existed.
  Re-checking just before the rename only narrows the window it is checking for. The lock is held
  across observe-decide-replace and never across the download, so a waiter is never queued behind
  a network fetch. A lock arrives already stamped with its owner's token (built aside, then
  renamed into place, which rename refuses onto a non-empty target), and release is conditional on
  still holding that token.
- **A lock is broken only when its holder is provably dead, never merely because it is old.** A
  wall-clock lease cannot fence a holder that is still running: a filesystem operation can stall
  past any deadline, and the holder then resumes inside its critical section and mutates the
  destination on a decision it made before its successor published. Ownership rules make that
  holder's cleanup harmless; nothing outside it can stop its body. So the timeout authorizes
  nothing. It bounds how long a waiter waits before failing closed with an actionable error
  naming the lock, the holder, and the remedy. Breaking requires the token to name this host and
  that pid to be gone (`os.kill(pid, 0)` raising `ESRCH`, or a recorded start time that no longer
  matches). A holder on another host, or one whose token cannot be read, is never assumed dead:
  this cache is per-user under `~/.cache`, so same-host is the case that matters, and a shared
  network cache is exactly where the answer should be "cannot tell".
- **Clearing a dead holder's lock names that holder, not the path.** A proof is about a
  generation, and between proving and acting the path can already be a live successor that
  cleared the same corpse first. So recovery unlinks exactly the token that was proved dead and
  removes the directory only while it is empty; if the token is gone, someone else dealt with it
  and whatever is there now is not this call's to touch. Acting on the path instead, by delete or
  by rename, evicts that successor from its own critical section and puts two publishers inside
  it. This is the same conditional protocol a holder uses to release its own lock, performed on a
  dead holder's behalf.
- **The revision pin is cryptographic, not just an address.** The registry gives a
  `content_hash` per task, and this port recomputes it over every extracted package before
  publishing (`dataset.content_hash`, the hub publisher's own algorithm: sha256 over
  `<relpath>\0<sha256 of the file>\n` for each collected file, in path order). Verified against
  the registry's recorded value for all **755** tasks of the pinned revision. Using that hash only
  to *form the URL* would leave the cache holding whatever answered the request, and phase 2
  executes what the cache holds: the compose file, `tests/test.sh`, and the judge. A mismatch
  fails closed, naming the task and both hashes, and publishes nothing.
- **A warm cache is re-authenticated, not just counted.** Every construction re-hashes the cached
  tasks and compares them with the pin, so a file changed after publication (an instruction, a
  compose file, a verifier, an expected answer) is caught rather than served as the pinned
  revision forever. It costs about 3 seconds serially and 1.6 in parallel for the whole 755-task
  revision (176 MB), measured, which is not enough to justify a cheaper proxy. A task whose bytes
  moved is **repaired**, exactly like an incomplete one: unlike residue, the pin says precisely
  what those bytes must be, so re-fetching is deterministic and cannot destroy anything a person
  authored.
- **A cached dataset is an exact set, not a superset.** Every pinned task present and complete is
  half the claim; the other half is that nothing else task-shaped is in there. A leftover
  directory from an older or interrupted provisioner is indexed like any other task, which
  changes `num_tasks`, shifts every id after it, and quietly moves what a slice or a published
  result refers to. So provisioning refuses a cache holding directories the pin does not name,
  before it fetches anything and again as a postcondition, and the index is built from the pinned
  identities rather than from a directory scan. Residue is **named, never removed**: this code
  cannot tell an older run's leftovers from your own work or from evidence of a bug, and nothing
  it writes lands there (its staging directories are hidden and prefixed), so residue means
  something happened that it did not do. The error says which directories and what to run.
- **A cached task is its files, not its `task.toml`.** `dataset.REQUIRED_TASK_FILES` is the
  contract: the three the index and `describe` read (`task.toml`, `instruction.md`,
  `environment/Dockerfile`) plus the four a graded episode needs (`environment/docker-compose.yaml`
  and the `tests/` verifier trio). An extracted archive is validated against it *before* it is
  published and again on every warm read, and a task that fails is re-fetched rather than raised
  on. An atomic rename proves nobody saw the extraction in progress; it proves nothing about what
  the archive contained. The oracle (`solution/`) and `tests/rubrics/` are deliberately not
  required: the port never runs the oracle, and rubrics exist only for incident tasks.
- **`task.toml` is the answer key.** Anything new that reads it (a report, a log line, an
  analysis script) has to keep it away from the agent. `tasks.answer_strings()` names exactly
  which strings that means, so a new surface can be checked the same way `describe` is.
- **Do not read `granularity` when you mean `difficulty`.** Both ladders use the word `hard` for
  different tiers, so the mistake is silent: it just redefines every per-tier number. The index
  refuses a task whose `difficulty` is outside `easy`/`medium`/`hard` for that reason.

## Layout

| File | What it is |
|---|---|
| [`env_v1.py`](env_v1.py) | The registered env: describe (redacted), session lifecycle, `finalize`, and the pure `score_evidence`. |
| [`dataset.py`](dataset.py) | The pins and the on-demand download: resolve the revision, list its tasks, fetch and publish each one into the cache. |
| [`tasks.py`](tasks.py) | The task model and index over a cached dataset directory: labels, slicing, snapshot grouping, and `answer_strings`. |
| [`judge.py`](judge.py) | The judge's configuration and preflight, and the verdict parsing that makes a judge failure explicit. |
| [`backend.py`](backend.py) | The phase-2 seam: the backend protocol, the pinned image digest, the staging contract. No Docker code. |
| [`mcp_server.py`](mcp_server.py) | The served tools (`exec` / `read_file` / `write_file` / `submit_report`) and the session registry. |
