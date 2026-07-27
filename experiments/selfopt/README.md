# Self-improvement study — cell #1 (Claude Code / Claude Sonnet 5 on AutomationBench, "Get Better")

The reusable experiment harness for [#57](https://github.com/anndvision/hgym/issues/57): hand a
harness a stream of tasks + one instruction — **"Get Better"** — and measure, honestly, whether
it improves. Experiment-side only (imports the hgym core + the AutomationBench env, modifies
neither). This is the template every other matrix cell inherits.

## The measurement spine
| Piece | File | What it guarantees |
|---|---|---|
| Disjoint train / held-out split | `split.py` | The held-out pool (private-set proxy) is a deterministic, seeded, **disjoint** partition of the 600 public tasks and is **never** placed in the train stream. |
| Curriculum broker | `broker.py` | Dispenses a **non-repeating** train stream; hides index/target/handle; serves AutomationBench's native tools; **seal-scores authoritatively** (RFC-009, #52) — the agent can't forge the number. Keyless (AutomationBench scoring is deterministic + offline). |
| Whole-workdir snapshots | `snapshot.py` | "The self" is archived under its content hash at every task boundary — tamper-evident, recoverable; the diffs are the narrative. |
| Authoritative held-out eval | `heldout.py` | Runs the **real evolving self** over the held-out pool, seal-scored, at start/mid/end — the honest generalization curve. **Both arms** measured (treatment vs a control baseline). The treatment probe plays a **throwaway copy** of the checkpoint self (held-out never trains the self) and runs **web-off** (the held-out split's answers are online — measure capability, not lookup). |
| MetricSink | `sink.py` | `LocalSink` (JSONL + console) by default — no account, no key. `WandbSink` opt-in behind `SELFOPT_WANDB=1` + `WANDB_API_KEY`, wired **broker-side** (the authoritative scorer streams reward/success + self-snapshot artifacts live). |
| The two arms | `arms.py` | **Treatment** (one persistent process, bare **"Get Better"** — tools discovered from `get_task`, full self-surface incl web) vs **control** (fresh context per task, no persistence, no instruction). |
| Two-container sandbox | `sandbox/` | The agent runs free; the broker (targets + held-out answers + provenance) is an isolated container the agent can't reach or forge → "cheating is a finding" is scientifically clean. `study.py` runs the **full two-arm + held-out study** under this topology (the real launch vehicle); `run_sandbox.sh` is the single-arm treatment-train demo. |

## Run it

Smoke test (keyless, deterministic, no model spend, no Docker — proves the whole loop):

```bash
uv run python -m experiments.selfopt.test_broker     # integrity tests
uv run python -m experiments.selfopt.smoke           # end-to-end spine
uv run python -m experiments.selfopt.smoke --real    # + ONE real Claude Code train task
```

The full two-arm study (**real spend — maintainer's call**) prints its plan by default:

```bash
uv run python -m experiments.selfopt.experiment                 # dry-run: prints the plan
uv run python -m experiments.selfopt.experiment --go            # local run; real Claude Code held-out
uv run python -m experiments.selfopt.experiment --go --stub-heldout   # cheap: keyless stub held-out
experiments/selfopt/sandbox/run_sandbox.sh                      # isolated two-container treatment-train demo
uv run python experiments/selfopt/sandbox/study.py --go --build --arm both --wandb  # FULL study, sandboxed + live W&B
```

## Secrets & artifacts
- Claude / W&B credentials are read from the **runtime environment only** (on macOS the Claude
  OAuth token comes from the Keychain — see `sandbox/run_sandbox.sh`). Never written to a tracked
  file.
- All run outputs (traces, snapshots, provenance, metrics) go to the **gitignored** `runs/` dir
  (override with `$HGYM_SELFOPT_RUNS`, e.g. `~/.cache/hgym`). Nothing large is committed.

## Cell #2 — Codex on the same env + same study

Same environment, same seeded task stream, same authoritative scoring — the harness swaps Claude
Code → **Codex** (`codex exec`, GPT-5.6 "terra"). Everything below the harness is **reused
unchanged** (`split.py` / `broker.py` / `snapshot.py` / `sink.py` / `heldout.py` and the broker
container image); only the harness is adapted, behind a small parallel adapter so cell #1 keeps
working untouched:

| Piece | File | Notes |
|---|---|---|
| Codex command / config / trace / auth | `codex_arms.py` | `codex exec --json` (JSONL trace = the stream-json analog); the curriculum served as a **streamable-HTTP MCP server** in an isolated `config.toml` (Codex consumes HTTP MCP natively — no stdio shim); Codex's own sandbox at `danger-full-access` (the container is the isolation boundary); the **prompts are imported verbatim from `arms.py`**; subscription-only auth (billed `OPENAI_API_KEY` stripped from the run env). |
| Codex agent image | `sandbox/agent.codex.Dockerfile` | mirrors `agent.Dockerfile`; the `codex` CLI in place of `claude`. |
| Two-container study | `sandbox/study_codex.py` | mirrors `study.py`; container names prefixed `selfopt-c2-` on network `selfopt-c2-net` — **disjoint from cell #1**, so a concurrent cell-#1 run is never touched. |

Tool policy maps to two Codex levers (no per-tool allow-list): the local self-surface comes
uniformly from `danger-full-access`; the one per-arm capability toggle is **`web_search`**
(treatment-train ON, held-out + control OFF), matching cell #1's web split. **The self** =
the persistent workdir (Codex's `AGENTS.md`), snapshotted whole per task boundary as in cell #1;
Codex's `~/.codex` durable surface (skills/rules/memories) lives in the per-run `CODEX_HOME`
**outside** the snapshotted self so the subscription credential is never archived (see the
open-question note in the issue).

```bash
uv run python -m experiments.selfopt.smoke_codex          # keyless spine + Codex plumbing
uv run python -m experiments.selfopt.smoke_codex --real   # + ONE real codex exec via a local HTTP broker
uv run python experiments/selfopt/sandbox/study_codex.py --plan  # the two-container plan (dry-run)
```
