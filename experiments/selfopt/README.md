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
| Two-container sandbox | `sandbox/` | The agent runs free; the broker (targets + held-out answers + provenance) is an isolated container the agent can't reach or forge → "cheating is a finding" is scientifically clean. `study.py` runs the **full study** under this topology (the real launch vehicle); a single-arm smoke is just `study.py --arm treatment --train-size N`. |

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
uv run python experiments/selfopt/sandbox/study.py --go --build --arm both --wandb  # FULL study, sandboxed + live W&B
```

## Secrets & artifacts
- Claude / W&B credentials are read from the **runtime environment only** (on macOS the Claude
  OAuth token comes from the Keychain, not `~/.claude`). Never written to a tracked
  file.
- All run outputs (traces, snapshots, provenance, metrics) go to the **gitignored** `runs/` dir
  (override with `$HGYM_SELFOPT_RUNS`, e.g. `~/.cache/hgym`). Nothing large is committed.
