"""Self-improvement experiment harness — study cell #1 (issue #57).

Claude Code (Claude Sonnet 5) on AutomationBench, one instruction: **"Get Better."**

This package is *experiment-side* only — it imports the hgym core + the AutomationBench env
but modifies neither. It provides the reusable spine every other matrix cell inherits:

  - :mod:`.split`    — deterministic, disjoint train / held-out partition of the public tasks.
  - :mod:`.broker`   — the curriculum broker over AutomationBench (non-repeating train stream,
                       authoritative seal-based scoring, external self-snapshots).
  - :mod:`.sink`     — :class:`MetricSink` (LocalSink default, WandbSink optional).
  - :mod:`.snapshot` — content-hashed whole-workdir snapshots ("the self" over time).
  - :mod:`.policy`   — a Policy protocol + StubPolicy for keyless, deterministic smoke runs.
  - :mod:`.heldout`  — the authoritative held-out evaluator (seal-scored, keyless).
  - :mod:`.arms`     — the real Claude Code runners: treatment ("Get Better", persistent self)
                       and control (fresh context per task, no persistence, no instruction).
  - :mod:`.experiment` — the two-arm orchestrator (the maintainer triggers the full run).
  - :mod:`.smoke`    — the cheap end-to-end smoke test (proves the loop, does not run the study).

Nothing here writes a secret or a large file into the tree: all run artifacts land in the
gitignored ``experiments/selfopt/runs/`` dir (or ``$HGYM_SELFOPT_RUNS``).
"""
