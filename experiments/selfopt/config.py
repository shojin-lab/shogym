"""Central configuration for the self-improvement harness (all overridable via env vars).

Nothing here holds a secret. Claude/W&B credentials are sourced at RUNTIME from the process
environment (or, for Claude on macOS, the Keychain) and never written to a tracked file.
"""

from __future__ import annotations

import os
from pathlib import Path

# The experiment package dir (…/experiments/selfopt).
PKG_DIR = Path(__file__).resolve().parent

# All run artifacts (traces, workdir snapshots, metrics, provenance) go here. Gitignored.
# Override with $HGYM_SELFOPT_RUNS to point at e.g. ~/.cache/hgym so nothing lands in the tree.
RUNS_DIR = Path(os.environ.get("HGYM_SELFOPT_RUNS", PKG_DIR / "runs"))

# --- The environment under study -----------------------------------------------------

ENV_NAME = "automationbench"
# The public benchmark's six business domains (the 600 distributed tasks). The env's
# `public` alias expands to sales/marketing/operations/support/finance/hr.
ENV_DOMAIN = "public"
ENV_CONFIG = {"domain": ENV_DOMAIN}

# --- The train / held-out split (see split.py) ---------------------------------------

# Fixed seed for the deterministic, disjoint partition. Change it and you get a *different*
# (still disjoint) split — but keep it fixed across a study so held-out never bleeds in.
SPLIT_SEED = int(os.environ.get("SELFOPT_SPLIT_SEED", "20260726"))
# Fraction of the public tasks reserved as the private-set proxy (never trained on).
HELDOUT_FRAC = float(os.environ.get("SELFOPT_HELDOUT_FRAC", "0.2"))

# --- The model under study -----------------------------------------------------------

# Cell #1 studies **Claude Opus 5** driven by Claude Code. The `claude` CLI resolves the
# concrete model id from this alias; override with $SELFOPT_MODEL (e.g. a cheaper model for
# the smoke test, or whatever the CLI advertises).
MODEL = os.environ.get("SELFOPT_MODEL", "claude-opus-5")
# Reasoning effort. Low keeps the smoke cheap; the real study picks its own.
EFFORT = os.environ.get("SELFOPT_EFFORT", "low")

# --- Metrics sink --------------------------------------------------------------------

# LocalSink (JSONL + console) is the zero-setup default. Opt into W&B with SELFOPT_WANDB=1
# *and* a WANDB_API_KEY in the environment; otherwise it never touches the network.
USE_WANDB = os.environ.get("SELFOPT_WANDB", "").lower() in ("1", "true", "yes")
WANDB_PROJECT = os.environ.get("SELFOPT_WANDB_PROJECT", "hgym-selfopt-cell1")


def run_dir(run_id: str) -> Path:
    """The per-run artifact dir under RUNS_DIR (created on demand)."""
    d = RUNS_DIR / run_id
    d.mkdir(parents=True, exist_ok=True)
    return d
