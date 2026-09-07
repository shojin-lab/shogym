"""One admission bundle that actually verifies, for the tests that serve a dealt family.

A bank filled by the registered bars, a screen artifact and a review pack. The bank and its
population are real. The screen's score rows and the pack's render bytes are SYNTHETIC:
structurally valid, and standing in for a pilot nobody ran here and renders nobody read. They
are written rather than asserted because the production open path recomputes everything
mechanical about them, so a fixture that only recorded that the stages had happened would be
testing the claim, not the path.

It lives here rather than in one test module because two of them now open an environment on a
bundle, and a second copy of this would be a second answer to what a bundle that verifies is.
"""

from __future__ import annotations

import json
from pathlib import Path

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts import streams
from shogym.envs.receipts.generators.ledger import GENERATOR


def verified_bundle(room: Path, size: int = 2) -> Path:
    """Build a bundle of ``size`` admitted instances under ``room``, and return its directory."""
    bank, held = bank_mod.materialized(GENERATOR, streams.new_master_key(), size)
    outcomes = room / "screen.json"
    outcomes.write_text(json.dumps(screen_artifact()), encoding="utf-8")
    pack = review_pack(room, held, bank)
    built = bundle_mod.build(room / "bundles", GENERATOR, bank, outcomes, pack)
    assert bundle_mod.verify(built, GENERATOR).problems == ()
    return built.root


def screen_artifact(pairs: int = 40) -> dict:
    """A pilot run and the bars it is judged against, as one artifact.

    Three numbers say what was measured; they do not say what it was measured on or what it had
    to clear, so both travel with the rows.
    """
    return {
        "family": GENERATOR.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.9}
            for i in range(pairs)
        ],
        "min_room": 0.05, "min_ratio": 0.25, "min_pairs": 36,
        "floor": 0.0, "floor_rule": "drop",
        "candidates_screened": 1, "selection_note": "",
    }


def review_pack(room: Path, held: bank_mod.Population, bank: bank_mod.Bank) -> Path:
    """A pack that covers what the family declares, with artifacts of a plausible size.

    Every surface, every option of every axis, every filing shape, every row count the bank
    holds, and a counterfactual render. The bytes stand in for what a reviewer actually read;
    what is being exercised is the coverage rule, not the reading.
    """
    from shogym.envs.receipts.checks import FILING_CLASSES
    from shogym.envs.receipts.review import required_coverage

    coverage = required_coverage(
        GENERATOR, FILING_CLASSES,
        [i.a.n_rows for i in held.instances] + [i.b.n_rows for i in held.instances],
    )
    envelope_size = min(i.envelope.size for i in held.instances)
    folder = room / "renders"
    folder.mkdir(exist_ok=True)
    renders = []
    for index, (category, key) in enumerate(coverage.required):
        kind = "task" if category == "surface" else "cell"
        floor = 400 if kind == "task" else envelope_size
        artifact = folder / f"{index:03d}.txt"
        artifact.write_text("R" * (floor + 8), encoding="utf-8")
        renders.append({
            "category": category, "key": key, "kind": kind,
            "path": f"renders/{artifact.name}",
        })
    pack = room / "review-pack.json"
    pack.write_text(
        json.dumps({
            "reviewer": "test",
            "checklist": ["surface templates", "every option", "filing shapes"],
            "seeds": [0, 1],
            "family": GENERATOR.name,
            "bank": bank_mod.bank_identity(bank),
            "renders": renders,
        }),
        encoding="utf-8",
    )
    return pack


__all__ = ["review_pack", "screen_artifact", "verified_bundle"]
