"""One admission bundle that actually verifies, for the tests that serve a dealt family.

A bank filled by the registered bars, a screen artifact and a review pack. The bank and its
population are real. The screen's score rows and the pack's render bytes are SYNTHETIC:
structurally valid, and standing in for a pilot nobody ran here and renders nobody read. They
are written rather than asserted because the production open path recomputes everything
mechanical about them, so a fixture that only recorded that the stages had happened would be
testing the claim, not the path.

It lives here rather than in one test module because two of them now open an environment on a
bundle, and a second copy of this would be a second answer to what a bundle that verifies is.

BUILDING ONE IS THE EXPENSIVE THING THIS SUITE DOES, so it is built once. Filling the bank walks
admission over every ordinal it considers, and the build walks it again to record the instances and
a third time to verify what it just wrote. Seven modules each wanted a bundle of the same shape, so
those three walks ran seven times to produce seven bundles that differed only in the key they were
frozen under, and no test read the key. :func:`verified_template` runs them once per process and
:func:`private_bundle` hands out copies, which is the whole saving.

WHAT MAY BE SHARED AND WHAT MAY NOT. The template is frozen input: a bank, a population and a
verification, none of which any test writes to. Everything a test forks, seals, captures or edits
has to be its own, and for a bundle that is not a matter of taste. The environment files its forks
at ``root.parent / "forks"`` and its seals underneath them, so two tests over one root would file
into one directory and one module's filing could answer another module's idempotency test. A copy
under a private parent is therefore the unit that is handed out, never the template itself, and
the template is left read-only so that a caller who takes it anyway fails where the mistake is.
Each copy is its own absolute path, so each still pays its own first open and its own real
verification: what the sharing removes is the building, not the checking.
"""

from __future__ import annotations

import atexit
import hashlib
import json
import shutil
import stat
import tempfile
from pathlib import Path

from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts import streams
from shogym.envs.receipts.generators.ledger import GENERATOR

#: The template bundle for each size this process has been asked for. One per process, which under
#: CI is one per shard job, because a shard is one pytest process and a session fixture would be
#: no wider than this.
_TEMPLATES: dict[int, Path] = {}

#: The key the one-instance template is filled under. The band on the ideal level is read at a
#: bank's mean, so a bank of one reads it at its only instance, and a single ledger instance can
#: stand under the 0.05 bar: 3 of 160 fresh keys drew a first instance with no more room than that,
#: and the bank then refused to fill before any test that asked for it had started. Under this key
#: ordinal 0 is admitted with an ideal level of 0.8254 standing 0.0869 above its lookup floor, so
#: the bank fills on every run, and a change that moves it under the bar fails on every run rather
#: than on about one in fifty.
ONE_INSTANCE_MASTER = bytes(range(96, 128))

#: The key the two-instance template is filled under. Under it admission admits ordinals 0 and 1,
#: so every walk over the bank is two reports, and the bank of two clears the registered band read
#: at its mean. Under a fresh key on every run, how many reports a walk took moved with the key,
#: and so did what every test that builds or opens a copy of this template cost.
TWO_INSTANCE_MASTER = hashlib.sha256(b"receipts-test-key:a").digest()

#: The key each template is filled under, by its size. Each was chosen by a real fill at that size,
#: so a size with no key here has no template until one is.
TEMPLATE_MASTERS = {1: ONE_INSTANCE_MASTER, 2: TWO_INSTANCE_MASTER}


def verified_bundle(room: Path, size: int = 2, master: bytes | None = None) -> Path:
    """Build a bundle of ``size`` admitted instances under ``room``, and return its directory.

    The bank is filled under ``master``, or under a fresh key when none is named.
    """
    key = master if master is not None else streams.new_master_key()
    bank, held = bank_mod.materialized(GENERATOR, key, size)
    outcomes = room / "screen.json"
    outcomes.write_text(json.dumps(screen_artifact()), encoding="utf-8")
    pack = review_pack(room, held, bank)
    # The build verifies what it wrote before it returns, and removes it and raises when it does
    # not verify, so a second verification here would be the same walk for the same answer.
    built = bundle_mod.build(room / "bundles", GENERATOR, bank, outcomes, pack)
    return built.root


def verified_template(size: int = 2) -> Path:
    """The one bundle of this size this process builds, read-only and shared.

    Built under the system temporary directory rather than under a test's own, because it outlives
    every test that reads it and belongs to none of them. It is removed when the process ends. It
    is filled under its size's key in :data:`TEMPLATE_MASTERS`.
    """
    known = _TEMPLATES.get(size)
    if known is not None:
        return known
    master = TEMPLATE_MASTERS.get(size)
    if master is None:
        raise ValueError(
            f"no key is chosen for a template of {size} instances; fill one for real at that "
            "size and add it to TEMPLATE_MASTERS"
        )
    room = Path(tempfile.mkdtemp(prefix=f"shogym-receipts-template-{size}-"))
    atexit.register(_discard, room)
    built = verified_bundle(room, size=size, master=master)
    _mode(built, writable=False)
    _TEMPLATES[size] = built
    return built


def private_bundle(room: Path, size: int = 2) -> Path:
    """A writable copy of that template under ``room``, which is this caller's alone.

    The copy keeps the template's name, because a bundle is addressed by its own contents and a
    directory called anything else is not the bundle it holds. ``room`` is what has to differ:
    it is where the forks and seals of whoever opens this copy will be written.
    """
    source = verified_template(size)
    room.mkdir(parents=True, exist_ok=True)
    private = room / source.name
    shutil.copytree(source, private)
    _mode(private, writable=True)
    return private


def _mode(root: Path, *, writable: bool) -> None:
    """Set the whole tree readable, and writable or not, from the bottom up.

    A directory has to keep its execute bit either way or nothing under it can be reached, and the
    read-only pass runs deepest first so that a directory is not sealed before what it holds.
    """
    files = stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH
    folders = files | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    if writable:
        files |= stat.S_IWUSR
        folders |= stat.S_IWUSR
    for item in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        item.chmod(folders if item.is_dir() else files)
    root.chmod(folders)


def _discard(room: Path) -> None:
    """Remove a template's room at exit, unsealing it first so that removal can happen."""
    if room.is_dir():
        _mode(room, writable=True)
    shutil.rmtree(room, ignore_errors=True)


def screen_artifact(pairs: int = 40) -> dict:
    """A pilot run and the bars it is judged against, as one artifact.

    Three numbers say what was measured; they do not say what it was measured on or what it had
    to clear, so both travel with the rows. It is made under the registered bars and names them,
    as a record a pilot writes now does.
    """
    return {
        "family": GENERATOR.name,
        "model": "a scripted policy",
        "task_seeds": [str(i) for i in range(pairs)],
        "pairs": [
            {"instance": f"task-{i:02d}", "filing": f"filing-{i:02d}",
             "placebo": 0.4, "graded": 0.6, "oracle": 0.95, "ideal": 0.82}
            for i in range(pairs)
        ],
        "bars": "receipts-screen-v2", "min_pairs": 36, "min_learning_gap": 0.10,
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


__all__ = [
    "ONE_INSTANCE_MASTER",
    "TEMPLATE_MASTERS",
    "TWO_INSTANCE_MASTER",
    "private_bundle",
    "review_pack",
    "screen_artifact",
    "verified_bundle",
    "verified_template",
]
