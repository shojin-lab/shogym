"""``shogym receipts``: materialize a bank, freeze a bundle, draw from it, read the roster.

    shogym receipts materialize ledger --size 16
    shogym receipts draw ledger
    shogym receipts gate ledger
    shogym receipts check ledger
    shogym receipts screen ledger --outcomes pilot.json
    shogym receipts bundle ledger --screen pilot.json --review pack.json
    shogym receipts verify ledger
    shogym receipts list

A bank is what a family looks like while it is being worked on. A BUNDLE is what can
be dealt: one frozen directory holding the bank, its instances, the bars it was
filtered under, the code it was certified against, the room screen and the human's
review pack, addressed by the hash of its own manifest. `verify` recomputes all of it
and is the only thing that says a family is eligible.

`draw` is for a human. It prints one admitted instance from the bank: both sibling
task texts, and the three cells a fork could serve, rendered through the same
atomic path a run uses. A generator enters a release only after someone has read
one, and the review pack is many of these, not one.

There is no `--seed`. A free seed makes the gate universe and the review
cherry-pickable, and a live run must never accept a draw from outside the bank, so
the only thing that can be drawn is an ordinal the bank already holds.

Every command here is CONTROLLER-SIDE. They read the master key, the drawn
convention and the answer key. None of this is reachable from a lineage sandbox.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from shogym.envs.receipts import admission as admission_mod
from shogym.envs.receipts import bank as bank_mod
from shogym.envs.receipts import bundle as bundle_mod
from shogym.envs.receipts import streams
from shogym.envs.receipts.receipt_ast import GRADED, ORACLE, PLACEBO
from shogym.receipts.screen import REGISTERED_MIN_PAIRS
from shogym.envs.receipts.registry import (
    FIXTURES,
    GENRES,
    bank_path,
    bundle_dir,
    bundles,
    is_fixture,
    load_generator,
)

RULE = "-" * 78


def add_parser(sub: argparse._SubParsersAction) -> None:
    """Hang the `receipts` command group off the top-level parser."""
    parser = sub.add_parser("receipts", help="task families under a hidden convention")
    inner = parser.add_subparsers(dest="receipts_command", required=True)

    inner.add_parser("list", help="the genres, their banks, and their bundles")

    made = inner.add_parser("materialize", help="build and freeze a bank of gate passers")
    made.add_argument("name", help="a genre name")
    made.add_argument("--size", type=_count, default=16, help="instances to admit (default: 16)")
    made.add_argument(
        "--force", action="store_true", help="replace a bank that already exists"
    )

    gated = inner.add_parser(
        "gate", help="run the registered gate set per instance; nonzero on a fail"
    )
    gated.add_argument("name", help="a genre name or a gate vector")
    gated.add_argument(
        "--instances", type=_count, default=4, help="how many ordinals to gate (default: 4)"
    )
    gated.add_argument(
        "--min-arity", type=_count, default=admission_mod.SETTLED_MIN_ARITY,
        help="diagnostic only; moving it renames the result",
    )
    gated.add_argument(
        "--min-blocks", type=_count, default=admission_mod.SETTLED_MIN_BLOCKS,
        help="diagnostic only; moving it renames the result",
    )
    gated.add_argument(
        "--min-headroom", type=_unit,
        default=admission_mod.REGISTERED_MIN_HEADROOM,
        help="H's threshold (registered: %(default)s)",
    )

    checked = inner.add_parser("check", help="the named checks per instance; nonzero on a fail")
    checked.add_argument("name", help="a genre name or a gate vector")
    checked.add_argument(
        "--instances", type=_count, default=4, help="how many ordinals to check (default: 4)"
    )
    _thresholds(checked)

    screened = inner.add_parser(
        "screen", help="score a recorded room screen and print the verdict"
    )
    screened.add_argument("name", help="a genre name")
    screened.add_argument(
        "--outcomes", required=True,
        help="a screen artifact: model, task_seeds, pairs, and the bars it is judged "
             "against",
    )

    bundled = inner.add_parser(
        "bundle", help="freeze a bank, a screen and a review pack into one bundle"
    )
    bundled.add_argument("name", help="a genre name")
    bundled.add_argument("--screen", required=True, help="the screen artifact")
    bundled.add_argument("--review", required=True, help="the review pack that was read")
    # There is no --bank, for the same reason there is no --seed: a bundle is frozen
    # from the genre's own bank file and there is no argument that points it at
    # another. That is a command-line property and NOT a proof that nobody chose the
    # bank. The master key fixes the whole gate universe, and an operator can reroll
    # it with `materialize --force` as often as they like, can point the bank
    # directory somewhere else with SHOGYM_RECEIPTS_BANKS, and can write the five
    # fields of a bank record by hand: the record carries no count of rolls and no
    # provenance for its key. Resistance to operator selection is process here, not
    # tooling, and closing it takes append-only external provenance the v0 hash set
    # does not have.

    verified = inner.add_parser(
        "verify", help="recompute a bundle's admission evidence; nonzero when it fails"
    )
    verified.add_argument("name", help="a genre name")
    verified.add_argument(
        "--bundle", default=None,
        help="a bundle digest or directory (default: the genre's only bundle)",
    )

    drawn = inner.add_parser("draw", help="print one admitted instance for reading")
    drawn.add_argument("name", help="a genre name")
    drawn.add_argument(
        "--instance", type=int, default=None,
        help="which admitted ordinal to render (default: the bank's first)",
    )
    drawn.add_argument("--side", choices=("a", "b"), default="a", help="which sibling")
    drawn.add_argument(
        "--filing", choices=bank_mod.FILING_SHAPES, default="mixed",
        help="the registered filing shape the cells are rendered against",
    )
    drawn.add_argument(
        "--tasks-only", action="store_true", help="print the task texts and stop"
    )


def _count(value: str) -> int:
    """A positive count. Zero instances is not a run, and it must not exit clean."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("has to be one or more")
    return number


def _unit(value: str) -> float:
    """A finite threshold in [0, 1]. Every comparison against NaN is false."""
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise argparse.ArgumentTypeError("has to be a finite number between 0 and 1")
    return number


def _thresholds(parser: argparse.ArgumentParser) -> None:
    """The named checks' registered bars, and the flags that override them.

    They have values because someone chose them against a measured distribution.
    Moving one is diagnostic and goes nowhere durable: a bank record is five fields
    and carries no bars at all, and a bundle stores the registered set and refuses to
    verify against any other. What an override leaves is the printed line naming the
    bar it ran against.
    """
    parser.add_argument(
        "--max-copy-score", type=_unit,
        default=admission_mod.REGISTERED_MAX_COPY_SCORE,
        help="most the closed transfer family may earn on B: every relabelling of A's "
             "answers under every registered row permutation (registered: %(default)s)",
    )
    parser.add_argument(
        "--max-flip-score", type=_unit,
        default=admission_mod.REGISTERED_MAX_FLIP_SCORE,
        help="most that getting one axis wrong may earn on B (registered: %(default)s)",
    )
    parser.add_argument(
        "--min-leverage", type=_unit,
        default=admission_mod.REGISTERED_MIN_LEVERAGE,
        help="least that getting an axis right must be worth on B (registered: %(default)s)",
    )
    # No --min-headroom here. `check` runs the named checks and never runs gate H, so
    # the flag was accepted, ignored, and reported nothing: `gate` is where H's bar
    # lives and where moving it renames the result.


def _threshold_set(args: argparse.Namespace) -> admission_mod.Thresholds:
    """The named checks' bars. H's is not among them: `check` does not run H."""
    return admission_mod.Thresholds(
        max_copy_score=args.max_copy_score,
        max_flip_score=args.max_flip_score,
        min_leverage=args.min_leverage,
    )


def run(args: argparse.Namespace) -> int:
    """Dispatch one receipts command, and refuse rather than crash.

    An unknown genre, a missing bank, a malformed artifact and a screen taken on
    another family are all operator mistakes, and a command group whose entrypoint
    promises to exit nonzero when something fails should say what is wrong in a line
    rather than in a traceback. `draw` already did; the rest did not, so which of them
    printed a message depended on which one you ran.

    The five caught here are the shapes a bad input takes: a file that is not there, a
    name nothing maps to, a file that will not read, a number too large to be one, and
    a value that is not what its record says it is. A programmer error is none of
    those and still raises.
    """
    command = args.receipts_command
    handlers = {
        "list": lambda: _list(),
        "materialize": lambda: _materialize(args),
        "draw": lambda: _draw(args),
        "gate": lambda: _gate(args),
        "check": lambda: _check(args),
        "screen": lambda: _screen(args),
        "bundle": lambda: _bundle(args),
        "verify": lambda: _verify(args),
    }
    handler = handlers.get(command)
    if handler is None:
        raise ValueError(f"unknown receipts command {command!r}")
    try:
        return handler()
    except (OSError, KeyError, OverflowError, TypeError, ValueError) as exc:
        # A KeyError's own str is the repr of its argument, so it is unwrapped here
        # rather than printed as a quoted blob.
        print(exc.args[0] if isinstance(exc, KeyError) and exc.args else exc)
        return 1


def _screen(args: argparse.Namespace) -> int:
    """Score a recorded screen. The bars it is judged against are in the artifact.

    They are in the artifact because they are part of the claim. A screen scored under
    bars supplied at the command line is a screen whose verdict depends on who reran
    it, and the selection disclosure in particular is worth nothing if it can be left
    off and defaulted to a single candidate.
    """
    from shogym.receipts import ScreenRecord, read_payload

    try:
        record = ScreenRecord.from_payload(
            read_payload(Path(args.outcomes).read_text(encoding="utf-8"))
        )
    except (OSError, ValueError) as exc:
        print(f"the screen artifact is not a readable record: {exc}")
        return 1
    if record.min_pairs < REGISTERED_MIN_PAIRS:
        print(
            "the sample bar is %d against the registered %d, and a screen below the "
            "registered minimum is not deal evidence"
            % (record.min_pairs, REGISTERED_MIN_PAIRS)
        )
    result = record.result(args.name)
    print(RULE)
    print("\n".join(result.lines()))
    print(RULE)
    print("model %s, %d task seeds" % (record.run.model, len(record.run.task_seeds)))
    print(_bars(record))
    return 0 if result.verdict else 1


def _bars(record) -> str:
    """One line saying what a screen was judged against, and whether that is registered."""
    stated = "room %g, ratio %g, pairs %d" % (
        record.min_room, record.min_ratio, record.min_pairs
    )
    if record.registered:
        return "screen bars: %s (registered)" % stated
    return "screen bars: %s, OVERRIDDEN: %s" % (stated, "; ".join(record.overrides()))


def _bundle(args: argparse.Namespace) -> int:
    """Freeze one bank, one screen and one review pack into an addressed bundle."""
    path = bank_path(args.name)
    if not path.is_file():
        print(f"no frozen bank for {args.name!r} at {path}; materialize one first")
        return 1
    generator = load_generator(args.name)
    try:
        built = bundle_mod.build(
            bundle_dir(args.name),
            generator,
            bank_mod.load_bank(path),
            Path(args.screen),
            Path(args.review),
        )
    except (OSError, ValueError) as exc:
        print(f"this does not make a bundle: {exc}")
        return 1
    print(f"bundle {built.digest}")
    print(f"at {built.root}")
    print("%d files, %d bytes" % (len(built.files), sum(built.files.values())))
    # `build` returns nothing that did not verify, so the bars are read out of the
    # frozen file rather than by rebuilding the whole population a third time to
    # print one line.
    from shogym.receipts import ScreenRecord

    print(_bars(ScreenRecord.from_payload(built.payload(bundle_mod.SCREEN))))
    return 0


def _verify(args: argparse.Namespace) -> int:
    """Recompute a bundle's evidence and say what does not hold."""
    generator = load_generator(args.name)
    roots = (
        [bundle_dir(args.name) / args.bundle if not Path(args.bundle).exists()
         else Path(args.bundle)]
        if args.bundle
        else bundles(args.name)
    )
    if not roots:
        print(f"no bundles for {args.name!r} under {bundle_dir(args.name)}")
        return 1
    failures = 0
    for root in roots:
        checked = bundle_mod.verify_at(root, generator)
        print(RULE)
        print("bundle %s" % (checked.digest[:16] or root.name[:16]))
        if checked.verified:
            print(
                "VERIFIED: %d instances %s, %.0f%% of %d ordinals considered passed"
                % (len(checked.instances), list(checked.ordinals),
                   100.0 * checked.passing_fraction, checked.considered)
            )
            print("gate bars:   %s (registered)" % ", ".join(
                "%s=%g" % (k, v)
                for k, v in sorted(admission_mod.Thresholds().as_record().items())
            ))
            print(_bars(checked.screen))
        else:
            failures += 1
            print("NOT VERIFIED")
            for problem in checked.problems:
                print("   - " + problem)
    return 1 if failures else 0


def _instances(name: str, count: int):
    """Instances to report on: the bank's own when there is one, else the first few.

    A gate vector has no bank and never will, so it is walked directly. A family with
    a bank is walked through its recomputed population, because those are the
    instances that would actually be dealt and generating fresh ones instead would
    reopen a cherry-pickable universe.
    """
    from shogym.envs.receipts.protocol import draw

    generator = load_generator(name)
    if is_fixture(name):
        master = bytes(range(32))
        return generator, master, [draw(generator, master, n) for n in range(count)]
    path = bank_path(name)
    if not path.is_file():
        raise FileNotFoundError(
            f"no frozen bank for {name!r}. These commands report on the instances that "
            "would actually be dealt, and generating fresh ones instead would reopen a "
            f"cherry-pickable universe. Materialize a bank for {name!r} first."
        )
    held = bank_mod.load_bank(path)
    found = bank_mod.population(held, generator)
    return generator, held.master, list(found.instances)[:count]


def _gate(args: argparse.Namespace) -> int:
    from shogym.envs.receipts.observe import observe
    from shogym.receipts import gate as run_gate

    generator, master, instances = _instances(args.name, args.instances)
    failures = 0
    for instance in instances:
        result = run_gate(
            observe(generator, instance, "a"),
            min_arity=args.min_arity,
            min_blocks=args.min_blocks,
            min_headroom=args.min_headroom,
        )
        print(RULE)
        print("\n".join(result.lines()))
        failures += 0 if result.verdict else 1
    # ONE NAMING RULE, and it is the library's. Two conjunctions, one here and one in
    # `Thresholds`, are two answers to which rule a run published under.
    bars = admission_mod.Thresholds(
        min_arity=args.min_arity,
        min_blocks=args.min_blocks,
        min_headroom=args.min_headroom,
    )
    registered = admission_mod.Thresholds().as_record()
    moved = [
        "%s=%g against the registered %g" % (name, bars.as_record()[name], registered[name])
        for name in bars.moved_gate_constants
    ]
    version = bars.gate_version
    print(RULE)
    print(
        "%s: %d of %d instances rejected by %s"
        % (args.name, failures, len(instances), version)
    )
    if moved:
        print(
            "moved: %s, so this is a diagnostic run and its results may not fill a bank"
            % "; ".join(moved)
        )
    return 1 if failures else 0


def _check(args: argparse.Namespace) -> int:
    from shogym.envs.receipts.checks import run_checks

    generator, master, instances = _instances(args.name, args.instances)
    thresholds = _threshold_set(args)
    failures = 0
    for instance in instances:
        results = run_checks(
            generator, instance, master,
            max_copy_score=thresholds.max_copy_score,
            max_flip_score=thresholds.max_flip_score,
            min_leverage=thresholds.min_leverage,
            min_material_rows=thresholds.min_material_rows,
        )
        print(RULE)
        print("instance %s/%d" % (args.name, instance.ordinal))
        for result in results:
            print("   " + result.line())
        failures += 0 if all(r.passed for r in results) else 1
    print(RULE)
    print(
        "%s: %d of %d instances failed a named check"
        % (args.name, failures, len(instances))
    )
    return 1 if failures else 0


def _list() -> int:
    """The roster. Every quantity printed here was recomputed to print it.

    A development bank is mentioned and never described. Its stored fields are
    unverified by construction, and printing them beside a verified bundle would give
    an operator two descriptions of what looks like one family, with nothing on the
    line to say which one was checked.
    """
    print("%-10s %-28s %s" % ("genre", "description", "state"))
    for name in sorted(GENRES):
        generator = load_generator(name)
        print("%-10s %-28s %s" % (
            name, generator.genre,
            "a development bank is present, unverified and not dealable"
            if bank_path(name).is_file() else "no bank; nothing to bundle yet",
        ))
        held_bundles = bundles(name)
        if not held_bundles:
            print("%-10s %-28s NOT DEALABLE: no admission bundle" % ("", ""))
            continue
        for root in held_bundles:
            checked = bundle_mod.verify_at(root, generator)
            if checked.verified:
                print(
                    "%-10s %-28s %s DEALABLE, %d instances, %.0f%% of %d passed"
                    % ("", "", root.name[:16], len(checked.instances),
                       100.0 * checked.passing_fraction, checked.considered)
                )
                print("%-10s %-28s %s" % ("", "", _bars(checked.screen)))
            else:
                print(
                    "%-10s %-28s %s NOT DEALABLE: %s"
                    % ("", "", root.name[:16], "; ".join(checked.problems[:2]))
                )
    print()
    print("gate vectors (never dealt): " + ", ".join(sorted(FIXTURES)))
    return 0


def _materialize(args: argparse.Namespace) -> int:
    """Freeze a bank: a generator, a fresh key, and how many passers it holds."""
    if is_fixture(args.name):
        print(
            f"{args.name} is a gate vector, not a family; vectors are never dealt and have "
            "no bank"
        )
        return 1
    path = bank_path(args.name)
    if path.is_file() and not args.force:
        print(f"{path} already holds a frozen bank; pass --force to replace it")
        return 1
    generator = load_generator(args.name)
    try:
        # The population comes back from the fill rather than from a second walk of
        # the same ordinals: admission is what costs, and it has already run.
        built, found = bank_mod.materialized(
            generator, streams.new_master_key(), args.size
        )
    except ValueError as exc:
        print(f"this bank cannot be filled: {exc}")
        return 1
    written = bank_mod.save_bank(built, path)
    print(f"materialized {built.size} instances of {args.name} at {path}")
    print(f"bank digest {written[:16]}, ordinals {list(found.ordinals)}")
    # In full, because the review pack has to name it and a bundle refuses a pack read
    # from another bank.
    print(f"bank identity {bank_mod.bank_identity(built)}")
    print(
        "%.1f%% of %d ordinals considered passed %s"
        % (100.0 * found.passing_fraction, found.considered, bank_mod.GATE_LABEL)
    )
    print(
        "a bank is not dealable. Record a room screen and a review pack, then "
        f"`shogym receipts bundle {args.name} --screen ... --review ...`"
    )
    return 0


def _draw(args: argparse.Namespace) -> int:
    generator = load_generator(args.name)
    path = bank_path(args.name)
    if not path.is_file():
        print(f"no frozen bank for {args.name!r} at {path}; materialize one first")
        return 1
    held = bank_mod.load_bank(path)
    found = bank_mod.population(held, generator)
    ordinal = args.instance if args.instance is not None else found.ordinals[0]
    if ordinal not in found.ordinals:
        print(f"instance {ordinal} is not in this bank; it holds {list(found.ordinals)}")
        return 1
    instance = found.instance(ordinal)

    print(RULE)
    print(f"genre {instance.genre}, generator {instance.generator}, instance {ordinal}")
    print("drawn convention: " + ", ".join("%s=%s" % kv for kv in instance.convention.items()))
    print(
        "commitment %s"
        % bank_mod.commitment(held.master, ordinal, instance.convention)[:16]
    )
    print(RULE)
    for side in ("a", "b"):
        task = instance.side(side)
        print()
        print(
            "TASK %s   reference %s   surface %s   %d rows"
            % (task.label, task.task_id, task.surface, task.n_rows)
        )
        print()
        print(task.text)
    if args.tasks_only:
        return 0

    raw = bank_mod.review_filing(generator, instance, args.side, args.filing, held.master)
    fork = bank_mod.render_fork(generator, instance, args.side, raw)
    print(RULE)
    print(
        "the three cells for task %s, rendered against a %s filing"
        % (instance.side(args.side).label, args.filing)
    )
    print(
        "component score %.6f, %s%s"
        % (
            fork.component_score,
            bank_mod.outcome_summary(fork.outcomes),
            ", no filing (%s)" % bank_mod.no_filing_reason(fork.canonical)
            if bank_mod.no_filing_reason(fork.canonical)
            else "",
        )
    )
    for kind in (GRADED, PLACEBO, ORACLE):
        payload = fork.agent_bytes(kind)
        print()
        print(RULE)
        print("%s cell, %d bytes, digest %s" % (kind.upper(), len(payload),
                                                fork.digests[kind][:16]))
        print(RULE)
        print(payload.decode("ascii"))
    print(RULE)
    sizes = {k: len(fork.agent_bytes(k)) for k in (GRADED, PLACEBO, ORACLE)}
    print("envelope %d, cells %s" % (instance.envelope.size, sizes))
    print("all three match the envelope: %s" % (set(sizes.values()) == {instance.envelope.size}))
    return 0


__all__ = ["add_parser", "run"]
