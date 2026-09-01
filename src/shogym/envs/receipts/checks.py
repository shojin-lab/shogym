"""The named checks: siblinghood, the copy screen, and the artifact invariants.

Gates R, S and H ask what one receipt can tell an agent. These ask the questions
that are not about the receipt at all, and each is named separately from the gates
because failing one means something different.

  exercise     every axis is exercised in A's receipt at min(3, arity) blocks. This
               is NOT gate R: it is asked of every axis at its own arity, so a
               binary axis passes at two blocks and is never asked for three. An
               axis nothing on A's receipt responds to is an axis the link cannot
               teach.
  materiality  every axis is material in B: under the drawn convention, changing
               the option changes B's correct answers on some rows. An axis A
               resolves and B is inert on carries nothing at this link, because the
               thing the agent would have learnt changes no answer on the task it
               is scored on next.
  copy         a registered family of low-complexity maps from A's answers to a B
               filing is enumerated, each family is optimized over its own
               parameter, and the best must sit below a registered threshold, and
               every axis must carry positive B leverage. The screen's scope is the
               enumerated maps. It does not claim to exclude every derivation,
               because no extensional test can separate "infer the convention, then
               solve B" from another program computing the same function. The
               enumerated family is CLOSED under composition, and the bar and the
               family are one registration: NO_INDUCTION_MAPS says what is in it.
  fixation     the instance is materialized and hashed before the link's first
               execution, and every rebuild is byte-identical, so every branch of a
               fork receives the same B.
  envelope     on BOTH siblings, over the whole convention support: the placebo's
               bytes do not move,
               the graded cell's bytes do not move outside the registered slots,
               row alignment is identical between the two, and all three cells
               total the envelope.
  graded       on BOTH siblings, every graded row says what the scorer said, under
               every convention: the verdict is the row's matched bit and the
               correction is that row's own answer. A licensed value chosen for what
               it encodes is still a channel, and this is what closes it.
  placebo      on BOTH siblings, the placebo prints its committed tokens under every
               registered filing
               class and every convention, checked on what it actually rendered
               rather than on what it was handed.
  neutral      no token the placebo prints reads as a verdict or as a legal answer
               on either side, checked on the tokens the committed stream actually
               drew rather than on the alphabet it drew them from.
  oracle       the oracle states the rule that was drawn, and reads back to it, for
               every convention in the sampler's support. It is the denominator of
               the room the screen measures, so a false one moves the measurement
               rather than merely failing to teach.
  lint         no option token appears in either task text.
  invariance   every model-visible pre-feedback artifact is byte-identical under
               every convention, so possession of the surface teaches nothing about
               the draw.

THRESHOLDS ARE ARGUMENTS. What a family must show is the maintainer's call, and a
default buried here would quietly become that call.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Callable, Iterable, Mapping, Sequence

from shogym.envs.receipts.observe import canonical_filing_text, observe
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse_body as parse_oracle_body
from shogym.envs.receipts.protocol import (
    Generator,
    Instance,
    Task,
    conventions,
    draw,
    option_mentions,
    support_of,
)
from shogym.envs.receipts.receipt_ast import (
    GRADED,
    PLACEBO,
    frozen_envelope,
    mask_slots,
    serialize,
    slot_ranges,
)
from shogym.envs.receipts.oracle import render as oracle_render
from shogym.envs.receipts.render import frozen_template, judge_cells
from shogym.envs.receipts.render import oracle_difference
from shogym.receipts import resolution_blocks

#: The maps the copy screen enumerates. Each takes A's correct answers and produces
#: a filing for B without inducing anything: they are the ways a lazy agent could
#: try to turn one task's answers into the other's.
COPY_MAPS = ("identity", "permutation", "relabel", "closure", "option_flip")


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    detail: str

    def line(self) -> str:
        return "%-11s %-4s  %s" % (self.name, "PASS" if self.passed else "FAIL", self.detail)


# --------------------------------------------------------------------------
# siblinghood
# --------------------------------------------------------------------------


def check_exercise(generator: Generator, instance: Instance) -> CheckResult:
    """Every axis exercised in A's receipt at min(3, arity) blocks."""
    obs = observe(generator, instance, "a")
    index = obs.space.index()
    arity = obs.space.arity
    blocks = {a: resolution_blocks(obs, a, index) for a in obs.space.axes}
    want = {a: min(3, arity[a]) for a in obs.space.axes}
    short = [a for a in obs.space.axes if blocks[a] < want[a]]
    detail = ", ".join(f"{a} {blocks[a]} of {want[a]}" for a in obs.space.axes)
    if short:
        return CheckResult(
            "exercise", False,
            f"A's receipt does not exercise {', '.join(short)} ({detail})",
        )
    return CheckResult("exercise", True, f"blocks against the bar: {detail}")


def axis_materiality(
    generator: Generator, instance: Instance
) -> dict[str, tuple[int, float]]:
    """Per axis, the fewest and mean rows of B that move when the option changes."""
    base = generator.key_for(instance.b.table, instance.convention)
    out: dict[str, tuple[int, float]] = {}
    for axis in generator.AXES:
        moved: list[int] = []
        for option in axis.options:
            if option == instance.convention[axis.name]:
                continue
            alt = dict(instance.convention)
            alt[axis.name] = option
            other = generator.key_for(instance.b.table, MappingProxyType(alt))
            moved.append(sum(1 for x, y in zip(base, other) if x != y))
        out[axis.name] = (min(moved), sum(moved) / float(len(moved))) if moved else (0, 0.0)
    return out


def check_materiality(
    generator: Generator, instance: Instance, min_rows: int = 1
) -> CheckResult:
    """Every axis has to move at least `min_rows` of B's answers, on every option."""
    moved = axis_materiality(generator, instance)
    inert = [a for a, (low, _) in moved.items() if low < min_rows]
    detail = ", ".join(f"{a} {low}/{mean:.1f}" for a, (low, mean) in moved.items())
    if inert:
        return CheckResult(
            "materiality", False,
            f"B is inert on {', '.join(inert)} (fewest and mean rows moved: {detail})",
        )
    return CheckResult("materiality", True, f"fewest and mean rows of B moved: {detail}")


# --------------------------------------------------------------------------
# the copy screen
# --------------------------------------------------------------------------


def _fit(values: Sequence[str], width: int) -> list[str]:
    """A filing for B of the right length, whatever A's row count was."""
    padded = list(values) + [""] * width
    return padded[:width]


def _distinct(filings: Iterable[Sequence[str]]) -> list[list[str]]:
    """The same filings with the repeats dropped, in the order they first appeared.

    A closed family names one filing many ways: the row move that undoes a rotation
    of a symmetric filing, the two token maps that agree on the answers A actually
    filed. Scoring each of those again costs a parse and decides nothing, and the
    maximum is over the set rather than the enumeration.
    """
    seen: dict[tuple[str, ...], list[str]] = {}
    for filing in filings:
        seen.setdefault(tuple(filing), list(filing))
    return list(seen.values())


def _permutations(values: Sequence[str], width: int) -> list[list[str]]:
    """The registered row moves, as the CLOSED family the two generators produce.

    The generators are the rotation and the reversal, and a family that listed only
    those was not closed: reversing and THEN rotating is a composition of registered
    moves that no listed member equals, and on a real draw one of them scored above
    the bar while the reported maximum sat under it. Rotation and reversal generate
    the dihedral group, so the closed family is every rotation of the sequence and
    every rotation of the reversed sequence: 2n moves for n rows, 48 at the ledger's
    24. Closing it here is what makes the composition with the relabels closed too,
    because a product of a closed family with a commuting one is closed.

    The moves are positional, so they are taken on the FITTED sequence: the filing
    they rearrange is the one B would receive, whatever A's row count was.
    """
    fitted = _fit(values, width)
    reversed_fitted = list(reversed(fitted))
    out: list[list[str]] = []
    for base in (fitted, reversed_fitted):
        for shift in range(max(len(base), 1)):
            out.append(list(base[shift:]) + list(base[:shift]))
    return out


def _token_generators(
    values: Sequence[str],
    source_ranks: Sequence[str],
    target_ranks: Sequence[str],
) -> list[dict[str, str]]:
    """The registered token dictionaries themselves, before anything is closed.

    EVERY TARGET SIDE IS A PUBLISHED VOCABULARY. The RANK map takes the two tasks'
    own published answer orders and sends first to first: it is the cheapest transfer
    there is, because both orders are printed in the two task texts and reading them
    off needs no induction at all. Its rotations come with it. The LEXICAL map is the
    same construction over the published vocabularies sorted alphabetically, which is
    what a reader with no sense of the order would write. The FILED map narrows the
    source to the answers the agent actually filed on A, which it knows without
    inducing anything, and sends them into the same published target order.

    The target side is never the tokens B's drawn key happened to realize. A map into
    the realized target prices a transfer nobody could perform: producing it means
    already knowing what the hidden draw did to B, which is the thing the screen
    exists to bound. It is also not measuring the surface pair, since it moves with
    the draw, and the bar was read against the numbers it inflated.

    These are the GENERATORS. What the screen is read against is what they generate,
    which `_token_maps` computes.
    """
    out: list[dict[str, str]] = []
    for source, into in (
        (list(source_ranks), list(target_ranks)),
        (sorted(set(source_ranks)), sorted(set(target_ranks))),
        (sorted(set(values)), sorted(set(target_ranks))),
    ):
        if not source or not into:
            continue
        for shift in range(len(into)):
            out.append({s: into[(i + shift) % len(into)] for i, s in enumerate(source)})
    return out


def _token_maps(
    values: Sequence[str],
    source_ranks: Sequence[str],
    target_ranks: Sequence[str],
) -> list[dict[str, str]]:
    """The registered token dictionaries CLOSED under composition, identity first.

    A LIST OF DICTIONARIES IS NOT THE FAMILY, and the difference has a price. Each
    registered dictionary is applied by leaving anything outside it alone, so it is a
    total function on the finite set of tokens the dictionaries and the filing
    mention, and two of them compose into a third total function on that same set. A
    rank map that carries A's bands into B's while leaving an untouched token alone,
    followed by a filed map that carries that token onward while leaving the already
    translated bands alone, is a map an agent builds from the same two published
    dictionaries and no more. Listing the dictionaries and not their compositions left
    such a map out, and on a real draw it scored 13 of 24 where the reported maximum
    was 11 of 24, so the bar admitted it.

    So this closes them, generically rather than by hand. Compositions are taken until
    no new map appears, which terminates because there are finitely many functions
    from a finite token set to itself. The result is a monoid: the identity is in it,
    and it is closed, so nothing an agent can build by chaining registered
    dictionaries is outside what the screen prices.
    """
    universe = sorted(set(values) | set(source_ranks) | set(target_ranks))
    if not universe:
        return [{}]
    position = {token: n for n, token in enumerate(universe)}

    def total(table: Mapping[str, str]) -> tuple[str, ...]:
        """One dictionary as its images over the whole token set, in one order."""
        return tuple(table.get(token, token) for token in universe)

    def after(first: tuple[str, ...], second: tuple[str, ...]) -> tuple[str, ...]:
        """`first` and then `second`, which is a map on the same token set."""
        return tuple(second[position[image]] for image in first)

    identity = tuple(universe)
    found = [identity]
    known = {identity}
    for table in _token_generators(values, source_ranks, target_ranks):
        made = total(table)
        if made not in known:
            known.add(made)
            found.append(made)
    frontier = list(found)
    while frontier:
        fresh: list[tuple[str, ...]] = []
        against = list(found)
        for one in frontier:
            for other in against:
                for made in (after(one, other), after(other, one)):
                    if made not in known:
                        known.add(made)
                        found.append(made)
                        fresh.append(made)
        frontier = fresh
    return [dict(zip(universe, image)) for image in found]


def _relabellings(
    values: Sequence[str],
    source_ranks: Sequence[str],
    target_ranks: Sequence[str],
    width: int,
) -> list[list[str]]:
    """Every filing the closed token family can make of A's answers, deduplicated.

    The identity is among the maps, so the untouched filing leads the list and
    composing this with the row moves produces those moves themselves.
    """
    seen: dict[tuple[str, ...], list[str]] = {}
    for table in _token_maps(values, source_ranks, target_ranks):
        filing = _fit([table.get(value, value) for value in values], width)
        seen.setdefault(tuple(filing), filing)
    return list(seen.values())


def copy_map_filings(
    generator: Generator, instance: Instance
) -> dict[str, list[list[str]]]:
    """Per registered map, every filing for B that map can produce.

    Each family is enumerated rather than sampled, so the screen optimizes over the
    whole map and not over one arbitrary member of it. A screen that tried the first
    rotation and reported its score would be reporting the wrong number.
    """
    a_key = list(instance.a.key)
    width = len(instance.b.key)
    relabels = _relabellings(
        a_key,
        generator.answer_ranks(instance.a.table),
        generator.answer_ranks(instance.b.table),
        width,
    )
    out: dict[str, list[list[str]]] = {
        "identity": [_fit(a_key, width)],
        "permutation": _permutations(a_key, width),
        "relabel": relabels,
        # THE REGISTERED FAMILY: the product of two families that are each closed
        # under composition, which is therefore closed itself. `relabels` is every
        # filing the closed token monoid makes of A's answers, and `_permutations` is
        # the closed group the row moves generate. Token maps act on answer values and
        # row moves on positions, so the two commute, and a product of commuting
        # closed families is closed. It contains the identity, every row move alone
        # and every relabel alone, because both families contain their identity, so
        # the three maps above are sub-families of it and are reported for diagnosis
        # rather than barred separately.
        "closure": _distinct(
            filing
            for relabelled in relabels
            for filing in _permutations(relabelled, width)
        ),
    }
    flips: list[list[str]] = []
    for axis in generator.AXES:
        for option in axis.options:
            if option == instance.convention[axis.name]:
                continue
            alt = dict(instance.convention)
            alt[axis.name] = option
            flips.append(
                _fit(list(generator.key_for(instance.b.table, MappingProxyType(alt))), width)
            )
    out["option_flip"] = flips or [_fit(a_key, width)]
    return out


def copy_scores(generator: Generator, instance: Instance) -> dict[str, float]:
    """Each registered map's BEST component score on B, optimized over the family."""
    identifiers = generator.row_identifiers(instance.b.table)

    def score_of(values: Sequence[str]) -> float:
        raw = "\n".join(f"{i},{v}" for i, v in zip(identifiers, values))
        canonical = generator.parse_and_canonicalize(instance.b, raw)
        return generator.score(instance.b, canonical)[0]

    return {
        name: max((score_of(v) for v in candidates), default=0.0)
        for name, candidates in copy_map_filings(generator, instance).items()
    }


def axis_leverage(generator: Generator, instance: Instance) -> dict[str, float]:
    """Per axis, what getting it right is worth on B against getting it wrong.

    The drawn convention's B score minus the mean B score of the same convention with
    that one axis changed. An axis with no leverage is an axis the link cannot be
    scored on, whatever A's receipt said about it.
    """
    identifiers = generator.row_identifiers(instance.b.table)

    def score_of(key: Sequence[str]) -> float:
        raw = "\n".join(f"{i},{v}" for i, v in zip(identifiers, key))
        canonical = generator.parse_and_canonicalize(instance.b, raw)
        return generator.score(instance.b, canonical)[0]

    right = score_of(instance.b.key)
    out: dict[str, float] = {}
    for axis in generator.AXES:
        wrong: list[float] = []
        for option in axis.options:
            if option == instance.convention[axis.name]:
                continue
            alt = dict(instance.convention)
            alt[axis.name] = option
            wrong.append(
                score_of(generator.key_for(instance.b.table, MappingProxyType(alt)))
            )
        out[axis.name] = right - (sum(wrong) / len(wrong)) if wrong else 0.0
    return out


#: THE REGISTERED NO-INDUCTION FAMILY, and the bar is read against exactly it.
#:
#: A map is in it when an agent can build the filing from A's receipt and B's surface
#: alone: what it filed on A, the two published answer orders, and B's printed rows.
#: `closure` is that family, and the three names before it are sub-families of it,
#: reported so a failure says which cheap map did it rather than only that one did.
#:
#: The family is CLOSED under composition, which is why it is the family and not a
#: list. It has two kinds of registered move, each closed on its own before they are
#: combined. The ROW MOVES are generated by the rotation and the reversal, so their
#: closure is the dihedral group, every rotation of the sequence AND of its reversal,
#: 48 moves at the ledger's 24 rows. The TOKEN MAPS are the registered dictionaries
#: and every composition of them, computed generically rather than listed, which
#: terminates because they are functions on a finite token set. Token maps act on
#: answer values and row moves on positions, so the two commute and the product of two
#: commuting closed families is closed.
#:
#: A LIST OF GENERATORS IS NOT THE FAMILY, and this cost twice. Listing the rotation
#: and the reversal without their compositions left out every reversal followed by a
#: rotation, and one of those scored 0.5417 on a draw whose reported maximum was
#: 0.5000. Listing the token dictionaries without theirs left out a rank map followed
#: by a filed map, which scored 0.5417 where the maximum reported 0.4583. Both draws
#: were admitted. Nothing about either omitted move is harder than the moves that were
#: listed; the enumerations were simply not the closures they claimed to be, which
#: product size and sub-family containment cannot see and a composition test can.
#:
#: `option_flip` is outside it, because producing that filing means having induced
#: every axis but one, which is not transfer.
NO_INDUCTION_MAPS = ("identity", "permutation", "relabel", "closure")

#: Sub-families of the closure, enumerated and printed for diagnosis. They cannot
#: exceed it, so they add nothing to the maximum the bar is read against.
REPORTED_MAPS = ("identity", "permutation", "relabel")


def check_copy(
    generator: Generator,
    instance: Instance,
    max_copy_score: float,
    max_flip_score: float,
    min_leverage: float,
) -> CheckResult:
    """What transfer can earn on B without inducing the rule, and what a near miss earns.

    Two numbers, thresholded separately, because they answer different questions.
    The NO-INDUCTION best is what an agent gets for reusing A's answers: identity,
    any registered row permutation, any registered token relabel. If that is high,
    B is not a different task and the link measures nothing.

    The FLIP best is what an agent gets for inducing every axis but one and being
    wrong about that one. It is a near miss, not a copy, so bounding it is a
    demand about how discriminating B's score is rather than about transfer: it
    asks that getting an axis wrong actually costs something. A single threshold
    over both would silently make the stricter of the two questions the only one
    asked.

    The no-induction maximum is taken over the CLOSED family, so the number the bar is
    read against is a number some map in the family actually reaches. Its sub-families
    are printed beside it, because "the rank relabel alone earns this" is a different
    repair from "only relabelling and then permuting earns it". See NO_INDUCTION_MAPS.
    """
    scores = copy_scores(generator, instance)
    leverage = axis_leverage(generator, instance)
    plain = {k: v for k, v in scores.items() if k in NO_INDUCTION_MAPS}
    best_plain = max(plain, key=lambda k: plain[k]) if plain else ""
    plain_score = plain.get(best_plain, 0.0)
    flip_score = scores.get("option_flip", 0.0)
    weak = [a for a, value in leverage.items() if value < min_leverage]
    detail = (
        "no-induction best %s at %.4f (bar %.4f, over the closed family); alone: %s; "
        "one axis wrong scores %.4f (bar %.4f); leverage %s"
        % (
            best_plain,
            plain_score,
            max_copy_score,
            ", ".join(f"{name} {scores[name]:.4f}" for name in REPORTED_MAPS),
            flip_score,
            max_flip_score,
            ", ".join(f"{a} {v:+.4f}" for a, v in leverage.items()),
        )
    )
    if plain_score > max_copy_score:
        return CheckResult(
            "copy", False,
            f"reusing A's answers earns {plain_score:.4f} on B through {best_plain}, over "
            f"the registered {max_copy_score:.4f}; {detail}",
        )
    if flip_score > max_flip_score:
        return CheckResult(
            "copy", False,
            f"getting one axis wrong still earns {flip_score:.4f} on B, over the registered "
            f"{max_flip_score:.4f}; {detail}",
        )
    if weak:
        return CheckResult(
            "copy", False,
            f"B pays too little for {', '.join(weak)} (bar {min_leverage:.4f}); {detail}",
        )
    return CheckResult("copy", True, detail)


# --------------------------------------------------------------------------
# the artifact invariants
# --------------------------------------------------------------------------


def check_fixation(generator: Generator, instance: Instance, master: bytes) -> CheckResult:
    """The instance rebuilds byte-identically, so every branch of a fork gets one B.

    THE WHOLE INSTANCE, by its digest, not a list of fields anybody thought to name.
    The bank's own record of what is committed covers both task identifiers, both
    surfaces, both keys, both texts, the drawn convention, the envelope schema, and
    the committed filler and neutral tokens. Comparing four of those left A's key and
    the whole envelope unchecked, and the envelope is what pads and fills all three
    cells: a generator whose filler was not reproducible would pass this and still
    hand two branches of one fork different bytes, which is the one thing this check
    exists to stop.
    """
    from shogym.envs.receipts.bank import instance_digest

    again = draw(generator, master, instance.ordinal)
    if instance_digest(again, generator) == instance_digest(instance, generator):
        return CheckResult(
            "fixation", True,
            f"the whole instance rebuilds to one digest at ordinal {instance.ordinal}",
        )
    # It differs. Name where, so the failure is actionable rather than a hash mismatch.
    for what, got, want in (
        ("the sibling task's rows", generator.table_record(again.b.table),
         generator.table_record(instance.b.table)),
        ("A's rows", generator.table_record(again.a.table),
         generator.table_record(instance.a.table)),
        ("the sibling task's text", again.b.text, instance.b.text),
        ("the sibling task's answers", again.b.key, instance.b.key),
        ("A's text", again.a.text, instance.a.text),
        ("A's answers", again.a.key, instance.a.key),
        ("the drawn convention", dict(again.convention), dict(instance.convention)),
        ("the envelope", again.envelope, instance.envelope),
    ):
        if got != want:
            return CheckResult(
                "fixation", False,
                f"{what} does not rebuild identically, so two branches of one fork would "
                "not receive the same bytes",
            )
    return CheckResult(
        "fixation", False,
        "the instance does not rebuild to the digest the bank would commit to",
    )


def check_envelope(
    generator: Generator, instance: Instance, side: str = "a"
) -> CheckResult:
    """The envelope, asserted over the whole convention support on one filing.

    Both cells are rendered inside the loop from a task retasked to that convention,
    so the placebo's invariance is measured rather than assumed, and the identifiers
    it prints are checked against the public row identities.
    """
    task = instance.side(side)
    envelope = frozen_envelope(instance.envelope)
    canonical = generator.parse_and_canonicalize(
        task, canonical_filing_text(generator, task)
    )
    reference_placebo: bytes | None = None
    reference_mask: bytes | None = None
    alignment: tuple[str, ...] | None = None
    seen = 0
    for convention in conventions(generator.AXES):
        truth = generator.key_for(task.table, convention)
        # Retask for this convention and render BOTH cells from it. Rendering the
        # placebo once outside the loop would check that one render is stable, which
        # is not the question: the question is whether the renderer produces the same
        # bytes when the whole task is retasked under a different rule.
        retasked = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text=task.text, key=tuple(truth),
        )
        graded_ast = generator.render_receipt(retasked, canonical, truth)
        ranges = slot_ranges(graded_ast, envelope)
        graded = serialize(graded_ast, envelope)
        placebo_ast = generator.render_placebo(retasked.public(), canonical, envelope)
        placebo = serialize(placebo_ast, envelope)
        if reference_placebo is None:
            reference_placebo = placebo
        elif placebo != reference_placebo:
            return CheckResult(
                "envelope", False,
                "the placebo's bytes move when the task is retasked under another "
                "convention, so what admission certifies as inert is carrying the rule",
            )
        if tuple(r.identifier for r in placebo_ast.rows) != tuple(
            generator.row_identifiers(task.table)
        ):
            return CheckResult(
                "envelope", False,
                "the placebo's row identifiers are not the public row identities",
            )
        oracle = serialize(
            generator.render_oracle(task.task_id, convention, task.n_rows), envelope
        )
        seen += 1
        for kind, payload in (("graded", graded), ("placebo", placebo), ("oracle", oracle)):
            if len(payload) != envelope.size:
                return CheckResult(
                    "envelope", False,
                    f"the {kind} cell is {len(payload)} bytes against an envelope of "
                    f"{envelope.size}",
                )
        masked = mask_slots(graded, ranges)
        if reference_mask is None:
            reference_mask = masked
            alignment = tuple(row.identifier for row in graded_ast.rows)
        elif masked != reference_mask:
            return CheckResult(
                "envelope", False,
                "the graded cell moves outside the registered slots when the convention "
                "moves, so a byte the slots do not account for carries the rule",
            )
        if tuple(row.identifier for row in graded_ast.rows) != alignment:
            return CheckResult(
                "envelope", False, "the graded cell's row alignment moves with the convention"
            )
        if mask_slots(placebo, ranges) != masked:
            return CheckResult(
                "envelope", False,
                "the graded and placebo cells differ outside the registered slots",
            )
    return CheckResult(
        "envelope", True,
        f"over all {seen} conventions: three cells at {envelope.size} bytes, one alignment, "
        "and nothing outside the slots moves",
    )


def check_neutral(generator: Generator, instance: Instance) -> CheckResult:
    """No neutral token the placebo prints may read as a judgement or an answer.

    The placebo fills the registered slots with committed neutral tokens. If one of
    them coincides with a verdict token, that row carries an apparent grade, and an
    apparent grade is an active feedback event in the arm whose whole purpose is to
    carry none. Checking the ALPHABET is not enough: what matters is the tokens the
    stream actually drew, so those are what this reads.
    """
    envelope = instance.envelope
    forbidden = set(_verdict_vocabulary(envelope))
    for convention in conventions(generator.AXES):
        forbidden |= set(generator.key_for(instance.a.table, convention))
        forbidden |= set(generator.key_for(instance.b.table, convention))
    forbidden.discard("")
    hits: list[str] = []
    for name, tokens in sorted(envelope.neutral.items()):
        for position, token in enumerate(tokens):
            if token.strip() in forbidden:
                hits.append(f"{name} row {position + 1} prints {token.strip()!r}")
    if hits:
        return CheckResult(
            "neutral", False,
            "the placebo's committed tokens read as a grade: " + "; ".join(hits[:4]),
        )
    counted = sum(len(t) for t in envelope.neutral.values())
    return CheckResult(
        "neutral", True,
        f"all {counted} committed neutral tokens sit outside every verdict token and "
        "every legal answer on both sides",
    )


def _verdict_vocabulary(envelope) -> tuple[str, ...]:
    """Every literal the registered slots may print as a judgement."""
    out: list[str] = []
    for spec in envelope.slots:
        out.extend(spec.vocabulary)
    return tuple(out)


#: The filing shapes admission renders under. One perfect filing is not a sample:
#: a renderer can be honest on it and carry the rule on every other. `unprintable` is
#: among them because the filed value is the one column with no closed grammar: it is
#: echoed into both cells, so whatever the agent typed is what the serializer has to
#: take, and a shape admission never walks is a shape that first arrives at a seal.
FILING_CLASSES = (
    "canonical", "partial", "blank", "duplicate", "none", "malformed", "unprintable",
)

#: What an `unprintable` filing puts in the value column: a letter outside ASCII, a
#: quotation mark outside it, two control bytes that are inside it, and a lone
#: surrogate, which is a code point a JSON string can carry and UTF-8 cannot encode.
UNPRINTABLE_VALUE = "Routïne “x”\x1b[7m\x00\ud800"


def filing_of(generator: Generator, instance: Instance, side: str, shape: str) -> object:
    """One registered filing shape, as raw text, deterministic per instance."""
    task = instance.side(side)
    identifiers = list(generator.row_identifiers(task.table))
    key = list(task.key)
    if shape == "none":
        return ""
    if shape == "malformed":
        return "no identifiers here\njust prose about the schedule"
    if shape == "canonical":
        return "\n".join(f"{i},{v}" for i, v in zip(identifiers, key))
    if shape == "partial":
        half = max(1, len(identifiers) // 2)
        return "\n".join(f"{i},{v}" for i, v in zip(identifiers[:half], key[:half]))
    if shape == "blank":
        return "\n".join(f"{i}," for i in identifiers)
    if shape == "duplicate":
        lines = [f"{i},{v}" for i, v in zip(identifiers, key)]
        return "\n".join(lines + lines[: max(1, len(lines) // 4)])
    if shape == "unprintable":
        # Half the rows filed correctly, so the shape is a filing that scores rather
        # than a filing the parser throws away before a cell is ever rendered.
        half = max(1, len(identifiers) // 2)
        return "\n".join(
            f"{identifier},{key[n] if n < half else UNPRINTABLE_VALUE}"
            for n, identifier in enumerate(identifiers)
        )
    raise ValueError(f"a filing class is one of {FILING_CLASSES}, not {shape!r}")


def check_graded(generator: Generator, instance: Instance, side: str = "a") -> CheckResult:
    """Every cell the fork would commit is acceptable, under every convention.

    A slot schema stops a field the bytes never show. A grammar stops a numeric code.
    Neither stops a licensed value being chosen for what it encodes: corrections
    spelled as ordinary band names, one per axis, hand the child the whole rule while
    every gate sees a legal answer in a legal slot. So the graded cell is required to
    BE the one the shared grader builds from the outcomes, and this walks the whole
    convention space to say so.

    The judgement is `judge_cells`, the same function the post-filing fork runs, so a
    cell this accepts is a cell the fork accepts. On top of it this asserts what only
    a walk of the support can see: that the bytes outside the two graded slots do not
    move with the convention, which is what makes the slots the only channel there is.
    """
    task = instance.side(side)
    envelope = frozen_envelope(instance.envelope)
    canonical = generator.parse_and_canonicalize(
        task, filing_of(generator, instance, side, "canonical")
    )
    reference: bytes | None = None
    seen = 0
    for convention in conventions(generator.AXES):
        truth = tuple(generator.key_for(task.table, convention))
        retasked = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text=task.text, key=truth,
        )
        judged = judge_cells(generator, retasked, canonical, convention, envelope)
        if judged.problems:
            return CheckResult("graded", False, judged.problems[0])
        payload = judged.payloads[GRADED]
        masked = mask_slots(payload, slot_ranges(judged.asts[GRADED], envelope))
        if reference is None:
            reference = masked
        elif masked != reference:
            return CheckResult(
                "graded", False,
                "the graded cell moves outside its two slots when the convention "
                "moves, so a byte the slots do not account for carries the rule",
            )
        seen += 1
    return CheckResult(
        "graded", True,
        f"over all {seen} conventions the three cells the fork would commit are the "
        "ones it commits, and nothing outside the slots moves",
    )


def check_placebo(generator: Generator, instance: Instance, side: str = "a") -> CheckResult:
    """The placebo prints its committed tokens, whatever was filed and whatever was drawn.

    Checking the committed tokens is not the same as checking what the placebo
    printed. A renderer can be handed neutral tokens and print something else, and it
    can be honest on the one perfect filing admission happens to use and carry a code
    on every other. So this runs the fork's own judgement under every registered
    filing class and every convention, which is what catches a placebo that drops
    rows or returns the wrong wrapper on one filing class and is otherwise honest,
    and adds the byte invariance only a walk of the support can see.
    """
    task = instance.side(side)
    # Snapshot before any renderer runs. A renderer handed the live mapping can
    # rewrite the commitment it is about to be compared against.
    envelope = frozen_envelope(instance.envelope)
    seen = 0
    for shape in FILING_CLASSES:
        raw = filing_of(generator, instance, side, shape)
        reference: bytes | None = None
        for convention in conventions(generator.AXES):
            truth = tuple(generator.key_for(task.table, convention))
            retasked = Task(
                label=task.label, task_id=task.task_id, surface=task.surface,
                table=task.table, text=task.text, key=truth,
            )
            canonical = generator.parse_and_canonicalize(retasked, raw)
            judged = judge_cells(generator, retasked, canonical, convention, envelope)
            if judged.problems:
                return CheckResult(
                    "placebo", False, f"under a {shape} filing, {judged.problems[0]}"
                )
            payload = judged.payloads[PLACEBO]
            if reference is None:
                reference = payload
            elif payload != reference:
                return CheckResult(
                    "placebo", False,
                    f"under a {shape} filing the placebo's bytes move with the drawn "
                    "convention, so the arm that is meant to carry nothing carries the rule",
                )
            seen += 1
    return CheckResult(
        "placebo", True,
        f"over {len(FILING_CLASSES)} filing classes and {seen} judgements, every cell "
        "the fork would commit is one it commits and no cell moved with the convention",
    )


def _blank_token(generator: Generator) -> str:
    return str(getattr(generator, "BLANK_TOKEN", "(empty)"))


def _unfiled_token(generator: Generator) -> str:
    return str(getattr(generator, "UNFILED_TOKEN", "(none)"))


def check_oracle(generator: Generator, instance: Instance) -> CheckResult:
    """The oracle states the rule that was drawn, read by this package's own parser.

    The oracle arm is the denominator of the room the whole screen is taken against,
    so what it says has to be established rather than confirmed. It is read with the
    library parser over the family's declared phrase table, not with the family's own
    reader: a renderer and a reader supplied together have no fixed point outside
    themselves, so a round trip through both is not evidence about either.

    A family that declares no phrase table cannot have its oracle read at all, and an
    oracle nobody can read is not a statement of the rule.

    AND THE CELL IS THE ONE THE TABLE RENDERS, exactly. Reading one option per axis
    back out leaves the wrapper and the rest of the body free, so a renderer could
    print another task's reference, another record count, and coaching under the rule,
    all inside the arm the whole screen measures its denominator against.
    """
    template = getattr(generator, "ORACLE", None)
    if not isinstance(template, OracleTemplate):
        return CheckResult(
            "oracle", False,
            "the family declares no oracle phrase table, so nothing here can read what "
            "its oracle says",
        )
    if set(template.axes) != {axis.name for axis in generator.AXES}:
        return CheckResult(
            "oracle", False,
            f"its oracle table covers {sorted(template.axes)} against the axes "
            f"{sorted(axis.name for axis in generator.AXES)}",
        )
    seen = 0
    for convention in support_of(generator):
        # The comparison target is built before the renderer is handed anything, and
        # what it is handed is a read-only view of that target.
        expected = dict(convention)
        want = oracle_render(
            frozen_template(template),
            instance.a.task_id,
            MappingProxyType(expected),
            instance.a.n_rows,
        )
        ast = generator.render_oracle(
            instance.a.task_id, MappingProxyType(expected), instance.a.n_rows
        )
        try:
            stated = parse_oracle_body(template, ast.body)
        except ValueError as exc:
            return CheckResult(
                "oracle", False, f"the oracle cannot be read under one convention: {exc}"
            )
        if stated != expected:
            return CheckResult(
                "oracle", False,
                f"the oracle states {stated} where {expected} was drawn",
            )
        if ast != want:
            return CheckResult(
                "oracle", False,
                "the oracle is not the cell the registered template renders under one "
                "convention, " + oracle_difference(ast, want),
            )
        seen += 1
    return CheckResult(
        "oracle", True,
        f"over all {seen} conventions the oracle is the cell the registered template "
        "renders, and this package's own parser reads the drawn rule back out of it",
    )


def check_lint(generator: Generator, instance: Instance) -> CheckResult:
    """No option token in either task text."""
    for task in (instance.a, instance.b):
        hits = option_mentions(generator.AXES, task.text)
        if hits:
            return CheckResult(
                "lint", False,
                "task %s names %s" % (task.label, ", ".join(f"{a}={o}" for a, o in hits)),
            )
    return CheckResult("lint", True, "neither task text names an option")


def check_invariance(generator: Generator, instance: Instance) -> CheckResult:
    """Every model-visible pre-feedback artifact is identical under every convention.

    The task texts are what the agent sees before any receipt exists. If one byte of
    them moved with the draw, possession of the surface would teach the draw, and a
    public surface could never be published.
    """
    seen: set[tuple[str, str]] = set()
    for convention in conventions(generator.AXES):
        seen.add(_with_convention(generator, instance, convention))
    if len(seen) != 1:
        return CheckResult(
            "invariance", False,
            f"the task texts take {len(seen)} forms across the convention space, so the "
            "surface carries the draw",
        )
    return CheckResult(
        "invariance", True,
        "both task texts are byte-identical under every convention in the space",
    )


def _with_convention(
    generator: Generator, instance: Instance, convention: Mapping[str, str]
) -> tuple[str, str]:
    """The two task texts as they would read under another convention."""
    out: list[str] = []
    for task in (instance.a, instance.b):
        rebuilt = Task(
            label=task.label,
            task_id=task.task_id,
            surface=task.surface,
            table=task.table,
            text="",
            key=tuple(generator.key_for(task.table, convention)),
        )
        out.append(generator.describe(rebuilt.public()))
    return out[0], out[1]


# --------------------------------------------------------------------------
# the whole set
# --------------------------------------------------------------------------


def _both(
    generator: Generator,
    instance: Instance,
    name: str,
    check: Callable[[Generator, Instance, str], CheckResult],
) -> CheckResult:
    """Run one artifact check on A and on B, and report the first side that fails."""
    seen: list[str] = []
    for side in ("a", "b"):
        result = check(generator, instance, side)
        if not result.passed:
            return CheckResult(name, False, f"on side {side.upper()}: {result.detail}")
        seen.append(f"{side.upper()} {result.detail}")
    return CheckResult(name, True, "; ".join(seen))


def _guarded(name: str, run: Callable[[], CheckResult]) -> CheckResult:
    """Run one check, turning a generator's exception into a failed check.

    The generator module is not trusted. A module that raises when it is checked has
    not passed the check, and an exception escaping to the caller would stop the
    admission report rather than record a refusal, which is the wrong shape of
    outcome for something a bank is supposed to decide about.
    """
    try:
        return run()
    except Exception as exc:  # noqa: BLE001 - any failure here is a failed check
        return CheckResult(name, False, f"the generator raised: {type(exc).__name__}: {exc}")


def run_checks(
    generator: Generator,
    instance: Instance,
    master: bytes,
    *,
    max_copy_score: float,
    max_flip_score: float,
    min_leverage: float,
    min_material_rows: int = 1,
) -> list[CheckResult]:
    """Every named check, in the order a reader wants them."""
    planned: list[tuple[str, Callable[[], CheckResult]]] = [
        ("exercise", lambda: check_exercise(generator, instance)),
        ("materiality", lambda: check_materiality(generator, instance, min_material_rows)),
        ("copy", lambda: check_copy(
            generator, instance, max_copy_score, max_flip_score, min_leverage)),
        ("fixation", lambda: check_fixation(generator, instance, master)),
        # Both siblings. B is a measured finalization path, so an author's mistake in
        # a cell only B renders is a branch failure at seal time rather than anything
        # admission saw, and the chain records it as an outcome.
        ("envelope", lambda: _both(generator, instance, "envelope", check_envelope)),
        ("graded", lambda: _both(generator, instance, "graded", check_graded)),
        ("placebo", lambda: _both(generator, instance, "placebo", check_placebo)),
        ("neutral", lambda: check_neutral(generator, instance)),
        ("oracle", lambda: check_oracle(generator, instance)),
        ("lint", lambda: check_lint(generator, instance)),
        ("invariance", lambda: check_invariance(generator, instance)),
    ]
    return [_guarded(name, run) for name, run in planned]


__all__ = [
    "COPY_MAPS",
    "NO_INDUCTION_MAPS",
    "REPORTED_MAPS",
    "UNPRINTABLE_VALUE",
    "CheckResult",
    "axis_leverage",
    "axis_materiality",
    "check_copy",
    "FILING_CLASSES",
    "check_envelope",
    "check_graded",
    "check_placebo",
    "filing_of",
    "check_exercise",
    "check_fixation",
    "check_invariance",
    "check_lint",
    "check_neutral",
    "check_oracle",
    "check_materiality",
    "copy_map_filings",
    "copy_scores",
    "run_checks",
]
