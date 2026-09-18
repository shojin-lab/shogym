"""What a sampled receipt tells a reader, averaged over the law rather than the draw.

WHY THIS EXISTS. The siblinghood exercise check asks what the receipt in front of it
resolved: every axis, at its own arity, on the realized artifact. That question has no
answer under a policy that reports four rows of twenty four. A mask can miss an axis's
witnesses entirely, and it is not redrawn when it does, because what is registered is
the LAW and not the realized receipt: a mask redrawn until every hidden decision had a
witness would be a mask that depended on the convention, and the average the gate takes
would be an average over a law nobody registered.

So the question this module asks instead is about the registered law. Over every mask
the policy can draw, uniformly and without replacement:

    E|C|        the expected number of conventions consistent with what the receipt
                showed, which is how much is left to be uncertain about
    singleton   how often the receipt pins the convention outright
    entropy     the mean entropy of the uniform posterior on that consistent set
    P0          the best level a reader with no receipt at all reaches on the sibling
    U           the best level an ideal reader of this receipt reaches on the sibling
    floor       the lookup floor, recomputed over masks from the reduced receipt
    room        U minus the floor, which is what an ideal reader gains over a lookup

and, per single-axis alternative to the drawn convention, the probability that the
receipt distinguishes it at all.

THE ARITHMETIC IS EXACT AND THERE IS NO MONTE CARLO. Every mask is visited. Two things
make that cheap enough. A mask's partition of the convention space depends only on the
SET OF DISTINCT ANSWER COLUMNS it selects, so two masks that select the same columns,
or the same columns twice over, are one calculation. And the class a partition produces
is an intersection of those columns' agreement sets, so the same class recurs across
masks and its score on the sibling is computed once.

WHAT AN IDEAL READER IS. A partition and an action rule. The reader observes its class,
holds a uniform posterior on the conventions in it, and answers each row of the sibling
task independently; scoring is row additive, so its best action per row is that row's
posterior mode. This is the same reader the gates price, and the quantities here
reproduce the gate's own ceiling and floor when they are asked about one mask rather
than averaged over all of them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from math import comb
from typing import Mapping, Sequence

import numpy as np

from shogym.envs.receipts.protocol import (
    Generator,
    Instance,
    ReceiptPolicy,
    conventions,
    policy_of,
    support_of,
)

#: THE REGISTERED LAW-LEVEL BARS, and they are read against the registered mask law.
#:
#: The distinguishing bar is 0.30, and the number is the sampling law's own: ledger's
#: weakest registered movement is two rows, and a four-row mask misses both of them
#: with probability choose(22,4)/choose(24,4), so an alternative that moves two rows is
#: distinguished with probability 0.3116. A bar above that would refuse the movement
#: the table builder already guarantees; a bar much below it would admit an alternative
#: the receipt almost never speaks to.
REGISTERED_MIN_DISTINGUISHING = 0.30
#: The band the bank-mean ideal level has to sit in. It is a design choice and not an
#: inherited admission bar: above it the receipt identifies too much for a later step to
#: improve on, which is the saturation this policy exists to undo, and below it the
#: receipt is not teaching enough for the task to be about reading it.
REGISTERED_MIN_IDEAL = 0.75
REGISTERED_MAX_IDEAL = 0.90
#: What an ideal reader has to gain over a lookup, bank-mean. Below this the receipt's
#: own arithmetic is reachable without inferring anything.
REGISTERED_MIN_LAW_ROOM = 0.05


@dataclass(frozen=True)
class LawResult:
    """What one instance's receipt law comes to, averaged over every mask it can draw."""

    tag: str
    policy: str
    rows: int
    reported: int
    masks: int
    compatible: float
    singleton: float
    entropy: float
    no_receipt: float
    ideal: float
    floor: float
    #: Per single-axis alternative to the drawn convention, the probability over masks
    #: that the receipt shows a row the two disagree on.
    distinguishing: Mapping[str, float]
    #: The axes a comparator hands the reader for nothing, and the floor it leaves.
    #:
    #: SOME FAMILIES HAVE ANSWERS THAT LEAK AN AXIS. A corrected daughter form carries
    #: the phone the replacement introduced and one fewer of the vowel the deletion
    #: took, so a reader of one corrected row can name two of sound change's four
    #: choices by direct correspondence, without inferring anything. A floor that
    #: counted whole words as indivisible answers would concede none of that and would
    #: report room the family does not have. So a family's own audit names the axes its
    #: answers hand over and asks for the floor computed with them conceded as well.
    #:
    #: Naming none leaves this equal to the floor, which is the statement that nothing
    #: the receipt prints hands an axis over for free.
    given: tuple[str, ...] = ()
    augmented: float = 0.0

    @property
    def room(self) -> float:
        return self.ideal - self.floor

    @property
    def augmented_room(self) -> float:
        """What an ideal reader gains over a lookup that was handed the given axes."""
        return self.ideal - self.augmented

    @property
    def weakest(self) -> tuple[str, float]:
        """The single-axis alternative the receipt speaks to least often."""
        if not self.distinguishing:
            return ("", 1.0)
        name = min(self.distinguishing, key=lambda k: self.distinguishing[k])
        return (name, self.distinguishing[name])

    def detail(self) -> str:
        """One line a reader of a check result can price the law from.

        The augmented floor is named only where a family gave an axis away, so the line
        a family that gave none prints is the line it printed before there was anything
        to give.
        """
        return self._detail() + (
            ""
            if not self.given
            else "; with %s given away the floor is %.4f and the room %.4f"
            % (" and ".join(self.given), self.augmented, self.augmented_room)
        )

    def _detail(self) -> str:
        name, weakest = self.weakest
        return (
            "over all %d masks: consistent conventions %.4f, singleton %.4f, entropy "
            "%.4f bits, no receipt %.4f, ideal %.4f, lookup floor %.4f, room %.4f; "
            "weakest single-axis alternative %s at %.4f"
            % (
                self.masks,
                self.compatible,
                self.singleton,
                self.entropy,
                self.no_receipt,
                self.ideal,
                self.floor,
                self.room,
                name,
                weakest,
            )
        )

    def as_record(self) -> dict[str, object]:
        return {
            "tag": self.tag,
            "policy": self.policy,
            "rows": self.rows,
            "reported": self.reported,
            "masks": self.masks,
            "compatible": self.compatible,
            "singleton": self.singleton,
            "entropy": self.entropy,
            "no_receipt": self.no_receipt,
            "ideal": self.ideal,
            "floor": self.floor,
            "room": self.room,
            "distinguishing": {k: v for k, v in sorted(self.distinguishing.items())},
            "given": list(self.given),
            "augmented": self.augmented,
        }


def _codes(keys: Sequence[Sequence[str]], normalize) -> np.ndarray:
    """One integer per answer, with a vocabulary per row, as the gates code them.

    Normalized first, because two spellings the scorer treats as one answer are one
    legal action and coding them apart would report room in a distinction nobody can
    act on.
    """
    if not keys or not keys[0]:
        return np.zeros((len(keys), 0), dtype=np.int64)
    width = len(keys[0])
    out = np.zeros((len(keys), width), dtype=np.int64)
    for row in range(width):
        vocabulary: dict[str, int] = {}
        for position, key in enumerate(keys):
            out[position, row] = vocabulary.setdefault(
                normalize(key[row]), len(vocabulary)
            )
    return out


class _Sibling:
    """The sibling task's answers, and what an ideal reader scores on a class of rules.

    A CLASS IS A BITMASK, one bit per convention, and it is the cache key. The same
    class recurs across masks and across references by the thousand, and an integer is
    what makes recognizing that free: the partitions below are computed by intersecting
    masks rather than by relabelling seventy two rows each time.
    """

    def __init__(self, answers: np.ndarray) -> None:
        self._answers = answers
        self._rows = int(answers.shape[1])
        self._cache: dict[int, float] = {}

    def score(self, members: int) -> float:
        """The rowwise posterior-mode score of a reader holding this class."""
        if members.bit_count() == 1:
            # A reader that has pinned the convention answers every row of the sibling
            # from it. Counting that out row by row is the same one every time.
            return 1.0
        known = self._cache.get(members)
        if known is not None:
            return known
        rows = [position for position in range(self._answers.shape[0]) if members >> position & 1]
        block = self._answers[rows]
        size = float(len(rows))
        if not self._rows or not size:
            return 0.0
        total = 0.0
        for row in range(self._rows):
            total += float(np.bincount(block[:, row]).max())
        value = total / (self._rows * size)
        self._cache[members] = value
        return value


def _dependence(
    columns: np.ndarray,
    space_axes: Sequence[str],
    options: Sequence[Sequence[str]],
    index: Mapping[tuple[str, ...], int],
    chat: tuple[str, ...],
) -> tuple[list[frozenset[str]], np.ndarray]:
    """Which axes each printed row responds to, and which rows hand one over outright.

    The same derivation the gates make from the rendered receipt, made here from the
    answers behind it: a reported row prints the verdict and that row's own answer, so
    two conventions print the same thing on it exactly when their answers agree. Rows
    the mask did not draw respond to nothing, which is what makes them contribute no
    column to either partition.

    EVIDENT IS DERIVED, NEVER DECLARED. A row is evident when it responds to exactly
    one axis and prints a distinct thing for every option of it, which makes it an
    index a reader can read the option off without inferring anything.
    """
    width = int(columns.shape[1])
    dependence: list[set[str]] = [set() for _ in range(width)]
    distinct = [{axis: 0 for axis in space_axes} for _ in range(width)]
    for position, axis in enumerate(space_axes):
        reached = []
        for option in options[position]:
            alternative = list(chat)
            alternative[position] = option
            reached.append(columns[index[tuple(alternative)]])
        stacked = np.asarray(reached)
        for row in range(width):
            seen = len(set(stacked[:, row].tolist()))
            distinct[row][axis] = seen
            if seen > 1:
                dependence[row].add(axis)
    evident = np.zeros(width, dtype=bool)
    for row in range(width):
        if len(dependence[row]) != 1:
            continue
        axis = next(iter(dependence[row]))
        position = list(space_axes).index(axis)
        evident[row] = distinct[row][axis] == len(options[position])
    return [frozenset(s) for s in dependence], evident


def _refined(blocks: Sequence[int], split: Sequence[int]) -> list[int]:
    """The partition, cut again by one observation whose values are these masks.

    A block of one convention is left where it is rather than intersected with every
    value: nothing can cut it further, and at eight reported rows of twenty four most
    of the partition is singletons after the first few columns. Skipping them is what
    keeps an exact walk of seven hundred thousand masks a walk anyone can run.
    """
    out: list[int] = []
    for block in blocks:
        if block.bit_count() == 1:
            out.append(block)
            continue
        for part in split:
            piece = block & part
            if piece:
                out.append(piece)
    return out


def _resolved(blocks: Sequence[int], total: int) -> bool:
    """Whether every class holds one convention, so no further column can cut it."""
    return len(blocks) == total


def _partition_value(blocks: Sequence[int], sibling: _Sibling, total: int) -> float:
    """The mean score over conventions of a reader that observes this partition."""
    return sum(block.bit_count() * sibling.score(block) for block in blocks) / total


#: How the lookup floor averages over the reference convention the reader is assumed
#: to have applied. ALL is the law-level number and the default: the floor is a
#: property of the table and the policy, and one conditioned on the rule that happened
#: to be drawn would price a different experiment on every instance. DRAWN is the
#: gate's own choice, kept so that this arithmetic can be held against the gate's floor
#: on the receipt the gate is actually looking at.
ALL_REFERENCES = "all"
DRAWN_REFERENCE = "drawn"
REFERENCE_RULES = (ALL_REFERENCES, DRAWN_REFERENCE)


#: How many computed laws are kept in memory, and why any are.
#:
#: The law is a pure function of the two tables, the drawn convention and the policy, and
#: it is an exact walk of every mask rather than a sample of them. Verification recomputes
#: a bank's population by rerunning admission, and a run that verifies several bundles
#: over one bank walks the same masks for the same answer every time. The key is the two
#: task identifiers, which are HMACs over the master key and the coordinates, so two banks
#: under two keys never share an entry and a redrawn instance of the same bank does.
_CACHE_SIZE = 512
_CACHE: "dict[tuple[object, ...], LawResult]" = {}


def law_for(
    generator: Generator,
    instance: Instance,
    side: str = "a",
    references: str = ALL_REFERENCES,
    policy: ReceiptPolicy | None = None,
    given: Sequence[str] = (),
) -> LawResult:
    """The receipt law for one instance, computed exactly over every mask.

    `side` is the task the receipt grades; the reader is scored on the other one.
    `references` says which reference conventions the lookup floor averages over.

    `policy` is the family's own unless a caller names another, which is what pricing a
    candidate count takes: what eight reported rows would leave on these tables is a
    question about the tables and the law, and it is asked without rendering a single
    cell under a policy nothing is registered as.

    `given` names axes a comparator concedes to the reader for nothing, on top of the
    lookup observations the floor already concedes. It moves the floor and never the
    ceiling: what an ideal reader of the receipt reaches is the same whatever a
    comparator is handed. A family names its own in its audit, because which of a
    family's answers hand an axis over is a fact about what its answers spell.
    """
    if references not in REFERENCE_RULES:
        raise ValueError(
            f"a floor averages over {REFERENCE_RULES}, not {references!r}"
        )
    policy = policy or policy_of(generator)
    conceded = tuple(given)
    named = {axis.name for axis in generator.AXES}
    unknown = [axis for axis in conceded if axis not in named]
    if unknown:
        raise ValueError(
            f"{generator.name!r} has no axis {', '.join(unknown)}, so nothing can be "
            "given away on it"
        )
    remembered = (
        type(generator).__qualname__,
        instance.a.task_id,
        instance.b.task_id,
        tuple(sorted(instance.convention.items())),
        policy.as_record()["name"],
        policy.as_record()["reported"],
        side.strip().lower(),
        references,
        conceded,
    )
    known = _CACHE.get(remembered)
    if known is not None:
        return known
    task = instance.side(side)
    sibling = instance.side("b" if side.strip().lower() == "a" else "a")
    axes = tuple(axis.name for axis in generator.AXES)
    options = tuple(tuple(axis.options) for axis in generator.AXES)
    drawable = [
        tuple(convention[axis] for axis in axes) for convention in support_of(generator)
    ]
    whole = [tuple(convention[axis] for axis in axes) for convention in conventions(generator.AXES)]
    index = {combo: position for position, combo in enumerate(drawable)}
    if len(index) != len(drawable) or len(drawable) != len(whole):
        # The law averages over the conventions the sampler can reach. A family with a
        # smaller declared support is a different calculation and is not priced here.
        raise ValueError(
            f"{generator.name!r} declares a support smaller than its axis product, and "
            "the receipt law is registered over the whole product"
        )
    graded = _codes(
        [
            tuple(generator.key_for(task.table, dict(zip(axes, combo))))
            for combo in drawable
        ],
        generator.normalize_answer,
    )
    answers = _codes(
        [
            tuple(generator.key_for(sibling.table, dict(zip(axes, combo))))
            for combo in drawable
        ],
        generator.normalize_answer,
    )
    reader = _Sibling(answers)
    # THE FLOOR IS AVERAGED OVER EVERY CANONICAL REFERENCE, not taken at the drawn
    # one. What the reader could have read off a receipt without inferring anything is
    # a property of the table and the policy, and a floor conditioned on the rule that
    # was drawn would move with the draw and would price a different experiment on
    # every instance. The gate's own per-instance floor stays at the drawn reference,
    # because that is the receipt it is looking at; this is the law behind it.
    chat = tuple(instance.convention[axis] for axis in axes)
    walked = drawable if references == ALL_REFERENCES else [chat]
    profiles = {
        index[combo]: _dependence(graded, axes, options, index, combo)
        for combo in walked
    }
    rows = int(graded.shape[1])
    reported = rows if not policy.samples else policy.reported
    masks = comb(rows, reported)

    # A mask's partition depends on the DISTINCT answer columns it selects and on
    # nothing else, so every mask is walked and the calculation is done once per set of
    # columns. Two masks that select the same column twice over are one calculation.
    kinds: dict[bytes, int] = {}
    kind_of: list[int] = []
    for column in range(rows):
        key = graded[:, column].tobytes()
        kind_of.append(kinds.setdefault(key, len(kinds)))
    seen: dict[frozenset[int], int] = {}
    for mask in combinations(range(rows), reported):
        selected = frozenset(kind_of[column] for column in mask)
        seen[selected] = seen.get(selected, 0) + 1
    representative: dict[int, int] = {}
    for column in range(rows):
        representative.setdefault(kind_of[column], column)

    total = float(sum(seen.values()))
    # The answer table as plain rows, and each column as the masks its values cut the
    # convention space into. Every partition below is an intersection of those masks,
    # which is one integer operation where relabelling seventy two rows is seventy two.
    table = [[int(value) for value in row] for row in graded.tolist()]
    population = len(drawable)
    everyone = (1 << population) - 1
    # WHAT A CONCEDED AXIS COSTS THE FLOOR, as one more refinement rather than as a
    # second construction: the conventions grouped by their options on the given axes,
    # one bitmask per group. Cutting the lookup partition by those groups is exactly
    # handing the reader those choices on top of what it could already read off.
    given_masks: list[int] | None = None
    if conceded:
        places = [axes.index(name) for name in conceded]
        groups: dict[tuple[str, ...], int] = {}
        for position, combo in enumerate(drawable):
            key = tuple(combo[place] for place in places)
            groups[key] = groups.get(key, 0) | (1 << position)
        given_masks = list(groups.values())
    column_masks: list[list[int]] = []
    column_mask_of: list[dict[int, int]] = []
    for column in range(rows):
        by_value: dict[int, int] = {}
        for h in range(population):
            value = table[h][column]
            by_value[value] = by_value.get(value, 0) | (1 << h)
        column_masks.append(list(by_value.values()))
        column_mask_of.append(by_value)
    # Per reference, the role it derives for each column: evident, or the identity of
    # the dependence class it joins. Two references give one floor calculation when
    # they agree on the selected columns AND derive the same roles for them, which is
    # what keeps an average over seventy two references from costing seventy two times
    # as much.
    #
    # EMPTY LOCAL DEPENDENCE IS AN ORDINARY CLASS. `_dependence` moves one axis at a
    # time, so a column it finds empty is a column no SINGLE-axis alternative moves;
    # a joint alternative can still move it, and the reader who sees it match learns
    # that. Dropping that class would concede less to the lookup than the receipt
    # actually shows, which understates the floor and overstates the room above it.
    # This is the construction `shogym.receipts.resolution` has always used for the
    # gate's own floor, where every non-evident row joins the class of its dependence
    # signature whatever that signature is.
    signature_id: dict[frozenset[str], int] = {}
    roles: dict[int, list[int]] = {}
    for reference, (dependence, evident) in profiles.items():
        row_roles: list[int] = []
        for column in range(rows):
            if evident[column]:
                row_roles.append(-1)
            else:
                row_roles.append(
                    signature_id.setdefault(dependence[column], len(signature_id))
                )
        roles[reference] = row_roles

    compatible = singleton = entropy = ideal = floor = augmented = 0.0
    for selected, weight in seen.items():
        columns = sorted(representative[kind] for kind in selected)
        blocks: list[int] = [everyone]
        for column in columns:
            if _resolved(blocks, population):
                break
            blocks = _refined(blocks, column_masks[column])
        share = weight / total
        sizes = [block.bit_count() for block in blocks]
        compatible += share * (sum(size * size for size in sizes) / population)
        singleton += share * (sum(1 for size in sizes if size == 1) / population)
        entropy += share * (
            sum(size * math.log2(size) for size in sizes) / population
        )
        ideal += share * _partition_value(blocks, reader, population)
        # The lookup floor over the same reduced receipt: the evident rows it shows,
        # plus one all-matched bit per class of shown rows a reader can see and not
        # resolve, the class of empty local dependence included. Rows the mask did not
        # draw respond to nothing, so their bit is constant and contributes no column,
        # which is what makes the floor a statement about the reduced receipt rather
        # than about the full one.
        #
        # Two references give one calculation when they agree on the selected columns
        # AND derive the same roles for them, because the labels are built from
        # exactly those two things. That is what keeps an average over seventy two
        # references from costing seventy two times as much.
        grouped: dict[tuple[tuple[int, ...], tuple[int, ...]], list[int]] = {}
        for reference in roles:
            key = (
                tuple(roles[reference][c] for c in columns),
                tuple(table[reference][c] for c in columns),
            )
            grouped.setdefault(key, []).append(reference)
        for (derived, _), members in grouped.items():
            reference = members[0]
            lookup: list[int] = [everyone]
            for position, column in enumerate(columns):
                if derived[position] != -1:
                    continue
                if _resolved(lookup, population):
                    break
                lookup = _refined(lookup, column_masks[column])
            together: dict[int, int] = {}
            for position, column in enumerate(columns):
                role = derived[position]
                if role < 0:
                    continue
                agreed = column_mask_of[column][table[reference][column]]
                together[role] = together.get(role, everyone) & agreed
            for role in sorted(together):
                if _resolved(lookup, population):
                    break
                agreed = together[role]
                lookup = _refined(lookup, (agreed, everyone & ~agreed))
            weight = share * (len(members) / len(roles))
            value = _partition_value(lookup, reader, population)
            floor += weight * value
            augmented += weight * (
                value
                if given_masks is None
                else _partition_value(
                    _refined(lookup, given_masks), reader, population
                )
            )

    no_receipt = _partition_value([everyone], reader, population)
    chat_position = index[chat]
    distinguishing: dict[str, float] = {}
    for position, axis in enumerate(axes):
        for option in options[position]:
            if option == chat[position]:
                continue
            alternative = list(chat)
            alternative[position] = option
            moved = int(
                np.count_nonzero(graded[index[tuple(alternative)]] != graded[chat_position])
            )
            missed = comb(rows - moved, reported) / masks if rows - moved >= reported else 0.0
            distinguishing[f"{axis}={option}"] = 1.0 - missed

    found = LawResult(
        tag=f"{instance.generator}/{instance.ordinal}/{task.label}",
        policy=policy.name,
        rows=rows,
        reported=reported,
        masks=masks,
        compatible=compatible,
        singleton=singleton,
        entropy=entropy,
        no_receipt=no_receipt,
        ideal=ideal,
        floor=floor,
        distinguishing=distinguishing,
        given=conceded,
        augmented=augmented,
    )
    if len(_CACHE) >= _CACHE_SIZE:
        _CACHE.clear()
    _CACHE[remembered] = found
    return found


@dataclass(frozen=True)
class BankLaw:
    """The bank-mean law quantities, and whether they clear the registered band."""

    instances: int
    ideal: float
    floor: float
    room: float
    min_ideal: float
    max_ideal: float
    min_room: float

    @property
    def reasons(self) -> tuple[str, ...]:
        out: list[str] = []
        if not self.min_ideal <= self.ideal <= self.max_ideal:
            out.append(
                "the bank-mean ideal graded level is %.4f, outside the registered "
                "%.2f to %.2f: a receipt above the band identifies too much for a "
                "later step to improve on, and one below it is not teaching enough"
                % (self.ideal, self.min_ideal, self.max_ideal)
            )
        if self.room <= self.min_room:
            out.append(
                "the bank-mean ideal level stands %.4f above the recomputed lookup "
                "floor, which is not more than the registered %.2f: what the receipt "
                "leaves is reachable without inferring anything"
                % (self.room, self.min_room)
            )
        return tuple(out)

    @property
    def passed(self) -> bool:
        return not self.reasons

    def line(self) -> str:
        return (
            "bank law over %d instances: ideal %.4f (band %.2f to %.2f), lookup floor "
            "%.4f, room %.4f (over %.2f)"
            % (
                self.instances,
                self.ideal,
                self.min_ideal,
                self.max_ideal,
                self.floor,
                self.room,
                self.min_room,
            )
        )


def bank_law(
    results: Sequence[LawResult],
    *,
    min_ideal: float = REGISTERED_MIN_IDEAL,
    max_ideal: float = REGISTERED_MAX_IDEAL,
    min_room: float = REGISTERED_MIN_LAW_ROOM,
) -> BankLaw:
    """The bank-mean law, which is the level the registered band is read at.

    THE BAND IS A BANK QUANTITY AND NOT A PER-INSTANCE ONE. A single table's ideal
    level moves with how much its own rows happen to move, and the consultation's six
    pairs run from 0.798 to 0.846 around a mean of 0.817. Reading the band at each
    instance would refuse tables for being at the edge of a distribution the band was
    computed as the centre of; reading it at the bank is what the arithmetic supports.
    """
    if not results:
        raise ValueError("a bank law is taken over at least one instance")
    ideal = sum(r.ideal for r in results) / len(results)
    floor = sum(r.floor for r in results) / len(results)
    return BankLaw(
        instances=len(results),
        ideal=ideal,
        floor=floor,
        room=sum(r.room for r in results) / len(results),
        min_ideal=min_ideal,
        max_ideal=max_ideal,
        min_room=min_room,
    )


__all__ = [
    "ALL_REFERENCES",
    "DRAWN_REFERENCE",
    "REFERENCE_RULES",
    "REGISTERED_MAX_IDEAL",
    "REGISTERED_MIN_DISTINGUISHING",
    "REGISTERED_MIN_IDEAL",
    "REGISTERED_MIN_LAW_ROOM",
    "BankLaw",
    "LawResult",
    "bank_law",
    "law_for",
]
