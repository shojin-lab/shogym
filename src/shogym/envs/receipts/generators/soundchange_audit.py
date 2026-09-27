"""A second implementation of the cascade, and the predicates the pair is filtered on.

WHY A SECOND IMPLEMENTATION. `judge_cells` compares what the renderer printed with what
the scorer computed, so it catches a renderer that disagrees with the scorer and cannot
catch a scorer that is wrong: a rewrite rule that applies only the first match, or reads
its own growing output, or lets a word boundary satisfy a consonant condition, is
carried identically into the receipt, the placebo, the key and the gate, and every named
check passes on it. The serializer can also fit an overwide field by truncating it
rather than refusing it, so a correction longer than its slot becomes a shorter legal
looking answer. Both are mistakes an author makes and cannot see, and the only thing
that finds them is another implementation that was not written from the first.

So this module never imports the production matcher and never uses an intermediate form
the production passes emitted. It shares the inventory, the option identifiers and the
dataclasses, which are declarations rather than computation, and it derives the
operation each option names from the option's own identifier rather than from the
production table: `out_g` introduces g and `erase_e` removes e, so an edited production
table is a disagreement here rather than a change both sides make together.

The passes below are regular expressions with lookaround. A lookaround is zero width and
the scan reads the subject string, so every position is tested against the string as it
stood when the pass began, which is the semantics the task states, arrived at by a
different route from the production scan.

THE PREDICATES ARE PURE AND CONVENTION FREE. `pair_refusal` takes two tuples of proto
forms and returns a reason or the empty string. It never draws, never sees the live
convention, and quantifies over the whole support, which is what makes acceptance a
function of the two batches alone: a filter that read the live draw would make rejection
informative, and an agent that knew which pairs are accepted could exclude conventions
from the fact that this pair was served.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Mapping, Sequence

import numpy as np

from shogym.envs.receipts.checks import CheckResult
from shogym.envs.receipts.generators.soundchange import (
    ALL_CONVENTIONS,
    AXES,
    CONSONANTS,
    MAX_DAUGHTER,
    ROWS,
    VOWELS,
    SoundTable,
)
from shogym.envs.receipts.protocol import Instance
from shogym.envs.receipts.receipt_ast import (
    GAP,
    ORDINAL_WIDTH,
    frozen_envelope,
    row_lines,
    serialize,
)
from shogym.envs.receipts.render import feedback_for
from shogym.receipts.resolution import rowwise_scores

# ----- the registered bars this module reads ---------------------------------
#
# The construction bars, which are stronger than the admission bars they sit above and
# are checked over the whole support rather than under the drawn convention.

#: Forms carrying the original nb cluster, per batch, so the public pass is exercised.
MIN_CLUSTER_ROWS = 4
#: Rows one option change has to move, minimized over every draw and every alternative.
MIN_MOVED_ROWS = 4
#: What a careful reader with no receipt can score on a batch by filing each row's modal
#: daughter over the whole support.
MAX_NO_RECEIPT = 0.25
#: The lookup floor, the augmented lookup floor, and the room each has to leave.
MAX_LOOKUP_FLOOR = 0.90
MIN_LOOKUP_ROOM = 0.05
#: What the character copy family and the skeleton transfer may earn on the sibling.
MAX_TRANSFER = 0.50
#: The longest proto form the grammar produces, held here so a drifting grammar is a
#: failed check rather than a wider envelope nobody noticed.
MAX_PROTO = 14

#: The phones the replacement pass can introduce. Derived from the option identifiers
#: below rather than declared twice.
AXIS_NAMES = tuple(axis.name for axis in AXES)


def introduced_phone(option: str) -> str:
    """The phone the reflex option named `out_g`, `out_x` or `out_s` introduces."""
    head, _, phone = option.partition("_")
    if head != "out" or len(phone) != 1 or phone not in CONSONANTS:
        raise ValueError(f"a reflex option names a consonant it introduces, not {option!r}")
    return phone


def removed_vowel(option: str) -> str:
    """The vowel the loss option named `erase_e`, `erase_i` or `erase_a` removes."""
    head, _, vowel = option.partition("_")
    if head != "erase" or len(vowel) != 1 or vowel not in VOWELS:
        raise ValueError(f"a loss option names a vowel it removes, not {option!r}")
    return vowel


def replaces_first(option: str) -> bool:
    """Whether the sequence option puts the replacement pass before the deletion."""
    if option == "replace_then_erase":
        return True
    if option == "erase_then_replace":
        return False
    raise ValueError(f"a sequence option names one of the two orders, not {option!r}")


def two_sided(option: str) -> bool:
    """Whether the environment option requires a vowel on both sides of the k."""
    if option == "vowel_pair":
        return True
    if option == "right_vowel":
        return False
    raise ValueError(f"an environment option names one of the two contexts, not {option!r}")


# --------------------------------------------------------------------------
# the cascade, by lookaround
# --------------------------------------------------------------------------

_NASAL = re.compile(r"n(?=b)")
_RIGHT_VOWEL_K = re.compile(r"k(?=[%s])" % VOWELS)
_VOWEL_PAIR_K = re.compile(r"(?<=[%s])k(?=[%s])" % (VOWELS, VOWELS))
_DELETE = {
    vowel: re.compile(r"(?<=[%s])%s(?=[%s])" % (CONSONANTS, vowel, CONSONANTS))
    for vowel in VOWELS
}


def audit_nasal(word: str) -> str:
    """Every n immediately before b becomes m, tested against the pass input."""
    return _NASAL.sub("m", word)


def audit_replace(word: str, option_reflex: str, option_environment: str) -> str:
    """Every k in the named environment becomes the phone the reflex option names."""
    pattern = _VOWEL_PAIR_K if two_sided(option_environment) else _RIGHT_VOWEL_K
    return pattern.sub(introduced_phone(option_reflex), word)


def audit_delete(word: str, option_loss: str) -> str:
    """Every occurrence of the named vowel with consonants on both sides goes."""
    return _DELETE[removed_vowel(option_loss)].sub("", word)


def audit_daughter(proto: str, convention: Mapping[str, str]) -> str:
    """The daughter this cascade produces, computed without the production passes."""
    word = audit_nasal(proto)
    if replaces_first(convention["sequence"]):
        word = audit_replace(word, convention["reflex"], convention["environment"])
        return audit_delete(word, convention["loss"])
    word = audit_delete(word, convention["loss"])
    return audit_replace(word, convention["reflex"], convention["environment"])


def audit_key(protos: Sequence[str], convention: Mapping[str, str]) -> tuple[str, ...]:
    return tuple(audit_daughter(proto, convention) for proto in protos)


# --------------------------------------------------------------------------
# the whole support, as arrays
# --------------------------------------------------------------------------

_COMBO_INDEX = {
    tuple(convention[name] for name in AXIS_NAMES): position
    for position, convention in enumerate(ALL_CONVENTIONS)
}


def _substituted(position: int, axis: str, option: str) -> int:
    """Where the convention at `position` sits once one axis is changed."""
    combo = list(ALL_CONVENTIONS[position][name] for name in AXIS_NAMES)
    combo[AXIS_NAMES.index(axis)] = option
    return _COMBO_INDEX[tuple(combo)]


@lru_cache(maxsize=64)
def support_keys(protos: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """One answer key per convention, in the declared product order."""
    return tuple(audit_key(protos, convention) for convention in ALL_CONVENTIONS)


def answer_codes(keys: Sequence[Sequence[str]]) -> np.ndarray:
    """The keys as integer codes, one vocabulary per row.

    Codes rather than strings because the partitions below are row comparisons over the
    whole support, and because this is the same reduction the gate's observation makes:
    what a reader can tell apart is which rows agree, not what they spell.
    """
    rows = len(keys[0]) if keys else 0
    out = np.zeros((len(keys), rows), dtype=np.int64)
    for row in range(rows):
        vocabulary: dict[str, int] = {}
        for position, key in enumerate(keys):
            out[position, row] = vocabulary.setdefault(key[row], len(vocabulary))
    return out


def _classes(matrix: np.ndarray) -> np.ndarray:
    """Group identical integer rows; returns the group id of each row."""
    if matrix.shape[1] == 0:
        return np.zeros(matrix.shape[0], dtype=np.int64)
    return np.unique(
        np.ascontiguousarray(matrix), axis=0, return_inverse=True
    )[1].reshape(-1)


def row_dependence(shown: np.ndarray, reference: int) -> list[frozenset[str]]:
    """Which axes each printed row responds to, with the rest held at the reference.

    THE RECEIPT ROW IS THE ANSWER. Under the canonical filing a row prints PASS with an
    empty correction when the truth agrees with the reference on that row, and FAIL with
    that row's whole daughter otherwise, and the correction is never truncated because
    the slot is wider than the longest daughter. So the printed row and the corrected
    answer determine each other, and a dependence computed from the answers is the
    dependence the gate computes from the bytes. Admission verifies that equality on the
    serialized cells rather than taking it on trust here.
    """
    out: list[frozenset[str]] = []
    for row in range(shown.shape[1]):
        responds = set()
        for axis in AXES:
            positions = [_substituted(reference, axis.name, o) for o in axis.options]
            if len(set(shown[positions, row].tolist())) > 1:
                responds.add(axis.name)
        out.append(frozenset(responds))
    return out


def evident_rows(
    shown: np.ndarray, reference: int, dependence: Sequence[frozenset[str]]
) -> np.ndarray:
    """The rows that hand one axis over outright, derived and never declared.

    A row is evident when it responds to exactly one axis and prints a distinct thing
    for every option of that axis: such a row is an index, and reading the option off it
    costs no induction. The derivation is deliberately generous, so the floor it feeds
    is high and the room reported is less than the design has rather than more.
    """
    out = np.zeros(shown.shape[1], dtype=bool)
    for row, responds in enumerate(dependence):
        if len(responds) != 1:
            continue
        axis = next(axis for axis in AXES if axis.name in responds)
        positions = [_substituted(reference, axis.name, o) for o in axis.options]
        out[row] = len(set(shown[positions, row].tolist())) == len(axis.options)
    return out


def lookup_signature(shown: np.ndarray, reference: int) -> np.ndarray:
    """What a reader confined to the receipt's own labels observes, per convention.

    The evident rows in full, plus one all-passed bit for each class of rows a reader
    can see but not resolve. Both derived from the receipt.
    """
    dependence = row_dependence(shown, reference)
    evident = evident_rows(shown, reference, dependence)
    columns = [shown[:, evident]] if evident.any() else []
    hidden = ~evident
    if hidden.any():
        signatures = sorted(
            {dependence[i] for i in np.where(hidden)[0]}, key=sorted
        )
        for signature in signatures:
            members = np.array(
                [hidden[i] and dependence[i] == signature for i in range(shown.shape[1])]
            )
            same = (shown[:, members] == shown[reference][members][None, :]).all(axis=1)
            columns.append(same.astype(np.int64)[:, None])
    if not columns:
        return np.zeros((shown.shape[0], 0), dtype=np.int64)
    return np.concatenate(columns, axis=1)


_REFLEX_COLUMN = np.array(
    [AXES[0].options.index(c["reflex"]) for c in ALL_CONVENTIONS], dtype=np.int64
)
_LOSS_COLUMN = np.array(
    [AXES[2].options.index(c["loss"]) for c in ALL_CONVENTIONS], dtype=np.int64
)


def augmented_signature(shown: np.ndarray, reference: int) -> np.ndarray:
    """The lookup signature with the reflex and the deleted vowel given away for free.

    The replacement phones never occur in a proto form and the deleted vowel can be read
    off the difference in vowel counts, so a reader of A's corrected key can name those
    two axes by direct subword correspondence. Counting whole words as indivisible
    answers would conceal that, so this hands both over and asks what is left.
    """
    return np.concatenate(
        [lookup_signature(shown, reference), _REFLEX_COLUMN[:, None], _LOSS_COLUMN[:, None]],
        axis=1,
    )


def _partition_score(groups: np.ndarray, answers: np.ndarray) -> float:
    """The mean rowwise Bayes-action score of a reader whose observation is `groups`."""
    return float(rowwise_scores(groups, answers)[0].mean())


@dataclass(frozen=True)
class Room:
    """What a reader of one side's receipt can reach on the other side."""

    ceiling: float
    floors: tuple[float, ...]
    augmented: tuple[float, ...]

    @property
    def worst_floor(self) -> float:
        return max(self.floors)

    @property
    def worst_augmented(self) -> float:
        return max(self.augmented)


def room(shown: np.ndarray, answers: np.ndarray) -> Room:
    """The ceiling, and both floors at every reference draw in the support."""
    ceiling = _partition_score(_classes(shown), answers)
    floors: list[float] = []
    augmented: list[float] = []
    for reference in range(len(ALL_CONVENTIONS)):
        floors.append(_partition_score(_classes(lookup_signature(shown, reference)), answers))
        augmented.append(
            _partition_score(_classes(augmented_signature(shown, reference)), answers)
        )
    return Room(ceiling=ceiling, floors=tuple(floors), augmented=tuple(augmented))


# --------------------------------------------------------------------------
# aliases, movement and the no-receipt optimum
# --------------------------------------------------------------------------


def distinct_keys(keys: Sequence[Sequence[str]]) -> int:
    """How many of the support's answer vectors are distinguishable from each other."""
    return len({tuple(key) for key in keys})


def movement(keys: Sequence[Sequence[str]]) -> dict[str, tuple[int, float]]:
    """Per axis, the fewest and the mean rows one option change moves.

    Minimized over EVERY draw and every alternative option, not under the draw this
    instance happens to have made. An axis that moves four rows somewhere in the support
    and twenty everywhere else is an axis a reader can be unlucky with, and the bar is
    about the unlucky case.
    """
    out: dict[str, tuple[int, float]] = {}
    for axis in AXES:
        moved: list[int] = []
        for position in range(len(ALL_CONVENTIONS)):
            for option in axis.options:
                if option == ALL_CONVENTIONS[position][axis.name]:
                    continue
                other = _substituted(position, axis.name, option)
                moved.append(
                    sum(1 for x, y in zip(keys[position], keys[other]) if x != y)
                )
        out[axis.name] = (min(moved), sum(moved) / float(len(moved)))
    return out


def no_receipt_optimum(keys: Sequence[Sequence[str]]) -> float:
    """What a careful reader with no receipt scores: each row's modal daughter.

    Scoring is row additive, so the best filing without feedback is the per-row mode
    over the support and its expected grade is the mode's share. Choosing one whole
    convention and answering as it would is generally worse, and is not the baseline
    this prices.
    """
    rows = len(keys[0]) if keys else 0
    if not rows:
        return 0.0
    total = 0.0
    for row in range(rows):
        counts: dict[str, int] = {}
        for key in keys:
            counts[key[row]] = counts.get(key[row], 0) + 1
        total += max(counts.values()) / float(len(keys))
    return total / rows


def modal_filing(keys: Sequence[Sequence[str]]) -> tuple[str, ...]:
    """The reproducible no-receipt filing: each row's modal daughter, ties broken low.

    The tie rule is diagnostic and is not a hidden task decision: modal ties all earn
    the same expected grade, and taking the lexicographically smallest makes a
    diagnostic filing the same filing twice.
    """
    rows = len(keys[0]) if keys else 0
    out: list[str] = []
    for row in range(rows):
        counts: dict[str, int] = {}
        for key in keys:
            counts[key[row]] = counts.get(key[row], 0) + 1
        best = max(counts.values())
        out.append(min(value for value, count in counts.items() if count == best))
    return tuple(out)


# --------------------------------------------------------------------------
# the character copy family, computed here as well
# --------------------------------------------------------------------------


def character_copy_maximum(source: Sequence[str], target: Sequence[str]) -> float:
    """The best a global character map plus a row move can earn on the sibling.

    Computed here rather than through the production screen, because the construction
    filter has to ask it of every draw in the support and the screen asks it of one.
    The families are the same two: the 36 maps that permute the deletable vowels and the
    replacement phones among themselves, and the 2n row moves the rotation and the
    reversal generate.
    """
    width = len(target)
    if not width:
        return 0.0
    wanted = set(target)
    best = 0.0
    for table in _CHARACTER_MAPS:
        mapped = ["".join(table.get(ch, ch) for ch in value) for value in source]
        # A map whose image shares no value with the target matches on no position under
        # any row move, so the row moves are enumerated only when something could match.
        if not (set(mapped) & wanted):
            continue
        fitted = (mapped + [""] * width)[:width]
        agreement = [
            [fitted[i] == target[j] for j in range(width)] for i in range(width)
        ]
        for base in (list(range(width)), list(reversed(range(width)))):
            for shift in range(width):
                moved = base[shift:] + base[:shift]
                hits = sum(1 for j in range(width) if agreement[moved[j]][j])
                best = max(best, hits / float(width))
    return best


_CHARACTER_MAPS: list[dict[str, str]] = []
for _vowels in itertools.permutations(("a", "e", "i")):
    for _phones in itertools.permutations(("g", "x", "s")):
        _table: dict[str, str] = dict(zip(("a", "e", "i"), _vowels))
        _table.update(zip(("g", "x", "s"), _phones))
        _CHARACTER_MAPS.append(_table)


# --------------------------------------------------------------------------
# the skeleton transfer
# --------------------------------------------------------------------------

KEEP, REPLACE, DELETE = "keep", "replace", "delete"
#: The operation order the lexicographically first alignment walks, which is the order
#: these three words sort in.
OPERATION_ORDER = (KEEP, REPLACE, DELETE)


def skeleton(proto: str) -> str:
    """The form after the public pass, with every inert consonant written as C.

    Vowels and k stay literal because they are what the hidden rule looks at; every
    other consonant is one symbol because the rule never distinguishes them. An
    assimilated mb is therefore CC, and two forms with one skeleton undergo the same
    operations at the same positions under every convention.
    """
    word = audit_nasal(proto)
    return "".join(
        character if character in VOWELS or character == "k" else "C"
        for character in word
    )


def align(source: str, target: str) -> tuple[tuple[str, str], ...] | None:
    """The position operations that carry `source` to `target`, or None if none do.

    Three operations and no others: keep consumes one phone from each and they have to
    be equal, replace consumes a k and the phone it became, delete consumes one source
    vowel and nothing from the target. Every candidate alignment consumes both strings
    completely. Several can exist, so the walk is depth first over source positions left
    to right, trying keep, then replace, then delete, and the first complete sequence it
    finds is the lexicographically first one under that order.
    """
    stack: list[tuple[int, int, tuple[tuple[str, str], ...]]] = [(0, 0, ())]
    while stack:
        i, j, operations = stack.pop()
        if i == len(source) and j == len(target):
            return operations
        options: list[tuple[int, int, tuple[tuple[str, str], ...]]] = []
        if i < len(source) and j < len(target) and source[i] == target[j]:
            options.append((i + 1, j + 1, operations + ((KEEP, ""),)))
        if (
            i < len(source)
            and j < len(target)
            and source[i] == "k"
            and target[j] in _INTRODUCED
        ):
            options.append((i + 1, j + 1, operations + ((REPLACE, target[j]),)))
        if i < len(source) and source[i] in VOWELS:
            options.append((i + 1, j, operations + ((DELETE, ""),)))
        stack.extend(reversed(options))
    return None


_INTRODUCED = frozenset(introduced_phone(option) for option in AXES[0].options)


def transfer(operations: Sequence[tuple[str, str]], source: str) -> str | None:
    """The same operations applied at the same positions of another form."""
    out: list[str] = []
    position = 0
    for kind, produced in operations:
        if position >= len(source):
            return None
        if kind == KEEP:
            out.append(source[position])
        elif kind == REPLACE:
            out.append(produced)
        position += 1
    if position != len(source):
        return None
    return "".join(out)


def analogy_score(
    source_protos: Sequence[str],
    source_key: Sequence[str],
    target_protos: Sequence[str],
    target_key: Sequence[str],
) -> float:
    """What an optimistic reader earns by reusing A's edits on the forms that match.

    For each target row it takes every source row with exactly the same skeleton,
    aligns that source form to its own corrected daughter, transfers the position
    operations, and is allowed the best of the candidates on that row. Where no
    skeleton matches, the unchanged form after the public pass is the only candidate.
    This is a specified lookup baseline. It is not a claim that these transformations
    are the registered copy family, and it does not exhaust what a reader might try.
    """
    by_skeleton: dict[str, list[int]] = {}
    for index, proto in enumerate(source_protos):
        by_skeleton.setdefault(skeleton(proto), []).append(index)
    hits = 0
    for index, proto in enumerate(target_protos):
        shape = skeleton(proto)
        after_nasal = audit_nasal(proto)
        candidates: set[str] = set()
        for source_index in by_skeleton.get(shape, ()):
            operations = align(
                audit_nasal(source_protos[source_index]), source_key[source_index]
            )
            if operations is None:
                raise ValueError(
                    "a generated form and its own daughter admit no alignment under the "
                    "registered edit model, so the analogy baseline cannot be priced"
                )
            carried = transfer(operations, after_nasal)
            if carried is not None:
                candidates.add(carried)
        if not candidates:
            candidates.add(after_nasal)
        if target_key[index] in candidates:
            hits += 1
    return hits / float(len(target_protos)) if target_protos else 0.0


def worst_analogy(
    source_protos: Sequence[str], target_protos: Sequence[str]
) -> tuple[float, int]:
    """The analogy baseline's worst score over the whole support, and where."""
    source_keys = support_keys(tuple(source_protos))
    target_keys = support_keys(tuple(target_protos))
    best = 0.0
    at = 0
    for position in range(len(ALL_CONVENTIONS)):
        score = analogy_score(
            source_protos, source_keys[position], target_protos, target_keys[position]
        )
        if score > best:
            best, at = score, position
    return best, at


# --------------------------------------------------------------------------
# the construction filter
# --------------------------------------------------------------------------


def grammar_refusal(protos: Sequence[str]) -> str:
    """What is wrong with a batch as a batch, before anything is asked of the pair."""
    if len(protos) != ROWS:
        return f"a batch is {ROWS} forms and this one is {len(protos)}"
    if len(set(protos)) != len(protos):
        return "a batch repeats a form"
    for proto in protos:
        if not proto or len(proto) > MAX_PROTO:
            return f"the form {proto!r} is not between one and {MAX_PROTO} phones"
        if any(character not in VOWELS + CONSONANTS for character in proto):
            return f"the form {proto!r} carries a character outside the inventory"
        if not set("aei") <= set(proto):
            return f"the form {proto!r} does not carry all three deletable vowels"
    if sum(1 for proto in protos if "nb" in proto) < MIN_CLUSTER_ROWS:
        return f"a batch carries fewer than {MIN_CLUSTER_ROWS} forms with the nb cluster"
    return ""


def side_refusal(protos: Sequence[str]) -> str:
    """What is wrong with one batch's answers over the whole support."""
    keys = support_keys(tuple(protos))
    if distinct_keys(keys) != len(ALL_CONVENTIONS):
        return (
            f"the batch separates {distinct_keys(keys)} of {len(ALL_CONVENTIONS)} draws"
        )
    for axis, (low, _) in movement(keys).items():
        if low < MIN_MOVED_ROWS:
            return f"changing {axis} moves {low} rows somewhere in the support"
    for key in keys:
        for value in key:
            if not value:
                return "a draw produces an empty daughter form"
            if len(value) > MAX_DAUGHTER:
                return f"the daughter {value!r} is longer than {MAX_DAUGHTER} phones"
    optimum = no_receipt_optimum(keys)
    if optimum > MAX_NO_RECEIPT:
        return f"a reader with no receipt scores {optimum:.6f} on the batch"
    return ""


def direction_refusal(shown: np.ndarray, answers: np.ndarray, name: str) -> str:
    """What is wrong with what one side's receipt leaves to do on the other."""
    measured = room(shown, answers)
    if abs(measured.ceiling - 1.0) > 1e-12:
        return f"{name}: the full corrected key reaches {measured.ceiling:.6f}, not 1"
    if measured.worst_floor > MAX_LOOKUP_FLOOR:
        return f"{name}: the lookup floor reaches {measured.worst_floor:.6f}"
    if measured.ceiling - measured.worst_floor <= MIN_LOOKUP_ROOM:
        return f"{name}: the room above the lookup floor is {measured.ceiling - measured.worst_floor:.6f}"
    if measured.worst_augmented > MAX_LOOKUP_FLOOR:
        return f"{name}: the augmented floor reaches {measured.worst_augmented:.6f}"
    if measured.ceiling - measured.worst_augmented <= MIN_LOOKUP_ROOM:
        return (
            f"{name}: the room above the augmented floor is "
            f"{measured.ceiling - measured.worst_augmented:.6f}"
        )
    return ""


def pair_refusal(a_protos: tuple[str, ...], b_protos: tuple[str, ...]) -> str:
    """Why this candidate pair cannot be served, or the empty string if it can.

    Cheap predicates first, so an attempt that fails on four missing clusters does not
    pay for the support-wide partitions. Nothing here reads a convention.
    """
    for name, protos in (("A", a_protos), ("B", b_protos)):
        refusal = grammar_refusal(protos)
        if refusal:
            return f"{name}: {refusal}"
    if set(a_protos) & set(b_protos):
        return "the two batches share a form"
    for name, protos in (("A", a_protos), ("B", b_protos)):
        refusal = side_refusal(protos)
        if refusal:
            return f"{name}: {refusal}"
    a_codes = answer_codes(support_keys(a_protos))
    b_codes = answer_codes(support_keys(b_protos))
    for name, shown, answers in (
        ("A to B", a_codes, b_codes),
        ("B to A", b_codes, a_codes),
    ):
        refusal = direction_refusal(shown, answers, name)
        if refusal:
            return refusal
    for name, source, target in (
        ("A to B", a_protos, b_protos),
        ("B to A", b_protos, a_protos),
    ):
        score, _ = worst_analogy(source, target)
        if score > MAX_TRANSFER:
            return f"{name}: the skeleton transfer earns {score:.6f}"
    a_keys = support_keys(a_protos)
    b_keys = support_keys(b_protos)
    for name, source_keys, target_keys in (
        ("A to B", a_keys, b_keys),
        ("B to A", b_keys, a_keys),
    ):
        for position in range(len(ALL_CONVENTIONS)):
            score = character_copy_maximum(source_keys[position], target_keys[position])
            if score > MAX_TRANSFER:
                return f"{name}: a character map earns {score:.6f} on one draw"
    return ""


# --------------------------------------------------------------------------
# the three named checks this profile adds
# --------------------------------------------------------------------------


def _protos(table: SoundTable) -> tuple[str, ...]:
    return tuple(row.proto for row in table.rows)


def _read_back(generator, instance: Instance, side: str) -> str:
    """What the serialized graded cell says the identifiers and corrections are.

    The serializer fits a field by truncating it rather than refusing it, so a value
    longer than its registered slot becomes a shorter legal looking one and every other
    check still passes. This reads the columns back out of the bytes the agent would
    receive and compares them with the values they were built from.
    """
    task = instance.side(side)
    envelope = frozen_envelope(instance.envelope)
    raw = "\n".join(
        "%s,%s" % (identifier, "")
        for identifier in generator.row_identifiers(task.table)
    )
    canonical = generator.parse_and_canonicalize(task, raw)
    ast = generator.render_receipt(
        task, canonical, task.key, feedback_for(generator, task, envelope)
    )
    payload = serialize(ast, envelope)
    identifier_start = len(GAP) + ORDINAL_WIDTH + len(GAP)
    identifier_end = identifier_start + envelope.identifier_width
    low, high = envelope.slot_span("correction")
    for line, row in zip(row_lines(payload, ast, envelope), ast.rows):
        printed_id = line[identifier_start:identifier_end].decode("ascii").strip()
        printed_correction = line[low:high].decode("ascii").strip()
        wanted = dict((slot.name, slot.value) for slot in row.slots)["correction"]
        if printed_id != row.identifier:
            return (
                f"side {side.upper()} prints the identifier {printed_id!r} where the row "
                f"names {row.identifier!r}"
            )
        if printed_correction != wanted:
            return (
                f"side {side.upper()} prints the correction {printed_correction!r} where "
                f"the row's own answer is {wanted!r}"
            )
    return ""


def check_cascade(generator, instance: Instance) -> CheckResult:
    """The production cascade, recomputed by a second implementation on every draw.

    It fails if any of the 36 cascades disagrees with this validator on either side, if
    a form leaves the declared grammar, if an answer is empty or longer than the
    registered correction slot, if the two batches intersect, if the 36 answer vectors
    are not distinct, if one option change moves fewer than four rows somewhere in the
    support, or if a printed identifier or correction changes under serialize and read
    back.
    """
    reports: list[str] = []
    for side in ("a", "b"):
        table = instance.side(side).table
        protos = _protos(table)
        refusal = grammar_refusal(protos)
        if refusal:
            return CheckResult("cascade", False, f"side {side.upper()}: {refusal}")
        refusal = side_refusal(protos)
        if refusal:
            return CheckResult("cascade", False, f"side {side.upper()}: {refusal}")
        for convention in ALL_CONVENTIONS:
            produced = tuple(generator.key_for(table, convention))
            expected = audit_key(protos, convention)
            if produced != expected:
                wrong = next(
                    n for n, (x, y) in enumerate(zip(produced, expected)) if x != y
                )
                return CheckResult(
                    "cascade", False,
                    f"side {side.upper()} row {wrong + 1} ({protos[wrong]!r}) is scored "
                    f"{produced[wrong]!r} and an independent cascade makes it "
                    f"{expected[wrong]!r}",
                )
        reports.append(
            "%s %d forms, longest %d, longest daughter %d"
            % (
                side.upper(), len(protos), max(len(p) for p in protos),
                max(len(v) for key in support_keys(protos) for v in key),
            )
        )
    if set(_protos(instance.a.table)) & set(_protos(instance.b.table)):
        return CheckResult("cascade", False, "the two batches share a form")
    for side in ("a", "b"):
        wrong = _read_back(generator, instance, side)
        if wrong:
            return CheckResult("cascade", False, wrong)
    return CheckResult(
        "cascade", True,
        "an independent cascade agrees with the scorer on all %d draws of both sides; %s"
        % (len(ALL_CONVENTIONS), "; ".join(reports)),
    )


def check_phone_lookup(generator, instance: Instance) -> CheckResult:
    """What is left once the reflex and the deleted vowel are given away for free.

    It fails if handing a reader both directly readable axes, on top of the lookup
    observations the gate's floor already concedes, leaves the sibling all but solved:
    the augmented floor has to stay at or under 0.90 and the room above it above 0.05,
    at every reference draw and in both directions. A family that cleared H on room a
    reader can reach by reading two subword correspondences has the room the gate
    measured and not the room the release would describe.
    """
    a_codes = answer_codes(support_keys(_protos(instance.a.table)))
    b_codes = answer_codes(support_keys(_protos(instance.b.table)))
    lines: list[str] = []
    for name, shown, answers in (
        ("A to B", a_codes, b_codes),
        ("B to A", b_codes, a_codes),
    ):
        measured = room(shown, answers)
        if measured.worst_augmented > MAX_LOOKUP_FLOOR:
            return CheckResult(
                "phone_lookup", False,
                f"{name}: giving the reflex and the deleted vowel away puts the floor at "
                f"{measured.worst_augmented:.6f}, over the registered "
                f"{MAX_LOOKUP_FLOOR:.4f}",
            )
        room_left = measured.ceiling - measured.worst_augmented
        if room_left <= MIN_LOOKUP_ROOM:
            return CheckResult(
                "phone_lookup", False,
                f"{name}: {room_left:.6f} is left above the augmented floor, under the "
                f"registered {MIN_LOOKUP_ROOM:.4f}, so what the receipt adds is the two "
                "axes a reader can already read off the phones",
            )
        lines.append(
            "%s ceiling %.6f, augmented floor %.6f to %.6f, room %.6f"
            % (
                name, measured.ceiling, min(measured.augmented),
                measured.worst_augmented, room_left,
            )
        )
    return CheckResult("phone_lookup", True, "; ".join(lines))


def check_analogy(generator, instance: Instance) -> CheckResult:
    """What reusing A's own edits earns on the forms of B that look like A's.

    It fails if an optimistic reader allowed the best skeleton transfer on every row
    earns more than 0.50 on the sibling under any draw, in either direction, or if a
    generated form and its own daughter admit no alignment at all. The disjoint final
    consonants are not what this measures and are not evidence by themselves: a batch
    made by renaming A's inert consonants would keep every skeleton and is what this
    refuses.
    """
    a_protos = _protos(instance.a.table)
    b_protos = _protos(instance.b.table)
    lines: list[str] = []
    for name, source, target in (
        ("A to B", a_protos, b_protos),
        ("B to A", b_protos, a_protos),
    ):
        try:
            score, position = worst_analogy(source, target)
        except ValueError as exc:
            return CheckResult("analogy", False, str(exc))
        if score > MAX_TRANSFER:
            return CheckResult(
                "analogy", False,
                f"{name}: the skeleton transfer earns {score:.6f} under one draw, over "
                f"the registered {MAX_TRANSFER:.4f}",
            )
        shared = len(
            {skeleton(p) for p in source} & {skeleton(p) for p in target}
        )
        lines.append(
            "%s worst %.6f at draw %d, %d shared skeletons"
            % (name, score, position, shared)
        )
    return CheckResult("analogy", True, "; ".join(lines))


# --------------------------------------------------------------------------
# the detail report, for the review pack and the roster release
# --------------------------------------------------------------------------


def pair_report(a_protos: Sequence[str], b_protos: Sequence[str]) -> dict[str, object]:
    """Everything the construction filter measured about one pair, for a human.

    The roster release has to carry the exact minima and the alternative means of each
    table rather than the construction lower bound, because four is what a pair had to
    clear and not what it reached.
    """
    a_keys = support_keys(tuple(a_protos))
    b_keys = support_keys(tuple(b_protos))
    a_codes = answer_codes(a_keys)
    b_codes = answer_codes(b_keys)
    out: dict[str, object] = {}
    for name, protos, keys in (("a", a_protos, a_keys), ("b", b_protos, b_keys)):
        out[name] = {
            "forms": list(protos),
            "cluster_rows": sum(1 for p in protos if "nb" in p),
            "longest_form": max(len(p) for p in protos),
            "longest_daughter": max(len(v) for key in keys for v in key),
            "distinct_keys": distinct_keys(keys),
            "movement": {
                axis: {"fewest": low, "mean": mean}
                for axis, (low, mean) in movement(keys).items()
            },
            "no_receipt_optimum": no_receipt_optimum(keys),
            "modal_filing": list(modal_filing(keys)),
        }
    for name, shown, answers in (
        ("a_to_b", a_codes, b_codes),
        ("b_to_a", b_codes, a_codes),
    ):
        measured = room(shown, answers)
        out[name] = {
            "ceiling": measured.ceiling,
            "lookup_floor": {
                "lowest": min(measured.floors), "highest": measured.worst_floor
            },
            "augmented_floor": {
                "lowest": min(measured.augmented), "highest": measured.worst_augmented
            },
            "room_above_augmented": measured.ceiling - measured.worst_augmented,
        }
    for name, source, target in (
        ("a_to_b", a_protos, b_protos), ("b_to_a", b_protos, a_protos)
    ):
        score, position = worst_analogy(source, target)
        entry = out[name]
        assert isinstance(entry, dict)
        entry["analogy"] = {"worst": score, "at_draw": position}
        source_keys = support_keys(tuple(source))
        target_keys = support_keys(tuple(target))
        entry["character_copy"] = max(
            character_copy_maximum(source_keys[i], target_keys[i])
            for i in range(len(ALL_CONVENTIONS))
        )
    out["shared_forms"] = sorted(set(a_protos) & set(b_protos))
    return out


def intermediate_forms(proto: str, convention: Mapping[str, str]) -> dict[str, str]:
    """The proto, the form after each pass and the daughter, for the private worksheet.

    These belong in the human's explanatory pack and nowhere else. No cell prints them,
    and a reader of a receipt sees only whole daughter forms.
    """
    after_nasal = audit_nasal(proto)
    if replaces_first(convention["sequence"]):
        first = audit_replace(after_nasal, convention["reflex"], convention["environment"])
        second = audit_delete(first, convention["loss"])
        first_name, second_name = "after_replacement", "after_deletion"
    else:
        first = audit_delete(after_nasal, convention["loss"])
        second = audit_replace(first, convention["reflex"], convention["environment"])
        first_name, second_name = "after_deletion", "after_replacement"
    return {
        "proto": proto,
        "after_nasal": after_nasal,
        first_name: first,
        second_name: second,
        "daughter": second,
        "skeleton": skeleton(proto),
    }


__all__ = [
    "MAX_LOOKUP_FLOOR",
    "MAX_NO_RECEIPT",
    "MAX_PROTO",
    "MAX_TRANSFER",
    "MIN_CLUSTER_ROWS",
    "MIN_LOOKUP_ROOM",
    "MIN_MOVED_ROWS",
    "Room",
    "align",
    "analogy_score",
    "answer_codes",
    "audit_daughter",
    "audit_delete",
    "audit_key",
    "audit_nasal",
    "audit_replace",
    "augmented_signature",
    "character_copy_maximum",
    "check_analogy",
    "check_cascade",
    "check_phone_lookup",
    "distinct_keys",
    "evident_rows",
    "grammar_refusal",
    "intermediate_forms",
    "introduced_phone",
    "lookup_signature",
    "modal_filing",
    "movement",
    "no_receipt_optimum",
    "pair_refusal",
    "pair_report",
    "removed_vowel",
    "replaces_first",
    "room",
    "row_dependence",
    "side_refusal",
    "skeleton",
    "support_keys",
    "transfer",
    "two_sided",
    "worst_analogy",
]
