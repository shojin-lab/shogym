"""Render one instance under every convention, and hand the gates what it printed.

This is the bridge between a generator and `shogym.receipts`. It runs the real
renderer and the real serializer once per convention in the space, and reduces the
result to what a reader could actually distinguish: per printed row, the tuple of
slot values that row shows.

WHAT THE AGENT IS ASSUMED TO HAVE DONE. The receipt is read by an agent deciding
what to do differently next time given what it did this time, so every counterfactual
is rendered against ONE fixed filing: the canonical filing under the drawn
convention, which is the filing of an agent that applied the drawn rule exactly.
Holding the filing fixed while the truth varies is what makes the printed
differences attributable to the convention rather than to the agent.

The cost is one render and one serialize per convention, and no execution at all.
For a family of 72 conventions over 24 rows that is milliseconds.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from shogym.envs.receipts.protocol import (
    ROW_ADDITIVE_EQUAL_WEIGHT,
    Filing,
    Generator,
    Instance,
    Task,
    support_of,
)
from shogym.envs.receipts.receipt_ast import row_lines, serialize
from shogym.receipts import AxisSpace, Observation


def axis_space(generator: Generator) -> AxisSpace:
    """The generator's axis catalogue, as the gates want it."""
    return AxisSpace(
        axes=tuple(a.name for a in generator.AXES),
        options=tuple(tuple(a.options) for a in generator.AXES),
    )


def canonical_filing_text(generator: Generator, task: Task) -> str:
    """What an agent that applied the drawn convention exactly would have filed."""
    return "\n".join(
        f"{identifier},{value}"
        for identifier, value in zip(generator.row_identifiers(task.table), task.key)
    )


def _codes(views: Sequence[Sequence[bytes]]) -> np.ndarray:
    """Map printed row observations to small integers, one vocabulary per column."""
    if not views or not views[0]:
        return np.zeros((len(views), 0), dtype=np.int64)
    n_rows = len(views[0])
    out = np.zeros((len(views), n_rows), dtype=np.int64)
    for row in range(n_rows):
        vocabulary: dict[bytes, int] = {}
        for j, view in enumerate(views):
            value = view[row]
            out[j, row] = vocabulary.setdefault(value, len(vocabulary))
    return out


def _row_count(values: Sequence[object]) -> int:
    """How many rows the rendered cells carry, from the first one that exists."""
    for item in values:
        if item is not None:
            return len(item)  # type: ignore[arg-type]
    return 0


def _whole_codes(payloads: Sequence[bytes]) -> np.ndarray:
    """One integer per convention, standing for the whole serialized cell.

    Gate R asks which options an agent can tell apart, and an agent reads the cell,
    not one region of it. Coding the whole payload is what makes a rule smuggled into
    the wrapper, the body or the padding count as something the receipt says.
    """
    vocabulary: dict[bytes, int] = {}
    return np.array(
        [vocabulary.setdefault(p, len(vocabulary)) for p in payloads], dtype=np.int64
    )


def _answer_codes(
    keys: Sequence[Sequence[str]], normalize: Callable[[str], str]
) -> np.ndarray:
    """Map the sibling task's answers to small integers, one vocabulary per row.

    Normalized first. Two spellings the scorer treats as one answer are one legal
    action, and coding them apart would let the gate report room in a distinction
    the agent cannot act on.
    """
    if not keys or not keys[0]:
        return np.zeros((len(keys), 0), dtype=np.int64)
    n_rows = len(keys[0])
    out = np.zeros((len(keys), n_rows), dtype=np.int64)
    for row in range(n_rows):
        vocabulary: dict[str, int] = {}
        for j, key in enumerate(keys):
            out[j, row] = vocabulary.setdefault(normalize(key[row]), len(vocabulary))
    return out


def observe(
    generator: Generator,
    instance: Instance,
    side: str = "a",
    filing: Filing | None = None,
) -> Observation:
    """What this instance's receipt prints, under every convention in the space.

    `side` is the task the receipt grades; the reader is scored on the other one,
    which is the sibling the whole family relation exists to connect.
    """
    declared = getattr(generator, "SCORING", None)
    if declared != ROW_ADDITIVE_EQUAL_WEIGHT:
        raise ValueError(
            f"{generator.name!r} declares scoring {declared!r}; the shared gate prices "
            f"only {ROW_ADDITIVE_EQUAL_WEIGHT!r}, and gating a different objective "
            "would report room the family does not have. Register an exact optimizer "
            "for that scorer before gating it."
        )
    space = axis_space(generator)
    index = space.index()
    # The sampler's real support, which is the product for every ordinary family and
    # smaller for the correlated exhibit. Gating the advertised product instead would
    # gate a space the sampler cannot reach.
    support = support_of(generator)
    task = instance.side(side)
    sibling = instance.side("b" if side.strip().lower() == "a" else "a")
    envelope = instance.envelope

    canonical = filing or generator.parse_and_canonicalize(
        task, canonical_filing_text(generator, task)
    )

    # Indexed by the space's own position, never by iteration order: the two happen
    # to agree, and a gate that depended on them agreeing would be a gate that broke
    # silently the day an enumeration changed.
    views: list[tuple[bytes, ...] | None] = [None] * space.n_combos
    whole: list[bytes | None] = [None] * space.n_combos
    reachable: list[int] = []
    keys: list[tuple[str, ...] | None] = [None] * space.n_combos
    payloads: dict[int, bytes] = {}
    orders: dict[int, tuple[str, ...]] = {}
    graded_keys: list[tuple[str, ...]] = []
    realized: dict[str, set[str]] = {spec.name: set() for spec in envelope.slots}
    for convention in support:
        position = index[tuple(convention[a.name] for a in generator.AXES)]
        truth = tuple(generator.key_for(task.table, convention))
        graded_keys.append(truth)
        reachable.append(position)
        ast = generator.render_receipt(task, canonical, truth)
        for row in ast.rows:
            for slot in row.slots:
                realized.setdefault(slot.name, set()).add(slot.value)
        payload = serialize(ast, envelope)
        views[position] = row_lines(payload, ast, envelope)
        whole[position] = payload
        orders[position] = tuple(row.identifier for row in ast.rows)
        payloads[position] = payload
        keys[position] = tuple(generator.key_for(sibling.table, convention))
    if any(views[j] is None or keys[j] is None for j in reachable):
        raise ValueError("the convention enumeration did not cover the sampler's support")
    # Unreachable conventions keep their slot in every array so that an index still
    # means the same convention everywhere, and carry a mask saying the sampler
    # cannot draw them. They are excluded from every partition and every score: a
    # reader is choosing among the rules that can occur, not among the rules the
    # axis catalogue could spell.
    reached = np.zeros(space.n_combos, dtype=bool)
    reached[np.asarray(reachable, dtype=int)] = True
    blank_row: tuple[bytes, ...] = tuple(b"" for _ in range(_row_count(views)))
    blank_key: tuple[str, ...] = ("",) * _row_count(keys)
    filled_views: list[tuple[bytes, ...]] = [
        v if v is not None else blank_row for v in views
    ]
    filled_whole: list[bytes] = [w if w is not None else b"" for w in whole]
    filled_keys: list[tuple[str, ...]] = [
        k if k is not None else blank_key for k in keys
    ]
    # The graded task's own answer vocabulary, which is what a correction slot is
    # entitled to print. The sibling's answers are a different vocabulary and are
    # not licensed here.
    answers_seen = frozenset(value for key in graded_keys for value in key)

    return Observation(
        space=space,
        chat=index[tuple(instance.convention[a.name] for a in generator.AXES)],
        shown=_codes(filled_views),
        whole=_whole_codes(filled_whole),
        reachable=reached,
        answers=_answer_codes(filled_keys, generator.normalize_answer),
        identifiers=tuple(generator.row_identifiers(task.table)),
        labels=tuple(generator.row_label(task.table)),
        orders=orders,
        payloads=payloads,
        answer_vocabulary=answers_seen,
        slot_grammar={
            spec.name: spec.allowed(answers_seen) for spec in envelope.slots
        },
        slot_realized={k: frozenset(v) for k, v in realized.items()},
        tag=f"{instance.generator}/{instance.ordinal}/{task.label}",
    )


__all__ = ["axis_space", "canonical_filing_text", "observe"]
