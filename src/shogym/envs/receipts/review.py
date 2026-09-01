"""The frozen review pack: what a human has to have looked at, and how that is checked.

A person reading rendered instances is the boundary against a family that is wrong in
a way no mechanical check can see. That only holds if the person saw the family, and a
pack that names one artifact has not shown them the family: it has shown them one
draw, under one convention, on one surface, against one filing.

So the coverage is enumerated from the generator's own declarations rather than
listed by whoever wrote the pack. Every surface template the family can draw, every
option of every axis, every registered filing shape, every row count the bank holds,
and at least one counterfactual render under a convention that was not the one drawn.
A pack missing a category fails, and so does one whose artifact for a category is too
small to be the thing it claims to be: a rendered cell is the envelope size, and a
task text is hundreds of bytes, so a single byte is not either.

None of this establishes that the reading was careful. It establishes that the
material was in front of the reader, which is the part a machine can check.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from shogym.envs.receipts.protocol import Generator

#: The categories a pack has to cover. Each is enumerated from what the family
#: declares, so adding an axis or a surface adds a required render rather than
#: leaving the pack as it was.
CATEGORIES = ("surface", "option", "filing", "rows", "counterfactual")

#: What kind of artifact a render entry is, and the smallest it can plausibly be. A
#: cell's floor is the family's own envelope size; a task text's is a few hundred
#: bytes, which no real schedule is under.
KINDS = ("cell", "task")
MIN_TASK_BYTES = 400

#: Fields the manifest carries, and the whole of them. Exactly this set: a field the
#: verifier ignores is a field a stale conclusion travels in, and a reader who finds
#: `reviewed: true` beside a coverage list has no way to know nothing checked it.
#:
#: `family` and `bank` are what the reading was OF. A pack that named neither is a
#: pack any bundle can claim: two banks under two master keys draw two different
#: conventions, and a human who read renders under the first has attested to nothing
#: about the second.
REQUIRED_FIELDS = ("reviewer", "checklist", "seeds", "family", "bank", "renders")

#: What one render entry says. The bytes are the bundle's business, so an entry says
#: only what was read and where it sits; a digest here would be a second copy of one
#: the manifest already holds, and two copies of a hash is one of them going stale.
RENDER_FIELDS = ("category", "key", "kind", "path")


@dataclass(frozen=True)
class Coverage:
    """What this family's pack has to contain, derived from its declarations."""

    required: tuple[tuple[str, str], ...]

    def missing(self, seen: Sequence[tuple[str, str]]) -> list[str]:
        have = set(seen)
        return [f"{kind}:{key}" for kind, key in self.required if (kind, key) not in have]


def required_coverage(
    generator: Generator, filing_classes: Sequence[str], row_counts: Sequence[int]
) -> Coverage:
    """Everything a pack for this family has to show, enumerated from its own SHAPE and AXES."""
    required: list[tuple[str, str]] = []
    for surface in generator.surface_templates():
        required.append(("surface", surface))
    for axis in generator.AXES:
        for option in axis.options:
            required.append(("option", f"{axis.name}={option}"))
    for shape in filing_classes:
        required.append(("filing", shape))
    for count in sorted(set(int(c) for c in row_counts)):
        required.append(("rows", str(count)))
    required.append(("counterfactual", "alternative convention"))
    return Coverage(required=tuple(required))


def identity(value: object, what: str) -> str:
    """One identity out of a stored pack, refused rather than defaulted when absent.

    `str(None)` is the nonempty string "None", so a pack export that lost the attesting
    person writes a null and one conversion later has an ordinary-looking reviewer
    nobody can find. Numbers are allowed, because a seed is often one. A null, a blank,
    a structure or a nonfinite float is a missing identity, which is what this catches.
    """
    if value is None:
        raise ValueError(f"the review pack names no {what}")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"the review pack's {what} is not a name")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"the review pack's {what} is {value!r}, which names nothing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"the review pack's {what} is blank")
    return text


def identities(values: object, what: str) -> list[str]:
    """A list of identities, each refused on the same rule."""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError(f"the review pack lists no {what}")
    return [identity(value, f"{what} {n}") for n, value in enumerate(values)]


def verify(
    manifest: Mapping[str, Any],
    coverage: Coverage,
    envelope_size: int,
    files: Mapping[str, int],
    family: str,
    bank_identity: str,
) -> list[str]:
    """What this pack does not establish. Empty when it covers the family.

    `files` is the bundle's own file list, path to size. Every render a pack names has
    to be one of them, which is part of the binding: the bundle's manifest hashed
    those bytes, and the manifest's own hash is the bundle's name. A pack cannot point
    at a render outside the bundle, so there is nothing to rebind and no second digest
    to keep in step with the first.

    The rest of the binding is `family` and `bank_identity`, which the pack has to
    name and which have to be the bundle's own. Being inside the directory says the
    bytes were not swapped after the fact; it does not say the reading was of this
    family and this draw, and a pack read for one bank is otherwise portable to any
    other with renders of the right size.

    WHAT THIS STILL DOES NOT ESTABLISH, and a recorded boundary rather than an
    oversight: that the artifacts are the renders they claim to be. Their size is
    checked and their bytes are not, because the pack is what a person read and a
    person reads whatever the renderer put in front of them. The control against a
    pack of the wrong bytes is the person who signed it.
    """
    problems: list[str] = []
    if set(manifest) != set(REQUIRED_FIELDS):
        return [
            "a review pack names exactly %s, and this one names %s"
            % (", ".join(REQUIRED_FIELDS), ", ".join(sorted(manifest)) or "nothing")
        ]
    try:
        identity(manifest["reviewer"], "reviewer")
        identities(manifest["checklist"], "checklist item")
        identities(manifest["seeds"], "seed")
        read_of = identity(manifest["family"], "family")
        read_from = identity(manifest["bank"], "bank")
    except ValueError as exc:
        return [str(exc)]
    if read_of != family:
        return [
            f"this pack is a read of {read_of!r} and it is being offered as the read "
            f"of {family!r}"
        ]
    if read_from != bank_identity:
        return [
            f"this pack was read from bank {read_from[:12]} and this bundle holds "
            f"{bank_identity[:12]}, which draws its own conventions"
        ]
    renders = manifest.get("renders")
    if not isinstance(renders, list) or not renders:
        return ["the review pack lists no rendered instances that were read"]

    seen: list[tuple[str, str]] = []
    for entry in renders:
        if not isinstance(entry, dict):
            problems.append("a render entry is not a record")
            continue
        if set(entry) != set(RENDER_FIELDS):
            problems.append(
                "a render entry names %s and a render is exactly %s"
                % (", ".join(sorted(entry)) or "nothing", ", ".join(RENDER_FIELDS))
            )
            continue
        category = str(entry["category"])
        key = str(entry["key"])
        kind = str(entry["kind"])
        named = str(entry["path"])
        if category not in CATEGORIES:
            problems.append(f"a render entry names category {category!r}")
            continue
        if kind not in KINDS:
            problems.append(f"the render for {category}:{key} names kind {kind!r}")
            continue
        if named not in files:
            problems.append(
                f"the render for {category}:{key} is not a file this bundle holds"
            )
            continue
        floor = envelope_size if kind == "cell" else MIN_TASK_BYTES
        size = files[named]
        if size < floor:
            problems.append(
                f"the {kind} render for {category}:{key} is {size} bytes, under the "
                f"{floor} a {kind} takes, so it is not one"
            )
            continue
        seen.append((category, key))

    problems.extend(
        f"the review pack covers no {absent}" for absent in coverage.missing(seen)
    )
    return problems


__all__ = [
    "CATEGORIES",
    "KINDS",
    "MIN_TASK_BYTES",
    "REQUIRED_FIELDS",
    "RENDER_FIELDS",
    "Coverage",
    "identities",
    "identity",
    "required_coverage",
    "verify",
]
