"""The oracle cell: one template, rendered and read back by this package.

A generator declares WHAT its options mean, as a phrase per option. It does not
render the oracle and it does not read one. Both are done here, from the same
declared data, so the sentence an oracle child reads and the rule an admission check
believes it states are two views of one table rather than two methods that have to
agree.

WHY THIS IS NOT LEFT TO THE FAMILY. A renderer and a reader supplied together cannot
establish anything about each other: a reader that returns what the renderer was
handed will confirm any sentence at all, including one that states a different rule
or no rule. That is not a claim about anyone's intentions. It is that a round trip
through one author's two functions has no fixed point outside them, so it cannot be
evidence, and the oracle arm is the denominator the whole room measurement is taken
against.

The phrase table is the registration. Adding an option means adding a phrase, and a
phrase that another option's phrase contains is refused when the table is built,
because a reader cannot tell those two apart in prose.
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass
from typing import Mapping, Sequence

from shogym.envs.receipts.receipt_ast import ORACLE, ReceiptAST

#: How wide the rendered rule is filled, and how its items are numbered.
WIDTH = 72
INDENT = "     "


@dataclass(frozen=True)
class OracleTemplate:
    """What each option means, in words, and the frame the rule is stated in.

    `head` is the standing preamble. `sentences` gives, per axis, the sentence frame
    with a single `{}` where the option's phrase goes. `phrases` gives the phrase for
    every option of every axis.
    """

    head: tuple[str, ...]
    sentences: Mapping[str, str]
    phrases: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        for axis, options in self.phrases.items():
            if axis not in self.sentences:
                raise ValueError(f"axis {axis!r} has phrases and no sentence frame")
            wording = list(options.values())
            if len(set(wording)) != len(wording):
                raise ValueError(f"axis {axis!r} gives two options the same phrase")
            for one in wording:
                for other in wording:
                    if one is not other and _flat(one) in _flat(other):
                        raise ValueError(
                            f"axis {axis!r} has a phrase contained in another, so prose "
                            "cannot tell the two options apart"
                        )

    @property
    def axes(self) -> tuple[str, ...]:
        return tuple(self.phrases)


def _flat(text: str) -> str:
    """Prose with its wrapping removed, which is how a phrase is matched."""
    return re.sub(r"\s+", " ", text).strip()


def render_body(template: OracleTemplate, convention: Mapping[str, str]) -> tuple[str, ...]:
    """The rule, stated in words, from the declared phrases alone."""
    lines = list(template.head)
    for position, axis in enumerate(template.axes):
        option = convention[axis]
        try:
            phrase = template.phrases[axis][option]
        except KeyError as exc:
            raise ValueError(f"axis {axis!r} declares no phrase for {option!r}") from exc
        sentence = template.sentences[axis].format(phrase)
        lines.extend(
            textwrap.fill(
                sentence,
                width=WIDTH,
                initial_indent="  %d. " % (position + 1),
                subsequent_indent=INDENT,
            ).split("\n")
        )
    return tuple(lines)


def parse_body(template: OracleTemplate, lines: Sequence[str]) -> dict[str, str]:
    """The convention a rendered oracle states, read out of its own words.

    Wrapping is undone first, because the rule is filled to a width and a phrase can
    straddle a line break. Exactly one option's phrase has to be present on each
    axis: none means the oracle does not state that part of the rule, and two means
    the prose does not say which one it means.
    """
    text = _flat(" ".join(lines))
    out: dict[str, str] = {}
    for axis in template.axes:
        found = [
            option
            for option, phrase in template.phrases[axis].items()
            if _flat(phrase) in text
        ]
        if len(found) != 1:
            raise ValueError(
                f"the oracle states {len(found)} options on {axis}; it has to state one"
            )
        out[axis] = found[0]
    return out


def render(
    template: OracleTemplate,
    task_id: str,
    convention: Mapping[str, str],
    row_count: int = 0,
) -> ReceiptAST:
    """The whole oracle cell. No rows to align, and the rule as its body."""
    return ReceiptAST(
        kind=ORACLE,
        task_id=task_id,
        row_count=row_count,
        body=render_body(template, convention),
    )


def parse(template: OracleTemplate, ast: ReceiptAST) -> dict[str, str]:
    """The convention an oracle cell states."""
    return parse_body(template, ast.body)


__all__ = [
    "INDENT",
    "WIDTH",
    "OracleTemplate",
    "parse",
    "parse_body",
    "render",
    "render_body",
]
