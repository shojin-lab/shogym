"""The sound change genre, second version: a third environment, and 54 conventions.

Everything the first version publishes is here unchanged except one list of options. The
batch grammar, the four surfaces, the public nasal pass, the three reflexes, the three
deleted vowels, the two orders, the receipt that reports two forms of twenty four, the
copy profile, the envelope and the construction bars are the first version's, read from
its module. What is new is a third option on the environment axis:

    reflex        3 options   which phone a matching k is replaced by
    environment   3 options   which neighbours a k needs before it is replaced
    loss          3 options   which vowel is deleted between consonants
    sequence      2 options   whether replacement runs before deletion or after

That is 54 conventions, drawn uniformly and independently. The new environment replaces
a k that has a vowel on its right and no vowel on its left: a k after a consonant, or a k
that begins the form. With the first version's two it sorts every k before a vowel by
what is on its left: a vowel, anything at all, or no vowel.

WHY A SECOND VERSION AND NOT AN EDIT TO THE FIRST. A bank names its generator, its key
and its size, and its population is recomputed by rerunning admission. Widening the
first version's support would redraw every instance an existing bank of it holds, under
the same name. So the first version stays at 36 and this one is registered beside it
under a name of its own, with streams of its own: its forms, row orders, identifiers,
filler, masks and task identifiers are all drawn under `soundchange_v2`, so a bank of
each under one key share only the convention stream every genre shares.

WHY THIS THIRD ENVIRONMENT, AND NOT THE TWO OBVIOUS ONES. A third environment is worth
adding only if both of its orders give answers no other convention gives, on some form
the grammar produces. Two obvious candidates fail that, and both fail for reasons that
are properties of the grammar rather than of any batch.

Replacing every k unconditionally commutes with the deletion, as the first version's
docstring says, so its two orders are one function.

Replacing a k that needs only a vowel on its LEFT fails in a different way. Every k in a
proto form is a cell, and every cell has a vowel on its right, so when replacement runs
first every k it sees has a vowel on its right and "a vowel on the left" decides exactly
what "vowels on both sides" decides. Each of the nine conventions pairing that
environment with replacement first gives the same answer as one pairing both sides with
replacement first, on every form there is.

What any third environment has to do follows from the same fact. Replacement first sees
a k with a vowel on its right and either a vowel or the start of the form on its left,
and the first version's two environments already cover "replace the one with a vowel on
the left" and "replace both". The two behaviours left are "replace neither", under which
the reflex never shows, and "replace only the k that begins the form". So a third
environment has to refuse a k between two vowels and accept a k that begins the form,
and the natural one of those is this one: a vowel on the right and no vowel on the left.
Under replacement first it replaces only a k that begins the form. Under deletion first
it also replaces a k whose left vowel the deletion took, provided the deletion left the
k's right vowel alone. Those are two behaviours no other convention has, so all 54
answer vectors differ, and the tests hold that on words the grammar produces.

WHAT THIS ADDS, AND WHAT IT DOES NOT. The new option sits on the environment axis, and
the environment composed with the order is the part of the rule a reader has to infer.
The reflex and the deleted vowel are still read straight off a corrected daughter, and
`phone_lookup` still gives both away for free before it asks what is left. So the
extension adds to the hard part: the environment by the order is six cells here where
it was four, and nothing a corrected form hands over has changed.

WHAT IT COSTS, MEASURED under the first version's construction filter with the support
widened to 54. Under one key, all of 40 ordinals built a pair: a candidate pair is
accepted on about a third of attempts, where the first version's is accepted on about
three quarters, so a pair takes three attempts on average and eleven at most of the 400
allowed. What the extra attempts are refused for is mostly the augmented floor: a reader
of every corrected form who is also handed the reflex and the deleted vowel reaches more
than 0.90 somewhere in the support. A bank of 24 under another key admitted every
ordinal it considered, at a bank-mean ideal level of 0.8325 against a lookup floor of
0.2169, which is inside the registered band at the first version's two reported forms.
Two is still the only registered count that is: on four pairs one reported form leaves an
ideal reader at 0.70, two at 0.84, three at 0.92 and four at 0.95. The tests hold a small
bank to the band.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Mapping

from shogym.envs.receipts import streams
from shogym.envs.receipts.generators import soundchange
from shogym.envs.receipts.generators import soundchange_audit as audit
from shogym.envs.receipts.generators.soundchange import (
    DELETED_VOWEL,
    FINAL_A,
    FINAL_B,
    LOSS_WORDS,
    MAX_ATTEMPTS,
    ORACLE_HEAD,
    REFLEX_PHONE,
    REFLEX_WORDS,
    REPLACE_FIRST,
    SEQUENCE_WORDS,
    VOWELS,
    SoundTable,
    delete_pass,
    invent_batch,
    nasal_pass,
)
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.protocol import Axis, ConstructionExhausted

NAME = "soundchange_v2"

#: The third environment's option identifier. It names the condition that sets it apart
#: from `right_vowel`, as `right_vowel` names the one that sets it apart from
#: `vowel_pair`, and no task text or oracle sentence prints it.
NO_LEFT_VOWEL = "no_left_vowel"

AXES: tuple[Axis, ...] = (
    soundchange.AXES[0],
    Axis(
        "environment", ("vowel_pair", "right_vowel", NO_LEFT_VOWEL),
        "which neighbours a k needs before it is replaced",
    ),
    soundchange.AXES[2],
    soundchange.AXES[3],
)


# --------------------------------------------------------------------------
# the cascade: the replacement pass, with three environments
# --------------------------------------------------------------------------


def replace_pass(word: str, phone: str, environment: str) -> str:
    """Every k whose neighbours satisfy the drawn environment becomes `phone`.

    A required vowel has to exist, so a k at the end of a form fails all three and a k
    at the start fails the two-sided condition. The third condition asks only that the
    left neighbour is not a vowel, and a word boundary is not a vowel, so a k at the
    start of a form meets it. Every position is tested against the pass input.
    """
    if environment not in AXES[1].options:
        raise ValueError(f"an environment is one of {AXES[1].options}, not {environment!r}")
    out: list[str] = []
    for position, source in enumerate(word):
        if source != "k":
            out.append(source)
            continue
        right = word[position + 1] if position + 1 < len(word) else ""
        left = word[position - 1] if position > 0 else ""
        right_vowel = right != "" and right in VOWELS
        left_vowel = left != "" and left in VOWELS
        if environment == "vowel_pair":
            matched = right_vowel and left_vowel
        elif environment == "right_vowel":
            matched = right_vowel
        else:
            matched = right_vowel and not left_vowel
        out.append(phone if matched else source)
    return "".join(out)


def daughter(proto: str, convention: Mapping[str, str]) -> str:
    """The daughter form one proto takes under one convention."""
    word = nasal_pass(proto)
    phone = REFLEX_PHONE[convention["reflex"]]
    vowel = DELETED_VOWEL[convention["loss"]]
    if convention["sequence"] == REPLACE_FIRST:
        return delete_pass(replace_pass(word, phone, convention["environment"]), vowel)
    return replace_pass(delete_pass(word, vowel), phone, convention["environment"])


def key_for(table: SoundTable, convention: Mapping[str, str]) -> tuple[str, ...]:
    """The correct answer for every row under one convention: one daughter per row."""
    return tuple(daughter(row.proto, convention) for row in table.rows)


ALL_CONVENTIONS: tuple[dict[str, str], ...] = tuple(
    {"reflex": r, "environment": e, "loss": lo, "sequence": s}
    for r in AXES[0].options
    for e in AXES[1].options
    for lo in AXES[2].options
    for s in AXES[3].options
)

#: The support the construction filter quantifies over, as the audit derives it from
#: the declared options rather than from the production passes above.
SUPPORT = audit.support_of_axes(AXES)


# --------------------------------------------------------------------------
# the pair
# --------------------------------------------------------------------------


@lru_cache(maxsize=32)
def build_pair(master: bytes, ordinal: int) -> tuple[SoundTable, SoundTable]:
    """Both sibling tables for one ordinal, from the public draw and nothing else.

    The first version's search, under this version's name and over this version's
    support. The filter is the first version's filter and quantifies over all 54
    conventions, never the one drawn, so acceptance is a function of the two batches
    alone here too.
    """
    for attempt in range(MAX_ATTEMPTS):
        a_protos = invent_batch(
            streams.rng(master, streams.SURFACE_A, NAME, ordinal, attempt, "forms"),
            FINAL_A,
        )
        b_protos = invent_batch(
            streams.rng(master, streams.SURFACE_B, NAME, ordinal, attempt, "forms"),
            FINAL_B,
        )
        if audit.pair_refusal(tuple(a_protos), tuple(b_protos), SUPPORT):
            continue
        return (
            soundchange._side_table(master, ordinal, "A", attempt, a_protos, NAME),
            soundchange._side_table(master, ordinal, "B", attempt, b_protos, NAME),
        )
    raise ConstructionExhausted(
        f"no {NAME} pair at ordinal {ordinal} cleared the construction filter in "
        f"{MAX_ATTEMPTS} attempts"
    )


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------

#: The first version's text with the environment list grown by one bullet, and one
#: sentence saying what a word boundary does to the one condition that asks for the
#: absence of a vowel rather than its presence. Written out rather than derived, because
#: a registered text a reader is served should be readable in one place.
TASK_TEMPLATE = """{batch_title}

Produce one daughter form for each proto form in the batch below.

A form is a sequence of single-character phones. Its vowels are a, e, i,
o and u. The other permitted phones are b, d, g, k, m, n, p, s, t and x;
all of these are consonants. There are no silent letters, digraphs,
syllable boundaries or stress marks.

Apply three passes to each whole form.

First apply the public nasal pass: change every n immediately before b
to m. This pass runs once, before either of the other two passes.

One remaining pass replaces k. Its replacement phone is one of g, x
or s. Its environment is one of these three possibilities:
* Both immediate neighbors must be vowels.
* The immediate neighbor on the right must be a vowel; the left
  neighbor is unrestricted.
* The immediate neighbor on the right must be a vowel, and the left
  neighbor must not be a vowel: it is a consonant, or the k is the
  first phone of the form.

The other remaining pass deletes one selected vowel, chosen from e, i
and a. Delete that vowel only where its immediate neighbors on both
sides are consonants.

The two remaining passes run in one of these two orders:
* Replacement, then deletion.
* Deletion, then replacement.

The replacement phone, replacement environment, deleted vowel and order
are drawn independently and uniformly from the choices just listed.
Their selected values are not stated in this task.

For each pass, test every position against the string as it stood at
the start of that pass. Apply all matching changes together to make
the next string. Every matching occurrence changes, not just the first.
Do not test against a partially rewritten string and do not repeat a
pass. Later passes see the completed output of the previous pass.
The nasal pass is never run again, even if deletion creates a new nb.

An immediate neighbor must exist to satisfy a vowel or consonant
condition. A word boundary is neither a vowel nor a consonant.
A missing required neighbor prevents that match. A left neighbor that
must not be a vowel is the one condition a word boundary meets, because
a boundary is not a vowel. A phone that does not match a pass stays
unchanged. If no position matches, the whole string stays unchanged on
that pass. All sources are single phones, so there is no longest-match
choice or tie to resolve.

{scope}
{receipt}
BATCH ({format_name})

{body}

Submit one line per printed row to submit_filing: form_id, a comma,
and the daughter form. Use no header, quotation marks, spaces inside
a form, slash marks or explanatory text. File the rows in printed
order. A daughter answer is a nonempty string of at most 16 permitted
phones. Different rows may be answered independently.

The reading rules are fixed. Blank lines are ignored. Split each
remaining line at its first comma. For identifiers and answers, trim
outer whitespace, collapse each run of internal whitespace to one
ASCII space, and replace every remaining character outside printable
ASCII with ?. Match identifiers and answers without ASCII letter case.
Internal spaces, punctuation, ? and non-phone characters do not become
phones and cannot match a correct answer. Accents are not transliterated.

The first line matching an identifier supplies its value; later lines
for that identifier are duplicates and do not replace it. Lines naming
unknown identifiers are ignored. A missing row or an explicitly empty
value is incorrect. If the entire filing contains no comma, it is read
positionally only when it has exactly one nonempty line per printed
row. Otherwise that comma-free filing supplies no answers. No other
fallback reading is used.

Every printed row has equal weight. The grade is the fraction with an
exact correct daughter form after the stated normalization, rounded to
six decimal places. If feedback is provided, a failed row's correction
is that row's correct daughter form.
"""


# --------------------------------------------------------------------------
# the oracle template
# --------------------------------------------------------------------------

#: The first version's environment phrases and one more. No phrase contains another, so
#: the parser reads exactly one option back out of a rendered statement.
ENVIRONMENT_WORDS = {
    **soundchange.ENVIRONMENT_WORDS,
    NO_LEFT_VOWEL: (
        "only when the immediate neighbor on the right is a vowel and the neighbor "
        "on the left is a consonant or absent"
    ),
}

ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences=dict(soundchange.ORACLE_TEMPLATE.sentences),
    phrases={
        "reflex": REFLEX_WORDS,
        "environment": ENVIRONMENT_WORDS,
        "loss": LOSS_WORDS,
        "sequence": SEQUENCE_WORDS,
    },
)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class SoundChangeV2Generator(soundchange.SoundChangeGenerator):
    """The second version, as the generator protocol wants it.

    Everything not overridden here is the first version's: the surfaces, the table
    record, the reading, the scorer, the three cells, the envelope widths, the copy
    profile and the receipt policy. The envelope's filler is drawn under this version's
    name, because the first version's generator names its streams by `name`.
    """

    name = NAME
    AXES = AXES
    TASK_TEMPLATE = TASK_TEMPLATE
    ORACLE = ORACLE_TEMPLATE

    def build_table(self, master: bytes, ordinal: int, label: str) -> SoundTable:
        """This side of the pair the construction filter accepted for this ordinal."""
        a_table, b_table = build_pair(bytes(master), int(ordinal))
        return a_table if label.upper() == "A" else b_table

    def key_for(
        self, table: SoundTable, convention: Mapping[str, str]
    ) -> tuple[str, ...]:
        return key_for(table, convention)


GENERATOR = SoundChangeV2Generator()


__all__ = [
    "ALL_CONVENTIONS",
    "AXES",
    "ENVIRONMENT_WORDS",
    "GENERATOR",
    "NAME",
    "NO_LEFT_VOWEL",
    "ORACLE_TEMPLATE",
    "SUPPORT",
    "TASK_TEMPLATE",
    "SoundChangeV2Generator",
    "build_pair",
    "daughter",
    "key_for",
    "replace_pass",
]
