"""The sound change genre: three passes over invented words, two of them hidden.

A batch of invented proto forms and the complete public mechanics of a cascade that
turns each one into a daughter form. The mechanics are deliberately incomplete. Four
decisions they never make are the hidden convention:

    reflex        3 options   which phone a matching k is replaced by
    environment   2 options   which neighbours a k needs before it is replaced
    loss          3 options   which vowel is deleted between consonants
    sequence      2 options   whether replacement runs before deletion or after

That is 36 conventions, drawn uniformly and independently. Sibling tasks A and B are
two disjoint batches of invented forms, presented under different surface templates
and scored under the same drawn convention.

THIRTY SIX AND NOT FIFTY FOUR. A third environment, replacing every k with no
condition at all, was proposed and is not here. Unconditional replacement of one
consonant by another commutes with the deletion of a vowel between consonants on every
word: replacement preserves the consonant and vowel classification, so deletion sees
the same contexts either way, and deletion cannot remove a condition an unconditional
rule does not have. The nine pairs of order choices under that environment would
therefore produce identical answers on every possible form, and no search for better
words could separate them. Keeping them would have advertised a decision no data can
reveal.

WHY THE RECEIPT CAN CARRY SOMETHING. The scored column holds a whole daughter form,
which the rule COMPUTES, not the option the agent chose. Two reflexes that leave a row
untouched are indistinguishable on that row and distinguishable on another; an
environment shows itself only where a k has a consonant on its left; an order shows
itself only where the deletion removes a context the replacement needed. So the
verdicts across rows cut the option set into more than two pieces and the agent can
narrow the rule rather than only learn that it was wrong. `build_pair` searches for a
pair of batches where that actually happens, over the whole support rather than under
the convention this instance drew.

WHY A SHORT EXECUTABLE RULE, AND WHAT THAT IS NOT AN ANSWER TO. The rule here is three
passes over at most fourteen phones with at most two k occurrences: no date arithmetic,
no rounding, no lookup. That is a deliberate choice about what a copy has to do once it
has the rule, and it is NOT a repair for the low oracle grade the fourth engineering run
recorded. The diagnostic of that run located the loss in uptake and in scope, not in
execution: 350 of its 357 filings were the exact answer key of some convention the
family draws from, an independent recomputation agreed with the grader on every row, and
no copy skipped a row or ran out of turns. What went wrong was that copies refused the
statement as untrusted content or read its scope word as the name printed on the surface
and did not carry the rule to the task they were about to file. Those are repairs to
registered wording, and they were made. A short rule here buys a cleaner separation
between a copy that did not take the rule and a copy that took it and got the arithmetic
wrong; calling the earlier loss an execution problem would have made this design a fix
for something that was not broken.

WHAT IS EASY HERE AND WHAT IS NOT, said plainly. The replacement phones g, x and s
never occur in a proto form, and the deleted vowel can be read off the difference in
vowel counts, so a reader holding A's corrected key can name two of the four axes
directly. The `phone_lookup` check gives those two away for free and asks what room is
left, which is the two-way environment composed with the two-way order. This family
does not offer four equally demanding inferences and the release should not say it
does.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Mapping, Sequence

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts import streams
from shogym.envs.receipts.copy_profiles import SOUNDCHANGE_V1
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.oracle import OracleTemplate
from shogym.envs.receipts.oracle import parse as parse_oracle_cell
from shogym.envs.receipts.oracle import render as render_oracle_cell
from shogym.envs.receipts.protocol import (
    ROW_ADDITIVE_EQUAL_WEIGHT,
    Axis,
    Column,
    ConstructionExhausted,
    Filing,
    PublicTask,
    RowOutcome,
    Shape,
    Task,
)
from shogym.envs.receipts.receipt_ast import (
    Envelope,
    ReceiptAST,
    SlotSpec,
    envelope_size_for,
)
from shogym.envs.receipts.render import graded_receipt, placebo_receipt
from shogym.receipts import ROW_LABEL

# ----- the phone inventory ----------------------------------------------------
# ASCII only, and smaller than the benchmark this genre is grounded in. Byte offsets
# and character offsets have to be the same number for the slot arithmetic to mean
# anything, and a phone a reader cannot type is a phone the parser folds to a question
# mark.

VOWELS = "aeiou"
CONSONANTS = "bdgkmnpstx"
#: The consonants a generated cell is drawn from. g, x and s are deliberately absent:
#: they are what the replacement pass introduces, so their presence in a daughter form
#: is a consequence of the hidden rule rather than of the surface.
CELL_CONSONANTS = "bdmnpt"
#: The final consonant pools, disjoint between the siblings. The final consonant is
#: untouched by every pass, so disjoint pools give disjoint A and B answer sets without
#: reading A's forms while B is being drawn. They establish NOTHING about copying
#: resistance: the analogy check exists because a superficial difference of this kind
#: is exactly what a reader would see through.
FINAL_A = "bdm"
FINAL_B = "npt"
#: The public nasal pass. Fixed, stated in the task, and carrying no hidden choice: it
#: is there so that a form can hold a cluster whose first phone a later pass can strand,
#: and so a reader has something to get wrong that is not the drawn rule.
NASAL_SOURCE = "n"
NASAL_TRIGGER = "b"
NASAL_RESULT = "m"

#: What the receipt prints for a row filed with an empty value, and for a row filed
#: with nothing at all. Two different acts, two different tokens. Neither is a legal
#: daughter form: a daughter is a nonempty string of phones and these are neither.
BLANK_TOKEN = "(empty)"
UNFILED_TOKEN = "(none)"

AXES: tuple[Axis, ...] = (
    Axis("reflex", ("out_g", "out_x", "out_s"),
         "which phone a matching k becomes"),
    Axis("environment", ("vowel_pair", "right_vowel"),
         "which neighbours a k needs before it is replaced"),
    Axis("loss", ("erase_e", "erase_i", "erase_a"),
         "which vowel is deleted between consonants"),
    Axis("sequence", ("replace_then_erase", "erase_then_replace"),
         "which of the two hidden passes runs first"),
)

#: The option to the phone it introduces, and the option to the vowel it removes. These
#: two tables are the option-to-operation mapping, and the audit module reimplements
#: them rather than importing them.
REFLEX_PHONE = {"out_g": "g", "out_x": "x", "out_s": "s"}
DELETED_VOWEL = {"erase_e": "e", "erase_i": "i", "erase_a": "a"}
REPLACE_FIRST = "replace_then_erase"

SHAPE = Shape(
    columns=(
        Column("form_id", "F- and eight lowercase hexadecimal digits from the side's "
                          "identifier stream, collisions rejected"),
        Column("proto_form", "a 7 to 14 character lowercase word from the batch grammar"),
    ),
    rows=24,
    case="one proto form awaiting its daughter form",
    note="every form carries each of the three deletable vowels between consonants",
)

#: The word grammar, as counts. A form is `n` consonant cells interleaved with `n`
#: vowels and closed by one final consonant, so every vowel has a consonant on both
#: sides and every cell has a vowel on its right.
CELL_COUNTS = (3, 4, 5, 6)
REQUIRED_VOWELS = ("a", "e", "i")
K_COUNTS = (1, 2)
#: One candidate form in four carries the two-phone cluster nb, which the public nasal
#: pass turns into mb. One in four rather than every form, so the batch holds both kinds
#: of row and the pass is exercised without the surface announcing it.
CLUSTER_IN = 4
CLUSTER = "nb"
#: How many candidate words one batch may draw before the draw is a failure rather than
#: a slow success, and how many whole pairs one ordinal may try.
MAX_CANDIDATES = 10000
MAX_ATTEMPTS = 400
ROWS = SHAPE.rows

# ----- the registered envelope constants -------------------------------------
# Every one of these is a family maximum or a fixed width. None is read off a
# particular draw, which is what makes the envelope size convention-independent.

IDENTIFIER_WIDTH = 12
OBSERVED_WIDTH = 16
VERDICT_WIDTH = 4
#: SIXTEEN, AND LEDGER'S TWELVE WILL NOT DO. The longest proto form is 14 phones and
#: every form holds at least one occurrence of whichever vowel the rule deletes, all of
#: them between consonants, so the longest daughter this family can produce is 13 bytes.
#: The legal filed answer is bounded at 16, so every well formed answer is echoed and
#: corrected without truncation, and a longer malformed filing keeps its whole value for
#: scoring while the observed column truncates it identically in both cells.
CORRECTION_WIDTH = 16
#: The longest daughter form the grammar can produce, and the longest value a filing may
#: offer as one. The first is what the correction slot has to hold; the second is what
#: the observed column has to echo.
MAX_DAUGHTER = 13
MAX_FILED_ANSWER = 16
VERDICT_TOKENS = ("PASS", "FAIL")
SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec("verdict", VERDICT_WIDTH, vocabulary=VERDICT_TOKENS),
    SlotSpec(
        "correction", CORRECTION_WIDTH,
        vocabulary=(BLANK_TOKEN,), allows_answers=True, allows_empty=True,
    ),
)
COLUMN_TITLES = ("mark", "note")
#: Digits only. With letters in it a neutral token eventually spells a verdict, and it
#: can never spell a daughter form either, which is a string of phones.
FILLER_ALPHABET = "0123456789"
ORACLE_BODY_ALLOWANCE = 1100
ENVELOPE_SIZE = envelope_size_for(
    max_rows=SHAPE.rows,
    identifier_width=IDENTIFIER_WIDTH,
    observed_width=OBSERVED_WIDTH,
    slots=SLOTS,
    body_allowance=ORACLE_BODY_ALLOWANCE,
)

# ----- the four surfaces ------------------------------------------------------


@dataclass(frozen=True)
class Surface:
    """One presentation template. A surface decides how the batch is printed and
    nothing else: every option of every axis means the same thing on all four."""

    name: str
    title: str
    format_name: str
    separator: str


SURFACES: dict[tuple[str, int], Surface] = {
    ("A", 0): Surface("field_csv", "Field transcription batch", "comma-separated", ","),
    ("A", 1): Surface("archive_tsv", "Archive transcription batch", "tab-separated", "\t"),
    ("B", 0): Surface(
        "comparative_csv", "Comparative transcription batch", "comma-separated", ","
    ),
    ("B", 1): Surface(
        "catalogue_tsv", "Catalogue transcription batch", "tab-separated", "\t"
    ),
}
SURFACE_BY_NAME = {surface.name: surface for surface in SURFACES.values()}
HEADER = ("form_id", "proto_form")


# --------------------------------------------------------------------------
# the table
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SoundRow:
    row_id: str
    proto: str


@dataclass(frozen=True)
class SoundTable:
    surface: str
    rows: tuple[SoundRow, ...]
    body: str

    @property
    def template(self) -> Surface:
        return SURFACE_BY_NAME[self.surface]


# --------------------------------------------------------------------------
# the cascade: the scoring function's arithmetic
# --------------------------------------------------------------------------
#
# EVERY PASS READS ITS OWN INPUT. A pass tests each position against the string as it
# stood when the pass began and emits a new string; it never consults the output it is
# building. That is the difference between `nb` becoming `mb` once and a deletion that
# creates a new `nb` being assimilated again, and it is the mistake this genre is most
# likely to elicit, so it is stated in the task and implemented here in the one way the
# task states.


def nasal_pass(word: str) -> str:
    """The public pass: every n immediately before b becomes m. Runs once, first."""
    out: list[str] = []
    for position, phone in enumerate(word):
        following = word[position + 1] if position + 1 < len(word) else ""
        if phone == NASAL_SOURCE and following == NASAL_TRIGGER:
            out.append(NASAL_RESULT)
        else:
            out.append(phone)
    return "".join(out)


def replace_pass(word: str, phone: str, environment: str) -> str:
    """Every k whose neighbours satisfy the drawn environment becomes `phone`.

    A required neighbour has to exist. A word boundary is neither a vowel nor a
    consonant, so a k at the start of a form fails the two-sided condition and a k at
    the end fails both.
    """
    out: list[str] = []
    for position, source in enumerate(word):
        if source != "k":
            out.append(source)
            continue
        right = word[position + 1] if position + 1 < len(word) else ""
        left = word[position - 1] if position > 0 else ""
        matched = right in VOWELS and right != ""
        if environment == "vowel_pair":
            matched = matched and left != "" and left in VOWELS
        out.append(phone if matched else source)
    return "".join(out)


def delete_pass(word: str, vowel: str) -> str:
    """Every occurrence of `vowel` with a consonant immediately on both sides goes."""
    out: list[str] = []
    for position, phone in enumerate(word):
        if phone == vowel:
            left = word[position - 1] if position > 0 else ""
            right = word[position + 1] if position + 1 < len(word) else ""
            if left in CONSONANTS and left != "" and right in CONSONANTS and right != "":
                continue
        out.append(phone)
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


# --------------------------------------------------------------------------
# inventing a batch
# --------------------------------------------------------------------------


def invent_word(rng: random.Random, finals: str) -> str:
    """One candidate proto form, consuming the forms stream in the registered order.

    The shape is `c v c v ... c v f`: every vowel sits between two consonants, so every
    vowel the rule might delete is deletable, and every cell has a vowel on its right,
    so the two environments differ only where a cell has a consonant on its left or a
    deletion has taken its right-hand vowel away.
    """
    count = rng.choice(CELL_COUNTS)
    while True:
        vowels = tuple(rng.choice(VOWELS) for _ in range(count))
        if set(REQUIRED_VOWELS) <= set(vowels):
            break
    cells = [rng.choice(CELL_CONSONANTS) for _ in range(count)]
    final = rng.choice(finals)
    for index in rng.sample(range(count), rng.choice(K_COUNTS)):
        cells[index] = "k"
    if rng.randrange(CLUSTER_IN) == 0:
        free = [index for index in range(count) if cells[index] != "k"]
        cells[free[rng.randrange(len(free))]] = CLUSTER
    return "".join(cell + vowel for cell, vowel in zip(cells, vowels)) + final


def invent_batch(rng: random.Random, finals: str, rows: int = ROWS) -> list[str]:
    """One batch of distinct proto forms, or a named failure for this whole attempt."""
    seen: list[str] = []
    held: set[str] = set()
    for _ in range(MAX_CANDIDATES):
        word = invent_word(rng, finals)
        if word in held:
            continue
        held.add(word)
        seen.append(word)
        if len(seen) == rows:
            return seen
    raise ConstructionExhausted(
        f"a soundchange batch of {rows} distinct forms did not appear in "
        f"{MAX_CANDIDATES} candidates"
    )


def _identifiers(rng: random.Random, rows: int = ROWS) -> list[str]:
    """One printed identifier per row, distinct within the side."""
    out: list[str] = []
    held: set[str] = set()
    for _ in range(MAX_CANDIDATES):
        made = "F-%08x" % rng.getrandbits(32)
        if made in held:
            continue
        held.add(made)
        out.append(made)
        if len(out) == rows:
            return out
    raise ConstructionExhausted(
        f"a soundchange side did not produce {rows} distinct identifiers"
    )


def render_body(protos: Sequence[str], identifiers: Sequence[str], surface: Surface) -> str:
    """The printed batch: a header and one line per row, joined by LF and no more."""
    separator = surface.separator
    lines = [separator.join(HEADER)]
    lines += [
        separator.join((identifier, proto))
        for identifier, proto in zip(identifiers, protos)
    ]
    return "\n".join(lines)


def _side_table(
    master: bytes, ordinal: int, label: str, attempt: int, protos: Sequence[str]
) -> SoundTable:
    """One side's frozen table: the shuffled forms, their identifiers and the body."""
    stream = streams.SURFACE_A if label == "A" else streams.SURFACE_B
    order = list(protos)
    streams.rng(master, stream, "soundchange", ordinal, attempt, "row-order").shuffle(order)
    identifiers = _identifiers(
        streams.rng(master, stream, "soundchange", ordinal, attempt, "row-identifiers")
    )
    surface = SURFACES[(label, int(ordinal) % 2)]
    return SoundTable(
        surface=surface.name,
        rows=tuple(
            SoundRow(row_id=identifier, proto=proto)
            for identifier, proto in zip(identifiers, order)
        ),
        body=render_body(order, identifiers, surface),
    )


@lru_cache(maxsize=32)
def build_pair(master: bytes, ordinal: int) -> tuple[SoundTable, SoundTable]:
    """Both sibling tables for one ordinal, from the public draw and nothing else.

    PURE, AND BOTH SIDES AT ONCE. The construction filter is a statement about the
    PAIR: it asks what a reader of A's corrected key can do on B, so neither side can
    be accepted on its own. The two `build_table` calls therefore read one answer out
    of here rather than searching twice and agreeing by luck.

    THE FILTER NEVER CONSULTS THE DRAWN CONVENTION. Every predicate quantifies over the
    whole support, so acceptance is a function of the two batches alone. A filter that
    read the live draw would make rejection informative: an agent that knew which pairs
    are accepted could exclude conventions from the fact that this pair was served,
    and `describe` being invariant would not save it.

    A candidate is rejected and the next attempt is drawn; running out of attempts is
    `ConstructionExhausted` and is a whole-bank failure with a name, not a skipped
    ordinal. The cache is small and controller-side: the master key is already in
    memory wherever this is called from, and the alternative is repeating a search that
    is deterministic in its arguments once per check that draws the instance again.
    """
    from shogym.envs.receipts.generators import soundchange_audit as audit

    for attempt in range(MAX_ATTEMPTS):
        a_protos = invent_batch(
            streams.rng(master, streams.SURFACE_A, "soundchange", ordinal, attempt, "forms"),
            FINAL_A,
        )
        b_protos = invent_batch(
            streams.rng(master, streams.SURFACE_B, "soundchange", ordinal, attempt, "forms"),
            FINAL_B,
        )
        if audit.pair_refusal(tuple(a_protos), tuple(b_protos)):
            continue
        return (
            _side_table(master, ordinal, "A", attempt, a_protos),
            _side_table(master, ordinal, "B", attempt, b_protos),
        )
    raise ConstructionExhausted(
        f"no soundchange pair at ordinal {ordinal} cleared the construction filter in "
        f"{MAX_ATTEMPTS} attempts"
    )


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------

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
or s. Its environment is one of these two possibilities:
* Both immediate neighbors must be vowels.
* The immediate neighbor on the right must be a vowel; the left
  neighbor is unrestricted.

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
A missing required neighbor prevents that match. A phone that does
not match a pass stays unchanged. If no position matches, the whole
string stays unchanged on that pass. All sources are single phones,
so there is no longest-match choice or tie to resolve.

{scope}

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
# the oracle template, and reading it back
# --------------------------------------------------------------------------

#: THE HEADING NAMES THE SCOPE BY THE BATCHES THE READER FILES.
#: A heading that scoped the rule to a word for the body of work it was drawn for was
#: read, and what it did was measured: of 119 readers given a correct statement of the
#: rule, 64 restated it correctly, wrote it into their own notes, and then filed the
#: held-out task under a different rule, because they had taken the scope word for the
#: name printed at the top of the task and the two siblings never carry one name between
#: them. A scope word a reader can map onto the surface is therefore a scope word that
#: names the wrong thing. This heading names the scope by what the reader does, which is
#: file this batch and then the next one, and says in the same breath that the batch
#: after those is scored under a cascade of its own, which leaves a cascade worked out
#: earlier somewhere to live. It is ledger's heading with the domain noun changed and
#: the claim untouched.
ORACLE_HEAD: tuple[str, ...] = (
    "CASCADE FOR THIS BATCH AND THE NEXT ONE YOU FILE (it governs the",
    "batch above and the next batch you are given, whichever title it",
    "carries; the batch after that is scored under a cascade of its own)",
    "Apply the public nasal change first, once; then use the choices below.",
    "",
)

REFLEX_WORDS = {
    "out_g": "the output phone g",
    "out_x": "the output phone x",
    "out_s": "the output phone s",
}
ENVIRONMENT_WORDS = {
    "vowel_pair": "only when the immediate neighbors on both sides are vowels",
    "right_vowel": (
        "whenever the immediate neighbor on the right is a vowel, with no left condition"
    ),
}
LOSS_WORDS = {
    "erase_e": "the vowel e",
    "erase_i": "the vowel i",
    "erase_a": "the vowel a",
}
SEQUENCE_WORDS = {
    "replace_then_erase": "replacement followed by vowel deletion",
    "erase_then_replace": "vowel deletion followed by replacement",
}

#: What each option means, in words. This is a declaration, not a renderer: the package
#: renders the oracle from it and reads one back with the same table, so the sentence an
#: oracle child reads and the rule an admission check believes it states cannot come
#: apart. There are no axis names in it, no intermediate forms and no worked row.
ORACLE_TEMPLATE = OracleTemplate(
    head=ORACLE_HEAD,
    sentences={
        "reflex": "During the k replacement pass, use {}.",
        "environment": "During that pass, replace k {}.",
        "loss": "During the deletion pass, remove {} between consonants.",
        "sequence": "The two passes after nasal assimilation are {}.",
    },
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


class SoundChangeGenerator:
    """The sound change genre, as the generator protocol wants it."""

    name = "soundchange"
    genre = "phonological transformation composition"
    SHAPE = SHAPE
    AXES = AXES
    SCORING: str = ROW_ADDITIVE_EQUAL_WEIGHT
    #: A daughter form is an invented word, not a token of a printed vocabulary, so the
    #: copy screen prices this family through the character maps rather than through
    #: maps between two published answer orders. See `copy_profiles`.
    COPY_PROFILE: str = SOUNDCHANGE_V1
    BLANK_TOKEN = BLANK_TOKEN
    UNFILED_TOKEN = UNFILED_TOKEN

    # ----- the instance -----

    def surface_for(self, ordinal: int, label: str) -> str:
        return SURFACES[(label.upper(), int(ordinal) % 2)].name

    def surface_templates(self) -> tuple[str, ...]:
        return tuple(surface.name for surface in SURFACES.values())

    def build_table(self, master: bytes, ordinal: int, label: str) -> SoundTable:
        """This side of the pair the construction filter accepted for this ordinal."""
        a_table, b_table = build_pair(bytes(master), int(ordinal))
        return a_table if label.upper() == "A" else b_table

    def table_record(self, table: SoundTable) -> dict[str, Any]:
        """Every field of a sound change table, as one canonical value.

        The surface names the presentation template, the rows carry each printed
        identifier and its proto form, and the body is the batch the agent reads. All
        three are read somewhere between describing the task and rendering the receipt,
        so all three are what the bank commits to.
        """
        return {
            "surface": table.surface,
            "rows": [
                {"row_id": row.row_id, "proto": row.proto} for row in table.rows
            ],
            "body": table.body,
        }

    def build_envelope(self, master: bytes, ordinal: int) -> Envelope:
        """The registered envelope, with this instance's committed filler and neutrals.

        The coordinates name this generator, so a bank of one genre and a bank of
        another under one master key do not share a filler stream.
        """
        neutral: dict[str, tuple[str, ...]] = {}
        for spec in SLOTS:
            neutral[spec.name] = tuple(
                streams.filler_stream(
                    master, FILLER_ALPHABET, spec.width,
                    "soundchange", ordinal, "neutral", spec.name, row,
                )
                for row in range(SHAPE.rows)
            )
        return Envelope(
            size=ENVELOPE_SIZE,
            identifier_width=IDENTIFIER_WIDTH,
            observed_width=OBSERVED_WIDTH,
            slots=SLOTS,
            filler=streams.filler_stream(
                master, FILLER_ALPHABET, ENVELOPE_SIZE, "soundchange", ordinal, "pad"
            ),
            column_titles=COLUMN_TITLES,
            neutral=neutral,
        )

    def key_for(
        self, table: SoundTable, convention: Mapping[str, str]
    ) -> tuple[str, ...]:
        return key_for(table, convention)

    def normalize_answer(self, value: str) -> str:
        # The same function the scorer compares with, so the two cannot drift.
        return shared_filing.fold(value)

    def row_identifiers(self, table: SoundTable) -> tuple[str, ...]:
        return tuple(row.row_id for row in table.rows)

    def row_classes(self, table: SoundTable) -> tuple[str, ...]:
        # One declared class: a reader looking at the batch sees 24 forms and no
        # printed distinction between them. The dependence classes the gate derives
        # from the receipt are a different and finer thing, and this does not replace
        # them.
        return ("form",) * len(table.rows)

    def answer_ranks(self, table: SoundTable) -> None:
        """No published ordered vocabulary, and NOT an empty one.

        A daughter form is an invented word. The forms this instance's draw happens to
        realize are not something a reader could read off the two tasks, and the legal
        words up to sixteen phones are not a list anyone enumerates, so there is no
        complete ordered vocabulary to publish and none is invented here. The declared
        copy profile says which maps price this family instead.
        """
        return None

    def row_label(self, table: SoundTable) -> tuple[str, ...]:
        # Every row names the form it grades. Nothing on this receipt names an axis.
        return (ROW_LABEL,) * len(table.rows)

    # ----- reading the filing -----

    def parse_and_canonicalize(self, task: Task, raw: object) -> Filing:
        """One canonical value per printed row, by the shared registered rules.

        The reading is the one in `filing.py`: a line is a form identifier, a comma and
        a daughter form; identifiers match without case after whitespace is collapsed;
        the first line for an identifier wins; a comma-free filing is read positionally
        only when it has exactly one line per printed row. A known identifier with a
        value that is not a legal form stays filed and wrong, because failing to read it
        would turn a bad answer into no answer.
        """
        return shared_filing.parse_rows(self.row_identifiers(task.table), raw)

    def score(self, task: Task, canonical: Filing) -> tuple[float, tuple[RowOutcome, ...]]:
        """The sealed scalar and what the filing did on every row.

        Equal weight per row, the denominator is the printed row count, rounded to six
        places. A row counts only if it was FILED: no correct daughter is ever the
        empty string, but the distinction between a filed empty value and an omission
        is what the receipt prints, so the scorer keeps it too.
        """
        identifiers = self.row_identifiers(task.table)
        truth = task.key if task.key else ("",) * len(identifiers)
        return shared_filing.score_rows(
            identifiers, truth, canonical, self.normalize_answer
        )

    # ----- the task text -----

    def describe(self, task: PublicTask) -> str:
        """The mechanics, which batches share a cascade, and the batch itself.

        It takes the PUBLIC task, so there is no argument here the drawn rule could
        arrive through, and the scope sentence is chosen by the sibling label and by
        nothing else: the same bytes go to every arm of a fork.
        """
        table: SoundTable = task.table
        surface = table.template
        return TASK_TEMPLATE.format(
            batch_title=surface.title,
            scope=scope_sentence(task.label),
            format_name=surface.format_name,
            body=table.body,
        )

    # ----- the three cells -----

    def render_receipt(
        self, task: Task, canonical: Filing, truth: Sequence[str]
    ) -> ReceiptAST:
        """One verdict per form, on what the filing did.

        The rows are built by the shared grader from the scorer's own outcomes, so what
        a row says is not a choice this module gets to make. There are no axis names,
        no intermediate forms, no alignment marks and no grouped counts: a failed row
        carries its own complete daughter form and a passed row carries nothing.
        """
        graded = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text=task.text, key=tuple(truth),
        )
        _, outcomes = self.score(graded, canonical)
        return graded_receipt(task.task_id, outcomes, BLANK_TOKEN, UNFILED_TOKEN)

    def render_placebo(
        self, task: PublicTask, canonical: Filing, envelope: Envelope
    ) -> ReceiptAST:
        """The inert cell: congruent with the graded one outside the registered slots.

        It takes the PUBLIC task and the envelope, so there is no argument here through
        which the hidden rule could reach it. The keyless task standing in for the
        scorer's shape supplies the row identities and the filed values and nothing
        else, and the slot values are this instance's committed neutral tokens.
        """
        blind = Task(
            label=task.label, task_id=task.task_id, surface=task.surface,
            table=task.table, text="", key=(),
        )
        _, outcomes = self.score(blind, canonical)
        return placebo_receipt(task.task_id, outcomes, envelope, BLANK_TOKEN, UNFILED_TOKEN)

    #: The declared phrase table. Rendering and reading both go through it.
    ORACLE = ORACLE_TEMPLATE

    def render_oracle(
        self, task_id: str, convention: Mapping[str, str], row_count: int = 0
    ) -> ReceiptAST:
        """The drawn cascade, from the declared phrases, with no rows to align."""
        return render_oracle_cell(ORACLE_TEMPLATE, task_id, convention, row_count)

    def parse_oracle(self, ast: ReceiptAST) -> dict[str, str]:
        return parse_oracle_cell(ORACLE_TEMPLATE, ast)


GENERATOR = SoundChangeGenerator()


__all__ = [
    "ALL_CONVENTIONS",
    "AXES",
    "BLANK_TOKEN",
    "CELL_CONSONANTS",
    "CONSONANTS",
    "DELETED_VOWEL",
    "ENVELOPE_SIZE",
    "FINAL_A",
    "FINAL_B",
    "GENERATOR",
    "MAX_DAUGHTER",
    "MAX_FILED_ANSWER",
    "ORACLE_TEMPLATE",
    "REFLEX_PHONE",
    "REPLACE_FIRST",
    "ROWS",
    "SHAPE",
    "SLOTS",
    "SURFACES",
    "UNFILED_TOKEN",
    "VOWELS",
    "SoundChangeGenerator",
    "SoundRow",
    "SoundTable",
    "Surface",
    "build_pair",
    "daughter",
    "delete_pass",
    "invent_batch",
    "invent_word",
    "key_for",
    "nasal_pass",
    "render_body",
    "replace_pass",
]
