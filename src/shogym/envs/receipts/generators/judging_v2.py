"""The judging genre, second version: the same rules, and an extract none of them contradicts.

Everything the first version decides is here unchanged: the four hidden decisions and
their 64 rules, the eight surfaces, the heat search, the rule code, the oracle, the
envelope, the copy profile and the full receipt, all read from its module. What is new is
the task text, on three points: two sentences the rule a task is scored under can
contradict, and a column the list on one surface never prints.

DISQUALIFICATION. The first version's extract says "Some entries are disqualified". Under
`bar_none` nobody is, and the oracle cell of such a draw says "no entry is disqualified",
so a reader in the oracle arm holds two statements that cannot both be true. Here some
entries MAY be disqualified.

LATENESS. The first version's extract says a late entry is penalised. Under `late_last` a
late entry that is disqualified is placed among the disqualified by its mark as usual, so
it is placed exactly where it would be placed on time. Here a late entry MAY be penalised.

THE ID. The brass surface prints each record as a bracketed block, whose header is the
entry's number and whose keys never name the id column. The first version names that
column twice: as where the number is filed from, and as the running order the tie rules
fall back on under both options. Here both sentences point at the brackets that open a
block on that surface, and at the id column on the other seven, which print it under its
name.

WHY A SECOND VERSION AND NOT AN EDIT TO THE FIRST. Finished studies registered the first
version's code pin, and an edit to its module, or to any module it imports, moves that
pin. So the first version stays as the studies ran it, and this one is registered beside
it under a name of its own. Its heats, names, filler, masks and task identifiers are all
drawn under `judging_v2`, because the first version's generator names its streams by
`name`, so a bank of each under one key share only the convention stream every genre
shares.
"""

from __future__ import annotations

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.generators import judging
from shogym.envs.receipts.generators.banded_rows import FORMAT_NOTE, prose
from shogym.envs.receipts.generators.judging import RECEIPT_SENTENCE, JudgingTable
from shogym.envs.receipts.protocol import PublicTask

NAME = "judging_v2"


def id_place(fmt: str, column: str) -> str:
    """Where an entry's number is printed, as the extract and the filing line name it.

    Seven formats print it under its column's name. A bracketed block prints it as the
    block's header and never names the column, so the column is not where to look. Each
    second version of a banded family words this for itself, so that a change to one
    family's text moves no other family's pin.
    """
    if fmt == "ini":
        return "in the brackets that open its block"
    return "in column %s" % column


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------

#: The first version's template with the filing line filled to width like the extract,
#: because the number's place is a phrase whose length differs between surfaces.
TASK_TEMPLATE = """{org} {title}

You are the steward. Give every {entry} below its placing within its {heat_word},
using the rules extract that follows.

RULES EXTRACT ({manual})

{rule}

  The placings, in order, are: {placings}.

{scope}
{receipt}
ENTRIES ({fmtnote}; in {heat_word} order, running order within each)

{body}

{filing}
"""

#: The extract's paragraph. Disqualification and the late penalty are what the rules may
#: do rather than what they do, and running order names where the number is printed.
RULE_PARAGRAPH = (
    "Every {entry} is scored out of 20 by three judges; the head judge's score is in "
    "column {c_head} and the other two are in columns {c_two} and {c_three}. Within each "
    "{heat_word} the {entry}s are placed from {first} to {last} by their mark, highest "
    "first. Some {entry}s may be disqualified, and a disqualified {entry} is placed below "
    "every qualified {entry} of its {heat_word}, the disqualified being placed among "
    "themselves by mark in the same way. A late {entry} (column {c_late}) may be "
    "penalised. Equal marks are separated by the house tie rule; running order means the "
    "{entry} number {id_place}, lower first. How the three scores make a mark, which "
    "{entry}s are disqualified, what the late penalty is and how ties are broken are set "
    "by the house rules, which this extract does not reproduce. Column {c_name} plays no "
    "part."
)

#: The filing line, filled to width once the number's place is in.
FILING_PARAGRAPH = (
    "File one line per {entry}: the {entry} number {id_place}, a comma, and its placing, "
    "with no header and no other text. File every {entry}, in the order they appear above."
)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class JudgingV2Generator(judging.JudgingGenerator):
    """The second version, as the generator protocol wants it.

    Everything not overridden here is the first version's: the surfaces, the heat
    search, the table record, the key, the reading, the scorer, the three cells, the
    oracle, the envelope widths, the copy profile and the receipt policy. The heats and
    the envelope's filler are drawn under this version's name, because the first
    version's generator names its streams by `name`.
    """

    name = NAME

    def describe(self, task: PublicTask) -> str:
        table: JudgingTable = task.table
        dom = table.dom
        cols = dom["cols"]
        heat_word = cols[1]
        places = dom["places"]
        place = id_place(dom["fmt"], cols[0])
        rule = RULE_PARAGRAPH.format(
            entry=dom["entry"], heat_word=heat_word, id_place=place, c_name=cols[2],
            c_head=cols[3], c_two=cols[4], c_three=cols[5], c_late=cols[6],
            first=places[0], last=places[-1],
        )
        filing = FILING_PARAGRAPH.format(entry=dom["entry"], id_place=place)
        return TASK_TEMPLATE.format(
            org=dom["org"], title=dom["title"], entry=dom["entry"], manual=dom["manual"],
            heat_word=heat_word, rule=prose(rule),
            placings=", ".join(places),
            scope=scope_sentence(task.label),
            receipt=shared_filing.receipt_sentence(self.RECEIPT_POLICY, RECEIPT_SENTENCE),
            fmtnote=FORMAT_NOTE[dom["fmt"]], body=table.body,
            filing=prose(filing, indent=""),
        )


GENERATOR = JudgingV2Generator()


__all__ = [
    "FILING_PARAGRAPH",
    "GENERATOR",
    "NAME",
    "RULE_PARAGRAPH",
    "TASK_TEMPLATE",
    "JudgingV2Generator",
    "id_place",
]
