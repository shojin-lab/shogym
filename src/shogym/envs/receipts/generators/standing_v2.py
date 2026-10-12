"""The standing genre, second version: the same rules, and a filing line true on every surface.

Everything the first version decides is here unchanged: the four hidden decisions and
their 64 rules, the eight surfaces, the log search, the rule code, the rules extract, the
oracle, the envelope, the copy profile and the full receipt, all read from its module.
What is new is one line of the task text.

THE ID. The flats surface prints each record as a bracketed block, whose header is the
entry's id and whose keys never name the id column. The first version tells a reader to
file "the id in column breach_ref", a column the log never prints. Here the filing line
points at the brackets that open a block on that surface, and at the id column on the
other seven, which print it under its name. On those seven the words are the first
version's.

WHY A SECOND VERSION AND NOT AN EDIT TO THE FIRST. Finished studies registered the first
version's code pin, and an edit to its module, or to any module it imports, moves that
pin. So the first version stays as the studies ran it, and this one is registered beside
it under a name of its own. Its logs, member numbers, filler, masks and task identifiers
are all drawn under `standing_v2`, because the first version's generator names its
streams by `name`, so a bank of each under one key share only the convention stream every
genre shares.
"""

from __future__ import annotations

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.generators import standing
from shogym.envs.receipts.generators.banded_rows import FORMAT_NOTE, prose
from shogym.envs.receipts.generators.standing import (
    READ_PARAGRAPH,
    REST_PARAGRAPH,
    RECEIPT_SENTENCE,
    StandingTable,
    level_table,
    points_table,
)
from shogym.envs.receipts.protocol import PublicTask

NAME = "standing_v2"


def id_place(fmt: str, column: str) -> str:
    """Where an entry's id is printed, as the filing line names it.

    Seven formats print the id under its column's name. A bracketed block prints it as
    the block's header and never names the column, so the column is not where to look.
    Each second version of a banded family words this for itself, so that a change to
    one family's text moves no other family's pin.
    """
    if fmt == "ini":
        return "in the brackets that open its block"
    return "in column %s" % column


# --------------------------------------------------------------------------
# the task text
# --------------------------------------------------------------------------

#: The first version's template with the filing line filled to width like the extract's
#: paragraphs, because the id's place is a phrase whose length differs between surfaces.
TASK_TEMPLATE = """{org} {title}

You are the membership clerk. For every entry in the log below, give the standing the
{member} concerned holds immediately after that entry, using the rules extract that
follows.

RULES EXTRACT ({manual})

  Each infraction carries points by its type:

{points}

{read}

{levels}

{rest}

{scope}
{receipt}
LOG ({fmtnote}; entries in date order)

{body}

{filing}
"""

#: The filing line, filled to width once the id's place is in.
FILING_PARAGRAPH = (
    "File one line per entry: the id {id_place}, a comma, and the standing, with no header "
    "and no other text. File every entry, in the order the entries appear above."
)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class StandingV2Generator(standing.StandingGenerator):
    """The second version, as the generator protocol wants it.

    Everything not overridden here is the first version's: the surfaces, the log
    search, the table record, the key, the reading, the scorer, the three cells, the
    oracle, the envelope widths, the copy profile and the receipt policy. The logs and
    the envelope's filler are drawn under this version's name, because the first
    version's generator names its streams by `name`.
    """

    name = NAME

    def describe(self, task: PublicTask) -> str:
        table: StandingTable = task.table
        dom = table.dom
        cols = dom["cols"]
        words = dict(member=dom["member"], c_staff=cols[4], c_place=cols[5])
        filing = FILING_PARAGRAPH.format(id_place=id_place(dom["fmt"], cols[0]))
        return TASK_TEMPLATE.format(
            org=dom["org"], title=dom["title"], member=dom["member"], manual=dom["manual"],
            points=points_table(dom), levels=level_table(dom),
            read=prose(READ_PARAGRAPH.format(**words)),
            rest=prose(REST_PARAGRAPH.format(**words)),
            scope=scope_sentence(task.label),
            receipt=shared_filing.receipt_sentence(self.RECEIPT_POLICY, RECEIPT_SENTENCE),
            fmtnote=FORMAT_NOTE[dom["fmt"]], body=table.body,
            filing=prose(filing, indent=""),
        )


GENERATOR = StandingV2Generator()


__all__ = [
    "FILING_PARAGRAPH",
    "GENERATOR",
    "NAME",
    "TASK_TEMPLATE",
    "StandingV2Generator",
    "id_place",
]
