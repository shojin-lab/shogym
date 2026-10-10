"""The scorecard genre, second version: the same rules, and an extract none of them contradicts.

Everything the first version decides is here unchanged: the four hidden decisions and
their 96 rules, the eight surfaces, the table search, the rule code, the oracle, the
envelope, the copy profile and the full receipt, all read from its module. What is new is
the task text, on two points: a sentence the rule a task is scored under can contradict,
and a column the list on one surface never prints.

THE SCORE. The first version's extract says an order's priority score is the sum of the
points it earns on four factors, the fourth being whether it repeats an earlier fault, and
that the band is read off the score with the printed table. Under `oneband_up` a repeat
earns no points and is placed one band above the band its score gives, so on a list drawn
under that option the key follows no sum the extract describes: a repeat's band is lifted
after the table is read, not by points before it. Here the score is the sum over three
factors, and the extract says a repeat can move the order to a higher band. That is true
under both options: two extra points before the band is read can lift the band, and one
band up always does unless the band is the last.

THE ID. The marina surface prints each record as a bracketed block, whose header is the
order's id and whose keys never name the id column. The first version tells a reader to
file "the id in column defect_ref", a column the list never prints. Here the filing line
points at the brackets that open a block on that surface, and at the id column on the
other seven, which print it under its name.

WHAT IS KEPT. The sentence that exposure is scored either from the people affected or
from the days open is the first version's. Under `larger_of` each order's exposure is
still read from one of the two, whichever earns more, and the extract already leaves how
exposure is measured to the house scorecard. The wordings that would say more either name
that option or say "both", which reads against the oracle's "never both".

WHY A SECOND VERSION AND NOT AN EDIT TO THE FIRST. Finished studies registered the first
version's code pin, and an edit to its module, or to any module it imports, moves that
pin. So the first version stays as the studies ran it, and this one is registered beside
it under a name of its own. Its tables, filler, masks and task identifiers are all drawn
under `scorecard_v2`, because the first version's generator names its streams by `name`,
so a bank of each under one key share only the convention stream every genre shares.
"""

from __future__ import annotations

from shogym.envs.receipts import filing as shared_filing
from shogym.envs.receipts.filing import scope_sentence
from shogym.envs.receipts.generators import scorecard
from shogym.envs.receipts.generators.banded_rows import FORMAT_NOTE, prose
from shogym.envs.receipts.generators.scorecard import (
    REST_PARAGRAPH,
    RECEIPT_SENTENCE,
    ScoreTable,
    band_table,
)
from shogym.envs.receipts.protocol import PublicTask

NAME = "scorecard_v2"


def id_place(fmt: str, column: str) -> str:
    """Where a record's id is printed, as the filing line names it.

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

You are the duty planner. Give every {entity} on the list below one priority band,
using the scorecard extract that follows.

SCORECARD EXTRACT ({manual})

{rule}

{table}

{rest}

{scope}
{receipt}
LIST ({fmtnote})

{body}

{filing}
"""

#: The extract's first paragraph. The score is a sum over three factors, and a repeat
#: fault is named as what can move an order above the band its score gives.
RULE_PARAGRAPH = (
    "Every open {entity} is given a priority score, which is the sum of the points it "
    "earns on the house scorecard's three factors: its hazard grade (column {c_haz}), "
    "the criticality of the asset it concerns (column {c_crit}) and its exposure. "
    "Exposure is scored either from the number of people affected (column {c_ppl}) or "
    "from the days the {entity} has been open (column {c_days}). The band is read off the "
    "score with this table, and a repeat of an earlier fault (column {c_rep}) can move "
    "the {entity} to a higher band:"
)

#: The filing line, filled to width once the id's place is in.
FILING_PARAGRAPH = (
    "File one line per {entity}: the id {id_place}, a comma, and the band, with no header "
    "and no other text. File every {entity}, in the order they appear above."
)


# --------------------------------------------------------------------------
# the generator
# --------------------------------------------------------------------------


class ScorecardV2Generator(scorecard.ScorecardGenerator):
    """The second version, as the generator protocol wants it.

    Everything not overridden here is the first version's: the surfaces, the table
    search, the table record, the key, the reading, the scorer, the three cells, the
    oracle, the envelope widths, the copy profile and the receipt policy. The tables and
    the envelope's filler are drawn under this version's name, because the first
    version's generator names its streams by `name`.
    """

    name = NAME

    def describe(self, task: PublicTask) -> str:
        table: ScoreTable = task.table
        dom = table.dom
        cols = dom["cols"]
        words = dict(
            entity=dom["entity"], c_haz=cols[1], c_crit=cols[2], c_ppl=cols[3],
            c_days=cols[4], c_rep=cols[5], c_trade=cols[6], c_place=cols[7],
        )
        filing = FILING_PARAGRAPH.format(
            entity=dom["entity"], id_place=id_place(dom["fmt"], cols[0])
        )
        return TASK_TEMPLATE.format(
            org=dom["org"], title=dom["title"], entity=dom["entity"], manual=dom["manual"],
            rule=prose(RULE_PARAGRAPH.format(**words)),
            rest=prose(REST_PARAGRAPH.format(**words)),
            table=band_table(dom), scope=scope_sentence(task.label),
            receipt=shared_filing.receipt_sentence(self.RECEIPT_POLICY, RECEIPT_SENTENCE),
            fmtnote=FORMAT_NOTE[dom["fmt"]], body=table.body,
            filing=prose(filing, indent=""),
        )


GENERATOR = ScorecardV2Generator()


__all__ = [
    "FILING_PARAGRAPH",
    "GENERATOR",
    "NAME",
    "RULE_PARAGRAPH",
    "TASK_TEMPLATE",
    "ScorecardV2Generator",
    "id_place",
]
