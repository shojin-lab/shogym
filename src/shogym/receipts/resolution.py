"""Gates R, S and H, computed per instance from what the receipt actually prints.

WHO THESE GATES ARE FOR

A generator is first-party code, and a family becomes admissible by passing these
gates and the named checks and then by a person reading a frozen pack of its
rendered instances. Once a family is admitted its scorer defines the ground truth
for that family: what its `score` returns is what the chain seals, and these gates
measure the receipt that scorer produces rather than adjudicating it.

So what the checks here are for is mistakes. A readout that turns out to be
injective, a receipt that prints a correction nobody meant to license, a placebo
whose committed tokens drifted, a rule the oracle states differently from the one
that was drawn: these are the errors an author makes and cannot see, and the
package finds them by rendering every convention and reading the bytes. They are not
a defence against a generator written to deceive. A family that wanted to defeat
them could, and the answer to that is the human read, not a longer list of
mechanical checks.

Two habits follow from that and are worth keeping even so. Everything a check
compares against is built by this package before any family code runs, and family
code is handed read-only views, because a comparison built after a callback is a
comparison against whatever the callback left behind and that is as likely to be an
accident as anything else. And a family that raises is a failed check rather than a
crashed report.

A receipt is graded against a hidden convention: named axes, each holding an
interchangeable option set, one option drawn per axis. These gates ask what that
receipt can possibly tell the agent about the drawn convention, and they ask it of
the RENDERED artifact rather than of the scoring function behind it.

WHY THE RENDERED ARTIFACT. What an agent can learn is a property of the bytes it
reads. A gate that read the scorer's per-row verdict bits would be reading the
renderer's intentions: a receipt that prints a correction on a failed row says far
more than a bit, one that groups or reorders rows says something the scorer never
computed, and one that prints an identifier correlated with the answer says it
without printing anything the scorer would recognise. So the observation here is
the tuple of slot values the receipt prints for each row, read off the structure
the renderer produced, evaluated under every convention in the space.

THE THREE GATES

  R, resolution. For axis k, hold every other axis at the agent's own applied
      convention and vary k over its options. Each option produces a rendered
      receipt; options whose receipts print the same thing are options the agent
      can never tell apart. The number of distinct printed receipts is the axis's
      SIGNATURE BLOCK COUNT. An instance FAILS R when every axis with three or
      more options sits at two blocks or fewer. Binary axes are outside R's
      quantifier: two blocks is full resolution for them, so asking for three
      would be asking for something that cannot exist. The siblinghood exercise
      check covers them at their own arity.

      Two corollaries follow and both are checked. A readout that is injective on
      an axis puts the receipt at the drawn option against everything else, for
      any number of rows. And a single reported row on a verdict-only receipt can
      produce at most two signatures, so an axis carried by one row fails R
      however rich its readout is.

  S, non-self-interpretation. The map from a printed symbol to its constraint must
      not be printed by the receipt. Five exact checks, on the declared labels AND
      on the serialized bytes, because a renderer can print an interpretation the
      declaration hides: no row is labelled by axis; the resolution the
      label-evident rows alone reach does not already equal the whole receipt's; no
      axis name or option token appears anywhere in the bytes; every slot prints
      only what its registered grammar allows, so a short numeric code stating the
      whole rule is refused even though it spells nothing; and the printed row order
      does not move with the drawn convention. What is NOT checked is prose in the
      task text, which stays a human read.

  H, room above lookup. The ceiling must stand above the LOOKUP FLOOR, and both
      are optimized over the LEGAL ACTION SPACE OF THE SIBLING TASK.

THE ACTION SPACE IS ROWWISE, NOT A CONVENTION KEY

A reader is two things: a partition, which says what it observes, and an action
rule, which says what it does with the uniform posterior on the observed class.
The legal action on the sibling task is one answer per row. Scoring is row
additive, so the best action is the per-row posterior mode, and the reader is never
required to commit to a whole convention and answer as that convention would.

    score(class) = mean over rows r of  max over answers a of
                   Pr( key_B[r] = a  |  the convention is in this class )

Committing to a single convention key is a strictly worse rule whenever two
conventions in the class agree on some rows and differ on others, which is the
ordinary case. Pricing a design with it understates both the ceiling and the floor,
and understates the floor by more, so it reports headroom that is not there.

THE FLOOR, WITH NOTHING DECLARED

    LOOKUP FLOOR: the score of the best strategy confined to what the receipt's own
    labels state, evaluating no readout on any hypothetical convention.

A ceiling alone is not a gate. A named-slot receipt's ceiling looks like ample room
above a placebo, and every point of it is reachable by a rule that costs no
induction: keep your option where it passed, draw from the rest where it failed.
Priced against the placebo the design looks alive; priced against what it concedes
for free its headroom is exactly zero, because floor and ceiling are then the same
partition and no action rule can separate them.

Which rows are EVIDENT is derived, never declared. A row is evident when it
responds to exactly one axis and prints a distinct thing for every option of it:
such a row is an index, and reading the option off it costs no induction. Taking an
author's word for which rows are evident would let a design set its own floor.

That derivation is deliberately GENEROUS, and the direction is chosen. It counts a
row as evident whenever its observation happens to turn on one axis, including rows
whose single-axis dependence a reader could only discover by doing the induction the
floor is supposed to exclude. Being generous raises the floor and lowers headroom,
so the gate under-reports what a design has rather than over-reports it: a design
this floor passes has the room, and a design it rejects may still have some. A
strict floor would err the other way, and a floor that erred the other way is what
licensed a payload whose whole effect was a lookup.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

#: The two things a printed row can be labelled by. An axis label states the
#: constraint; a row label does not.
AXIS_LABEL = "axis"
ROW_LABEL = "row"


@dataclass(frozen=True)
class AxisSpace:
    """The convention space: named axes, each with an interchangeable option set.

    Combinations are enumerated in a fixed order so an integer index can stand for
    a convention everywhere below. Enumeration is deliberate. A skeptic can print
    the space.
    """

    axes: tuple[str, ...]
    options: tuple[tuple[str, ...], ...]

    def __post_init__(self) -> None:
        if len(self.axes) != len(self.options):
            raise ValueError("every axis needs an option set")
        if len(set(self.axes)) != len(self.axes):
            raise ValueError("axis names have to be distinct")

    @property
    def arity(self) -> dict[str, int]:
        return {a: len(o) for a, o in zip(self.axes, self.options)}

    @property
    def combos(self) -> list[tuple[str, ...]]:
        out: list[tuple[str, ...]] = [()]
        for opts in self.options:
            out = [c + (o,) for c in out for o in opts]
        return out

    @property
    def n_combos(self) -> int:
        n = 1
        for o in self.options:
            n *= len(o)
        return n

    def index(self) -> dict[tuple[str, ...], int]:
        return {c: j for j, c in enumerate(self.combos)}

    def substitute(self, combo: tuple[str, ...], axis: str, option: str) -> tuple[str, ...]:
        out = list(combo)
        out[self.axes.index(axis)] = option
        return tuple(out)


@dataclass(frozen=True)
class Observation:
    """One instance, as the gates see it: what the receipt prints, under every convention.

    `shown` is (n_combos, n_rows) of integer codes, one per printed row, where the
    code stands for the whole tuple of slot values that row shows. Two rows with the
    same code print the same thing, which is the only thing that matters to a reader.

    `answers` is (n_combos, n_rows_b) of integer codes for the sibling task's correct
    answers, which is the space the reader's action is scored in.

    `chat` is the convention the agent applied, the one place the question is asked:
    the agent is deciding what to do differently next time given what it did this
    time.

    `labels` is one label per printed row, `orders` the printed identifier sequence
    under each convention, and `payloads` the serialized bytes per convention, all
    three for gate S.
    """

    space: AxisSpace
    chat: int
    shown: np.ndarray
    #: One code per convention for the WHOLE serialized cell. What an agent can tell
    #: apart is whole cells, so this is what R counts and what the ceiling partitions
    #: on; `shown` carries the per-row detail the floor is built from.
    whole: np.ndarray
    answers: np.ndarray
    #: Which conventions the sampler can actually draw. All of them for an ordinary
    #: family; fewer when a generator declares a support smaller than its axis
    #: product, which is the shape the gates exist to reject.
    reachable: np.ndarray | None = None
    identifiers: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    orders: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    payloads: Mapping[int, bytes] = field(default_factory=dict)
    #: Per registered slot, the closed set of values it may print, and the set it
    #: actually realized over the whole convention support. A value outside the
    #: registered set is a symbol whose meaning the receipt is free to define, which
    #: is a self-interpretation whatever it is spelled as.
    slot_grammar: Mapping[str, frozenset[str]] = field(default_factory=dict)
    slot_realized: Mapping[str, frozenset[str]] = field(default_factory=dict)
    #: Every value the scorer can produce as a correct answer on the graded task,
    #: over the whole convention space. A receipt is entitled to print these: they
    #: are what it is grading. Anything else it prints is there to be interpreted.
    answer_vocabulary: frozenset[str] = frozenset()
    tag: str = ""

    def __post_init__(self) -> None:
        if self.shown.shape[0] != self.space.n_combos:
            raise ValueError(
                f"the receipt was rendered under {self.shown.shape[0]} conventions and the "
                f"space holds {self.space.n_combos}"
            )
        if self.answers.shape[0] != self.space.n_combos:
            raise ValueError("the sibling answers have to cover the same convention space")
        if self.whole.shape[0] != self.space.n_combos:
            raise ValueError("the whole-cell codes have to cover the same convention space")

    @property
    def n_rows(self) -> int:
        return int(self.shown.shape[1])

    @property
    def drawable(self) -> np.ndarray:
        """The mask of conventions the sampler can reach."""
        if self.reachable is None:
            return np.ones(self.space.n_combos, dtype=bool)
        return self.reachable

    def truths(self) -> list[int]:
        """The conventions a reader could actually be facing."""
        return [int(j) for j in np.where(self.drawable)[0]]

    def row_labels(self) -> tuple[str, ...]:
        return self.labels or (ROW_LABEL,) * self.n_rows


# --------------------------------------------------------------------------
# gate R: what the printed receipt resolves
# --------------------------------------------------------------------------


def axis_receipts(obs: Observation, axis: str, index: Mapping[tuple[str, ...], int]):
    """(n_options, n_rows) int: what the receipt prints as the axis is varied.

    Everything except `axis` is held at the agent's own applied convention.
    """
    base = obs.space.combos[obs.chat]
    k = obs.space.axes.index(axis)
    drawable = obs.drawable
    positions = [index[obs.space.substitute(base, axis, o)] for o in obs.space.options[k]]
    kept = [j for j in positions if drawable[j]]
    return np.asarray([obs.shown[j] for j in kept]) if kept else np.zeros((0, obs.n_rows))


def axis_cells(obs: Observation, axis: str, index: Mapping[tuple[str, ...], int]):
    """(n_options,) int: the whole cell's code as the axis is varied."""
    base = obs.space.combos[obs.chat]
    k = obs.space.axes.index(axis)
    drawable = obs.drawable
    positions = [index[obs.space.substitute(base, axis, o)] for o in obs.space.options[k]]
    # An option the sampler cannot reach in combination with the rest is not an option
    # the agent is choosing among here, so it is not a block it could resolve.
    return np.asarray([obs.whole[j] for j in positions if drawable[j]])


def resolution_blocks(
    obs: Observation, axis: str, index: Mapping[tuple[str, ...], int] | None = None
) -> int:
    """How many distinct receipts the axis's options produce.

    Options whose receipts are identical are options the agent can never tell apart,
    so the count of distinct printed receipts is what this axis resolves. Its arity
    is full resolution; two is the pin that leaves the agent knowing only that it was
    wrong.
    """
    index = index or obs.space.index()
    reached = axis_cells(obs, axis, index)
    return len(set(reached.tolist())) if reached.size else 1


def row_dependence(
    obs: Observation, index: Mapping[tuple[str, ...], int] | None = None
) -> list[frozenset[str]]:
    """Which axes each printed row actually responds to, derived from the receipt."""
    index = index or obs.space.index()
    out: list[set[str]] = [set() for _ in range(obs.n_rows)]
    for axis in obs.space.axes:
        printed = axis_receipts(obs, axis, index)
        if printed.shape[0] == 0:
            continue
        for row in range(obs.n_rows):
            if len(set(printed[:, row].tolist())) > 1:
                out[row].add(axis)
    return [frozenset(s) for s in out]


def evident_rows(
    obs: Observation, index: Mapping[tuple[str, ...], int] | None = None
) -> np.ndarray:
    """The mask of rows that hand an axis over outright, derived from the receipt.

    A row is EVIDENT when it responds to exactly one axis AND prints a distinct
    thing for every option of it. Such a row is an index: the reader sees a symbol
    and reads the option off it, with no model relating this row to any other, so
    the constraint it carries costs no induction and belongs in the floor.

    Both halves are load-bearing. Without "one axis", a row responding to three
    would be counted as free when a reader could not disentangle it. Without
    "distinct per option", every row of a single-axis design would count, which is
    vacuous: a merging row on a one-axis space responds to that axis and still tells
    the reader only that the answer lies in a set. That is the case the whole design
    turns on, and calling it free would put the floor at the ceiling for every
    design that works.

    Derived, never declared. An author who could nominate the evident rows could
    nominate the floor.
    """
    index = index or obs.space.index()
    dependence = row_dependence(obs, index)
    arity = obs.space.arity
    out = np.zeros(obs.n_rows, dtype=bool)
    printed_by_axis = {a: axis_receipts(obs, a, index) for a in obs.space.axes}
    for row, axes in enumerate(dependence):
        if len(axes) != 1:
            continue
        axis = next(iter(axes))
        printed = printed_by_axis[axis]
        if printed.shape[0] == 0:
            continue
        distinct = len(set(printed[:, row].tolist()))
        # Reachable options, not declared ones: a row cannot be asked to distinguish
        # a combination the sampler never produces.
        out[row] = distinct == printed.shape[0] and printed.shape[0] == arity[axis]
    return out


# --------------------------------------------------------------------------
# the action space: one answer per row of the sibling task
# --------------------------------------------------------------------------


def _classes(rows: np.ndarray) -> np.ndarray:
    """Group identical integer rows; returns the group id of each row."""
    if rows.shape[1] == 0:
        return np.zeros(rows.shape[0], dtype=np.int64)
    _, inv = np.unique(np.ascontiguousarray(rows), axis=0, return_inverse=True)
    return inv.reshape(-1)


def rowwise_scores(
    groups: np.ndarray, answers: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Per observation class, the rowwise Bayes-action score and the sampling score.

    The reader observes its class, holds a uniform posterior on the conventions in
    it, and answers each row of the sibling task independently. Scoring is row
    additive, so its best action per row is that row's posterior mode and its
    expected compliance is the mode's share:

        bayes(class)    = mean over rows of  max over answers of  share in class
        sampling(class) = mean over rows of  sum over answers of  share squared

    The sampling figure is the same reader drawing from its posterior instead of
    taking the mode. It is never better and is strictly worse on any row whose
    posterior is not flat, and it is reported beside the Bayes figure because the
    gap is a property of the receipt worth seeing.
    """
    n_rows = answers.shape[1]
    keep = np.ones(len(groups), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    groups = np.where(keep, groups, -1)
    ids = np.unique(groups[keep]) if keep.any() else np.unique(groups)
    bayes = np.zeros(len(ids))
    sampling = np.zeros(len(ids))
    for position, group in enumerate(ids):
        members = answers[(groups == group) & keep]
        size = float(members.shape[0])
        if n_rows == 0 or size == 0:
            continue
        top = 0.0
        drawn = 0.0
        for row in range(n_rows):
            counts = np.bincount(
                np.unique(members[:, row], return_inverse=True)[1].reshape(-1)
            )
            shares = counts / size
            top += float(shares.max())
            drawn += float(np.square(shares).sum())
        bayes[position] = top / n_rows
        sampling[position] = drawn / n_rows
    lookup = {int(g): i for i, g in enumerate(ids)}
    order = np.array([lookup.get(int(g), 0) for g in groups])
    return bayes[order], sampling[order]


def score_partition(
    groups: np.ndarray,
    answers: np.ndarray,
    truths: Sequence[int],
    mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """The mean Bayes-action and sampling score of a reader whose observation is `groups`."""
    bayes, sampling = rowwise_scores(groups, answers, mask)
    picked = np.asarray(list(truths))
    return float(bayes[picked].mean()), float(sampling[picked].mean())


# --------------------------------------------------------------------------
# gate S: what the receipt says about itself
# --------------------------------------------------------------------------


def _tokens(word: str) -> set[str]:
    return {word, word.replace("_", " "), word.replace("_", "-")}


def vocabulary_leaks(obs: Observation) -> list[str]:
    """Words the serialized receipt prints that can only be there to be interpreted.

    Two kinds, and the distinction matters. An AXIS NAME is never legitimate: a
    receipt that prints one states which decision a row turns on, and there is
    nothing left to infer. An OPTION TOKEN is legitimate exactly when it is also an
    answer the scorer can produce, because then the receipt is printing a graded
    value and not an interpretation of one. A ledger whose empty-band option is
    called `pending` and whose correct answer on an undated record is the band
    `PENDING` is printing the answer; a receipt carrying a legend that reads
    `lowest` is printing the map.

    Judging every option token a leak would fail every receipt that prints a
    correction, which is the channel the whole design runs on.
    """
    allowed = {value.strip().lower() for value in obs.answer_vocabulary}
    words: list[tuple[str, str, str]] = []
    for axis, options in zip(obs.space.axes, obs.space.options):
        words.append((axis, axis, "names the axis"))
        for option in options:
            if option.strip().lower() not in allowed:
                words.append((axis, option, "prints an option that is not an answer"))
    hits: list[str] = []
    for payload in obs.payloads.values():
        text = payload.decode("ascii", "ignore").lower()
        for axis, word, why in words:
            for token in _tokens(word.lower()):
                if re.search(r"(?<![a-z0-9])%s(?![a-z0-9])" % re.escape(token), text):
                    hit = f"{word!r} ({why}, axis {axis})"
                    if hit not in hits:
                        hits.append(hit)
                    break
    return hits


def grammar_violations(obs: Observation) -> list[str]:
    """Slot values the receipt printed that its registered grammar does not allow.

    Searching the bytes for axis names catches a receipt that says `anchor`. It does
    not catch one that says `2100`, and a four-digit code is a complete statement of
    the rule to a child that has seen two of them. The defence is not a longer list
    of forbidden words: it is a closed list of permitted ones.
    """
    out: list[str] = []
    for name, realized in sorted(obs.slot_realized.items()):
        allowed = obs.slot_grammar.get(name)
        if allowed is None:
            out.append(f"{name!r} prints values under no registered grammar")
            continue
        stray = sorted(v for v in realized if v not in allowed)
        if stray:
            shown = ", ".join(repr(v) for v in stray[:4])
            more = "" if len(stray) <= 4 else f" and {len(stray) - 4} more"
            out.append(f"{name!r} prints {shown}{more}, which its grammar does not allow")
    return out


def order_moves_with_the_convention(obs: Observation) -> bool:
    """Whether the receipt reorders its rows when the drawn convention changes.

    Ordering is an observation. A renderer that sorted rows by their answer would
    print the answer in the sequence of identifiers without printing it in any
    field, and a gate reading only the fields would see nothing. Exact: the printed
    identifier sequence is compared across every convention in the space, and any
    two that differ are a receipt whose order carries the rule.
    """
    if len(obs.orders) < 2:
        return False
    return len({tuple(order) for order in obs.orders.values()}) > 1


# --------------------------------------------------------------------------
# the gate
# --------------------------------------------------------------------------


@dataclass
class GateResult:
    """One instance's gate report. Nothing here is averaged over instances."""

    tag: str
    arity: dict[str, int]
    blocks: dict[str, int]
    dependence: tuple[frozenset[str], ...]
    n_rows: int
    n_evident: int
    placebo: float
    ceiling: float
    floor: float
    sampling: dict[str, float]
    r_pass: bool
    r_axes: list[str]
    s_pass: bool
    s_structural: str
    s_label_resolution_equal: bool
    s_leaks: list[str]
    s_order_moves: bool
    h_pass: bool
    min_headroom: float
    verdict: bool
    reasons: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)

    @property
    def headroom(self) -> float:
        return self.ceiling - self.floor

    def lines(self) -> list[str]:
        width = max((len(a) for a in self.arity), default=6) + 4
        head = "".join((f"{a} (c={self.arity[a]})").ljust(width + 4) for a in self.arity)
        blk = "".join((f"{self.blocks[a]}").ljust(width + 4) for a in self.arity)
        out = [
            f"instance               {self.tag}",
            f"printed rows           {self.n_rows} ({self.n_evident} evident)",
            "",
            "                       " + head,
            "signature blocks       " + blk,
            "",
            f"placebo (no receipt)   {self.placebo:8.4f}",
            f"LOOKUP FLOOR           {self.floor:8.4f}   "
            f"(sampling {self.sampling.get('floor', float('nan')):.4f})",
            f"CEILING                {self.ceiling:8.4f}   "
            f"(sampling {self.sampling.get('ceiling', float('nan')):.4f})",
            f"HEADROOM               {self.headroom:8.4f}   "
            f"(needs more than {self.min_headroom:.4f})",
            "",
            "GATE R  resolution     "
            + ("PASS on " + ", ".join(self.r_axes) if self.r_pass else "FAIL"),
            f"GATE S  non-self-int.  {'PASS' if self.s_pass else 'FAIL'}   "
            f"({self.s_structural})",
            f"GATE H  headroom       {'PASS' if self.h_pass else 'FAIL'}",
            f"VERDICT                {'USABLE' if self.verdict else 'REJECTED'}",
        ]
        for reason in self.reasons:
            out.append("   - " + reason)
        for caveat in self.caveats:
            out.append("   ? " + caveat)
        return out


def gate(
    obs: Observation,
    min_arity: int = 3,
    min_blocks: int = 2,
    min_headroom: float = 0.0,
) -> GateResult:
    """Run gates R, S and H on one rendered instance.

    An instance passes only if some axis of at least `min_arity` options resolves
    past `min_blocks`, the receipt does not print its own interpretation, and the
    ceiling stands above the lookup floor. Failing any one of the three, the receipt
    can produce a large, tight, significant and completely flat effect, which is the
    outcome the whole exercise exists to prevent.

    Thresholds are arguments. Their admission values are the maintainer's call, and
    nothing here bakes one in beyond the arithmetic: two blocks is where the agent
    learns only that it was wrong, and zero headroom is where the ceiling and the
    floor are literally the same partition.
    """
    space = obs.space
    index = space.index()
    arity = space.arity
    truths = obs.truths()

    blocks = {a: resolution_blocks(obs, a, index) for a in space.axes}
    dependence = row_dependence(obs, index)
    evident = evident_rows(obs, index)

    reasons: list[str] = []
    caveats: list[str] = []

    # ---- GATE R ----------------------------------------------------------
    wide = [a for a in space.axes if arity[a] >= min_arity]
    r_axes = [a for a in wide if blocks[a] > min_blocks]
    r_pass = bool(r_axes)
    if not r_pass:
        if not wide:
            reasons.append(
                f"R: no axis has {min_arity} or more options, so no axis can resolve past "
                "the pin"
            )
        else:
            worst = ", ".join(f"{a} {blocks[a]} of {arity[a]}" for a in wide)
            reasons.append(
                f"R: every axis of {min_arity} or more options is pinned at or below "
                f"{min_blocks} blocks ({worst}); the receipt tells the agent it was wrong "
                "and not what it should have done instead"
            )

    # ---- the readers -----------------------------------------------------
    drawable = obs.drawable
    ceiling_groups = np.unique(obs.whole, return_inverse=True)[1].reshape(-1)
    ceiling, ceiling_sampled = score_partition(
        ceiling_groups, obs.answers, truths, drawable
    )

    # the lookup floor: the evident rows, plus one all-passed bit per class of rows
    # a reader can see but not resolve. Both derived from the receipt.
    columns = [obs.shown[:, evident]] if evident.any() else []
    hidden = ~evident
    if hidden.any():
        for signature in sorted({dependence[i] for i in np.where(hidden)[0]}, key=sorted):
            members = np.array(
                [hidden[i] and dependence[i] == signature for i in range(obs.n_rows)]
            )
            same = (obs.shown[:, members] == obs.shown[obs.chat][members][None, :]).all(axis=1)
            columns.append(same.astype(np.int64)[:, None])
    floor_matrix = (
        np.concatenate(columns, axis=1)
        if columns
        else np.zeros((space.n_combos, 0), dtype=np.int64)
    )
    floor_groups = _classes(floor_matrix)
    floor, floor_sampled = score_partition(floor_groups, obs.answers, truths, drawable)

    placebo, _ = score_partition(
        np.zeros(space.n_combos, dtype=np.int64), obs.answers, truths, drawable
    )

    # ---- GATE S ----------------------------------------------------------
    labels = obs.row_labels()
    n_axis_labelled = sum(1 for label in labels if label == AXIS_LABEL)
    if n_axis_labelled == len(labels) and labels:
        s_structural = "every printed row is labelled by axis"
    elif n_axis_labelled:
        s_structural = f"{n_axis_labelled} of {len(labels)} printed rows are labelled by axis"
    else:
        s_structural = "every printed row is labelled by scored row"
    s_equal = abs(ceiling - floor) < 1e-12
    leaks = vocabulary_leaks(obs) + grammar_violations(obs)
    order_moves = order_moves_with_the_convention(obs)
    s_pass = not n_axis_labelled and not s_equal and not leaks and not order_moves
    if n_axis_labelled:
        reasons.append(
            "S: the receipt is labelled by axis, so it prints its own interpretation and "
            "nothing has to be inferred"
        )
    if s_equal:
        reasons.append(
            "S: the resolution reachable from the receipt's evident rows alone already "
            "equals the whole receipt's, so evaluating readouts buys nothing"
        )
    if leaks:
        reasons.append("S: the serialized receipt prints " + "; ".join(leaks))
    if order_moves:
        reasons.append(
            "S: the printed row order moves with the drawn convention, so the sequence of "
            "identifiers carries the answer"
        )

    # ---- GATE H ----------------------------------------------------------
    headroom = ceiling - floor
    h_pass = headroom > min_headroom + 1e-12
    if not h_pass:
        reasons.append(
            f"H: the ceiling {ceiling:.4f} stands at the lookup floor {floor:.4f}, so every "
            "point of the effect is reachable without induction and the effect is a constant"
        )

    caveats.append(
        "S is checked on the declared labels, the serialized bytes, and the printed order. "
        "Prose in the task text that gives the rule away needs a human read."
    )
    if obs.n_rows == 1:
        caveats.append(
            "a single printed row on a verdict-only receipt can produce at most two "
            "signatures, so this instance cannot pass R however rich its readout is"
        )

    return GateResult(
        tag=obs.tag,
        arity=arity,
        blocks=blocks,
        dependence=tuple(dependence),
        n_rows=obs.n_rows,
        n_evident=int(evident.sum()),
        placebo=placebo,
        ceiling=ceiling,
        floor=floor,
        sampling={"ceiling": ceiling_sampled, "floor": floor_sampled},
        r_pass=r_pass,
        r_axes=r_axes,
        s_pass=s_pass,
        s_structural=s_structural,
        s_label_resolution_equal=s_equal,
        s_leaks=leaks,
        s_order_moves=order_moves,
        h_pass=h_pass,
        min_headroom=min_headroom,
        verdict=bool(r_pass and s_pass and h_pass),
        reasons=reasons,
        caveats=caveats,
    )


def bits(obs: Observation) -> float:
    """How much the printed receipt says about the convention, in bits.

    A diagnostic, not a gate. It is what an entropy pre-check reports, and the whole
    point of R, S and H is that a large number here settles nothing.
    """
    drawable = obs.drawable
    groups = np.unique(obs.whole[drawable], return_inverse=True)[1].reshape(-1)
    sizes = np.bincount(groups)[groups].astype(float)
    return math.log2(int(drawable.sum())) - float(np.mean(np.log2(sizes)))


__all__ = [
    "AXIS_LABEL",
    "ROW_LABEL",
    "AxisSpace",
    "GateResult",
    "Observation",
    "axis_cells",
    "axis_receipts",
    "bits",
    "evident_rows",
    "gate",
    "grammar_violations",
    "resolution_blocks",
    "row_dependence",
    "rowwise_scores",
    "score_partition",
]
