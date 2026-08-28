"""The filing backlog, the house conventions that band it, and the answer key each one gives.

A task's world is seeded with a backlog of filing requests. Every dated request records four
dates; the appended paragraph prints a band table and asks for one band per request, and it
never says which recorded date the window starts from, which days in the window are counted,
which band a count on a printed figure takes, or what an undated request gets. Those four
unstated choices are the **convention**, and there are 64 of them.

The convention is what makes the chore gradeable without storing the answer anywhere. A band is
*computed* from a request's own dates rather than picked off a list, so two conventions agree on
some requests and disagree on others, and which requests they disagree on is different for every
pair. A verdict vector over the backlog is therefore something to explain rather than a bit to
invert, which is the property the whole measurement rests on.

A backlog is only shipped if it earns that property, and :meth:`Backlog.audit` is the test: the
64 conventions must give 64 distinct answer keys, and moving any one axis on its own must change
at least two requests' bands. A backlog that fails is redrawn, up to :data:`ATTEMPTS` times; one
that never passes is not served.

Everything here is pure, offline, and a deterministic function of the seed and the reference
date. No AppWorld import, no world state, no clock.
"""

from __future__ import annotations

import datetime as dt
import math
import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import numpy as np

# ----- the convention space -----

#: The four dates every dated request records, in the order the world prints them. The `anchor`
#: axis is which of these the ageing window starts from.
ROLES: Tuple[str, ...] = ("raised", "acknowledged", "actioned", "closed")

#: Which days inside the window are counted. The two independent toggles of a real ageing
#: configuration: weekends in or out, and the project's closed dates in or out.
BASIS_OPTIONS: Tuple[str, ...] = (
    "calendar",
    "calendar_minus_closures",
    "business",
    "business_minus_closures",
)

#: Which band a count that lands exactly on one of the printed figures takes. The printed ranges
#: overlap at their ends, which is what makes the question a real one.
BOUNDARY_OPTIONS: Tuple[str, ...] = ("lower", "upper")

#: What a request with no recorded dates gets. Both ends of the printed table; a reader can
#: construct either, and neither is named.
MISSING_OPTIONS: Tuple[str, ...] = ("lowest", "highest")

#: The figures printed in the band table, and the bands they separate.
CUTS: Tuple[int, ...] = (5, 10, 20, 30)
BANDS: Tuple[str, ...] = ("Routine", "Standard", "Priority", "Urgent", "Critical")

#: How far back a request may be dated. Wide enough that every band above `Routine` is reachable
#: under every counting basis, and no wider: a window past the top cut lands in `Critical`
#: whatever the convention, which is a request that separates nothing.
SPAN = 46

#: How many times a backlog is redrawn before the reference date is given up on.
#:
#: Raising this is additive and nothing else: attempts are tried in order and the first admissible
#: draw wins, so a task that already had a backlog keeps exactly the backlog it had. What a higher
#: cap changes is only the tasks that had none. A manifest that lists a task the generator gives
#: up on is a task that raises when it is served.
ATTEMPTS = 500

#: How many requests a shipped backlog holds. Four options on two of the axes need this many to
#: separate all 64 conventions; shorter backlogs collide.
DATED = 28
UNDATED = 1

#: The sections seeded into the project, and the shape of its description. The section names
#: collide with none the corpus ships, so a section read back is one this backlog put there.
SECTIONS: Tuple[str, ...] = ("Inbound", "Outward", "Standby", "Cleared")
WORKED_DAYS = "Monday to Friday"


class Convention(NamedTuple):
    """One house convention: the four choices the instruction leaves open."""

    anchor: str
    basis: str
    boundary: str
    missing: str


#: Every convention, in a fixed order. The order is what a drawn index means, so it is a
#: constant of the environment rather than something a caller may sort.
CONVENTIONS: Tuple[Convention, ...] = tuple(
    Convention(anchor=a, basis=b, boundary=o, missing=m)
    for a in ROLES
    for b in BASIS_OPTIONS
    for o in BOUNDARY_OPTIONS
    for m in MISSING_OPTIONS
)

#: The convention a reader reaches for unprompted, measured on a panel of readers rather than
#: assumed: file from the last recorded date, exclude everything the world gives grounds to
#: exclude, take the upper band on a tie, and give an undated request the lowest band. It is the
#: applied convention the roster's own verdict counts are enumerated against
#: (:func:`pass_count_marginal`), and it is never what the environment draws.
REFERENCE_CONVENTION = Convention(
    anchor="closed", basis="business_minus_closures", boundary="upper", missing="lowest"
)

_AXES: Tuple[str, ...] = ("anchor", "basis", "boundary", "missing")
_OPTIONS: Dict[str, Tuple[str, ...]] = {
    "anchor": ROLES,
    "basis": BASIS_OPTIONS,
    "boundary": BOUNDARY_OPTIONS,
    "missing": MISSING_OPTIONS,
}
_CONVENTION_INDEX: Dict[Convention, int] = {c: i for i, c in enumerate(CONVENTIONS)}


def band_of(count: int, boundary: str) -> str:
    """The band a window of ``count`` counted days falls in, under one boundary rule.

    The printed ranges share their ends, so a count that lands on a printed figure belongs to
    two of them and the boundary rule says which. ``lower`` keeps it in the range below,
    ``upper`` pushes it into the range above."""
    above = (count > cut if boundary == "lower" else count >= cut for cut in CUTS)
    return BANDS[sum(1 for step in above if step)]


# ----- one seeded backlog -----


class Request(NamedTuple):
    """One waiting request: its printed reference, and the four dates it records (or none)."""

    reference: str
    dates: Optional[Dict[str, dt.date]]


@dataclass(frozen=True)
class Backlog:
    """One task's seeded backlog, with every convention's answer key precomputed.

    ``keys`` is a 64-by-29 array of band indices, one row per convention in
    :data:`CONVENTIONS` order and one column per request in printed order, which is the object
    every gate and every scorer reads. Building it once is what makes the audit, the scoring and
    the roster enumeration cheap enough to run on the whole split."""

    reference: dt.date
    closures: Tuple[dt.date, ...]
    requests: Tuple[Request, ...]
    keys: np.ndarray
    attempts: int

    @property
    def description(self) -> str:
        """The project description: the working week and the closed dates, and nothing else.

        It is the only place the closed dates appear, and it carries no band, no section and no
        due date, because a seeded row that named one would hand over the answer on every task
        at once."""
        closed = ", ".join(d.isoformat() for d in self.closures)
        return f"Worked: {WORKED_DAYS}. Closed: {closed}"

    def key(self, convention: Convention) -> Tuple[str, ...]:
        """The bands ``convention`` gives, one per request, in printed order."""
        row = self.keys[_CONVENTION_INDEX[convention]]
        return tuple(BANDS[i] for i in row)

    def audit(self) -> Tuple[bool, Dict[str, int]]:
        """Whether this backlog separates the conventions, and by how much per axis.

        Two conditions. Every convention must give a different answer key, or a verdict vector
        cannot tell two of them apart however well it is read. And moving one axis while holding
        the other three must change at least two requests, so no axis is decided by a single
        request: one request produces two signatures whatever its readout, which caps what any
        reader can recover from it however rich the request is.

        The `missing` axis is exempt from the second condition by construction: it is read off
        the one undated request, so a flip on it moves exactly one row and no backlog can do
        better."""
        distinct = len(set(map(tuple, self.keys.tolist()))) == len(CONVENTIONS)
        return distinct, {axis: self._least_flip(axis) for axis in _AXES}

    def _least_flip(self, axis: str) -> int:
        """The fewest requests any single move on ``axis`` changes, over every convention."""
        position = _AXES.index(axis)
        least = self.keys.shape[1]
        for convention in CONVENTIONS:
            base = self.keys[_CONVENTION_INDEX[convention]]
            for option in _OPTIONS[axis]:
                if option == convention[position]:
                    continue
                moved = convention._replace(**{axis: option})
                other = self.keys[_CONVENTION_INDEX[moved]]
                least = min(least, int(np.count_nonzero(base != other)))
        return least


def _cover_gains(pool: np.ndarray, cover: np.ndarray, scale: int) -> np.ndarray:
    """Each candidate's gain against the cover so far, in exact integers.

    Named and separated from the loop that uses it so a test can assert what makes it
    architecture independent, which is a property of the dtypes on this path rather than of the
    machine the test runs on. One machine can therefore check a claim about every machine: if
    nothing here is a float, no BLAS kernel's summation order can reach the result."""
    return pool @ (3 * scale // (1 + cover))


def _admissible(backlog: Backlog) -> bool:
    """Whether a built backlog clears both gates (see :meth:`Backlog.audit`)."""
    distinct, flips = backlog.audit()
    if not distinct:
        return False
    return min(flips[axis] for axis in ("anchor", "basis", "boundary")) >= 2


# ----- the generator -----


def build_backlog(
    seed: int,
    reference: dt.date,
    *,
    dated: int = DATED,
    undated: int = UNDATED,
    attempts: int = ATTEMPTS,
) -> Optional[Backlog]:
    """Draw a backlog for ``reference`` that clears the gates, or ``None`` if none does.

    Deterministic in ``seed`` and ``reference``: the same pair returns the same backlog, down to
    the references printed on the requests, on any machine and in any process. A backlog that
    fails the audit is redrawn rather than repaired, because repairing one means choosing which
    request to move and that choice would be made against the answer key."""
    for attempt in range(attempts):
        rng = random.Random(seed * 100003 + attempt)
        requests, closures = _draw(rng, reference, dated, undated)
        backlog = _with_keys(reference, closures, requests, attempt)
        if _admissible(backlog):
            return backlog
    return None


def _with_keys(
    reference: dt.date,
    closures: Sequence[dt.date],
    requests: Sequence[Request],
    attempt: int,
) -> Backlog:
    """Attach every convention's answer key to a drawn set of requests."""
    counts = _count_table(reference, closures)
    keys = np.empty((len(CONVENTIONS), len(requests)), dtype=np.int8)
    for i, convention in enumerate(CONVENTIONS):
        missing_band = BANDS[0] if convention.missing == "lowest" else BANDS[-1]
        missing_index = BANDS.index(missing_band)
        for j, request in enumerate(requests):
            if request.dates is None:
                keys[i, j] = missing_index
            else:
                days = counts[request.dates[convention.anchor]][convention.basis]
                keys[i, j] = BANDS.index(band_of(days, convention.boundary))
    return Backlog(
        reference=reference,
        closures=tuple(closures),
        requests=tuple(requests),
        keys=keys,
        attempts=attempt,
    )


def _count_table(
    reference: dt.date, closures: Iterable[dt.date]
) -> Dict[dt.date, Dict[str, int]]:
    """How many days each basis counts in the window opened by each candidate date.

    The window begins the day *after* the date it is anchored to and ends on the reference date,
    which is the world's own "today"."""
    closed = set(closures)
    table: Dict[dt.date, Dict[str, int]] = {}
    calendar = closures_out = business = business_out = 0
    for step in range(0, SPAN + 14):
        day = reference - dt.timedelta(days=step)
        table[day] = {
            "calendar": calendar,
            "calendar_minus_closures": closures_out,
            "business": business,
            "business_minus_closures": business_out,
        }
        # Walking backwards one day extends every window by that day, so the next entry is this
        # one plus whatever the day itself counts for.
        calendar += 1
        if day not in closed:
            closures_out += 1
        if day.weekday() < 5:
            business += 1
            if day not in closed:
                business_out += 1
    return table


def _draw_closures(rng: random.Random, reference: dt.date) -> List[dt.date]:
    """Three closed dates, one from each third of the span, all on working days.

    Spread rather than drawn freely so a closure falls inside most windows: three closures
    bunched at one end would leave the two `_minus_closures` bases agreeing with their
    counterparts on most requests, and an axis whose options agree everywhere separates
    nothing."""
    chosen: List[dt.date] = []
    for low, high in ((3, 9), (11, 22), (24, SPAN - 4)):
        for _ in range(300):
            day = reference - dt.timedelta(days=rng.randint(low, high))
            if day.weekday() < 5 and day not in chosen:
                chosen.append(day)
                break
    return sorted(chosen)


def _draw(
    rng: random.Random, reference: dt.date, dated: int, undated: int
) -> Tuple[List[Request], List[dt.date]]:
    """One attempt at a backlog: closures, requests, and the references printed on them.

    Two thirds of the dated requests are *designed*, chosen greedily so that as many
    (anchor, basis) settings as possible have some request whose count lands exactly on a printed
    figure. Those are the requests the `boundary` axis is decided by, and spreading them over the
    settings is what stops one axis from being carried by a single request. The rest are drawn
    freely, so the backlog does not read as sixteen constructed cases.

    References are assigned **after** the shuffle, so their order carries nothing: sorted
    references would group requests by construction order and let a reader partition the backlog
    without evaluating anything."""
    closures = _draw_closures(rng, reference)
    counts = _count_table(reference, closures)
    settings = [(a, b) for a in ROLES for b in BASIS_OPTIONS]
    cut_values = set(CUTS)

    candidates: List[Tuple[dt.date, ...]] = []
    hits: List[List[int]] = []
    for back in range(4, SPAN + 1):
        first = reference - dt.timedelta(days=back)
        for gap1 in range(1, 6):
            second = first + dt.timedelta(days=gap1)
            if second >= reference:
                continue
            for gap2 in range(1, 7):
                third = second + dt.timedelta(days=gap2)
                if third >= reference:
                    continue
                for gap3 in range(1, 8):
                    fourth = third + dt.timedelta(days=gap3)
                    if fourth >= reference:
                        continue
                    dates = (first, second, third, fourth)
                    on_cut = [
                        1 if counts[dates[ROLES.index(a)]][b] in cut_values else 0
                        for a, b in settings
                    ]
                    if any(on_cut):
                        candidates.append(dates)
                        hits.append(on_cut)

    order = list(range(len(candidates)))
    rng.shuffle(order)
    order = order[:2500]
    pool = (
        np.array([hits[i] for i in order], dtype=np.int64)
        if order
        else np.zeros((0, len(settings)), dtype=np.int64)
    )

    designed = max(3, (dated * 2) // 3)
    # Integers, and that is the point of this block rather than a preference.
    #
    # A float dot product is not the same number on every machine: BLAS sums in whatever order
    # its kernel picks, so one candidate's gain differs in the last bit between one architecture's
    # kernel and another's, and two gains that are equal in arithmetic can compare either way. The
    # pick changes, and with it the draw, whether the draw is admissible, which attempt first
    # succeeds, and the frozen pass-count table built out of all of it. A backlog is meant to be
    # the same on every machine.
    #
    # Every weight is `3 / (1 + cover)` for an integer cover bounded by the number of picks, so a
    # common denominator makes the comparison exact. Integer addition is associative, so the sum
    # is the same whatever order numpy adds it in, which is what floats could not promise.
    scale = math.lcm(*range(1, designed + 2))
    cover = np.zeros(len(settings), dtype=np.int64)
    taken = np.zeros(len(order), dtype=bool)
    chosen: List[Tuple[dt.date, ...]] = []
    for _ in range(min(designed, dated)):
        if not len(order) or taken.all():
            break
        gains = _cover_gains(pool, cover, scale)
        # Gains are non-negative, so -1 is below every real candidate. `argmax` takes the first
        # maximum, which is well defined on exact integers.
        gains[taken] = -1
        pick = int(np.argmax(gains))
        taken[pick] = True
        cover = cover + pool[pick]
        chosen.append(candidates[order[pick]])

    requests: List[Request] = [Request(reference="", dates=dict(zip(ROLES, d))) for d in chosen]
    guard = 0
    while len(requests) < dated and guard < 500:
        guard += 1
        first = reference - dt.timedelta(days=rng.randint(4, SPAN))
        second = first + dt.timedelta(days=rng.randint(1, 5))
        third = second + dt.timedelta(days=rng.randint(1, 6))
        fourth = third + dt.timedelta(days=rng.randint(1, 7))
        if fourth >= reference:
            continue
        requests.append(
            Request(reference="", dates=dict(zip(ROLES, (first, second, third, fourth))))
        )
    requests = requests[:dated]
    requests.extend(Request(reference="", dates=None) for _ in range(undated))
    rng.shuffle(requests)
    return (
        [
            Request(reference="RQ-%04d" % (1000 + 7 * i + rng.randint(0, 6)), dates=r.dates)
            for i, r in enumerate(requests)
        ],
        closures,
    )


# ----- what a roster's real verdict counts look like -----


class Posterior(NamedTuple):
    """What one receipt leaves standing, enumerated rather than argued about.

    ``distinct`` is how many different verdict vectors the 64 conventions produce against one
    submission, ``largest`` the size of the biggest set of conventions that produce the same one,
    and ``mean_size`` the average number of conventions still standing after the receipt is read,
    under a uniform prior."""

    distinct: int
    largest: int
    mean_size: float


def posterior_profile(backlog: "Backlog", applied: Convention) -> Posterior:
    """What a receipt on ``applied``'s own answer leaves standing about the drawn convention.

    The observation is the vector of bits saying, per request, whether the drawn convention would
    have written what the agent wrote. Two conventions that produce the same vector are two a
    reader cannot separate however well it reads, so counting the vectors is the exact statement
    of what one grade can and cannot settle.

    **It is not one, and no length of backlog makes it one.** Some pairs of conventions agree on
    every request the world can supply: when no window under the drawn anchor lands on a printed
    figure, the boundary rule changes nothing anywhere, and no extra request can make it. What the
    receipt supports is matching the drawn convention closely, not naming it, and this is the
    number that says by how much."""
    submission = backlog.keys[_CONVENTION_INDEX[applied]]
    tally: Dict[Tuple[bool, ...], int] = {}
    for row in backlog.keys:
        vector = tuple((row == submission).tolist())
        tally[vector] = tally.get(vector, 0) + 1
    total = len(CONVENTIONS)
    return Posterior(
        distinct=len(tally),
        largest=max(tally.values()),
        mean_size=sum(size * size for size in tally.values()) / total,
    )


def pass_count_marginal(backlogs: Iterable[Backlog]) -> Tuple[int, ...]:
    """How often each number of passing requests occurs across a roster, as raw counts.

    Enumerated over every backlog crossed with every convention the draw can produce, with
    :data:`REFERENCE_CONVENTION` filed against it. It is the distribution a real receipt's pass
    count is drawn from, so a payload that draws its count from here is not separable from a real
    one by its count alone: a fixed count, or a symmetric one, would be.

    Returned as counts over ``0 .. dated`` rather than as probabilities, so the table is exact
    integers and two builds of the same roster are byte-comparable."""
    tally: Optional[np.ndarray] = None
    for backlog in backlogs:
        dated = np.array([r.dates is not None for r in backlog.requests])
        applied = backlog.keys[_CONVENTION_INDEX[REFERENCE_CONVENTION]][dated]
        if tally is None:
            tally = np.zeros(int(dated.sum()) + 1, dtype=np.int64)
        agreed = (backlog.keys[:, dated] == applied).sum(axis=1)
        for count in agreed:
            tally[int(count)] += 1
    if tally is None:
        return ()
    return tuple(int(x) for x in tally)


__all__ = [
    "ATTEMPTS",
    "BANDS",
    "BASIS_OPTIONS",
    "BOUNDARY_OPTIONS",
    "CONVENTIONS",
    "CUTS",
    "DATED",
    "MISSING_OPTIONS",
    "REFERENCE_CONVENTION",
    "ROLES",
    "SECTIONS",
    "SPAN",
    "UNDATED",
    "Backlog",
    "Convention",
    "Posterior",
    "Request",
    "band_of",
    "build_backlog",
    "pass_count_marginal",
    "posterior_profile",
]
