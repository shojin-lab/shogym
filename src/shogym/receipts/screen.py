"""The room screen: what one graded receipt was worth, on a family that has been run.

The gates in `resolution` are enumerable at zero execution cost and say whether a
receipt CAN carry a measurement. The room screen is empirical and says what one
receipt DID carry. It is the one-receipt criterion: each task pair is one
execution of task A and three of task B, a graded branch, a byte-matched placebo
branch, and an oracle branch that is told the rule, all at zero prior dose.

Per pair j the two contrasts are

    x_j = graded_j - placebo_j      the gain the graded receipt produced
    y_j = oracle_j - placebo_j      the room a perfect reader had

and the family's extraction ratio is a ratio of two AGGREGATED differences,

    rho = mean_j(x_j) / mean_j(y_j),

never a mean of per-pair ratios. A family whose pairs sit at the ceiling has
y_j at or below zero and no room to extract, which is the failure this screen
exists to catch: a receipt can pass every gate and still be worth nothing on a
task the agent already solves.

THE FLOOR

A pooled denominator below `floor` is not turned into a ratio, because a ratio
whose denominator is noise around zero is not a number. Three ways of not turning
it into a ratio are implemented: `drop` reports the ratio as NaN and the screen
fails on room rather than pretending to a value, `clamp` replaces the denominator
with the floor, and `none` divides anyway and is kept only to show what the floor
buys. A denominator at zero is NaN under all three: a family with no room has no
ratio, whatever the floor was set to.

THE VARIANCE

Delta method, with x_bar and y_bar the means over P pairs:

    rho_hat - rho  ~  (x_bar - rho * y_bar) / mu_y
    Var(rho_hat)   ~  (s_x^2 + rho^2 s_y^2 - 2 rho r s_x s_y) / (P mu_y^2)

so the per-pair quantity whose SD sets the budget on this scale is the influence
value (x_j - rho y_j) / mu_y, and `sd_influence` computes it from the five
moments.

THE BARS ARE REGISTERED. What a family must show to be admitted is the maintainer's
call, and the call has been made: the oracle must beat the placebo by at least
`REGISTERED_MIN_ROOM` with the interval's lower bound above zero, and one graded
receipt must take at least `REGISTERED_MIN_RATIO` of that room, over at least
`REGISTERED_MIN_PAIRS` DISTINCT tasks. They are defaults rather than constants,
because a diagnostic run may want to see what a family does against another bar: a
result says whether the bars it was judged against were the registered ones, and a
caller that deals families requires the registered ones or does not deal.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

FLOOR_RULES = ("drop", "clamp", "none")

#: The registered sample. A pair is one execution of A and three of B, so 36 pairs is
#: 36 A executions and 108 B executions, 144 in all, which is the costing a cheap
#: generated family was planned against. It is a default rather than a caller's
#: choice: a screen is the only evidence that a family's room can actually be
#: converted, and a caller free to pick the sample can pick the one that passes.
REGISTERED_MIN_PAIRS = 36
#: The registered bars. A family whose oracle beats its placebo by less than this has
#: too little room for one receipt to carry anything, and a family whose one graded
#: receipt takes less than a quarter of the room it had is not converting it.
REGISTERED_MIN_ROOM = 0.05
REGISTERED_MIN_RATIO = 0.25

#: How many candidates a screen may have been selected from and still be DEAL
#: EVIDENCE. The best of several clears a bar more easily than one does, and nothing
#: here corrects for it: the interval, the bars and the verdict are identical for one
#: candidate and for a thousand. Until an adjustment is registered, the conservative
#: reading of a selected winner is that it has not met this stage, so deal evidence
#: takes one candidate. A screen that names more is still scored and still printed,
#: as the diagnostic it is.
REGISTERED_MAX_CANDIDATES = 1

#: THE REGISTERED RESOLUTION OF A COMPARISON. Room and ratio are means of binary
#: floats, and a sample that is exactly at a bar in decimal arrives here a fraction
#: under it: 36 pairs of oracle 0.35 against placebo 0.30 give a room of
#: 0.049999999999999996, which prints as 0.0500 and fails a bar of 0.05. The bars are
#: registered as inclusive, so a comparison has to be inclusive at the resolution the
#: numbers carry. This is far below any difference a screen could mean and far above
#: the accumulated error of averaging a few hundred six-place scores.
COMPARISON_RESOLUTION = 1e-9


def at_least(value: float, bar: float) -> bool:
    """Whether `value` clears an inclusive `bar` at the registered resolution.

    ONE PLACE, so screening and the bundle's re-verification cannot disagree about
    what "at least" means, and a family is not admitted by one and refused by the
    other on the last bit of a float.
    """
    return bool(math.isfinite(value) and value >= bar - COMPARISON_RESOLUTION)
#: How many resamples the reported interval is taken over, and how wide it is.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_MASS = 0.90


def _reject_constant(token: str) -> float:
    """JSON's nonfinite extensions, refused where a run is read.

    Python's JSON reader accepts `NaN` and `Infinity` by default. They are not JSON,
    and a sentinel that arrives as a float becomes a plausible name one `str()` later,
    so the shortest place to stop them is the parse.
    """
    raise ValueError(f"a screen artifact carries {token}, which is not a JSON number")


def _one_value(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Object members, refusing a name that appears twice.

    A JSON object with two members of one name has no single value. Python keeps the
    last, another reader or a person auditing the file can take the first, and an
    artifact that two readers disagree about is not a record of anything. It is worth
    refusing rather than resolving: whichever one is kept, one of the two readings was
    silently discarded.
    """
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(
                f"the name {key!r} appears twice in one object, so this file has no "
                "single value"
            )
        seen[key] = value
    return seen


def read_payload(text: str) -> object:
    """Parse a stored run: no nonfinite extensions, and no repeated names."""
    return json.loads(text, parse_constant=_reject_constant, object_pairs_hook=_one_value)


def _identity(value: object, name: str) -> str:
    """One name out of a stored run, refused rather than defaulted when absent.

    Numbers are allowed, because a task seed is often one, and they are written out
    canonically. Everything else has to arrive as text with something in it: a null,
    a blank, a structure or a nonfinite float is a missing identity, and a missing
    identity is what this exists to catch. `str(float("nan"))` is the perfectly
    ordinary-looking name "nan".
    """
    if value is None:
        raise ValueError(f"the run names no {name}")
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError(f"the run's {name} is not a name")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"the run's {name} is {value!r}, which names nothing")
    text = str(value).strip()
    if not text:
        raise ValueError(f"the run's {name} is blank")
    return text


def _pair(row: object) -> "PairRecord":
    """One pair row, requiring exactly the five fields a pair is."""
    if not isinstance(row, dict):
        raise ValueError("a pair is a record of five fields")
    if set(row) != set(PAIR_FIELDS):
        raise ValueError(
            "a pair names exactly %s, and this one names %s"
            % (", ".join(PAIR_FIELDS), ", ".join(sorted(row)) or "nothing")
        )
    return PairRecord(
        instance=_identity(row["instance"], "instance"),
        filing=_identity(row["filing"], "filing"),
        placebo=_score(row["placebo"], "placebo"),
        graded=_score(row["graded"], "graded"),
        oracle=_score(row["oracle"], "oracle"),
    )


def _score(value: object, name: str) -> float:
    """One branch score, refused rather than coerced when absent."""
    if value is None or isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"a pair's {name} is not a score")
    return _as_float(value, f"a pair's {name}")


#: What one pair row says, and the whole of it. A field the reader ignores is a field
#: a stale conclusion travels in, and a reader who finds `passed: true` beside three
#: scores has no way to know nothing checked it.
PAIR_FIELDS = ("instance", "filing", "placebo", "graded", "oracle")


@dataclass(frozen=True)
class PairRecord:
    """One task pair's three branch scores, and which pair they came from.

    The identities are the point. Three numbers say what was measured; they do not
    say what was measured ON, and a file of anonymous scores verifies as readily
    against a family and a model it never touched.
    """

    instance: str
    filing: str
    placebo: float
    graded: float
    oracle: float


@dataclass(frozen=True)
class ScreenRun:
    """A pilot: which family, which model, which task seeds, and one record per pair.

    This is the artifact a screen record hashes and binds to. It carries its own
    provenance, so a claim about which family, which model and which task set produced
    it is checkable against the thing itself rather than taken from whoever typed it.

    THE FAMILY IS PART OF THE RUN. A run that named only its numbers is a run anything
    can claim: a pilot taken on one family reads as evidence for another, and the
    reader being asked to deal a family sees three statistics with nothing in them
    that says what they were taken on.
    """

    family: str
    model: str
    task_seeds: tuple[str, ...]
    pairs: tuple[PairRecord, ...]

    def __post_init__(self) -> None:
        if not self.family.strip():
            raise ValueError("a screen run names the family it was taken on")
        if not self.model.strip():
            raise ValueError("a screen run names the model it was taken with")
        if not self.task_seeds:
            raise ValueError("a screen run names the task seeds it was taken over")
        if not self.pairs:
            raise ValueError("a screen run holds at least one pair")
        if len(self.task_seeds) != len(self.pairs):
            raise ValueError(
                f"the run names {len(self.task_seeds)} task seeds and holds "
                f"{len(self.pairs)} pairs; the two are one ordered set"
            )
        # ONE ROW PER TASK. A pair is one execution of A and three of B on ONE task,
        # and the sample the screen reports is a count of tasks. Forty filings against
        # one clerical table are forty observations of a single unit: they clear the
        # floor, and the pair bootstrap prices them as forty independent draws, which is
        # a sample size the pilot does not have.
        for name, values in (
            ("task seeds", self.task_seeds),
            ("instances", tuple(p.instance for p in self.pairs)),
        ):
            if len(set(values)) != len(values):
                repeated = sorted({v for v in values if list(values).count(v) > 1})
                raise ValueError(
                    f"this run repeats {name} ({', '.join(repeated[:3])}); a pair is one "
                    "task, and repeated observations of one task are not a wider sample"
                )
        for pair in self.pairs:
            if not str(pair.instance).strip() or not str(pair.filing).strip():
                raise ValueError("a pair of this run names no instance or no filing")

    @property
    def distinct_instances(self) -> int:
        """How many task units this run was taken over. One per pair, by construction."""
        return len({p.instance for p in self.pairs})

    def outcomes(self) -> "Outcomes":
        return Outcomes(
            placebo=tuple(p.placebo for p in self.pairs),
            graded=tuple(p.graded for p in self.pairs),
            oracle=tuple(p.oracle for p in self.pairs),
        )

    @classmethod
    def from_payload(cls, payload: object) -> "ScreenRun":
        """Read a run out of its stored form, refusing anything that is not one.

        Every identity is validated BEFORE it is converted. `str(None)` is the
        nonempty string "None", so converting first turns a pilot export that lost
        its model field into a run that names a model, and the absence becomes
        unrecoverable one line before anything would have noticed it.
        """
        if not isinstance(payload, dict):
            raise ValueError(
                "a screen artifact is a run naming its family, its model, its task "
                "seeds and its pairs, not a bare list of scores"
            )
        try:
            family = _identity(payload["family"], "family")
            model = _identity(payload["model"], "model")
            seeds = payload["task_seeds"]
            if not isinstance(seeds, (list, tuple)):
                raise ValueError("task_seeds is a list of seeds")
            pairs = tuple(_pair(row) for row in payload["pairs"])
            return cls(
                family=family,
                model=model,
                task_seeds=tuple(
                    _identity(seed, f"task seed {n}") for n, seed in enumerate(seeds)
                ),
                pairs=pairs,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"the screen artifact is not a readable run: {exc}") from exc


#: Everything a stored screen has to name. The rows say what was measured; these say
#: what it was measured against, and a decision input that can be absent is a decision
#: input the reader supplies, which means the record does not say what was decided.
RUN_FIELDS = ("family", "model", "task_seeds", "pairs")
DECISION_FIELDS = (
    "min_room",
    "min_ratio",
    "min_pairs",
    "floor",
    "floor_rule",
    "candidates_screened",
    "selection_note",
)


@dataclass(frozen=True)
class ScreenRecord:
    """A run and the bars it was judged against, as one artifact.

    The two halves are one file because they are one claim. A file of rows beside
    thresholds someone types at verification time is a screen whose result depends on
    who reruns it, and the selection disclosure in particular is worth nothing when it
    can be omitted and defaulted to a single candidate.

    Nothing here reads the count or the note into a number: a record of a thousand
    candidates is SCORED exactly as a record of one, so what the arithmetic establishes
    about a selected winner is what it establishes about any single candidate, and the
    disclosure adds only that somebody said it was selected. That is why the count
    decides eligibility separately, in `dealable_selection`: the statistics stay a
    diagnostic and a bundle may not be frozen on a selected record. The registered
    adjustment is an open maintainer call.
    """

    run: "ScreenRun"
    min_room: float
    min_ratio: float
    min_pairs: int
    floor: float
    floor_rule: str
    candidates_screened: int
    selection_note: str

    @property
    def registered(self) -> bool:
        """Whether this record was judged against the registered bars."""
        return not self.overrides()

    def overrides(self) -> list[str]:
        """Which bars were moved off their registered values, named for printing."""
        moved = []
        for name, registered in (
            ("min_room", REGISTERED_MIN_ROOM),
            ("min_ratio", REGISTERED_MIN_RATIO),
            ("min_pairs", REGISTERED_MIN_PAIRS),
        ):
            value = getattr(self, name)
            if value != registered:
                moved.append(f"{name}={value:g} against the registered {registered:g}")
        return moved

    @property
    def dealable_selection(self) -> bool:
        """Whether this record's selection makes it deal evidence.

        Separate from the verdict on purpose. `screen` scores a selected record and
        prints it, because a diagnostic run wants the number; what a bundle may be
        frozen on is a different question, and until an adjustment is registered the
        answer for a selected winner is no.
        """
        return self.candidates_screened <= REGISTERED_MAX_CANDIDATES

    def result(self, family: str) -> "ScreenResult":
        """Rerun the screen on this record's own rows and its own bars.

        The family a caller asks about has to be the family the run says it was taken
        on. A label supplied here and nowhere else is a label anyone can change, and a
        pilot taken on one family would answer for another.
        """
        if family != self.run.family:
            raise ValueError(
                f"this screen was taken on {self.run.family!r} and is being read as "
                f"evidence for {family!r}"
            )
        return screen(
            family,
            self.run.outcomes(),
            min_room=self.min_room,
            min_ratio=self.min_ratio,
            min_pairs=self.min_pairs,
            candidates_screened=self.candidates_screened,
            selection_note=self.selection_note,
            floor=self.floor,
            floor_rule=self.floor_rule,
        )

    @classmethod
    def from_payload(cls, payload: object) -> "ScreenRecord":
        """Read a record, requiring exactly the fields a screen is decided by."""
        if not isinstance(payload, dict):
            raise ValueError("a screen artifact is a mapping")
        expected = set(RUN_FIELDS) | set(DECISION_FIELDS)
        if set(payload) != expected:
            missing = sorted(expected - set(payload))
            extra = sorted(set(payload) - expected)
            raise ValueError(
                "a screen artifact names exactly %s; this one %s%s"
                % (
                    ", ".join(sorted(expected)),
                    ("is missing " + ", ".join(missing)) if missing else "",
                    (("; and carries " if missing else "carries ") + ", ".join(extra))
                    if extra
                    else "",
                )
            )
        run = ScreenRun.from_payload({name: payload[name] for name in RUN_FIELDS})
        rule = payload["floor_rule"]
        if not isinstance(rule, str) or rule not in FLOOR_RULES:
            raise ValueError(f"a floor rule is one of {FLOOR_RULES}, not {rule!r}")
        note = payload["selection_note"]
        if not isinstance(note, str):
            raise ValueError("a selection note is text")
        candidates = _whole(payload["candidates_screened"], "candidates_screened")
        if candidates > 1 and not note.strip():
            # An undisclosed selection is refused; a disclosed one is recorded and
            # scored identically. This is the whole of what the count does.
            raise ValueError(
                f"{candidates} candidates were screened and the record says nothing "
                "about the selection, so the best of several would be read as one"
            )
        return cls(
            run=run,
            min_room=_bar(payload["min_room"], "min_room"),
            min_ratio=_bar(payload["min_ratio"], "min_ratio"),
            min_pairs=_whole(payload["min_pairs"], "min_pairs"),
            floor=_bar(payload["floor"], "floor"),
            floor_rule=rule,
            candidates_screened=candidates,
            selection_note=note,
        )


def _as_float(value: object, what: str) -> float:
    """One JSON number as a float, refusing what will not become one.

    JSON integers are unbounded and Python floats are not, so a record carrying
    10**400 raised `OverflowError` out of the conversion, past every range check and
    out of the command that promised to refuse rather than crash. An out-of-range
    number is a malformed record like any other and is refused as one.
    """
    try:
        return float(value)  # type: ignore[arg-type]
    except OverflowError as exc:
        raise ValueError(f"{what} is out of range for a score: {exc}") from exc


def _bar(value: object, name: str) -> float:
    """One recorded threshold, refused rather than coerced."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} is {value!r}; a screen bar is a number")
    number = _as_float(value, name)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(
            f"{name} is {value!r}; a screen bar is a finite number between 0 and 1"
        )
    return number


def _whole(value: object, name: str) -> int:
    """One recorded count, refused rather than coerced."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} is {value!r}; a count is a whole number")
    if value < 1:
        raise ValueError(f"{name} is {value!r}; a count is one or more")
    return int(value)


@dataclass(frozen=True)
class Outcomes:
    """One family's screen data: three branch scores per task pair, in pair order.

    Every sequence has one entry per pair, and the three entries at index j are the
    same pair's placebo, graded and oracle executions of task B.
    """

    placebo: tuple[float, ...]
    graded: tuple[float, ...]
    oracle: tuple[float, ...]

    def __post_init__(self) -> None:
        n = len(self.placebo)
        if not n:
            raise ValueError("the screen needs at least one pair")
        if len(self.graded) != n or len(self.oracle) != n:
            raise ValueError("the three branches must carry the same number of pairs")
        # A branch score is a component score, and a component score lives in [0, 1].
        # A screen that accepted 2.0 would report a ratio above one and call a family
        # admissible on evidence no run could have produced.
        for branch, values in (
            ("placebo", self.placebo), ("graded", self.graded), ("oracle", self.oracle)
        ):
            for value in values:
                if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                    raise ValueError(
                        f"the {branch} branch carries {value!r}; a component score is a "
                        "finite number between 0 and 1"
                    )

    @property
    def n_pairs(self) -> int:
        return len(self.placebo)

    @classmethod
    def from_rows(cls, rows: Sequence[dict]) -> "Outcomes":
        """Build from a list of `{"placebo": ..., "graded": ..., "oracle": ...}` rows."""
        return cls(
            placebo=tuple(float(r["placebo"]) for r in rows),
            graded=tuple(float(r["graded"]) for r in rows),
            oracle=tuple(float(r["oracle"]) for r in rows),
        )


@dataclass(frozen=True)
class ScreenResult:
    family: str
    n_pairs: int
    room: float
    gain: float
    ratio: float
    saturated: float
    sd_x: float
    sd_y: float
    r_xy: float
    sd_influence: float
    floor: float
    floor_rule: str
    min_pairs: int
    room_low: float
    room_high: float
    ratio_low: float
    ratio_high: float
    candidates_screened: int
    selection_note: str
    floor_binds: bool
    min_room: float
    min_ratio: float
    room_pass: bool
    ratio_pass: bool
    verdict: bool
    reasons: tuple[str, ...] = ()

    @property
    def registered(self) -> bool:
        """Whether the bars this was judged against are the registered ones."""
        return (
            self.min_room == REGISTERED_MIN_ROOM
            and self.min_ratio == REGISTERED_MIN_RATIO
            and self.min_pairs == REGISTERED_MIN_PAIRS
        )

    def lines(self) -> list[str]:
        out = [
            f"family                 {self.family}",
            f"pairs                  {self.n_pairs}   (needs {self.min_pairs})",
            "",
            f"ROOM   oracle - placebo   {self.room:8.4f}   (needs {self.min_room:.4f}, "
            f"interval {self.room_low:.4f} to {self.room_high:.4f})",
            f"GAIN   graded - placebo   {self.gain:8.4f}",
            f"RATIO  gain / room        {self.ratio:8.4f}   (needs {self.min_ratio:.4f}, "
            f"interval {self.ratio_low:.4f} to {self.ratio_high:.4f})",
            f"candidates screened       {self.candidates_screened:8d}",
            f"pairs already at ceiling  {self.saturated:8.4f}",
            "",
            f"per-pair SD of the gain   {self.sd_x:8.4f}",
            f"per-pair SD of the room   {self.sd_y:8.4f}",
            f"their correlation         {self.r_xy:8.4f}",
            f"influence SD              {self.sd_influence:8.4f}",
            f"denominator floor         {self.floor:8.4f}   "
            f"({self.floor_rule}, {'binds' if self.floor_binds else 'clear'})",
            "",
            f"ROOM   {'PASS' if self.room_pass else 'FAIL'}",
            f"RATIO  {'PASS' if self.ratio_pass else 'FAIL'}",
            f"VERDICT                {'ADMITTED' if self.verdict else 'REJECTED'}",
            "BARS                   %s"
            % ("registered" if self.registered else "OVERRIDDEN, not the registered set"),
        ]
        for r in self.reasons:
            out.append("   - " + r)
        return out


def canonical_order(outcomes: Outcomes) -> Outcomes:
    """The same sample, in one order, so a verdict is a fact about the sample.

    THE BOOTSTRAP ADDRESSES POSITIONS. Its resampling indices come from a fixed seed,
    so two serializations of ONE multiset of paired observations draw different
    resamples and can report different intervals, and the room interval's lower bound
    decides dealability. That makes the verdict a fact about the order rows happened
    to be written in. Sorting the whole triples first removes it: the multiset is the
    input, and a permutation of the file is the same input.

    Whole triples, never the three columns separately, because a pair is one A
    execution and its three B branches and pulling the columns apart would invent
    pairings the pilot never ran.
    """
    ordered = sorted(zip(outcomes.placebo, outcomes.graded, outcomes.oracle))
    return Outcomes(
        placebo=tuple(row[0] for row in ordered),
        graded=tuple(row[1] for row in ordered),
        oracle=tuple(row[2] for row in ordered),
    )


def contrasts(outcomes: Outcomes) -> tuple[np.ndarray, np.ndarray]:
    """The per-pair gain and room, x = graded - placebo and y = oracle - placebo."""
    placebo = np.asarray(outcomes.placebo, dtype=float)
    return (
        np.asarray(outcomes.graded, dtype=float) - placebo,
        np.asarray(outcomes.oracle, dtype=float) - placebo,
    )


def floored_ratio(num: float, den: float, floor: float, rule: str = "drop") -> float:
    """The extraction ratio under one of the three floor rules."""
    if rule == "none":
        return float("nan") if abs(den) < 1e-9 else num / den
    if rule == "clamp":
        clamped = max(den, floor)
        return float("nan") if abs(clamped) < 1e-9 else num / clamped
    if rule == "drop":
        if den < floor or abs(den) < 1e-9:
            return float("nan")
        return num / den
    raise ValueError(f"unknown floor rule {rule!r}")


def sd_influence(s_x: float, s_y: float, r_xy: float, ratio: float, room: float) -> float:
    """Per-pair SD of the ratio's influence value, (x - rho y) / mu_y.

    The covariance term is zero when either contrast is constant, rather than the NaN
    a correlation of a constant is. A deterministic sample has no spread in its
    influence value, and reporting that as a non-number where a standard deviation is
    promised is a diagnostic nobody can read: zero uncertainty is the answer.
    """
    if room <= 0.0 or math.isnan(ratio):
        return float("nan")
    # NaN is exactly what a correlation is when one contrast does not vary, which
    # is the case whose covariance is zero. Tying the two here means the guard
    # cannot drift from the condition that produced the NaN.
    covariance = 0.0 if math.isnan(r_xy) else r_xy * s_x * s_y
    v = s_x**2 + ratio**2 * s_y**2 - 2.0 * ratio * covariance
    return math.sqrt(max(v, 0.0)) / room


def _interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    """A bootstrap interval for the mean of one contrast, over the pairs.

    Seeded from the sample rule so a record is reproducible from what it stores.
    """
    n = len(values)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))].mean(axis=1)
    tail = (1.0 - BOOTSTRAP_MASS) / 2.0
    return float(np.quantile(draws, tail)), float(np.quantile(draws, 1.0 - tail))


def _ratio_interval(
    gain: np.ndarray, room: np.ndarray, floor: float, rule: str, seed: int
) -> tuple[float, float]:
    """The same interval for the extraction ratio, resampling pairs together.

    Pairs are resampled as pairs, not each contrast separately, because the two share
    a placebo execution and pulling them apart would price a correlation that is not
    there.
    """
    n = len(gain)
    if n < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed + 1)
    picks = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    ratios = []
    for row in picks:
        try:
            value = floored_ratio(float(gain[row].mean()), float(room[row].mean()), floor, rule)
        except ZeroDivisionError:
            value = float("nan")
        if math.isfinite(value):
            ratios.append(value)
    if not ratios:
        return float("nan"), float("nan")
    tail = (1.0 - BOOTSTRAP_MASS) / 2.0
    return float(np.quantile(ratios, tail)), float(np.quantile(ratios, 1.0 - tail))


def screen(
    family: str,
    outcomes: Outcomes,
    *,
    min_room: float = REGISTERED_MIN_ROOM,
    min_ratio: float = REGISTERED_MIN_RATIO,
    min_pairs: int = REGISTERED_MIN_PAIRS,
    candidates_screened: int = 1,
    selection_note: str = "",
    floor: float = 0.0,
    floor_rule: str = "drop",
) -> ScreenResult:
    """Score one family's screen data against the bars a caller supplies.

    `min_pairs` is REGISTERED and defaults to it. A caller may lower it for a
    diagnostic run, and the result says which bars it was judged against, because a
    screen is the only evidence that a family's room can be converted at all and a
    caller free to choose the sample can choose the one that passes.

    UNCERTAINTY IS PART OF THE DECISION. Room and ratio are reported with a bootstrap
    interval over the pairs, and the rule is that the interval's lower bound on room
    has to clear zero: a point estimate above the bar on a sample whose interval
    straddles no room at all is a number, not a result.

    SELECTION IS DECLARED AND NOT ADJUSTED FOR. A record that screened more than one
    candidate has to say so and say what was done about it, and a record that does not
    is refused. That is the entire effect. The interval, the bars and the verdict are
    the same for one candidate and for a thousand, so a family selected as the best of
    many clears this stage on exactly the arithmetic one candidate clears it on, and
    the disclosure is audit text a reader has to price themselves. The registered
    adjustment is an open maintainer call and no adjustment is implemented here.

    `min_room` and `min_ratio` default to the REGISTERED bars. A caller may move them,
    and a record says which values it was judged against, so what a family cleared is
    always the number beside it rather than whatever the reader assumes.
    """
    if floor_rule not in FLOOR_RULES:
        raise ValueError(f"unknown floor rule {floor_rule!r}")
    if min_pairs < 2:
        raise ValueError(
            f"min_pairs is {min_pairs!r}; a screen taken on fewer than two pairs has no "
            "spread to report and is a point, not a result"
        )
    if candidates_screened < 1:
        raise ValueError("a screen was run on at least one candidate")
    for name, value in (
        ("min_room", min_room), ("min_ratio", min_ratio), ("floor", floor)
    ):
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(
                f"{name} is {value!r}; a screen threshold is a finite number between "
                "0 and 1, and a bar of minus infinity is not a bar"
            )
    # One order for the sample before anything positional touches it, so the verdict
    # is a fact about the observations rather than about how the file was written.
    x, y = contrasts(canonical_order(outcomes))
    room = float(y.mean())
    gain = float(x.mean())
    ratio = floored_ratio(gain, room, floor, floor_rule)
    ddof = 1 if len(x) > 1 else 0
    s_x = float(x.std(ddof=ddof))
    s_y = float(y.std(ddof=ddof))
    if s_x > 0.0 and s_y > 0.0:
        r_xy = float(np.corrcoef(x, y)[0, 1])
    else:
        r_xy = float("nan")

    room_low, room_high = _interval(y, min_pairs)
    ratio_low, ratio_high = _ratio_interval(x, y, floor, floor_rule, min_pairs)

    reasons: list[str] = []
    enough = outcomes.n_pairs >= min_pairs
    if not enough:
        reasons.append(
            f"the screen was taken on {outcomes.n_pairs} pairs against a required "
            f"{min_pairs}"
        )
    certain = bool(math.isfinite(room_low) and room_low > 0.0)
    if not certain:
        reasons.append(
            f"the room interval reaches {room_low:.4f}, so this sample does not "
            "establish that there was any room at all"
        )
    declared = bool(candidates_screened == 1 or selection_note)
    if not declared:
        reasons.append(
            f"{candidates_screened} candidates were screened and the record says nothing "
            "about the selection, so the best of several is being read as one"
        )
    room_pass = at_least(room, min_room)
    if not room_pass:
        reasons.append(
            f"the oracle beats the placebo by only {room:.4f}, so there was nothing for a "
            "receipt to carry on this family"
        )
    ratio_pass = bool(not math.isnan(ratio) and at_least(ratio, min_ratio))
    if math.isnan(ratio):
        reasons.append(
            f"the pooled room {room:.4f} sits under the denominator floor {floor:.4f}, so the "
            "extraction ratio is not a number on this family"
        )
    elif not ratio_pass:
        reasons.append(
            f"one graded receipt took {ratio:.4f} of the room the oracle had"
        )
    if declared and candidates_screened > 1:
        # Not a failure and not an adjustment. Last, so it never crowds a real reason
        # out of the two a bundle prints, and on the result so a reader is told that
        # a selected winner cleared the same arithmetic one candidate clears.
        reasons.append(
            f"{candidates_screened} candidates were screened and this verdict carries "
            "no selection adjustment, because none is registered"
        )

    return ScreenResult(
        family=family,
        n_pairs=outcomes.n_pairs,
        room=room,
        gain=gain,
        ratio=ratio,
        saturated=float((y <= 0.0).mean()),
        sd_x=s_x,
        sd_y=s_y,
        r_xy=r_xy,
        sd_influence=sd_influence(s_x, s_y, r_xy, ratio, room),
        floor=floor,
        floor_rule=floor_rule,
        min_pairs=min_pairs,
        room_low=room_low,
        room_high=room_high,
        ratio_low=ratio_low,
        ratio_high=ratio_high,
        candidates_screened=candidates_screened,
        selection_note=selection_note,
        floor_binds=bool(floor_rule != "none" and room < floor),
        min_room=min_room,
        min_ratio=min_ratio,
        room_pass=room_pass,
        ratio_pass=ratio_pass,
        verdict=bool(room_pass and ratio_pass and enough and certain and declared),
        reasons=tuple(reasons),
    )


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_MASS",
    "DECISION_FIELDS",
    "FLOOR_RULES",
    "PAIR_FIELDS",
    "REGISTERED_MIN_PAIRS",
    "REGISTERED_MIN_RATIO",
    "REGISTERED_MIN_ROOM",
    "RUN_FIELDS",
    "Outcomes",
    "PairRecord",
    "ScreenRecord",
    "ScreenResult",
    "ScreenRun",
    "COMPARISON_RESOLUTION",
    "REGISTERED_MAX_CANDIDATES",
    "at_least",
    "canonical_order",
    "contrasts",
    "floored_ratio",
    "read_payload",
    "screen",
    "sd_influence",
]
