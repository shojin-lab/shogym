"""Admission: the gates and the named checks, run together over one instance.

The gate set is a NAMED chain-specific version. It implements R, S and H from the
instrument and deliberately excludes the instrument's later count gate, because the
channel that gate closes is a paid mechanism in this design rather than a defect. The
name is what this publishes its verdicts under, so nothing here can be mistaken for
the instrument's own.

An instance is admissible when every gate and every check passes on it. A bank holds
the instances this admits, in ordinal order, and which ones those are is recomputed by
rerunning this rather than read from a list.

THE BARS ARE REGISTERED and are the defaults here. A run may move them for diagnosis,
and a report carries the ones it judged against, but a bundle is verified against the
registered set exactly: a family admitted under an easier rule was not admitted under
the rule the measurement is registered under.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Callable

from shogym.envs.receipts.checks import CheckResult, run_checks
from shogym.envs.receipts.observe import observe
from shogym.envs.receipts.protocol import Generator, Instance
from shogym.receipts import GateResult, gate

#: The name of this gate set and the rule it publishes under. Recorded in every bank;
#: nothing claims the instrument's own verdict.
#:
#: v2 BECAUSE THE COPY REGISTRATION CHANGED. v1 read its copy bar against three
#: enumerated maps at 0.40; this reads it against the closure of those maps under
#: composition at 0.50. Two different rules under one label is exactly what a named
#: version exists to prevent, so a family admitted under the earlier rule does not
#: publish under this name and a bundle frozen under the earlier bars does not verify.
GATE_VERSION = "receipts-gates-v2"
#: What that name means, frozen. R's arity and block constants are the settled rule,
#: not a dial: two blocks is where the agent learns only that it was wrong, and three
#: options is where an axis can resolve past it at all. A run that moved them and
#: still reported this name would publish two different rules under one label.
CUSTOM_VERSION = "receipts-gates-custom"
SETTLED_MIN_ARITY = 3
SETTLED_MIN_BLOCKS = 2

#: The registered admission bars. These are the numbers a family is admitted under,
#: and a bundle is verified against exactly these; a caller may move them for a
#: diagnostic run, and the report says which it judged against.
#:
#: They sit where the ledger's own distribution makes them bite without emptying the
#: bank. Its one-axis-wrong score is never below five sixths, so a flip bar under that
#: admits nothing; 0.875 rejects the worst draws and keeps the rest. Its weakest axis
#: leverage never exceeds one sixth, so a leverage bar above that admits nothing;
#: 0.10 leaves room under the ceiling. Headroom above 0.05 cuts the thin ones.
#:
#: THE COPY BAR IS 0.50, AND IT IS READ AGAINST THE CLOSED FAMILY. A bar and the maps
#: it is a maximum over are ONE registration: a screen that enumerates more maps
#: reports a larger maximum whether or not the extra maps are a real channel, so a
#: number measured against a narrower family means nothing held against a wider one.
#: The family is the closure of the registered maps under composition, generators and
#: all compositions of them (`checks.NO_INDUCTION_MAPS`), and 0.50 is calibrated
#: against that closure: over sixty ledger draws under one key the closed maximum runs
#: 0.4167 to 0.5833 with a median of 0.5000, and 50 of the 60 clear 0.50, so the bar
#: keeps most of the bank and refuses the worst copies. Held against the same closure
#: a bar of 0.40 clears none of the 60, which is what a bar calibrated against a
#: narrower enumeration is worth once the family it maximizes over is the real one.
REGISTERED_MIN_HEADROOM = 0.05
REGISTERED_MAX_COPY_SCORE = 0.50
REGISTERED_MAX_FLIP_SCORE = 0.875
REGISTERED_MIN_LEVERAGE = 0.10


@dataclass(frozen=True)
class Thresholds:
    """What an instance has to clear. Every bar defaults to its registered value.

    A caller may move any of them for a diagnostic run, and `registered` says whether
    a set is still the one a bundle may be dealt under. The defaults are the whole
    registration: a bar with no default would be a bar whose value came from whoever
    ran the command.
    """

    max_copy_score: float = REGISTERED_MAX_COPY_SCORE
    max_flip_score: float = REGISTERED_MAX_FLIP_SCORE
    min_leverage: float = REGISTERED_MIN_LEVERAGE
    min_arity: int = SETTLED_MIN_ARITY
    min_blocks: int = SETTLED_MIN_BLOCKS
    min_headroom: float = REGISTERED_MIN_HEADROOM
    min_material_rows: int = 1

    @property
    def registered(self) -> bool:
        """Whether these are the registered bars rather than an override."""
        return self.as_record() == Thresholds().as_record()

    def __post_init__(self) -> None:
        for name in ("max_copy_score", "max_flip_score", "min_leverage", "min_headroom"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} is {value!r}; a score threshold has to be a finite number "
                    "between 0 and 1, and a comparison against NaN is silently false"
                )
        if self.min_arity < 2 or self.min_blocks < 1 or self.min_material_rows < 1:
            raise ValueError("the gate constants have to be positive")

    @property
    def moved_gate_constants(self) -> tuple[str, ...]:
        """Which of the gate set's own constants are not the registered ones.

        R's arity and blocks AND H's headroom. All three are part of what the name
        `receipts-gates-v2` means, so a run that moved any of them and still published
        that name would put two different rules under one label. The copy, flip,
        leverage and material-row bars are the named checks rather than the gates, and
        `registered` is what covers those.
        """
        return tuple(
            name
            for name, value, settled in (
                ("min_arity", self.min_arity, SETTLED_MIN_ARITY),
                ("min_blocks", self.min_blocks, SETTLED_MIN_BLOCKS),
                ("min_headroom", self.min_headroom, REGISTERED_MIN_HEADROOM),
            )
            if value != settled
        )

    @property
    def gate_version(self) -> str:
        """The name this configuration may publish its results under."""
        return CUSTOM_VERSION if self.moved_gate_constants else GATE_VERSION

    @property
    def settled(self) -> bool:
        return self.gate_version == GATE_VERSION

    def as_record(self) -> dict[str, float]:
        return {
            "max_copy_score": self.max_copy_score,
            "max_flip_score": self.max_flip_score,
            "min_leverage": self.min_leverage,
            "min_arity": float(self.min_arity),
            "min_blocks": float(self.min_blocks),
            "min_headroom": self.min_headroom,
            "min_material_rows": float(self.min_material_rows),
        }


@dataclass(frozen=True)
class Report:
    """One instance's whole admission report, and the bars it was judged against."""

    tag: str
    gates: GateResult
    checks: tuple[CheckResult, ...]
    thresholds: "Thresholds" = None  # type: ignore[assignment]

    def digest(self) -> str:
        """The content hash of this instance's whole admission report.

        One value standing for the whole verdict, for a caller comparing two runs of
        this over one instance.
        """
        from shogym.envs.receipts import streams

        payload = {
            "tag": self.tag,
            # The name THIS report may publish under, not the settled one. A run that
            # moved R's constants publishes under the custom name, and stamping the
            # settled one here would put two different rules under one label in the
            # hashed payload as well as in the printout.
            "gate_version": self.thresholds.gate_version,
            "thresholds": {k: round(v, 12) for k, v in sorted(
                self.thresholds.as_record().items()
            )},
            "verdict": self.gates.verdict,
            "r_pass": self.gates.r_pass,
            "r_axes": list(self.gates.r_axes),
            "s_pass": self.gates.s_pass,
            "h_pass": self.gates.h_pass,
            "blocks": {k: int(v) for k, v in sorted(self.gates.blocks.items())},
            "ceiling": round(self.gates.ceiling, 12),
            "floor": round(self.gates.floor, 12),
            "checks": [[c.name, c.passed] for c in self.checks],
        }
        return streams.digest(json.dumps(payload, sort_keys=True).encode())

    @property
    def failed_checks(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.checks if not c.passed)

    @property
    def admitted(self) -> bool:
        return self.gates.verdict and not self.failed_checks

    def lines(self) -> list[str]:
        out = list(self.gates.lines())
        out += ["", "NAMED CHECKS"]
        out += ["   " + c.line() for c in self.checks]
        out += [
            "",
            "ADMISSION              %s" % ("ADMITTED" if self.admitted else "EXCLUDED"),
        ]
        return out


def report(
    generator: Generator,
    instance: Instance,
    master: bytes,
    thresholds: Thresholds,
    side: str = "a",
) -> Report:
    """Gate and check one instance, and say whether it is admissible."""
    try:
        result = gate(
            observe(generator, instance, side),
            min_arity=thresholds.min_arity,
            min_blocks=thresholds.min_blocks,
            min_headroom=thresholds.min_headroom,
        )
    except Exception as exc:  # noqa: BLE001 - a generator that raises has not passed
        return Report(
            tag=f"{instance.generator}/{instance.ordinal}/{side.upper()}",
            gates=_refused(generator, exc),
            checks=(
                CheckResult("gates", False, f"the generator raised: {type(exc).__name__}"),
            ),
            thresholds=thresholds,
        )
    checks = run_checks(
        generator,
        instance,
        master,
        max_copy_score=thresholds.max_copy_score,
        max_flip_score=thresholds.max_flip_score,
        min_leverage=thresholds.min_leverage,
        min_material_rows=thresholds.min_material_rows,
    )
    return Report(
        tag=result.tag, gates=result, checks=tuple(checks), thresholds=thresholds
    )


def _refused(generator: Generator, exc: Exception) -> GateResult:
    """A gate report for a generator that could not be observed at all."""
    arity = {axis.name: len(axis.options) for axis in generator.AXES}
    return GateResult(
        tag=generator.name,
        arity=arity,
        blocks={name: 0 for name in arity},
        dependence=(),
        n_rows=0,
        n_evident=0,
        placebo=0.0,
        ceiling=0.0,
        floor=0.0,
        sampling={},
        r_pass=False,
        r_axes=[],
        s_pass=False,
        s_structural="not evaluable",
        s_label_resolution_equal=False,
        s_leaks=[],
        s_order_moves=False,
        h_pass=False,
        min_headroom=0.0,
        verdict=False,
        reasons=[f"the generator raised while being observed: {type(exc).__name__}: {exc}"],
    )


def admitter(
    generator: Generator, master: bytes, thresholds: Thresholds
) -> Callable[[Instance], bool]:
    """The predicate a bank build filters with."""

    def admit(instance: Instance) -> bool:
        return report(generator, instance, master, thresholds).admitted

    return admit


__all__ = [
    "CUSTOM_VERSION",
    "GATE_VERSION",
    "REGISTERED_MAX_COPY_SCORE",
    "REGISTERED_MAX_FLIP_SCORE",
    "REGISTERED_MIN_HEADROOM",
    "REGISTERED_MIN_LEVERAGE",
    "SETTLED_MIN_ARITY",
    "SETTLED_MIN_BLOCKS",
    "Report",
    "Thresholds",
    "admitter",
    "report",
]
