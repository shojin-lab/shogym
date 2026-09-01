# `shogym.receipts`: can this task source carry a measurement

A graded receipt is worth measuring only if it can tell an agent something it could not have
looked up. This library answers that, before anything is run.

It is a library and not part of the [`receipts_v1`](../envs/receipts/README.md) environment
because the question is about **any** task source. A rented world, a scenario pair someone
else authored, a receipt from a benchmark this repo does not own: all of them can be handed to
`gate`, as long as their readouts can be enumerated. The environment is one caller.

## What admission is, and what these checks are for

A generator is first-party code. A family becomes admissible by passing gates R, S and H
and the named checks on every instance of its bank, by a passing one-receipt room screen on
an independent sample, and by a person reading a frozen pack of its rendered instances. The
environment freezes those three into one addressed bundle and, before it serves anything,
recomputes everything about them that is mechanically checkable: the population, the instance
digests, the thresholds, the code pin, the screen's arithmetic and its family, and the review
pack's coverage and bindings. It does not recompute the pilot that produced the screen's
scores or the reading that produced the pack; those are attested, by whoever ran the pilot and
whoever signed the pack. This library is the half that decides what passing means.

**Once a family is admitted, its scorer defines that family's ground truth**: what `score`
returns is what the chain seals, and these gates measure the receipt that scorer produces
rather than adjudicating it.

The checks exist to catch **mistakes**. A readout that turns out to be injective, a receipt
that prints a correction nobody meant to license, a placebo whose committed tokens drifted, a
rule the oracle states differently from the one that was drawn: these are errors an author
makes and cannot see, and the package finds them by rendering every convention and reading the
bytes. They are not a defence against a generator written to deceive. The human read is what
stands there.

Two habits follow and are kept anyway: everything a check compares against is built by this
package before any family code runs and family code is handed read-only views, because a
comparison built after a callback compares against whatever the callback left behind; and a
family that raises is a failed check rather than a crashed report.

## The failure this exists to catch

A receipt is graded against a hidden convention: named axes, each holding an interchangeable
option set, one option drawn per axis. The distinction that decides everything is between a
graded field that **stores** the convention and one the convention **computes**.

If the scorer checks the option the agent picked, the verdict can only say right or wrong. It
can never say which of the wrong options should have been picked instead, so the posterior
after a FAIL is uniform on the `c - 1` survivors, compliance is exactly `2/c`, and the lift is
exactly `1/c`. That value depends on the arity and on nothing else. **Twenty copies of such an
item carry exactly what one copy carries.**

A receipt like that produces a large, tight, statistically significant effect that is also a
**constant**. An entropy pre-check approves it. A run built on it draws a flat line with a
narrow interval, at every checkpoint, for every agent, forever.

## The three gates

```python
from shogym.receipts import gate
result = gate(design, min_arity=3, min_blocks=2.0, min_headroom=0.0)
print("\n".join(result.lines()))
```

**R, resolution.** For axis `k`, hold every other axis at the agent's own applied convention
and vary `k` over its options. Each option produces a rendered receipt, and options whose
receipts print the same thing are options the agent can never tell apart, so the number of
distinct printed receipts is what the axis resolves. An instance fails when **every axis of
three or more options** sits at two blocks or fewer. Binary axes are outside R: two blocks is
full resolution for them. Exact, fully automated, and computed per instance with no averaging.

The observation is what the receipt **prints**, not the scorer's verdict bits. A gate reading
the scorer would be reading the renderer's intentions: a receipt that prints a correction on a
failed row says far more than a bit, and one that reorders rows says something the scorer
never computed.

Two corollaries, both tested:

- an **injective** readout pins the axis at two blocks for any number of items;
- a **single reported item** can produce at most two signatures, so an axis carried by one
  item fails R however rich its readout is.

**S, non-self-interpretation.** The map from a printed symbol to its constraint must not be
printed by the receipt. `date selection: FAIL` states its constraint; `RQ-1042: FAIL` says
nothing until you work out what RQ-1042 reads. Five checks, all exact, on the declared labels AND on the
serialized bytes: no row labelled by its axis; the resolution the evident rows alone reach must
not already equal the resolution the whole receipt reaches; no axis name in the printed
vocabulary; no option token that is not also a legitimate answer; and a printed row order that
does not move with the convention. Which rows count as evident is derived from the receipt
rather than declared: a row is evident when it responds to exactly one axis and prints a
distinct thing for every option of it.

`gate` says in its own output that S is **only structurally checked**. What it does not catch
needs the rendered receipt read by a human: item ids that encode the constraint in a way no
listed token matches, a legend that prints the mapping in prose, prose in the task text that
gives it away, and an axis label so opaque nobody could use it.

**H, room above lookup.** The ceiling must stand above the **lookup floor**, and both are
optimized over the **sibling task's legal action space**: one answer per row, so scoring is
row additive and the best action is the per-row posterior mode. Committing to a whole
convention key is strictly worse whenever two conventions in a class agree on some rows and
differ on others, which is the ordinary case, and pricing a design with it understates the
floor by more than the ceiling and so reports headroom that is not there.

A ceiling on its own is not a gate. A named-slot receipt's ceiling
looks like ample room above a placebo, and every point of it is reachable by a rule that costs
no induction: keep your option on PASS, draw uniformly from the rest on FAIL. Priced against
the placebo it looks alive; priced against what it concedes for free its headroom is exactly
zero, because the floor and the ceiling are literally the same partition there.

Which rows are **evident** is derived from the receipt, never declared: a row is evident when
it responds to exactly one axis and prints a distinct thing for every option of it, so reading
the option off it costs no induction. Taking an author's word for it would let a design set
its own floor. The derivation is deliberately generous, which raises the floor and lowers
headroom, so the gate under-reports what a design has rather than over-reporting it.

## Every score is a rowwise Bayes-action score

A reader is two things: a partition, which says what it observes, and an action rule, which
says what it does with the uniform posterior on the observed class. Everything here fixes the
second and varies only the first.

    score(class) = mean over rows r of  max over answers a of  Pr( key_B[r] = a | class )

The sampling score, the same reader drawing from its posterior per row instead of taking the
mode, is computed and reported beside it (`GateResult.sampling`), because the gap is a
property of the receipt worth seeing. The draw is never better and is strictly worse on any
row whose posterior is not flat.

**Headroom is always a difference of two scores under the same rule.** Mixing them would price
the action rule rather than the receipt. The placebo does not move under either: an ungraded
arm has no posterior to act on.

## The room screen

The gates are enumerable at zero execution cost and say whether a receipt **can** carry a
measurement. `screen` is empirical and says what one receipt **did** carry. Per task pair: one
execution of task A and three of task B, a graded branch, a byte-matched placebo branch, and
an oracle branch that is told the rule, all at zero prior dose.

```python
from shogym.receipts import Outcomes, screen
result = screen("ledger", Outcomes.from_rows(rows))   # the registered bars
```

    room  = mean(oracle - placebo)
    gain  = mean(graded - placebo)
    ratio = gain / room

The ratio is a ratio of two **aggregated** differences, never a mean of per-pair ratios. A
pooled denominator below `floor` is not turned into a ratio at all, under one of three rules
(`drop`, `clamp`, `none`), because a ratio whose denominator is noise around zero is not a
number. `sd_influence` gives the per-pair SD the delta method puts on the ratio scale, which is
what sets a budget.

`saturated`, the fraction of pairs the oracle could not improve, is the number that kills a
family quietly: a receipt can pass every gate and be worth nothing on a task the agent already
solves.

**The bars are registered.** What a family must show to earn a roster place is the
maintainer's call, and the call is `min_room = 0.05` with the bootstrap interval's lower bound
above zero, `min_ratio = 0.25`, and 36 DISTINCT tasks. The first two are defaults rather than
constants, so a diagnostic run can ask what a family does against another bar; a result says
whether the bars it was judged against were the registered ones. A caller that deals families
requires the registered ones: recording that a bar was moved is not refusing to deal a family
admitted under an easier rule, and `receipts_v1` refuses one.

A pair is one task, not one observation of one. `ScreenRun` requires distinct task seeds and
distinct task instances, because repeated filings against a single table clear the sample floor
while the pilot has one sampled unit, and the pair bootstrap would price them as independent
draws.

## Provenance

Gates R, S and H are ported from the instrument that measured them. Gate C, its fourth and
later gate, is not carried, and the set is named `receipts-gates-v2` so nothing here claims
the instrument's own verdict.

Two things were deliberately changed in the port, and both change numbers:

- the observation is the **serialized receipt**, not the scorer's verdict bits, so a receipt
  that prints a correction resolves more than the instrument's model of it would;
- the action rule is **rowwise** over the sibling task's legal actions, not commitment to a
  whole convention key, so ceilings and floors both rise.

What survives unchanged is the arithmetic the instrument's hand-checkable vectors pin, and
that is not a coincidence: a verbatim slot receipt has independent rows, so answering row by
row and committing to a key are the same act on it. `tests/test_receipts_gates.py` runs those
vectors as generators rather than hand-built matrices, through the shipped `draw`, `observe`
and `gate` the CLI itself calls, so the shipped code is what is being checked: named-slot receipts at two blocks per
axis with ceiling `2/c`, placebo `1/c` and headroom exactly zero; crossed merges resolving
three; one row capped at two signatures; a binary axis outside R; twenty copies of a row
carrying exactly what one carries; and the correlated affine sampler REJECTED, with its
support size pinned directly rather than inferred by comparison.

Because the two implementations now price different objectives on purpose, there is no
cross-implementation golden file and no claim of one. A drift from the instrument shows up as
a vector whose closed form no longer holds.

## API

| Name | What it is |
|---|---|
| `AxisSpace` | the convention space: named axes with option sets, enumerated in a fixed order |
| `Observation` | one instance, as the gates see it: what the receipt prints under every convention |
| `gate` | R, S and H on one instance, with the thresholds as arguments |
| `GateResult` | the verdict, the numbers behind it, its reasons and its caveats |
| `resolution_blocks`, `axis_receipts` | gate R's two primitives, usable on their own |
| `row_dependence`, `evident_rows` | the derived evidence the floor is built from |
| `rowwise_scores`, `score_partition` | the action space, usable on their own |
| `bits` | the entropy reading, a diagnostic and never a gate |
| `Outcomes`, `screen`, `ScreenResult` | the room screen |
| `sd_influence`, `floored_ratio`, `contrasts` | the screen's arithmetic, usable on their own |
