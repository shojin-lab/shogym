# `receipts_v1`: sibling tasks under a hidden convention

An experiment wants to know what one graded receipt is worth: give an agent a task, grade
what it filed, hand it back a receipt, and see whether the next task goes better. Nothing in
that sentence is measurable unless the two tasks turn on the same hidden decision, unless the
receipt says something the agent could not have looked up, and unless the arm that gets no
receipt reads bytes of exactly the same shape as the arm that does.

`receipts_v1` serves tasks from a **family**: one generator plus one draw of its hidden
convention. The generator emits sibling tasks **A** and **B** over different surface data,
both scored under that same drawn convention, so the family relation holds **by
construction** rather than by assertion.

Read that claim exactly. What is mechanical is that one convention object, frozen before
either sibling is built, computes both answer keys. What an option MEANS on each surface is
the generator's own code, and nothing here can check that `event` reads the same way on a
claims table and a library table: admitted generators are trusted first-party code, and the
human read of the review pack is the control against one that does not.

No extra dependencies. `receipts_v1` runs on core shogym.

## Start here: read a drawn instance

```bash
shogym receipts materialize ledger --size 16   # freeze a bank of gate passers
shogym receipts draw ledger                    # print one, for reading
```

`draw` prints both sibling task texts, the drawn convention, and the three cells a fork could
serve, rendered through the same atomic path a run uses. A generator enters a release only
after a human has read instances of it, and the review pack is many of these, not one.

There is no `--seed`. A free seed makes the gate universe and the review cherry-pickable, and
a live run must never serve a draw from outside the bank, so the only thing that can be drawn
is an ordinal the bank already holds.

## The trust boundary

The build side and the run side are separate trust domains. Everything that could reproduce
the hidden convention stays **controller-side**: generator code, the sampler's live draws, the
drawn convention, the answer key, the oracle renderer, the bank, the bundle, and this CLI. What crosses to
the agent is **rendered bytes** and an **opaque task identifier**.

Identifiers are HMACs under a controller-side master key, so they encode no seed, no family
index and no draw ordinal. An agent that collects every identifier it was ever served learns
how many there were and nothing else.

Every draw comes from a keyed, domain-separated stream. Surface A, surface B, the convention
and the filler are four independent streams under one key, and there is no ambient PRNG.

## What a family is

A generator implements:

| Part | What it is |
|---|---|
| `SHAPE` | the table's columns and how their values are invented |
| `AXES` | the hidden decisions, each a named axis with an interchangeable option set |
| `parse_and_canonicalize` | the mechanical reading of what the agent filed, returning a reason-coded `NoFiling` rather than raising |
| `score` | the sealed `[0, 1]` scalar and the per-row outcomes; pure |
| `render_receipt` / `render_placebo` / `render_oracle` | the three cells, each a `ReceiptAST` |
| `parse_oracle` | reads a convention back out of an oracle cell |
| `describe` | the instruction the agent sees, leaving every axis undetermined |

`NoFiling` is what keeps "answered badly" mechanically distinct from "did not answer". An
exception would erase the difference, and the chain's failure taxonomy depends on it.

The **same canonical filing** feeds the scorer and both renderers, so a receipt can never
grade something the score did not.

## The receipt is a structure, not text

A renderer returns a `ReceiptAST`. One serializer, shared by every family, turns an AST into
bytes. The renderer does not choose its own layout.

The gates ask what a receipt can tell an agent, and that is a property of what the agent
actually reads. A renderer that returned text could tell the gates one thing and the agent
another: it could order rows informatively, pad one class of row differently from another, or
print an identifier that correlates with the drawn option, and a gate reading the renderer's
declared intentions would see none of it.

## The envelope: three cells, one shape

The three cells a fork can serve are read into a context, so they share one envelope: a single
registered size that does not depend on the drawn convention, reached by padding with a
committed filler stream.

- **graded** and **placebo** are structurally congruent. Identical wrapper, identifiers,
  order, offsets, whitespace, column headers and padding. They differ **only** inside
  registered fixed-width slots: `verdict` (4 bytes) and `correction` (12 bytes).
- And on some rows they do not differ at all. A family declares a **receipt policy**, and
  under one that samples the graded cell reports a verdict and a same-row correction on
  only the rows a committed mask drew. Every other row carries that position's committed
  neutral tokens in both slots, which is exactly what the placebo carries there, so
  outside the drawn rows the two cells are identical inside the slots as well as outside
  them.
- The placebo fills those slots with neutral tokens from the family's registered filler
  alphabet, drawn by the committed stream and fixed before launch. No character of that
  alphabet appears in a verdict token or a band. A placebo may not group, reorder, highlight,
  tutor or analyze: in a chain a placebo child can be the next link's parent, so any
  organizing work it does rides forward as treatment.
- **oracle** shares the size and the outer wrapper but has no rows to align. Its body is the
  registered rule template, padded.

Fixed-width everywhere is what makes this checkable rather than hoped for. Every row line is
the same length, built from registered widths, so a slot occupies a known byte range and the
envelope check masks those ranges and asserts the rest never moves. A layout that stripped
trailing spaces, or sized a column from the longest value in it, would put the answer key into
the byte count.

The envelope check runs at the moment the cells are made and refuses to persist a fork that
fails it. It caught a real break during development: graded and placebo were carrying
different column headers.

## The receipt policy: which rows a receipt reports

A generator declares `RECEIPT_POLICY`, and `registry.load_generator` refuses one that does
not, the way it refuses one that declares no copy profile. Three policies are registered.

| Policy | What the graded cell carries |
|---|---|
| `full` | a verdict and that row's own answer on **every** printed row |
| `sampled-2-of-24` | a verdict and that row's own answer on the **two** rows a committed mask drew, and that position's committed neutral tokens in both slots on the other twenty two |
| `sampled-4-of-24` | a verdict and that row's own answer on the **four** rows a committed mask drew, and that position's committed neutral tokens in both slots on the other twenty |

**The count is registered per generator and the bars are the same for all.** Ledger
declares four records of twenty four, sound change two forms of twenty four, components
two rows of twenty four, and every gate vector the full receipt. How many rows a receipt
has to report before it stops teaching is a property of the family's own tables, not of the shape they share: four
fully reported rows leave an ideal reader at about 0.82 on ledger and at 0.95 on sound
change, because a corrected daughter form exposes the replacement phone and the lost
vowel directly. Two rows of sound change leave 0.849 against a lookup floor of 0.220, and
they are the only count from one to eight whose bank-mean ideal level sits in the
registered band. Components is two for a different reason: a board's count is one of the
two values its own pattern can take, so a verdict alone names the other and any form that
keeps every verdict saturates, while two reported rows leave 329/368 against a lookup
floor of 305/368. A count is chosen from a bank's arithmetic before the bank is looked
at, and a family whose fresh bank fails a bar is not admitted at that count and is not
admitted at a count chosen afterwards.

**The mask is drawn before any filing and independently of the drawn convention**, from a
fifth committed stream keyed by the generator, the ordinal and the sibling label. It joins
the instance commitment beside both answer keys, so a re-render after a crash reports the
same records, and it reaches the renderer as an immutable `render.Feedback` the shared
judge builds from the frozen envelope before any family callback runs. A cell that reports
a row the mask did not draw, or goes quiet on one it did, is refused by name.

**A mask that happens to omit an axis is not redrawn.** What is registered is the law, not
the realized receipt: a mask redrawn until every hidden decision had a witness would be a
mask the convention could be read off, and the arithmetic the gate does averages over the
uniform law rather than over a filtered one.

**The task says so.** Under a policy that samples, the description carries one registered
sentence, after the scope sentence and before the table. `filing.receipt_sentence` holds
when it is printed and that it is the same bytes in every arm and under every convention;
the words are the genre's, because a schedule has records and bands where a batch has
forms and daughter forms:

```
The receipt for this schedule reports the verdict and the correct band for four
selected records only, and the lines for the other records say nothing about
whether they were right.

The receipt for this batch reports the verdict and the correct daughter form for
two selected forms only, and the lines for the other forms say nothing about
whether they were right.

The receipt for this schedule reports the verdict and the correct count for
two selected rows only, and the lines for the other rows say nothing about
whether they were right.
```

Neither names a row: the selection is drawn before any filing exists, a reader can see it
in the receipt anyway, and a reader who could see it in the task could file the rest at
random and lose nothing.

## Rendering order: what exists when

The graded and placebo cells depend on the agent's A filing, which does not exist until the
agent files. Nothing here pretends otherwise.

- **Before launch**, materialized and hashed: every instance (both surfaces, both task texts),
  the convention commitments, the envelope template with its registered slots, the committed
  filler stream, the receipt policy with each side's drawn mask, and the renderer
  configuration.
- **After A seals**, in one act: the parser canonicalizes the filing, the three cells are
  rendered, they are judged, and the blobs are hashed. That is `bank.render_fork`, and it is
  the only place the three cells are made. Every retry replays those blobs; nothing rerenders,
  because rerendering is how two branches of one fork come to differ.

  Once is once under concurrency as well as over time, and that takes a claim: two seals of
  one filing arriving together would both find the record absent and both render it, so the
  fork is claimed on its own name, per fork rather than per store, and the loser reads what
  the winner wrote. Nothing here is durable against an abrupt host failure mid-write: a fork
  that was never published is rendered again on the next attempt, which is the same answer.

**What "judged" means is one function**, `render.judge_cells`, and admission runs it too, at
every sample it takes, over the whole convention space and every registered filing class on
both siblings. There is no second, weaker version: a cell admission accepts is a cell the fork
accepts, because the same function decided both. Otherwise a renderer with a wrapper bug, or
one that drops rows on a partial filing, passes admission and then fails at every seal, or at
a filing-dependent subset of them, which reaches an experiment as branch-specific missing
outcomes on a family the instrument said was usable. Everything it compares against is built
before any renderer runs, because family code can mutate a structure it is handed.

Instances are rebuilt from the controller-side key rather than stored row by row, and every
rebuild is checked against the digest the bundle committed to. A generator edit that would
change one byte of a task text **fails verification** instead of quietly serving a different
task.

## The genre: `ledger`, clerical date counting

A table of records, each with three dates and some with none, and a policy extract that says
to band every record by "days elapsed as at the reference date". The extract is deliberately
incomplete. Four decisions it never makes are the hidden convention:

| Axis | Options | The decision |
|---|---|---|
| `anchor` | `event`, `intake`, `last_action` | which of the three dates the count runs from |
| `basis` | `calendar`, `business`, `business_minus_holidays` | which days count |
| `boundary` | `lower`, `upper` | a count landing exactly on a printed figure |
| `missing` | `lowest`, `highest`, `blank`, `pending` | a record with no dates entered |

72 conventions. Sibling tasks are two different organisations, formats and band vocabularies
(eight domains, four reserved for A and four for B), scored under one draw.

The receipt can carry something because **the scored column holds a value the rule computes**,
a band, not the option the agent chose. Two anchor options that put a record in the same band
are indistinguishable on that record and distinguishable on another, so the verdicts cut the
option set into more than two pieces. `build_table` searches for a table where that actually
happens, and rejects a draw where varying an axis moves too few records.

The filing format is one line per record: the record id, a comma, and the band. The registered
reading rules cover identifier matching, case and whitespace, duplicates (the first line for an
identifier wins), extras, and omissions. One forgiving reading is registered and it is narrow:
a filing with no commas anywhere is read positionally **only** when it has exactly one line per
printed row, so a paragraph of prose cannot be read as an answer to the first rows.

## The genre: `soundchange`, phonological transformation composition

A batch of 24 invented proto forms and the complete public mechanics of a cascade that turns
each one into a daughter form. The mechanics are deliberately incomplete. Four decisions they
never make are the hidden convention:

| Axis | Options | The decision |
|---|---|---|
| `reflex` | `out_g`, `out_x`, `out_s` | which phone a matching k becomes |
| `environment` | `vowel_pair`, `right_vowel` | which neighbours a k needs before it is replaced |
| `loss` | `erase_e`, `erase_i`, `erase_a` | which vowel is deleted between consonants |
| `sequence` | `replace_then_erase`, `erase_then_replace` | which of the two hidden passes runs first |

36 conventions. Sibling tasks are two disjoint batches of invented words under four
presentation templates (`field_csv`, `archive_tsv`, `comparative_csv`, `catalogue_tsv`), scored
under one draw. A public nasal pass runs first, once, and carries no hidden choice.

**Thirty six and not fifty four.** A third environment replacing every k with no condition was
proposed and is not here. Unconditional replacement of one consonant by another commutes with
the deletion of a vowel between consonants on every word, so the two order choices under it
would be the same function on every possible form and no search for better words could separate
them. Keeping them would have advertised a decision no data can reveal.

**What is easy here, said plainly.** The replacement phones g, x and s never occur in a proto
form, and the deleted vowel can be read off the difference in vowel counts, so a reader holding
A's corrected key can name two of the four axes by direct subword correspondence. This family
does not offer four equally demanding inferences. The `phone_lookup` check gives those two away
for free and requires room above what is left, which is the two-way environment composed with
the two-way order.

**A wider correction slot.** The longest proto form is 14 phones and every form carries each of
the three deletable vowels between consonants, so the longest daughter is 13 bytes and the
registered correction slot is **16**, not ledger's 12. Registered widths are 12 / 16 / 4 / 16
with a 1100-byte oracle body allowance, which is a 63-byte row line and a **2757-byte**
envelope.

### The three checks this genre adds

`copy_profiles` gives it the `soundchange_v1` profile, and that profile brings three checks
beyond the eleven every family runs. They live in `generators/soundchange_audit.py`, which is a
second implementation of the cascade by lookaround rather than by scan: `judge_cells` compares
the renderer with the scorer and cannot catch a scorer that is wrong in both.

| Check | What it asks |
|---|---|
| `cascade` | an independently implemented validator recomputes every answer on all 36 draws and both sides, the forms stay inside the declared grammar, and no printed identifier or correction changes under serialize and read back |
| `phone_lookup` | giving the reflex and the deleted vowel away for free, on top of the lookup observations the floor already concedes, leaves the floor at or under 0.90 and more than 0.05 of room, at every reference draw and in both directions, and the same of the receipt this family serves |
| `analogy` | an optimistic reader who transfers A's own edits to every B form with the same skeleton earns at most 0.50 under every draw, in both directions |

**Two of the three read the tables and not the receipt, and the count does not move them.**
`cascade` recomputes the answers and `analogy` transfers the proto forms' skeletons, so both
are construction facts about a pair and ask the same question whatever a receipt reports.
`phone_lookup` reads one side's whole support key, which is what a receipt reporting every row
prints, so under a count that reports two of twenty four it prices a reader the family does not
serve. Its lines say so, and the same two bars are put to the served receipt beside them: over
every mask the registered law can draw, with both readable axes conceded on top of what the
floor already concedes. On sound change's evidence bank at two reported rows that floor is
`0.688284` with `0.160965` of room above it.

The disjoint final-consonant pools give the two batches disjoint answers without either side
reading the other, and they establish **nothing** about copying resistance: the registered
character maps fix them, so the copy screen scores zero on every pair by construction. The
analogy check is what prices a sibling built by renaming the consonants the rule never looks at,
and a release that presented the copy score as evidence by itself would be presenting an
arithmetic fact about the alphabets.

### The model room screen for this genre

The `screen` command analyzes recorded outcomes and runs no model. The procedure, in one place
for whoever runs it:

1. After the mechanical checks and the human read, use the intended harness and resource limits
   to execute one A copy and three B copies per case, from a common sealed A state. Receipt,
   placebo and the registered prose oracle are assigned at the ordinary delivery point. B grades
   are withheld and every artifact reference and transcript is preserved. No executable solver
   is inserted into the oracle treatment.
2. If usability exploration is needed, use the **first 12 admitted cases** as a separately
   labelled diagnostic and freeze the candidate before any admission measurement. Then use the
   **next 36 previously unused admitted cases**, fixed before outcomes are inspected. Without
   exploration, use the first 36.
3. Allocate **nine cases to each of the four frozen initial states**, in their registered order,
   cyclically by case position. That allocation and the model configuration live in run
   provenance: the screen schema has no fields for them.
4. That is **36 A executions plus 108 B executions, 144 in all**, at zero prior receipt dose
   within each process. Every assigned case and the registered failure dispositions are kept; a
   run does not keep only the completed favourable triplets.
5. Record O, P and G by initial state as well as pooled, with rule-reading mistakes, execution
   mistakes, omitted rows, tool failures, context use, time and tokens.
6. Judge it against the registered bars: `min_room=0.05`, `min_ratio=0.25`, `min_pairs=36`,
   `min_oracle=0.90`, `min_learning_gap=0.10`, `floor=0.0`, `floor_rule="drop"`,
   `candidates_screened=1`. The last two of those used to be a roster-release condition beside
   the record; they are fields of it now, and a bundle re-verifies against them.

A pooled mean can conceal a state-specific uptake failure, which is why the by-state report is
part of the procedure rather than an extra. A solved reference implementation proves
attainability and says nothing about model uptake.

### Why a short executable rule, and what that is not an answer to

The rule is three passes over at most 14 phones with at most two k occurrences: no date
arithmetic, no rounding, no lookup. That is a choice about what a copy has to do once it has the
rule, and it is **not** a repair for the low oracle grade an earlier engineering run recorded.
The diagnostic of that run located the loss in **uptake and scope**, not in execution: almost
every filing was the exact answer key of some convention the family draws from, an independent
recomputation agreed with the grader on every row, and no copy skipped a row or ran out of
turns. What went wrong was that copies refused the statement as untrusted content, or read its
scope word as the name printed on the surface and did not carry the rule to the task they were
about to file. Those are repairs to registered wording and they were made. A short rule here
buys a cleaner separation between a copy that did not take the rule and a copy that took it and
got the arithmetic wrong.

### What this genre does not establish

Passing one-step admission does not establish that this candidate will detect the recursion
effect. A high graded arm as well as a high oracle arm leaves little scope for later
improvement, so the mixed-chain pilot has to report the graded arm's ceiling frequency and the
available gain by initial state. Corrections are not removed to avoid saturation without a
separately registered change.

## The genre: `components`, geometric contact connectivity

A schedule of 24 separate 5 by 5 boards, each with one to six occupied cells printed as
coordinate pairs, and one question per board: how many islands are there. The mechanics of an
island are completely public. One decision they never make is the hidden convention:

| Axis | Options | The decision |
|---|---|---|
| `contact_kernel` | `side_contacts`, `corner_contacts`, `combined_contacts` | which two occupied cells count as touching |

Three conventions. Sibling schedules are two different sets of boards on one notation, scored
under one draw. Transitivity is the whole of the work: two cells with no direct contact still
share an island when a path joins them, and a cycle merges nothing extra, so the count is not a
sum of contacts and cannot be reached by subtracting contacts from cells. Six cells with five
side contacts can still be two islands.

**The recipe, and what it fixes.** Each side holds six rows on which a shared side answers
differently from the other two rules, six on which a corner-only rule does, six on which the
combined rule does, and six invariant boards of mutually separated cells that every rule answers
alike. Every pair of rules therefore differs on exactly **12 of 24** rows on both siblings, which
fixes all of the arithmetic: the whole receipt resolves three blocks, no informative row is
evident, the full-receipt ceiling is 1, the registered lookup floor is 5/6 and H is **1/6**. A
reader with no receipt does no better than each row's modal count, which is **3/4**; a reader who
has resolved the rule scores 1; and the ideal oracle-minus-placebo room is **1/4**.

**A catalogue rather than a sampler.** Every subset of one to six of the 25 cells is enumerated,
245505 of them. A board is informative when exactly two of its three counts agree, and it is then
classified by its colour-preserving contact graph, where every pair of cells carries a colour for
a shared side, a shared corner or neither. That gives **46 / 96 / 78** classes in the three
strata. Boards with no contact at all under any rule are the control pools. "Random small boards"
is not a specification: isolated cells merge all three rules, pure side chains merge two of them
and pure diagonal chains merge the other two, and the fixed strata reject those by construction.

**The bounded construction.** For each ordinal the search tries proposals `p = 0..4095`. Each
proposal samples six classes per stratum for A and six more for B from what is left, so no
informative contact graph occurs twice within or across the siblings, then draws one coordinate
embedding per class and one control of each size, with A's control of that size removed from B's
pool. A proposal is refused when a count occurs more than eight times under any rule on either
side, or when neither side holds an informative board whose side contacts close a cycle. A
surviving proposal tries orderings `q = 0..63` and accepts the first whose copy maximum, over
every rotation of A's filing and of its reversal composed with every bijection of the six counts,
is at most **12 of 24** under all three rules. Running out is `ConstructionExhausted`, which is a
named whole-bank failure and never a skipped ordinal or a widened bound. The proposal bound is
4096 because a keyed sweep of ordinals 0 to 63 under a public test master needed proposal indices
from 2 to 1047.

**Ledger's receipt geometry, unchanged.** An island count is one byte and the longest coordinate
list is 35 bytes, but a coordinate list belongs in the task table and never in a receipt field:
an observed value is the submitted count, not the input pattern. Registered widths are therefore
12 / 16 / 4 / 12 with a 1100-byte oracle body allowance, which is a 59-byte row line and a
**2657-byte** envelope, exactly ledger's. That is why this genre moves neither
`receipts-gates-v3` nor `receipts-render-v2`.

### The six checks this genre adds

`copy_profiles` gives it the `ordered_tokens` profile, because the six counts are a complete
published vocabulary in a published order. The added checks are brought by the **genre name**
rather than by the profile, because a profile says what a family's answers are and two families
can share one and have nothing else in common: a geometry audit run over a schedule of dates
would be asking a ledger whether its boards are on the board. They live in
`generators/components_audit.py`, which never imports the production contact predicate: it builds
its own coordinate neighbourhoods from each option's own name and floods from every unvisited
cell.

| Check | What it asks |
|---|---|
| `components_semantics` | an independent flood fill agrees with the scorer on every row under every rule, every board is nonempty, in range, distinct and no larger than six cells, every key value is in the published vocabulary, and no printed identifier or correction changes under serialize and read back |
| `components_shape` | both sides hold six rows in each of the three informative strata and six separated controls, realize every count under every rule, keep no count over eight times, hold a side-contact cycle, and share no informative contact graph and no exact board |
| `components_support` | at each of the three reference rules on both siblings the keys are injective; the serialized receipts REPORTING EVERY ROW are injective and give three blocks, zero evident rows, ceiling 1, floor 5/6 and H 1/6 with exercise holding; movement is exactly 12; the registered mask law leaves 329/368, 305/368, 3/46 and 35/46; and the one-rule-wrong score, the leverage and the registered copy calculation hold across the support |
| `components_bijection_copy` | the exact maximum over all 48 row moves and all 720 value bijections is at most 12 of 24 under every rule, in both directions, with the attaining transformation printed |
| `components_analogy` | no informative contact graph occurs on both siblings, so a correction cannot be carried across by recognition, and the public modal fallback that remains scores exactly 3/4 |
| `components_public_contract` | no committed neutral token normalizes onto a legal answer, a verdict or a status; each task carries its own registered scope sentence and not the other's; identifiers and coordinate lists survive the shared reading; and neither the printed coordinate order nor any of the eight symmetries of the square changes a count |

They are mandatory: they run wherever the eleven common checks run, which is every admission
report and therefore every bank fill and every bundle reverification. A family whose geometry
check fails is not admitted on the strength of the common ones.

The registered `ordered_tokens` closure here is exactly six cyclic token maps, which with 48 row
moves is 288 transformations. `components_bijection_copy` tests 34560 per rule. Island counts are
genuinely ordered numbers, so the nominal-value objection is less acute here than for a family
whose answers are destination codes, and the stronger bound is retained anyway rather than making
substantive disjointness rest on calling a relabelling "ordered".

### One use per chain, and a master key of its own

**The receipt-shaped half of `components_support` is asked of the receipt that reports
every row.** Which two rows a stream drew is not what a recipe check is about, and
several of those numbers are deliberately false on masks this policy can draw: a mask
that drew two of the six controls resolves one block and aliases all three rules, and it
is not redrawn. So the injectivity, the blocks, the evident rows and the gate's own
ceiling and floor are computed on the full receipt, which is a fact about six rows of
each informative type and six controls, and the law beside them is the same recipe's
statement about what is served. The row recipe makes the law a closed form in the
reported count alone, so those four numbers are the same on every compliant pair.

**At most one components process per chain**, and that is a release restriction rather than a
caution. If a previous draw were known and repetition forbidden, two rules would remain, and the
lookup floor's single all-passed bit separates those two outright: the floor reaches the ceiling
and H is zero. Adding an axis is not an automatic repair. A second use needs every possible prior
chain history and the remaining support evaluated first, and a larger support used more than once
still needs distinct draws and documented conditional evidence.

**A master key of its own, separate from every other genre's.** `draw_convention`'s coordinates
are the stream label and the ordinal and carry no generator name, so two genres under one key
draw the *same* rule at every ordinal however different their surfaces are. That is refused where
the key is recorded: `begin_attempt` refuses a commitment the key history already holds for
another generator, and a master that is the key of another genre's bank in the same evidence
directory.

### The review pack for this genre

`components_review.export` writes the first admitted instance's two task texts, all three oracle
cells, the row count, and the graded, placebo and oracle cells of all seven filing classes on both
siblings under **each of the three rules**. Each rule's filings are its own, so a canonical filing
is correct under the rule being shown. The cells taken under the two rules that were not drawn are
labelled review counterfactuals in the manifest and are offered as coverage of nothing: they show
what a receipt would have said had that rule been drawn, and no bank member holds them. Beside the
renders the exporter writes private worksheets, which are not renders and are in no bundle field.
The pack names no reviewer, so `review.verify` refuses it until a person does.

### The model room screen for this genre

The `screen` command analyzes recorded outcomes and runs no model. **No model screen has been run
for this genre.** The procedure, in one place for whoever runs it, with
`components_review.screen_refusals` as the companion audit of the allocation. That audit takes the
bank's admitted order and the recorded `ScreenRecord` as well as the allocation, and reads the
identities against the first and the outcomes against the second, because an allocation checked
against its own fields is an allocation that says whatever it likes. The cases and the recorded
pairs are matched one to one, with each case naming the pair it executed, an ordinal the
allocation puts in the final screen and the A filing the record binds to that pair; the release
statistic is the mean over those matched pairs and over no others, because counting 36 cases
beside 36 pairs says nothing about which case was which execution:

1. Reserve the **first 12 admitted identities** as an exploratory prefix, at most three per
   initial state in cyclic order. Unused reserved identities do not become final cases. Freeze
   the complete instrument and model configuration before final execution; any change after
   exploration needs the prescribed rebuild and fresh final evidence.
2. The **next 36 admitted identities** are the final screen. Assign case j, in bank population
   order, to initial state `j mod 4` with j starting at zero, which gives **nine distinct cases
   per state** and 36 distinct cases in all. Use the roster's common four frozen initial states
   with their digests and histories predeclared. Each case is a fresh restoration at zero prior
   dose and never a continuation of another case.
3. Execute A once, seal its filing, and make a complete common checkpoint. Resume three copies
   with graded, byte-matched placebo and oracle payloads from that seal, then give each exactly
   the same B bytes. There is no B feedback and no further task in a case. Hold model, budgets,
   tools, standing instructions and continuation policy fixed. That is **36 A executions plus 108
   B executions, 144 in all**, excluding exploration. Report actual attempts when a failure
   prevents a later execution, and do not replace an unfortunate filing or a failed branch.
4. Record the case identities and A filing digests in the standard `ScreenRecord`: family, model,
   task_seeds and the five pair fields. Use `min_pairs=36`, `min_room=0.05`, `min_ratio=0.25`,
   `floor=0.0`, `floor_rule="drop"` and a truthful candidate-selection disclosure. Do not add a
   state field to that strict pair record: a hashed companion allocation binds each identity to
   its initial-state digest, bank ordinal, source and task digests, sealed checkpoint and
   execution evidence.
5. Require mean O minus mean P at least 0.05, its paired 90 percent bootstrap lower endpoint
   strictly above zero, and `(mean G - mean P) / (mean O - mean P)` at least 0.25, which is a
   ratio of pooled differences and never an average of case ratios.
6. **Beside them**, as a predeclared roster-release condition and not a `screen.json` field: mean
   executed oracle grade at least **0.90**. Report O, P, G, room, gain, saturation and failure
   counts by initial state as well as pooled. The bundle verifier does not enforce this and the
   release audit recomputes it from the same sealed outcomes.
7. Classify errors from observable evidence: a payload rejected as instructions, scope limited to
   the prior task, a complete key under another rule, mixed counting mistakes, malformed or
   omitted filings, and execution failure. Match B filings against all three canonical keys and
   keep any diagnosis that needs prose labelled as human assessment. Preserve ambiguous cases: a
   high complete-key rate does not by itself separate refusal from a wrongly inferred rule.

The common standing instruction every arm starts with is the registered one, and
`components_review.STANDING_INSTRUCTION` is what the screen evidence is audited against. **This
repository does not install it.** The harness that carries standing instructions is a different
one and still carries the shorter sentence this one replaces, so evidence gathered before it is
installed is refused by the audit rather than accepted and explained afterwards.

### What this genre does not establish

Small boards and one-digit answers make correct execution attainable and say nothing about
whether a reader will use the oracle's prose; that is what the screen is for. Only `log2(3)` bits
are unknown, so easy inference or rapid saturation stays possible and the pilot has to report the
graded arm's ceiling frequency and the available gain by initial state. The catalogue exclusion
stops a correction being carried across by recognizing the same complete coloured contact graph;
it is not a bound on every program that examines geometry and corrections, and comparing candidate
rules against a collection of corrected boards is the intended induction rather than prohibited
lookup. A green mathematical fixture is not a filled production bank.
## The genre: `retail_refund`, economic destination choice

A schedule of 24 already authorized returns. Each prints its order, its returned items, its
refund total, a purchase flag and the complete list of payment instruments the house may refund
it to. The public rules settle eligibility, class membership, the empty-class fallback, ties and
normalization. Three decisions they never make are the hidden convention:

| Axis | Options | The decision |
|---|---|---|
| `route_policy` | `route_gift`, `route_origin`, `route_usedgift` | which destination class the house prefers |
| `gift_pick` | `gift_list`, `gift_max`, `gift_min` | which gift card is taken inside the gift-card class |
| `origin_pick` | `origin_list`, `origin_max`, `origin_min` | which original contributor is taken inside its class |

27 conventions. Sibling tasks are two schedules over different organizations, item pools,
amounts, identifiers and code bindings under four presentation templates (`apparel`, `home`,
`electronics`, `outdoors`), scored under one draw. The answer is a two-byte destination code.

**The classes are disjoint, and that is a derivative rule.** A gift card belongs to the
gift-card class whether or not it paid for the purchase; the original class is every listed
credit card or PayPal instrument with a positive contribution. Native tau2 permits an original
payment method or an existing gift card and its return tool restricts the non-gift destination
to the first payment in the order's history. This genre permits every positive non-gift
contributor instead and puts a purchasing gift in the gift class alone. Neither change may be
presented as unchanged upstream behaviour.

**Codes are row-local, and that is what keeps the correction honest.** `D2` in one case and `D2`
in another name different instruments of different types, so a code by itself is not a selector
label. The published vocabulary is `D1` through `D6` in listing order and every accepted table
realizes all six under every draw. Registered widths are 12 / 16 / 4 / 12 with a 1100-byte
oracle body allowance, which is a 59-byte row line and a **2657-byte** envelope, the same
geometry the ledger uses. A native payment ID would not fit: `credit_card_1234567` is 19 bytes
and the 12-byte correction would truncate it into a shorter legal looking answer. A full-ID
instrument is a separately admitted revision, not an alternate spelling in this one.

**What the profile multiset is for.** Each side prints six cases of each of four public kinds:
gift cards only, original instruments only, both classes with a gift purchase, and both classes
without one. A prints two candidates per class, three cases with the smaller amount first and
three with the larger first. B prints three per class with the median amount first and the
minimum and maximum after it in a random order. The single-class strata are what keep the
otherwise unused selector material under an unconditional route; the two orientations are what
separate the first-listed selector from an extremum; the two purchase types are what separate
the three routes. B's median-first triple is what makes an inferred extremum preference file a
different code from the printed first candidate, and the two-against-three cardinality is what
leaves no structural position a corrected destination could be reused from.

**The room, as registered arithmetic rather than as a measurement.** A careful uninformed reader
scores 5/9 on A and 5/18 on B; a copy executing the stated rule scores 1. Through the shared
gate on real serialized observations the full receipt's ceiling is 1 and the lookup floor is
7/12 under either unconditional route and 11/18 under the conditional one, so the smallest
headroom is 7/18 against a registered bar of 0.05. B's best one-axis-wrong score is 18/24 and
its weakest axis leverage is 0.25.

**Under the full receipt this genre saturates.** A consultation priced the reduced receipt
forms for it: four of the 24 rows sampled uniformly and reported in full, with both slots
neutral on the other 20, leave an ideal level of 0.799323 against a law-level lookup floor of
0.576664, room 0.222660 above it, a minimum single-axis distinction probability of 0.436759, an
expected 2.933476 compatible conventions, a singleton probability of 0.167482 and 1.322865 bits
of posterior entropy. Uniform samples of four, five and six rows pass the three law bars and
four is the smallest that does. **This build declares the full receipt.** The sampled policy is
a separate mechanism, declared per generator, and when it arrives the declaration of four rows
for this genre is one line.

### The checks this genre adds

The family declares `ADDITIONAL_CHECKS` and a `check_additional`, and `run_checks` runs one
guarded callback per declared name beside the eleven. They live in
`generators/retail_validation.py`, which is a second implementation: it reads the printed body
back rather than the stored table, derives each option's operation from the option's own
identifier, and picks by explicit stable sorting where the production selector picks by a keyed
minimum. `judge_cells` compares the renderer with the scorer and cannot catch a scorer that is
wrong in both.

| Check | What it asks |
|---|---|
| `retail_surface` | the printed body reads back, every field stays inside its registered bounds, the class partition, contribution totals, purchase flag, profile counts and class cardinality are what the schedule promises, all 27 keys agree with an independently implemented selector on both sides, and no case id or correct code changes under serialize and read back |
| `retail_support` | the registered gates and the exercise, materiality and copy checks pass under **every one of the 27 conventions**, not only under the one drawn, and the all-support movement minima and six-code realization hold |
| `retail_profile_transfer` | no case of one schedule shares a code-free structural fingerprint with any case of the other, so a corrected destination's position is not reusable |

`retail_support` is what makes the pass predicate independent of which convention was sampled.
Without it a deliberately constructed pair can pass for one draw and fail for another, and the
conditional prior given the public surface moves.

### The two copying families, and the one that does not qualify

The shipped screen is the registered `ordered_tokens` closure: the six rotations of the
published code order composed with the 48 row moves, at most 288 filings, read against the
registered 0.50. Accepted pairs clear it under every convention, by construction.

A wider family was asked for separately: every one of the **720 relabellings** of the six codes
composed with the same 48 row moves, computed by a six-by-six match-count matrix and exact
assignment optimization, quantified over the whole support, at a conservative 0.50.
`retail_validation.bijection_copy_maximum` computes it and `check_retail_bijection` reports it.

**It does not qualify, and no schedule built to this recipe can make it.** Over 4000 pairs built
to the profile recipe the lowest support-wide all-bijection maximum is 13 of 24 (0.541667), the
median is 14 of 24, and none reaches 12. This is a property of 24 rows over a six-code
vocabulary rather than of any particular draw: the assignment optimum for one row move already
runs near 9 of 24 on unstructured keys, and the maximum over 48 moves carries it past half. The
number is reported beside every admitted pair and in the review pack, and it is never
relabelled as the registered closure. Whether this candidate is admitted under the shipped
screen alone, or rejected because the stronger claim is the one the roster needs, is not a
question this build settles.

### The model room screen for this genre

The `screen` command analyzes recorded outcomes and runs no model. The procedure, in one place
for whoever runs it:

1. After the mechanical checks and the human read, use the intended harness and resource limits
   to execute one A copy and three B copies per case, from a common sealed A state. Receipt,
   placebo and the registered prose oracle are assigned at the ordinary delivery point. B grades
   are withheld and every artifact reference and transcript is preserved. No executable solver
   is inserted into the oracle treatment.
2. If usability exploration is needed, use the **first 12 admitted cases** as a separately
   labelled diagnostic and freeze the candidate before any admission measurement. Then use the
   **next 36 previously unused admitted cases**, fixed before outcomes are inspected. Without
   exploration, use the first 36.
3. Allocate **nine cases to each of the four frozen initial states**, in their registered order,
   cyclically by case position. That allocation and the model configuration live in run
   provenance: the screen schema has no fields for them.
4. That is **36 A executions plus 108 B executions, 144 in all**, at zero prior receipt dose
   within each process. Every assigned case and the registered failure dispositions are kept; a
   run does not keep only the completed favourable triplets.
5. Record O, P and G by initial state as well as pooled, with rule-reading mistakes, execution
   mistakes, omitted rows, filing omissions, token length and row-level oracle errors.
6. Judge it against the registered bars: `min_room=0.05`, `min_ratio=0.25`, `min_pairs=36`,
   `floor=0.0`, `floor_rule="drop"`, `candidates_screened=1`. **Beside them**, as a predeclared
   roster-release condition and not a `screen.json` field: **mean executed oracle grade at least
   0.90**, with the independent reader scoring 1 on every admitted draw.

A pooled mean can conceal a state-specific uptake failure, which is why the by-state report is
part of the procedure rather than an extra. A solved reference implementation proves
attainability and says nothing about model uptake.

### What this genre does not establish

It is a table served by the receipts engine and not a tau2 session: there is no retrieval, no
customer authentication, no database mutation and no proof that a captured destination is the
sealed action. A tool-backed version would add all of those and no new hidden decision, so it is
not built here. Passing one-step admission does not establish that this candidate detects the
recursion effect, and under the full receipt it saturates, which is what the reduced-receipt
arithmetic above exists to answer. The disjointness claim is about hidden decision functions:
retail shares printed-order selection, public fallback and exact row matching with the families
beside it, and what makes it a different decision is that it routes value to an eligible
economic instrument rather than measuring an elapsed count, resolving an entity or composing
string transformations. Two additions plus ledger make three generators, and the release wants
four.

## Tools (served over MCP)

- **`submit_filing(filing: str)`** is the env's **score terminal**: the call validates its args,
  atomically seals the episode, and runs `finalize`, so a filing is graded exactly once and on
  an episode that can no longer be continued.
- **`terminate()`** is the reserved episode-completion tool. Ending without a filing records
  `no_filing`, not a low score.

**B is a measured path.** The protocol builds A and B independently, so a bank can hold two
row counts, and `receipts_v1` serves either side and seals either side. The artifact checks
therefore run on both siblings: an author's mistake in a cell only B renders would otherwise
surface as a branch finalization failure at seal time, which the chain records as an outcome
and which has nothing to do with what the learner did.

**The terminal returns no verdict, on either channel.** A tool result has two: the JSON
`content`, which on a successful filing is `filed`, `rows` and core's `finalize_error`; and the
`_meta["shogym/feedback"]` sidecar, which by default carries the episode feedback and therefore
the score. This env declares `inband_terminal_feedback = False`, so the sidecar carries the
terminate flag and nothing else. The scalar is still verified, still written to the trace and
still returned by `evaluate`; it does not cross into the agent's process, and the served tests
assert the whole result rather than half of it.

The reason is that an exact score narrows the draw. Filing one candidate convention's answers
and reading back the score partitions the 72 conventions, and a perfect score names the drawn
one outright. The environment does not deliver a receipt either: an experiment decides which
cell a branch is served. An env whose terminal carried the grade would be putting a receipt in
every arm, including the one that is meant to be empty.

`{filed, rows, finalize_error}` is the SUCCESS shape and not the only one. A terminate before
filing gives core's abort record; a failure inside the terminal transaction gives
`{"correct": false, "finalize_error": true}`; a call whose arguments the schema refuses gives
`{"error": ..., "validation_error": true}` with an empty `_meta` and the episode still open;
and any call after the seal that would be dispatched into the episode, an unknown tool
included, gives the constant `<episode sealed; no further tool calls are dispatched>`. The
server's own `describe` is not one of those: it is public discovery, it answers the same task
contract before and after the seal, and it carries no verdict. None of these forms carries a
score, and none carries anything that moves with the draw.

## Scoring

- **`component_score`**, the sealed `[0, 1]` scalar: the fraction of records the filing got
  right, equal weight per row, over the printed row count, rounded to six places.
- **`solved`**, the whole table right.
- **`rows_filed`** and **`rows_omitted`**, what the reading had to decide, so a malformed filing
  is visible as one.
- **`no_filing`**, the reason code, when nothing scorable arrived.
- **`grade_error`**, and nothing else, when the act this environment owns failed closed. The
  render, the envelope check, the hash and the commit are one act; when it fails there is no
  committed fork, so there is no score to report and the episode says so instead of reporting
  one. That covers parsing, rendering, judging, publishing, and a lost session.

  It does **not** cover a failure of the core verifier that runs after `finalize` returned. By
  then the fork is committed and valid, and core's own contract on that path is to publish no
  feedback at all, so such an episode carries an **empty** feedback list beside a committed
  fork. `finalize_error` means the terminal transaction failed; on its own it implies neither
  "no fork" nor "`grade_error` is present".

## The gates: `receipts-gates-v4`

Three questions, all answered controller-side at zero execution cost, per instance, from the
**serialized bytes** of the receipt the renderer actually produced. Not from the structure
behind them: a renderer can carry a field the bytes never show, or two values a registered
width truncates into one, and a gate reading the structure would score a receipt the agent
cannot read. `shogym receipts gate <name>` runs them and exits
nonzero on a fail.

- **R, resolution.** Hold every axis but one at the drawn convention and vary that one. Each
  option produces a rendered receipt; options whose receipts print the same thing are options
  the agent can never tell apart. An instance fails when every axis of three or more options
  sits at two blocks or fewer. Binary axes are outside R: two blocks is full resolution for
  them. The siblinghood exercise check covers them at their own arity, where the family
  reports every row; where it reports only the rows a committed mask drew, the `law` check
  covers them over the registered mask law instead.
- **S, non-self-interpretation.** Five exact checks, on the declared labels **and** on the
  serialized bytes: no row is labelled by axis; the evident rows alone do not already reach
  the whole receipt's resolution; the bytes print no axis name and no option token that is not
  also a legitimate answer; every slot prints only what its registered grammar allows; and the
  printed row order does not move with the convention. Prose in the task text stays a human
  read.

  The grammar is not the last line. A licensed value can still be chosen for what it
  encodes: four corrections spelled as ordinary band names, one per axis, hand over the
  whole rule while every gate sees a legal answer in a legal slot. So a graded row is not
  a choice at all. It is built from the scorer's own `RowOutcome` by a shared grader, and
  the `graded` check walks the whole convention support asserting that what the generator
  rendered is what that grader would have rendered.

  The **grammar** is what catches a rule that spells nothing. A slot is not a free string: a
  verdict is one of two literals, a correction is empty or the row's actual answer, and every
  value a slot realizes over the whole convention support is checked against that closed set.
  Without it a renderer can print `2100` and hand the child the entire rule while passing a
  search for forbidden words.
- **H, room above lookup.** The ceiling must stand above the lookup floor, and **both are
  optimized over the sibling task's legal action space**: one answer per row, so the best
  action is the per-row posterior mode. Committing to a whole convention key is a strictly
  worse rule, and pricing a design with it reports headroom that is not there.

Which rows are **evident** is derived, never declared. A row is evident when it responds to
exactly one axis and prints a distinct thing for every option of it: such a row is an index,
and reading the option off it costs no induction. The derivation is deliberately generous, so
the floor is high and the gate under-reports headroom rather than over-reporting it.

**Which of the three decide depends on what the receipt reports.** For a family that
reports every row, all three: there is one receipt and no other it could have served. For a
family that reports the rows a committed mask drew, what decides is S's printed form, which
is no row labelled by axis, no vocabulary or grammar leak in the bytes, and a printed order
that does not move with the convention. R, H and S's information equality are quantities of
the realized mask. A mask that happens to omit an axis leaves R short and one that happens to
draw the evident rows leaves no headroom, and such a mask is not redrawn: refusing the
instance instead is the same filter by another route, and the masks the bank would then hold
are the masks that happened to be rich. What replaces them is the registered law, read in two
places, the distinguishing bar in the `law` check and the band over the bank. Every realized
number is still computed and printed, and `shogym receipts gate` says so under the report.

The gate set is named because it is chain-specific: it implements R, S and H from the
instrument and deliberately excludes the instrument's later count gate, whose channel is a
paid mechanism here rather than a defect. Nothing claims the instrument's own verdict.

The name is at **v4** because what a receipt reports is now a registration of its own: a family
says which of its rows carry a verdict and a correction, and for one that reports only some of
them the demand that every axis is exercised by the realized receipt is replaced by a demand on
the registered mask law. v3 was the copy bar being read against whichever registered family a
generator declares, with the further checks a declared profile brings. The numbers are the
numbers they were; the rule they are read against is not, which is what a named version exists
to keep apart. A family admitted under an earlier label does not publish under this one, and a
bundle frozen under an earlier label does not verify. The renderer configuration is at
`receipts-render-v3` for the same kind of reason: which rows a cell reports is part of how it is
built, as the second genre's wider correction slot and larger envelope were before it.

The **policy identity** is separate and is the policy's own name, recorded in every instance
the bank commits. A gate version says which rule judged a family; a policy name says which
instrument it judged, and the two move independently.

### The vectors

The instrument's hand-checkable vectors ship as real generators (`generators/vectors.py`) so
the gates are validated through the same path a family runs: `slots-c3` / `c4` / `c6`
(ceiling `2/c`, placebo `1/c`, headroom exactly zero, two blocks per axis), `merge` (crossed
merges resolve three), `one-row` (one row prints at most two signatures), `binary`,
`copies-20` against `copies-1` (item count with an unchanged readout is worth exactly
nothing), and `affine`, the correlated latent sampler, asserted REJECTED.

What they exercise is the gate observation and serialization path, not the shared fork judge:
they register one slot and emit raw empty values, so they do not satisfy the two-slot and
missing-value rules `judge_cells` enforces on a family. They are **never dealt**: `list` names
them, `materialize` refuses them, and the environment refuses to be constructed on one.

## The named checks

`shogym receipts check <name>` runs eleven for every family, each named separately from the
gates because failing one means something different, plus whichever further checks the
generator's declared copy profile brings with it, whichever its **genre name** brings, and
whichever further checks the generator declares for itself.

Three dispatches, because they answer different questions. A **profile** says what a family's
answers are and therefore which maps a copy screen prices, so it brings the checks a cheap reader
of answers of that kind needs. A **genre** brings the evidence its own hidden function needs, and
two families can share a profile and have nothing else in common: island counts and ledger's band
names are both tokens of a complete published vocabulary, and a geometry audit run over a
schedule of dates would be asking a ledger whether its boards are on the board. A family's own
**declaration**, `ADDITIONAL_CHECKS` beside a `check_additional` that runs them, is the third:
it is optional, it is refused a name that shadows one of the eleven or one a profile or a genre
already brought, and it runs wherever the eleven run.

| Check | What it asks |
|---|---|
| `exercise` | every axis exercised in A's receipt at `min(3, arity)` blocks. Asked of a family that reports **every** row |
| `law` | for a family whose receipt reports only the rows a committed mask drew: over every mask the policy can draw, the probability that the receipt distinguishes each single-axis alternative is at least `0.30`. It reports the expected number of consistent conventions, the singleton probability, the posterior entropy, the no-receipt level, the ideal graded level and the recomputed lookup floor |
| `materiality` | every axis moves B's answers under the drawn convention |
| `copy` | what transfer earns on B, and what a near miss earns |
| `fixation` | the instance rebuilds byte-identically, so every branch of a fork gets one B |
| `envelope` | on both siblings, over the whole convention support: three cells at one size, one alignment, nothing outside the slots moves |
| `graded` | on both siblings, every graded row says what the scorer said about that row |
| `placebo` | on both siblings, the placebo prints its committed tokens, under every filing class |
| `neutral` | no committed placebo token reads as a verdict or a legal answer |
| `oracle` | the oracle states the drawn rule, read by this package's own parser, on every convention |
| `lint` | no option token in either task text |
| `invariance` | both task texts byte-identical under every convention |

The **copy screen** enumerates a registered family of low-complexity maps from A's answers to
a B filing and optimizes the family over its own parameters.

**The registered family is `closure`, and it is closed.** A map is in it when an agent can
build the filing from A's receipt and B's surface alone: what it filed on A, the two published
answer orders, and B's printed rows. It has two kinds of registered move, each closed on its
own before they are combined.

- **Row moves.** Generated by the rotation and the reversal, so their closure is the dihedral
  group: every rotation of the sequence **and** of its reversal, 48 moves at the ledger's 24
  rows.
- **Token maps.** The registered dictionaries and every composition of them. Each is applied by
  leaving anything outside it alone, so it is a total function on the tokens the dictionaries
  and the filing mention, and the compositions are computed generically until no new map
  appears. That terminates because there are finitely many functions on a finite token set.

Token maps act on answer values and row moves on positions, so the two commute and the product
of two commuting closed families is closed. `identity`, `permutation` and `relabel` are
sub-families of it, printed beside the maximum so a failure says which cheap map did it.
`option_flip` (every one-axis substitution) is outside it, because producing that filing means
having induced every axis but one.

A list of generators is not the family, and this cost twice. Listing the rotation and the
reversal without their compositions left out every reversal followed by a rotation, one of
which scored 0.5417 where the maximum reported 0.5000. Listing the token dictionaries without
theirs left out a rank map followed by a filed map, which scored 0.5417 where the maximum
reported 0.4583. Both draws were admitted. A test now applies every registered transformation
to every registered filing and requires the result to be a member, which is the property
product size and sub-family containment cannot see.

Every target vocabulary is a published one. A map into the tokens B's drawn key happened to
realize would price a transfer nobody can perform, because producing it means already knowing
what the hidden draw did to B.

### Every generator declares its copy profile

Which value maps the screen enumerates depends on what a task publishes, so a generator
declares `COPY_PROFILE` and `registry.load_generator` **refuses one that does not**. The
declared profiles live in `copy_profiles.py`:

| Profile | Value maps | Who declares it |
|---|---|---|
| `ordered_tokens` | the registered token dictionaries between two published answer orders, closed under composition | `ledger`, `components`, `retail_refund`, the gate vectors |
| `soundchange_v1` | the 36 global character maps: every permutation of the three deletable vowels crossed with every permutation of the three replacement phones | `soundchange` |

A generator added later declares one of these or is refused at registration, and a profile
name this build does not register is refused the same way. The refusal is the point: a family
whose answers are whole invented words has no complete ordered vocabulary, so pricing it under
`ordered_tokens` would report the maximum of a transfer nobody can perform, and the bar would
be read against a number that measured nothing. `answer_ranks` may return `None` for a profile
that consumes no ranks, and only for such a profile; an empty tuple is not a fictitious
complete vocabulary, and a family that publishes ranks while declaring a profile that consumes
none is refused rather than quietly priced by fewer maps.

### A family may declare checks of its own

A copy profile brings the checks its own cheap readers need. A family can also have an invariant
nothing shared can see, and it declares that itself:

```python
ADDITIONAL_CHECKS = ("retail_surface", "retail_support", "retail_profile_transfer")

def check_additional(self, name, instance, master) -> CheckResult: ...
```

`run_checks` plans one guarded callback per declared name after the eleven and after the
profile's. An absent attribute is the empty tuple, so a family that declares none is untouched.
The declaration is **not** trusted with the names: a repeated name, a name that shadows one of
the eleven or one a profile already brought, and a result handed back under a name other than
the one asked for are refused as failures of that named check rather than run, because a check
whose result can arrive under another check's name is a check that can report a pass somebody
else earned. It is deliberately not a member of the runtime-checkable `Generator` protocol, and
`protocol.py` says why beside it. A declared check is pinned code: it runs at materialization
and again when a bundle rebuilds its population, and the family reaches its implementation by
importing it, so the code pin's walk covers it without a manual hash exemption.

`soundchange_v1` and its bar are a **new registration**. The number 0.50 is carried over as a
conservative initial ceiling and inherits none of `ordered_tokens`'s empirical calibration, so
the combined registration was named `receipts-gates-v3`, and the receipt policy has taken it to
`receipts-gates-v4`; a bundle frozen under an earlier label does not verify against it.

It reports **two** numbers against two thresholds, because they answer different questions. The
no-induction best is what an agent gets for reusing A's answers. The flip best is what it gets
for inducing every axis but one and being wrong about that one, which is a near miss rather
than a copy. A single threshold over both would silently make the stricter question the only
one asked.

A bar and the maps it is a maximum over are **one registration**: a screen that enumerates more
maps reports a larger maximum whether or not the extra maps are a real channel, so a number
measured against a narrower family means nothing held against a wider one. The bar is therefore
`0.50`, calibrated against this closure. Over sixty ledger draws under one key the closed
maximum runs 0.4167 to 0.5833 with a median of 0.5000, and 50 of 60 clear 0.50; held against
the same closure a bar of `0.40` clears none of the 60, which is what a bar calibrated against
a narrower enumeration is worth. Under the whole registered rule 32 of those 60 draws are
admitted, and 15 of the 60 fail copy.

The screen's scope is the enumerated maps. It does not claim to exclude every derivation,
because no extensional test can separate "infer the convention, then solve B" from another
program computing the same function.

## Thresholds are registrations

The gate bars are **registered**: headroom above `0.05`, no-induction copy over the closed
transfer family at most `0.50`, one axis wrong at most `0.875`, per-axis leverage at least
`0.10`. They are the defaults and
`materialize` takes no flag to move them, because which rule filled a bank is the whole of
what the bank means. `check` can be pointed at other bars for diagnosis; what it prints is a
diagnosis, and a bundle carries the registered eight exactly or does not verify.

They sit where the ledger's own distribution makes them bite without emptying the bank. Its
one-axis-wrong score is never below five sixths, so a flip bar under that admits nothing;
`0.875` rejects the worst draws and keeps the rest. Its weakest axis leverage never exceeds
one sixth, so a leverage bar above that admits nothing; `0.10` leaves room under the ceiling.

R's arity and block constants are the settled rule rather than dials, and so is the law
check's distinguishing bar of `0.30`. `gate` will run with them moved, for diagnosis, but it
renames its output `receipts-gates-custom`.

**The law's own bars are read where the bank is.** For a family whose receipt reports only
some rows, the bank-mean ideal graded level has to sit in **`0.75` to `0.90`** and the
bank-mean room above the recomputed lookup floor has to be **above `0.05`**. They are bank
quantities rather than per-instance ones because one table's ideal level moves with how much
its own rows happen to move, and reading the band at each instance would refuse tables for
sitting at the edge of the distribution the band was computed as the centre of. A bank whose
mean sits outside the band is not filled. A bank of ONE instance reads the band at that
instance, which is why nothing here freezes one: ledger's own rooms run from about `0.048` to
`0.093` around a mean well clear of the bar.

**The recomputed floor is the consultation's construction**, and the two agree to nine
places: the evident rows the reduced receipt shows, plus one all-matched bit per dependence
class among the rest, the class of empty local dependence included, averaged over every
canonical reference convention. That class belongs in it because local dependence is derived
one axis at a time, so a row it finds empty is a row no single-axis alternative moves and a
joint alternative can still move it. The six run 5 table pairs give `0.753568219`,
`0.733884414`, `0.751611431`, `0.734439942`, `0.708373223` and `0.767741008`, mean
`0.741603040`, against an ideal level of `0.817285062`.

The `0.30` is the sampling law's own arithmetic: ledger's weakest registered movement is two
rows, and a four-row mask misses both with probability `choose(22,4)/choose(24,4)`, so such an
alternative is distinguished with probability `0.3116`. The `0.75` to `0.90` band is a design
choice: above it the receipt identifies too much for a later step to improve on, which is the
saturation the sampled policy exists to undo.

The room screen is registered too, and it has five bars rather than three:
**`min_room = 0.05`** with the bootstrap interval's lower bound above zero,
**`min_ratio = 0.25`**, over **36 distinct tasks**, plus a mean executed oracle level of at
least **`min_oracle = 0.90`** and a graded level at least **`min_learning_gap = 0.10`** below
the level a perfect reader of the same receipts reaches, with that paired interval's lower
bound above zero. That level rides on each pair row as `ideal`, carried from the gate's exact
computation over the registered mask law rather than measured in the pilot. A low oracle level
is as consistent with copies that could not carry out a rule they were handed as with a family
that leaves nothing to carry, and a family already at its receipt's own ceiling clears the
first three bars emphatically while having nothing left for a later step to read better. A pair is one execution of
A and three of B, so that is 36 A executions and 108 B executions, the costing a cheap
generated family was planned against. Verification
recomputes room, gain, ratio and the interval from the raw rows and compares them with those
bars: recording that a bar was moved is not refusing to deal a family admitted under an easier
rule, so a bundle whose recorded bars are not the registered ones is refused. `screen` is a
diagnostic and will report against other bars, saying which it used.

When more than one candidate was screened the record has to say so and say what was done
about it, and a record that does not is refused outright. Disclosure is not an adjustment: no
correction is applied, and the interval, the bars and the diagnostic verdict are the same for
one candidate and for a million, so a selected winner clears that arithmetic on exactly what
one candidate clears it on. The result says so on its own printed lines.

So **deal evidence takes one candidate**. `candidates_screened == 1` is the registered rule,
and `verify` refuses to freeze a bundle on a record that names more, however well disclosed. A
selected record keeps its unchanged diagnostic statistics and is still scored and printed by
`screen`; what it cannot do is establish eligibility. Registering an adjustment, and dealing
selected evidence under it, is an open maintainer call.

The outcomes artifact is a **run, not a list of scores**. It names the model it was taken
with, the task seeds it was taken over, one identified record per pair, and the bars it is
judged against. Three numbers say what was measured and not what it was measured on, and a
file of anonymous scores verifies as readily against a family and a model it never touched. A
pair is one task: seeds and instances must each be distinct, because repeated filings against
a single table clear the sample floor while the pilot has one sampled unit and the pair
bootstrap would price them as independent draws.

`materialize` freezes a bank of passers under the registered rule. Which ordinals passed and
what fraction got in are recomputed wherever they are wanted and stored nowhere. A bank is not
dealable; a bundle over one is.

Counts must be positive and bars must be finite numbers in `[0, 1]`: a comparison against NaN
is silently false, so a NaN bar would admit everything while looking like a bar.

## The admission bundle

Gates passing is necessary and is **not** admission. What may be dealt is a **bundle**: one
frozen directory holding everything admission rests on, addressed by the hash of its own
manifest. `receipts_v1` opens a bundle by that digest and nothing else, and a bank that was
never bundled is not dealable.

```
<digest>/
  manifest.json     every file below, with its size and its hash, in canonical form
  bank.json         the generator, the genre, the renderer, the master key, the size
  instances.json    per instance: its ordinal, its content digest, its convention commitment
  thresholds.json   the bars it was filtered under
  code.json         the hash of the code that certified it, and the modules it covers
  screen.json       the room screen: the family, the model, task seeds, rows, and the bars
  review.json       the reviewer, the checklist, the seeds, the family and bank it was
                    read of, and the renders that were read
  renders/...       those renders, byte for byte
```

The bundle's digest is the manifest's digest, and the directory is named by it, so there is
one name over all of it and nothing to shuffle between directories. That says these files were
frozen together; it does not say they are about the same family and the same draw. What says
that is written inside the two artifacts a person supplies: the screen names the family it was
taken on, the review pack names the family and the identity of the bank it was read from, and
verification refuses either when it is not this bundle's. A pack read for one bank is
otherwise portable to any other with renders of the right size.

**Every quantity is recomputed from those files and the running code.** `verify` is the one
eligibility operation, and production, the roster and `shogym receipts verify` all call it:

- the **manifest** is reserialized and required to be the bytes on disk, the directory's name
  is required to be its hash, every listed file is hashed and sized, and a file in the
  directory that the manifest does not list is a refusal;
- the **code pin** is compared with a hash computed here over the code that decides what this
  family means. The closure is walked from the import graph rather than listed, so a module that
  starts deciding something joins the pin by being imported; what stays written down is the
  short set that decides nothing (the name-to-module registry, and the gate exhibits nothing
  deals), each with its reason;
- the **thresholds** are required to be exactly the registered field set at the registered
  values, and those are the bars admission is rerun under;
- the **population** is rebuilt: ordinals are considered from zero, the settled rule is rerun
  on each, and the passers are what the bundle holds. Which instances a bank holds is a
  consequence of the key, the code and the rule rather than a list, so there is no ordinal to
  duplicate, none to insert, and no prefix to shorten after the fact. The fraction that passed
  is computed and printed, never stored;
- each **instance entry** is compared, in order, with the recomputed sequence: its digest
  against the rebuilt canonical record, its commitment against the rebuilt convention;
- the **screen** is rerun on its own rows and the recomputed room, ratio and interval are
  compared with the REGISTERED bars: `min_room = 0.05` with the bootstrap interval's lower
  bound above zero, `min_ratio = 0.25`, 36 distinct tasks, a mean oracle level of at least
  `0.90`, and a graded level at least `0.10` below the receipts' own ideal with that interval
  above zero. A bundle carries those bars
  exactly, the way it carries the gate thresholds. A diagnostic run may still ask what a
  family does against another bar, and `screen` prints which bars it used, but recording that
  a bar was moved is not refusing to deal a family admitted under an easier rule. A pair is
  one task: task seeds and task instances must each be distinct, because forty filings against
  one clerical table clear the sample floor while the pilot has one sampled unit, and the pair
  bootstrap would price them as forty independent draws. Every decision input is in the
  artifact and none may be absent, identities must be finite names, and the reader rejects
  JSON's nonfinite extensions, since `str(float("nan"))` is the ordinary-looking name `nan`;
- the **review coverage** is enumerated from the rebuilt instances and the family's own
  declarations: every surface template (both pools, since B is served too), every option of
  every axis, every registered filing shape, every row count on either sibling, and at least
  one counterfactual render. A family that declares a count adds eleven more, none of which any
  other category would ever show a reader: a reported row that passed, one that failed, one
  nobody filed and one filed with an empty value; a row that failed and was suppressed anyway;
  a receipt whose reported rows witness every axis and one that leaves an axis unwitnessed; a
  posterior of one convention and a posterior of several with different held-out keys; the mask
  commitment beside the three cells; and a pair of conventions whose reduced receipts are byte
  identical, with what that costs on the held-out task. Each named render has to be a file the
  manifest hashed and large enough to be what it claims: a rendered cell is the envelope size, a
  task text is hundreds of bytes, and a **document** is material assembled out of committed
  values for a reader to read rather than anything the family served. A reviewer that is null,
  blank or not a name is refused when the bundle is built and again when it is read, because
  `str(None)` is the nonempty string `"None"` and a pack export that lost the attesting person
  would otherwise arrive with one.

There are no summary fields, and every file has an EXACT field set. Nothing records how many
ordinals were considered, what fraction passed, how many rows a sibling has, or which stages a
bank claims to have cleared, because every one of those was a place where a file could say one
thing while the family said another. A field the verifier merely ignores is no better: a reader
who finds `reviewed: true` beside a coverage list has no way to know that nothing checked it, so
an unexpected field is refused rather than skipped. Derived values are printed; none is read.

The roster prints nothing it did not recompute. A development bank is mentioned and never
described: its stored fields are unverified by construction, and printing them beside a verified
bundle would give an operator two descriptions of what looks like one family.

### What verification recomputes

Every input a bundle persists or the roster prints, and how it is established. Nothing
is trusted from a summary, and there are no summary fields to trust: a value that is
not recomputed here is either bounded, or human text with no artifact to check it
against, or the one declared trust boundary at the end.

| Input | How it is established |
|---|---|
| every JSON file's bytes | recomputed: reserialized canonically and required to be the bytes on disk, so a file's bytes and what it says are one statement |
| every JSON object's names | exact: a name that appears twice is refused, because Python keeps the last and another reader can take the first |
| every scalar's type | exact: nonblank strings where text is required, non-boolean integers for counts and ordinals, finite non-boolean numbers for bars. `int(1.9)` is 1 and `int(True)` is 1, so nothing is coerced |
| `manifest.json` version and field set | exact: a fixed version and exactly `bundle` and `files` |
| manifest entries | exact: each is exactly a path, a size and a digest, and no path twice |
| manifest bytes | recomputed: reserialized canonically and required to be the bytes on disk |
| bundle digest and directory name | recomputed: the manifest's hash, and the directory is named by it |
| every listed file | recomputed: hashed and sized, and a file the manifest does not list is refused |
| filesystem shape | refused: a link anywhere, the root included, and any file with a second hard link |
| `bank.json` field set | exact: exactly generator, genre, renderer, master, size |
| `bank.generator` / `genre` | cross-checked against the generator being verified |
| `bank.renderer` | cross-checked against the shipped renderer configuration |
| `bank.master` | the key everything else derives from; it is the secret, not a claim |
| `bank.size` | bounded, then used to rebuild; the population it produces is the check |
| `thresholds.json` | exact: the registered field set at the registered values, then used to rerun admission |
| `code.json` | exact two fields, recomputed: the aggregate digest and the module-to-digest map of the import-graph closure, hashed from the running code; a mismatch names the module that moved |
| the population | recomputed: admission rerun from ordinal zero, and the passers are what the bundle holds |
| ordinals considered, passing fraction | derived at verification and printed; stored nowhere |
| `instances.json` entries | exact fields, recomputed: compared in order with the rebuilt sequence |
| `instances[].digest` | recomputed: the canonical instance record rebuilt and hashed |
| `instances[].commitment` | recomputed from the rebuilt convention and the master key |
| `screen.json` field set | exact: the four run fields and all nine decision inputs, none absent |
| `screen.family` | exact: the family the pilot was taken on, refused when it is not this bundle's |
| `screen.model`, `task_seeds` | validated: nonblank finite names, refused rather than defaulted |
| `screen.pairs[]` field set | exact: exactly instance, filing, placebo, graded, oracle, ideal |
| `screen.pairs[]` identities | validated and required distinct: one row per task, and one task seed per row. That the labels name genuinely different tasks is reported by whoever ran the pilot, in the same class as the render boundary below |
| `screen.pairs[]` scores | bounded to `[0, 1]` and finite, then used to recompute |
| `screen` bars (`min_room`, `min_ratio`, `min_pairs`, `min_oracle`, `min_learning_gap`) | exact: the registered `0.05`, `0.25`, `36`, `0.90` and `0.10` |
| `screen.floor`, `floor_rule`, `selection_note` | bounded and used verbatim in the rerun; a selection of more than one candidate must be disclosed |
| `screen.candidates_screened` | exact: the registered `1`. A larger disclosed count is scored and printed as a diagnostic and refused as deal evidence, because nothing adjusts for selection |
| room, gain, ratio, oracle level, ideal-minus-graded gap, intervals, verdict | recomputed from the raw rows and compared with the registered bars here |
| distinct task units | recomputed: counted from the rows and compared with the registered 36 |
| `review.json` field set | exact: exactly reviewer, checklist, seeds, family, bank, renders |
| `review.family`, `review.bank` | exact: the family and the bank identity the pack was read of, refused when they are not this bundle's |
| `review.reviewer` | validated: a nonblank finite name, refused at build and again at verification |
| `review.checklist`, `seeds` | validated as names; what they SAY is human text with nothing to check it against |
| `review.renders[]` field set | exact: exactly category, key, kind, path, and no digest of its own |
| `review.renders[]` path | cross-checked: it has to be a file this bundle's manifest hashed |
| `review.renders[]` size | bounded: a cell is the rebuilt envelope size, a task text is 400 bytes |
| review coverage | recomputed: enumerated from the rebuilt instances and the generator's declarations |
| whether the renders depict THIS bank | **the declared human trust boundary.** A machine can check that the material is present, hashed and the right size. That it is the right material is what the reviewer attests |
| whether the reading was careful | **the second declared human input.** Not mechanically checkable, and not claimed to be |

None of this establishes that the reading was careful. It establishes that the material was in
front of the reader, which is the part a machine can check.

**A generator is first-party code, and an admitted family's scorer defines its ground
truth.** What `score` returns is what the chain seals. The gates and checks measure the
receipt that scorer produces; they exist to catch **mistakes** an author cannot see, not
deceit by one. The human read of the pack inside the bundle is what stands against a generator
written to mislead, and it is also what stands behind the claim that those renders are of this
bank: the machine checks that the material is there and hashed, the person checks that it is
the right material.

Two habits are kept regardless. Everything a check compares against is computed by this
package and captured **before** any renderer is called: the frozen envelope commitment, the
scorer's own outcomes, the drawn convention. Family code is handed read-only views, because a
comparison built after a callback compares against whatever the callback left behind. And a
generator that raises is a failed check, not a crashed report.

Admission samples filings, and the parser's legal filing space is open-ended, so sampling
cannot be where semantics are enforced. When the real filing seals, the committed cells are
checked against **that filing's own outcomes**: every graded row must say what the scorer
said, the placebo must print its committed tokens, and the oracle must parse back to the rule
that was actually drawn. A fork that fails is never serialized or persisted.

After a filing seals, the three cells are **committed once**, keyed by the frozen source, the
task and the whole filing hash. Every later read replays those bytes, and a record whose task,
filing, renderer or source does not match what was asked for is refused rather than replayed.
Committed once is not rendered once: two callers arriving on one filing at the same instant can
each render before one publication wins, and the loser drops its own bytes for the winner's. Forks are written outside the bundle: anything written inside one would make it
disagree with its own manifest.

There is no fallback. An absent bundle is a refusal, not a rebuild: two environments that each
invented one would draw different conventions and serve them under the same visible task id,
so a run would compare tasks that are not siblings while every published field agreed. `gate`
and `check` refuse to run on a genre with no bank, because they report on the instances that
would actually be dealt and generating fresh ones would reopen a cherry-pickable universe.

**Verified once per bundle per process, on sequential opens.** Verification reserializes the
manifest, hashes every file, recomputes the code pin and rebuilds the whole population from the
master key, which on an eight family bundle is seconds of work; and the serve layer builds a fresh
env for every attempt, inside the call that hands the agent its task. A run of F families would
pay it `2F + 2` times, all but two of them while the agent waits, which is how a client with a
short tool timeout turns into a run that looks like an agent that stopped asking for work. So a
verified open is remembered for the life of the process, keyed by the **absolute** directory, the
digest that names it, and the code that recomputed it. Absolute because a relative spelling names
a different directory from a different working directory, so it is not a name for the thing that
was opened; lexically absolute rather than resolved, because the load refuses symlinks and it has
to still see one. The first open is byte for byte what it was. Two callers opening one bundle at
the same instant can both verify it, the way two callers can both render one fork; what they
cannot do is disagree, since both verify the same frozen bytes.

Within one process that cache **trusts the digest**: a hit does not reopen the directory, so a
file edited under it after the first open is served from the verification that ran before the
edit. That is safe because a bundle is content addressed. The directory's name is the digest of
the manifest that hashes every file in it, so an edit makes the directory stop matching its own
name and the next process to open it is refused. Editing a frozen artifact under a live run is
what the content address exists to catch, and it is caught at the next load.

`receipts_dev_v1` serves a bank that was never bundled, for local work. It is **the one
unbundled path**, it is a separate registered name rather than a flag, and it reports
`dealable` as False: the production constructor takes no argument that turns the refusal off,
because one would make the development name decorative.

The identifier a task is published under is its HMAC, never the selector. The numeric
position stays controller-side.

## Still to build

The frozen manifest release with its pairwise disjointness matrix, the rented-family
protocol, and the further genres. Three generators are not a roster: the release still needs the
pairwise semantic matrix over every implemented hidden axis, the independent validator results,
the independent master-key provenance, the chain-use plan proving that `components` occurs at most
once and that repeated other generators use distinct draws, and a named human attestation tying
all of it to final code and bundle hashes. A registry entry is not admission and a verified bundle
is not a roster place.

Three families now declare a count of their own and each was priced on its own tables:
four records of twenty four for ledger, two forms of twenty four for sound change, two
rows of twenty four for components. None is in the recursion roster on the strength of
that arithmetic. What a count establishes is an information form worth screening;
whether a model reads a receipt that reports two rows at all is what the actual-model
screen measures, and none of the three has been screened under its count.

Recorded boundaries, held by process rather than by a check:

- **Which bank was frozen.** `bundle` takes no argument pointing at another bank and `draw`
  takes no seed, but that is the command line and not a proof that nobody chose the key.
  `materialize` writes the key and its commitment before construction begins and keeps them
  whether or not the bank fills, in **two** places that are deliberately not one. The record
  beside the banks is a per-genre summary: it is rewritten when an attempt ends and it moves when
  the evidence directory moves. The **key history** is neither. It is one file for every genre and
  every evidence directory, reached through `SHOGYM_RECEIPTS_HISTORY` rather than through the bank
  directory; it is only ever appended to, so an outcome is a line after the line that recorded the
  attempt and never a rewrite of it; each line carries the digest of the line before it, so a
  removed, reordered or edited line is a chain that does not verify and is refused rather than
  continued; and each started line binds the attempt to the code pin, the gate and renderer
  labels, the generator's declared construction bounds and the evidence directory it was made in,
  because the same key under a wider bound or a later renderer is a different attempt at a
  different instrument.

  `materialize` reads the history rather than the local record, so redirecting
  `SHOGYM_RECEIPTS_BANKS` no longer turns a second attempt into a first. A second invocation has
  the acts the procedure allows: `--retry` makes the recorded attempt again under the key it
  kept, `--reroll REASON` rolls another key and says why, and `--retry --changed REASON` makes an
  attempt under the kept key at a size, a code pin, an instrument or a construction bound that is
  not the recorded one. The complete recorded identity is compared before construction, so a
  plain `--retry` that would change any of those four is refused and names which; a changed
  attempt is recorded with its own identity. All of them leave the first attempt where it was,
  and so does `--force`. A key already committed to another genre is refused where it is
  recorded, in the history or in a bank file beside it.

  An evidence directory made by the build before the history holds its attempts in the local
  record and nothing in the chain, and the refusal reads the chain, so an upgrade would have
  rolled a second key and written it down as a first. A genre in that state is refused until the
  attempts already recorded are reconciled into the chain with `--import-record`, which imports
  what the earlier record actually held: no code pin, no instrument labels and no construction
  bounds, because the earlier build wrote none. A retry of an imported attempt is therefore a
  changed retry.

  Attempts are made one at a time. The claim is a file beside the key history, so two commands
  pointed at two evidence directories wait on each other rather than both reading a chain with
  nothing in it and both writing themselves into it as a first attempt. Two commands naming two
  different histories are two chains and are not serialized by it.

  A line is flushed to the disk, and its directory entry with it, before construction begins,
  because the attempt it records is the one that may not finish. Every valid prefix of a chain is
  a valid chain, so removing lines from the END of one breaks nothing the chain itself can see:
  the file that is left verifies and the genre whose last attempt was in the part removed reads
  as a genre that has attempted nothing. What says otherwise is a POSITION, the last sequence and
  the digest of the line at it, retained in two places that are not the history: a file beside it
  and the record beside the banks. A history shorter than either, or whose line at that sequence
  is not the one recorded, is refused before construction.

  That is still not closure. An operator who rewrites the history, the position beside it and the
  record beside the banks has rewritten three files, and an operator who deletes all of them has
  deleted them; a chain that starts from nothing verifies. What the position adds is that
  removing lines is no longer a one-file edit that nothing notices. Closing it takes provenance
  retained where this process cannot reach it, and publishing the commitment before the bank is
  built is what the record is shaped for.

- **The task text is a stable instance name.** A surface is a pure function of the ordinal, so a
  lineage that remembers a previous link can recognize an instance it has been served before.
- **A review pack's bytes are not recomputed.** The pack names its family and the bank it was
  read from, and its artifacts are inside the bundle at a plausible size; that they are the
  renders they claim to be rests on the person who signed the pack.
- **The code pin stops at two packages.** `shogym.envs.receipts` and `shogym.receipts`. The
  serve lifecycle and the core types that decide what a seal is are outside it, along with the
  interpreter and the dependency lock. The bundle records the module list, so where it stopped
  is readable.

## Layout

| File | Role |
|---|---|
| `protocol.py` | what a genre implements, the published sampler law, and `draw`. |
| `receipt_ast.py` | the receipt structure, the one canonical serializer, and the envelope. |
| `streams.py` | keyed domain-separated randomness and the opaque task identifiers. |
| `filing.py` | the shared line reading, the printable fold, the row parser, the equal-row scorer, and the sentences that say which tasks share a convention. |
| `copy_profiles.py` | the registered families of maps the copy screen prices, and which generator declares which. |
| `generators/ledger.py` | the ledger genre: domains, the scoring function, the three renderers, the oracle template. |
| `generators/soundchange.py` | the sound change genre: the phone inventory, the three passes, the keyed pair construction and its filter, the four surfaces, the oracle template. |
| `generators/soundchange_audit.py` | a second implementation of that cascade, the predicates the pair is filtered on, and the three checks the profile adds. |
| `soundchange_review.py` | the deterministic review pack and the private trace worksheets for that genre. |
| `generators/components.py` | the components genre: the contact rules, the exact board catalogue, the bounded keyed pair construction, the one surface, the oracle template. |
| `generators/components_audit.py` | a second implementation of the island count by flood fill, the predicates the pair is filtered on, and the six checks the genre adds. |
| `components_review.py` | the deterministic review pack and private worksheets for that genre, and the audit of a recorded screen's allocation. |
| `generators/retail_refund.py` | the retail refund genre: the two destination classes, the three selectors, the keyed pair construction, the four surfaces, the oracle template. |
| `generators/retail_validation.py` | a second implementation of that rule reading the printed body, the predicates the pair is filtered on, the three checks the family declares, and the two copying maxima. |
| `retail_review.py` | the deterministic review pack, the equal-amount fixtures, the separating matrix and the private worksheets for that genre. |
| `bank.py` | what is frozen before launch, and the one atomic render after a filing seals. |
| `bundle.py` | the admission bundle: one addressed directory, and the one verifier over it. |
| `review.py` | what a review pack has to cover, enumerated from the family's declarations. |
| `registry.py` | genre name to module, to its materialized bank, and to its bundles. |
| `env_v1.py` | `ReceiptsV1Env` (the registered `receipts_v1`) and its verifier. |
| `mcp_server.py` | the in-process server backing `submit_filing`. |
| `cli.py` | `shogym receipts materialize / draw / gate / check / screen / bundle / verify / list`. |
