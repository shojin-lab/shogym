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

## Rendering order: what exists when

The graded and placebo cells depend on the agent's A filing, which does not exist until the
agent files. Nothing here pretends otherwise.

- **Before launch**, materialized and hashed: every instance (both surfaces, both task texts),
  the convention commitments, the envelope template with its registered slots, the committed
  filler stream, and the renderer configuration.
- **After A seals**, in one act: the parser canonicalizes the filing, the three cells are
  rendered, they are judged, and the blobs are hashed. That is `bank.render_fork`, and it is
  the only place the three cells are made. Every retry replays those blobs; nothing rerenders,
  because rerendering is how two branches of one fork come to differ.

  The record is **committed once**, which is the property the chain needs, and that is not the
  same as rendered once. Two callers arriving on one filing at the same instant both render;
  the record is published by an exclusive link, the loser's bytes are dropped, and both read
  the winner's record back before returning, so every branch of one fork holds one set of
  bytes. Nothing here is durable against an abrupt host failure mid-write: a fork that was
  never published is rendered again on the next attempt, which is the same answer.

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

## The gates: `receipts-gates-v2`

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
  them. The siblinghood exercise check covers them at their own arity.
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

The gate set is named because it is chain-specific: it implements R, S and H from the
instrument and deliberately excludes the instrument's later count gate, whose channel is a
paid mechanism here rather than a defect. Nothing claims the instrument's own verdict.

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

`shogym receipts check <name>` runs eleven, each named separately from the gates because
failing one means something different.

| Check | What it asks |
|---|---|
| `exercise` | every axis exercised in A's receipt at `min(3, arity)` blocks |
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
diagnosis, and a bundle carries the registered seven exactly or does not verify.

They sit where the ledger's own distribution makes them bite without emptying the bank. Its
one-axis-wrong score is never below five sixths, so a flip bar under that admits nothing;
`0.875` rejects the worst draws and keeps the rest. Its weakest axis leverage never exceeds
one sixth, so a leverage bar above that admits nothing; `0.10` leaves room under the ceiling.

R's arity and block constants are the settled rule rather than dials. `gate` will run with
them moved, for diagnosis, but it renames its output `receipts-gates-custom`.

The room screen is registered too: **`min_room = 0.05`** with the bootstrap interval's lower
bound above zero, **`min_ratio = 0.25`**, over **36 distinct tasks**. A pair is one execution of
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
  bound above zero, `min_ratio = 0.25`, and 36 distinct tasks. A bundle carries those bars
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
  one counterfactual render. Each named render has to be a file the manifest hashed and large
  enough to be what it claims: a rendered cell is the envelope size, a task text is hundreds of
  bytes. A reviewer that is null, blank or not a name is refused when the bundle is built and
  again when it is read, because `str(None)` is the nonempty string `"None"` and a pack export
  that lost the attesting person would otherwise arrive with one.

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
| `screen.json` field set | exact: the four run fields and all seven decision inputs, none absent |
| `screen.family` | exact: the family the pilot was taken on, refused when it is not this bundle's |
| `screen.model`, `task_seeds` | validated: nonblank finite names, refused rather than defaulted |
| `screen.pairs[]` field set | exact: exactly instance, filing, placebo, graded, oracle |
| `screen.pairs[]` identities | validated and required distinct: one row per task, and one task seed per row. That the labels name genuinely different tasks is reported by whoever ran the pilot, in the same class as the render boundary below |
| `screen.pairs[]` scores | bounded to `[0, 1]` and finite, then used to recompute |
| `screen` bars (`min_room`, `min_ratio`, `min_pairs`) | exact: the registered `0.05`, `0.25` and `36` |
| `screen.floor`, `floor_rule`, `selection_note` | bounded and used verbatim in the rerun; a selection of more than one candidate must be disclosed |
| `screen.candidates_screened` | exact: the registered `1`. A larger disclosed count is scored and printed as a diagnostic and refused as deal evidence, because nothing adjusts for selection |
| room, gain, ratio, interval, verdict | recomputed from the raw rows and compared with the registered bars here |
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

`receipts_dev_v1` serves a bank that was never bundled, for local work. It is **the one
unbundled path**, it is a separate registered name rather than a flag, and it reports
`dealable` as False: the production constructor takes no argument that turns the refusal off,
because one would make the development name decorative.

The identifier a task is published under is its HMAC, never the selector. The numeric
position stays controller-side.

## Still to build

The frozen manifest release with its pairwise disjointness matrix, the rented-family
protocol, and the further genres.

Recorded boundaries, held by process rather than by a check:

- **Which bank was frozen.** `bundle` takes no argument pointing at another bank and `draw`
  takes no seed, but that is the command line and not a proof that nobody chose the key.
  `materialize --force` rerolls the master, the bank directory is redirectable by
  `SHOGYM_RECEIPTS_BANKS`, and a bank record is five fields anyone can write; nothing counts
  the rolls. Closing it takes append-only external provenance the v0 hash set does not have.

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
| `generators/ledger.py` | the ledger genre: domains, the scoring function, the three renderers, the oracle template. |
| `bank.py` | what is frozen before launch, and the one atomic render after a filing seals. |
| `bundle.py` | the admission bundle: one addressed directory, and the one verifier over it. |
| `review.py` | what a review pack has to cover, enumerated from the family's declarations. |
| `registry.py` | genre name to module, to its materialized bank, and to its bundles. |
| `env_v1.py` | `ReceiptsV1Env` (the registered `receipts_v1`) and its verifier. |
| `mcp_server.py` | the in-process server backing `submit_filing`. |
| `cli.py` | `shogym receipts materialize / draw / gate / check / screen / bundle / verify / list`. |
