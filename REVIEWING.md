# Reviewing shogym

How this repository reviews changes to the serve layer (`src/shogym/serve/`) and to env ports.
Both surfaces share one failure mode: a run that reports itself intact while the record it left
and the answer it gave disagree. A test suite that only drives the happy path never sees it, so
review here is adversarial by construction. The reviewer's job is to find the input that makes
the layer lie, and the author's job is to reproduce that input before touching any code.

The method was not designed up front. It accumulated over roughly thirty review rounds on four
pull requests, and those threads are the worked examples cited throughout:

| PR | what it changed | rounds |
|---|---|---|
| [#112](https://github.com/shojin-lab/shogym/pull/112) | a failed seal hands back a row, never a recomposition (`stream.py`) | one review, one clean re-review |
| [#113](https://github.com/shojin-lab/shogym/pull/113) | a lifecycle deadline test taken off the wall clock, plus a coverage claim refuted | one review, no findings |
| [#114](https://github.com/shojin-lab/shogym/pull/114) | fail closed wherever a served episode runs the env's own code (`episode.py`, `stream.py`) | fifteen and counting |
| [#115](https://github.com/shojin-lab/shogym/pull/115) | the `orca_bench` port, phase one | sixteen |

Citations below name the PR and the round: "#115 round 5" is the fifth review and its fix.

## The loop

Review, fix, re-review. Repeat until a pass comes back clean. That is the whole shape, and every
property this document describes exists to make one of those three steps mean something.

**One pass is never exhaustive.** Every one of the fourteen reviews on #114 found a defect at a
head the previous round had fixed and re-verified, and #115 ran sixteen rounds the same way. Five
consecutive rounds of #115 (3 through 7) landed on one publish path, each on a window the previous
fix had opened. The correct inference from a clean round is that the last pass found nothing, not
that nothing is there, which is why a clean round is a stopping rule rather than a proof (see
[When to stop](#when-to-stop)).

**Strictly bottom-up on a stack.** A PR is not reviewed above an unclean one below it. While #112
held the lower position on `stream.py`, #114 kept its own edit to a single 36-line hunk clear of
the region #112 was rewriting, and only dropped that constraint once #112 had passed a clean
re-review and merged (#114 rounds two and eleven). Re-review after every fix round, not only at
the end: a fix that closes a finding is itself unreviewed code, and in both #114 and #115 the
next round's finding was usually inside it.

**The fixer is briefed adversarially.** A finding is a hypothesis with evidence attached, not an
instruction. Three verdicts are available and each has to be earned by reproduction:

- **As filed.** Most findings.
- **Bigger than filed.** Also common, and worth stating rather than quietly fixing the larger
  thing. #114 round two's finding named the `verdict` field; `status` was the same shape and
  worse, because the commit compares against it. #114 round three's named the initial read of the
  tool collection; the episode reads it a second time to find the terminal it must enforce, so an
  env that answers once and refuses after raised from the middle of construction. #114 round ten's
  named a wrong-typed schema; a perfectly valid `{"type": "array"}` root reaches the same fail-open
  without raising at all. #112's named lost provenance; the same branch also discarded an outcome
  the agent had earned, which is the worse half and is not about provenance.
- **Does not hold.** Rebut with evidence, in the same comment, and say what shipped instead.
  #114 round two refused half a finding (spans opened before `open_env` do not bypass a cleanup
  block, because that method drops spans on every refusal branch deliberately) and refused the
  suggested fix as too broad, shipping a narrower classification (`TaskContractError`) that keeps
  a task-local failure task-local. #115 round 9 went after the reviewer's preferred option first
  and reported it unreachable: the hub computes its dataset hash server-side, nine plausible
  reformulations miss, so the port ships a pinned manifest instead. That was a negative result
  checked rather than assumed. #112's body rebuts two claims in the issue it closes, and #113
  refutes an issue outright by showing the coverage it wants already exists at the tip.

## What a round is made of

### The review comment

Written against one exact head, from a clean workspace, and it changes no code. It contains:

- each finding with a priority (`[P1]`/`[P2]`, or `[blocking]`) and `path:line` anchors;
- a **reproduction** the author can run, with its printed evidence pasted in. For a race, the
  exact interleaving and where each participant is held. #115 round 4 gives a pure in-process
  reproduction of an endpoint audited from a different reading than the one that was equipped;
  #114 round two gives four-line state dumps (`state`, `terminated`, durable record, `close()`);
- what the fix has to establish, in properties rather than in edits, and where the reviewer is
  unsure which of two shapes is right;
- the validation performed at that head: the PR-specific module, the full offline suite, `ruff`,
  `pyright`, and any live-path test that the change touches;
- the sentence "no code changes were made", and the workspace left clean.

### The fix comment

- a **verdict per finding**: as filed, bigger, or does not hold, each with the reproduction's own
  output;
- the fix, and the reasoning for the shape chosen, including the alternatives rejected and why.
  #115 round 6 states that a fenced publish would be stronger, that POSIX offers no conditional
  rename to build it from, and that the liveness fence is therefore the shape actually available;
- the **sweep**: every other site of the same class, enumerated with its result, so the reviewer
  can check a claim instead of hunting the next member;
- the tests, each named with the finding it closes and the mutation it fails under;
- **verification numbers**: suite before and after, at that head, plus lint and types;
- what changed in the artifact: the env README, the backend contract, the PR body's round
  summary. A decision that has been made stops being described as open (#115 round 12).

## Reproduce first

**Every finding is reproduced exactly as probed, before any code changes.** Not approximately, and
not by reading the code and agreeing. Three things come out of this rule:

- The reproduction becomes the regression test. #115 round 5's probe (an archive whose judge is
  replaced with `raise SystemExit("tampered judge")`, served under the pinned hash) is now a test.
- Reproducing catches a test that would have proved nothing. #115 round 4's first attempt passed
  against the buggy code, because the gate sat before the fetch, so the loser observed the
  destination only after the winner had published. Moving the gate between the observation and the
  replace reproduced the defect exactly. A test written from the description rather than from a
  reproduction would have shipped green and empty.
- A finding that does not reproduce is a result, not a gap. #112's second issue does not
  reproduce: the invariant probe still fires, and the consequence drawn from it does not follow,
  because a dispatch on that lease reaches the episode zero times at every event-loop yield. The
  answer is a test pinning the shape that genuinely was uncovered, plus a written account of which
  commit closed the rest. #113 answers an issue asking for deleted coverage to be restored by
  showing both properties are covered at the tip and proving the covering tests load-bearing with
  mutations, then restoring nothing.

## Fail first, and mutation evidence

**Every new test fails before the fix**, and the comment says on which head and how many times.
Race tests report a count (`3/3` against the pre-fix head, `5/5` after) because a race test that
passes for scheduling reasons is worse than no test.

**Every fix hunk carries a mutation entry.** Revert the hunk with its test in place, the test
fails; restore, and the file is verified byte-identical by `sha256`. #114 carries the whole list
in each round's comment and it grows with the branch: 11 hunks in the PR body, then 15, 19, 21,
25, 27, 26, 30, 35, 41, 44, 47, 49, 54.

**A mutation that does not fail is the finding.** It means the test is vacuous or the hunk is
decoration, and both are reported:

- #115 round 7's first ABA test passed against the buggy code on some runs. Its interleaving was
  scheduling-dependent: nothing forced the second waiter to read the doomed generation before the
  first cleared it. The mutation check caught it by not failing. The replacement uses a barrier so
  neither waiter acts until both hold a proof about the same generation, which is the only
  arrangement in which the second proof can go stale.
- #114 round three found that reverting an earlier round's inner containment no longer changed any
  observable outcome, because a guard added since caught it either way. The hunk was not kept on
  the strength of a mutation that had stopped biting: it was re-pinned against the property it
  actually preserves, with a new test.

**A test can also be vacuous in a way no mutation catches, and then the reviewer finds it.** A
mutation only asks whether the test notices the fix being removed; it cannot notice that the test
arranged the easy ordering. Twice in #114 the reviewer caught exactly that, and both times the
round agreed in writing rather than patching around it. Round twelve's finding was that the
previous round's test released its blocker and polled to a closed state before calling shutdown,
so it proved that the background release makes progress and said nothing about shutdown waiting
for it. Round thirteen's was that the round-twelve test passed only because it kept the failing
cleanup pending until shutdown was already joining it: "the same fault stopped the run when it was
slow and vanished when it was quick, which is not a policy". Both replacements pin both orderings
as two arms of one test.

**Entries retire explicitly, with reasoning, or not at all.** #114 retired two: a widened catch
whose only reachable distinguishing input had been removed one layer up, and a comparison that a
later flattening made unreachable. Both were disclosed in the round that retired them ("this is me
saying so"), the hunk count went down rather than being quietly held flat, and both retirements
had the same cause: a fix upstream removed the reachable input for a guard downstream. A coverage
change on `main` rather than on the branch is disclosed the same way even though no entry retires
for it.

**A refusal test always has a companion asserting the gate is not a wall.** Every round of #114
that added a refusal added a test that the legal shapes still open, still advertise the same
markers, and still seal on the tool they advertised. A gate that refuses everything passes every
test written about what it refuses.

**Tests are retargeted, not quietly dropped, when a fix closes a vector at its source.** Twice in
#114 a test that proved the stream contained a hostile env value stopped being reachable, because
the value was flattened at the episode instead. Both were renamed to assert the better outcome
(the record is now the seal the agent earned rather than an unscored infrastructure failure), and
the round's comment says which of the two outcomes is better and why.

## Class tables

When a finding is one member of a finite class, enumerate the class in the PR comment. This is
the single highest-leverage artifact in these threads: it converts "here is another one" into a
claim the reviewer can check, and it makes the next member's absence visible.

#114 opened two in round five, when three findings from three sites turned out to be members of
two classes, and has maintained both since.

**Class A: every pre-dispense use of the env's published contract lands in the stop path.** One
row per site, with the env-reachable code, the classification, and the test:

| # | site | env-reachable code | classified as | test |
|---|---|---|---|---|
| 2 | `open_env` reaching `env.load_task` | task selection | dispense-local **by definition**: this task's record may be the only bad one | `..._does_not_replace_the_setup_failure` |
| 4 | `open_env` reaching `env.describe` | publishing the contract | **contract refusal, stop the run** | `[the contract cannot be obtained at all]` |

Nineteen rows by round eleven. The line the class turns on is stated once, above the table: the
contract is env-wide, so failing to publish, read, copy or honour it will fail identically for
every later task from that env, while a task and its session are not. Six of the nine contract
rows are asserted by one parametrized test.

**Class B: the terminal-call state machine, and what leaves a check by reference.** A numbered
table of every step between the seal and the finalization claim, with what is env-reachable at
each, and a second table of every object handed to env code with a column for whether the core
reads it back afterwards.

Rules for maintaining them:

- **Amend rows; do not rewrite the table.** Rounds six, eight, nine, ten, eleven, twelve and
  fourteen each add or amend rows and say "rows 1 to 13 unchanged". The table is a running claim
  about a surface, so its history is part of what it asserts.
- **State a reversal plainly, with the reason the original clearance was wrong.** #114 cleared the
  feedback items as safe by reference in round six and reversed that in round eight; it cleared the
  terminal `args` in round six and reversed that in round fourteen. The reversed rows are kept in
  the table with the old justification struck through and the new verdict beside it. Both
  reversals had one cause, and the round that found the second said so: the justification counted
  what is written down and not what the module reads back itself. A justification of that shape is
  now suspect on sight.
- **Say what the sweep cleared, not only what it found.** #114 round four's sweep table has four
  rows, two of them live and fixed, two clear with the reasoning attached "so the next reader does
  not have to re-derive it". #115 round 12 does the same for every value its package hands out,
  naming the frozen dataclasses, the tuples and the per-call constructions individually.
- **A row can be marked unreachable rather than tested**, if the comment says why, and the test
  that would have covered it injects the condition directly instead. #114's row for a reader-side
  copy became unreachable once the snapshot held only plain data, so the test that pins the
  ordering rule now injects the refusal, because the rule is what is under test and it has to hold
  for whichever site raises there next.

## The rules, and the incidents that minted them

Each of these started as a single finding, recurred, and became a rule the sweeps are run against.

**One reading per decision.** A value read twice is two values. #115 round 4 recorded a judge
endpoint resolved at grading time against one equipped at session start; round 13 found the same
confusion one layer down, where the endpoint id and the verifier's environment were derived from
two separate reads of the same variable. The fix both times: read once, freeze, derive everything
from the frozen object. An episode takes one environment snapshot; the dataset pin is one
manifest, read once and handed out read-only (#115 round 12).

**A check whose subject can be swapped after the check is not a check.** #115 round 15: sealing an
episode stops the agent's tool calls, not the processes it already started, so validating the
report and letting the verifier reopen the same path grades whatever is there the second time. The
report is captured once at the seal through a single descriptor, and the verifier is given exactly
those bytes.

**Never check and then reuse: the rendering is what is retained.** A predicate that answers "can
this be serialized" and then hands the original object onward is a second walk with no guarantee
attached. #114 round three ("a second walk is a second question") and round four deleted the
predicate entirely: the wire helper returns the product, and the product is what the transaction
carries.

**Nothing foreign holds a reference between a check and its commit.** "The value checked is the
value committed" only holds if nothing else can reach it in between. #114 round six: the verifier
was handed the same evidence object the commit would read, and could rewrite the status and the
core-stamped provenance after their boundary checks. Round fourteen: the finalizer could rewrite
the submission that the terminal trajectory shows the verifier, while the durable digest still
witnessed the original.

**Detachment comes from a rebuild out of plain data, never from a foreign copy that did not
raise.** #114 round four found a contract whose `model_copy` returned a different contract on
every call without ever raising, so containment answered nothing. Round seven removed the copy:
detachment is now a property of having rebuilt the object from rendered values, rather than an
inference from a successful call into env code.

**A comparison never dispatches on a foreign value.** #114 round four: an object whose `__eq__`
returns true for `"ok"` is accepted as a declared status. No arrangement of the operands fixes it,
since a `str` subclass is offered the reflected comparison first. The value has to be a declared
string, checked by type before any comparison, and what is kept is the core's own constant.

**Rendering a document is not validating it.** #114 round ten: an advertised tool schema was
proved to be JSON and never proved to be a schema, so an agent's correct answer came back as its
own mistake. The rebuilt schema is now checked with the validator the seal itself will use, and
its root is required to permit an object, because that is what a call is.

**Classify by cause, and classify before anything cancellable.** #115 round 14: the upstream
verifier reads the agent's report inside the same handler as the judge call, so a missing report
produced exactly the payload a dead judge does, and this port filters judge errors out of results.
An agent that never wrote a report converted a failed attempt into no attempt at all. The class is
decided by cause now, before the verifier runs: anything wrong with the agent's own artifact is an
ordinary verified zero, only grading infrastructure is an exclusion. #114 round six is the
ordering half of the same rule: a classification of something already discovered runs before
anything that can be cancelled, never after it, because a caller's cancellation may end the
caller's call and may not unfind a fault.

**A lease bounds patience, not permission.** #115 rounds 5 to 7, three rounds on one lock. A
wall-clock lease cannot fence a stalled filesystem operation, so a timeout is not authority to
take another holder's lock. Breaking now requires proof that the holder is dead (same host, pid
gone), a live or foreign or unreadable holder is never broken, and the waiter fails closed with an
actionable error. The wrong guess in the safe direction costs an error message; the wrong guess in
the other direction costs a reader the tree it was handed.

**Act on the generation, never on the path.** #115 round 7, the third round on that lock: two
waiters proved the same holder dead, one cleared it and acquired, and the other spent its stale
proof on the path and evicted the live successor. Recovery is now conditional on the exact
generation that was proved dead. A proof is about a holder, so anything that acts on "whatever is
there now" discards it.

**A release that is the only owner of what it releases may not be endable by whoever is waiting
for it, and the run may not report itself over until every such release has been waited for.**
Assembled over four rounds of #114 (ten, eleven, twelve, fourteen): claim the cleanup in its own
task, join it under a shield, drain every claimed cleanup at shutdown until a pass finds none, and
do not let a shutdown that has already completed make a later claim invisible.

**The transaction is total, with or without an owner above it.** #114 round ten: a serializer
failure was left to the stream deliberately, and that policy is unchanged, but the record
underneath it is the episode's own and `evaluate()` drives the same class with no stream at all.
The record now commits fail closed and attributed before the failure is handed on.

**An address is not an authentication.** #115 rounds 5, 9, 10 and 11, four rounds on one pin. A
revision hash used only to build a URL pins nothing: the cache holds whatever answered the
request. Authenticating each archive is not the same claim as authenticating the task set, proving
every pinned task present is not the same claim as the cache being an exact set, and hashing on
the cold path says nothing about the warm one. Each of those four was a separate round.

**Redaction has a before and an after, and they are different claims.** #115 rounds 8 and 9: the
public verdict carried the ground-truth control label that `describe()` withholds, and removing
the key did not close it, because a passing control still shows a score relation no incident task
can produce. The scoring rule stayed aligned with upstream and the disclosure was documented,
pinned by a test, and signed off by the owner in the thread rather than decided inside the fix.

## Working notes

Cheap rules, each of which cost real time at least once.

- **Assert a replacement pattern matches exactly once before substituting.** #115 round 10: a
  substitution silently matched nothing and the intended refusal was never wired. Round 11: twice
  more, one landing in the wrong test. The suite caught all three, which is luck, not method.
- **Never pipe a long test run through `tail` or `head`.** The buffering reads as a hang. Run the
  suite in the background and poll its output file.
- **Expect the full suite to stall at a live verifier.** Several rounds of both #114 and #115
  report interrupting a broad run at around 142 passed while it waited inside an unrelated live
  Docker verifier. Run the PR-specific module first, then `-m "not network"` for the offline
  suite, and report which one produced the numbers quoted.
- **Do not monkeypatch `os.environ` wholesale.** #115 round 13's first attempt replaced it with a
  stand-in mapping and broke pytest's own machinery for the rest of the run (97 errors). Patch the
  seam the code reads through instead, and assert the property directly. The failure was loud
  here; the same instinct applied somewhere quieter would not have been.
- **A test for an uncancellable blocking syscall needs a daemon-thread guard.** #115 round 16: a
  blocking `open` cannot be cancelled from the caller, and `asyncio.to_thread` leaves a
  non-daemon executor thread stuck in the syscall, so the process hangs at interpreter exit and
  the regression check becomes the hang it is testing for. A daemon thread with a bounded `join`
  fails the mutation in about eleven seconds and exits cleanly.
- **Pin a race with a barrier, never with a sleep**, and write the assertion so it does not depend
  on who wins: whatever tree first became complete was handed to a reader, so the reader's file
  must still be there at the end (#115 round 4). The hold is a timeout rather than an assertion
  precisely because a correct implementation makes the interleaving impossible.
- **Measure before choosing between the cheap-and-arguable and the expensive-and-airtight.** #115
  round 11 timed a full re-hash of the 755-task cache at 3.07s serially and 1.56s in parallel, so
  the airtight option shipped and no completion marker or mtime scheme had to be defended.
- **Verify a platform claim rather than citing it.** #115 round 16 confirmed empirically that
  `O_NONBLOCK` has no effect on a regular file, then pinned it with a test that captures a report
  at the size bound, so the property is checked on both platforms the suite runs on.
- **Re-run the live-path test whenever its path changed, and say so when it was not re-run.**
  Several of #115's later rounds record "the network test was not re-run: this round touches
  neither the fetch nor the publish path", and every round that moved the fetch, publish or warm
  path re-ran it rather than assuming it.

## When to stop

**Each round's defense is the next round's attack surface.** #115's publish lock took four rounds
(4, 5, 6, 7): the lock introduced to close a race, then its ownership token, then its liveness
fence, then the generation the recovery acts on. #114's cleanup took five (10 through 14): the
claimed release task, then the stream's own, then the shutdown drain, then the discard in the done
callback, then the memoized release that hid later claims. The mechanism is not the mistake; each
one was necessary. The cost is real, and it is the reason the stopping rule is explicit rather
than "when it looks done".

**Stop when a round comes back clean.** #112's re-review reproduced nothing new at the exact head,
re-ran the focused suites, `ruff` and `pyright` there, and said so; the PR merged on it. #113's
single review found nothing at all, blocking or otherwise. A clean round is a stopping rule, not a
proof that nothing is left.

**Or stop when the findings no longer touch the surface under review.** #115 merged after round
16, in which the last three rounds were about a report capture belonging to a phase the PR does
not implement. The residue was redirected rather than dropped: what landed was the backend
contract stating capture-once as a requirement with its reason, the plumbing that makes the
verifier unreachable without a capture, and the tests that pin both, so the implementation cannot
quietly skip it when it arrives. A finding that will become live somewhere else belongs there, in
a form that fails if it is ignored.

**Owner calls are named and signed off in the thread**, not decided inside a fix. #115 round 8
named a residual it did not close (a metric key upstream leaves undefined for control tasks, so
its presence implies the class) and offered to take the owner's call either way; round 9's
class-dependent scoring rule was flagged as provisional, signed off in the thread during round 10,
and the README and PR body stopped calling it open in round 12, which is itself a finding the
reviewer filed.

**A round that only finds smaller things is not automatically the last one.** #114 is open at
round fifteen, and the finding there is a valid integer feedback value silently coerced to a float
between two sinks, which is a score-corruption path found after fourteen rounds of hardening the
same two files.
