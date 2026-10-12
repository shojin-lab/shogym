# Review rules

## Procedure

1. Build the whole stack with no reviews.
   - Building every PR before reviewing any shows early where the design does not fit together.
   - Reviews then do not polish code that a later PR will change.
   - Each builder opens its PR ready for review.
   - Tests cover each thing the PR makes true with a few cases, about as many test lines as code lines.
   - Builders write lean comments from the start: what and why, in a few lines.
2. Launch one cold review per PR, bottom up.
   - The reviewer runs `codex exec -m gpt-6-astra -c model_reasoning_effort="ultra"`: maximum reasoning, with delegated agents at xhigh.
   - The brief gives the spec the PR must meet and names every other PR in the stack with one line of scope.
   - Fix a finding only if, in the runs we plan to do, it could change:
     - a grade;
     - an outcome row;
     - whether a chain goes on to its next step;
     - what the agent sees, such as the bytes of a task, a receipt or the prompt;
     - what the report records, such as a stop's cause or a rerun.
   - Show each finding with a test before fixing it, and answer a finding the test refutes with that evidence.
   - Note everything else in one line in the PR body.
   - Fixes add no new machinery, and a finding that needs some goes to the user as a proposal.
   - After the fixes, rebase every PR above it and push the PR branches, never main.
3. Launch a holistic review of the whole stack.
   - Make its fixes one at a time, branch by branch, from the bottom of the stack up.
   - After each change, rebase every PR above it and push.

## Guidance

- CI does the testing.
  - Tests run in CI, not locally, unless CI minutes run out.
  - After every push, CI must be green on each PR before work goes on.
  - Time CI on every push.
  - If CI passes 10 minutes, an efficiency PR goes into the stack.
  - It reviews what made CI slow, then fixes it by sharding or cutting cost, never by raising the timeout.
- Independent stacks may be built and reviewed side by side.
- Do not merge any PRs.
  - The user will perform merges.
  - After each merge, rebase the rest of the stack and push.
- Every update to the user gives each PR's size, in lines of code and lines of tests.
- A Codex review is finished when its review file is written.
  - Do not judge it by whether its process is still running.
