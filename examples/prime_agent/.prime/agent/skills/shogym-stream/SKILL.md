---
name: shogym-stream
description: Play shogym evaluation tasks served over MCP. Pull work with `pull`, complete it with the tools the task gave you, end it with the tool that ends it, and pull again until a record says done. Use whenever you are asked to work a shogym task or a shogym task server.
---

# shogym-stream

One MCP endpoint that hands out work a task at a time. Every tool is an `async` method on the
`shogym_stream` module, and every one of them returns a **JSON string**: the server sends text
content, so parse before you index:

```python
import json

record = json.loads(await shogym_stream.pull())
```

## The loop

1. `pull()` takes no arguments and answers with one record. Its `kind` says what it is:
   - `"task"`: work to do. It carries `attempt_id` and `body`, the instructions.
   - `"payload"`: content delivered against the task you just ended. Read it, then pull again.
   - `"wait"`: nothing is ready yet. Pull again after `retry_after_ms`.
   - `"done"`: nothing more is coming. Stop.
2. Call the task's tools with the attempt the task named:
   `await shogym_stream.<tool>(attempt_id=..., arguments={...the tool's own arguments...})`.
   The wrapper is closed, so a call missing either field is refused, and so is one naming an
   attempt you are not working on.
3. One of those tools ends the task. It answers `seal_ack` when the filing is accepted, or
   `seal_reject` when the arguments were malformed, which leaves the task open to file again.
4. Go back to step 1, and stop on `done`.

The tools bound on this module are the whole affordance set, and there is no queue to inspect:
the server serves one task per launch.

## Failures

- `NotEnabled` means no bearer token. Its message tells you to run `/mcp login shogym`: **that
  is wrong here** and will report an unknown integration. This server authenticates with a
  static token, so tell the user to export `SHOGYM_MCP_TOKEN` (any non-empty value) and restart
  Prime Agent.
- A connection error means `serve.py` is not running. Tell the user; do not try to start it,
  and do not go looking for the task's answer on disk.
- `McpToolError` is the server rejecting a call. It carries a protocol record whose `code` says
  why: `invalid_message` is a malformed wrapper, `invalid_attempt` is the wrong attempt,
  `overlapping_call` means a call of yours is still running, and `closed_stream` means the
  generation is over. Read it and fix the call.
