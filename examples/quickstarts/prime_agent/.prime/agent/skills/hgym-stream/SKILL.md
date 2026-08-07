---
name: hgym-stream
description: Play a stream of hgym evaluation tasks served over MCP. Pull a task with get_task, complete it with the tools that task lists, end it, and repeat until the stream is exhausted. Use whenever you are asked to work through a queue of hgym tasks or a task server.
---

# hgym-stream

A queue of evaluation tasks behind one MCP endpoint. Every tool is an `async` method on the
`hgym_stream` module, and every one of them returns a **JSON string**: the server sends text
content, so parse before you index:

```python
import json

task = json.loads(await hgym_stream.get_task())
```

## The loop

1. `get_task()` takes the next task off the queue. It answers either
   `{env, instructions, budget, tools}` or `{done: true, ...}`, and `done` means stop.
2. `tools` lists the tools that task published, by name and schema. They are methods on this
   same module: `await hgym_stream.<name>(**kwargs)`. Nothing else is available for the task.
3. One of them ends the episode. Call it when you are finished; then go back to step 1.
4. Stop when `get_task()` reports `done`.

Do not assume tool names between tasks; read `tools` each time. `queue_info()` reports
`{remaining, consumed, in_flight}` if you want to know how much is left.

## Failures

- `NotEnabled` means no bearer token. Its message tells you to run `/mcp login hgym`: **that
  is wrong here** and will report an unknown integration. This server authenticates with a
  static token, so tell the user to export `HGYM_MCP_TOKEN` (any non-empty value) and restart
  Prime Agent.
- A connection error means `serve.py` is not running. Tell the user; do not try to start it,
  and do not go looking for the task's answer on disk.
- `McpToolError` is the server rejecting a call. Read the message and fix the arguments.
