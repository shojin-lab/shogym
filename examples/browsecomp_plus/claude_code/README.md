# Claude Code plays BrowseComp-Plus (through a served hgym env)

The end-to-end demo of the env-as-center design (RFC 008) on the BrowseComp-Plus port (issue
#43): hgym serves one Deep-Research question — with `search` / `get_document` tools over a
**fixed corpus** — as an MCP server, **Claude Code** answers it as an external harness, and hgym
scores it with a **model-graded judge** (correctness) plus deterministic **retrieval-recall** and
**citation** metrics off the trace. hgym never sees Claude Code's model, prompt, or loop — only
the tool calls and the verifier's feedback.

```
  Claude Code  ──spawns──▶  hgym serve browsecomp_plus          (stdio MCP server)
  (the harness)             ├─ describe               → the question + tools
      │                     ├─ search(query)          → top-k {docid, score, snippet}
      │                     ├─ get_document(docid)     → full text
      └──tool calls────────▶└─ submit_answer(a,c)      → seal → finalize (LLM judge) → verdict
                                     (score terminal)     → episode scored → ./hgym_logs/…jsonl
                                                                              │
                            hgym.result_from_trace(...) ◀────────────────────┘
```

## The flow: `search` → `submit_answer` (the score terminal)

1. Call **`describe`** to read the question.
2. Call **`search`** (and **`get_document`**) to gather evidence from the fixed corpus — there is
   **no web access**; retrieval goes only through these tools.
3. Call **`submit_answer`** once with the final `answer` (cite supporting docids as `[docid]`) and
   a `confidence` (0–100). This is the **score terminal**: it **seals** the episode, then the env's
   `finalize` hook grades the sealed submission with the LLM judge (seal-before-verdict) and hgym
   computes the retrieval/citation metrics into the terminal feedback. That one call ends the
   episode — there is no second submission and **no separate `terminate`** (a later call is
   tombstoned). An explicit `terminate` without submitting aborts the episode (scores `correct=False`).

## Prerequisites

- **Python 3.12 + the `browsecomp_plus` extra.** `uv sync` builds the 3.12 `.venv` with the extra
  (it's in the dev group). Confirm: `uv run python -c "import datasets, openai, pyserini; print('ok')"`.
- **Java 21.** pyserini's BM25 backend runs on Lucene (JVM). Install a JDK 21 and ensure
  `java -version` reports 21.
- **`OPENAI_API_KEY`** — the model-graded judge calls an OpenAI model (or point `judge_base_url`
  at a keyless vLLM Qwen3-32B, the upstream judge). `export OPENAI_API_KEY=sk-...`
- **Hugging Face access to `Tevatron/browsecomp-plus`.** The queries are XOR-encrypted; hgym
  decrypts them **in memory** (never persisted) and downloads once to
  `~/.cache/hgym/browsecomp_plus`. Please do not redistribute decrypted queries.
- **The prebuilt BM25 index (~2.78 GB).** **Auto-downloads once** to `~/.cache/hgym/browsecomp_plus/`
  on first served use (from the upstream HF dataset repo `Tevatron/browsecomp-plus-indexes`) — no
  manual step. Set `HGYM_BROWSECOMP_PLUS_BM25_INDEX=/path/to/index` only to reuse a
  pre-provisioned index. You need ~disk space + HF access for the one-time download.
- The [`claude`](https://www.anthropic.com/claude-code) CLI on your `PATH`, with credentials.

`run.py` runs a **preflight** that fails fast with the exact fix if the extra can't import, the
judge key is missing, or the env won't build — checking **Java 21** (the failure users actually
hit) *before* the multi-GB index/dataset downloads, rather than letting Claude connect to a
crashed, toolless server.

## Run it

```bash
export OPENAI_API_KEY=sk-...
uv run python examples/browsecomp_plus/claude_code/run.py --task 0

# print Claude Code's turn-by-turn reasoning and tool calls as it plays:
uv run python examples/browsecomp_plus/claude_code/run.py --task 0 --transcript

# pick the model / reasoning effort (defaults: claude-sonnet-5 / high):
uv run python examples/browsecomp_plus/claude_code/run.py --task 0 --model opus --effort max
```

The script writes a per-run `.mcp.json`, runs Claude Code, then prints the score (`correct`,
`retrieval_recall`, `citation_recall`) read back off the trace.

## Drive it by hand

Using the checked-in [`.mcp.json`](./.mcp.json) (task 0, trace at `./hgym_logs/browsecomp_plus.jsonl`) —
run from the repo root so `uv run` resolves this project's `.venv`:

```bash
claude -p "Answer the BrowseComp-Plus question: call describe, search for evidence, then submit_answer once with [docid] citations (that seals and grades your answer and ends the episode — do not call terminate)." \
  --mcp-config examples/browsecomp_plus/claude_code/.mcp.json \
  --strict-mcp-config \
  --allowedTools "mcp__browsecomp_plus__*" \
  --disallowedTools "WebFetch,WebSearch,Bash,Read,Write,Edit,Glob,Grep,Task,TodoWrite,NotebookEdit,BashOutput,KillShell" \
  --permission-mode dontAsk

# then read the score hgym recorded (judge + retrieval/citation metrics):
uv run python -c "import hgym; print(hgym.result_from_trace('hgym_logs/browsecomp_plus.jsonl'))"
```

Why these flags: `--strict-mcp-config` isolates the session to this one server;
`--allowedTools "mcp__browsecomp_plus__*"` pre-approves the env's tools; `--permission-mode dontAsk`
runs non-interactively by **denying** anything not pre-allowed; and `--disallowedTools "WebFetch,WebSearch,…"`
removes the built-ins so the agent can't reach the live web — BrowseComp-Plus scores retrieval
over the **fixed corpus** alone, so the score stays attributable to the served tool surface.

## Swapping the harness

Nothing here is Claude-specific. Point any MCP-speaking harness at the same
`hgym serve browsecomp_plus` server and the trace/score path is unchanged — hold `(env, task)`
fixed, swap the harness, and the delta in the trace is attributable to the harness.
