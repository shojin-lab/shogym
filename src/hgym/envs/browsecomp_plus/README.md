# `browsecomp_plus` — BrowseComp-Plus, a Deep-Research retrieval env ("HLE with a fixed corpus")

A faithful hgym port of [**BrowseComp-Plus**](https://github.com/texttron/BrowseComp-Plus) (ACL
2026) — answer OpenAI BrowseComp's reasoning-heavy queries against a **fixed, human-verified
~100K-doc corpus** served as `search` / `get_document` tools, instead of the live web. Freezing
the corpus isolates search + reasoning from web noise and makes runs reproducible. `submit_answer`
is the **score terminal**: submitting seals the episode and the env's `finalize` hook grades the
sealed answer with an LLM judge (as in the [HLE](../hle/README.md) port), and the env adds
deterministic **retrieval-recall** and **citation** metrics computed purely off the recorded
trajectory against the query's relevance judgements (qrels) — so it exercises hgym's verification
surface with both a model judge *and* deterministic retrieval metrics.

Like every hgym env this **describes** a task, **serves** its tools over MCP, and **verifies** a
recorded trajectory while an external harness drives the tools — see
[`../README.md`](../README.md). The runnable demo is
[`examples/quickstarts/`](../../../../examples/quickstarts/).

## Running it

> Requires **Python 3.12 + the `browsecomp_plus` extra**, an OpenAI key for the judge, Hugging
> Face access to the (encrypted) query dataset, and a **Java 21** runtime (pyserini/Lucene) — the
> prebuilt **BM25 index auto-downloads once** on first served use. See
> [Requirements](#requirements). Offline tests need none of it.

### Construct + serve

```python
import hgym

env = hgym.make("browsecomp_plus")     # train split; decrypts queries in memory, loads BM25 index
spec = env.describe("0")                # task 0: the query + tool manifest
```

Serve it as a stdio MCP server any harness can drive:

```bash
export OPENAI_API_KEY=sk-...    # the judge is an OpenAI model
# The prebuilt BM25 Lucene index auto-downloads once to ~/.cache/hgym on first served use
# (needs Java 21). Set HGYM_BROWSECOMP_PLUS_BM25_INDEX only to reuse a pre-provisioned index.
uv run python -m hgym.cli serve browsecomp_plus --task 0 --trace ./hgym_logs/bcp.jsonl
```

The harness reads the query via `describe`, calls **`search`** / **`get_document`** to gather
evidence, then calls **`submit_answer`** — the score terminal that seals the episode, grades the
sealed answer in `finalize` (the LLM judge), and ends the episode in one call (no separate
`terminate`). See the shared
[terminal lifecycle](../README.md#terminal-lifecycle-seal-terminal-score-terminal-abort). hgym
reads the score off the trace via `hgym.result_from_trace(...)`.

**Config** (via `hgym.make(name, config)` / `env_config`): `task_split` (`"train"`/`"test"`),
`tasks` (an explicit task list — bypasses the dataset/decryption, used by offline tests),
`searcher` (an injected [`Searcher`](searcher.py) — an `InMemorySearcher` for offline runs),
`judge` (an injected [`Judge`](judge.py) — a scripted judge for offline runs), `judge_model` /
`judge_base_url` (the default judge's model + endpoint), `k` / `snippet_max_tokens` (retrieval
knobs; upstream defaults 5 / 512), and `max_turns` (the tool-call horizon).

### Quickstart

Any quickstart under [`examples/quickstarts/`](../../../../examples/quickstarts/) serves this env: one MCP endpoint
hands out a queue of tasks and scores each one server-side. Point it here with the single
variable at the top of its `serve.py`:

```python
ENV = "browsecomp_plus"
```

The default judge is model-graded, so this needs `OPENAI_API_KEY` set in the environment
the server runs in.

## Requirements

The Python pin and the `uv sync` / `pip install` / `import hgym` mechanics are the shared
[requirements boilerplate](../README.md#requirements-boilerplate). The `browsecomp_plus` extra
pulls `datasets` (the encrypted queries), `openai` (the default judge client), and `pyserini`
(the BM25/Lucene retriever). On top of that:

- **Java 21.** pyserini's BM25 backend runs on Lucene (JVM). Without a JDK 21 the real retriever
  can't run; offline tests use an in-memory searcher and need no Java.
- **`OPENAI_API_KEY`.** `submit_answer` grades with an LLM judge. With the default judge, an
  episode **fails fast at startup** if no key is set (so a keyless run never silently scores
  everything wrong). Opt out by injecting a scripted `judge`, or point `judge_base_url` at a
  keyless OpenAI-compatible endpoint (e.g. a vLLM **Qwen3-32B** — the upstream judge).
- **The query dataset (`Tevatron/browsecomp-plus`).** Queries + answers are **XOR-encrypted**
  with a canary to keep the benchmark off plain-text crawls. hgym decrypts them **in memory
  only** — it never writes or commits decrypted queries/answers, and the canary is preserved.
  The dataset downloads once to `~/.cache/hgym/browsecomp_plus` (honor `HF_HOME` or
  `HGYM_BROWSECOMP_PLUS_DATA_DIR`). The per-query qrels are lazy-downloaded from the pinned
  upstream commit and cached alongside.
- **The prebuilt BM25 index (~2.78 GB).** **Auto-downloads once** to
  `~/.cache/hgym/browsecomp_plus/bm25/` on first real (served) use — from the upstream HF *dataset*
  repo `Tevatron/browsecomp-plus-indexes` (`bm25/*`), the same source as upstream's
  `scripts_build_index/download_indexes.sh`, provisioned lazily like the queries/qrels. The
  genuine prerequisites are **Java 21 + HF access + ~disk space**. `HGYM_BROWSECOMP_PLUS_BM25_INDEX`
  remains an optional override to reuse a pre-provisioned index (used verbatim).

## How it works

### describe → TaskSpec

`env.describe(task_id)` publishes the task contract: the **query** (in `instructions`, with the
retriever named so the score is only compared at a fixed backend) and the **tool manifest** —
`search`, `get_document`, `submit_answer` (all `env-mandatory`) and the reserved `terminate`.

### Tools (served over MCP)

Backed by the in-process server in `mcp_server.py` (hgym's near-verbatim reuse of upstream's
`searcher/mcp_server.py` surface):

- **`search(query)`** — rank the fixed corpus and return the top-k hits (`docid`, `score`,
  `snippet`). The docids returned across a run are what the verifier reads back as
  `retrieved_docids`.
- **`get_document(docid)`** — fetch a document's full text (upstream's optional tool).
- **`submit_answer(answer, confidence)`** — the **score terminal**. The serve layer validates
  its args, atomically **seals** the episode, then runs the env's `finalize` hook — so its handler
  body is never dispatched inward. The **judge runs in `finalize`**: it grades `answer` against
  the session's gold answer (BrowseComp-Plus's own `GRADER_TEMPLATE`, temp 0 for determinism) and
  returns core-owned, **sanitized** `TerminalEvidence`. Submitting ends the episode; single
  submission is structural (a second `submit_answer` is tombstoned).
- **`terminate()`** — the reserved `abort` terminal (ending without a submission scores
  `correct=False`).

The env pushes the (injectable) **searcher** into the in-process server via `begin_session`; the
query, its gold answer, and the (injectable) **judge** are held on the env for `finalize` (grading
runs after the seal, not in a served handler). State is keyed by session id and dropped via
`end_session`; the searcher is read-only and shared across episodes.

### finalize + verify

`finalize` runs the LLM judge on the **already-sealed** submission and returns core-owned
`TerminalEvidence`. It is **sanitized**: the public verdict carries only `correct` / `confidence` /
`judge_error` — the judge's `reasoning` / `extracted_answer` (answer oracles) and any exception
text go to a private diagnostic, never to the agent. A judge failure **fails closed** to
`correct=False` with `judge_error=True`.

`verify` is a **pure** function over the recorded trajectory + the terminal evidence + the task's
qrels. `score_trajectory` reads:

- **`correct`** — the judge's verdict, off the core-owned `evidence.verdict` (never a marker in a
  tool result the agent can forge). A judge / finalize failure is flagged `judge_error=True`.
- **`confidence`** (0–1) + **`calibration_error`** (`|confidence − correct|`) — HLE-style, read
  from the validated submission (`evidence.args`); omitted on a horizon/abort end.
- **`retrieval_recall`** — fraction of the query's evidence docids (`qrel_evidence`) retrieved
  across all `search` steps (BrowseComp-Plus's retrieval recall).
- **`citation_recall` / `citation_precision` / `num_citations`** — cited-docid metrics vs
  `qrel_evidence` (citations parsed from the submitted answer as `[docid]`).

Deterministic metrics are emitted whenever the query has evidence qrels, even on a premature end
(a run that retrieved well but never answered still gets a recall score). The episode terminates
when `submit_answer` **seals** it, or `terminate` aborts it, or the horizon (`max_turns`, default
50) is reached with no submission (`zero_unsubmitted` → `correct=False`).

## Tasks

`browsecomp_plus` loads the queries from `Tevatron/browsecomp-plus` (split `test`), decrypts them
**in memory**, joins each with its `qrel_golds` / `qrel_evidence` relevance judgements, and slices
the set **positionally 80/20** into `train` / `test`, like the HLE/wordle ports. It defaults to
`train`; pass `config={"task_split": "test"}` for the held-out set. Task indices are split-relative.

## Scoring

The episode scores ride out on the terminal result's `_meta` sidecar and the terminal trace row.
Read the score back with:

```python
import hgym
result = hgym.result_from_trace("hgym_logs/bcp.jsonl", env="browsecomp_plus", task="0")
print(result.terminated, result.value("correct"), result.value("retrieval_recall"))
```

`result_from_trace` treats `env` / `task` / `session_id` as **filters** — see
[Reading a score back](../README.md#reading-a-score-back-result_from_trace) for the shared
semantics (give each run its own trace file for a guaranteed 1:1 mapping).

## Fidelity & deviations

- **Upstream pin.** Upstream commit `0469490` (MIT). The auto-downloaded BM25 index is pinned to
  an immutable commit of `Tevatron/browsecomp-plus-indexes` (`INDEX_REVISION` in `data.py`), so a
  cold-cache download is reproducible and can't drift with an upstream index replacement.
- **Retriever pinned to BM25.** The retriever materially changes scores (BM25 vs dense
  Qwen3-Embedding), so the backend is named in the TaskSpec and this first cut pins **BM25**
  (CPU/Java-only; the dense path needs Faiss + GPU — a deferred follow-up).
- **Judge.** Pinned to temp 0 for determinism; upstream reports GPT-4.1 (paper) / Qwen3-32B
  (vLLM) — hgym defaults to GPT-4.1, overridable via `judge_model` / `judge_base_url`. A judge
  failure fails closed to `correct=False` with `judge_error=True`.
- **Queries stay encrypted at rest.** The XOR/canary decryption happens only in memory at load;
  hgym never writes or commits decrypted queries/answers. The canary constant is preserved.

## Gotchas

- **The index is heavy but auto-provisioned.** ~2.78 GB Lucene index (Java 21); it auto-downloads
  once to `~/.cache/hgym/browsecomp_plus/bm25/` on first served use (override with
  `HGYM_BROWSECOMP_PLUS_BM25_INDEX`). The Java check runs *before* that download, so a missing JVM
  fails fast without paying for it. Offline tests inject a tiny synthetic corpus (no Java, no
  download).
- **No web access.** The point of the fixed corpus is reproducibility — a harness must deny
  `WebFetch` / `WebSearch` (the example does), so retrieval goes only through the served tools.
- **Judge keying.** Like HLE, starting an episode without `OPENAI_API_KEY` (and no injected judge
  / `judge_base_url`) raises early rather than scoring everything wrong.
- **Offline vs keyed tests.** Follows the shared
  [offline-vs-keyed split](../README.md#tests-offline-vs-keyed): the pure verifier / metric /
  judge-parser / decryption tests and the served in-memory-searcher + scripted-judge test run
  offline; the keyed fidelity test (the real judge grading a served episode) is skipped unless
  `OPENAI_API_KEY` is set.

## Layout

- `env_v1.py` — the registered env: `describe` (query + manifest), session wiring, the `finalize`
  hook (LLM judge on the sealed submission), and the pure `score_trajectory` verifier (verdict from
  the terminal evidence + retrieval/citation metrics).
- `mcp_server.py` — the in-process MCP server: `search` / `get_document` / `submit_answer` (the
  score terminal, never dispatched inward once sealed).
- `searcher.py` — the `Searcher` seam: `InMemorySearcher` (fixtures) + `BM25Searcher` (pyserini).
- `judge.py` — the LLM judge (`GRADER_TEMPLATE` + `parse_judge_response`, upstream verbatim).
- `metrics.py` — pure retrieval-recall + citation precision/recall (upstream verbatim).
- `data.py` — in-memory decryption (canary preserved), qrel + lazy BM25-index auto-download, and
  the Java-21 fast-check (before any multi-GB download).
</content>
