"""Retrieval backends for the ``browsecomp_plus`` env — the ``search`` / ``get_document`` seam.

BrowseComp-Plus factors its retriever behind a ``BaseSearcher`` interface (``searcher/searchers/``)
so the corpus/index is swappable (BM25/Lucene, dense Faiss, …). shogym reuses that seam:

- :class:`Searcher` — the injectable protocol (``search(query, k)`` + ``get_document(docid)``),
  mirroring upstream's ``BaseSearcher``. The registered env pins **BM25** (CPU/Java-only);
  offline tests inject an :class:`InMemorySearcher` over a tiny synthetic corpus.
- :class:`InMemorySearcher` — a dependency-free, deterministic lexical searcher (token-overlap
  scoring) for fixtures and unit tests: **no Java, no pyserini, no network**.
- :class:`BM25Searcher` — the faithful pyserini/Lucene backend (``LuceneSearcher``), ported from
  upstream. It imports ``pyserini`` lazily (needs **Java 21**), so importing this module stays
  light; it is only constructed when the real env is served against a prebuilt BM25 index.

A result is ``{"docid": str, "score": float | None, "text": str}``; the served ``search`` tool
snippets ``text`` and drops ``text`` for the wire (see :mod:`mcp_server`).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@runtime_checkable
class Searcher(Protocol):
    """A retrieval backend: rank the corpus for ``query`` and fetch a document by id.

    Mirrors BrowseComp-Plus's ``BaseSearcher``. ``search`` returns a list of
    ``{"docid", "score", "text"}`` (best first); ``get_document`` returns ``{"docid", "text"}``
    or ``None``. ``search_type`` names the backend (``"BM25"``, ``"in_memory"``, …) for the
    tool description and the TaskSpec (retriever fidelity is only comparable at a fixed backend).
    """

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]: ...

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]: ...

    @property
    def search_type(self) -> str: ...


_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> List[str]:
    return _WORD_RE.findall(text.lower())


class InMemorySearcher:
    """A deterministic, dependency-free lexical searcher over an in-memory corpus.

    For fixtures and offline tests only — **not** a faithful BM25. Scores each document by the
    number of distinct query terms it contains (ties broken by total term frequency, then docid
    for stability), which is enough to exercise the served ``search`` / ``get_document`` tools and
    the retrieval/citation metrics without Java, pyserini, or the 2.78 GB corpus.

    ``corpus`` is a mapping of ``docid -> text`` (docids are compared as strings).
    """

    def __init__(self, corpus: Dict[str, str]) -> None:
        self._corpus: Dict[str, str] = {str(k): str(v) for k, v in corpus.items()}
        self._doc_tokens: Dict[str, List[str]] = {
            docid: _tokens(text) for docid, text in self._corpus.items()
        }

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        q_terms = set(_tokens(query))
        scored: List[tuple] = []
        for docid, toks in self._doc_tokens.items():
            distinct_hits = len(q_terms & set(toks))
            if distinct_hits == 0:
                continue
            total_hits = sum(1 for t in toks if t in q_terms)
            # Sort key: more distinct query terms first, then total frequency; docid ascending
            # (as int when possible) breaks ties deterministically.
            scored.append((distinct_hits, total_hits, _docid_sort_key(docid), docid))
        scored.sort(key=lambda s: (-s[0], -s[1], s[2]))
        results: List[Dict[str, Any]] = []
        for distinct_hits, _total, _key, docid in scored[: max(0, k)]:
            results.append(
                {"docid": docid, "score": float(distinct_hits), "text": self._corpus[docid]}
            )
        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        text = self._corpus.get(str(docid))
        if text is None:
            return None
        return {"docid": str(docid), "text": text}

    @property
    def search_type(self) -> str:
        return "in_memory"


def _docid_sort_key(docid: str) -> tuple:
    """Sort docids numerically when they are integers (the BrowseComp-Plus corpus uses integer
    ids), else lexically — so tie-breaking is stable and human-sensible."""
    try:
        return (0, int(docid))
    except (TypeError, ValueError):
        return (1, docid)


class BM25Searcher:
    """The faithful BM25/Lucene backend, ported from BrowseComp-Plus ``BM25Searcher``.

    Wraps pyserini's ``LuceneSearcher`` over a prebuilt Lucene index. Requires **Java 21** and
    ``pyserini`` (the ``browsecomp_plus`` extra); ``pyserini`` is imported lazily in ``__init__``
    so merely importing this module needs neither. Construct it with the local path to a prebuilt
    BrowseComp-Plus BM25 index (see :mod:`data` for lazy provisioning)."""

    def __init__(self, index_path: str) -> None:
        if not index_path:
            raise ValueError("index_path is required for BM25Searcher")
        try:
            from pyserini.search.lucene import LuceneSearcher
        except ImportError as exc:
            raise RuntimeError(
                "the `browsecomp_plus` extra (pyserini) and a Java 21 runtime are required for "
                "the BM25 backend — install it with `pip install 'shogym[browsecomp_plus]'` and a "
                "JDK 21. Offline tests inject an InMemorySearcher instead and need neither."
            ) from exc
        self._index_path = index_path
        try:
            self._searcher = LuceneSearcher(index_path)
        except Exception as exc:
            raise ValueError(
                f"'{index_path}' is not a valid local Lucene index path."
            ) from exc

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        import json

        hits = self._searcher.search(query, k)
        results: List[Dict[str, Any]] = []
        for hit in hits:
            raw = json.loads(hit.lucene_document.get("raw"))
            results.append({"docid": hit.docid, "score": float(hit.score), "text": raw["contents"]})
        return results

    def get_document(self, docid: str) -> Optional[Dict[str, Any]]:
        import json

        doc = self._searcher.doc(str(docid))
        if doc is None:
            return None
        return {"docid": str(docid), "text": json.loads(doc.raw())["contents"]}

    @property
    def search_type(self) -> str:
        return "BM25"
