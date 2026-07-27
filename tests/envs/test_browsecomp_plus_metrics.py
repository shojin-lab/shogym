"""Unit tests for the deterministic retrieval + citation metrics (ported verbatim from upstream
``evaluate_run.py``). Pure, stdlib-only — run in the offline core suite.
"""

from __future__ import annotations

from hgym.envs.browsecomp_plus.metrics import (
    compute_citation_metrics,
    extract_citations_from_response,
    retrieval_recall,
)


def test_extract_single_and_multi_citations() -> None:
    assert set(extract_citations_from_response("The answer is [42].")) == {"42"}
    assert set(extract_citations_from_response("See [1, 2, 3] and [7].")) == {"1", "2", "3", "7"}


def test_extract_fullwidth_citations() -> None:
    # Upstream also recognises the full-width 【docid】 form some models were tuned on.
    assert set(extract_citations_from_response("evidence 【18639】 and 【5412, 82002】")) == {
        "18639",
        "5412",
        "82002",
    }


def test_extract_empty_and_no_citations() -> None:
    assert extract_citations_from_response("") == []
    assert extract_citations_from_response("no docids here") == []


def test_citation_precision_and_recall() -> None:
    m = compute_citation_metrics(["1", "2", "3"], ["2", "3", "4"])
    assert m["num_citations"] == 3.0
    assert m["num_relevant"] == 3.0
    assert abs(m["precision"] - 2 / 3) < 1e-9  # {2,3} cited & relevant / 3 cited
    assert abs(m["recall"] - 2 / 3) < 1e-9  # {2,3} / 3 relevant


def test_citation_metrics_no_citations_is_zero() -> None:
    m = compute_citation_metrics([], ["1", "2"])
    assert m["precision"] == 0.0
    assert m["recall"] == 0.0
    assert m["num_citations"] == 0.0


def test_citation_metrics_dedupes_and_coerces_types() -> None:
    # Integer docids and duplicates are handled as string sets (as in the trajectory).
    m = compute_citation_metrics([1, 1, 2], [2])
    assert m["num_citations"] == 2.0
    assert m["recall"] == 1.0
    assert m["precision"] == 0.5


def test_retrieval_recall() -> None:
    assert retrieval_recall(["1", "2", "3"], ["2", "4"]) == 0.5  # {2} / {2,4}
    assert retrieval_recall(["1", "2"], ["1", "2"]) == 1.0
    assert retrieval_recall([], ["1"]) == 0.0
    # No relevant docids: no recall to measure (guarded divide-by-zero).
    assert retrieval_recall(["1"], []) == 0.0
