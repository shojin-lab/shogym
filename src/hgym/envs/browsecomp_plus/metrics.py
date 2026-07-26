"""Deterministic retrieval + citation metrics for the ``browsecomp_plus`` env (issue #43).

These are the *non-judge* half of BrowseComp-Plus scoring: given the docids a run retrieved
(off the recorded ``search`` steps), the docids it cited (off the submitted answer text), and
the query's relevance judgements (``qrel_evidence`` / ``qrel_golds``), compute recall and
citation precision/recall exactly as upstream ``scripts_evaluation/evaluate_run.py`` does.

Everything here is **pure** and dependency-free (stdlib ``re`` only), so it runs in the offline
core test suite and is called from the env's pure ``_verify``. The functions are ported
verbatim from BrowseComp-Plus commit ``0469490`` to keep the metrics faithful.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, List


def extract_citations_from_response(response_text: str) -> List[str]:
    """Extract cited docids from an answer, matching upstream's citation grammar.

    Recognises ``[docid]`` / ``[docid1, docid2, ...]`` and the full-width ``【docid】`` variants
    (upstream notes some models were fine-tuned on the full-width form). Returns the distinct
    docids as strings. Ported verbatim from BrowseComp-Plus ``evaluate_run.py``."""
    if not response_text:
        return []

    single_matches = re.findall(r"\[(\d+)\]", response_text)
    multi_matches = re.findall(r"\[([^\[\]]*?)\]", response_text)
    single_fullwidth_matches = re.findall(r"【(\d+)】", response_text)
    multi_fullwidth_matches = re.findall(r"【([^【】]*?)】", response_text)

    all_docids: set[str] = set()
    all_docids.update(single_matches)
    all_docids.update(single_fullwidth_matches)

    for match in multi_matches:
        if match in single_matches:
            continue
        all_docids.update(re.findall(r"\d+", match))

    for match in multi_fullwidth_matches:
        if match in single_fullwidth_matches:
            continue
        all_docids.update(re.findall(r"\d+", match))

    return list(all_docids)


def compute_citation_metrics(
    cited_docids: Iterable[str], relevant_docids: Iterable[str]
) -> Dict[str, float]:
    """Citation precision/recall of ``cited_docids`` against ``relevant_docids`` (upstream verbatim).

    - ``precision`` — fraction of cited docids that are relevant;
    - ``recall`` — fraction of relevant docids that were cited.
    Empty citations yield 0.0 for both; ``num_citations`` / ``num_relevant`` are the set sizes."""
    cited_set = {str(d) for d in cited_docids}
    relevant_set = {str(d) for d in relevant_docids}
    metrics: Dict[str, float] = {
        "num_citations": float(len(cited_set)),
        "num_relevant": float(len(relevant_set)),
        "precision": 0.0,
        "recall": 0.0,
    }
    if not cited_set:
        return metrics

    relevant_cited = cited_set & relevant_set
    metrics["precision"] = len(relevant_cited) / len(cited_set)
    if relevant_set:
        metrics["recall"] = len(relevant_cited) / len(relevant_set)
    return metrics


def retrieval_recall(retrieved_docids: Iterable[str], relevant_docids: Iterable[str]) -> float:
    """Fraction of relevant docids that were retrieved (upstream's ``retrieval_recall``).

    ``|retrieved ∩ relevant| / |relevant|``. An empty relevant set has no recall to measure and
    returns 0.0 (upstream divides by ``len(positives)``; the env only computes this when the
    query has evidence qrels, so this guard just avoids a divide-by-zero on a degenerate task)."""
    relevant_set = {str(d) for d in relevant_docids}
    if not relevant_set:
        return 0.0
    retrieved_set = {str(d) for d in retrieved_docids}
    return len(retrieved_set & relevant_set) / float(len(relevant_set))
