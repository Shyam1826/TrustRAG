r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/fusion.py
   - Role: Hybrid rank fusion and intent-adaptive document diversification engine.
   - Purpose: Combines dense semantic and sparse lexical ranking lists into a single,
     robust candidate list using Reciprocal Rank Fusion (RRF), eliminating the need for
     arbitrary score calibration across disparate scoring scales, with strict per-document
     quota diversification to prevent large corpora from starving smaller documents.

2. INPUT (IP):
   - dense_ranks (list[tuple[str, int, float]]): Ranked dense results from `src/pipeline_2_retrieval/search_dense.py`.
   - sparse_ranks (list[tuple[str, int, float]]): Ranked sparse results from `src/pipeline_2_retrieval/search_sparse.py`.
   - k (int): RRF smoothing constant (default 60 from `src/common/config.py`).
   - top_n (int): Number of top fused candidates to return (default 20).
   - child_chunk_map (dict[str, Any], optional): Map of chunk IDs to ChildChunk instances for per-doc quota.
   - max_chunks_per_doc (int, optional): Maximum allowed chunks per document during fusion.
   - query (str, optional): Cleaned query string for intent-adaptive allocation.
   - is_comparative (bool, optional): Explicit override for comparative diversification intent.
   - doc_filter (str or list[str], optional): Explicit document routing filter.

3. PROCESS UNDER THE HOOD:
   - Evaluates the standard Reciprocal Rank Fusion formula:
     RRF_Score(d) = sum_{s in systems} (1.0 / (k + rank_s(d)))
   - Maintains an accumulator dictionary: `dict[child_id, float]`.
   - Iterates through dense ranked pairs and adds `1.0 / (k + rank)`.
   - Iterates through sparse ranked pairs and adds `1.0 / (k + rank)`.
   - Deduplicates items across dense and sparse retrievers.
   - Sorts candidate IDs descending by cumulative RRF score.
   - Strict Diversification Quota Allocation:
     * Exception Rule: When a single solitary document is targeted via `doc_filter` (or only 1
       document exists in candidate set), allow candidates from that document to fill up to `top_n`.
     * Comparative / Multi-Document Intent: Computes proportional dynamic quota
       `max(1, top_n // min(num_docs, 4))` bounded by `max_chunks_per_doc`.
     * General Multi-Document Intent: Enforces strict per-document quota ceiling (`max_chunks_per_doc`, default: 3).
     * Backfill Pass: If quota-based selection produces fewer than `top_n` candidates, backfills from
       the remaining highest-scoring candidates across all documents until `top_n` is fulfilled.

4. OUTPUT (OP):
   - list[tuple[str, float]]: List of `(child_id, rrf_score)` candidate tuples.
   - Consumed by: `src/pipeline_2_retrieval/reranker.py` for cross-encoder reranking.

5. LIBRARIES & DEPENDENCIES:
   - collections.defaultdict: Dictionary with default float values for fast score accumulation.
   - re: Regex pattern matching for comparative intent classification.
   - typing (Dict, List, Optional, Tuple, Any, Union): Standard type hints.
   - src.common.config: Central configuration instance.
================================================================================
"""

from collections import defaultdict
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from src.common.config import config

_COMPARATIVE_QUERY_REGEX = re.compile(
    r"\b(?:"
    r"compare|comparison|difference|differences|versus|vs|both|"
    r"contrast|contrasting|across|between|each|all\s+(?:candidates|documents|resumes|specs|papers|files|records)|"
    r"and\s+.*(?:difference|compare|contrast|versus|vs)"
    r")\b",
    re.IGNORECASE,
)


def is_comparative_query(query: Optional[str]) -> bool:
    """Detect if a user query demonstrates comparative or cross-document intent.

    Args:
        query: Query string to evaluate.

    Returns:
        True if comparative intent is detected, False otherwise.
    """
    if not query:
        return False
    return bool(_COMPARATIVE_QUERY_REGEX.search(query))


def apply_rrf(
    dense_ranks: List[Tuple[str, int, float]],
    sparse_ranks: List[Tuple[str, int, float]],
    k: int = 60,
    top_n: int = 20,
    child_chunk_map: Optional[Dict[str, Any]] = None,
    max_chunks_per_doc: Optional[int] = None,
    query: Optional[str] = None,
    is_comparative: Optional[bool] = None,
    doc_filter: Optional[Union[str, List[str]]] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> List[Tuple[str, float]]:
    """Combine dense and sparse ranked lists using Reciprocal Rank Fusion (RRF) with strict diversification quotas.

    Formula:
        RRF_Score(d) = sum_{s in systems} (1.0 / (k + rank_s(d)))

    Args:
        dense_ranks: List of (child_id, rank, score) from dense search.
        sparse_ranks: List of (child_id, rank, score) from sparse search.
        k: Smoothing parameter mitigating the impact of high-ranking outliers (default 60).
        top_n: Maximum number of fused candidate tuples to return.
        child_chunk_map: Optional mapping of child_id to ChildChunk for document-level diversification.
        max_chunks_per_doc: Optional override for max chunks per document.
        query: Optional user query string for intent-adaptive allocation.
        is_comparative: Optional explicit boolean flag declaring comparative intent.
        doc_filter: Optional document routing filter to evaluate solitary document scope.
        user_id: Optional tenant user_id filter.
        thread_id: Optional tenant thread_id filter.

    Returns:
        List of tuples sorted descending by RRF score: [(child_id, rrf_score), ...].
    """
    rrf_scores: defaultdict[str, float] = defaultdict(float)

    # Accumulate RRF contributions from dense retrieval
    for child_id, rank, _ in dense_ranks:
        rrf_scores[child_id] += 1.0 / (k + rank)

    # Accumulate RRF contributions from sparse retrieval
    for child_id, rank, _ in sparse_ranks:
        rrf_scores[child_id] += 1.0 / (k + rank)

    # Sort descending by cumulative RRF score
    sorted_candidates = sorted(
        rrf_scores.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    # Filter by tenant coordinates if child_chunk_map is provided
    if child_chunk_map and user_id is not None:
        sorted_candidates = [
            item for item in sorted_candidates
            if item[0] not in child_chunk_map or getattr(child_chunk_map[item[0]], "user_id", None) == user_id
        ]
    if child_chunk_map and thread_id is not None:
        sorted_candidates = [
            item for item in sorted_candidates
            if item[0] not in child_chunk_map or getattr(child_chunk_map[item[0]], "thread_id", None) == thread_id
        ]

    # Apply strict document diversification if child_chunk_map is provided
    if child_chunk_map and getattr(config.retrieval, "enable_document_diversification", True):
        matching_docs = {
            getattr(child_chunk_map[cid], "doc_id", None)
            for cid, _ in sorted_candidates
            if cid in child_chunk_map and hasattr(child_chunk_map[cid], "doc_id")
        }
        matching_docs.discard(None)
        num_docs = len(matching_docs)

        if num_docs > 1:
            # Exception rule: solitary document targeted by scope router -> no quota suppression
            if isinstance(doc_filter, str) and doc_filter.strip():
                return sorted_candidates[:top_n]

            # Determine comparative vs general multi-document intent
            comparative_intent = (
                is_comparative
                if is_comparative is not None
                else (is_comparative_query(query) if query is not None else True)
            )

            configured_cap = (
                max_chunks_per_doc
                if max_chunks_per_doc is not None
                else getattr(config.retrieval, "max_chunks_per_doc", 3)
            )

            if comparative_intent:
                dynamic_quota = max(1, top_n // min(num_docs, 4))
                effective_quota = min(configured_cap, dynamic_quota) if configured_cap > 0 else dynamic_quota
            else:
                effective_quota = configured_cap if configured_cap > 0 else max(1, top_n // min(num_docs, 4))

            effective_quota = max(1, effective_quota)

            selected: List[Tuple[str, float]] = []
            doc_counts: Dict[str, int] = defaultdict(int)
            remaining: List[Tuple[str, float]] = []

            # 1. Quota-based diversification pass
            for cid, score in sorted_candidates:
                doc_id = getattr(child_chunk_map[cid], "doc_id", None) if cid in child_chunk_map else None
                if doc_id and doc_counts[doc_id] < effective_quota and len(selected) < top_n:
                    selected.append((cid, score))
                    doc_counts[doc_id] += 1
                elif not doc_id and len(selected) < top_n:
                    selected.append((cid, score))
                else:
                    remaining.append((cid, score))

            # 2. Backfill pass from remaining highest-scoring candidates if under top_n
            if len(selected) < top_n:
                for item in remaining:
                    if len(selected) >= top_n:
                        break
                    selected.append(item)

            return selected

    return sorted_candidates[:top_n]


# Functional alias for unified interface naming
apply_rrf_fusion = apply_rrf
