r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/fusion.py
   - Role: Hybrid rank fusion and document diversification engine.
   - Purpose: Combines dense semantic and sparse lexical ranking lists into a single,
     robust candidate list using Reciprocal Rank Fusion (RRF), eliminating the need for
     arbitrary score calibration across disparate scoring scales, with optional dynamic
     per-document quota diversification.

2. INPUT (IP):
   - dense_ranks (list[tuple[str, int, float]]): Ranked dense results from `src/pipeline_2_retrieval/search_dense.py`.
   - sparse_ranks (list[tuple[str, int, float]]): Ranked sparse results from `src/pipeline_2_retrieval/search_sparse.py`.
   - k (int): RRF smoothing constant (default 60 from `src/common/config.py`).
   - top_n (int): Number of top fused candidates to return (default 20).
   - child_chunk_map (dict[str, Any], optional): Map of chunk IDs to ChildChunk instances for per-doc quota.
   - max_chunks_per_doc (int, optional): Maximum allowed chunks per document during fusion.

3. PROCESS UNDER THE HOOD:
   - Evaluates the standard Reciprocal Rank Fusion formula:
     RRF_Score(d) = sum_{s in systems} (1.0 / (k + rank_s(d)))
   - Maintains an accumulator dictionary: `dict[child_id, float]`.
   - Iterates through dense ranked pairs and adds `1.0 / (k + rank)`.
   - Iterates through sparse ranked pairs and adds `1.0 / (k + rank)`.
   - Deduplicates items across dense and sparse retrievers.
   - Sorts candidate IDs descending by cumulative RRF score.
   - If `child_chunk_map` is provided and multi-document diversification is active:
     * Calculates dynamic quota cap: `max_chunks_per_source = max(1, top_n // min(num_unique_matching_docs, 3))`.
     * Selects candidates in descending RRF order respecting per-document quota limits.
     * Backfills from remaining candidates if fewer than `top_n` are selected.
   - Otherwise, truncates directly to the top_n items.

4. OUTPUT (OP):
   - list[tuple[str, float]]: List of `(child_id, rrf_score)` candidate tuples.
   - Consumed by: `src/pipeline_2_retrieval/reranker.py` for cross-encoder reranking.

5. LIBRARIES & DEPENDENCIES:
   - collections.defaultdict: Dictionary with default float values for fast score accumulation.
   - typing (Dict, List, Optional, Tuple, Any): Standard type hints.
   - src.common.config: Central configuration instance.
================================================================================
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from src.common.config import config


def apply_rrf(
    dense_ranks: List[Tuple[str, int, float]],
    sparse_ranks: List[Tuple[str, int, float]],
    k: int = 60,
    top_n: int = 20,
    child_chunk_map: Optional[Dict[str, Any]] = None,
    max_chunks_per_doc: Optional[int] = None,
) -> List[Tuple[str, float]]:
    """Combine dense and sparse ranked lists using Reciprocal Rank Fusion (RRF) with dynamic document diversification.

    Formula:
        RRF_Score(d) = sum_{s in systems} (1.0 / (k + rank_s(d)))

    Args:
        dense_ranks: List of (child_id, rank, score) from dense search.
        sparse_ranks: List of (child_id, rank, score) from sparse search.
        k: Smoothing parameter mitigating the impact of high-ranking outliers (default 60).
        top_n: Maximum number of fused candidate tuples to return.
        child_chunk_map: Optional mapping of child_id to ChildChunk for document-level diversification.
        max_chunks_per_doc: Optional override for max chunks per document.

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

    # Apply dynamic document diversification if child_chunk_map is provided
    if child_chunk_map and getattr(config.retrieval, "enable_document_diversification", True):
        matching_docs = {
            getattr(child_chunk_map[cid], "doc_id", None)
            for cid, _ in sorted_candidates
            if cid in child_chunk_map and hasattr(child_chunk_map[cid], "doc_id")
        }
        matching_docs.discard(None)
        num_docs = len(matching_docs)

        if num_docs > 1:
            configured_cap = max_chunks_per_doc if max_chunks_per_doc is not None else getattr(config.retrieval, "max_chunks_per_doc", 3)
            dynamic_quota = max(1, top_n // min(num_docs, 3))
            max_chunks_per_source = min(configured_cap, dynamic_quota) if configured_cap > 0 else dynamic_quota
            max_chunks_per_source = max(1, max_chunks_per_source)

            selected: List[Tuple[str, float]] = []
            doc_counts: Dict[str, int] = defaultdict(int)
            remaining: List[Tuple[str, float]] = []

            for cid, score in sorted_candidates:
                doc_id = getattr(child_chunk_map[cid], "doc_id", None) if cid in child_chunk_map else None
                if doc_id and doc_counts[doc_id] < max_chunks_per_source and len(selected) < top_n:
                    selected.append((cid, score))
                    doc_counts[doc_id] += 1
                elif not doc_id and len(selected) < top_n:
                    selected.append((cid, score))
                else:
                    remaining.append((cid, score))

            if len(selected) < top_n:
                for item in remaining:
                    if len(selected) >= top_n:
                        break
                    selected.append(item)

            return selected

    return sorted_candidates[:top_n]


# Functional alias for unified interface naming
apply_rrf_fusion = apply_rrf
