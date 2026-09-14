"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/fusion.py
   - Role: Hybrid rank fusion engine.
   - Purpose: Combines dense semantic and sparse lexical ranking lists into a single,
     robust candidate list using Reciprocal Rank Fusion (RRF), eliminating the need for
     arbitrary score calibration across disparate scoring scales.

2. INPUT (IP):
   - dense_ranks (list[tuple[str, int, float]]): Ranked dense results from `src/pipeline_2_retrieval/search_dense.py`.
   - sparse_ranks (list[tuple[str, int, float]]): Ranked sparse results from `src/pipeline_2_retrieval/search_sparse.py`.
   - k (int): RRF smoothing constant (default 60 from `src/common/config.py`).
   - top_n (int): Number of top fused candidates to return (default 20).

3. PROCESS UNDER THE HOOD:
   - Evaluates the standard Reciprocal Rank Fusion formula:
     RRF_Score(d) = sum_{s in systems} (1.0 / (k + rank_s(d)))
   - Maintains an accumulator dictionary: `dict[child_id, float]`.
   - Iterates through dense ranked pairs and adds `1.0 / (k + rank)`.
   - Iterates through sparse ranked pairs and adds `1.0 / (k + rank)`.
   - Deduplicates items across dense and sparse retrievers.
   - Sorts candidate IDs descending by cumulative RRF score.
   - Truncates to the top_n items.

4. OUTPUT (OP):
   - list[tuple[str, float]]: List of `(child_id, rrf_score)` candidate tuples.
   - Consumed by: `src/pipeline_2_retrieval/reranker.py` for cross-encoder reranking.

5. LIBRARIES & DEPENDENCIES:
   - collections.defaultdict: Dictionary with default float values for fast score accumulation.
   - typing (List, Tuple): Standard type hints.
================================================================================
"""

from collections import defaultdict
from typing import List, Tuple


def apply_rrf(
    dense_ranks: List[Tuple[str, int, float]],
    sparse_ranks: List[Tuple[str, int, float]],
    k: int = 60,
    top_n: int = 20,
) -> List[Tuple[str, float]]:
    """Combine dense and sparse ranked lists using Reciprocal Rank Fusion (RRF).

    Formula:
        RRF_Score(d) = sum_{s in systems} (1.0 / (k + rank_s(d)))

    Args:
        dense_ranks: List of (child_id, rank, score) from dense search.
        sparse_ranks: List of (child_id, rank, score) from sparse search.
        k: Smoothing parameter mitigating the impact of high-ranking outliers (default 60).
        top_n: Maximum number of fused candidate tuples to return.

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

    return sorted_candidates[:top_n]
