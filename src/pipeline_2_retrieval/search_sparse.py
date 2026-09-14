r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/search_sparse.py
   - Role: Lexical/sparse search engine with metadata pre-filtering.
   - Purpose: Implements BM25Okapi scoring across child chunks to capture exact keyword,
     technical term, and number matches, supporting document-level entity pre-filtering.

2. INPUT (IP):
   - Initialization: chunks (list[ChildChunk]) from `src/pipeline_1_ingestion/chunker.py`.
   - Search: query_tokens (list[str]) representing tokenized query keywords.
   - top_k (int): Number of top lexical matches to retrieve (default 20).
   - doc_filter (str, optional): Target document ID to isolate search to.

3. PROCESS UNDER THE HOOD:
   - Tokenizes child chunks into lowercase word tokens (`re.findall(r'\b\w+\b', chunk.text.lower())`).
   - Builds inverted index structures and document frequency stats via `rank_bm25.BM25Okapi`.
   - On search:
     * Evaluates BM25 scores across the indexed corpus.
     * If `doc_filter` is specified, filters out chunks belonging to other documents.
     * Sorts candidate chunks descending by BM25 score.
     * Takes top_k scored entries and assigns 1-indexed ranks.

4. OUTPUT (OP):
   - list[tuple[str, int, float]]: Ranked list of `(child_id, rank, score)`.
   - Consumed by: `src/pipeline_2_retrieval/fusion.py` for Reciprocal Rank Fusion (RRF).

5. LIBRARIES & DEPENDENCIES:
   - rank_bm25.BM25Okapi: Standard library for BM25 Okapi lexical scoring.
   - re: Standard library module for regex tokenization.
   - src.common.schemas.ChildChunk: Typed data contract for fine-grained chunks.
================================================================================
"""

import re
from typing import List, Optional, Tuple, Union
from rank_bm25 import BM25Okapi

from src.common.schemas import ChildChunk


class BM25Searcher:
    """Performs sparse Okapi BM25 keyword retrieval over indexed ChildChunk structures."""

    def __init__(self, chunks: List[ChildChunk]) -> None:
        """Initialize BM25 index with pre-extracted ChildChunk objects.

        Args:
            chunks: List of ChildChunk instances representing the retrieval corpus.
        """
        self.chunks = chunks
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]

        # Tokenize corpus for BM25Okapi
        self.corpus_tokens = [self._tokenize(chunk.text) for chunk in chunks]

        if self.corpus_tokens:
            self.bm25 = BM25Okapi(self.corpus_tokens)
        else:
            self.bm25 = None

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercased alphanumeric word list."""
        if not text:
            return []
        return re.findall(r"\b\w+\b", text.lower())

    def search(
        self,
        query_tokens: List[str],
        top_k: int = 20,
        doc_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[str, int, float]]:
        """Score indexed child chunks against query tokens using BM25.

        Args:
            query_tokens: List of lowercased keyword tokens.
            top_k: Maximum number of results to return.
            doc_filter: Optional document ID or list of document IDs to pre-filter candidate pool.

        Returns:
            A 1-indexed ranked list of tuples: [(child_id, rank, score), ...].
        """
        if not self.bm25 or not query_tokens or not self.chunks:
            return []

        # Ensure query tokens are cleaned and lowercased
        cleaned_tokens = [t.lower().strip() for t in query_tokens if t.strip()]
        if not cleaned_tokens:
            return []

        scores = self.bm25.get_scores(cleaned_tokens)

        # Handle balanced multi-document retrieval
        if isinstance(doc_filter, list) and len(doc_filter) > 0:
            if len(doc_filter) == 1:
                doc_filter = doc_filter[0]
            else:
                k_per_doc = max(2, top_k // len(doc_filter))
                balanced_pairs: List[Tuple[str, float]] = []
                for target_doc in doc_filter:
                    doc_pairs = [
                        (chunk.chunk_id, float(score))
                        for chunk, score in zip(self.chunks, scores)
                        if chunk.doc_id == target_doc
                    ]
                    doc_pairs.sort(key=lambda x: x[1], reverse=True)
                    balanced_pairs.extend(doc_pairs[:k_per_doc])

                balanced_pairs.sort(key=lambda x: x[1], reverse=True)
                return [
                    (chunk_id, rank_idx, float(score))
                    for rank_idx, (chunk_id, score) in enumerate(balanced_pairs[:top_k], start=1)
                ]

        # Pair chunks with BM25 scores and apply single doc_filter if active
        scored_pairs: List[Tuple[str, float]] = []
        for chunk, score in zip(self.chunks, scores):
            if isinstance(doc_filter, str) and doc_filter and chunk.doc_id != doc_filter:
                continue
            scored_pairs.append((chunk.chunk_id, float(score)))

        # Sort descending by BM25 score
        scored_pairs.sort(key=lambda x: x[1], reverse=True)

        top_pairs = scored_pairs[:top_k]

        ranked_results: List[Tuple[str, int, float]] = [
            (chunk_id, rank_idx, float(score))
            for rank_idx, (chunk_id, score) in enumerate(top_pairs, start=1)
        ]

        return ranked_results
