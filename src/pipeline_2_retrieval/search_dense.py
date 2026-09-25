r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/search_dense.py
   - Role: Dense vector semantic search executor with native metadata pre-filtering.
   - Purpose: Encodes normalized queries into dense embedding vectors and executes
     native cosine similarity searches against Qdrant vector store child chunk collections
     with dynamic document-level balanced quota allocations and parent deduplication.

2. INPUT (IP):
   - query_text (str): Preprocessed search query from `src/pipeline_2_retrieval/rewriter.py`.
   - embedder (DualEmbedder): Dense embedding instance from `src/pipeline_1_ingestion/embedder.py`.
   - vector_store (QdrantVectorStore / LocalStore): Persistent vector store from `src/pipeline_1_ingestion/vector_store.py`.
   - top_k (int): Number of top semantic candidates to retrieve (default 20).
   - doc_filter (str or list[str], optional): Target document ID(s) to isolate search scope.

3. PROCESS UNDER THE HOOD:
   - Encodes query_text into a 384-dimensional vector using `embedder.embed_dense([query_text])[0]`.
   - For multi-document filters (`list[str]`), allocates balanced dynamic per-document retrieval quotas
     (`k_per_doc = max(1, top_k // min(len(clean_filters), 3))`) to ensure fair cross-document representation.
   - For single-document or global search, delegates directly to `vector_store.search()`.
   - Deduplicates matching points by `parent_id` / `child_id` to prevent redundant overlapping slices.
   - Formats results into 1-indexed ranked tuples: `(child_id, rank, similarity_score)`.

4. OUTPUT (OP):
   - list[tuple[str, int, float]]: Ranked list of `(child_id, rank, score)`.
   - Consumed by: `src/pipeline_2_retrieval/fusion.py` for Reciprocal Rank Fusion (RRF).

5. LIBRARIES & DEPENDENCIES:
   - qdrant_client.http.models: Qdrant filtering dataclasses.
   - typing (List, Tuple, Any, Optional, Union, Dict, Set): Type annotations.
   - src.common.config: Central configuration instance.
   - src.pipeline_1_ingestion.vector_store (QdrantVectorStore): Vector storage engine.
================================================================================
"""

from typing import Any, Dict, List, Optional, Set, Tuple, Union
from qdrant_client.http import models

from src.common.config import config


class DenseSearcher:
    """Executes dense vector semantic search against QdrantVectorStore with balanced quota allocation."""

    def __init__(self, vector_store: Any, embedder: Any) -> None:
        """Initialize DenseSearcher with vector store and embedding engine.

        Args:
            vector_store: QdrantVectorStore or compatible vector store instance.
            embedder: DualEmbedder or compatible dense encoder.
        """
        self.vector_store = vector_store
        self.embedder = embedder

    def search(
        self,
        query_text: str,
        top_k: int = 20,
        doc_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[Tuple[str, int, float]]:
        """Retrieve top-k child chunks via dense cosine similarity search in Qdrant.

        Args:
            query_text: Normalized search query.
            top_k: Maximum number of nearest neighbors to retrieve.
            doc_filter: Optional document ID or list of document IDs.

        Returns:
            A 1-indexed ranked list of tuples: [(child_id, rank, score), ...].
        """
        if not query_text or not query_text.strip():
            return []

        query_vectors = self.embedder.embed_dense([query_text])
        if not query_vectors:
            return []
        query_vector = query_vectors[0]

        # Multi-document balanced quota search
        if isinstance(doc_filter, list) and len(doc_filter) > 0:
            clean_filters = [d.strip() for d in doc_filter if isinstance(d, str) and d.strip()]
            if len(clean_filters) == 1:
                return self._search_single(query_vector, top_k, clean_filters[0])
            elif len(clean_filters) > 1:
                configured_cap = getattr(config.retrieval, "max_chunks_per_doc", 3)
                dynamic_quota = max(1, top_k // min(len(clean_filters), 3))
                k_per_doc = min(configured_cap, dynamic_quota) if configured_cap > 0 else dynamic_quota
                k_per_doc = max(1, k_per_doc)

                all_points: List[Tuple[str, float]] = []
                seen_parents: Set[str] = set()
                seen_cids: Set[str] = set()

                for target_doc in clean_filters:
                    res = self.vector_store.search(
                        query_vector=query_vector,
                        limit=k_per_doc,
                        doc_filter=target_doc,
                    )
                    for pt in res:
                        cid = pt.get("child_id", "")
                        pid = pt.get("parent_id", "")
                        score = float(pt.get("score", 0.0))

                        # Deduplicate by parent_id or child_id
                        dedup_key = pid if pid else cid
                        if dedup_key not in seen_parents and cid not in seen_cids:
                            seen_parents.add(dedup_key)
                            seen_cids.add(cid)
                            all_points.append((cid, score))

                # Sort combined candidates descending by similarity score
                all_points.sort(key=lambda x: x[1], reverse=True)
                return [(cid, rank, score) for rank, (cid, score) in enumerate(all_points[:top_k], start=1)]

        # Single document or global search
        return self._search_single(query_vector, top_k, doc_filter if isinstance(doc_filter, str) else None)

    def _search_single(
        self,
        query_vector: List[float],
        top_k: int,
        doc_filter: Optional[str],
    ) -> List[Tuple[str, int, float]]:
        """Execute single-filter or global search."""
        res = self.vector_store.search(
            query_vector=query_vector,
            limit=top_k,
            doc_filter=doc_filter,
        )
        ranked_results: List[Tuple[str, int, float]] = []
        for rank_idx, pt in enumerate(res, start=1):
            cid = pt.get("child_id", "")
            score = float(pt.get("score", 0.0))
            ranked_results.append((cid, rank_idx, score))

        return ranked_results


def retrieve_dense(
    query_text: str,
    embedder: Any,
    client_or_store: Any,
    collection_name: str = "trustrag_enterprise",
    top_k: int = 20,
    doc_filter: Optional[Union[str, List[str]]] = None,
) -> List[Tuple[str, int, float]]:
    """Functional wrapper for dense retrieval supporting both QdrantVectorStore and raw QdrantClient.

    Args:
        query_text: Normalized search query.
        embedder: Instance of DualEmbedder or compatible dense encoder.
        client_or_store: QdrantVectorStore instance or raw QdrantClient.
        collection_name: Target collection name.
        top_k: Maximum number of nearest neighbors to retrieve.
        doc_filter: Optional document ID or list of document IDs.

    Returns:
        A 1-indexed ranked list of tuples: [(child_id, rank, score), ...].
    """
    if hasattr(client_or_store, "search") and callable(client_or_store.search) and not hasattr(client_or_store, "get_collections"):
        searcher = DenseSearcher(vector_store=client_or_store, embedder=embedder)
        return searcher.search(query_text=query_text, top_k=top_k, doc_filter=doc_filter)

    # Fallback to direct client execution if raw QdrantClient is passed
    if not query_text or not query_text.strip():
        return []

    query_vectors = embedder.embed_dense([query_text])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    # Handle multi-document balanced retrieval on raw client
    if isinstance(doc_filter, list) and len(doc_filter) > 0:
        clean_filters = [d.strip() for d in doc_filter if isinstance(d, str) and d.strip()]
        if len(clean_filters) == 1:
            doc_filter = clean_filters[0]
        elif len(clean_filters) > 1:
            configured_cap = getattr(config.retrieval, "max_chunks_per_doc", 3)
            dynamic_quota = max(1, top_k // min(len(clean_filters), 3))
            k_per_doc = min(configured_cap, dynamic_quota) if configured_cap > 0 else dynamic_quota
            k_per_doc = max(1, k_per_doc)
            all_points: List[Tuple[str, float]] = []
            seen_cids: Set[str] = set()

            for target_doc in clean_filters:
                q_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=target_doc),
                        )
                    ]
                )
                res = client_or_store.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=q_filter,
                    limit=k_per_doc,
                    with_payload=True,
                )
                for pt in res:
                    payload = pt.payload or {}
                    cid = payload.get("child_id", payload.get("chunk_id", str(pt.id)))
                    if cid not in seen_cids:
                        seen_cids.add(cid)
                        all_points.append((cid, float(pt.score)))

            all_points.sort(key=lambda x: x[1], reverse=True)
            return [(cid, rank, score) for rank, (cid, score) in enumerate(all_points[:top_k], start=1)]

    query_filter = None
    if isinstance(doc_filter, str) and doc_filter.strip():
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_filter.strip()),
                )
            ]
        )

    search_results = client_or_store.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    ranked_results: List[Tuple[str, int, float]] = []
    for rank_idx, point in enumerate(search_results, start=1):
        payload = point.payload or {}
        child_id = payload.get("child_id", payload.get("chunk_id", str(point.id)))
        score = float(point.score)
        ranked_results.append((child_id, rank_idx, score))

    return ranked_results

