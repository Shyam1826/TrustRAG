r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/search_dense.py
   - Role: Dense vector semantic search executor with metadata pre-filtering.
   - Purpose: Encodes normalized queries into dense embedding vectors and performs
     approximate/exact cosine similarity search against Qdrant child chunk indices
     with optional document-level entity pre-filtering.

2. INPUT (IP):
   - query_text (str): Preprocessed search query from `src/pipeline_2_retrieval/rewriter.py`.
   - embedder (DualEmbedder): Dense embedding instance from `src/pipeline_1_ingestion/embedder.py`.
   - client (QdrantClient): Vector database client from `src/pipeline_1_ingestion/indexer.py`.
   - collection_name (str): Target Qdrant collection (default "child_chunks").
   - top_k (int): Number of top semantic candidates to retrieve (default 20).
   - doc_filter (str, optional): Target document ID to isolate search to.

3. PROCESS UNDER THE HOOD:
   - Encodes query_text into a 384-dimensional L2-normalized vector using `embedder.embed_dense([query_text])[0]`.
   - Constructs Qdrant metadata `query_filter` if `doc_filter` is specified.
   - Executes nearest-neighbor search in Qdrant via `client.search(collection_name, query_vector, query_filter, limit=top_k)`.
   - Iterates through scored points and extracts `chunk_id` from point payload.
   - Formats results into 1-indexed ranked tuples: `(child_id, rank, similarity_score)`.

4. OUTPUT (OP):
   - list[tuple[str, int, float]]: Ranked list of `(child_id, rank, score)`.
   - Consumed by: `src/pipeline_2_retrieval/fusion.py` for Reciprocal Rank Fusion (RRF).

5. LIBRARIES & DEPENDENCIES:
   - qdrant_client: Vector database search engine.
   - qdrant_client.http.models: Filter and FieldCondition dataclasses.
   - typing (List, Tuple, Any, Optional): Type annotations.
================================================================================
"""

from typing import Any, List, Optional, Tuple, Union
from qdrant_client.http import models


def retrieve_dense(
    query_text: str,
    embedder: Any,
    client: Any,
    collection_name: str = "child_chunks",
    top_k: int = 20,
    doc_filter: Optional[Union[str, List[str]]] = None,
) -> List[Tuple[str, int, float]]:
    """Retrieve top-k child chunks via dense cosine similarity search in Qdrant.

    Args:
        query_text: Normalized search query.
        embedder: Instance of DualEmbedder or compatible dense encoder.
        client: QdrantClient instance holding the index.
        collection_name: Target collection name in Qdrant.
        top_k: Maximum number of nearest neighbors to retrieve.
        doc_filter: Optional document ID or list of document IDs to filter search scope.

    Returns:
        A 1-indexed ranked list of tuples: [(child_id, rank, score), ...].
    """
    if not query_text or not query_text.strip():
        return []

    # Generate dense vector for query
    query_vectors = embedder.embed_dense([query_text])
    if not query_vectors:
        return []
    query_vector = query_vectors[0]

    # Handle multi-document balanced retrieval
    if isinstance(doc_filter, list) and len(doc_filter) > 0:
        if len(doc_filter) == 1:
            doc_filter = doc_filter[0]
        else:
            k_per_doc = max(2, top_k // len(doc_filter))
            all_points = []
            seen_cids = set()

            for target_doc in doc_filter:
                q_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=target_doc),
                        )
                    ]
                )
                res = client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=q_filter,
                    limit=k_per_doc,
                    with_payload=True,
                )
                for pt in res:
                    payload = pt.payload or {}
                    cid = payload.get("chunk_id", str(pt.id))
                    if cid not in seen_cids:
                        seen_cids.add(cid)
                        all_points.append((cid, float(pt.score)))

            # Sort combined results descending by score
            all_points.sort(key=lambda x: x[1], reverse=True)
            return [(cid, rank, score) for rank, (cid, score) in enumerate(all_points[:top_k], start=1)]

    # Single-document or global search
    query_filter = None
    if isinstance(doc_filter, str) and doc_filter:
        query_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_filter),
                )
            ]
        )

    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        query_filter=query_filter,
        limit=top_k,
        with_payload=True,
    )

    ranked_results: List[Tuple[str, int, float]] = []
    for rank_idx, point in enumerate(search_results, start=1):
        payload = point.payload or {}
        child_id = payload.get("chunk_id", str(point.id))
        score = float(point.score)
        ranked_results.append((child_id, rank_idx, score))

    return ranked_results
