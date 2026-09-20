r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/indexer.py
   - Role: Hybrid storage and vector indexing coordinator.
   - Purpose: Re-exports `QdrantVectorStore` and `LocalStore` from `vector_store.py`
     for full backwards compatibility across ingestion, retrieval, and reranking pipelines.

2. INPUT (IP):
   - chunks (list[ChildChunk]): Embedder-enriched ChildChunk models with dense vectors and metadata.
   - parents (list[ParentChunk]): Coarse context ParentChunk models.
   - Source: `src/pipeline_1_ingestion/chunker.py` and `src/pipeline_1_ingestion/embedder.py`.

3. PROCESS UNDER THE HOOD:
   - Re-exports `QdrantVectorStore` and `LocalStore` implementation.

4. OUTPUT (OP):
   - QdrantVectorStore and LocalStore classes.
   - Consumed by: `src/pipeline_2_retrieval/search_dense.py`, `src/pipeline_2_retrieval/reranker.py`,
     and `src/main.py`.

5. LIBRARIES & DEPENDENCIES:
   - src.pipeline_1_ingestion.vector_store: QdrantVectorStore, LocalStore.
================================================================================
"""

from src.pipeline_1_ingestion.vector_store import LocalStore, QdrantVectorStore

__all__ = ["QdrantVectorStore", "LocalStore"]

