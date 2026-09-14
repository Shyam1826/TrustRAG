r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/indexer.py
   - Role: Hybrid storage and vector indexing coordinator.
   - Purpose: Indexes dense vectors into Qdrant collections for fast similarity search
     and maintains an in-memory/disk store of complete ParentChunk context passages.

2. INPUT (IP):
   - chunks (list[ChildChunk]): Embedder-enriched ChildChunk models with dense vectors and metadata.
   - parents (list[ParentChunk]): Coarse context ParentChunk models.
   - Source: `src/pipeline_1_ingestion/chunker.py` and `src/pipeline_1_ingestion/embedder.py`.

3. PROCESS UNDER THE HOOD:
   - Initializes QdrantClient (`:memory:` or local persistent directory).
   - Verifies / creates the target collection (`child_chunks`) configured with Cosine distance metric.
   - Converts `ChildChunk` instances to Qdrant `PointStruct` entries using deterministic UUIDs
     (`uuid5(NAMESPACE_DNS, chunk_id)`).
   - Stores chunk metadata payloads (`chunk_id`, `parent_id`, `doc_id`, `page_number`, `text`, `sparse_tokens`, `chunk_index`).
   - Indexes `ParentChunk` objects in an internal key-value lookup store keyed by `parent_id`.

4. OUTPUT (OP):
   - Upserted points in Qdrant vector database.
   - Parent retrieval via `get_parent(parent_id) -> Optional[ParentChunk]`.
   - Consumed by: `src/pipeline_2_retrieval/search_dense.py` and `src/pipeline_2_retrieval/reranker.py`.

5. LIBRARIES & DEPENDENCIES:
   - qdrant_client: Vector database search engine client supporting in-memory and disk-based vector storage.
   - qdrant_client.http.models: Dataclasses for PointStruct, VectorParams, and Distance metrics.
   - uuid: Standard library for RFC 4122 compliant deterministic UUID generation.
   - src.common.schemas: Pydantic v2 schemas (`ChildChunk`, `ParentChunk`).
================================================================================
"""

import uuid
from typing import Dict, List, Optional
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.common.schemas import ChildChunk, ParentChunk


class LocalStore:
    """Manages dense vector indexing in Qdrant and parent document chunk retrieval."""

    def __init__(
        self,
        location: str = ":memory:",
        path: Optional[str] = None,
        collection_name: str = "child_chunks",
        vector_size: int = 384,
    ) -> None:
        """Initialize Qdrant client and parent cache.

        Args:
            location: Qdrant location (default ':memory:').
            path: Local storage path on disk (e.g. 'data/storage/qdrant').
            collection_name: Name of Qdrant collection for child vectors.
            vector_size: Dimension of dense embedding vectors (default 384 for bge-small).
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._parent_store: Dict[str, ParentChunk] = {}

        if path:
            self.client = QdrantClient(path=path)
        else:
            self.client = QdrantClient(location=location)

        self._init_collection()

    def _init_collection(self) -> None:
        """Create Qdrant collection if it does not already exist."""
        collections = self.client.get_collections().collections
        existing_names = {c.name for c in collections}

        if self.collection_name not in existing_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size,
                    distance=models.Distance.COSINE,
                ),
            )

    def upsert_child_chunks(self, chunks: List[ChildChunk]) -> None:
        """Upsert child chunk vectors and payloads into Qdrant.

        Args:
            chunks: List of ChildChunk instances (must include vector).
        """
        if not chunks:
            return

        points: List[models.PointStruct] = []
        for chunk in chunks:
            if chunk.vector is None:
                raise ValueError(f"ChildChunk {chunk.chunk_id} missing dense vector embedding.")

            # Generate deterministic UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            payload = {
                "chunk_id": chunk.chunk_id,
                "parent_id": chunk.parent_id,
                "doc_id": chunk.doc_id,
                "page_number": chunk.page_number,
                "text": chunk.text,
                "sparse_tokens": chunk.sparse_tokens,
                "chunk_index": getattr(chunk, "chunk_index", 0),
            }

            points.append(
                models.PointStruct(
                    id=point_id,
                    vector=chunk.vector,
                    payload=payload,
                )
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=points,
        )

    def store_parents(self, parents: List[ParentChunk]) -> None:
        """Store parent chunks in the local cache keyed by parent_id.

        Args:
            parents: List of ParentChunk instances.
        """
        for parent in parents:
            self._parent_store[parent.parent_id] = parent

    def get_parent(self, parent_id: str) -> Optional[ParentChunk]:
        """Retrieve a parent chunk by its parent_id.

        Args:
            parent_id: Unique identifier for the parent chunk.

        Returns:
            The ParentChunk instance if found, otherwise None.
        """
        return self._parent_store.get(parent_id)
