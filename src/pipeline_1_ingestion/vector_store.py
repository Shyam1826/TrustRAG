r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/vector_store.py
   - Role: Persistent enterprise vector store and parent-child metadata storage engine.
   - Purpose: Manages persistent disk-based or remote Qdrant vector database collections,
     stores dense child chunk embeddings with hydrated parent payloads, and executes
     native Qdrant payload-filtered nearest-neighbor vector similarity searches.

2. INPUT (IP):
   - chunks (list[ChildChunk]): Embedder-enriched ChildChunk models with dense vectors and metadata.
   - parents (list[ParentChunk], optional): Coarse context ParentChunk models.
   - query_vector (list[float]): 384-dimensional query embedding vector for search.
   - doc_filter (str or list[str], optional): Document ID(s) to constrain search space.
   - Source: `src/pipeline_1_ingestion/chunker.py`, `src/pipeline_1_ingestion/embedder.py`,
     and `src/pipeline_2_retrieval/search_dense.py`.

3. PROCESS UNDER THE HOOD:
   - Initializes official `qdrant_client.QdrantClient` targeting local disk persistence
     (default `path="data/qdrant_db"`), remote URL (`os.getenv("QDRANT_URL")`), or in-memory.
   - Idempotently creates collection `trustrag_enterprise` (384 dimensions, Cosine distance).
   - Generates deterministic RFC 4122 UUIDs (`uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)`).
   - Ingests child chunk vectors alongside complete metadata payloads:
     * `doc_id`, `parent_id`, `child_id`, `section_path`, `child_text`, `parent_text`,
       `page_number`, `chunk_index`, and `sparse_tokens`.
   - Executes native Qdrant filtered search using `models.MatchValue` (single string filter)
     or `models.MatchAny` (multi-doc list filter).
   - Returns structured point dictionaries with hydrated parent text and similarity scores.

4. OUTPUT (OP):
   - Ingestion: Persisted points in Qdrant collection and parent chunk cache.
   - Search: list[dict[str, Any]] containing `child_id`, `parent_id`, `doc_id`, `score`,
     `child_text`, `parent_text`, and metadata payloads.
   - Consumed by: `src/pipeline_2_retrieval/search_dense.py` and `src/main.py`.

5. LIBRARIES & DEPENDENCIES:
   - qdrant_client: Official Qdrant vector database client.
   - qdrant_client.http.models: Collection configuration, PointStruct, Filter, and Match models.
   - uuid: Deterministic UUID generation.
   - os: Reading environment variables and directory creation.
   - src.common.schemas: Strict schemas (`ChildChunk`, `ParentChunk`).
================================================================================
"""

import os
import uuid
from typing import Any, Dict, List, Optional, Union
from qdrant_client import QdrantClient
from qdrant_client.http import models

from src.common.schemas import ChildChunk, ParentChunk


class QdrantVectorStore:
    """Manages dense vector indexing and native payload-filtered retrieval in Qdrant."""

    def __init__(
        self,
        location: Optional[str] = None,
        path: Optional[str] = None,
        url: Optional[str] = None,
        collection_name: str = "trustrag_enterprise",
        vector_size: int = 384,
        api_key: Optional[str] = None,
    ) -> None:
        """Initialize Qdrant client and parent chunk storage.

        Args:
            location: Optional direct location (e.g. ':memory:').
            path: Local storage path on disk (default: 'data/qdrant_db').
            url: Remote Qdrant server URL (or from QDRANT_URL env var).
            collection_name: Name of Qdrant collection (default: 'trustrag_enterprise').
            vector_size: Dimension of dense embedding vectors (default 384 for bge-small).
            api_key: Optional API key for remote Qdrant cluster.
        """
        self.collection_name = collection_name
        self.vector_size = vector_size
        self._parent_store: Dict[str, ParentChunk] = {}

        # Resolve connection mode
        target_url = url or os.getenv("QDRANT_URL")
        target_api_key = api_key or os.getenv("QDRANT_API_KEY")

        if target_url:
            self.client = QdrantClient(url=target_url, api_key=target_api_key)
        elif location == ":memory:":
            self.client = QdrantClient(location=":memory:")
        else:
            db_path = path or "data/qdrant_db"
            os.makedirs(db_path, exist_ok=True)
            self.client = QdrantClient(path=db_path)

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

    def upsert(
        self,
        chunks: List[ChildChunk],
        parents: Optional[List[ParentChunk]] = None,
    ) -> None:
        """Upsert child chunk vectors and hydrated parent metadata into Qdrant.

        Args:
            chunks: List of ChildChunk instances (must include vector).
            parents: Optional list of ParentChunk instances to hydrate parent payloads.
        """
        if not chunks:
            return

        if parents:
            self.store_parents(parents)

        points: List[models.PointStruct] = []
        for chunk in chunks:
            if chunk.vector is None:
                raise ValueError(f"ChildChunk {chunk.chunk_id} missing dense vector embedding.")

            # Generate deterministic UUID from chunk_id
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))

            # Hydrate parent text from store if available
            parent = self._parent_store.get(chunk.parent_id)
            parent_text = parent.text if parent else getattr(chunk, "text", "")
            section_path = getattr(chunk, "section_name", "General")

            payload = {
                "doc_id": chunk.doc_id,
                "parent_id": chunk.parent_id,
                "child_id": chunk.chunk_id,
                "section_path": section_path,
                "child_text": chunk.text,
                "parent_text": parent_text,
                "page_number": chunk.page_number,
                "sparse_tokens": getattr(chunk, "sparse_tokens", None),
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

    def upsert_child_chunks(self, chunks: List[ChildChunk]) -> None:
        """Backward-compatible alias for upserting child chunks."""
        self.upsert(chunks)

    def store_parents(self, parents: List[ParentChunk]) -> None:
        """Store parent chunks in local cache keyed by parent_id.

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

    def search(
        self,
        query_vector: List[float],
        limit: int = 20,
        doc_filter: Optional[Union[str, List[str]]] = None,
    ) -> List[Dict[str, Any]]:
        """Search nearest neighbors with native payload filtering.

        Args:
            query_vector: Dense embedding vector for query.
            limit: Maximum points to retrieve.
            doc_filter: Optional single doc_id string or list of doc_ids.

        Returns:
            List of result dictionaries containing scores and hydrated payloads.
        """
        query_filter: Optional[models.Filter] = None

        if isinstance(doc_filter, str) and doc_filter.strip():
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="doc_id",
                        match=models.MatchValue(value=doc_filter.strip()),
                    )
                ]
            )
        elif isinstance(doc_filter, list) and len(doc_filter) > 0:
            clean_filters = [d.strip() for d in doc_filter if isinstance(d, str) and d.strip()]
            if len(clean_filters) == 1:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchValue(value=clean_filters[0]),
                        )
                    ]
                )
            elif len(clean_filters) > 1:
                query_filter = models.Filter(
                    must=[
                        models.FieldCondition(
                            key="doc_id",
                            match=models.MatchAny(any=clean_filters),
                        )
                    ]
                )

        search_results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            query_filter=query_filter,
            limit=limit,
            with_payload=True,
        )

        results: List[Dict[str, Any]] = []
        for point in search_results:
            payload = point.payload or {}
            results.append(
                {
                    "id": point.id,
                    "score": float(point.score),
                    "child_id": payload.get("child_id", str(point.id)),
                    "parent_id": payload.get("parent_id", ""),
                    "doc_id": payload.get("doc_id", ""),
                    "section_path": payload.get("section_path", "General"),
                    "child_text": payload.get("child_text", ""),
                    "parent_text": payload.get("parent_text", ""),
                    "page_number": payload.get("page_number", 1),
                    "chunk_index": payload.get("chunk_index", 0),
                    "payload": payload,
                }
            )

        return results


# Backward compatibility alias
LocalStore = QdrantVectorStore
