r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/vector_store.py
   - Role: Persistent enterprise vector store, point management, and parent-child metadata storage engine.
   - Purpose: Manages persistent disk-based or remote Qdrant vector database collections,
     stores dense child chunk embeddings with hydrated parent payloads, performs document-level
     point deletions for incremental updates, hydrates child and parent chunks during startup
     via scroll pagination, and executes native Qdrant payload-filtered nearest-neighbor searches.

2. INPUT (IP):
   - chunks (list[ChildChunk]): Embedder-enriched ChildChunk models with dense vectors and metadata.
   - parents (list[ParentChunk], optional): Coarse context ParentChunk models.
   - query_vector (list[float]): 384-dimensional query embedding vector for search.
   - doc_filter (str or list[str], optional): Document ID(s) to constrain search space.
   - doc_id (str): Target document ID to delete.
   - Source: `src/pipeline_1_ingestion/chunker.py`, `src/pipeline_1_ingestion/embedder.py`,
     and `src/pipeline_2_retrieval/search_dense.py`.

3. PROCESS UNDER THE HOOD:
   - Initializes official `qdrant_client.QdrantClient` targeting local disk persistence
     (default `path="data/qdrant_db"`), remote URL (`os.getenv("QDRANT_URL")`), or in-memory.
   - Idempotently creates collection `trustrag_enterprise` (384 dimensions, Cosine distance).
   - Generates deterministic RFC 4122 UUIDs (`uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id)`).
   - Ingests child chunk vectors alongside complete metadata payloads:
     * `doc_id`, `parent_id`, `child_id`, `section_path`, `child_text`, `parent_text`,
       `page_number`, `chunk_index`, `sparse_tokens`, `relative_path`, and `folder_hierarchy`.
   - `load_all_child_chunks()`: Scrolls all indexed points from Qdrant collection, reconstructs
     `ChildChunk` models, hydrates internal parent store cache, and returns complete list of chunks.
   - `delete_document(doc_id)`: Deletes all points matching `doc_id` from Qdrant collection
     and clears associated parents from local cache.
   - Executes native Qdrant filtered search using `models.MatchValue` or `models.MatchAny`.
   - Returns structured point dictionaries with hydrated parent text and similarity scores.

4. OUTPUT (OP):
   - Ingestion: Persisted points in Qdrant collection and parent chunk cache.
   - Hydration: list[ChildChunk] for startup state reconstruction and BM25 index re-fitting.
   - Deletion: Removed points and cache eviction.
   - Search: list[dict[str, Any]] containing `child_id`, `parent_id`, `doc_id`, `score`,
     `child_text`, `parent_text`, `relative_path`, `folder_hierarchy`, and metadata payloads.
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
    """Manages dense vector indexing, startup state hydration, and native payload-filtered retrieval in Qdrant."""

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
            try:
                self.client = QdrantClient(path=db_path)
            except RuntimeError as e:
                if "already accessed by another instance" in str(e):
                    self.client = QdrantClient(location=":memory:")
                else:
                    raise

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

            # Resolve relative_path and folder_hierarchy
            relative_path = getattr(chunk, "relative_path", None)
            if not relative_path and parent:
                relative_path = getattr(parent, "relative_path", None)
            if not relative_path:
                relative_path = chunk.doc_id

            folder_hierarchy = getattr(chunk, "folder_hierarchy", None)
            if folder_hierarchy is None and parent:
                folder_hierarchy = getattr(parent, "folder_hierarchy", None)
            if folder_hierarchy is None:
                if "/" in str(relative_path):
                    parts = str(relative_path).split("/")[:-1]
                    folder_hierarchy = [p for p in parts if p]
                elif "__" in str(relative_path):
                    parts = str(relative_path).split("__")[:-1]
                    folder_hierarchy = [p for p in parts if p]
                else:
                    folder_hierarchy = []

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
                "relative_path": relative_path,
                "folder_hierarchy": folder_hierarchy,
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

    def delete_document(self, doc_id: str) -> None:
        """Delete all points matching doc_id from Qdrant and clear associated parents from cache.

        Args:
            doc_id: Unique document identifier to remove.
        """
        if not doc_id:
            return

        delete_filter = models.Filter(
            must=[
                models.FieldCondition(
                    key="doc_id",
                    match=models.MatchValue(value=doc_id.strip()),
                )
            ]
        )
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=delete_filter),
        )

        # Clear parent cache entries matching doc_id
        parent_keys_to_delete = [
            pid for pid, parent in self._parent_store.items()
            if getattr(parent, "doc_id", "") == doc_id
        ]
        for pid in parent_keys_to_delete:
            self._parent_store.pop(pid, None)

    def load_all_child_chunks(self) -> List[ChildChunk]:
        """Scroll all indexed points from Qdrant, hydrate parent store, and return ChildChunk instances.

        Returns:
            List of reconstructed ChildChunk models from persistent vector store.
        """
        all_chunks: List[ChildChunk] = []
        offset = None

        while True:
            scroll_result, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in scroll_result:
                payload = point.payload or {}
                chunk_id = payload.get("child_id") or str(point.id)
                parent_id = payload.get("parent_id", "")
                doc_id = payload.get("doc_id", "")
                child_text = payload.get("child_text") or payload.get("text", "")
                section_name = payload.get("section_path") or payload.get("section_name", "General")
                page_number = payload.get("page_number", 1)
                chunk_index = payload.get("chunk_index", 0)
                relative_path = payload.get("relative_path")
                folder_hierarchy = payload.get("folder_hierarchy")
                sparse_tokens = payload.get("sparse_tokens")

                chunk = ChildChunk(
                    chunk_id=chunk_id,
                    parent_id=parent_id,
                    doc_id=doc_id,
                    text=child_text,
                    vector=None,
                    sparse_tokens=sparse_tokens,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    section_name=section_name,
                    relative_path=relative_path,
                    folder_hierarchy=folder_hierarchy,
                )
                all_chunks.append(chunk)

                # Hydrate parent store if parent_text is present
                parent_text = payload.get("parent_text")
                if parent_id and parent_text:
                    if parent_id not in self._parent_store:
                        self._parent_store[parent_id] = ParentChunk(
                            parent_id=parent_id,
                            doc_id=doc_id,
                            text=parent_text,
                            page_number=page_number,
                            child_ids=[chunk_id],
                            chunk_index=chunk_index,
                            section_name=section_name,
                            relative_path=relative_path,
                            folder_hierarchy=folder_hierarchy,
                        )
                    else:
                        if chunk_id not in self._parent_store[parent_id].child_ids:
                            self._parent_store[parent_id].child_ids.append(chunk_id)

            if next_offset is None:
                break
            offset = next_offset

        return all_chunks

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
                    "relative_path": payload.get("relative_path", ""),
                    "folder_hierarchy": payload.get("folder_hierarchy", []),
                    "payload": payload,
                }
            )

        return results

    def close(self) -> None:
        """Close Qdrant client connection and release local filesystem locks."""
        if hasattr(self, "client") and self.client is not None:
            try:
                self.client.close()
            except Exception:
                pass


# Backward compatibility alias
LocalStore = QdrantVectorStore
