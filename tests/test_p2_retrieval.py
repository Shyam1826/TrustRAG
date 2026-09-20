r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_p2_retrieval.py
   - Role: Test suite for Pipeline 2 (Retrieval & Reranking).
   - Purpose: Validates query rewriting, dense search, BM25 sparse search, RRF rank
     fusion, cross-encoder reranking, and document-aware neighbor context expansion.

2. INPUT (IP):
   - Sample synthetic documents across 3 distinct domains (Hardware, HR, Network).
   - Natural language test query with conversational filler.

3. PROCESS UNDER THE HOOD:
   - Tests individual components: QueryTransformer, BM25Searcher, apply_rrf.
   - Executes full end-to-end retrieval and reranking flow against an in-memory store.
   - Tests document-aware neighbor expansion when adjacent parent chunks are retrieved.
   - Asserts parent resolution extracts full context without truncation.

4. OUTPUT (OP):
   - Pytest test execution assertions and reports.

5. LIBRARIES & DEPENDENCIES:
   - pytest: Test execution framework.
   - src.pipeline_1_ingestion.*: Ingestion components.
   - src.pipeline_2_retrieval.*: Retrieval & reranking components.
   - src.common.schemas: Pydantic schemas.
================================================================================
"""

import pytest

from src.common.schemas import ChildChunk, ParentChunk
from src.pipeline_1_ingestion.chunker import create_hierarchical_chunks
from src.pipeline_1_ingestion.embedder import DualEmbedder
from src.pipeline_1_ingestion.indexer import LocalStore
from src.pipeline_1_ingestion.vector_store import QdrantVectorStore
from src.pipeline_2_retrieval.rewriter import QueryTransformer
from src.pipeline_2_retrieval.search_dense import DenseSearcher, retrieve_dense
from src.pipeline_2_retrieval.search_sparse import BM25Searcher
from src.pipeline_2_retrieval.fusion import apply_rrf
from src.pipeline_2_retrieval.reranker import CrossEncoderReranker


def test_query_transformer_filler_removal():
    transformer = QueryTransformer()

    q1 = "Can you please tell me what is the TDP wattage of Model-X processor?"
    transformed1 = transformer.transform(q1)
    assert "Can you please tell me" not in transformed1
    assert "TDP wattage of Model-X processor" in transformed1

    q2 = "Please search for 192.168.1.1 subnet?"
    transformed2 = transformer.transform(q2)
    assert "Please search for" not in transformed2
    assert "192.168.1.1 subnet" in transformed2


def test_bm25_searcher_scoring():
    chunk1 = ChildChunk(
        chunk_id="c1",
        parent_id="p1",
        doc_id="d1",
        text="Hardware specs: Model-X processor operates at 125W TDP.",
        page_number=1,
    )
    chunk2 = ChildChunk(
        chunk_id="c2",
        parent_id="p2",
        doc_id="d2",
        text="Corporate policy: employees are entitled to 20 days PTO.",
        page_number=1,
    )
    chunk3 = ChildChunk(
        chunk_id="c3",
        parent_id="p3",
        doc_id="d3",
        text="Network guide: gateway router IP is 192.168.1.1 subnet.",
        page_number=1,
    )

    bm25 = BM25Searcher([chunk1, chunk2, chunk3])
    results = bm25.search(["tdp", "processor"], top_k=3)

    assert len(results) == 3
    assert results[0][0] == "c1"  # chunk1 must rank highest for 'tdp'
    assert results[0][1] == 1  # Rank 1
    assert results[0][2] > results[1][2]  # c1 has positive score higher than c2/c3
    assert results[0][2] > 0.0


def test_apply_rrf_fusion():
    dense_ranks = [("c1", 1, 0.95), ("c2", 2, 0.80)]
    sparse_ranks = [("c2", 1, 4.5), ("c1", 2, 3.2)]

    fused = apply_rrf(dense_ranks, sparse_ranks, k=60, top_n=2)
    assert len(fused) == 2
    # Both c1 and c2 are present with combined scores: 1/(61) + 1/(62)
    score_c1 = (1.0 / 61) + (1.0 / 62)
    score_c2 = (1.0 / 62) + (1.0 / 61)
    assert pytest.approx(fused[0][1], 1e-5) == score_c1
    assert pytest.approx(fused[1][1], 1e-5) == score_c2


def test_neighbor_context_expansion():
    # Test that adjacent parent chunks from the same doc are merged
    p1 = ParentChunk(parent_id="doc_p0", doc_id="doc_1", text="Section 1 Project Alpha", page_number=1, chunk_index=0)
    p2 = ParentChunk(parent_id="doc_p1", doc_id="doc_1", text="Section 2 Project Beta", page_number=1, chunk_index=1)
    p3 = ParentChunk(parent_id="doc_p5", doc_id="doc_2", text="Unrelated doc context", page_number=1, chunk_index=5)

    c1 = ChildChunk(chunk_id="c0", parent_id="doc_p0", doc_id="doc_1", text="Project Alpha details", page_number=1, chunk_index=0)
    c2 = ChildChunk(chunk_id="c1", parent_id="doc_p1", doc_id="doc_1", text="Project Beta details", page_number=1, chunk_index=1)
    c3 = ChildChunk(chunk_id="c2", parent_id="doc_p5", doc_id="doc_2", text="Unrelated text", page_number=1, chunk_index=5)

    store = LocalStore(location=":memory:")
    store.store_parents([p1, p2, p3])

    reranker = CrossEncoderReranker()
    child_map = {"c0": c1, "c1": c2, "c2": c3}

    candidates = reranker.rerank_and_resolve(
        query="Tell me about all projects",
        candidate_child_ids=["c0", "c1", "c2"],
        child_chunk_map=child_map,
        local_store=store,
        top_k=5,
    )

    assert len(candidates) == 2  # p1 and p2 merged into 1, plus p3
    doc1_candidate = [c for c in candidates if c.doc_id == "doc_1"][0]
    assert "Section 1 Project Alpha" in doc1_candidate.text
    assert "Section 2 Project Beta" in doc1_candidate.text
    assert doc1_candidate.match_type == "cross_encoder_neighbor_expanded"


def test_p2_end_to_end_retrieval_and_reranking():
    # 1. Prepare 3 distinct documents
    docs_data = [
        (
            "doc_1",
            "Hardware Specs: The Model-X processor has 16 physical cores and operates at 125W TDP.",
        ),
        (
            "doc_2",
            "Corporate Policies: All employees are entitled to 20 days of annual paid time off.",
        ),
        (
            "doc_3",
            "Network Guide: The gateway router IP is 192.168.1.1 with subnet 255.255.255.0.",
        ),
    ]

    all_parents: list[ParentChunk] = []
    all_children: list[ChildChunk] = []
    child_map: dict[str, ChildChunk] = {}

    for doc_id, text in docs_data:
        pages = [{"page_number": 1, "raw_text": text}]
        parents, children = create_hierarchical_chunks(
            pages,
            doc_id=doc_id,
            parent_size=1400,
            child_size=150,
            overlap=30,
        )
        all_parents.extend(parents)
        all_children.extend(children)
        for c in children:
            child_map[c.chunk_id] = c

    # 2. Embed and index documents
    embedder = DualEmbedder()
    dense_vectors = embedder.embed_dense([c.text for c in all_children])
    for child, vec in zip(all_children, dense_vectors):
        child.vector = vec
        child.sparse_tokens = embedder.tokenize_sparse(child.text)

    store = LocalStore(location=":memory:", vector_size=384)
    store.upsert_child_chunks(all_children)
    store.store_parents(all_parents)

    # 3. Process Query
    query = "What is the TDP wattage of Model-X processor?"
    transformer = QueryTransformer()
    clean_query = transformer.transform(query)
    assert "TDP wattage of Model-X processor" in clean_query

    # 4. Dense Retrieval
    dense_results = retrieve_dense(clean_query, embedder, store.client, top_k=10)
    assert len(dense_results) > 0

    # 5. Sparse Retrieval
    bm25 = BM25Searcher(all_children)
    query_tokens = [t.lower() for t in clean_query.split()]
    sparse_results = bm25.search(query_tokens, top_k=10)
    assert len(sparse_results) > 0

    # 6. Reciprocal Rank Fusion
    fused_candidates = apply_rrf(dense_results, sparse_results, k=60, top_n=10)
    candidate_cids = [cid for cid, _ in fused_candidates]
    assert len(candidate_cids) > 0

    # 7. Cross-Encoder Reranking and Parent Resolution
    reranker = CrossEncoderReranker()
    resolved_candidates = reranker.rerank_and_resolve(
        query=clean_query,
        candidate_child_ids=candidate_cids,
        child_chunk_map=child_map,
        local_store=store,
        top_k=5,
    )

    # Assertions
    assert len(resolved_candidates) <= 5
    assert len(resolved_candidates) == len(docs_data)  # All 3 unique parents

    # Top-1 returned RetrievalCandidate must be from Doc 1
    top_candidate = resolved_candidates[0]
    assert top_candidate.doc_id == "doc_1"
    assert "125W TDP" in top_candidate.text
    assert "Model-X processor" in top_candidate.text


def test_dense_searcher_balanced_quota_multi_doc():
    """Verify DenseSearcher enforces balanced retrieval quotas and deduplication by parent_id."""
    store = QdrantVectorStore(location=":memory:", vector_size=384)
    embedder = DualEmbedder()

    # Create multiple child chunks under the same parent for doc_a and doc_b
    p_a = ParentChunk(parent_id="pa_1", doc_id="doc_a", text="Parent context doc_a with MongoDB", page_number=1, section_name="Skills")
    p_b = ParentChunk(parent_id="pb_1", doc_id="doc_b", text="Parent context doc_b with PostgreSQL", page_number=1, section_name="Skills")
    store.store_parents([p_a, p_b])

    ca_1 = ChildChunk(chunk_id="ca_1", parent_id="pa_1", doc_id="doc_a", text="Database MongoDB", page_number=1, vector=embedder.embed_dense(["Database MongoDB"])[0])
    ca_2 = ChildChunk(chunk_id="ca_2", parent_id="pa_1", doc_id="doc_a", text="MongoDB queries", page_number=1, vector=embedder.embed_dense(["MongoDB queries"])[0])
    cb_1 = ChildChunk(chunk_id="cb_1", parent_id="pb_1", doc_id="doc_b", text="Database PostgreSQL", page_number=1, vector=embedder.embed_dense(["Database PostgreSQL"])[0])
    cb_2 = ChildChunk(chunk_id="cb_2", parent_id="pb_1", doc_id="doc_b", text="PostgreSQL schemas", page_number=1, vector=embedder.embed_dense(["PostgreSQL schemas"])[0])

    store.upsert_child_chunks([ca_1, ca_2, cb_1, cb_2])

    searcher = DenseSearcher(vector_store=store, embedder=embedder)
    results = searcher.search(
        query_text="Databases",
        top_k=4,
        doc_filter=["doc_a", "doc_b"],
    )

    # Balanced search deduplicates by parent_id, so only 1 chunk per unique parent is emitted
    assert len(results) == 2
    cids = [cid for cid, rank, score in results]
    assert "ca_1" in cids or "ca_2" in cids
    assert "cb_1" in cids or "cb_2" in cids
    # Ranks must be 1 and 2
    assert [r[1] for r in results] == [1, 2]

