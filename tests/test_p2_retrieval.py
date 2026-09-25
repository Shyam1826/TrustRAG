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


def test_query_transformer_folder_domain_extraction():
    """Verify that QueryTransformer extracts folder-based domain filters and strips folder prepositions."""
    transformer = QueryTransformer()
    known_docs = ["legal/2026/nda", "legal/2025/agreement", "engineering/specs"]

    # 1. "in legal" query
    doc_filter, tokens = transformer.extract_doc_filter("What are the agreements in legal?", known_doc_ids=known_docs)
    assert isinstance(doc_filter, list)
    assert set(doc_filter) == {"legal/2026/nda", "legal/2025/agreement"}
    clean_q = transformer.transform("What are the agreements in legal?", entities_to_strip=tokens)
    assert "agreements" in clean_q
    assert "legal" not in clean_q

    # 2. "under engineering" query
    doc_filter_eng, tokens_eng = transformer.extract_doc_filter("Show me the specs under engineering", known_doc_ids=known_docs)
    assert doc_filter_eng == "engineering/specs"
    clean_q_eng = transformer.transform("Show me the specs under engineering", entities_to_strip=tokens_eng)
    assert "specs" in clean_q_eng
    assert "engineering" not in clean_q_eng


def test_query_transformer_broad_and_comparative_global_retrieval():
    """Verify that broad, comparative, and multi-topic queries return doc_filter=None for 100% global semantic retrieval."""
    transformer = QueryTransformer()
    known_docs = ["engineering/specs", "legal/2026/nda", "candidates/sanjeev_resume", "candidates/vaishalee_resume"]

    # Comparative queries across documents
    doc_filter, tokens = transformer.extract_doc_filter("Compare the candidates and their experience with Python", known_doc_ids=known_docs)
    assert doc_filter is None
    assert tokens == []

    # Queries with "across" or "all"
    doc_filter_across, _ = transformer.extract_doc_filter("What are the specs across all documents?", known_doc_ids=known_docs)
    assert doc_filter_across is None

    # Technical query with variable name (no folder scoping, avoiding eager single-keyword token lock)
    doc_filter_var, _ = transformer.extract_doc_filter("What is d_model in the attention mechanism?", known_doc_ids=known_docs)
    assert doc_filter_var is None


def test_reranker_dynamic_document_diversification():
    """Verify that dynamic per-document quota prevents a large document from monopolizing top-k retrieval over a smaller document."""
    # Create LocalStore with Parent chunks for large manual (doc_manual) and short brief (doc_brief)
    store = LocalStore(location=":memory:")

    # 6 parent chunks for doc_manual (non-consecutive indices representing distinct sections)
    manual_parents = [
        ParentChunk(parent_id=f"p_man_{i}", doc_id="doc_manual", text=f"Manual section {i} architecture details", page_number=i, chunk_index=i * 2)
        for i in range(6)
    ]
    # 2 parent chunks for doc_brief
    brief_parents = [
        ParentChunk(parent_id=f"p_brf_{i}", doc_id="doc_brief", text=f"Brief section {i} summary overview", page_number=1, chunk_index=i * 2)
        for i in range(2)
    ]
    store.store_parents(manual_parents + brief_parents)

    # 6 child chunks for doc_manual (high semantic overlap)
    manual_children = [
        ChildChunk(chunk_id=f"c_man_{i}", parent_id=f"p_man_{i}", doc_id="doc_manual", text=f"Manual chunk {i} architecture", page_number=i, chunk_index=i * 2)
        for i in range(6)
    ]
    # 2 child chunks for doc_brief
    brief_children = [
        ChildChunk(chunk_id=f"c_brf_{i}", parent_id=f"p_brf_{i}", doc_id="doc_brief", text=f"Brief chunk {i} summary", page_number=1, chunk_index=i * 2)
        for i in range(2)
    ]

    child_map = {c.chunk_id: c for c in (manual_children + brief_children)}

    reranker = CrossEncoderReranker()
    candidate_cids = [c.chunk_id for c in (manual_children + brief_children)]

    # Request top_k=4 across 2 documents -> dynamic quota = 4 // 2 = 2 per doc
    resolved = reranker.rerank_and_resolve(
        query="architecture and summary overview",
        candidate_child_ids=candidate_cids,
        child_chunk_map=child_map,
        local_store=store,
        top_k=4,
    )

    assert len(resolved) == 4
    doc_counts = {}
    for cand in resolved:
        doc_counts[cand.doc_id] = doc_counts.get(cand.doc_id, 0) + 1

    # Both documents must have representation bounded by dynamic quota
    assert doc_counts.get("doc_manual", 0) == 2
    assert doc_counts.get("doc_brief", 0) == 2


def test_apply_rrf_dynamic_document_diversification():
    """Verify that apply_rrf enforces per-document dynamic quota diversification when child_chunk_map is provided."""
    # 5 chunks from large doc_a, 2 chunks from small doc_b
    child_map = {
        "ca_1": ChildChunk(chunk_id="ca_1", parent_id="pa_1", doc_id="doc_a", text="A1", page_number=1),
        "ca_2": ChildChunk(chunk_id="ca_2", parent_id="pa_2", doc_id="doc_a", text="A2", page_number=1),
        "ca_3": ChildChunk(chunk_id="ca_3", parent_id="pa_3", doc_id="doc_a", text="A3", page_number=1),
        "ca_4": ChildChunk(chunk_id="ca_4", parent_id="pa_4", doc_id="doc_a", text="A4", page_number=1),
        "cb_1": ChildChunk(chunk_id="cb_1", parent_id="pb_1", doc_id="doc_b", text="B1", page_number=1),
        "cb_2": ChildChunk(chunk_id="cb_2", parent_id="pb_2", doc_id="doc_b", text="B2", page_number=1),
    }

    # Dense ranks heavily favoring doc_a
    dense_ranks = [("ca_1", 1, 0.99), ("ca_2", 2, 0.98), ("ca_3", 3, 0.97), ("ca_4", 4, 0.96), ("cb_1", 5, 0.80)]
    sparse_ranks = [("ca_1", 1, 9.0), ("ca_2", 2, 8.0), ("cb_1", 3, 7.0), ("ca_3", 4, 6.0)]

    # Request top_n=3 with diversification enabled -> dynamic quota = 3 // 2 = 1 per doc, backfilled to 3
    fused = apply_rrf(dense_ranks, sparse_ranks, k=60, top_n=3, child_chunk_map=child_map)

    assert len(fused) == 3
    fused_cids = [cid for cid, _ in fused]
    # Both doc_a and doc_b must be present
    assert "cb_1" in fused_cids
    assert "ca_1" in fused_cids




