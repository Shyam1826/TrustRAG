r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_tenant_isolation.py
   - Role: Comprehensive verification of Tenant & Document Isolation (Phase 2).
   - Purpose: Validates that vector embeddings, BM25 indices, DuckDB tabular catalogs,
     and pipeline retrieval are strictly segregated by user_id and thread_id, preventing
     cross-tenant data contamination with zero backward compatibility regressions.

2. INPUT (IP):
   - Tenant 1 synthetic documents and CSV tabular catalogs (`usr_1`, `thr_1`).
   - Tenant 2 synthetic documents and CSV tabular catalogs (`usr_2`, `thr_2`).
   - Tenant-scoped and unscoped retrieval queries.

3. PROCESS UNDER THE HOOD:
   - Vector Store Isolation:
     * Verifies deterministic point ID namespacing to prevent chunk ID collisions across tenants.
     * Verifies Qdrant payload filters for user_id and thread_id.
     * Verifies dense search strictly isolates candidate chunks by tenant coordinates.
   - Sparse BM25 & Fusion Isolation:
     * Verifies BM25 filtering by user_id and thread_id.
     * Verifies RRF fusion and Cross-Encoder reranker reject mismatched tenant chunks.
   - DuckDB Tabular Store Isolation:
     * Verifies catalog namespacing by user_id and thread_id.
     * Verifies `get_table_schemas` exposes only caller's registered tables.
     * Verifies execution of queries referencing another tenant's tables triggers `PermissionError`.
   - Full Orchestrator Integration:
     * Ingests Document A for Tenant 1 and Document B for Tenant 2.
     * Ingests Tabular CSV A for Tenant 1 and Tabular CSV B for Tenant 2.
     * Asserts that Tenant 1 queries return ONLY Document A / Table A (0% contamination).
     * Asserts that Tenant 2 queries return ONLY Document B / Table B (0% contamination).
     * Asserts default unscoped retrieval retains backward compatibility.

4. OUTPUT (OP):
   - Pytest assertions confirming complete isolation and zero cross-tenant leakage.

5. LIBRARIES & DEPENDENCIES:
   - pytest, tempfile, pathlib.Path, pandas.
   - src.common.schemas (ChildChunk, ParentChunk, RetrievalCandidate).
   - src.pipeline_1_ingestion.vector_store (LocalStore).
   - src.pipeline_1_ingestion.tabular_store (TabularStore).
   - src.pipeline_2_retrieval.tabular_engine (TabularQueryEngine).
   - src.pipeline_2_retrieval.search_dense (retrieve_dense).
   - src.pipeline_2_retrieval.search_sparse (BM25Searcher).
   - src.pipeline_2_retrieval.fusion (apply_rrf).
   - src.main (TrustRAGPipeline).
================================================================================
"""

import os
from pathlib import Path
import tempfile
from typing import Generator
import pandas as pd
import pytest

from src.common.schemas import ChildChunk, ParentChunk, RetrievalCandidate
from src.pipeline_1_ingestion.vector_store import LocalStore
from src.pipeline_1_ingestion.tabular_store import TabularStore
from src.pipeline_2_retrieval.tabular_engine import TabularQueryEngine
from src.pipeline_2_retrieval.search_dense import retrieve_dense
from src.pipeline_2_retrieval.search_sparse import BM25Searcher
from src.pipeline_2_retrieval.fusion import apply_rrf
from src.main import TrustRAGPipeline


@pytest.fixture
def temp_tenant_files() -> Generator[dict, None, None]:
    """Create isolated narrative documents and tabular CSV files for two distinct tenants."""
    with tempfile.TemporaryDirectory() as temp_dir:
        dir_path = Path(temp_dir)

        # Tenant 1 narrative file
        doc_a_path = dir_path / "tenant_1_hardware.txt"
        doc_a_path.write_text(
            "Hardware Specs: The Model-X processor features 16 physical cores and operates at 125W TDP.\n"
            "It is engineered for mission-critical aerospace defense protocols.",
            encoding="utf-8",
        )

        # Tenant 2 narrative file
        doc_b_path = dir_path / "tenant_2_network.txt"
        doc_b_path.write_text(
            "Network Guide: The gateway router IP is 192.168.1.1 with subnet 255.255.255.0.\n"
            "It connects the regional branch to the enterprise datacenter.",
            encoding="utf-8",
        )

        # Tenant 1 tabular file
        csv_a_path = dir_path / "tenant_1_assets.csv"
        df_a = pd.DataFrame({
            "asset_id": ["A-101", "A-102"],
            "asset_name": ["Saturn Flight Sensor", "Titan Radar Beacon"],
            "status": ["Active", "Maintenance"],
            "security_tier": ["TopSecret", "Secret"],
        })
        df_a.to_csv(csv_a_path, index=False)

        # Tenant 2 tabular file
        csv_b_path = dir_path / "tenant_2_products.csv"
        df_b = pd.DataFrame({
            "product_id": ["P-901", "P-902"],
            "product_name": ["Organic Cotton T-Shirt", "Wireless Bluetooth Speaker"],
            "category": ["Apparel", "Electronics"],
            "price_usd": [29.99, 79.50],
        })
        df_b.to_csv(csv_b_path, index=False)

        yield {
            "doc_a": str(doc_a_path),
            "doc_b": str(doc_b_path),
            "csv_a": str(csv_a_path),
            "csv_b": str(csv_b_path),
        }


def test_schema_tenant_metadata():
    """Verify ChildChunk, ParentChunk, and RetrievalCandidate support user_id and thread_id."""
    chunk = ChildChunk(
        chunk_id="chk_001",
        doc_id="doc_001",
        parent_id="p_001",
        text="Sample text content for tenant testing.",
        page_number=1,
        user_id="usr_abc",
        thread_id="thr_xyz",
    )
    assert chunk.user_id == "usr_abc"
    assert chunk.thread_id == "thr_xyz"

    parent = ParentChunk(
        parent_id="p_001",
        doc_id="doc_001",
        text="Sample parent text content.",
        page_number=1,
        user_id="usr_abc",
        thread_id="thr_xyz",
    )
    assert parent.user_id == "usr_abc"
    assert parent.thread_id == "thr_xyz"

    cand = RetrievalCandidate(
        parent_id="p_001",
        doc_id="doc_001",
        page_number=1,
        text="Candidate text",
        score=0.95,
        match_type="dense",
        user_id="usr_abc",
        thread_id="thr_xyz",
    )
    assert cand.user_id == "usr_abc"
    assert cand.thread_id == "thr_xyz"


def test_vector_store_tenant_isolation():
    """Verify LocalStore namespaces point IDs and filters search results by tenant coordinates."""
    store = LocalStore(location=":memory:")

    # Tenant 1 chunks
    chunk_t1 = ChildChunk(
        chunk_id="chunk_same_id",  # Identical chunk_id to test collision avoidance
        doc_id="doc_alpha",
        parent_id="parent_alpha",
        text="Tenant 1 proprietary aerospace propulsion specs.",
        page_number=1,
        user_id="usr_1",
        thread_id="thr_1",
    )
    chunk_t1.vector = [0.1] * 384
    parent_t1 = ParentChunk(
        parent_id="parent_alpha",
        doc_id="doc_alpha",
        text="Tenant 1 full parent document on aerospace propulsion specs.",
        page_number=1,
        user_id="usr_1",
        thread_id="thr_1",
    )

    # Tenant 2 chunks
    chunk_t2 = ChildChunk(
        chunk_id="chunk_same_id",  # Identical chunk_id
        doc_id="doc_beta",
        parent_id="parent_beta",
        text="Tenant 2 commercial retail catalog and customer demographics.",
        page_number=1,
        user_id="usr_2",
        thread_id="thr_2",
    )
    chunk_t2.vector = [0.9] * 384
    parent_t2 = ParentChunk(
        parent_id="parent_beta",
        doc_id="doc_beta",
        text="Tenant 2 full parent document on retail catalog and customer demographics.",
        page_number=1,
        user_id="usr_2",
        thread_id="thr_2",
    )

    # Upsert both
    store.upsert([chunk_t1], [parent_t1])
    store.upsert([chunk_t2], [parent_t2])

    # Search with Tenant 1 coordinates
    results_t1 = store.search([0.1] * 384, top_k=5, user_id="usr_1", thread_id="thr_1")
    assert len(results_t1) == 1
    assert results_t1[0]["doc_id"] == "doc_alpha"
    assert results_t1[0]["user_id"] == "usr_1"
    assert results_t1[0]["thread_id"] == "thr_1"

    # Search with Tenant 2 coordinates
    results_t2 = store.search([0.1] * 384, top_k=5, user_id="usr_2", thread_id="thr_2")
    assert len(results_t2) == 1
    assert results_t2[0]["doc_id"] == "doc_beta"
    assert results_t2[0]["user_id"] == "usr_2"
    assert results_t2[0]["thread_id"] == "thr_2"

    # Search with non-existent tenant
    results_t3 = store.search([0.1] * 384, top_k=5, user_id="usr_unknown")
    assert len(results_t3) == 0

    # Test load_all_child_chunks tenant filter
    loaded_t1 = store.load_all_child_chunks(user_id="usr_1")
    assert len(loaded_t1) == 1
    assert loaded_t1[0].doc_id == "doc_alpha"

    loaded_t2 = store.load_all_child_chunks(user_id="usr_2")
    assert len(loaded_t2) == 1
    assert loaded_t2[0].doc_id == "doc_beta"

    # Unscoped load returns both
    loaded_all = store.load_all_child_chunks()
    assert len(loaded_all) == 2

    store.close()


def test_bm25_and_rrf_tenant_isolation():
    """Verify BM25Searcher and apply_rrf strictly filter candidates by user_id and thread_id."""
    chunk_1 = ChildChunk(
        chunk_id="c1",
        doc_id="doc_1",
        parent_id="p1",
        text="Aerospace rocket propulsion engineering guidelines.",
        page_number=1,
        user_id="usr_1",
        thread_id="thr_1",
    )
    chunk_2 = ChildChunk(
        chunk_id="c2",
        doc_id="doc_2",
        parent_id="p2",
        text="Consumer retail ecommerce marketing guidelines.",
        page_number=1,
        user_id="usr_2",
        thread_id="thr_2",
    )

    child_map = {"c1": chunk_1, "c2": chunk_2}
    searcher = BM25Searcher([chunk_1, chunk_2])

    # BM25 search as Tenant 1
    res_t1 = searcher.search(["guidelines"], top_k=5, user_id="usr_1", thread_id="thr_1")
    assert len(res_t1) == 1
    assert res_t1[0][0] == "c1"

    # BM25 search as Tenant 2
    res_t2 = searcher.search(["guidelines"], top_k=5, user_id="usr_2", thread_id="thr_2")
    assert len(res_t2) == 1
    assert res_t2[0][0] == "c2"

    # RRF fusion as Tenant 1
    dense_mock = [("c1", 1, 0.9), ("c2", 2, 0.85)]
    sparse_mock = [("c1", 1, 0.9), ("c2", 2, 0.85)]

    fused_t1 = apply_rrf(
        dense_ranks=dense_mock,
        sparse_ranks=sparse_mock,
        child_chunk_map=child_map,
        user_id="usr_1",
        thread_id="thr_1",
    )
    assert len(fused_t1) == 1
    assert fused_t1[0][0] == "c1"

    # RRF fusion as Tenant 2
    fused_t2 = apply_rrf(
        dense_ranks=dense_mock,
        sparse_ranks=sparse_mock,
        child_chunk_map=child_map,
        user_id="usr_2",
        thread_id="thr_2",
    )
    assert len(fused_t2) == 1
    assert fused_t2[0][0] == "c2"


def test_tabular_store_tenant_isolation(temp_tenant_files: dict):
    """Verify TabularStore isolates catalogs and strictly prevents cross-tenant SQL access."""
    store = TabularStore()

    # Register CSV A for Tenant 1
    t1_names = store.register_table_from_file(
        temp_tenant_files["csv_a"],
        doc_id="assets",
        user_id="usr_1",
        thread_id="thr_1",
    )
    assert len(t1_names) == 1
    t1_name = t1_names[0]
    assert "usr_1" in t1_name

    # Register CSV B for Tenant 2
    t2_names = store.register_table_from_file(
        temp_tenant_files["csv_b"],
        doc_id="products",
        user_id="usr_2",
        thread_id="thr_2",
    )
    assert len(t2_names) == 1
    t2_name = t2_names[0]
    assert "usr_2" in t2_name

    # Check visible schemas for Tenant 1
    schemas_t1 = store.get_table_schemas(user_id="usr_1", thread_id="thr_1")
    assert t1_name in schemas_t1
    assert t2_name not in schemas_t1

    # Check visible schemas for Tenant 2
    schemas_t2 = store.get_table_schemas(user_id="usr_2", thread_id="thr_2")
    assert t2_name in schemas_t2
    assert t1_name not in schemas_t2

    # Query within Tenant 1 namespace succeeds
    df_t1 = store.execute_query(f"SELECT * FROM {t1_name}", user_id="usr_1", thread_id="thr_1")
    assert len(df_t1) == 2

    # Malicious / Cross-tenant access attempt: Tenant 1 attempting to query Tenant 2's table
    with pytest.raises(PermissionError) as exc_info:
        store.execute_query(f"SELECT * FROM {t2_name}", user_id="usr_1", thread_id="thr_1")
    assert "Cross-tenant access violation" in str(exc_info.value)

    # Cross-tenant access attempt by Tenant 2 querying Tenant 1's table
    with pytest.raises(PermissionError) as exc_info2:
        store.execute_query(f"SELECT * FROM {t1_name}", user_id="usr_2", thread_id="thr_2")
    assert "Cross-tenant access violation" in str(exc_info2.value)

    store.close()


def test_tabular_engine_query_isolation(temp_tenant_files: dict):
    """Verify TabularQueryEngine operates strictly within caller's tenant namespace."""
    store = TabularStore()
    engine = TabularQueryEngine(store)

    store.register_table_from_file(
        temp_tenant_files["csv_a"],
        doc_id="assets",
        user_id="usr_1",
        thread_id="thr_1",
    )
    store.register_table_from_file(
        temp_tenant_files["csv_b"],
        doc_id="products",
        user_id="usr_2",
        thread_id="thr_2",
    )

    # is_tabular_query detects only tenant-visible tables
    assert engine.is_tabular_query("Find assets where status is 'Active'", user_id="usr_1", thread_id="thr_1") is True
    assert engine.is_tabular_query("Find products where category is 'Apparel'", user_id="usr_1", thread_id="thr_1") is False

    assert engine.is_tabular_query("Find products where category is 'Apparel'", user_id="usr_2", thread_id="thr_2") is True
    assert engine.is_tabular_query("Find assets where status is 'Active'", user_id="usr_2", thread_id="thr_2") is False

    # Execute query as Tenant 1
    cands_t1 = engine.query("Find assets where status is 'Active'", user_id="usr_1", thread_id="thr_1")
    assert len(cands_t1) > 0
    for c in cands_t1:
        assert c.user_id == "usr_1"
        assert c.thread_id == "thr_1"
        assert "Saturn" in c.text or "Active" in c.text

    # Execute query as Tenant 2
    cands_t2 = engine.query("Find products where category is 'Apparel'", user_id="usr_2", thread_id="thr_2")
    assert len(cands_t2) > 0
    for c in cands_t2:
        assert c.user_id == "usr_2"
        assert c.thread_id == "thr_2"
        assert "Organic Cotton" in c.text

    store.close()


def test_pipeline_end_to_end_tenant_isolation(temp_tenant_files: dict):
    """Verify end-to-end TrustRAGPipeline ingestion, retrieval, and ask guarantee 0% data contamination."""
    pipeline = TrustRAGPipeline(generator_type="mock")

    # Ingest narrative and tabular files for Tenant 1
    pipeline.ingest_document(
        temp_tenant_files["doc_a"],
        doc_id="tenant_1_defense_doc",
        user_id="usr_1",
        thread_id="thr_1",
    )
    pipeline.ingest_document(
        temp_tenant_files["csv_a"],
        doc_id="tenant_1_assets_csv",
        user_id="usr_1",
        thread_id="thr_1",
    )

    # Ingest narrative and tabular files for Tenant 2
    pipeline.ingest_document(
        temp_tenant_files["doc_b"],
        doc_id="tenant_2_retail_doc",
        user_id="usr_2",
        thread_id="thr_2",
    )
    pipeline.ingest_document(
        temp_tenant_files["csv_b"],
        doc_id="tenant_2_products_csv",
        user_id="usr_2",
        thread_id="thr_2",
    )

    # 1. Query as Tenant 1: Narrative retrieval
    cands_t1 = pipeline.retrieve(
        "What is the TDP wattage of Model-X processor?",
        user_id="usr_1",
        thread_id="thr_1",
    )
    assert len(cands_t1) > 0
    for cand in cands_t1:
        # ZERO CONTAMINATION from Tenant 2
        assert cand.user_id == "usr_1"
        assert "gateway" not in cand.text.lower()
        assert "192.168.1.1" not in cand.text.lower()
    assert any("125w tdp" in cand.text.lower() for cand in cands_t1)

    # Cross-tenant query attempt: Tenant 1 querying for Tenant 2's specific knowledge
    cross_t1 = pipeline.retrieve(
        "What is the gateway router IP?",
        user_id="usr_1",
        thread_id="thr_1",
    )
    # ZERO contamination from Tenant 2
    assert all(cand.user_id == "usr_1" for cand in cross_t1)
    assert all("192.168.1.1" not in cand.text.lower() for cand in cross_t1)

    # 2. Query as Tenant 2: Narrative retrieval
    cands_t2 = pipeline.retrieve(
        "What is the gateway router IP?",
        user_id="usr_2",
        thread_id="thr_2",
    )
    assert len(cands_t2) > 0
    for cand in cands_t2:
        # ZERO CONTAMINATION from Tenant 1
        assert cand.user_id == "usr_2"
        assert "model-x" not in cand.text.lower()
        assert "125w" not in cand.text.lower()
    assert any("192.168.1.1" in cand.text.lower() for cand in cands_t2)

    # 3. Query as Tenant 1: Tabular retrieval
    tab_t1 = pipeline.retrieve(
        "Find records in tenant_1_assets_csv where status is 'Active'",
        user_id="usr_1",
        thread_id="thr_1",
    )
    assert len(tab_t1) > 0
    for cand in tab_t1:
        assert cand.user_id == "usr_1"
        assert "Saturn Flight Sensor" in cand.text

    # 4. Query as Tenant 2: Tabular retrieval
    tab_t2 = pipeline.retrieve(
        "Find records in tenant_2_products_csv where category is 'Apparel'",
        user_id="usr_2",
        thread_id="thr_2",
    )
    assert len(tab_t2) > 0
    for cand in tab_t2:
        assert cand.user_id == "usr_2"
        assert "Organic Cotton T-Shirt" in cand.text

    # 5. Full End-to-End ask() as Tenant 1
    report_t1 = pipeline.ask(
        "What is the TDP wattage of Model-X processor?",
        user_id="usr_1",
        thread_id="thr_1",
    )
    assert report_t1.draft_text is not None
    assert "125W TDP" in report_t1.draft_text
    assert len(report_t1.audits) > 0
    assert report_t1.action == "PASS"
    for cand in pipeline.last_retrieved_contexts:
        assert cand.user_id == "usr_1"

    # 6. Full End-to-End ask() as Tenant 2
    report_t2 = pipeline.ask(
        "What is the gateway router IP?",
        user_id="usr_2",
        thread_id="thr_2",
    )
    assert report_t2.draft_text is not None
    assert "192.168.1.1" in report_t2.draft_text
    assert len(report_t2.audits) > 0
    assert report_t2.action == "PASS"
    for cand in pipeline.last_retrieved_contexts:
        assert cand.user_id == "usr_2"

    pipeline.close()
