r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_incremental_ingestion.py
   - Role: Test suite for incremental SHA-256 manifest caching, Qdrant deletion, and startup state hydration.
   - Purpose: Validates incremental document caching, idempotent skips, document deletion from Qdrant,
     point scrolling, and in-memory state/BM25 hydration upon startup when documents are skipped.

2. INPUT (IP):
   - Synthetic PDF documents created in temporary directories.

3. PROCESS UNDER THE HOOD:
   - Tests file hashing and lifecycle updates in `IngestionManifest`.
   - Tests Qdrant vector deletion and parent cache clearance.
   - Tests scroll-based chunk hydration via `load_all_child_chunks()`.
   - Tests `TrustRAGPipeline.ingest_directory()` startup hydration when files are skipped by manifest.

4. OUTPUT (OP):
   - Pytest assertions verifying incremental ingestion integrity and state hydration.

5. LIBRARIES & DEPENDENCIES:
   - pytest, tempfile, pathlib, fitz (PyMuPDF).
   - src.pipeline_1_ingestion.manifest, src.pipeline_1_ingestion.vector_store, src.main.
================================================================================
"""

import tempfile
from pathlib import Path
import fitz
import pytest

from src.pipeline_1_ingestion.manifest import IngestionManifest
from src.pipeline_1_ingestion.vector_store import QdrantVectorStore
from src.main import TrustRAGPipeline


def _create_test_pdf(file_path: Path, content: str) -> Path:

    """Helper to create a test PDF."""
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((50, 72), content, fontsize=11)
    doc.save(str(file_path))
    doc.close()
    return file_path


def test_manifest_file_hashing_and_lifecycle():
    """Verify compute_file_hash, record_indexed, is_indexed_and_current, and remove_entry."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        manifest_file = tmp_path / "manifest.json"
        pdf_file = tmp_path / "contract.pdf"
        _create_test_pdf(pdf_file, "Article 1: Confidential Information terms.")

        manifest = IngestionManifest(manifest_path=manifest_file)
        doc_id = "legal/contract"

        # Initially not indexed
        assert not manifest.is_indexed_and_current(pdf_file, doc_id)

        # Record indexed
        manifest.record_indexed(pdf_file, doc_id=doc_id, chunk_count=3)
        assert manifest.is_indexed_and_current(pdf_file, doc_id)

        # Reload from disk to verify persistence
        manifest_reloaded = IngestionManifest(manifest_path=manifest_file)
        assert manifest_reloaded.is_indexed_and_current(pdf_file, doc_id)
        assert doc_id in manifest_reloaded.get_indexed_doc_ids()

        # Modify file content -> hash and size change -> should report not current
        _create_test_pdf(pdf_file, "Article 1: Revised Confidentiality and Non-Disclosure terms.")
        assert not manifest_reloaded.is_indexed_and_current(pdf_file, doc_id)

        # Remove entry
        manifest_reloaded.remove_entry(doc_id)
        assert doc_id not in manifest_reloaded.get_indexed_doc_ids()


def test_manifest_prune_missing_files():
    """Verify that deleting files on disk allows manifest.prune_missing_files() to clean up."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        manifest_file = tmp_path / "manifest.json"
        pdf_file = tmp_path / "temp_spec.pdf"
        _create_test_pdf(pdf_file, "Section 2.1 Hardware Clock Frequency.")

        manifest = IngestionManifest(manifest_path=manifest_file)
        manifest.record_indexed(pdf_file, doc_id="specs/temp_spec", chunk_count=2)
        assert "specs/temp_spec" in manifest.get_indexed_doc_ids()

        # Delete file on disk
        pdf_file.unlink()
        pruned = manifest.prune_missing_files()
        assert pruned == ["specs/temp_spec"]
        assert "specs/temp_spec" not in manifest.get_indexed_doc_ids()


def test_vector_store_delete_document():
    """Verify that QdrantVectorStore.delete_document deletes points matching doc_id."""
    store = QdrantVectorStore(location=":memory:")
    from src.common.schemas import ChildChunk, ParentChunk

    parent = ParentChunk(
        parent_id="p_1",
        doc_id="doc_a",
        text="Parent text for doc_a",
        page_number=1,
        child_ids=["c_1"],
        chunk_index=0,
    )
    child = ChildChunk(
        chunk_id="c_1",
        parent_id="p_1",
        doc_id="doc_a",
        text="Child text for doc_a",
        vector=[0.1] * 384,
        page_number=1,
        chunk_index=0,
    )

    store.upsert(chunks=[child], parents=[parent])

    # Search should find doc_a
    results = store.search(query_vector=[0.1] * 384, limit=5)
    assert len(results) == 1
    assert results[0]["doc_id"] == "doc_a"

    # Delete doc_a
    store.delete_document("doc_a")

    # Search should now return empty
    results_after = store.search(query_vector=[0.1] * 384, limit=5)
    assert len(results_after) == 0
    assert store.get_parent("p_1") is None


def test_pipeline_incremental_ingestion_skip_and_reindex():
    """Verify that TrustRAGPipeline skips unchanged files and re-indexes modified files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        manifest_file = tmp_path / "manifest.json"

        pdf_1 = raw_dir / "doc1.pdf"
        _create_test_pdf(pdf_1, "Section 1: Initial specifications.")

        manifest = IngestionManifest(manifest_path=manifest_file)
        pipeline = TrustRAGPipeline(manifest=manifest, qdrant_location=":memory:")

        # 1st Run: Should index doc1
        indexed_1 = pipeline.ingest_directory(raw_dir=str(raw_dir))
        assert len(indexed_1) > 0
        assert "doc1" in pipeline.known_doc_ids

        # 2nd Run: Should skip doc1 (unchanged)
        indexed_2 = pipeline.ingest_directory(raw_dir=str(raw_dir))
        assert len(indexed_2) == 0

        # Modify doc1
        _create_test_pdf(pdf_1, "Section 1: Completely updated specifications with new metrics.")

        # 3rd Run: Should detect modification and re-index doc1
        indexed_3 = pipeline.ingest_directory(raw_dir=str(raw_dir))
        assert len(indexed_3) > 0


def test_vector_store_load_all_child_chunks():
    """Verify that QdrantVectorStore.load_all_child_chunks scrolls and reconstructs ChildChunk models."""
    store = QdrantVectorStore(location=":memory:")
    from src.common.schemas import ChildChunk, ParentChunk

    parent = ParentChunk(
        parent_id="parent_1",
        doc_id="spec_doc",
        text="Full parent passage about architecture.",
        page_number=2,
        child_ids=["child_1"],
        chunk_index=0,
        section_name="Architecture",
    )
    child = ChildChunk(
        chunk_id="child_1",
        parent_id="parent_1",
        doc_id="spec_doc",
        text="[Section: Architecture] Child chunk text.",
        vector=[0.05] * 384,
        page_number=2,
        chunk_index=0,
        section_name="Architecture",
    )

    store.upsert(chunks=[child], parents=[parent])

    # Load all chunks
    loaded = store.load_all_child_chunks()
    assert len(loaded) == 1
    assert loaded[0].chunk_id == "child_1"
    assert loaded[0].parent_id == "parent_1"
    assert loaded[0].doc_id == "spec_doc"
    assert loaded[0].section_name == "Architecture"
    assert loaded[0].page_number == 2

    # Verify parent store was hydrated
    hydrated_parent = store.get_parent("parent_1")
    assert hydrated_parent is not None
    assert hydrated_parent.text == "Full parent passage about architecture."


def test_pipeline_state_hydration_from_qdrant_scroll():
    """Verify that a brand new pipeline instance on startup hydrates in-memory chunks and BM25 index from persistent vector store."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir()
        qdrant_db_path = tmp_path / "qdrant_db"
        manifest_file = tmp_path / "manifest.json"

        pdf_1 = raw_dir / "hardware_specs.pdf"
        _create_test_pdf(pdf_1, "Section 1: The Model-X processor operates at 125W TDP with 16 cores.")

        # Run 1: Initial ingestion with persistent Qdrant path
        manifest_1 = IngestionManifest(manifest_path=manifest_file)
        pipeline_1 = TrustRAGPipeline(
            manifest=manifest_1,
            qdrant_location=None,
            qdrant_path=str(qdrant_db_path),
        )
        indexed_1 = pipeline_1.ingest_directory(raw_dir=str(raw_dir))
        assert len(indexed_1) > 0
        assert len(pipeline_1.all_child_chunks) > 0
        initial_chunk_count = len(pipeline_1.all_child_chunks)
        pipeline_1.close()

        # Run 2: Brand new pipeline instance (simulating application restart)
        manifest_2 = IngestionManifest(manifest_path=manifest_file)
        pipeline_2 = TrustRAGPipeline(
            manifest=manifest_2,
            qdrant_location=None,
            qdrant_path=str(qdrant_db_path),
        )

        try:
            # Before ingestion, in-memory list is empty
            assert len(pipeline_2.all_child_chunks) == 0

            # Ingest directory -> file is skipped by manifest, but in-memory state is hydrated from Qdrant scroll!
            pipeline_2.ingest_directory(raw_dir=str(raw_dir))

            assert len(pipeline_2.all_child_chunks) == initial_chunk_count
            assert len(pipeline_2.child_chunk_map) == initial_chunk_count
            assert pipeline_2.bm25_searcher is not None
            assert "hardware_specs" in pipeline_2.known_doc_ids

            # Verify BM25 search works immediately on hydrated state
            bm25_results = pipeline_2.bm25_searcher.search(["processor", "125w"], top_k=3)
            assert len(bm25_results) > 0
        finally:
            pipeline_2.close()


