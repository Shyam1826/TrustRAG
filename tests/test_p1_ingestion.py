r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_p1_ingestion.py
   - Role: Integration and unit test suite for Pipeline 1 (Ingestion & Storage).
   - Purpose: Validates PDF layout extraction, text cleaning, hierarchical chunking,
     dense/sparse embedding generation, persistent QdrantVectorStore operations,
     native payload filtering, and parent-child metadata hydration.

2. INPUT (IP):
   - Temporary multi-page PDF documents and synthetic layout blocks.

3. PROCESS UNDER THE HOOD:
   - Tests `clean_text` hyphenation deconstruction and whitespace normalization.
   - Tests end-to-end ingestion pipeline with hierarchical chunking and embedding.
   - Tests `QdrantVectorStore` disk persistence, upsert with complete payloads,
     and native filtering with `MatchValue` and `MatchAny`.
   - Tests multi-column layout extraction block ordering in PyMuPDF parser.

4. OUTPUT (OP):
   - Pytest assertions and validation results.

5. LIBRARIES & DEPENDENCIES:
   - fitz (PyMuPDF): PDF generation and parsing.
   - pytest, tempfile, os: Test harness utilities.
   - src.pipeline_1_ingestion.*: Ingestion modules.
   - src.common.schemas: ChildChunk, ParentChunk models.
================================================================================
"""

import os
import shutil
import tempfile
from typing import Generator
import fitz  # PyMuPDF
import pytest

from src.common.schemas import ChildChunk, ParentChunk
from src.pipeline_1_ingestion.parser import extract_pdf_pages
from src.pipeline_1_ingestion.cleaner import clean_text
from src.pipeline_1_ingestion.chunker import create_hierarchical_chunks
from src.pipeline_1_ingestion.embedder import DualEmbedder
from src.pipeline_1_ingestion.indexer import LocalStore
from src.pipeline_1_ingestion.vector_store import QdrantVectorStore


@pytest.fixture
def sample_pdf_path() -> Generator[str, None, None]:
    """Create a temporary 2-page PDF with known sample content."""
    doc = fitz.open()

    # Page 1: Contains multi-paragraph text with hyphenation across newlines
    page1 = doc.new_page()
    page1_text = (
        "TrustRAG provides high-assurance retrieval augmented generation with claim-level "
        "verification.\n\n"
        "It supports inter-\nnational standards for enterprise AI safety, robustly "
        "evaluating groundedness across diverse document corpora."
    )
    page1.insert_text((50, 72), page1_text, fontsize=11)

    # Page 2: Contains second page content
    page2 = doc.new_page()
    page2_text = (
        "Hierarchical indexing enables precise vector retrieval on child chunks while "
        "preserving the complete parent context for downstream generation and reasoning."
    )
    page2.insert_text((50, 72), page2_text, fontsize=11)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    doc.save(tmp_path)
    doc.close()

    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_cleaner_hyphenation_and_normalization():
    raw_text = "inter-\nnational  cooperation\n\n\n\nand   security\r\nstandards."
    cleaned = clean_text(raw_text)
    assert "international cooperation" in cleaned
    assert "and security standards." in cleaned
    assert "\n\n" in cleaned
    assert "\n\n\n" not in cleaned


def test_p1_end_to_end_pipeline(sample_pdf_path: str):
    doc_id = "doc_test_42"

    # Step 1: Parse PDF
    pages = extract_pdf_pages(sample_pdf_path)
    assert len(pages) == 2
    assert pages[0]["page_number"] == 1
    assert pages[1]["page_number"] == 2
    assert "TrustRAG" in pages[0]["raw_text"]

    # Step 2: Create Hierarchical Chunks
    parents, children = create_hierarchical_chunks(
        pages,
        doc_id=doc_id,
        parent_size=800,
        child_size=150,
        overlap=30,
    )

    assert len(parents) >= 2
    assert len(children) >= len(parents)

    # Verify parent-child linkage contracts
    parent_id_map = {p.parent_id: p for p in parents}
    for child in children:
        assert child.doc_id == doc_id
        assert child.parent_id in parent_id_map
        assert child.chunk_id in parent_id_map[child.parent_id].child_ids
        assert child.page_number in (1, 2)
        assert len(child.text) > 0

    # Step 3: Embed Dense and Sparse
    embedder = DualEmbedder()
    child_texts = [c.text for c in children]
    dense_vectors = embedder.embed_dense(child_texts)
    assert len(dense_vectors) == len(children)
    assert len(dense_vectors[0]) == 384  # BGE-small embedding dimension

    for child, vec in zip(children, dense_vectors):
        child.vector = vec
        child.sparse_tokens = embedder.tokenize_sparse(child.text)
        assert isinstance(child.sparse_tokens, dict)
        assert len(child.sparse_tokens) > 0

    # Step 4: Index into LocalStore (Qdrant + Parent Store)
    store = LocalStore(location=":memory:", vector_size=384)
    store.upsert_child_chunks(children)
    store.store_parents(parents)

    # Assert 1: Qdrant collection contains exact number of points
    point_count = store.client.count(collection_name=store.collection_name).count
    assert point_count == len(children)

    # Assert 2: Querying get_parent returns exact parent passage
    for parent in parents:
        retrieved_parent = store.get_parent(parent.parent_id)
        assert retrieved_parent is not None
        assert retrieved_parent.parent_id == parent.parent_id
        assert retrieved_parent.text == parent.text
        assert retrieved_parent.child_ids == parent.child_ids


def test_qdrant_vector_store_persistence_and_filtering():
    """Verify persistent Qdrant on disk, payload hydration, and native filtering."""
    temp_dir = tempfile.mkdtemp(prefix="qdrant_test_")
    try:
        # 1. Initialize persistent store on disk
        store = QdrantVectorStore(path=temp_dir, vector_size=384)

        # 2. Prepare sample chunks across two documents
        parent1 = ParentChunk(
            parent_id="p_doc1_0",
            doc_id="doc_alpha",
            text="[Document: doc_alpha | Section: Architecture]\nBackend microservices use MongoDB and Python.",
            page_number=1,
            section_name="Architecture",
        )
        parent2 = ParentChunk(
            parent_id="p_doc2_0",
            doc_id="doc_beta",
            text="[Document: doc_beta | Section: Infrastructure]\nContainer orchestration runs on Docker and Kubernetes.",
            page_number=1,
            section_name="Infrastructure",
        )
        store.store_parents([parent1, parent2])

        dummy_vector1 = [0.1] * 384
        dummy_vector2 = [0.2] * 384

        child1 = ChildChunk(
            chunk_id="c_doc1_0",
            parent_id="p_doc1_0",
            doc_id="doc_alpha",
            text="Backend microservices use MongoDB.",
            page_number=1,
            vector=dummy_vector1,
            section_name="Architecture",
        )
        child2 = ChildChunk(
            chunk_id="c_doc2_0",
            parent_id="p_doc2_0",
            doc_id="doc_beta",
            text="Container orchestration runs on Docker.",
            page_number=1,
            vector=dummy_vector2,
            section_name="Infrastructure",
        )
        store.upsert_child_chunks([child1, child2])

        # 3. Test global search with parent_text hydration
        results_all = store.search(query_vector=dummy_vector1, limit=5)
        assert len(results_all) == 2
        for res in results_all:
            assert "child_id" in res
            assert "parent_id" in res
            assert "doc_id" in res
            assert "parent_text" in res
            assert len(res["parent_text"]) > 0

        # 4. Test single document filter (MatchValue)
        results_alpha = store.search(query_vector=dummy_vector1, limit=5, doc_filter="doc_alpha")
        assert len(results_alpha) == 1
        assert results_alpha[0]["doc_id"] == "doc_alpha"
        assert results_alpha[0]["child_id"] == "c_doc1_0"
        assert "MongoDB" in results_alpha[0]["parent_text"]

        # 5. Test multi-document filter (MatchAny)
        results_multi = store.search(query_vector=dummy_vector1, limit=5, doc_filter=["doc_alpha", "doc_beta"])
        assert len(results_multi) == 2
        doc_ids = {r["doc_id"] for r in results_multi}
        assert doc_ids == {"doc_alpha", "doc_beta"}

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_parser_multi_column_block_ordering():
    """Verify that multi-column PDFs are sorted left-to-right to prevent text interleaving."""
    doc = fitz.open()
    page = doc.new_page(width=600, height=800)

    # Insert left column block (e.g., Contact Info & Skills)
    page.insert_textbox(
        fitz.Rect(50, 50, 200, 400),
        "CONTACT INFO\nEmail: candidate@domain.com\n\nSKILLS\n- Python\n- AutoCad",
        fontsize=10,
    )

    # Insert right column block (e.g., Experience & Projects)
    page.insert_textbox(
        fitz.Rect(250, 50, 550, 400),
        "EXPERIENCE\nSenior Developer at TechCorp\n- Built distributed streaming pipelines",
        fontsize=10,
    )

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    doc.save(tmp_path)
    doc.close()

    try:
        pages = extract_pdf_pages(tmp_path)
        assert len(pages) == 1
        text = pages[0]["raw_text"]

        # Left column items should appear before right column items
        contact_pos = text.find("CONTACT INFO")
        skills_pos = text.find("SKILLS")
        exp_pos = text.find("EXPERIENCE")

        assert contact_pos != -1
        assert skills_pos != -1
        assert exp_pos != -1
        assert contact_pos < exp_pos
        assert skills_pos < exp_pos
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

