"""Integration and unit tests for Pipeline 1: Ingestion & Hierarchical Indexing."""

import os
import tempfile
from typing import Generator
import fitz  # PyMuPDF
import pytest

from src.pipeline_1_ingestion.parser import extract_pdf_pages
from src.pipeline_1_ingestion.cleaner import clean_text
from src.pipeline_1_ingestion.chunker import create_hierarchical_chunks
from src.pipeline_1_ingestion.embedder import DualEmbedder
from src.pipeline_1_ingestion.indexer import LocalStore


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

    # Step 4: Index into LocalStore (in-memory Qdrant + Parent Store)
    store = LocalStore(location=":memory:", vector_size=384)
    store.upsert_child_chunks(children)
    store.store_parents(parents)

    # Assert 1: Qdrant collection contains exact number of points
    point_count = store.client.count(collection_name="child_chunks").count
    assert point_count == len(children)

    # Assert 2: Querying get_parent returns exact parent passage
    for parent in parents:
        retrieved_parent = store.get_parent(parent.parent_id)
        assert retrieved_parent is not None
        assert retrieved_parent.parent_id == parent.parent_id
        assert retrieved_parent.text == parent.text
        assert retrieved_parent.child_ids == parent.child_ids


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

