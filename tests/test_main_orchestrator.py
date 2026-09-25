r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_main_orchestrator.py
   - Role: Integration test suite for TrustRAGPipeline orchestrator.
   - Purpose: Validates complete end-to-end execution of `TrustRAGPipeline.ingest_pdf`,
     `TrustRAGPipeline.ask`, recursive folder discovery, and 1-pass automated self-correction rewriting.

2. INPUT (IP):
   - Synthesized PDF document and natural language queries.

3. PROCESS UNDER THE HOOD:
   - Creates a temporary PDF document.
   - Instantiates `TrustRAGPipeline`.
   - Ingests the PDF and verifies chunks and BM25 index.
   - Queries the pipeline with a factual question.
   - Asserts the final `TrustAuditReport` structure, faithfulness score, and safety action.
   - Tests automated 1-pass corrective rewriting on ungrounded initial drafts.

4. OUTPUT (OP):
   - Pytest assertions and test outcomes.

5. LIBRARIES & DEPENDENCIES:
   - pytest: Test execution framework.
   - fitz (PyMuPDF): Test PDF creation.
   - src.main.TrustRAGPipeline: Full pipeline orchestrator.
================================================================================
"""

import os
from pathlib import Path
import tempfile
from typing import Generator
import fitz
import pytest

from src.main import TrustRAGPipeline, discover_raw_documents


@pytest.fixture
def orchestrator_pdf_path() -> Generator[str, None, None]:
    """Create a temporary PDF for orchestrator testing."""
    doc = fitz.open()
    page = doc.new_page()
    page_text = (
        "Hardware Specs: The Model-X processor features 16 physical cores and operates at 125W TDP.\n"
        "It is engineered for enterprise datacenters."
    )
    page.insert_text((50, 72), page_text, fontsize=11)

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp_file:
        tmp_path = tmp_file.name

    doc.save(tmp_path)
    doc.close()

    try:
        yield tmp_path
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def test_trustrag_pipeline_end_to_end(orchestrator_pdf_path: str):
    pipeline = TrustRAGPipeline(generator_type="mock")

    # Ingest PDF
    chunks = pipeline.ingest_pdf(orchestrator_pdf_path, doc_id="hardware_spec_doc")
    assert len(chunks) > 0
    assert pipeline.bm25_searcher is not None

    # Query Pipeline
    query = "Can you tell me what is the TDP wattage of Model-X processor?"
    report = pipeline.ask(query)

    # Assert Report
    assert report.draft_text is not None
    assert "125W TDP" in report.draft_text
    assert "[Doc-1]" in report.draft_text
    assert len(report.audits) > 0
    assert report.has_contradiction is False
    assert report.faithfulness_score >= 0.80
    assert report.action == "PASS"


def test_recursive_subfolder_document_discovery():
    """Verify recursive scanning discovers files across nested subfolders and prevents collision."""
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)

        sub_legal_2026 = root / "legal" / "2026"
        sub_legal_2025 = root / "legal" / "2025"
        sub_eng = root / "engineering"
        sub_hidden = root / ".hidden"

        for p in [sub_legal_2026, sub_legal_2025, sub_eng, sub_hidden]:
            p.mkdir(parents=True, exist_ok=True)

        doc1 = fitz.open()
        p1 = doc1.new_page()
        p1.insert_text((50, 72), "Legal NDA 2026 content.")
        doc1.save(str(sub_legal_2026 / "nda.pdf"))
        doc1.close()

        doc2 = fitz.open()
        p2 = doc2.new_page()
        p2.insert_text((50, 72), "Legal NDA 2025 content.")
        doc2.save(str(sub_legal_2025 / "nda.pdf"))
        doc2.close()

        doc3 = fitz.open()
        p3 = doc3.new_page()
        p3.insert_text((50, 72), "Engineering specifications content.")
        doc3.save(str(sub_eng / "specs.pdf"))
        doc3.close()

        doc4 = fitz.open()
        p4 = doc4.new_page()
        p4.insert_text((50, 72), "Hidden content.")
        doc4.save(str(sub_hidden / "secret.pdf"))
        doc4.close()

        discovered = discover_raw_documents(raw_dir=root)

        assert len(discovered) == 3
        doc_ids = {d[1] for d in discovered}
        assert doc_ids == {"legal/2026/nda", "legal/2025/nda", "engineering/specs"}

        pipeline = TrustRAGPipeline(generator_type="mock")
        chunks = pipeline.ingest_directory(raw_dir=str(root))
        assert len(chunks) >= 3
        assert "legal/2026/nda" in pipeline.known_doc_ids
        assert "legal/2025/nda" in pipeline.known_doc_ids
        assert "engineering/specs" in pipeline.known_doc_ids
        pipeline.close()


def test_self_correction_rewrite_loop(orchestrator_pdf_path: str):
    """Verify that unverified/contradictory drafts trigger 1-pass corrective rewrite to improve faithfulness."""
    class TwoPassCorrectiveGenerator:
        def __init__(self):
            self.call_count = 0

        def generate(self, prompt: str) -> str:
            self.call_count += 1
            if "CLOSED-WORLD REWRITE INSTRUCTIONS" in prompt:
                # Corrected response removing ungrounded assertion
                return "The Model-X processor features 16 physical cores and operates at 125W TDP [Doc-1]."
            # Initial buggy draft with ungrounded assertion
            return (
                "The Model-X processor features 16 physical cores and operates at 125W TDP [Doc-1].\n"
                "The processor also integrates an ungrounded 500W quantum chiller unit [Doc-1]."
            )

    pipeline = TrustRAGPipeline(generator_type="mock")
    pipeline.generator = TwoPassCorrectiveGenerator()

    # Ingest document
    pipeline.ingest_pdf(orchestrator_pdf_path, doc_id="hardware_spec_doc")

    # Ask query
    query = "What is the TDP wattage of Model-X processor?"
    report = pipeline.ask(query)

    # Asserts that generator was called twice (initial + self-correction rewrite)
    assert pipeline.generator.call_count == 2
    assert report.faithfulness_score == 1.0
    assert report.action == "PASS"
    assert "quantum chiller" not in report.draft_text
    assert "125W TDP" in report.draft_text
    pipeline.close()

