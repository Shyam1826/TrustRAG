r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_main_orchestrator.py
   - Role: Integration test suite for TrustRAGPipeline orchestrator.
   - Purpose: Validates complete end-to-end execution of `TrustRAGPipeline.ingest_pdf`
     and `TrustRAGPipeline.ask` across all 4 pipelines.

2. INPUT (IP):
   - Synthesized PDF document and natural language queries.

3. PROCESS UNDER THE HOOD:
   - Creates a temporary PDF document.
   - Instantiates `TrustRAGPipeline`.
   - Ingests the PDF and verifies chunks and BM25 index.
   - Queries the pipeline with a factual question.
   - Asserts the final `TrustAuditReport` structure, faithfulness score, and safety action.

4. OUTPUT (OP):
   - Pytest assertions and test outcomes.

5. LIBRARIES & DEPENDENCIES:
   - pytest: Test execution framework.
   - fitz (PyMuPDF): Test PDF creation.
   - src.main.TrustRAGPipeline: Full pipeline orchestrator.
================================================================================
"""

import os
import tempfile
from typing import Generator
import fitz
import pytest

from src.main import TrustRAGPipeline


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
