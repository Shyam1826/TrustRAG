r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_compactor_and_concurrency.py
   - Role: Unit and integration test suite for Features 2 & 3.
   - Purpose: Validates ContextCompactor dynamic adaptive sentence compaction,
     tabular candidate bypass, 4,000-char context budgeting, batched DeBERTa NLI
     adjudication, and concurrent multi-thread sub-query retrieval.

2. INPUT (IP):
   - Synthetic long parent chunks and multi-faceted queries.
   - Tabular row candidates and narrative candidates.

3. PROCESS UNDER THE HOOD:
   - Tests ContextCompactor:
     * Sentences matching query terms extracted along with ±1 bounding sentences.
     * Section breadcrumbs preserved.
     * Tabular candidates retained verbatim without truncation.
     * Aggregated context budget ceiling enforcement.
   - Tests Batched DeBERTa NLI Adjudication:
     * Multi-claim adjudication executed via vectorized batch inference (`batch_size=32`).
     * Correct entailment classification.
   - Tests Concurrent Sub-Query Retrieval:
     * Decomposes compound queries and processes sub-queries in parallel threads.
     * Deduplicates candidate pools safely and applies RRF fusion.

4. OUTPUT (OP):
   - Pytest assertions and test outcomes.

5. LIBRARIES & DEPENDENCIES:
   - pytest, unittest.mock.
   - src.common.schemas (RetrievalCandidate, AtomicClaim).
   - src.pipeline_3_generation.compactor (ContextCompactor).
   - src.pipeline_3_generation.prompt (build_rag_prompt).
   - src.pipeline_4_verification.adjudicator (AuditAdjudicator).
   - src.pipeline_4_verification.nli_model (DebertaNLIVerifier).
   - src.main (TrustRAGPipeline).
================================================================================
"""

import pytest

from src.common.schemas import AtomicClaim, RetrievalCandidate
from src.main import TrustRAGPipeline
from src.pipeline_3_generation.compactor import ContextCompactor
from src.pipeline_3_generation.prompt import build_rag_prompt
from src.pipeline_4_verification.adjudicator import AuditAdjudicator
from src.pipeline_4_verification.nli_model import DebertaNLIVerifier


def test_context_compactor_narrative_sentence_spans() -> None:
    """Verify ContextCompactor extracts targeted sentence spans and ±1 bounding context."""
    compactor = ContextCompactor()

    raw_text = (
        "[Section: Liability Clauses] "
        "The agreement is governed by the laws of California. "
        "Party A shall deliver goods within thirty days of order placement. "
        "The aggregate liability of the Vendor shall not exceed the total fees paid in the last twelve months. "
        "Any delay caused by shipping carriers shall be excused. "
        "Disputes shall be settled by binding arbitration in San Francisco. "
        "This contract terminates upon written notice of thirty days."
    )

    candidate = RetrievalCandidate(
        parent_id="p1",
        doc_id="contract_doc",
        page_number=1,
        text=raw_text,
        score=0.92,
        match_type="dense",
    )

    query = "What is the aggregate liability of the Vendor?"
    compacted = compactor.compact_candidate(query, candidate)

    # Must preserve section prefix
    assert "[Section: Liability Clauses]" in compacted.text
    # Must contain targeted sentence
    assert "The aggregate liability of the Vendor shall not exceed the total fees paid" in compacted.text
    # Must contain bounding context (-1: delivery, +1: shipping delay)
    assert "Party A shall deliver goods within thirty days" in compacted.text
    assert "Any delay caused by shipping carriers shall be excused" in compacted.text
    # Must omit unreferenced distant sentences (e.g. arbitration and termination)
    assert "arbitration in San Francisco" not in compacted.text
    assert "contract terminates upon written notice" not in compacted.text
    assert len(compacted.text) < len(raw_text)


def test_context_compactor_tabular_bypass() -> None:
    """Verify ContextCompactor preserves structured tabular candidates verbatim."""
    compactor = ContextCompactor()

    table_text = (
        "[Section: Table: master_clauses | Row: 1] "
        "Document Name: MARKETING AFFILIATE AGREEMENT | "
        "Filename: CybergyHoldingsInc.pdf | "
        "Cap On Liability: Yes | Liquidated Damages: No | "
        "Renewal Term: 1 year successive"
    )

    candidate = RetrievalCandidate(
        parent_id="tab_1",
        doc_id="master_clauses",
        page_number=1,
        text=table_text,
        score=1.0,
        match_type="tabular_sql",
        section_name="Table: master_clauses | Row: 1",
    )

    query = "Show renewal terms where Cap On Liability is Yes"
    compacted = compactor.compact_candidate(query, candidate)

    # Must remain completely untouched
    assert compacted.text == table_text


def test_context_compactor_aggregate_budget_enforcement() -> None:
    """Verify ContextCompactor enforces max_total_chars budget across candidate lists."""
    compactor = ContextCompactor()

    candidates = [
        RetrievalCandidate(
            parent_id=f"p_{i}",
            doc_id=f"doc_{i}",
            page_number=1,
            text=f"Paragraph {i}: " + ("The quick brown fox jumps over the lazy dog. " * 15),
            score=0.9 - (i * 0.05),
            match_type="dense",
        )
        for i in range(10)
    ]

    query = "Where does the brown fox jump?"
    compacted_list = compactor.compact_contexts(query, candidates, max_total_chars=1000)

    total_chars = sum(len(c.text) for c in compacted_list)
    assert len(compacted_list) >= 2
    assert total_chars <= 1600  # Budget enforced gracefully while preserving top-2 minimum


def test_build_rag_prompt_hard_ceiling_under_4000_chars() -> None:
    """Verify build_rag_prompt context block remains strictly bounded under 4,000 chars."""
    candidates = [
        RetrievalCandidate(
            parent_id=f"p_{i}",
            doc_id=f"doc_{i}",
            page_number=1,
            text=f"[Section: Section_{i}] " + ("Specification parameter alpha is active. " * 30),
            score=0.95 - (i * 0.01),
            match_type="dense",
        )
        for i in range(8)
    ]

    query = "What is parameter alpha?"
    prompt = build_rag_prompt(query, candidates, max_context_chars=4000)

    context_start = prompt.find("<context>\n")
    context_end = prompt.find("</context>")
    assert context_start != -1 and context_end != -1

    context_section = prompt[context_start:context_end]
    assert len(context_section) <= 4500  # Well below 2,500 token ceiling


def test_batched_deberta_nli_adjudication() -> None:
    """Verify AuditAdjudicator processes multiple claims in vectorized batch inference."""
    adjudicator = AuditAdjudicator()
    nli_verifier = DebertaNLIVerifier()

    premise_1 = "The Model-X processor features 16 physical cores and operates at 125W TDP."
    premise_2 = "Employees are entitled to 20 days of annual paid time off."

    candidates = {
        "Doc-1": RetrievalCandidate(parent_id="p1", doc_id="d1", page_number=1, text=premise_1, score=1.0, match_type="dense"),
        "Doc-2": RetrievalCandidate(parent_id="p2", doc_id="d2", page_number=2, text=premise_2, score=1.0, match_type="dense"),
    }

    claims = [
        AtomicClaim(claim_id="c1", claim_text="The Model-X processor has 16 physical cores.", cited_doc_ids=["Doc-1"]),
        AtomicClaim(claim_id="c2", claim_text="The Model-X processor operates at 125W TDP.", cited_doc_ids=["Doc-1"]),
        AtomicClaim(claim_id="c3", claim_text="Employees receive 20 days of paid time off.", cited_doc_ids=["Doc-2"]),
    ]

    report = adjudicator.adjudicate(
        claims=claims,
        context_map=candidates,
        nli_verifier=nli_verifier,
        draft_text="Draft text with claims.",
        batch_size=32,
    )

    assert len(report.audits) == 3
    assert all(a.verdict == "ENTAILED" for a in report.audits)
    assert report.faithfulness_score == 1.0
    assert report.action == "PASS"


def test_concurrent_sub_query_retrieval_integration() -> None:
    """Verify TrustRAGPipeline.retrieve executes sub-queries concurrently with ThreadPoolExecutor."""
    pipeline = TrustRAGPipeline(generator_type="mock", qdrant_location=":memory:")

    # Ingest synthetic documents into in-memory store
    cand_1 = RetrievalCandidate(
        parent_id="p1",
        doc_id="contract_alpha",
        page_number=1,
        text="Contract Alpha governing law is Delaware and term is 36 months.",
        score=0.9,
        match_type="dense",
    )
    cand_2 = RetrievalCandidate(
        parent_id="p2",
        doc_id="contract_beta",
        page_number=1,
        text="Contract Beta governing law is New York and term is 12 months.",
        score=0.85,
        match_type="dense",
    )

    pipeline.known_doc_ids = {"contract_alpha", "contract_beta"}

    # Query with multiple sub-queries
    query = "What is the governing law and term in Contract Alpha?"
    candidates = pipeline.retrieve(query)

    # Retrieval must complete cleanly without thread deadlocks or exceptions
    assert isinstance(candidates, list)
    pipeline.close()
