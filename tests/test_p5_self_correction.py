r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_p5_self_correction.py
   - Role: Unit & integration test suite for Pipeline 5 (Query Decomposition & Self-Correction).
   - Purpose: Validates multi-faceted query decomposition, selective claim pruning,
     specific directive scope resolution, and recursive compound carrier stripping.

2. INPUT (IP):
   - Synthetic compound and multi-entity queries.
   - Mocked draft responses and atomic claims.

3. PROCESS UNDER THE HOOD:
   - Tests QueryDecomposer across single-target multi-aspect and multi-target queries.
   - Tests SelfCorrector selective pruning behavior on partial failures.
   - Tests ScopeRouter specific directive precedence over shared generic tokens.
   - Tests AuditAdjudicator recursive compound carrier stripper on stacked nested framing.

4. OUTPUT (OP):
   - Pytest execution assertions and validation reports.

5. LIBRARIES & DEPENDENCIES:
   - pytest: Test execution framework.
   - src.pipeline_5_self_correction.*: Decomposition and self-correction modules.
   - src.pipeline_2_retrieval.router: ScopeRouter.
   - src.pipeline_4_verification.adjudicator: AuditAdjudicator.
================================================================================
"""

import pytest
from unittest.mock import MagicMock

from src.common.schemas import AtomicClaim, ClaimAudit, RetrievalCandidate, TrustAuditReport
from src.pipeline_2_retrieval.router import ScopeRouter
from src.pipeline_4_verification.adjudicator import AuditAdjudicator
from src.pipeline_5_self_correction.decomposer import QueryDecomposer, decompose_query
from src.pipeline_5_self_correction.corrector import SelfCorrector


def test_query_decomposer_multi_facet_and_multi_target():
    decomposer = QueryDecomposer()

    # 1. Single-target multi-aspect query
    q1 = "What are the termination terms and governing law in the LinkPlus Corp affiliate agreement?"
    subs1 = decomposer.decompose(q1)
    assert len(subs1) == 2
    assert "termination terms in the LinkPlus Corp affiliate agreement" in subs1
    assert "governing law in the LinkPlus Corp affiliate agreement" in subs1

    # 2. Multi-target multi-aspect comparative query
    q2 = "Compare the governing law and termination terms between the Chase affiliate agreement and the LinkPlus Corp affiliate agreement."
    subs2 = decomposer.decompose(q2)
    assert len(subs2) == 4
    assert any("governing law in the Chase affiliate agreement" in s for s in subs2)
    assert any("termination terms in the Chase affiliate agreement" in s for s in subs2)
    assert any("governing law in the LinkPlus Corp affiliate agreement" in s for s in subs2)
    assert any("termination terms in the LinkPlus Corp affiliate agreement" in s for s in subs2)

    # 3. Single-intent focused query
    q3 = "What is the TDP wattage of Model-X processor?"
    subs3 = decomposer.decompose(q3)
    assert len(subs3) == 1
    assert subs3[0] == q3


def test_scope_router_specific_directive_precedence():
    router = ScopeRouter()
    known_docs = [
        "Chase Affiliate Agreement",
        "LinkPlusCorp_Affiliate_Agreement",
        "Minimalist CV Resume",
        "master_clauses",
    ]

    # "LinkPlus Corp" is specific to LinkPlusCorp_Affiliate_Agreement; "affiliate" is shared with Chase
    # Scope router must prune Chase and return solitary LinkPlusCorp_Affiliate_Agreement
    doc_filter, tokens = router.extract_doc_filter(
        "What are the termination terms and governing law in the LinkPlus Corp affiliate agreement?",
        known_docs,
    )
    assert doc_filter == "LinkPlusCorp_Affiliate_Agreement"

    # Both Chase and LinkPlus Corp are specifically mentioned -> both must be retained
    doc_filter_comp, tokens_comp = router.extract_doc_filter(
        "Compare the governing law and termination terms between the Chase affiliate agreement and the LinkPlus Corp affiliate agreement.",
        known_docs,
    )
    assert isinstance(doc_filter_comp, list)
    assert set(doc_filter_comp) == {"Chase Affiliate Agreement", "LinkPlusCorp_Affiliate_Agreement"}


def test_recursive_compound_carrier_stripping():
    adjudicator = AuditAdjudicator()

    # Stacked section framing + entity framing
    claim_text = (
        "Under Termination Terms, the document specifies: Under the Chase Affiliate Agreement, "
        "either Affiliate or Chase may terminate the agreement at any time, with or without cause, "
        "by giving the other party written or e-mail notice of termination."
    )
    cleaned = adjudicator._clean_hypothesis_for_nli(claim_text)
    assert not cleaned.startswith("Under Termination Terms")
    assert not cleaned.startswith("Under the Chase Affiliate Agreement")
    assert "Either Affiliate or Chase may terminate the agreement at any time" in cleaned


def test_self_corrector_selective_claim_pruning():
    corrector = SelfCorrector()

    candidate = RetrievalCandidate(
        parent_id="p1",
        doc_id="doc_1",
        page_number=1,
        text="The server runs on port 8080 and uses PostgreSQL.",
        score=0.9,
        match_type="dense",
    )

    # Initial report with 1 ENTAILED claim and 1 NEUTRAL claim
    initial_audits = [
        ClaimAudit(claim_id="c1", claim_text="The server runs on port 8080.", cited_premise="port 8080", probabilities={"entailment": 0.99}, verdict="ENTAILED", confidence=0.99),
        ClaimAudit(claim_id="c2", claim_text="The server uses Redis cache.", cited_premise="redis cache", probabilities={"neutral": 0.60}, verdict="NEUTRAL", confidence=0.40),
    ]
    initial_report = TrustAuditReport(
        draft_text="- The server runs on port 8080 [Doc-1].\n- The server uses Redis cache [Doc-1].",
        faithfulness_score=0.5,
        has_contradiction=False,
        action="TRIGGER_REWRITE",
        audits=initial_audits,
    )

    # Mock generator returning fallback string on rewrite
    mock_generator = MagicMock()
    mock_generator.generate.return_value = "The provided documentation does not contain sufficient information to answer."

    mock_extractor = MagicMock()
    mock_extractor.extract_claims.return_value = [
        AtomicClaim(claim_id="c1", claim_text="The server runs on port 8080.", cited_doc_ids=["Doc-1"]),
    ]

    mock_adjudicator = MagicMock()
    mock_adjudicator.adjudicate.return_value = TrustAuditReport(
        draft_text="- The server runs on port 8080 [Doc-1].",
        faithfulness_score=1.0,
        has_contradiction=False,
        action="PASS",
        audits=[
            ClaimAudit(claim_id="c1", claim_text="The server runs on port 8080.", cited_premise="port 8080", probabilities={"entailment": 0.99}, verdict="ENTAILED", confidence=0.99),
        ],
    )

    result_report = corrector.correct(
        query="What port and cache does the server use?",
        draft_text=initial_report.draft_text,
        audit_report=initial_report,
        top_contexts=[candidate],
        generator=mock_generator,
        claim_extractor=mock_extractor,
        adjudicator=mock_adjudicator,
        nli_verifier=MagicMock(),
        context_map={"Doc-1": candidate},
    )

    # Must preserve the verified entailed claim instead of collapsing to full fallback
    assert result_report.faithfulness_score == 1.0
    assert result_report.action == "PASS"
    assert "port 8080" in result_report.draft_text
