r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_p4_verification.py
   - Role: Test suite for Pipeline 4 (Claim Extraction & NLI Verification).
   - Purpose: Validates sentence-level proposition decomposition, Unicode bracket
     normalization (【Doc-X】 -> [Doc-X]), DeBERTa-v3 sequence classification
     probabilities (entailment, contradiction, neutral), faithfulness scoring, and
     adjudicator safety gating actions (PASS, TRIGGER_REWRITE, WARN).

2. INPUT (IP):
   - Synthetic Drafts, Premises, and Hypothesis claim pairs.

3. PROCESS UNDER THE HOOD:
   - Tests `AtomicClaimExtractor` decomposition on complete sentences, markdown lists, and Unicode brackets.
   - Tests `DebertaNLIVerifier` on known entailment, contradiction, and neutral propositions.
   - Tests `AuditAdjudicator` threshold gating and action determination.

4. OUTPUT (OP):
   - Pytest assertions and validation reports.

5. LIBRARIES & DEPENDENCIES:
   - pytest: Test execution framework.
   - src.common.schemas: Strict Pydantic schemas.
   - src.pipeline_4_verification.*: Verification pipeline modules.
================================================================================
"""

import pytest

from src.common.schemas import AtomicClaim, GeneratedDraft
from src.pipeline_4_verification.claim_extractor import AtomicClaimExtractor
from src.pipeline_4_verification.nli_model import DebertaNLIVerifier
from src.pipeline_4_verification.adjudicator import AuditAdjudicator


def test_atomic_claim_extractor():
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "**Hardware Specs**\n"
            "- The Model-X processor operates at 125W TDP and features 16 physical cores【Doc-1】.\n"
            "- Corporate policy grants 20 days of annual paid time off [Doc-2]."
        ),
        cited_doc_ids=["Doc-1", "Doc-2"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    assert len(claims) == 2
    assert claims[0].claim_id == "claim_0"
    assert "The Model-X processor operates at 125W TDP and features 16 physical cores" in claims[0].claim_text
    assert claims[0].cited_doc_id == "Doc-1"

    assert claims[1].claim_id == "claim_1"
    assert "Corporate policy grants 20 days of annual paid time off" in claims[1].claim_text
    assert claims[1].cited_doc_id == "Doc-2"


def test_deberta_nli_entailment_and_contradiction():
    verifier = DebertaNLIVerifier()

    premise = "The Model-X processor operates at 125W TDP and has 16 execution cores."

    claims = [
        "Model-X operates at 125W TDP.",       # Entailment
        "Model-X operates at 65W TDP.",        # Contradiction
        "Model-X was designed in Sweden.",     # Neutral
    ]
    premises = [premise, premise, premise]

    results = verifier.predict_batch(claims=claims, premises=premises)

    assert len(results) == 3

    # Claim 1: Must verify as ENTAILED (prob >= 0.80)
    p_entail = results[0]["probabilities"]["entailment"]
    assert p_entail >= 0.80, f"Expected entailment >= 0.80, got {p_entail}"

    # Claim 2: Must verify as CONTRADICTION (prob >= 0.60)
    p_contra = results[1]["probabilities"]["contradiction"]
    assert p_contra >= 0.60, f"Expected contradiction >= 0.60, got {p_contra}"

    # Claim 3: Must verify as NEUTRAL (neutral probability is dominant)
    p_neutral = results[2]["probabilities"]["neutral"]
    assert p_neutral > results[2]["probabilities"]["entailment"]
    assert p_neutral > results[2]["probabilities"]["contradiction"]


def test_adjudicator_gate_actions():
    adjudicator = AuditAdjudicator(tau_entailment=0.80, tau_contradiction=0.60)

    class MockNLIVerifier:
        def __init__(self, prob_list):
            self.prob_list = prob_list

        def predict_batch(self, claims, premises):
            return [{"probabilities": p} for p in self.prob_list]

    context_map = {"Doc-1": "The Model-X processor operates at 125W TDP."}

    # Case 1: 100% Entailment -> PASS
    mock_pass = MockNLIVerifier([
        {"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04},
        {"entailment": 0.88, "contradiction": 0.02, "neutral": 0.10},
    ])
    claims_pass = [
        AtomicClaim(claim_id="c0", claim_text="operates at 125W", cited_doc_id="Doc-1"),
        AtomicClaim(claim_id="c1", claim_text="Model-X power", cited_doc_id="Doc-1"),
    ]
    report_pass = adjudicator.adjudicate(claims_pass, context_map, mock_pass, "Draft 1")
    assert report_pass.faithfulness_score == 1.0
    assert report_pass.has_contradiction is False
    assert report_pass.action == "PASS"

    # Case 2: Contradiction present -> TRIGGER_REWRITE
    mock_rewrite = MockNLIVerifier([
        {"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04},
        {"entailment": 0.05, "contradiction": 0.85, "neutral": 0.10},
    ])
    claims_rewrite = [
        AtomicClaim(claim_id="c0", claim_text="operates at 125W", cited_doc_id="Doc-1"),
        AtomicClaim(claim_id="c1", claim_text="operates at 65W", cited_doc_id="Doc-1"),
    ]
    report_rewrite = adjudicator.adjudicate(claims_rewrite, context_map, mock_rewrite, "Draft 2")
    assert report_rewrite.faithfulness_score == 0.5
    assert report_rewrite.has_contradiction is True
    assert report_rewrite.action == "TRIGGER_REWRITE"

    # Case 4: Zero claims with legitimate fallback text -> PASS (score 1.0)
    report_fallback = adjudicator.adjudicate(
        claims=[],
        context_map=context_map,
        nli_verifier=mock_pass,
        draft_text="The provided documentation does not contain sufficient information to answer.",
    )
    assert report_fallback.faithfulness_score == 1.0
    assert report_fallback.action == "PASS"

    # Case 5: Zero claims with substantive unverified draft text -> WARN (score 0.0)
    report_bypass = adjudicator.adjudicate(
        claims=[],
        context_map=context_map,
        nli_verifier=mock_pass,
        draft_text="The candidate is an expert in AutoCad and SolidWorks engineering design.",
    )
    assert report_bypass.faithfulness_score == 0.0
    assert report_bypass.action == "WARN"


def test_atomic_claim_extractor_short_items_and_universal_framing():
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### Technical Skills\n"
            "- AutoCad [Doc-1]\n"
            "- Python [Doc-1]\n"
            "### Work Experience\n"
            "- Developed automated data analysis pipelines [Doc-2]"
        ),
        cited_doc_ids=["Doc-1", "Doc-2"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    assert len(claims) == 3
    # 1-word item anchored with section
    assert "Under Technical Skills, the documented specification or item is: AutoCad." in claims[0].claim_text
    assert claims[0].cited_doc_id == "Doc-1"

    assert "Under Technical Skills, the documented specification or item is: Python." in claims[1].claim_text
    assert claims[1].cited_doc_id == "Doc-1"

    # Action verb proposition anchored with domain-neutral framing
    assert "Under Work Experience, the document specifies: Developed automated data analysis pipelines." in claims[2].claim_text
    assert claims[2].cited_doc_id == "Doc-2"


def test_multi_citation_atomic_claim_splitting():
    """Verify that multi-citation lines (e.g. - MongoDB [Doc-1][Doc-2]) split into independent atomic claims."""
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### Database Technologies\n"
            "- MongoDB [Doc-1][Doc-2]\n"
            "- Microservices architecture [Doc-1] [Doc-3]"
        ),
        cited_doc_ids=["Doc-1", "Doc-2", "Doc-3"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    # 2 citations on line 1 + 2 citations on line 2 = 4 atomic claims
    assert len(claims) == 4

    # Claims for MongoDB
    assert "MongoDB" in claims[0].claim_text
    assert claims[0].cited_doc_id == "Doc-1"
    assert claims[0].claim_id == "claim_0"

    assert "MongoDB" in claims[1].claim_text
    assert claims[1].cited_doc_id == "Doc-2"
    assert claims[1].claim_id == "claim_1"

    # Claims for Microservices architecture
    assert "Microservices architecture" in claims[2].claim_text
    assert claims[2].cited_doc_id == "Doc-1"
    assert claims[2].claim_id == "claim_2"

    assert "Microservices architecture" in claims[3].claim_text
    assert claims[3].cited_doc_id == "Doc-3"
    assert claims[3].claim_id == "claim_3"


def test_adjudicator_short_token_premise_window():
    """Verify that premise window extraction searches across complete parent passage for short technical tokens."""
    adjudicator = AuditAdjudicator()

    context = (
        "[Document: Doc-1 | Section: Technical Architecture]\n"
        "Microservices Cluster\n"
        "• Services communicate via REST APIs and gRPC.\n"
        "• Data persistence layer is backed by MongoDB and Redis.\n"
        "• Deployment uses Docker containers orchestrated on AWS ECS."
    )

    # Short technical token claim
    claim = "The document specifies: MongoDB."
    premise_window = adjudicator._extract_premise_window(claim, context)

    assert "MongoDB" in premise_window
    assert "[Document: Doc-1 | Section: Technical Architecture]" in premise_window


