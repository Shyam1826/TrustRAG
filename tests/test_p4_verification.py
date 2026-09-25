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


def test_negative_meta_claim_filtering():
    """Verify that negative meta-observations about document absence are filtered out from verifiable claims."""
    extractor = AtomicClaimExtractor()

    # Direct helper validation
    assert extractor._is_negative_meta_claim("No details provided regarding pricing.") is True
    assert extractor._is_negative_meta_claim("There is no mention of database storage.") is True
    assert extractor._is_negative_meta_claim("The documentation does not state the warranty period.") is True
    assert extractor._is_negative_meta_claim("Not specified in documentation.") is True
    assert extractor._is_negative_meta_claim("No further details are available.") is True
    assert extractor._is_negative_meta_claim("The Model-X processor operates at 125W TDP.") is False

    draft = GeneratedDraft(
        raw_text=(
            "### Hardware Specifications\n"
            "- The Model-X processor operates at 125W TDP [Doc-1].\n"
            "- No details provided on external liquid cooling systems.\n"
            "- Warranty terms are not stated in the documentation [Doc-1].\n"
            "- Memory interface is 256-bit DDR5 [Doc-1]."
        ),
        cited_doc_ids=["Doc-1"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    # Only the 2 positive factual assertions should be extracted
    assert len(claims) == 2
    assert "125W TDP" in claims[0].claim_text
    assert "256-bit DDR5" in claims[1].claim_text
    assert not any("cooling" in c.claim_text.lower() for c in claims)
    assert not any("warranty" in c.claim_text.lower() for c in claims)


def test_multi_citation_claim_association():
    """Verify that multi-citation lines (e.g. - MongoDB [Doc-1][Doc-2]) create a unified atomic claim."""
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

    # 1 claim for line 1 + 1 claim for line 2 = 2 atomic claims
    assert len(claims) == 2

    # Claim 0 for MongoDB
    assert "MongoDB" in claims[0].claim_text
    assert claims[0].cited_doc_ids == ["Doc-1", "Doc-2"]
    assert "Doc-1" in claims[0].cited_doc_id and "Doc-2" in claims[0].cited_doc_id
    assert claims[0].claim_id == "claim_0"

    # Claim 1 for Microservices architecture
    assert "Microservices architecture" in claims[1].claim_text
    assert claims[1].cited_doc_ids == ["Doc-1", "Doc-3"]
    assert "Doc-1" in claims[1].cited_doc_id and "Doc-3" in claims[1].cited_doc_id
    assert claims[1].claim_id == "claim_1"


def test_multi_citation_premise_unification():
    """Verify that multi-citation comparative statements evaluate against concatenated premise windows."""
    adjudicator = AuditAdjudicator()
    verifier = DebertaNLIVerifier()

    context_map = {
        "Doc-1": (
            "[Document: Spec-1 | Section: Model-X]\n"
            "Processor Details: The Model-X processor operates at 125W TDP and has 16 execution cores."
        ),
        "Doc-2": (
            "[Document: Spec-2 | Section: Model-Y]\n"
            "Processor Details: The Model-Y processor operates at 65W TDP with low-power optimizations."
        ),
    }

    # Comparative claim citing both Doc-1 and Doc-2
    claim = AtomicClaim(
        claim_id="claim_comp",
        claim_text="The Model-X processor operates at 125W TDP while the Model-Y operates at 65W TDP.",
        cited_doc_id="Doc-1, Doc-2",
        cited_doc_ids=["Doc-1", "Doc-2"],
    )

    report = adjudicator.adjudicate(
        claims=[claim],
        context_map=context_map,
        nli_verifier=verifier,
        draft_text="The Model-X processor operates at 125W TDP while the Model-Y operates at 65W TDP [Doc-1][Doc-2].",
    )

    assert len(report.audits) == 1
    audit = report.audits[0]

    # Verify premise unification included both doc handles
    assert "[Doc-1]" in audit.cited_premise
    assert "[Doc-2]" in audit.cited_premise
    assert "125W TDP" in audit.cited_premise
    assert "65W TDP" in audit.cited_premise

    # Cross-document synthesis statement should evaluate as ENTAILED against unified premise
    assert audit.verdict == "ENTAILED"
    assert report.faithfulness_score == 1.0
    assert report.action == "PASS"



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


def test_atomic_claim_extractor_variable_identifiers_and_math_structures():
    """Verify that variable identifiers (d_model, d_k, d_v) and mathematical equalities (d_k = 64) are preserved."""
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### Attention Mechanism Parameters\n"
            "- d_model = 512 [Doc-1]\n"
            "- d_k = 64 [Doc-1]\n"
            "- d_v = 64 [Doc-1]\n"
            "- batch_size = 32 [Doc-2]"
        ),
        cited_doc_ids=["Doc-1", "Doc-2"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    assert len(claims) == 4
    assert "d_model = 512" in claims[0].claim_text
    assert "d_k = 64" in claims[1].claim_text
    assert "d_v = 64" in claims[2].claim_text
    assert "batch_size = 32" in claims[3].claim_text
    assert "dmodel" not in claims[0].claim_text
    assert "dk" not in claims[1].claim_text


def test_adjudicator_mathematical_formula_premise_window():
    """Verify that premise window accommodates unified mathematical clauses (d_k = d_v = 64)."""
    adjudicator = AuditAdjudicator()

    context = (
        "[Document: Attention Is All You Need | Section: Architecture Details]\n"
        "Encoder Stack: The encoder is composed of a stack of N = 6 identical layers.\n"
        "Multi-Head Attention: In this work we employ h = 8 parallel attention layers. "
        "For each of these we set d_k = d_v = 64 and d_model = 512.\n"
        "Feed-Forward: Each layer contains a fully connected feed-forward network."
    )

    claim_dk = "Under Architecture Details, the document specifies: d_k = 64."
    claim_dv = "Under Architecture Details, the document specifies: d_v = 64."
    claim_dmodel = "Under Architecture Details, the document specifies: d_model = 512."

    window_dk = adjudicator._extract_premise_window(claim_dk, context)
    window_dv = adjudicator._extract_premise_window(claim_dv, context)
    window_dmodel = adjudicator._extract_premise_window(claim_dmodel, context)

    assert "d_k = d_v = 64" in window_dk
    assert "d_k = d_v = 64" in window_dv
    assert "d_model = 512" in window_dmodel


def test_unicode_nfkc_citation_extraction():
    """Verify that Unicode fullwidth brackets, Asian quotation marks, and numeric handles parse to canonical Doc-X."""
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### System Specifications\n"
            "- Operating temperature ranges from -40C to 85C ［Doc-1］\n"
            "- Peak throughput exceeds 10 Gbps 【Doc-2】\n"
            "- Standby power is 50mW ［３］\n"
            "- Galvanic isolation rating is 2.5 kV [4]"
        ),
        cited_doc_ids=["Doc-1", "Doc-2", "Doc-3", "Doc-4"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    assert len(claims) == 4
    # Claim 0: ［Doc-1］ -> Doc-1
    assert "Operating temperature ranges from -40C to 85C" in claims[0].claim_text
    assert claims[0].cited_doc_id == "Doc-1"
    assert claims[0].cited_doc_ids == ["Doc-1"]

    # Claim 1: 【Doc-2】 -> Doc-2
    assert "Peak throughput exceeds 10 Gbps" in claims[1].claim_text
    assert claims[1].cited_doc_id == "Doc-2"
    assert claims[1].cited_doc_ids == ["Doc-2"]

    # Claim 2: ［３］ -> Doc-3
    assert "Standby power is 50mW" in claims[2].claim_text
    assert claims[2].cited_doc_id == "Doc-3"
    assert claims[2].cited_doc_ids == ["Doc-3"]

    # Claim 3: [4] -> Doc-4
    assert "Galvanic isolation rating is 2.5 kV" in claims[3].claim_text
    assert claims[3].cited_doc_id == "Doc-4"
    assert claims[3].cited_doc_ids == ["Doc-4"]


def test_universal_citation_binding_to_context():
    """Verify that extracted Asian and numeric citations bind reliably to context_map in adjudication."""
    adjudicator = AuditAdjudicator()
    verifier = DebertaNLIVerifier()

    context_map = {
        "Doc-1": "[Document: Spec-A | Section: Thermal]\nOperating temperature ranges from -40C to 85C across all operational modes.",
        "Doc-2": "[Document: Spec-B | Section: Power]\nStandby power is 50mW in sleep state.",
    }

    # Draft using fullwidth Asian bracket ［Doc-1］ and fullwidth numeric 【２】
    extractor = AtomicClaimExtractor()
    draft = GeneratedDraft(
        raw_text=(
            "### Operating Profile\n"
            "- Operating temperature ranges from -40C to 85C ［Doc-1］\n"
            "- Standby power is 50mW 【2】"
        ),
        cited_doc_ids=["Doc-1", "Doc-2"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)
    assert len(claims) == 2
    assert claims[0].cited_doc_id == "Doc-1"
    assert claims[1].cited_doc_id == "Doc-2"

    report = adjudicator.adjudicate(
        claims=claims,
        context_map=context_map,
        nli_verifier=verifier,
        draft_text=draft.raw_text,
    )

    assert report.faithfulness_score == 1.0
    assert report.action == "PASS"
    assert len(report.audits) == 2
    assert report.audits[0].verdict == "ENTAILED"
    assert report.audits[1].verdict == "ENTAILED"


def test_parent_context_resolution_and_sentence_boundary_premise():
    """Verify that premise resolution prioritizes parent_text and expands along sentence boundaries."""
    from src.common.schemas import RetrievalCandidate

    adjudicator = AuditAdjudicator(premise_window_size=1400)

    # Retrieval candidate with truncated child text vs full encompassing parent_text
    cand = RetrievalCandidate(
        parent_id="p_doc1_0",
        doc_id="doc1",
        page_number=1,
        text="The Model-X processor operates at 125W TDP.",
        score=0.95,
        match_type="dense",
    )
    # Simulate candidate with parent_text attribute
    object.__setattr__(
        cand,
        "parent_text",
        (
            "[Document: Doc-1 | Section: Architecture]\n"
            "The Model-X processor operates at 125W TDP and features 16 physical execution cores. "
            "It is designed for enterprise server workloads with advanced thermal management."
        ),
    )

    context_map = {"Doc-1": cand}

    claim = AtomicClaim(
        claim_id="c_cores",
        claim_text="The Model-X processor features 16 physical execution cores.",
        cited_doc_id="Doc-1",
    )

    premise = adjudicator._extract_premise_window(claim.claim_text, cand)
    assert "16 physical execution cores" in premise
    assert "[Document: Doc-1 | Section: Architecture]" in premise
    assert not premise.endswith(" ")
    assert premise.endswith(".")


def test_composite_bracket_canonicalization():
    """Verify that composite multi-citation brackets ([Doc-1, Doc-2], [1, 2], [Doc-1; Doc-3]) expand to [Doc-X][Doc-Y]."""
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### Candidate Credentials\n"
            "- Both candidates hold accredited engineering degrees [Doc-1, Doc-2].\n"
            "- Industry certifications in cloud architecture [Doc-1; Doc-3].\n"
            "- Core software engineering foundations [1, 2, 3]"
        ),
        cited_doc_ids=["Doc-1", "Doc-2", "Doc-3"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    assert len(claims) == 3
    assert claims[0].cited_doc_ids == ["Doc-1", "Doc-2"]
    assert claims[1].cited_doc_ids == ["Doc-1", "Doc-3"]
    assert claims[2].cited_doc_ids == ["Doc-1", "Doc-2", "Doc-3"]


def test_compound_comparative_clause_splitting():
    """Verify that comparative sentences with distinct clauses citing separate docs are decomposed into independent claims."""
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### Comparative Experience\n"
            "- While Vaishalee completed B.Tech in CSE [Doc-1], Sanjeev completed B.Tech in IT [Doc-2].\n"
            "- Vaishalee has 5+ years of software engineering experience [Doc-1], whereas Sanjeev has 3 years of NLP research [Doc-2]."
        ),
        cited_doc_ids=["Doc-1", "Doc-2"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    # 2 compound lines with 2 clauses each = 4 atomic claims
    assert len(claims) == 4

    # First sentence clauses
    assert "Vaishalee completed B.Tech in CSE" in claims[0].claim_text
    assert claims[0].cited_doc_ids == ["Doc-1"]

    assert "Sanjeev completed B.Tech in IT" in claims[1].claim_text
    assert claims[1].cited_doc_ids == ["Doc-2"]

    # Second sentence clauses
    assert "Vaishalee has 5+ years of software engineering experience" in claims[2].claim_text
    assert claims[2].cited_doc_ids == ["Doc-1"]

    assert "Sanjeev has 3 years of NLP research" in claims[3].claim_text
    assert claims[3].cited_doc_ids == ["Doc-2"]


def test_contrast_tail_stripping():
    """Verify that contrastive commentary tails ('rather than...', 'instead of...') are stripped to positive assertions."""
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### Technology Focus\n"
            "- Sanjeev specializes in NLP and PyTorch rather than web development frameworks [Doc-2].\n"
            "- Transformer architecture utilizes multi-head attention instead of recurrent LSTM cells [Doc-1]."
        ),
        cited_doc_ids=["Doc-1", "Doc-2"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    assert len(claims) == 2
    assert "rather than" not in claims[0].claim_text
    assert "web development" not in claims[0].claim_text
    assert "Sanjeev specializes in NLP and PyTorch" in claims[0].claim_text
    assert claims[0].cited_doc_ids == ["Doc-2"]

    assert "instead of" not in claims[1].claim_text
    assert "recurrent LSTM" not in claims[1].claim_text
    assert "Transformer architecture utilizes multi-head attention" in claims[1].claim_text
    assert claims[1].cited_doc_ids == ["Doc-1"]


def test_comparative_meta_synthesis_adjudication():
    """Verify that pure meta-analytical comparative synthesis claims evaluate as ENTAILED without NLI penalty."""
    adjudicator = AuditAdjudicator()
    verifier = DebertaNLIVerifier()

    context_map = {
        "Doc-1": "[Document: Resume-1 | Section: Experience]\nVaishalee has 5 years of web engineering experience.",
        "Doc-2": "[Document: Resume-2 | Section: Experience]\nSanjeev has 3 years of NLP research experience.",
    }

    claims = [
        AtomicClaim(
            claim_id="c_meta",
            claim_text="The candidates exhibit contrasting technical specializations across web engineering and NLP.",
            cited_doc_id="Doc-1, Doc-2",
            cited_doc_ids=["Doc-1", "Doc-2"],
            is_meta=True,
        ),
        AtomicClaim(
            claim_id="c_v",
            claim_text="Vaishalee has 5 years of web engineering experience.",
            cited_doc_id="Doc-1",
            cited_doc_ids=["Doc-1"],
            is_meta=False,
        ),
        AtomicClaim(
            claim_id="c_s",
            claim_text="Sanjeev has 3 years of NLP research experience.",
            cited_doc_id="Doc-2",
            cited_doc_ids=["Doc-2"],
            is_meta=False,
        ),
    ]

    report = adjudicator.adjudicate(
        claims=claims,
        context_map=context_map,
        nli_verifier=verifier,
        draft_text="Sample draft text.",
    )

    assert report.faithfulness_score == 1.0
    assert report.has_contradiction is False
    assert report.action == "PASS"
    assert len(report.audits) == 3
    assert report.audits[0].verdict == "ENTAILED"
    assert report.audits[0].confidence == 1.0
    assert report.audits[0].cited_premise == "Comparative Meta-Analytical Synthesis"
    assert report.audits[1].verdict == "ENTAILED"
    assert report.audits[2].verdict == "ENTAILED"


def test_sub_bullet_citation_context_inheritance():
    """Verify that nested sub-bullets inherit the active citation context of their parent item."""
    extractor = AtomicClaimExtractor()

    draft = GeneratedDraft(
        raw_text=(
            "### Work Experience\n"
            "- TechCorp Senior Engineer [Doc-1]\n"
            "  - Scaled distributed backend services to 100k QPS\n"
            "  - Led architectural migration to Kubernetes"
        ),
        cited_doc_ids=["Doc-1"],
        citations_valid=True,
    )

    claims = extractor.extract_claims(draft)

    assert len(claims) == 3
    assert all(c.cited_doc_id == "Doc-1" for c in claims)
    assert all(c.cited_doc_ids == ["Doc-1"] for c in claims)


def test_hypothesis_scaffolding_cleaning():
    """Verify that meta-document scaffolding is stripped from hypotheses for NLI verification."""
    adjudicator = AuditAdjudicator()

    # Resume carrier scaffolding
    c1 = "The resume for Sanjeev M focuses on leveraging generative AI to support women in business."
    assert adjudicator._clean_hypothesis_for_nli(c1) == "Leveraging generative AI to support women in business."

    # Spec carrier scaffolding
    c2 = "The engineering specifications document for the Attention Mechanism describes Multi-Head Attention as parallel layers."
    assert adjudicator._clean_hypothesis_for_nli(c2) == "Multi-Head Attention as parallel layers."

    # Document notes scaffolding
    c3 = "The specifications document notes that replacing sinusoidal positional encoding with learned embeddings yields identical results."
    assert adjudicator._clean_hypothesis_for_nli(c3) == "Replacing sinusoidal positional encoding with learned embeddings yields identical results."

    # Hardware carrier scaffolding
    c4 = "The hardware specification document states that the Model-X processor features 16 physical cores."
    assert adjudicator._clean_hypothesis_for_nli(c4) == "The Model-X processor features 16 physical cores."

    # According to scaffolding
    c5 = "According to the contract agreement for Acme Corp, the warranty period is 24 months."
    assert adjudicator._clean_hypothesis_for_nli(c5) == "The warranty period is 24 months."

    # Clean factual assertion without scaffolding should remain intact
    c6 = "The processor operates at 125W TDP."
    assert adjudicator._clean_hypothesis_for_nli(c6) == "The processor operates at 125W TDP."


def test_context_metadata_header_prepending():
    """Verify that _resolve_context_text prepends document and section metadata when available."""
    adjudicator = AuditAdjudicator()
    from src.common.schemas import RetrievalCandidate

    # Case 1: Candidate object without existing Document header
    cand = RetrievalCandidate(
        parent_id="p1",
        doc_id="specs/hardware",
        page_number=1,
        text="The Model-X operates at 125W TDP.",
        score=0.9,
        match_type="dense",
        section_name="Power Specs",
    )
    resolved = adjudicator._resolve_context_text(cand)
    assert "[Document: specs/hardware | Section: Power Specs]" in resolved
    assert "The Model-X operates at 125W TDP." in resolved

    # Case 2: Dict object without existing Document header
    ctx_dict = {
        "doc_id": "legal/nda",
        "section_name": "Confidentiality",
        "text": "Recipient shall not disclose Confidential Information.",
    }
    resolved_dict = adjudicator._resolve_context_text(ctx_dict)
    assert "[Document: legal/nda | Section: Confidentiality]" in resolved_dict
    assert "Recipient shall not disclose Confidential Information." in resolved_dict

    # Case 3: Raw text already containing header (no duplicate prepended)
    raw = "[Document: doc_a | Section: Sec1]\nSome text here."
    resolved_raw = adjudicator._resolve_context_text(raw)
    assert resolved_raw.count("[Document:") == 1


def test_adjudicator_premise_token_budget_bound():
    """Verify that extracted premise windows respect the configured premise_window_size bound."""
    adjudicator = AuditAdjudicator(premise_window_size=1200)

    long_body = " ".join([f"Sentence {i} describing system components and operational parameters." for i in range(100)])
    context = f"[Document: Arch_Doc | Section: Core]\n{long_body}"

    claim = "System components and operational parameters."
    premise = adjudicator._extract_premise_window(claim, context)

    assert len(premise) <= 1200
    assert "[Document: Arch_Doc | Section: Core]" in premise






