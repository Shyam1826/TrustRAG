r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_knowledge_gap.py
   - Role: Test suite for dynamic Knowledge Gap Detection, citation-to-premise routing,
     and Dual-Mode generation fallback.
   - Purpose: Verifies programmatic citation-to-premise routing in the adjudicator,
     evaluates `KnowledgeGapDetector` edge cases, and verifies Dual-Mode routing
     (Closed-World vs Open-World Fallback) using dynamically randomized inputs.

2. INPUT (IP):
   - Programmatically generated synthetic chunks, random UUID tokens, and randomized probe strings.

3. PROCESS UNDER THE HOOD:
   - Test 1: Evaluates dynamic citation-to-premise routing in `AuditAdjudicator.verify_claims`:
     verifies that claims with `[Doc-2]` cite Chunk 2's premise, `[Doc-1]` cites Chunk 1's premise,
     and un-cited claims fall back to the top-ranked candidate.
   - Test 2: Validates `KnowledgeGapDetector` logic across NO_CANDIDATES, LOW_CONFIDENCE,
     SUFFICIENT_EVIDENCE, and refusal semantic detection.
   - Test 3 (Test A): Executes closed-world retrieval and generation with synthetic entities,
     verifying `GroundingMode.CLOSED_WORLD` and citation integrity.
   - Test 4 (Test B): Injects randomized out-of-domain probe queries, verifying that the
     Knowledge Gap is flagged, routing to `GroundingMode.OPEN_WORLD_FALLBACK`, outputting the
     standard enterprise disclaimer, and setting trust_score to 0.0 with empty audits.

4. OUTPUT (OP):
   - Pytest assertions and test status reports.

5. LIBRARIES & DEPENDENCIES:
   - pytest: Test execution and assertions.
   - uuid, random: Randomized synthetic fixtures.
   - src.common.schemas: GroundingMode, AtomicClaim, RetrievalCandidate, TrustAuditReport.
   - src.pipeline_2_retrieval.gap_detector: KnowledgeGapDetector.
   - src.pipeline_3_generation.generator: OPEN_WORLD_DISCLAIMER, MockGenerator.
   - src.pipeline_4_verification.adjudicator: AuditAdjudicator, Adjudicator.
   - src.main: TrustRAGPipeline.
================================================================================
"""

import random
import uuid
import pytest

from src.common.schemas import AtomicClaim, GroundingMode, RetrievalCandidate, TrustAuditReport
from src.pipeline_2_retrieval.gap_detector import KnowledgeGapDetector
from src.pipeline_3_generation.generator import OPEN_WORLD_DISCLAIMER, MockGenerator
from src.pipeline_4_verification.adjudicator import Adjudicator, AuditAdjudicator


class MockNLIVerifier:
    """Fast deterministic mock NLI verifier for unit testing."""

    def predict_batch(self, claims, premises, batch_size=32):
        results = []
        for c, p in zip(claims, premises):
            # If claim words appear in premise, entail; else neutral
            c_words = set(c.lower().split())
            p_words = set(p.lower().split())
            if len(c_words & p_words) >= 2:
                results.append({
                    "verdict": "ENTAILED",
                    "probabilities": {"entailment": 0.95, "contradiction": 0.01, "neutral": 0.04},
                })
            else:
                results.append({
                    "verdict": "NEUTRAL",
                    "probabilities": {"entailment": 0.10, "contradiction": 0.05, "neutral": 0.85},
                })
        return results


def test_dynamic_citation_to_premise_routing():
    """Verify that verify_claims dynamically routes premises based on citation tags [Doc-k]."""
    token_a = f"alpha_{uuid.uuid4().hex[:6]}"
    token_b = f"beta_{uuid.uuid4().hex[:6]}"
    freq_a = random.randint(100, 999)
    freq_b = random.randint(1000, 9999)

    chunk_1 = RetrievalCandidate(
        parent_id="p1",
        doc_id="doc_alpha",
        page_number=1,
        text=f"The primary engine {token_a} operates at frequency {freq_a} MHz with dedicated cooling.",
        score=2.5,
        match_type="dense",
    )
    chunk_2 = RetrievalCandidate(
        parent_id="p2",
        doc_id="doc_beta",
        page_number=2,
        text=f"The secondary coprocessor {token_b} delivers bandwidth {freq_b} Gbps across the bus.",
        score=1.8,
        match_type="dense",
    )

    adjudicator = AuditAdjudicator(nli_verifier=MockNLIVerifier())

    # Claim citing [Doc-2] should align to chunk_2
    claim_2 = AtomicClaim(
        claim_id="c_2",
        claim_text=f"The secondary coprocessor {token_b} delivers bandwidth {freq_b} Gbps [Doc-2].",
    )
    report_2 = adjudicator.verify_claims([claim_2], [chunk_1, chunk_2])
    assert len(report_2.audits) == 1
    assert token_b in report_2.audits[0].cited_premise
    assert token_a not in report_2.audits[0].cited_premise
    assert report_2.audits[0].verdict == "ENTAILED"

    # Claim citing [Doc-1] should align to chunk_1
    claim_1 = AtomicClaim(
        claim_id="c_1",
        claim_text=f"The primary engine {token_a} operates at frequency {freq_a} MHz [Doc-1].",
    )
    report_1 = adjudicator.verify_claims([claim_1], [chunk_1, chunk_2])
    assert len(report_1.audits) == 1
    assert token_a in report_1.audits[0].cited_premise
    assert token_b not in report_1.audits[0].cited_premise
    assert report_1.audits[0].verdict == "ENTAILED"

    # Claim without citation falls back to top-ranked context (chunk_1)
    claim_none = AtomicClaim(
        claim_id="c_none",
        claim_text=f"The primary engine {token_a} operates with dedicated cooling.",
    )
    report_none = adjudicator.verify_claims([claim_none], [chunk_1, chunk_2])
    assert len(report_none.audits) == 1
    assert token_a in report_none.audits[0].cited_premise


def test_knowledge_gap_detector_logic():
    """Verify KnowledgeGapDetector across NO_CANDIDATES, LOW_CONFIDENCE, and refusals."""
    detector = KnowledgeGapDetector()
    rand_query = f"query_{uuid.uuid4().hex[:8]} what is the status of the entity?"

    # 1. NO_CANDIDATES
    gap_detected, reason = detector.detect_gap(rand_query, [])
    assert gap_detected is True
    assert reason == "NO_CANDIDATES"

    # 2. LOW_CONFIDENCE (all scores < -6.0 and no lexical overlap)
    low_cand = RetrievalCandidate(
        parent_id="p_low",
        doc_id="unrelated_doc",
        page_number=1,
        text="Completely disjoint context about historical agriculture and farming implements.",
        score=-10.5,
        match_type="sparse",
    )
    gap_detected, reason = detector.detect_gap(rand_query, [low_cand])
    assert gap_detected is True
    assert reason == "LOW_CONFIDENCE"

    # 3. SUFFICIENT_EVIDENCE (score >= -6.0)
    high_cand = RetrievalCandidate(
        parent_id="p_high",
        doc_id="relevant_doc",
        page_number=1,
        text=f"Specific facts regarding {rand_query} are explicitly documented here.",
        score=2.1,
        match_type="dense",
    )
    gap_detected, reason = detector.detect_gap(rand_query, [high_cand])
    assert gap_detected is False
    assert reason == "SUFFICIENT_EVIDENCE"

    # 4. Refusal detection
    assert detector.is_refusal_response("The provided documentation does not contain sufficient information to answer.")
    assert detector.is_refusal_response("No relevant context could be retrieved.")
    assert detector.is_refusal_response("Insufficient information to determine the answer.")
    assert not detector.is_refusal_response("The device operates at 450 MHz clock speed [Doc-1].")


def test_pipeline_closed_world_mode_with_synthetic_fixtures(monkeypatch):
    """Test A: Verify pipeline routes to CLOSED_WORLD mode with PASS when confidence is sufficient."""
    from src.main import TrustRAGPipeline

    rand_tag = f"synth_{uuid.uuid4().hex[:6]}"
    rand_val = f"val_{random.randint(50, 500)}"

    cand = RetrievalCandidate(
        parent_id="p_synth",
        doc_id=f"doc_{rand_tag}",
        page_number=1,
        text=f"The specifications for {rand_tag} specify a operating metric of {rand_val}.",
        score=3.5,
        match_type="dense",
    )

    pipeline = TrustRAGPipeline(generator_type="mock")
    pipeline.nli_verifier = MockNLIVerifier()

    # Mock retrieve to return the synthetic candidate
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: [cand])

    query = f"What is the operating metric of {rand_tag}?"
    report = pipeline.ask(query)

    assert report.grounding_mode == GroundingMode.CLOSED_WORLD
    assert report.action in ("PASS", "WARN")
    assert report.trust_score is not None
    assert report.source_attribution == "DOCUMENT_VAULT"


def test_pipeline_open_world_fallback_mode_with_randomized_probes(monkeypatch):
    """Test B: Verify pipeline flags knowledge gap and routes to OPEN_WORLD_FALLBACK with disclaimer and 0.0 score."""
    from src.main import TrustRAGPipeline

    pipeline = TrustRAGPipeline(generator_type="mock")

    # Probe 1: Zero candidates returned
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: [])
    rand_probe_1 = f"probe_{uuid.uuid4().hex} what are the historical treaties of 1650?"
    report_1 = pipeline.ask(rand_probe_1)

    assert report_1.grounding_mode == GroundingMode.OPEN_WORLD_FALLBACK
    assert report_1.trust_score == 0.0
    assert report_1.verdict == "UNVERIFIED_OPEN_WORLD"
    assert report_1.source_attribution == "OPEN_WORLD_GENERAL_KNOWLEDGE"
    assert report_1.gap_reason == "NO_CANDIDATES"
    assert OPEN_WORLD_DISCLAIMER in report_1.draft_text
    assert len(report_1.audits) == 0

    # Probe 2: Low-confidence candidates below floor (-11.2) with no lexical overlap
    low_cand = RetrievalCandidate(
        parent_id="p_unrelated",
        doc_id="unrelated_doc",
        page_number=1,
        text="Standard contract boilerplates and liability waivers for equipment leasing.",
        score=-11.2,
        match_type="dense",
    )
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: [low_cand])
    rand_probe_2 = f"quantum_fluctuation_{uuid.uuid4().hex[:8]} astrophysics question"
    report_2 = pipeline.ask(rand_probe_2)

    assert report_2.grounding_mode == GroundingMode.OPEN_WORLD_FALLBACK
    assert report_2.trust_score == 0.0
    assert report_2.verdict == "UNVERIFIED_OPEN_WORLD"
    assert report_2.source_attribution == "OPEN_WORLD_GENERAL_KNOWLEDGE"
    assert report_2.gap_reason == "LOW_CONFIDENCE"
    assert OPEN_WORLD_DISCLAIMER in report_2.draft_text
    assert len(report_2.audits) == 0

    # Probe 3: Closed-world generator outputs refusal
    monkeypatch.setattr(
        pipeline.generator,
        "generate_answer",
        lambda *args, **kwargs: "The provided documentation does not contain sufficient information to answer.",
    )
    cand_high = RetrievalCandidate(
        parent_id="p_high",
        doc_id="some_doc",
        page_number=1,
        text="Some text that matched vaguely.",
        score=1.5,
        match_type="dense",
    )
    monkeypatch.setattr(pipeline, "retrieve", lambda *args, **kwargs: [cand_high])
    rand_probe_3 = f"arbitrary_probe_{uuid.uuid4().hex[:6]}"
    report_3 = pipeline.ask(rand_probe_3)

    assert report_3.grounding_mode == GroundingMode.OPEN_WORLD_FALLBACK
    assert report_3.trust_score == 0.0
    assert report_3.gap_reason == "CLOSED_WORLD_REFUSAL"
    assert OPEN_WORLD_DISCLAIMER in report_3.draft_text
