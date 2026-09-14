"""Tests for Pydantic v2 schemas instantiation and serialization."""

import pytest
from pydantic import ValidationError

from src.common.schemas import (
    ChildChunk,
    ParentChunk,
    RetrievalCandidate,
    GeneratedDraft,
    AtomicClaim,
    ClaimAudit,
    TrustAuditReport,
)


def test_child_chunk_instantiation_and_serialization():
    chunk = ChildChunk(
        chunk_id="c_001",
        parent_id="p_001",
        doc_id="doc_101",
        text="Sample child chunk text.",
        vector=[0.1, 0.2, 0.3],
        sparse_tokens={"sample": 1, "chunk": 2},
        page_number=1,
    )
    assert chunk.chunk_id == "c_001"
    assert chunk.vector == [0.1, 0.2, 0.3]
    assert chunk.sparse_tokens == {"sample": 1, "chunk": 2}

    # Serialization test
    dumped_dict = chunk.model_dump()
    assert dumped_dict["chunk_id"] == "c_001"
    assert dumped_dict["page_number"] == 1

    json_str = chunk.model_dump_json()
    reloaded = ChildChunk.model_validate_json(json_str)
    assert reloaded == chunk

    # Optional fields default to None
    minimal_chunk = ChildChunk(
        chunk_id="c_002",
        parent_id="p_001",
        doc_id="doc_101",
        text="Minimal chunk text",
        page_number=2,
    )
    assert minimal_chunk.vector is None
    assert minimal_chunk.sparse_tokens is None


def test_parent_chunk_instantiation_and_serialization():
    parent = ParentChunk(
        parent_id="p_001",
        doc_id="doc_101",
        text="Comprehensive parent document text spanning several paragraphs.",
        page_number=1,
        child_ids=["c_001", "c_002"],
    )
    assert parent.parent_id == "p_001"
    assert parent.child_ids == ["c_001", "c_002"]

    json_str = parent.model_dump_json()
    reloaded = ParentChunk.model_validate_json(json_str)
    assert reloaded == parent


def test_retrieval_candidate_instantiation_and_serialization():
    candidate = RetrievalCandidate(
        parent_id="p_001",
        doc_id="doc_101",
        page_number=1,
        text="Relevant context extracted from parent chunk.",
        score=0.925,
        match_type="dense_sparse_hybrid",
    )
    assert candidate.score == 0.925
    assert candidate.match_type == "dense_sparse_hybrid"

    dumped_dict = candidate.model_dump()
    reloaded = RetrievalCandidate.model_validate(dumped_dict)
    assert reloaded == candidate


def test_generated_draft_instantiation_and_serialization():
    draft = GeneratedDraft(
        raw_text="The company revenue increased by 20% in Q3 [doc_101].",
        cited_doc_ids=["doc_101"],
        citations_valid=True,
    )
    assert draft.citations_valid is True
    assert draft.cited_doc_ids == ["doc_101"]

    json_str = draft.model_dump_json()
    reloaded = GeneratedDraft.model_validate_json(json_str)
    assert reloaded == draft


def test_atomic_claim_instantiation_and_serialization():
    claim = AtomicClaim(
        claim_id="clm_001",
        claim_text="Revenue increased by 20% in Q3.",
        cited_doc_id="doc_101",
    )
    assert claim.claim_id == "clm_001"
    assert claim.cited_doc_id == "doc_101"

    # Test optional cited_doc_id
    claim_no_cite = AtomicClaim(
        claim_id="clm_002",
        claim_text="Revenue was positive.",
    )
    assert claim_no_cite.cited_doc_id is None


def test_claim_audit_instantiation_and_validation():
    audit = ClaimAudit(
        claim_id="clm_001",
        claim_text="Revenue increased by 20% in Q3.",
        cited_premise="In Q3, the company reported a 20% increase in revenue.",
        probabilities={"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01},
        verdict="ENTAILED",
        confidence=0.95,
    )
    assert audit.verdict == "ENTAILED"
    assert audit.probabilities["entailment"] == 0.95

    json_str = audit.model_dump_json()
    reloaded = ClaimAudit.model_validate_json(json_str)
    assert reloaded == audit

    # Invalid verdict Literal must fail
    with pytest.raises(ValidationError):
        ClaimAudit(
            claim_id="clm_002",
            claim_text="Test",
            cited_premise="Test",
            probabilities={"entailment": 0.1, "neutral": 0.1, "contradiction": 0.8},
            verdict="INVALID_VERDICT",  # type: ignore
            confidence=0.8,
        )


def test_trust_audit_report_instantiation_and_serialization():
    audit1 = ClaimAudit(
        claim_id="clm_001",
        claim_text="Revenue increased by 20% in Q3.",
        cited_premise="In Q3, the company reported a 20% increase in revenue.",
        probabilities={"entailment": 0.95, "neutral": 0.04, "contradiction": 0.01},
        verdict="ENTAILED",
        confidence=0.95,
    )
    report = TrustAuditReport(
        draft_text="The company revenue increased by 20% in Q3 [doc_101].",
        faithfulness_score=1.0,
        has_contradiction=False,
        action="PASS",
        audits=[audit1],
    )
    assert report.faithfulness_score == 1.0
    assert report.has_contradiction is False
    assert report.action == "PASS"
    assert len(report.audits) == 1

    json_str = report.model_dump_json()
    reloaded = TrustAuditReport.model_validate_json(json_str)
    assert reloaded == report

    # Invalid action Literal must fail
    with pytest.raises(ValidationError):
        TrustAuditReport(
            draft_text="Draft",
            faithfulness_score=0.5,
            has_contradiction=True,
            action="UNKNOWN_ACTION",  # type: ignore
            audits=[],
        )


def test_strict_extra_field_rejection():
    # Extra unexpected fields must raise ValidationError
    with pytest.raises(ValidationError):
        ChildChunk(
            chunk_id="c_001",
            parent_id="p_001",
            doc_id="doc_101",
            text="Text",
            page_number=1,
            extra_field="unexpected",  # type: ignore
        )
