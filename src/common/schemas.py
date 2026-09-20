r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/common/schemas.py
   - Role: Type system and data contracts layer for TrustRAG.
   - Purpose: Defines strict Pydantic v2 data models for inter-pipeline contracts,
     including hierarchical chunks with section breadcrumbs, retrieval candidates,
     generated drafts, atomic claims, and trust audit reports.

2. INPUT (IP):
   - Raw data structures, model outputs, and pipeline state dictionaries.

3. PROCESS UNDER THE HOOD:
   - Enforces Pydantic v2 `extra="forbid"` and `validate_assignment=True`.
   - Models data flow from ingestion (ChildChunk, ParentChunk) to retrieval
     (RetrievalCandidate), generation (GeneratedDraft), and verification (AtomicClaim,
     ClaimAudit, TrustAuditReport).
   - Tracks `chunk_index` and `section_name` for structural scope and neighbor expansion.

4. OUTPUT (OP):
   - Strictly validated Pydantic model classes and instances.
   - Consumed by all modules across Pipelines 1 to 4.

5. LIBRARIES & DEPENDENCIES:
   - pydantic: Schema validation and JSON serialization.
   - typing: Standard library typing constructs.
================================================================================
"""

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field


class StrictBaseModel(BaseModel):
    """Base schema with strict extra field validation."""
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ChildChunk(StrictBaseModel):
    """Fine-grained chunk used for dense vector and sparse lexical indexing."""
    chunk_id: str
    parent_id: str
    doc_id: str
    text: str
    vector: Optional[List[float]] = None
    sparse_tokens: Optional[Dict[str, int]] = None
    page_number: int
    chunk_index: int = 0
    section_name: str = "General"
    relative_path: Optional[str] = None
    folder_hierarchy: Optional[List[str]] = None


class ParentChunk(StrictBaseModel):
    """Coarse-grained context chunk holding rich surrounding document context."""
    parent_id: str
    doc_id: str
    text: str
    page_number: int
    child_ids: List[str] = Field(default_factory=list)
    chunk_index: int = 0
    section_name: str = "General"
    relative_path: Optional[str] = None
    folder_hierarchy: Optional[List[str]] = None


class RetrievalCandidate(StrictBaseModel):
    """Candidate chunk returned by hybrid retrieval and cross-encoder reranker."""
    parent_id: str
    doc_id: str
    page_number: int
    text: str
    score: float
    match_type: str
    chunk_index: int = 0
    section_name: str = "General"
    relative_path: Optional[str] = None
    folder_hierarchy: Optional[List[str]] = None


class GeneratedDraft(StrictBaseModel):
    """Synthesized response with inline citation mapping."""
    raw_text: str
    cited_doc_ids: List[str] = Field(default_factory=list)
    citations_valid: bool


class AtomicClaim(StrictBaseModel):
    """Atomic factual statement extracted from generated text for verification."""
    claim_id: str
    claim_text: str
    cited_doc_id: Optional[str] = None


class ClaimAudit(StrictBaseModel):
    """NLI verification result for an individual atomic claim against its cited premise."""
    claim_id: str
    claim_text: str
    cited_premise: str
    probabilities: Dict[str, float]
    verdict: Literal["ENTAILED", "CONTRADICTION", "NEUTRAL"]
    confidence: float


class TrustAuditReport(StrictBaseModel):
    """Aggregate trust evaluation and action verdict for a generated draft."""
    draft_text: str
    faithfulness_score: float
    has_contradiction: bool
    action: Literal["PASS", "TRIGGER_REWRITE", "WARN"]
    audits: List[ClaimAudit] = Field(default_factory=list)
