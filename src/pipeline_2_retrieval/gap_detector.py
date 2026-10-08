r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/gap_detector.py
   - Role: Dynamic Knowledge Gap Detection & Grounding Safety Gating for TrustRAG.
   - Purpose: Detects knowledge deficits prior to generation and auditing by inspecting
     candidate retrieval density, cross-encoder score distributions, query-context lexical
     overlap, and refusal semantics. Dispatches queries between Closed-World Vault
     Grounding and Open-World General Knowledge Fallback.

2. INPUT (IP):
   - query (str): Natural language search query or user inquiry.
   - contexts (List[Any]): Retrieved candidate chunks (RetrievalCandidate or dicts) with scores and text.
   - min_confidence_floor (float, default=-6.0): Minimum acceptable reranker score floor.
   - draft_text (str): Synthesized candidate answer text from generation stage.

3. PROCESS UNDER THE HOOD:
   - `detect_gap()`:
     * If `len(contexts) == 0`: returns `(True, "NO_CANDIDATES")`.
     * Inspects candidate scores against `min_confidence_floor` (default -6.0).
     * Extracts salient non-stopword query tokens and computes lexical intersection against candidate corpora.
     * If all candidate scores fall strictly below `min_confidence_floor` and lack sufficient
       lexical overlap (`max_overlap <= 1` or overlap ratio < 0.35): returns `(True, "LOW_CONFIDENCE")`.
     * Otherwise returns `(False, "SUFFICIENT_EVIDENCE")`.
   - `is_refusal_response()`:
     * Evaluates candidate draft against standard refusal regex patterns ("insufficient information to answer",
       "does not contain sufficient information", "no relevant context", etc.).
     * Returns True if the draft is an uninformative fallback refusal, triggering dual-mode open-world redirection.

4. OUTPUT (OP):
   - Tuple[bool, str]: (gap_detected, gap_reason) indicating whether to trigger open-world fallback.
   - bool: Refusal detection verdict.
   - Consumed by: `src/main.py` (TrustRAGPipeline) and verification orchestrators.

5. LIBRARIES & DEPENDENCIES:
   - re: Regex pattern compilation and tokenization.
   - typing: Standard library typing primitives.
   - sklearn.feature_extraction.text (ENGLISH_STOP_WORDS): Universal lexical stop words.
================================================================================
"""

import re
from typing import Any, List, Optional, Sequence, Set, Tuple
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.pipeline_3_generation.prompt import FALLBACK_INSUFFICIENT_INFO


class KnowledgeGapDetector:
    """Detects knowledge gaps and evaluates refusal semantics for dual-mode routing."""

    _DEFAULT_STOP_WORDS: Set[str] = set(ENGLISH_STOP_WORDS) | {
        "what", "when", "where", "which", "who", "whom", "whose", "why", "how",
        "tell", "explain", "describe", "show", "give", "list", "does", "did", "is", "are",
    }

    _REFUSAL_PATTERNS = [
        re.compile(r"insufficient\s+information", re.IGNORECASE),
        re.compile(r"does\s+not\s+contain\s+sufficient\s+information", re.IGNORECASE),
        re.compile(r"not\s+contain\s+sufficient\s+information", re.IGNORECASE),
        re.compile(r"no\s+relevant\s+context", re.IGNORECASE),
        re.compile(r"no\s+relevant\s+information", re.IGNORECASE),
        re.compile(r"not\s+enough\s+information", re.IGNORECASE),
        re.compile(r"cannot\s+be\s+answered\s+based\s+on", re.IGNORECASE),
        re.compile(r"provided\s+documentation\s+does\s+not\s+contain", re.IGNORECASE),
        re.compile(r"provided\s+context\s+does\s+not\s+contain", re.IGNORECASE),
        re.compile(r"is\s+not\s+mentioned\s+in\s+the\s+provided", re.IGNORECASE),
        re.compile(r"documents?\s+do(?:es)?\s+not\s+mention", re.IGNORECASE),
        re.compile(r"context\s+does\s+not\s+mention", re.IGNORECASE),
        re.compile(r"no\s+information\s+(?:is\s+)?provided", re.IGNORECASE),
        re.compile(r"information\s+is\s+not\s+available", re.IGNORECASE),
    ]

    def __init__(self, stop_words: Optional[Set[str]] = None) -> None:
        """Initialize the KnowledgeGapDetector.

        Args:
            stop_words: Optional custom stop words set.
        """
        self.stop_words = stop_words if stop_words is not None else self._DEFAULT_STOP_WORDS

    def _extract_query_tokens(self, query: str) -> Set[str]:
        """Extract salient non-stopword query tokens."""
        words = re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", query.lower())
        return {w for w in words if w not in self.stop_words}

    def _get_candidate_text(self, candidate: Any) -> str:
        """Extract textual content from diverse candidate representations."""
        if hasattr(candidate, "text") and candidate.text:
            return candidate.text
        if isinstance(candidate, dict):
            return candidate.get("text", "") or candidate.get("raw_text", "")
        if hasattr(candidate, "raw_text") and candidate.raw_text:
            return candidate.raw_text
        return str(candidate)

    def _get_candidate_score(self, candidate: Any) -> float:
        """Extract score from candidate object or dictionary."""
        if hasattr(candidate, "score") and candidate.score is not None:
            return float(candidate.score)
        if isinstance(candidate, dict) and "score" in candidate:
            return float(candidate["score"])
        return 0.0

    def detect_gap(
        self,
        query: str,
        contexts: Sequence[Any],
        min_confidence_floor: float = -6.0,
    ) -> Tuple[bool, str]:
        """Detect whether retrieved context exhibits a knowledge gap for the given query.

        Args:
            query: Natural language query.
            contexts: List or sequence of retrieved candidate objects.
            min_confidence_floor: Empirical confidence floor below which candidates are dubious.

        Returns:
            Tuple[bool, str]: (gap_detected, gap_reason).
        """
        if not contexts or len(contexts) == 0:
            return True, "NO_CANDIDATES"

        scores = [self._get_candidate_score(c) for c in contexts]
        all_low_scores = all(s < min_confidence_floor for s in scores)

        query_tokens = self._extract_query_tokens(query)

        if not query_tokens:
            if all_low_scores:
                return True, "LOW_CONFIDENCE"
            return False, "SUFFICIENT_EVIDENCE"

        # Calculate lexical token overlap with candidate corpora
        max_overlap = 0
        for cand in contexts:
            text = self._get_candidate_text(cand)
            cand_tokens = set(re.findall(r"\b[a-zA-Z0-9_-]{2,}\b", text.lower()))
            overlap = len(query_tokens & cand_tokens)
            if overlap > max_overlap:
                max_overlap = overlap

        overlap_ratio = max_overlap / max(len(query_tokens), 1)

        # Flag gap if all scores fall strictly below confidence floor and lack sufficient lexical overlap
        if all_low_scores and (max_overlap <= 1 or overlap_ratio < 0.35):
            return True, "LOW_CONFIDENCE"

        return False, "SUFFICIENT_EVIDENCE"

    def is_refusal_response(self, draft_text: str) -> bool:
        """Evaluate whether a generated draft is an uninformative refusal response.

        Args:
            draft_text: Synthesized answer string.

        Returns:
            bool: True if draft expresses refusal or lack of knowledge.
        """
        if not draft_text or not draft_text.strip():
            return True

        stripped = draft_text.strip()
        if stripped == FALLBACK_INSUFFICIENT_INFO.strip():
            return True

        for pattern in self._REFUSAL_PATTERNS:
            if pattern.search(stripped):
                return True

        return False
