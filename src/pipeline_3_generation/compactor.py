r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_3_generation/compactor.py
   - Role: Adaptive Context Compaction engine for TrustRAG.
   - Purpose: Minimizes prompt token payloads and eliminates non-informative boilerplate
     by dynamically compacting narrative text candidates to target sentence spans and
     their immediate bounding contexts (1 preceding, 1 succeeding sentence). Retains
     tabular key-value candidates verbatim and enforces a hard context ceiling to
     prevent provider rate-limiting (ITPM/TPM) and reduce generation latency.

2. INPUT (IP):
   - query (str): Natural language query used to locate salient concept spans.
   - candidates (list[RetrievalCandidate]): Candidate retrieval models from Pipeline 2.
   - max_total_chars (int): Hard safety ceiling on aggregate context characters (default 4000).

3. PROCESS UNDER THE HOOD:
   - Tabular Candidate Bypass:
     * Checks if candidate represents structured tabular data (`[Section: Table: ...]` or
       `match_type="tabular_sql"`). If so, preserves row key-value text verbatim.
   - Narrative Sentence Compaction:
     * Preserves section breadcrumbs (`[Section: ...]`) while isolating narrative body text.
     * Segments body text into sentence boundaries using syntactic boundary regex.
     * Extracts significant non-stop query terms (length >= 3).
     * Identifies sentence indices matching query terms and expands by ±1 bounding sentence.
     * Merges contiguous/overlapping sentence intervals.
     * Falls back to head sentences if no lexical overlap exists for semantic dense matches.
   - Aggregated Context Bounding:
     * Truncates context candidates once total character count reaches `max_total_chars`
       while guaranteeing at least top candidates are included.

4. OUTPUT (OP):
   - List[RetrievalCandidate]: Compacted retrieval candidates ready for prompt serialization.
   - Consumed by: `src/pipeline_3_generation/prompt.py`.

5. LIBRARIES & DEPENDENCIES:
   - re: Regex boundary splitting and token extraction.
   - typing (List, Optional, Set, Tuple).
   - sklearn.feature_extraction.text.ENGLISH_STOP_WORDS: Standard NLP stop words.
   - src.common.config: Configurable stop words and parameters.
   - src.common.schemas.RetrievalCandidate: Candidate schema.
================================================================================
"""

import re
from typing import List, Optional, Set
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config
from src.common.schemas import RetrievalCandidate

DEFAULT_MAX_COMPACT_CHARS = 4000


class ContextCompactor:
    """Dynamically compacts retrieval candidates to targeted sentence spans and bounding context."""

    _SENTENCE_BOUNDARY_REGEX = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z0-9\"'•\-])|\n+",
        re.MULTILINE,
    )
    _SECTION_PREFIX_REGEX = re.compile(r"^(\[[^\]]+\])\s*(.*)$", re.DOTALL)

    def __init__(self, custom_stop_words: Optional[Set[str]] = None) -> None:
        """Initialize compactor with stop words configuration."""
        configured_stops = set(config.retrieval.custom_stop_words) if hasattr(config, "retrieval") else set()
        self.stop_words: Set[str] = ENGLISH_STOP_WORDS | configured_stops | (custom_stop_words or set())

    def _extract_query_tokens(self, query: str) -> Set[str]:
        """Extract significant lowercase alphanumeric query tokens."""
        if not query:
            return set()
        tokens = re.findall(r"[a-zA-Z0-9_-]+", query.lower())
        return {t for t in tokens if len(t) >= 3 and t not in self.stop_words}

    def compact_candidate(self, query: str, candidate: RetrievalCandidate) -> RetrievalCandidate:
        """Compact a single retrieval candidate's text while preserving metadata.

        Args:
            query: Target query string.
            candidate: Candidate model to compact.

        Returns:
            New or modified RetrievalCandidate with compacted text.
        """
        raw_text = candidate.text.strip() if candidate.text else ""

        # 1. Retain Tabular Candidates Verbatim
        if (
            candidate.match_type == "tabular_sql"
            or "[Section: Table:" in raw_text
            or (candidate.section_name and candidate.section_name.startswith("Table:"))
        ):
            return candidate

        # 2. Skip compaction if candidate is already short (<= 350 chars)
        if len(raw_text) <= 350:
            return candidate

        # 3. Separate section prefix if present
        prefix = ""
        body = raw_text
        prefix_match = self._SECTION_PREFIX_REGEX.match(raw_text)
        if prefix_match:
            prefix = prefix_match.group(1).strip()
            body = prefix_match.group(2).strip()

        # 4. Segment into sentences
        sentences = [s.strip() for s in self._SENTENCE_BOUNDARY_REGEX.split(body) if s.strip()]
        if len(sentences) <= 2:
            return candidate

        # 5. Locate matching sentence spans
        query_tokens = self._extract_query_tokens(query)
        scored_matches: List[Tuple[int, int]] = []  # (overlap_count, sentence_index)

        for idx, sent in enumerate(sentences):
            s_tokens = {t.lower() for t in re.findall(r"[a-zA-Z0-9_-]+", sent)}
            overlap = len(s_tokens & query_tokens)
            if overlap > 0:
                scored_matches.append((overlap, idx))

        # 6. Expand by 1 preceding and 1 succeeding sentence
        if scored_matches:
            # If many sentences match, prioritize top 4 by query token overlap
            if len(scored_matches) > 4:
                top_matches = sorted(scored_matches, key=lambda x: x[0], reverse=True)[:4]
                target_indices = {m[1] for m in top_matches}
            else:
                target_indices = {m[1] for m in scored_matches}

            selected_indices: Set[int] = set()
            for m_idx in target_indices:
                start = max(0, m_idx - 1)
                end = min(len(sentences) - 1, m_idx + 1)
                for i in range(start, end + 1):
                    selected_indices.add(i)

            compacted_body = " ".join(sentences[i] for i in sorted(selected_indices))
        else:
            # Fallback for purely semantic dense matches without lexical overlap:
            # Retain leading sentences up to ~450 characters
            accumulated: List[str] = []
            curr_len = 0
            for sent in sentences:
                accumulated.append(sent)
                curr_len += len(sent)
                if curr_len >= 450:
                    break
            compacted_body = " ".join(accumulated)

        # Re-attach section prefix if present
        compacted_text = f"{prefix} {compacted_body}".strip() if prefix else compacted_body

        # Return a copy of candidate with compacted text
        return candidate.model_copy(update={"text": compacted_text})

    def compact_contexts(
        self,
        query: str,
        contexts: List[RetrievalCandidate],
        max_total_chars: int = DEFAULT_MAX_COMPACT_CHARS,
    ) -> List[RetrievalCandidate]:
        """Compact a list of retrieval candidates and enforce maximum aggregate character budget.

        Args:
            query: Target query string.
            contexts: List of candidate retrieval models.
            max_total_chars: Maximum character budget across all formatted candidates.

        Returns:
            List of compacted RetrievalCandidate objects fitting within character budget.
        """
        if not contexts:
            return []

        compacted_list: List[RetrievalCandidate] = []
        current_chars = 0

        for idx, candidate in enumerate(contexts, start=1):
            compacted = self.compact_candidate(query, candidate)
            cand_len = len(compacted.text)

            # Enforce budget guard: stop appending if we exceed the budget and have at least 2 passages
            if current_chars + cand_len > max_total_chars and idx > 2:
                break

            compacted_list.append(compacted)
            current_chars += cand_len

        return compacted_list
