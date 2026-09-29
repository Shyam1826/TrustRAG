r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_5_self_correction/decomposer.py
   - Role: Multi-faceted query decomposition and atomic sub-intent generation engine.
   - Purpose: Analyzes compound queries combining multiple distinct targets, documents,
     or legal/technical aspects (e.g. "termination terms and governing law in Agreement A
     and Agreement B"). Decomposes complex prompts into atomic, single-intent sub-queries
     to prevent vector representation dilution and guarantee focused retrieval across
     all requested facets without domain-specific hardcoding.

2. INPUT (IP):
   - query (str): User natural language search query.

3. PROCESS UNDER THE HOOD:
   - Structural & Conjunction Pattern Extraction:
     * Identifies aspect groupings joined by conjunctions (`and`, `as well as`, `along with`, commas).
     * Identifies target entities or document comparators joined by `between X and Y`, `X and Y`,
       `versus`, `vs`, or `compared to`.
   - Combinatorial Aspect-Target Mapping:
     * Decomposes compound cross-document queries into atomic queries:
       `Sub-query(i, j) = f"{Aspect_j} in {Target_i}"`
     * For single-target multi-aspect queries:
       `Sub-query(j) = f"{Aspect_j} in {Target}"`
     * For multi-target single-aspect queries:
       `Sub-query(i) = f"{Aspect} in {Target_i}"`
   - Single-intent fallback: If no compound conjunctions are present, returns `[query]`.

4. OUTPUT (OP):
   - list[str]: List of atomic, focused sub-query strings.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library module for regex pattern matching.
   - typing (List, Optional, Tuple): Type annotations.
================================================================================
"""

import re
from typing import List


class QueryDecomposer:
    """Decomposes compound, multi-entity, and multi-facet queries into atomic sub-queries."""

    _COMP_BETWEEN_REGEX = re.compile(
        r"\b(?:between|across|for)\s+(?P<target1>.+?)\s+(?:and|versus|vs)\s+(?P<target2>.+?)(?:$|[.?!])",
        re.IGNORECASE,
    )

    _ASPECT_IN_TARGET_REGEX = re.compile(
        r"^(?:what\s+(?:is|are)\s+(?:the\s+)?|compare\s+(?:the\s+)?|show\s+me\s+(?:the\s+)?|find\s+(?:the\s+)?|tell\s+me\s+about\s+(?:the\s+)?)"
        r"(?P<aspects>.+?)\s+(?:in|of|for|under|between|regarding)\s+(?P<targets>.+?)[.?!]?$",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        pass

    def decompose(self, query: str) -> List[str]:
        """Decompose compound query into atomic, single-intent sub-queries.

        Args:
            query: Raw user query string.

        Returns:
            List of atomic sub-query strings.
        """
        if not query or not query.strip():
            return []

        clean_q = query.strip()

        # Match "What are <Aspects> in <Targets>" or "Compare <Aspects> between <Target1> and <Target2>"
        aspect_match = self._ASPECT_IN_TARGET_REGEX.match(clean_q)
        if aspect_match:
            aspects_raw = aspect_match.group("aspects").strip()
            targets_raw = aspect_match.group("targets").strip()

            # Split aspects by 'and', 'as well as', 'along with', or commas
            aspect_tokens = [
                a.strip()
                for a in re.split(r",\s*|\s+(?:and|as\s+well\s+as|along\s+with)\s+", aspects_raw)
                if a.strip() and len(a.strip()) > 1
            ]

            # Check if targets contain "between X and Y"
            comp_match = self._COMP_BETWEEN_REGEX.search(targets_raw)
            if comp_match:
                t1 = comp_match.group("target1").strip()
                t2 = comp_match.group("target2").strip()
                target_tokens = [t1, t2]
            else:
                # Check for "X and Y" in targets
                target_tokens = [
                    t.strip()
                    for t in re.split(r"\s+(?:and|versus|vs|compared\s+to)\s+", targets_raw)
                    if t.strip() and len(t.strip()) > 1
                ]

            if len(aspect_tokens) > 1 or len(target_tokens) > 1:
                sub_queries: List[str] = []
                for target in target_tokens:
                    for aspect in aspect_tokens:
                        sub_queries.append(f"{aspect} in {target}")
                return sub_queries

        # If not matched by structured regex, check general comparison between two targets
        comp_match_general = self._COMP_BETWEEN_REGEX.search(clean_q)
        if comp_match_general:
            t1 = comp_match_general.group("target1").strip()
            t2 = comp_match_general.group("target2").strip()
            prefix = clean_q[:comp_match_general.start()].strip()
            if prefix:
                return [f"{prefix} for {t1}", f"{prefix} for {t2}"]

        return [clean_q]

    def decompose_query(self, query: str) -> List[str]:
        """Alias for decompose."""
        return self.decompose(query)


def decompose_query(query: str) -> List[str]:
    """Functional wrapper for query decomposition."""
    decomposer = QueryDecomposer()
    return decomposer.decompose(query)
