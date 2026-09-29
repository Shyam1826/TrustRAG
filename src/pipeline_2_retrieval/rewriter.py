r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/rewriter.py
   - Role: Semantic query preprocessing, conversational noise reduction, and scope routing.
   - Purpose: Strips conversational filler phrases and preambles from user queries.
     Enforces 100% global semantic vector and BM25 retrieval by default across all documents
     and folders without keyword-based filename locking. Dynamically routes explicit folder
     domain targets and scope directives (e.g. `scope: engineering` or `under legal/2026`).

2. INPUT (IP):
   - query (str): Raw user query string.
   - known_doc_ids (list[str], optional): List of indexed document identifiers in the corpus.
   - doc_entity_map (dict[str, str], optional): Optional mapping from entity aliases to doc_ids.

3. PROCESS UNDER THE HOOD:
   - Strips conversational preambles and normalizes punctuation.
   - Dynamic Folder & Scope Routing:
     * Delegates to `ScopeRouter` for canonical token normalization, n-gram overlap,
       and folder domain routing.
   - Global Semantic Default:
     * If no explicit folder scope or direct mapping is detected, defaults strictly
       to `(None, [])` so dense vector embeddings and BM25 retrieve globally based on semantic meaning.
   - Uses standard `sklearn.feature_extraction.text.ENGLISH_STOP_WORDS` combined with
     `config.retrieval.custom_stop_words` for stop token resolution.
   - Collapses consecutive whitespace characters into a single space.

4. OUTPUT (OP):
   - str: Focused, standalone query string.
   - tuple[Optional[Union[str, list[str]]], list[str]]: `(doc_filter, matched_entity_or_folder_tokens)`.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library module for regex pattern matching.
   - sklearn.feature_extraction.text (ENGLISH_STOP_WORDS): Standard NLP stop words.
   - typing (Dict, List, Optional, Tuple, Union, Set): Type annotations.
   - src.common.config: Provides configuration settings.
   - src.pipeline_2_retrieval.router (ScopeRouter): Canonical scope router.
================================================================================
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config
from src.pipeline_2_retrieval.router import ScopeRouter


class QueryTransformer:
    """Transforms raw conversational queries and extracts document and folder domain filters."""

    # Pre-compiled regex patterns for conversational noise and filler phrases
    _FILLER_PATTERNS = [
        r"^(?:can|could|would)\s+you\s+(?:please\s+)?(?:tell|explain|show|find|give|provide)\s+(?:me\s+)?(?:about\s+)?",
        r"^(?:please\s+)?(?:search\s+for|look\s+up|find\s+(?:out|information\s+about)?)\s*",
        r"^(?:what\s+(?:is|are|was|were)\s+(?:the\s+)?|who\s+(?:is|are|was|were)\s+(?:the\s+)?)",
        r"^(?:tell\s+me\s+about|i\s+want\s+to\s+know\s+(?:about\s+)?|i\s+need\s+information\s+(?:on|about)\s*)",
        r"^(?:give\s+me\s+(?:the\s+)?details\s+(?:on|about)\s*|explain\s+to\s+me\s*)",
        r"^(?:do\s+you\s+know\s+(?:about\s+)?|how\s+do\s+i\s+find\s*)",
    ]

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._FILLER_PATTERNS
        ]
        self._router = ScopeRouter()

    def _get_stop_words(self) -> Set[str]:
        """Combine standard English stop words with configured custom stop words."""
        custom = set(config.retrieval.custom_stop_words)
        return ENGLISH_STOP_WORDS | custom

    def transform(
        self,
        query: str,
        entities_to_strip: Optional[Union[str, List[str]]] = None,
    ) -> str:
        """Strip conversational filler patterns, folder prepositions, comparative noise, and entity names.

        Args:
            query: Raw user query string.
            entities_to_strip: Optional single entity name or list of entity/folder names to remove.

        Returns:
            Cleaned, focused search query string.
        """
        if not query:
            return ""

        transformed = query.strip()

        # Iteratively strip conversational preambles
        for pattern in self._compiled_patterns:
            transformed = pattern.sub("", transformed).strip()

        # If scoped to entities or folders, strip the entity/folder tokens and scoping prepositions
        if entities_to_strip:
            if isinstance(entities_to_strip, str):
                strip_list = [entities_to_strip]
            else:
                strip_list = list(entities_to_strip)

            tokens_to_strip: Set[str] = set()
            for ent in strip_list:
                if ent:
                    tokens_to_strip.add(ent.lower())
                    for chunk in re.findall(r"[a-zA-Z0-9]+", ent):
                        tokens_to_strip.add(chunk.lower())
                        for sub_part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", chunk):
                            tokens_to_strip.add(sub_part.lower())

            sorted_tokens = sorted([t for t in tokens_to_strip if len(t) >= 2], key=len, reverse=True)

            for ent in sorted_tokens:
                transformed = re.sub(
                    rf"\b(?:in|under|from|inside|within|of|for|regarding|about|between|across)\s+(?:the\s+|a\s+|an\s+)?{re.escape(ent)}(?:'s|\Ws)?\b",
                    " ",
                    transformed,
                    flags=re.IGNORECASE,
                ).strip()
                transformed = re.sub(
                    rf"\b(?:of\s+)?{re.escape(ent)}(?:'s|\Ws)?\b",
                    " ",
                    transformed,
                    flags=re.IGNORECASE,
                ).strip()
                if len(ent) >= 3:
                    transformed = re.sub(
                        rf"(?i){re.escape(ent)}",
                        " ",
                        transformed,
                    ).strip()

            # Clean residual prepositions and noise
            transformed = re.sub(r"^(?:in|under|from|inside|within|of|for|regarding|about|between|across|the|a|an)\s+", "", transformed, flags=re.I).strip()
            transformed = re.sub(r"\s+(?:in|under|from|inside|within|of|for|regarding|about|between|across|the|a|an)$", "", transformed, flags=re.I).strip()

        # Strip comparative and relational filler words for pure topic retrieval
        comparative_pattern = (
            r"\b(?:both|in\s+common|have\s+in\s+common|difference\s+between|"
            r"difference|compare|comparison|between|versus|vs|and|that|share|shared)\b"
        )
        if entities_to_strip and isinstance(entities_to_strip, list) and len(entities_to_strip) > 1:
            transformed = re.sub(comparative_pattern, " ", transformed, flags=re.IGNORECASE).strip()

        # Remove trailing question marks, colons, or punctuation noise
        transformed = re.sub(r"[?!.,:;]+$", "", transformed).strip()

        # Collapse whitespace
        transformed = re.sub(r"\s+", " ", transformed).strip()

        # If stripping eliminated the entire query, fall back to cleaned original
        if not transformed:
            transformed = re.sub(r"\s+", " ", query).strip()

        return transformed

    def extract_doc_filter(
        self,
        query: str,
        known_doc_ids: List[str],
        doc_entity_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Union[str, List[str]]], List[str]]:
        """Detect if the query explicitly targets an isolated document or subfolder domain.

        Delegates to ScopeRouter for canonical string-distance normalization, n-gram
        overlap, and folder domain routing.

        Args:
            query: User query string.
            known_doc_ids: List of known document identifiers in the index.
            doc_entity_map: Optional mapping of entity alias keywords to doc_ids.

        Returns:
            Tuple of `(matched_doc_filter, matched_entity_tokens)`:
            - If explicit document/folder domain matched: `('doc_id', ['token'])` or `(['doc1', 'doc2'], ['token'])`
            - If unrouted or global query: `(None, [])`
        """
        return self._router.extract_doc_filter(
            query=query,
            available_doc_ids=known_doc_ids,
            doc_entity_map=doc_entity_map,
        )
