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
     * Evaluates explicit folder domain scoping directives (`scope: <folder>`, `in <folder>`,
       `under <folder>`, `within <folder>`, `from <folder>`).
     * Matches target against dynamic folder hierarchies of indexed document identifiers.
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
================================================================================
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config


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

    _FOLDER_TARGET_REGEX = re.compile(
        r"\b(?:in|under|from|inside|within|folder|directory|section|domain|scope:?)\s+([a-zA-Z0-9_\-\/]+)\b",
        re.IGNORECASE,
    )

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._FILLER_PATTERNS
        ]

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

            for ent in strip_list:
                if ent:
                    transformed = re.sub(
                        rf"\b(?:in|under|from|inside|within|of|folder|directory|domain|scope:?)\s+{re.escape(ent)}\b",
                        "",
                        transformed,
                        flags=re.IGNORECASE,
                    ).strip()
                    transformed = re.sub(
                        rf"\b(?:of\s+)?{re.escape(ent)}(?:'s|\Ws)?\b",
                        "",
                        transformed,
                        flags=re.IGNORECASE,
                    ).strip()

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

        Disables eager single-keyword file locking. If no explicit folder directive
        or explicit scope prefix is present, defaults strictly to (None, []) so retrieval
        executes globally based 100% on semantic vector and BM25 relevance.

        Args:
            query: User query string.
            known_doc_ids: List of known document identifiers in the index.
            doc_entity_map: Optional mapping of entity alias keywords to doc_ids.

        Returns:
            Tuple of `(matched_doc_filter, matched_entity_tokens)`:
            - If explicit folder domain matched: `(['doc1', 'doc2'], ['folder'])` or `('doc1', ['folder'])`
            - If unrouted or global query: `(None, [])`
        """
        if not query:
            return None, []

        stop_words = self._get_stop_words()
        query_lower = query.lower()
        raw_tokens = set(re.findall(r"\b\w+\b", query_lower))

        # Check for explicit folder-based domain scoping (e.g. "in legal", "under engineering", "from legal/2026", "scope: legal")
        folder_matches = self._FOLDER_TARGET_REGEX.findall(query)
        matched_docs: Dict[str, str] = {}

        if folder_matches:
            for folder_target in folder_matches:
                folder_clean = folder_target.strip().lower()
                if folder_clean and folder_clean not in stop_words and len(folder_clean) >= 3:
                    # Match docs residing in this folder or having this folder prefix
                    for doc_id in known_doc_ids:
                        doc_id_lower = doc_id.lower()
                        is_folder_match = (
                            doc_id_lower.startswith(f"{folder_clean}/")
                            or doc_id_lower.startswith(f"{folder_clean}__")
                            or f"/{folder_clean}/" in f"/{doc_id_lower}/"
                            or f"__{folder_clean}__" in f"__{doc_id_lower}__"
                            or folder_clean in [p.lower() for p in re.split(r"[/\\__]+", doc_id) if p]
                        )
                        if is_folder_match:
                            matched_docs[doc_id] = folder_clean

        # If no explicit folder match, default strictly to None for 100% global semantic search
        if not matched_docs:
            return None, []

        if len(matched_docs) == 1:
            doc_id, entity_token = next(iter(matched_docs.items()))
            return doc_id, [entity_token]

        unique_tokens = list(dict.fromkeys(matched_docs.values()))
        return list(matched_docs.keys()), unique_tokens
