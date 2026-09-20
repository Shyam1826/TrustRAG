r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/rewriter.py
   - Role: Query pre-processing, conversational noise reduction, and entity/domain routing.
   - Purpose: Strips conversational filler phrases and preambles from user queries and
     performs dynamic metadata entity and folder-based domain routing against known document
     corpora and directory hierarchies to enable targeted document-level and subfolder-level
     pre-filtering and focused topic search.

2. INPUT (IP):
   - query (str): Raw user query string (e.g., "What are the contracts in legal/2026?").
   - known_doc_ids (list[str], optional): List of indexed document identifiers in the corpus.
   - doc_entity_map (dict[str, str], optional): Mapping from entity aliases/names to target doc_ids.

3. PROCESS UNDER THE HOOD:
   - Compiles regex patterns matching conversational preambles ("can you tell me", "what is", etc.).
   - Strips matching conversational preambles and normalizes punctuation.
   - Entity & Subfolder Routing:
     * Discovers folder-based domain triggers ("in legal", "under engineering", "from 2026", etc.).
     * Matches folder hierarchy prefixes across known document IDs (e.g. `legal/2026/nda` or `legal__2026__nda`).
     * Parses query tokens against known document IDs, entity aliases (person names, emails), and document stems.
     * Returns matched `(doc_filter, matched_entity_tokens)`.
   - Strips matched entity/folder tokens and possessives from intra-document search query when routed,
     focusing lexical/dense retrieval on pure section topics (e.g. "projects", "contracts", "specifications").
   - Collapses consecutive whitespace characters into a single space.

4. OUTPUT (OP):
   - str: Focused, standalone query string.
   - tuple[Optional[Union[str, list[str]]], list[str]]: `(doc_filter, matched_entity_or_folder_tokens)`.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library module for regex pattern matching and token extraction.
   - typing (Dict, List, Optional, Tuple, Union): Type annotations.
================================================================================
"""

import re
from typing import Dict, List, Optional, Tuple, Union


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
        r"\b(?:in|under|from|inside|within|folder|directory|section|domain)\s+([a-zA-Z0-9_\-\/]+)\b",
        re.IGNORECASE,
    )

    _STOP_TOKENS = {
        "cv", "resume", "doc", "pdf", "file", "spec", "specs", "guide", "the", "of", "and", "in",
        "for", "to", "a", "an", "is", "are", "projects", "academic", "experience", "internship",
        "skills", "what", "who", "which", "how", "with", "from", "about", "tell", "give", "details",
        "this", "that", "all", "any", "some", "his", "her", "their", "candidate", "person", "under",
        "inside", "within", "folder", "directory", "domain"
    }

    def __init__(self) -> None:
        self._compiled_patterns = [
            re.compile(pattern, re.IGNORECASE) for pattern in self._FILLER_PATTERNS
        ]

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
                    # Strip "in/under/from {ent}" as well as plain "{ent}"
                    transformed = re.sub(
                        rf"\b(?:in|under|from|inside|within|of|folder|directory|domain)\s+{re.escape(ent)}\b",
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
        """Detect if the query explicitly names one or more documents/entities or subfolder domains.

        Args:
            query: User query string.
            known_doc_ids: List of known document identifiers in the index.
            doc_entity_map: Optional mapping of entity alias keywords to doc_ids.

        Returns:
            Tuple of `(matched_doc_filter, matched_entity_tokens)`:
            - If multiple documents matched: `(['doc1', 'doc2'], ['ent1', 'ent2'])`
            - If one document matched: `('doc1', ['ent1'])`
            - If no document matched: `(None, [])`
        """
        if not query:
            return None, []

        query_lower = query.lower()
        raw_tokens = re.findall(r"\b\w+\b", query_lower)
        query_tokens = {t for t in raw_tokens if t not in self._STOP_TOKENS and len(t) >= 3}

        matched_docs: Dict[str, str] = {}

        # 1. Check folder-based domain phrasing (e.g. "in legal", "under engineering", "from legal/2026")
        folder_matches = self._FOLDER_TARGET_REGEX.findall(query)
        for folder_target in folder_matches:
            folder_clean = folder_target.strip().lower()
            if folder_clean and folder_clean not in self._STOP_TOKENS and len(folder_clean) >= 3:
                # Match all docs residing in this folder or having this folder prefix
                for doc_id in known_doc_ids:
                    doc_id_lower = doc_id.lower()
                    # Check prefix or folder path containment
                    is_folder_match = (
                        doc_id_lower.startswith(f"{folder_clean}/")
                        or doc_id_lower.startswith(f"{folder_clean}__")
                        or f"/{folder_clean}/" in f"/{doc_id_lower}/"
                        or f"__{folder_clean}__" in f"__{doc_id_lower}__"
                        or folder_clean in [p.lower() for p in re.split(r"[/\\__]+", doc_id) if p]
                    )
                    if is_folder_match:
                        matched_docs[doc_id] = folder_clean

        # 2. Check direct entity map (e.g. 'vaishalee' -> 'Minimalist CV Resume', 'sanjeev' -> 'Sanjeev_M_Resume')
        if doc_entity_map:
            for token in query_tokens:
                if token in doc_entity_map:
                    matched_docs[doc_entity_map[token]] = token

            for alias, target_doc_id in doc_entity_map.items():
                if alias in query_lower and alias not in self._STOP_TOKENS and len(alias) >= 3:
                    matched_docs[target_doc_id] = alias

        # 3. Check known doc_ids (full path, filename stem, or subfolder tokens)
        if known_doc_ids:
            for doc_id in known_doc_ids:
                doc_id_lower = doc_id.lower()

                # Exact substring match (e.g. 'legal/2026/nda' or 'karthik' in query)
                if doc_id_lower in query_lower and doc_id_lower not in self._STOP_TOKENS:
                    matched_docs[doc_id] = doc_id_lower

                # Token-level matching (e.g. 'sanjeev' in 'Sanjeev_M_Resume' or 'nda' in 'legal/2026/nda')
                doc_tokens = [
                    t.lower()
                    for t in re.split(r"[/\\__ -]+", doc_id)
                    if t.lower() not in self._STOP_TOKENS and len(t) >= 3
                ]
                for token in doc_tokens:
                    if token in query_tokens:
                        matched_docs[doc_id] = token

        if not matched_docs:
            return None, []

        if len(matched_docs) == 1:
            doc_id, entity_token = next(iter(matched_docs.items()))
            return doc_id, [entity_token]

        # Return unique list of matched doc_ids and matched tokens
        unique_tokens = list(dict.fromkeys(matched_docs.values()))
        return list(matched_docs.keys()), unique_tokens
