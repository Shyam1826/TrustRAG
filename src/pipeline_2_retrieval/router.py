r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/router.py
   - Role: Canonical string-distance normalization, scope routing, and entity resolution engine.
   - Purpose: Performs robust syntactic, canonical, and n-gram overlap routing across
     indexed document identifiers and folder hierarchies. Solves spacing, casing, and
     punctuation variations between conversational queries and raw document paths
     without domain-specific hardcoding.

2. INPUT (IP):
   - query (str): User natural language search query.
   - available_doc_ids (list[str]): List of indexed document identifiers in the corpus.
   - doc_entity_map (dict[str, str], optional): Optional mapping from entity aliases to doc_ids.

3. PROCESS UNDER THE HOOD:
   - Canonical Token Normalization:
     * Strips non-alphanumeric delimiters (spaces, underscores, hyphens, dots, slashes) and
       lowercases both the query and target identifiers:
       `canonical_id = re.sub(r'[^a-z0-9]', '', doc_id.lower())`
       `canonical_query = re.sub(r'[^a-z0-9]', '', query.lower())`
   - Multi-Tier Scope Matching Logic:
     1. Folder Domain Routing: Detects explicit scoping prepositions (`in <folder>`,
        `under <folder>`, `from <folder>`, `scope: <folder>`) across indexed folder paths.
     2. Exact Canonical Substring Matching: Matches if `canonical_id` or `canonical_stem`
        is an exact substring of `canonical_query`.
     3. N-gram Substring Projection: Evaluates query n-grams (lengths 2 to 5) against
        canonical document identifiers.
     4. Token-Level Jaccard / Overlap: Computes token overlap between significant non-stop
        words in the query and document path components (threshold >= 0.70).
     5. Broad / Across-All Override: Resolves queries with broad multi-document synthesis
        framing to global retrieval (`None`).

4. OUTPUT (OP):
   - tuple[Optional[Union[str, list[str]]], list[str]]:
     * Solitary target: `('doc_id', ['matched_token_1', ...])`
     * Multi-document target: `(['doc1', 'doc2'], ['matched_token_1', ...])`
     * Global unconstrained search: `(None, [])`

5. LIBRARIES & DEPENDENCIES:
   - pathlib.Path: Path stem and component manipulation.
   - re: Regular expressions for tokenization and canonical stripping.
   - typing (Dict, List, Optional, Set, Tuple, Union): Standard type annotations.
   - sklearn.feature_extraction.text.ENGLISH_STOP_WORDS: Standard NLP stop words.
   - src.common.config: System configuration and custom stop words.
================================================================================
"""

from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config


class ScopeRouter:
    """Routes user queries to target document identifiers using canonical normalization and token overlap."""

    # Generic structural and file type tokens to filter from document token sets
    _STRUCTURAL_STOP_TOKENS: Set[str] = {
        "resume", "resumes", "cv", "cvs", "document", "documents",
        "specifications", "specification", "specs", "spec",
        "template", "templates", "file", "files", "candidate", "candidates",
        "report", "reports", "contract", "contracts", "agreement", "agreements",
    }

    _FOLDER_TARGET_REGEX = re.compile(
        r"\b(?:in|under|from|inside|within|folder|directory|section|domain|scope:?)\s+(?:the\s+|a\s+|an\s+)?([a-zA-Z0-9_\-\/]+)\b",
        re.IGNORECASE,
    )

    _GLOBAL_BROAD_PATTERNS = [
        re.compile(r"\b(?:across\s+all\s+documents|across\s+all|in\s+all\s+documents|all\s+documents|all\s+files)\b", re.IGNORECASE),
        re.compile(r"\b(?:compare\s+the\s+candidates|compare\s+all|across\s+the\s+candidates)\b", re.IGNORECASE),
    ]

    _RELATIONAL_MARKERS = [
        re.compile(r"\b(?:where|whose|having|with\s+both|having\s+both)\b", re.IGNORECASE),
        re.compile(r"\b(?:is|equals|=|equal\s+to|is\s+not|!=)\s+['\"]?(?:yes|no|true|false|[a-zA-Z0-9_.-]+)['\"]?", re.IGNORECASE),
        re.compile(r"\b(?:filter(?:ed)?\s+by|rows?\s+where|clauses?\s+where|records?\s+where)\b", re.IGNORECASE),
        re.compile(r"\b(?:count|sum|average|avg|minimum|min|maximum|max|total)\s+(?:of|for|across)?\b", re.IGNORECASE),
        re.compile(r"\b(?:greater\s+than|less\s+than|above|below|at\s+least|at\s+most|[><]=?)\s+[0-9.]+", re.IGNORECASE),
    ]

    def __init__(self) -> None:
        pass

    def is_tabular_query(
        self,
        query: str,
        available_tables: Optional[List[str]] = None,
    ) -> bool:
        """Detect whether a query targets structured tabular documents or relational operations.

        Args:
            query: User search query string.
            available_tables: Optional list of known registered table names.

        Returns:
            True if query references known tabular tables or exhibits relational query markers.
        """
        if not query or not query.strip():
            return False

        has_relational = any(pat.search(query) for pat in self._RELATIONAL_MARKERS)
        if not available_tables:
            return has_relational

        # Canonical substring match against known tables
        query_lower = query.lower()
        canonical_q = re.sub(r"[^a-z0-9]", "", query_lower)
        for t in available_tables:
            t_clean = re.sub(r"[^a-z0-9]", "", t.lower())
            if len(t_clean) >= 3 and t_clean in canonical_q:
                return True

        return has_relational

    def _get_stop_words(self) -> Set[str]:
        """Combine standard English stop words with structural and configured custom stop words."""
        custom = set(config.retrieval.custom_stop_words)
        return ENGLISH_STOP_WORDS | self._STRUCTURAL_STOP_TOKENS | custom

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alphanumeric tokens, handling camelCase boundaries."""
        if not text:
            return []
        parts: List[str] = []
        for chunk in re.findall(r"[a-zA-Z0-9]+", text):
            sub_parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|\d|\W|$)|\d+", chunk)
            if sub_parts:
                parts.extend(sub_parts)
            else:
                parts.append(chunk)
        return [p.lower() for p in parts if p]

    def extract_doc_filter(
        self,
        query: str,
        available_doc_ids: List[str],
        doc_entity_map: Optional[Dict[str, str]] = None,
    ) -> Tuple[Optional[Union[str, List[str]]], List[str]]:
        """Detect if the query explicitly targets specific document identifier(s) or folder domains.

        Args:
            query: User search query string.
            available_doc_ids: List of known document identifiers in the corpus.
            doc_entity_map: Optional mapping of entity keywords to doc_ids.

        Returns:
            Tuple of `(doc_filter, matched_tokens)`:
            - Single document: `('doc_id', ['token_1', ...])`
            - Multiple documents: `(['doc1', 'doc2'], ['token_1', ...])`
            - Global unconstrained search: `(None, [])`
        """
        if not query or not available_doc_ids:
            return None, []

        # Check for broad cross-document queries requiring 100% global search
        for broad_pat in self._GLOBAL_BROAD_PATTERNS:
            if broad_pat.search(query):
                return None, []

        stop_words = self._get_stop_words()
        query_lower = query.lower()
        canonical_query = re.sub(r"[^a-z0-9]", "", query_lower)
        q_tokens = [t for t in self._tokenize(query_lower) if t not in stop_words and len(t) >= 2]

        matched_docs: Dict[str, List[str]] = {}
        folder_matched_docs: Set[str] = set()

        # 1. Check explicit folder domain scoping (e.g. "in legal", "under engineering", "scope: contracts")
        folder_matches = self._FOLDER_TARGET_REGEX.findall(query)
        if folder_matches:
            for folder_target in folder_matches:
                folder_clean = folder_target.strip().lower()
                if folder_clean and folder_clean not in stop_words and len(folder_clean) >= 3:
                    for doc_id in available_doc_ids:
                        doc_id_lower = doc_id.lower()
                        parts = [p.lower() for p in re.split(r"[/\\__]+", doc_id) if p]
                        if folder_clean in parts or doc_id_lower.startswith(f"{folder_clean}/"):
                            matched_docs.setdefault(doc_id, []).append(folder_clean)
                            folder_matched_docs.add(doc_id)

        # 2. Check Entity Alias Mapping if provided
        if doc_entity_map:
            for alias, target_doc in doc_entity_map.items():
                alias_lower = alias.lower()
                if alias_lower and alias_lower not in stop_words and len(alias_lower) >= 3:
                    if re.search(rf"\b{re.escape(alias_lower)}\b", query_lower):
                        matched_docs.setdefault(target_doc, []).append(alias_lower)

        # 3. Canonical String Normalization & Token-Level Jaccard/Overlap Matching
        raw_q_words = self._tokenize(query_lower)
        q_ngrams: List[Tuple[str, str]] = []
        for n in range(2, 6):
            for i in range(len(raw_q_words) - n + 1):
                ngram = " ".join(raw_q_words[i:i + n])
                c_ngram = re.sub(r"[^a-z0-9]", "", ngram)
                if len(c_ngram) >= 4 and c_ngram not in stop_words:
                    q_ngrams.append((ngram, c_ngram))

        for doc_id in available_doc_ids:
            canonical_id = re.sub(r"[^a-z0-9]", "", doc_id.lower())
            stem = Path(doc_id).stem
            canonical_stem = re.sub(r"[^a-z0-9]", "", stem.lower())

            doc_tokens = [t for t in self._tokenize(doc_id) if t not in stop_words and len(t) >= 2]
            stem_tokens = [t for t in self._tokenize(stem) if t not in stop_words and len(t) >= 2]

            is_matched = False
            matched_terms: List[str] = []

            # Exact canonical substring match
            if len(canonical_stem) >= 4 and canonical_stem in canonical_query and canonical_stem not in stop_words:
                is_matched = True
                matched_terms.extend(stem_tokens)
            elif len(canonical_id) >= 4 and canonical_id in canonical_query and canonical_id not in stop_words:
                is_matched = True
                matched_terms.extend(doc_tokens)

            # Query n-gram inside canonical document identifier
            if not is_matched:
                for ngram_text, c_ngram in q_ngrams:
                    if c_ngram in canonical_id or c_ngram in canonical_stem:
                        is_matched = True
                        matched_terms.extend([t for t in self._tokenize(ngram_text) if t not in stop_words])
                        break

            # Token-level overlap & Jaccard similarity
            if not is_matched and stem_tokens:
                common = set(stem_tokens) & set(q_tokens)
                overlap = len(common) / len(stem_tokens)
                if overlap >= 0.70 or (len(common) >= 2 and overlap >= 0.50):
                    is_matched = True
                    matched_terms.extend(list(common))
            elif not is_matched and len(doc_tokens) == 1 and doc_tokens[0] in q_tokens and len(doc_tokens[0]) >= 4:
                is_matched = True
                matched_terms.extend(doc_tokens)

            if is_matched:
                matched_docs.setdefault(doc_id, []).extend(matched_terms)

        if not matched_docs:
            return None, []

        # Specific Directive Scope Resolution:
        # If one document has a specific unique identifier match (e.g. 'linkpluscorp', 'chase') while
        # another document only matched via a shared category/descriptor token (e.g. 'agreement', 'affiliate'),
        # prune the broader shared match so the specific document takes strict precedence.
        # (Exception: If matches were produced by an explicit folder scoping directive like "in legal", retain the folder pool).
        if len(matched_docs) > 1 and len(folder_matched_docs) < len(matched_docs):
            doc_stems = {d: re.sub(r"[^a-z0-9]", "", Path(d).stem.lower()) for d in matched_docs}
            doc_tokens = {
                d: [t for t in self._tokenize(Path(d).stem) if t not in stop_words and len(t) >= 3]
                for d in matched_docs
            }
            doc_has_unique_specific: Dict[str, bool] = {}

            for doc_id, c_stem in doc_stems.items():
                other_stems = [s for d, s in doc_stems.items() if d != doc_id]
                other_tokens = [t for d, tokens in doc_tokens.items() if d != doc_id for t in tokens]

                has_unique_match = False

                # 1. Check query n-grams (len >= 4)
                for _, c_ng in q_ngrams:
                    if len(c_ng) >= 4 and c_ng in c_stem and not any(c_ng in oth for oth in other_stems):
                        has_unique_match = True
                        break

                # 2. Check individual significant tokens (len >= 3)
                if not has_unique_match:
                    for t in doc_tokens[doc_id]:
                        if t in q_tokens and not any(t == oth or t in oth_stem for oth, oth_stem in zip(other_tokens, other_stems)):
                            has_unique_match = True
                            break

                # 3. Exact full stem match
                if not has_unique_match and len(c_stem) >= 4 and c_stem in canonical_query and not any(c_stem in oth for oth in other_stems):
                    has_unique_match = True

                doc_has_unique_specific[doc_id] = has_unique_match

            # If some docs have unique specific identifiers and others only have shared terms, keep only the specific ones
            if any(doc_has_unique_specific.values()) and not all(doc_has_unique_specific.values()):
                filtered_docs = {
                    doc_id: terms for doc_id, terms in matched_docs.items()
                    if doc_has_unique_specific[doc_id]
                }
                if filtered_docs:
                    matched_docs = filtered_docs

        unique_tokens = list(dict.fromkeys([t for t_list in matched_docs.values() for t in t_list if t]))

        # Comparative Intent Handling:
        # If the query is comparative across multiple domains/entities, only lock scope if
        # multiple discrete documents are identified; if only 1 document is detected, fall back to None
        # to ensure global balanced multi-source retrieval across all candidates.
        from src.pipeline_2_retrieval.fusion import is_comparative_query
        if is_comparative_query(query):
            if len(matched_docs) > 1:
                return list(matched_docs.keys()), unique_tokens
            return None, []

        if len(matched_docs) == 1:
            return next(iter(matched_docs.keys())), unique_tokens

        return list(matched_docs.keys()), unique_tokens


def extract_document_scope(
    query: str,
    available_doc_ids: List[str],
    doc_entity_map: Optional[Dict[str, str]] = None,
) -> Tuple[Optional[Union[str, List[str]]], List[str]]:
    """Functional helper for scope routing."""
    router = ScopeRouter()
    return router.extract_doc_filter(query, available_doc_ids, doc_entity_map=doc_entity_map)


def is_tabular_query(
    query: str,
    available_tables: Optional[List[str]] = None,
) -> bool:
    """Functional helper for detecting tabular relational queries."""
    router = ScopeRouter()
    return router.is_tabular_query(query, available_tables=available_tables)
