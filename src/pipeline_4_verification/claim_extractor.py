r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_4_verification/claim_extractor.py
   - Role: Domain-agnostic proposition, comparative clause decomposition, and atomic claim extraction engine.
   - Purpose: Deconstructs synthesized draft responses into complete, standalone
     verifiable propositions across any enterprise domain (legal contracts, financial filings,
     technical specifications, medical literature, resumes). Standardizes Unicode brackets, dashes,
     and whitespace via canonical NFKC normalization, expands multi-citation bracket clusters
     ([Doc-1, Doc-2] -> [Doc-1][Doc-2]), splits compound comparative sentences into independent
     clause propositions, strips ungrounded contrastive tails (e.g. 'rather than...', 'instead of...'),
     filters negative meta-claims (absence observations), detects comparative meta-synthesis
     statements (is_meta=True), anchors short list items with active section headers, and
     associates multi-citation sentences with unified document handles for NLI verification.

2. INPUT (IP):
   - draft (GeneratedDraft): Synthesized draft model from `src/pipeline_3_generation/`.
   - default_section (str, optional): Default section name if no explicit header is in draft.

3. PROCESS UNDER THE HOOD:
   - Step 1: Applies canonical Unicode NFKC normalization (`unicodedata.normalize('NFKC', text)`)
     to automatically convert fullwidth brackets ('【', '】', '［', '］', '〔', '〕'), fullwidth
     digits ('0-9'), and symbols into standard ASCII equivalents.
   - Step 2: Normalizes whitespace (`\u202f`, `\xa0` -> `' '`) and Unicode dashes (`—`, `–`, `\u2011` -> `'-'`).
   - Step 3: Canonicalizes all citation tag variants (`[Doc-1]`, `[1]`, `【Doc-2】`, `［Doc-3］`, `[Doc: 1]`)
     and composite multi-citation brackets (`[Doc-1, Doc-2]`, `[1, 2]`) into canonical `[Doc-X]` format.
   - Step 4: Filters out negative meta-claims via `_is_negative_meta_claim()` asserting absence
     of documentation (e.g., "no details provided", "not specified in documentation").
   - Step 5: Identifies structural section headings, tracking `current_section` and resetting `last_seen_citations`.
   - Step 6: Splits compound comparative lines into independent clause propositions via `_split_compound_line()`
     only when multiple distinct citations are separated across clauses. Preserves abbreviations,
     initials ('V.', 'M.'), and technical decimals ('4.1') from splitting fragmentation.
   - Step 7: Cleans markdown formatting, strips contrastive tails (`rather than...`, `instead of...`),
     and removes comparative preamble framing (`In contrast to...`, `Unlike...`) while preserving
     variable identifiers (`d_model`, `d_k`, `d_v`, `batch_size`) and mathematical equalities (`d_k = 64`).
   - Step 8: Detects pure high-level comparative meta-synthesis connectors and tags `is_meta=True`.
   - Step 9: Context citation inheritance: inherits `last_seen_citations` within active section for
     sub-bullets lacking explicit inline citations.
   - Step 10: Structural proposition framing:
     * Strips colon-separated structural topic labels (`^[A-Za-z0-9\s/&_-]{2,30}:\s*`).
     * Anchors short items/specifications with active section scope.
   - Step 11: Emits `AtomicClaim` models with sequential IDs (`claim_0`, `claim_1`, ...).

4. OUTPUT (OP):
   - list[AtomicClaim]: Complete, multi-citation unified grammatical proposition models.
   - Consumed by: `src/pipeline_4_verification/adjudicator.py`.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library module for regex tokenization and pattern replacement.
   - unicodedata: Standard library module for canonical NFKC Unicode normalization.
   - src.common.schemas (GeneratedDraft, AtomicClaim): Strictly typed data schemas.
================================================================================
"""

import re
import unicodedata
from typing import List, Optional

from src.common.schemas import AtomicClaim, GeneratedDraft


class AtomicClaimExtractor:
    """Extracts domain-agnostic, section-anchored proposition claims from generated drafts."""

    _CITATION_REGEX = re.compile(r"\[(Doc-\d+)\]")
    _NEGATIVE_META_REGEX = re.compile(
        r"\b(?:"
        r"no\s+(?:mention|details?|information|data|evidence|specs?|specifications?|further\s+details?)"
        r"|not\s+(?:provided|mentioned|stated|specified|found|available|disclosed|listed|described|detailed|given)"
        r"|does\s+not\s+(?:state|provide|mention|specify|contain|disclose|list|detail)"
        r"|do\s+not\s+(?:state|provide|mention|specify|contain|disclose|list|detail)"
        r"|without\s+(?:mentioning|providing|specifying|giving)"
        r"|cannot\s+be\s+(?:found|determined|verified)"
        r"|neither\s+provided\s+nor\s+mentioned"
        r")\b",
        re.IGNORECASE,
    )
    _COMPARATIVE_META_REGEX = re.compile(
        r"\b(?:"
        r"in\s+contrast(?:\s+to)?|"
        r"contrasting\s+(?:approaches|paradigms|backgrounds|skillsets|domains|technologies|architectures|profiles|specifications)|"
        r"unlike\s+\[?Doc-\d+\]?|"
        r"compared\s+to\s+\[?Doc-\d+\]?|"
        r"difference\s+between\s+(?:the\s+two|both|these)\s+documents|"
        r"complementary\s+(?:approaches|skillsets|domains|profiles|specifications)|"
        r"presents?\s+a\s+contrast|"
        r"exhibits?\s+a\s+contrast"
        r")\b",
        re.IGNORECASE,
    )
    _CONTRAST_TAIL_REGEX = re.compile(
        r"\s+(?:rather\s+than|instead\s+of|as\s+opposed\s+to)\s+[^.;]+",
        re.IGNORECASE,
    )

    @staticmethod
    def _is_negative_meta_claim(claim_text: str) -> bool:
        """Detect phrases asserting the absence of documentation or unmentioned facts.

        Args:
            claim_text: Proposition or raw claim string.

        Returns:
            True if claim is a negative meta-observation, False otherwise.
        """
        if not claim_text:
            return False
        return bool(AtomicClaimExtractor._NEGATIVE_META_REGEX.search(claim_text))

    @staticmethod
    def _canonicalize_citations(text: str) -> str:
        """Canonicalize all bracket citation variations into standard [Doc-X] format.

        Expands composite multi-citation brackets (e.g. [Doc-1, Doc-2], [1, 2], [Doc-1; Doc-3])
        into separate adjacent tags [Doc-1][Doc-2][Doc-3].

        Args:
            text: Raw input text.

        Returns:
            Text with canonicalized [Doc-X] citation brackets.
        """
        def _expand_bracket(match: re.Match) -> str:
            inner = match.group(1)
            digits = re.findall(r"(?:Doc[\s:_-]*)?(\d+)", inner, flags=re.IGNORECASE)
            if digits:
                return "".join(f"[Doc-{d}]" for d in digits)
            return match.group(0)

        # Expand composite brackets
        text = re.sub(r"\[([^\]\n]+)\]", _expand_bracket, text)
        # Normalize standalone variations
        text = re.sub(r"\[\s*(?:Doc[\s:_-]*)?(\d+)\s*\]", r"[Doc-\1]", text, flags=re.IGNORECASE)
        return text

    def _split_compound_line(self, line: str) -> List[str]:
        """Split a compound line ONLY when distinct citations are distributed across clauses.

        Preserves abbreviations (e.g. 'e.g.', 'i.e.', 'Dr.', 'Sec.', 'Fig.') and single-letter
        initials ('V.', 'M.') across any domain. Never fragments single-citation lines.

        Args:
            line: Raw line string.

        Returns:
            List of sub-proposition strings.
        """
        citations = list(self._CITATION_REGEX.finditer(line))

        # RULE 1: If line has 0 or 1 citations, DO NOT split the sentence.
        # The entire line is a single coherent proposition for that document.
        if len(citations) < 2:
            return [line]

        # Check if citations are adjacent (e.g., "[Doc-1][Doc-2]") -> Joint citation, don't split
        is_adjacent = all(
            line[citations[i].end() : citations[i + 1].start()].strip() == ""
            for i in range(len(citations) - 1)
        )
        if is_adjacent:
            return [line]

        # RULE 2: If distinct citations are separated across clauses, split on contrastive/clause conjunctions
        temp = re.sub(r"(\[Doc-\d+\])[,;]\s*", r"\1 <SPLIT> ", line)
        temp = re.sub(
            r",\s*(?:while|whereas|meanwhile|whilst|in\s+contrast(?:\s+to)?|on\s+the\s+other\s+hand|but|however)\s+",
            " <SPLIT> ",
            temp,
            flags=re.IGNORECASE,
        )
        temp = re.sub(r"\b(?:while|whereas|whilst)\s+", " <SPLIT> ", temp, flags=re.IGNORECASE)
        temp = re.sub(r";\s*", " <SPLIT> ", temp)

        parts = temp.split("<SPLIT>")
        propositions: List[str] = []
        for p in parts:
            p_clean = p.strip()
            # Clean leading contrast connector words
            p_clean = re.sub(
                r"^(?:in\s+contrast(?:\s+to\s+[^,;]+)?|unlike\s+[^,;]+|while|whereas|whilst|meanwhile|on\s+the\s+other\s+hand|but|however)[,;\s]+",
                "",
                p_clean,
                flags=re.IGNORECASE,
            ).strip()
            if p_clean:
                propositions.append(p_clean)

        return propositions if propositions else [line]

    def _clean_claim_text(self, text: str) -> str:
        """Clean markdown formatting and structural markers while preserving variable identifiers and math equalities.

        Args:
            text: Raw claim text line.

        Returns:
            Cleaned proposition text with intact identifiers and equality structures.
        """
        if not text:
            return ""

        # Remove bullet markers, list numbering, and inline citation handles
        cleaned = re.sub(r"^[-*•\d.]+\s*", "", text)
        cleaned = self._CITATION_REGEX.sub("", cleaned).strip()
        cleaned = re.sub(r"\[(?:Doc-)?\d+\]", "", cleaned).strip()

        # Strip ungrounded contrastive tails (e.g., 'rather than web technologies', 'instead of X')
        cleaned = self._CONTRAST_TAIL_REGEX.sub("", cleaned).strip()

        # Remove bold/italic asterisks, backticks, tildes, and header hashes
        cleaned = re.sub(r"\*{1,3}", "", cleaned)
        cleaned = re.sub(r"[`~#]+", "", cleaned)

        # Remove standalone markdown emphasis underscores (__bold__ or _italic_) without touching variable identifiers
        cleaned = re.sub(r"(?<!\w)_{1,2}(\w+)_{1,2}(?!\w)", r"\1", cleaned)
        cleaned = re.sub(r"(?<!\w)_{1,3}|_{1,3}(?!\w)", "", cleaned)

        # Strip leading and trailing punctuation/whitespace (preserving internal math operators like =)
        cleaned = re.sub(r"^[\s,;:.-]+|[\s,;:.-]+$", "", cleaned).strip()
        return cleaned

    def extract_claims(
        self,
        draft: GeneratedDraft,
        default_section: Optional[str] = None,
    ) -> List[AtomicClaim]:
        """Extract complete section-anchored proposition claims from a generated draft response.

        Args:
            draft: GeneratedDraft instance containing raw synthesized text.
            default_section: Optional fallback section name if none is explicitly declared in text.

        Returns:
            List of AtomicClaim objects with associated cited document IDs.
        """
        if not draft or not draft.raw_text or not draft.raw_text.strip():
            return []

        # 1. Apply canonical Unicode NFKC normalization
        text = unicodedata.normalize("NFKC", draft.raw_text)

        # 2. Normalize non-breaking whitespace and Unicode dashes
        text = text.replace("\u202f", " ").replace("\xa0", " ")
        text = text.replace("—", " - ").replace("–", " - ").replace("\u2011", "-")

        # 3. Normalize all Asian and Unicode bracket variants
        text = (
            text.replace("【", "[")
            .replace("】", "]")
            .replace("〔", "[")
            .replace("〕", "]")
            .replace("〖", "[")
            .replace("〗", "]")
            .replace("〘", "[")
            .replace("〙", "]")
        )

        # 4. Canonicalize citation variations and expand composite brackets ([Doc-1, Doc-2] -> [Doc-1][Doc-2])
        text = self._canonicalize_citations(text)

        # Ignore standard insufficient evidence fallback response
        if "does not contain sufficient information" in text.lower():
            return []

        lines = text.split("\n")
        claims: List[AtomicClaim] = []
        claim_counter = 0
        current_section: Optional[str] = default_section
        last_seen_citations: List[str] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Filter negative meta-claims asserting absence of documentation
            if self._is_negative_meta_claim(line_str):
                continue

            # Extract all citations from line
            citation_matches = self._CITATION_REGEX.findall(line_str)
            if citation_matches:
                last_seen_citations = citation_matches

            # Check if line is purely a section header or item title
            is_header = False
            if line_str.startswith("#") or (line_str.startswith("**") and line_str.endswith("**") and not citation_matches) or (line_str.endswith(":") and len(line_str.split()) <= 6 and not citation_matches):
                is_header = True
            elif not line_str.startswith(("-", "*", "•")) and not any(line_str.startswith(f"{c}.") for c in range(10)) and not citation_matches and len(line_str.split()) <= 6:
                is_header = True

            if is_header:
                clean_h = re.sub(r"[*_~`#:]+", "", line_str).strip()
                clean_h = re.sub(r"^(?:section|document|doc)[\s:_-]+", "", clean_h, flags=re.IGNORECASE).strip()
                clean_h = re.split(r"\(", clean_h)[0].strip()
                if clean_h and len(clean_h) >= 2:
                    current_section = clean_h
                last_seen_citations = []
                continue

            # Split compound lines into distinct clause propositions
            sub_propositions = self._split_compound_line(line_str)

            for prop in sub_propositions:
                prop_citations = self._CITATION_REGEX.findall(prop)
                # Inherit from line or last_seen_citations if this sub-proposition has no citations
                effective_citations = prop_citations or citation_matches or list(last_seen_citations)

                # Clean proposition of bullets, inline citations, contrast tails, markdown formatting
                clean_claim = self._clean_claim_text(prop)
                if not clean_claim or self._is_negative_meta_claim(clean_claim):
                    continue

                tokens = clean_claim.split()
                if not tokens:
                    continue

                # Detect pure comparative meta-synthesis connectors
                is_pure_meta = bool(self._COMPARATIVE_META_REGEX.search(prop)) and len(tokens) <= 8 and not effective_citations

                sec_label = current_section

                # Universal domain-agnostic proposition framing
                if len(tokens) < 3:
                    # 1-2 word items or specifications
                    if sec_label and sec_label.lower() != "general":
                        anchored_claim = f"Under {sec_label}, the documented specification or item is: {clean_claim}."
                    else:
                        anchored_claim = f"The document specifies: {clean_claim}."
                else:
                    # 3+ tokens: generalized colon-separated prefix stripping
                    colon_prefix_match = re.match(r"^[A-Za-z0-9\s/&_-]{2,30}:\s*", clean_claim)
                    if colon_prefix_match:
                        body = clean_claim[colon_prefix_match.end():].strip()
                        if sec_label and sec_label.lower() != "general":
                            anchored_claim = f"Under {sec_label}, the documented items include: {body}"
                        else:
                            anchored_claim = f"The document specifies: {body}"
                    elif sec_label and sec_label.lower() != "general":
                        anchored_claim = f"Under {sec_label}, the document specifies: {clean_claim}"
                    else:
                        # Generic short noun phrase or statement framing
                        verbs = {
                            "is", "are", "was", "were", "has", "have", "had", "shall", "will", "may",
                            "can", "could", "provides", "contains", "specifies", "features", "includes",
                            "details", "describes", "focuses", "specializes", "developed", "built",
                            "holds", "completed", "architected", "serves", "works", "uses", "utilizes",
                        }
                        if len(tokens) <= 10 and not any(t.lower() in verbs for t in tokens):
                            anchored_claim = f"The document specifies: {clean_claim}"
                        else:
                            anchored_claim = clean_claim

                if not anchored_claim.endswith("."):
                    anchored_claim += "."

                # Multi-Citation Association: Emit AtomicClaim containing all cited handles
                claim_id = f"claim_{claim_counter}"
                cited_id = ", ".join(effective_citations) if len(effective_citations) > 1 else (effective_citations[0] if effective_citations else None)
                claims.append(
                    AtomicClaim(
                        claim_id=claim_id,
                        claim_text=anchored_claim,
                        cited_doc_id=cited_id,
                        cited_doc_ids=effective_citations,
                        is_meta=is_pure_meta,
                    )
                )
                claim_counter += 1

        return claims