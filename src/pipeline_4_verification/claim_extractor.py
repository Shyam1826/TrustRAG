r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_4_verification/claim_extractor.py
   - Role: Domain-agnostic proposition and atomic claim extraction engine with section context.
   - Purpose: Deconstructs synthesized draft responses into complete, standalone
     verifiable propositions across any domain (resumes, technical specifications, legal contracts,
     financial filings, API documentation). Standardizes Unicode brackets, dashes, and whitespace,
     anchors short list items and specifications with active section headers, and associates
     each claim with its cited document handle for NLI auditing.

2. INPUT (IP):
   - draft (GeneratedDraft): Synthesized draft model from `src/pipeline_3_generation/`.
   - default_section (str, optional): Default section name if no explicit header is in draft.

3. PROCESS UNDER THE HOOD:
   - Step 1: Normalizes non-breaking whitespace (`\u202f`, `\xa0` -> `' '`), Unicode dashes
     (`—`, `–`, `\u2011` -> `'-'`), and Unicode brackets (`【` -> `[`, `】` -> `]`).
   - Step 2: Identifies structural section headings, tracking `current_section`.
   - Step 3: Skips heading and fallback lines so they are not treated as atomic assertions.
   - Step 4: Discovers inline citation tags (e.g., `[Doc-1]`).
   - Step 5: For short specifications or 1-2 word list items (e.g. "AutoCad", "15W", "Python"),
     anchors with section scope (`Under {section}, the documented specification or item is: {item}.`).
   - Step 6: For technology, feature, or specification lists, anchors with domain-neutral prefixes.
   - Step 7: Emits `AtomicClaim` models with sequential IDs (`claim_0`, `claim_1`, ...).

4. OUTPUT (OP):
   - list[AtomicClaim]: Complete, section-anchored grammatical proposition models.
   - Consumed by: `src/pipeline_4_verification/adjudicator.py`.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library module for regex tokenization and pattern replacement.
   - src.common.schemas (GeneratedDraft, AtomicClaim): Strictly typed data schemas.
================================================================================
"""

import re
from typing import List, Optional

from src.common.schemas import AtomicClaim, GeneratedDraft


class AtomicClaimExtractor:
    """Extracts domain-agnostic, section-anchored proposition claims from generated drafts."""

    _CITATION_REGEX = re.compile(r"\[(Doc-\d+)\]")
    _ACTION_VERBS = {
        "built", "developed", "implemented", "integrated", "utilized", "used", "optimized",
        "enhanced", "designed", "analyzed", "created", "worked", "completed", "deployed",
        "engineered", "architected", "managed", "led", "participated", "volunteered", "evaluated",
        "contains", "includes", "requires", "provides", "defines", "establishes", "reported",
        "increased", "decreased", "achieved", "delivered", "supports", "ensures", "maintains",
        "specifies", "features"
    }

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

        # 1. Normalize non-breaking spaces, dashes, and Unicode brackets
        text = draft.raw_text.replace("\u202f", " ").replace("\xa0", " ")
        text = text.replace("—", " - ").replace("–", " - ").replace("\u2011", "-")
        text = text.replace("【", "[").replace("】", "]")
        text = re.sub(r"\[\s*Doc[\s:_-]*(\d+)\s*\]", r"[Doc-\1]", text, flags=re.IGNORECASE)

        # Ignore standard insufficient evidence fallback response
        if "does not contain sufficient information" in text.lower():
            return []

        lines = text.split("\n")
        claims: List[AtomicClaim] = []
        claim_counter = 0
        current_section: Optional[str] = default_section

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Extract citations from line
            citation_matches = self._CITATION_REGEX.findall(line_str)
            cited_doc_id: Optional[str] = citation_matches[0] if citation_matches else None

            # Check if line is purely a section header or item title
            is_header = False
            if line_str.startswith("#") or (line_str.startswith("**") and line_str.endswith("**") and not cited_doc_id) or (line_str.endswith(":") and len(line_str.split()) <= 6 and not cited_doc_id):
                is_header = True
            elif not line_str.startswith(("-", "*", "•")) and not any(line_str.startswith(f"{c}.") for c in range(10)) and not cited_doc_id and len(line_str.split()) <= 6:
                is_header = True

            if is_header:
                clean_h = re.sub(r"[*_~`#:]+", "", line_str).strip()
                clean_h = re.sub(r"^(?:section|document|doc)[\s:_-]+", "", clean_h, flags=re.IGNORECASE).strip()
                clean_h = re.split(r"\(", clean_h)[0].strip()
                if clean_h and len(clean_h) >= 2:
                    current_section = clean_h
                continue

            # Clean line of bullets, inline citations, markdown formatting
            clean_line = re.sub(r"^[-*•\d.]+\s*", "", line_str)
            clean_line = self._CITATION_REGEX.sub("", clean_line).strip()
            clean_line = re.sub(r"[*_~`#]+", "", clean_line).strip()

            clean_claim = re.sub(r"^[\s,;:.-]+|[\s,;:.-]+$", "", clean_line).strip()
            tokens = clean_claim.split()
            if not tokens:
                continue

            sec_label = current_section

            # Universal proposition framing
            if len(tokens) < 3:
                # 1-2 word items or specifications (e.g., "AutoCad", "15W", "Python", "GDPR")
                if sec_label and sec_label.lower() != "general":
                    anchored_claim = f"Under {sec_label}, the documented specification or item is: {clean_claim}."
                else:
                    anchored_claim = f"The document specifies: {clean_claim}."
            else:
                # 3+ tokens
                if clean_claim.lower().startswith(("tech stack:", "technologies:", "tools:", "features:", "specifications:", "specs:", "skills:")):
                    body = re.sub(r"^(?:tech stack|technologies|tools|features|specifications|specs|skills):\s*", "", clean_claim, flags=re.IGNORECASE).strip()
                    if sec_label and sec_label.lower() != "general":
                        anchored_claim = f"Under {sec_label}, the documented items include: {body}"
                    else:
                        anchored_claim = f"The document specifies: {body}"
                elif sec_label and sec_label.lower() != "general":
                    anchored_claim = f"Under {sec_label}, the document specifies: {clean_claim}"
                else:
                    # If it's a short title or noun phrase without an explicit predicate
                    has_explicit_verb = any(t.lower().rstrip(",;:") in self._ACTION_VERBS or t.lower() in {"is", "are", "was", "were", "operates", "grants", "contains", "provides"} for t in tokens)
                    if not has_explicit_verb and len(tokens) <= 10:
                        anchored_claim = f"The document specifies: {clean_claim}"
                    else:
                        anchored_claim = clean_claim


            if not anchored_claim.endswith("."):
                anchored_claim += "."

            claim_id = f"claim_{claim_counter}"
            claims.append(
                AtomicClaim(
                    claim_id=claim_id,
                    claim_text=anchored_claim,
                    cited_doc_id=cited_doc_id,
                )
            )
            claim_counter += 1

        return claims
