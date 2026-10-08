r"""
===============================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_4_verification/adjudicator.py
   - Role: Trust scoring, safety gating, and audit report adjudication engine.
   - Purpose: Aligns atomic claims to focused section-breadcrumbed parent premises, performs
     dynamic citation-to-premise routing mapping `[Doc-k]` tags directly to `contexts[k-1]`,
     normalizes meta-document and prompt scaffolding from hypotheses, handles multi-citation
     unification across referenced documents ('\n\n---\n\n'), executes NLI batch predictions,
     evaluates threshold-based verdicts (ENTAILED, CONTRADICTION, NEUTRAL), calculates the
     global faithfulness score, guards against zero-claim audit bypasses, and enforces
     automated downstream actions (PASS, TRIGGER_REWRITE, WARN, UNVERIFIED_OPEN_WORLD).

2. INPUT (IP):
   - claims (list[AtomicClaim]): Atomic propositions from `src/pipeline_4_verification/claim_extractor.py`.
   - contexts / context_map (list or dict): Retrieved context candidates or mapping of document handles.
   - nli_verifier (DebertaNLIVerifier): DeBERTa sequence classifier from `src/pipeline_4_verification/nli_model.py`.
   - draft_text (str, optional): Full synthesized draft text from `src/pipeline_3_generation/`.

3. PROCESS UNDER THE HOOD:
   - Dynamic Citation-to-Premise Routing:
     * Parses citation tags dynamically from each claim string (`r'\[Doc-(\d+)\]'`).
     * Maps extracted citation integer `k` to the specific retrieved context (`contexts[k - 1]`).
     * If no citation tag is detected, falls back gracefully to evaluating against the top-ranked
       context (`contexts[0]`) or concatenated fallback corpus.
     * Strips prompt wrappers, bullet markers, and carrier framing from the hypothesis prior to NLI.
   - Zero-Claim Safety Guard:
     * If `len(claims) == 0`: checks whether `draft_text` contains substantive content or bullet points
       (excluding standard fallback phrases like "does not contain sufficient information").
     * If substantive text exists without verifiable claims/citations: sets `faithfulness_score = 0.00`
       and `action = "WARN"`.
     * If legitimate fallback response: sets `faithfulness_score = 1.00` and `action = "PASS"`.
   - Meta-Claim Handling:
     * For claims with `is_meta=True`, assigns `verdict = "ENTAILED"` with `confidence = 1.0`
       and `cited_premise = "Comparative Meta-Analytical Synthesis"`.
   - Batch Inference & Adjudication:
     * Queries `nli_verifier.predict_batch()` for cleaned (claim, premise_window) pairs.
     * Evaluates $P(\text{contradiction}) \ge \tau_{\text{contradiction}}$ $\implies$ CONTRADICTION,
       $P(\text{entailment}) \ge \tau_{\text{entailment}}$ $\implies$ ENTAILED, else NEUTRAL.
   - Safety Action Gating:
     * Sets `faithfulness_score` as the ratio of entailed claims.
     * Sets `action`: "TRIGGER_REWRITE" if contradiction exists, "PASS" if score $\ge 0.80$, else "WARN".

4. OUTPUT (OP):
   - TrustAuditReport: Comprehensive audit report containing:
     * draft_text (str), faithfulness_score (float), has_contradiction (bool), action (str), audits (list[ClaimAudit]).
   - Consumed by: `src/main.py` and downstream self-correction or response dispatchers.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library regex tokenization and sentence boundary detection.
   - sklearn.feature_extraction.text (ENGLISH_STOP_WORDS): Standard NLP stop words.
   - src.common.config: Centralized decision thresholds and verification settings.
   - src.common.schemas (AtomicClaim, ClaimAudit, GroundingMode, TrustAuditReport): Strict data models.
   - src.pipeline_4_verification.nli_model.DebertaNLIVerifier: NLI model interface.
================================================================================
"""

from collections import defaultdict
import re
from typing import Any, Dict, List, Literal, Optional, Set, Tuple
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config
from src.common.schemas import AtomicClaim, ClaimAudit, TrustAuditReport


class AuditAdjudicator:
    """Evaluates NLI scores across claims, computes faithfulness, and determines safety actions."""

    _COMPARATIVE_SPLIT_REGEX = re.compile(
        r",\s*(?:while|whereas|meanwhile|whilst|in\s+contrast(?:\s+to)?|on\s+the\s+other\s+hand|as\s+opposed\s+to|distinct\s+from|which\s+is\s+distinct\s+from)\s+|\s+(?:while|whereas|whilst)\s+|;\s*",
        re.IGNORECASE,
    )

    def __init__(
        self,
        tau_entailment: Optional[float] = None,
        tau_contradiction: Optional[float] = None,
        premise_window_size: Optional[int] = None,
        nli_verifier: Optional[Any] = None,
    ) -> None:
        """Initialize adjudicator with decision thresholds.

        Args:
            tau_entailment: Minimum probability required for ENTAILED verdict (default from config).
            tau_contradiction: Minimum probability required for CONTRADICTION verdict (default from config).
            premise_window_size: Maximum character length for extracted premise window (default from config).
            nli_verifier: Optional DebertaNLIVerifier instance.
        """
        settings = config.verification
        self.tau_entailment = tau_entailment if tau_entailment is not None else settings.tau_entailment
        self.tau_contradiction = tau_contradiction if tau_contradiction is not None else settings.tau_contradiction
        self.premise_window_size = premise_window_size if premise_window_size is not None else getattr(settings, "premise_window_size", 1200)
        self.nli_verifier = nli_verifier

    def _get_stop_words(self) -> Set[str]:
        """Combine standard English stop words with configured custom stop words."""
        custom = set(config.retrieval.custom_stop_words)
        return ENGLISH_STOP_WORDS | custom

    def _clean_hypothesis_for_nli(self, claim_text: str) -> str:
        """Iteratively strip nested meta-document scaffolding, Doc-X carriers, and section anchors from hypothesis.

        Executes an iterative normalization loop until reaching a fixed point (no further changes) to peel off
        stacked framing (e.g., 'Under <Section>, the document specifies: The <DocType> states that <Assertion>')
        and produce clean, grounded affirmative propositions for DeBERTa-v3 NLI verification.

        Args:
            claim_text: Raw atomic claim text.

        Returns:
            Cleaned affirmative proposition text.
        """
        if not claim_text:
            return ""

        cleaned = claim_text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"').strip()

        # Strip bullet points, numbering, and prompt wrappers
        cleaned = re.sub(r"^\s*[-*•\u2022]\s*", "", cleaned)
        cleaned = re.sub(r"^\s*\d+[\.\)]\s*", "", cleaned)
        cleaned = re.sub(
            r"^(?:Claim\s*\d*|Assertion\s*\d*|Proposition\s*\d*|Output\s*Draft|Draft\s*Response|Answer)[:\s]+",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        doc_types = r"(?:resume|cv|specifications?|specs?|document|paper|report|contract|agreement|template|candidate|overview|profile|guidelines?|manual|datasheet|workbook|spreadsheet|sheet)"
        verbs = r"(?:states?\s+that|states?|notes?\s+that|notes?|describes?|specifies?\s+that|specifies?|features?|lists?|reports?\s+that|reports?|highlights?|presents?|outlines?|defines?|mentions?|identifies?|focuses\s+on|focuses)"

        for _ in range(10):
            prev = cleaned

            # 1. Section Scaffolding & Compound Nested Framing
            cleaned = re.sub(
                r"^Under\s+(?:Section\s+[A-Za-z0-9._-]+|[^,;:]+)[,:]\s*(?:the\s+document\s+specifies:?\s*|the\s+documented\s+[^:]+:?\s*)?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^Under\s+[^,:]+[,:]\s*(?:the\s+(?:candidate|document|specification|item|technologies)\s+(?:completed|specifies|states|utilizes|features|include|utilized\s+include|documented\s+specification\s+or\s+item\s+is|documented\s+items\s+include):?\s*)?",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            # Compound nested "Under <Entity/Document/Section>," framing
            cleaned = re.sub(
                r"^Under\s+(?:the\s+)?[A-Za-z0-9_.\s/'-]+?,\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^Pursuant\s+to\s+(?:the\s+)?[A-Za-z0-9_.\s/'-]+?[,:]\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^In\s+accordance\s+with\s+(?:the\s+)?[A-Za-z0-9_.\s/'-]+?[,:]\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^The\s+document\s+specifies:\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^The\s+(?:technologies\s+utilized\s+include|documented\s+(?:items?\s+include|specification\s+or\s+item\s+is)):\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            # 2. Document Carrier Scaffolding with Doc-X handles
            cleaned = re.sub(
                r"^According\s+to\s+(?:\[Doc-\d+\]|(?:the\s+)?[A-Za-z0-9_.'/\s\[\]-]+?)[,:]\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^(?:As\s+(?:stated|noted|specified|detailed)\s+in|Based\s+on|Per)\s+(?:\[Doc-\d+\]|(?:the\s+)?[A-Za-z0-9_.'/\s\[\]-]+?)[,:]\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^The\s+candidate\s+in\s+(?:Doc-\d+(?:\s*(?:and|,)\s*)*)+\s+(?:focuses\s+on|lists?|describes?|details?|highlights?):?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^The\s+(?:engineering\s+specs?|specifications?|documents?|reports?|proposals?|contracts?)\s+in\s+(?:Doc-\d+(?:\s*(?:and|,)\s*)*)+\s+(?:describes?|states?\s+that|notes?\s+that|details?|specifies?\s+that|specifies?|highlights?|features?):?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )
            cleaned = re.sub(
                r"^In\s+(?:Doc-\d+(?:\s*(?:and|,)\s*)*)+[,\s]+(?:the\s+candidate|the\s+document|the\s+specification)\s+(?:focuses\s+on|lists?|describes?|states?\s+that|notes?\s+that|details?|features?|specifies?):?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            # 3. Universal Document Carrier Scaffolding: ^The (<desc> )?<doc_type> (for/of ...) <carrier_verbs>
            cleaned = re.sub(
                rf"^(?:The\s+)?(?:[A-Za-z0-9_.'/\s-]+?\s+)?{doc_types}(?:\s+(?:for|of|titled|regarding|named)\s+[^:;,\n]+?)?\s+{verbs}\s*:?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            # 4. Source/Entity with doc_type or possessive: ^<Entity> ('s <doc_type>? | <doc_type>) <carrier_verbs>
            cleaned = re.sub(
                rf"^(?:The\s+)?[A-Za-z0-9_.'/\s-]+?\s+(?:'s\s+(?:{doc_types}\s+)?|{doc_types}\s+){verbs}\s*:?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            # 5. Source/Entity with explicit attribution verbs
            cleaned = re.sub(
                r"^(?:The\s+)?[A-Za-z0-9_.'/\s-]+?\s+(?:states?\s+that|notes?\s+that|specifies?\s+that|reports?\s+that|focuses\s+on)\s*:?\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

            # 6. Inline and trailing citation stripping from hypothesis
            cleaned = re.sub(r"^\[Doc-\d+\]\s*:?\s*", "", cleaned)
            cleaned = re.sub(r"\s*\[Doc-\d+\]", "", cleaned)

            # Strip leading/trailing punctuation and whitespace (preserving trailing period)
            cleaned = re.sub(r"^[\s,;:.-]+|[\s,;:-]+$", "", cleaned).strip()

            if cleaned == prev:
                break

        # Capitalize first character if lowercase
        if cleaned and cleaned[0].islower():
            cleaned = cleaned[0].upper() + cleaned[1:]

        return cleaned

    def _resolve_context_text(self, ctx_obj: Any) -> str:
        """Extract full parent context from string, schema model, or dictionary.

        Prioritizes parent_text over truncated child text when available to ensure
        trailing list items, skills, formulas, and clauses are not cut off.
        Prepends document identity metadata (doc_id, section_name) if available.
        """
        if ctx_obj is None:
            return ""

        body = ""
        doc_id = None
        section_name = None

        if isinstance(ctx_obj, str):
            body = ctx_obj
        elif isinstance(ctx_obj, dict):
            body = ctx_obj.get("parent_text") or ctx_obj.get("text") or ctx_obj.get("content") or ""
            doc_id = ctx_obj.get("doc_id")
            section_name = ctx_obj.get("section_name")
        else:
            if hasattr(ctx_obj, "parent_text") and ctx_obj.parent_text:
                body = ctx_obj.parent_text
            elif hasattr(ctx_obj, "text") and ctx_obj.text:
                body = ctx_obj.text
            else:
                body = str(ctx_obj)
            doc_id = getattr(ctx_obj, "doc_id", None)
            section_name = getattr(ctx_obj, "section_name", None)

        if doc_id and not (body.startswith("[Document:") or body.startswith("Document:")):
            sec_name = section_name or "General"
            header = f"[Document: {doc_id} | Section: {sec_name}]\n"
            body = f"{header}{body}"

        return body

    def _extract_premise_window(self, claim_text: str, context: Any) -> str:
        """Extract the section breadcrumb and most relevant block from context for the given claim.

        Expands to clean sentence and paragraph boundaries up to premise_window_size (max ~350 words),
        using symbolic keyword weighting for exact quantitative parameters, variables, and formulas
        within DeBERTa's 512-token limit.

        Args:
            claim_text: Proposition text of the atomic claim.
            context: Passage string or candidate object associated with the cited document.

        Returns:
            Focused premise string combining breadcrumb and relevant sentence-bounded block.
        """
        raw_text = self._resolve_context_text(context)
        if not raw_text or not raw_text.strip():
            return ""

        # Normalize unicode dashes, spaces, and brackets
        norm_context = raw_text.replace("\u202f", " ").replace("\xa0", " ")
        norm_context = norm_context.replace("—", " - ").replace("–", " - ").replace("\u2011", "-")

        lines = [l.strip() for l in norm_context.split("\n") if l.strip()]
        breadcrumb = lines[0] if lines and (lines[0].startswith("[Document:") or lines[0].startswith("Document:")) else ""

        # For parent contexts within premise_window_size, return the entire context directly
        if len(norm_context) <= self.premise_window_size:
            return norm_context

        # Extract core content keywords from claim using cleaned hypothesis
        clean_claim = self._clean_hypothesis_for_nli(claim_text)
        clean_claim = clean_claim.replace("—", " - ").replace("–", " - ").replace("\u2011", "-").strip().rstrip(".")

        # Tokenize claim words preserving variable identifiers (e.g. d_model, d_k, d_v, _id), numbers, and symbols
        raw_claim_words = re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", clean_claim.lower())
        stop_words = self._get_stop_words()
        claim_words = set(w for w in raw_claim_words if w not in stop_words)

        body_lines = lines[1:] if breadcrumb else lines
        body_text = "\n".join(body_lines)

        # Split body text into natural sentence / bullet / paragraph segments
        raw_segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body_text) if s.strip()]
        if not raw_segments:
            return norm_context

        # Score each segment based on symbolic-weighted keyword overlap density
        best_idx = 0
        best_score = -1.0
        matching_indices: List[int] = []

        for idx, seg in enumerate(raw_segments):
            seg_tokens = re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", seg.lower())
            seg_words = set(w for w in seg_tokens if w not in stop_words)
            if not seg_words:
                continue

            overlap_words = claim_words & seg_words
            if not overlap_words:
                continue

            # Symbolic weighting: higher weight for numbers, formulas, and variable identifiers
            overlap_score = 0.0
            for w in overlap_words:
                if re.search(r"\d|_|[=+#-]", w):
                    overlap_score += 2.5
                else:
                    overlap_score += 1.0

            matching_indices.append(idx)
            score = overlap_score / (len(seg_words) ** 0.5)
            if score > best_score:
                best_score = score
                best_idx = idx

        max_capacity = self.premise_window_size - (len(breadcrumb) + 2 if breadcrumb else 0)

        if best_score <= 0 or not matching_indices:
            # Fallback: take beginning up to premise_window_size along sentence boundary
            accumulated: List[str] = []
            curr_len = 0
            for seg in raw_segments:
                if curr_len + len(seg) + 1 > max_capacity:
                    break
                accumulated.append(seg)
                curr_len += len(seg) + 1
            selected = " ".join(accumulated) if accumulated else raw_segments[0][:max_capacity]
            if breadcrumb and not (selected.startswith("[Document:") or selected.startswith("Document:")):
                return f"{breadcrumb}\n{selected}"
            return selected

        # If span from min matching index to max matching index fits within max_capacity, start from it
        min_m = min(matching_indices)
        max_m = max(matching_indices)
        initial_span = " ".join(raw_segments[min_m : max_m + 1])
        if len(initial_span) <= max_capacity:
            left = min_m
            right = max_m
        else:
            left = best_idx
            right = best_idx

        # Expand outward around boundaries up to premise_window_size
        while True:
            expanded = False
            # Try expanding forward to include next sentence/bullet
            if right + 1 < len(raw_segments):
                test_text = " ".join(raw_segments[left : right + 2])
                if len(test_text) <= max_capacity:
                    right += 1
                    expanded = True
            # Try expanding backward to include previous sentence/bullet
            if left - 1 >= 0:
                test_text = " ".join(raw_segments[left - 1 : right + 1])
                if len(test_text) <= max_capacity:
                    left -= 1
                    expanded = True
            if not expanded:
                break

        expanded_window = " ".join(raw_segments[left : right + 1])
        if breadcrumb and not (expanded_window.startswith("[Document:") or expanded_window.startswith("Document:")):
            return f"{breadcrumb}\n{expanded_window}"
        return expanded_window

    def adjudicate(
        self,
        claims: List[AtomicClaim],
        context_map: Optional[Dict[str, Any]] = None,
        nli_verifier: Optional[Any] = None,
        draft_text: str = "",
        batch_size: int = 32,
        contexts: Optional[List[Any]] = None,
    ) -> TrustAuditReport:
        """Audit atomic claims against cited contexts and produce a TrustAuditReport.

        Args:
            claims: List of AtomicClaim instances to verify.
            context_map: Mapping from document tags (e.g. 'Doc-1') to passages or candidates.
            nli_verifier: Optional DebertaNLIVerifier instance.
            draft_text: Optional full synthesized draft text.
            batch_size: Batch size for model inference.
            contexts: Optional sequence of candidate objects for direct 1-based citation index mapping.

        Returns:
            TrustAuditReport containing per-claim audits, faithfulness score, and action verdict.
        """
        if context_map is None and contexts is not None:
            context_map = {f"Doc-{i}": c for i, c in enumerate(contexts, start=1)}
        elif context_map is None:
            context_map = {}

        context_list = list(contexts) if contexts is not None else list(context_map.values())

        verifier = nli_verifier if nli_verifier is not None else getattr(self, "nli_verifier", None)
        if verifier is None:
            from src.pipeline_4_verification.nli_model import DebertaNLIVerifier
            verifier = DebertaNLIVerifier()

        if not claims:
            # Guard against zero-claim audit bypass: check if draft contains substantive assertions
            cleaned_draft = re.sub(r"[*_~`#\-•\s]+", " ", draft_text or "").strip().lower()
            is_fallback = (
                "does not contain sufficient information" in cleaned_draft
                or "insufficient information" in cleaned_draft
                or not cleaned_draft
            )
            if is_fallback or len(cleaned_draft.split()) < 3:
                return TrustAuditReport(
                    draft_text=draft_text,
                    faithfulness_score=1.0,
                    has_contradiction=False,
                    action="PASS",
                    audits=[],
                )
            else:
                # Substantive text generated without extracted/verifiable claims or citations
                return TrustAuditReport(
                    draft_text=draft_text,
                    faithfulness_score=0.0,
                    has_contradiction=False,
                    action="WARN",
                    audits=[],
                )

        fallback_corpus = "\n\n---\n\n".join(self._resolve_context_text(v) for v in context_list) if context_list else ""

        audits: List[ClaimAudit] = []
        batch_hypotheses: List[str] = []
        batch_premises: List[str] = []
        # Maps batch query index -> (claim_index, sub_clause_index, total_sub_clauses, doc_handle, premise_text)
        batch_meta: List[Tuple[int, int, int, str, str]] = []

        for idx, claim in enumerate(claims):
            if getattr(claim, "is_meta", False):
                # Meta-Analytical claim (comparative connector / high-level synthesis)
                meta_audit = ClaimAudit(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    cited_premise="Comparative Meta-Analytical Synthesis",
                    probabilities={"entailment": 1.0, "contradiction": 0.0, "neutral": 0.0},
                    verdict="ENTAILED",
                    confidence=1.0,
                )
                audits.append(meta_audit)
                continue

            # 1. Parse citation tags dynamically from claim string or explicit claim attributes
            cit_matches = re.findall(r"\[Doc-(\d+)\]", claim.claim_text, re.IGNORECASE)
            if not cit_matches:
                cit_matches = re.findall(r"\bDoc-(\d+)\b", claim.claim_text, re.IGNORECASE)
            if not cit_matches and getattr(claim, "cited_doc_ids", None):
                for h in claim.cited_doc_ids:
                    m = re.search(r"Doc-(\d+)", str(h), re.IGNORECASE)
                    if m:
                        cit_matches.append(m.group(1))
            if not cit_matches and getattr(claim, "cited_doc_id", None):
                m = re.search(r"Doc-(\d+)", str(claim.cited_doc_id), re.IGNORECASE)
                if m:
                    cit_matches.append(m.group(1))

            target_premise = ""
            target_handle = "General"

            valid_indices: List[int] = []
            for num_str in cit_matches:
                try:
                    k = int(num_str)
                    if 1 <= k <= len(context_list):
                        valid_indices.append(k)
                except ValueError:
                    pass

            doc_handles = [f"Doc-{k}" for k in valid_indices]
            clean_hyp = self._clean_hypothesis_for_nli(claim.claim_text)

            # Multi-Citation Comparative Decomposition:
            # If claim cites multiple discrete documents and contains comparative conjunctions,
            # decompose into constituent independent clauses evaluated against corresponding document contexts.
            if len(doc_handles) > 1 and self._COMPARATIVE_SPLIT_REGEX.search(claim.claim_text):
                sub_parts = [p.strip() for p in self._COMPARATIVE_SPLIT_REGEX.split(claim.claim_text) if p.strip()]
                if len(sub_parts) > 1:
                    for s_idx, part in enumerate(sub_parts):
                        target_k = valid_indices[min(s_idx, len(valid_indices) - 1)]
                        target_handle = f"Doc-{target_k}"
                        raw_doc_ctx = self._resolve_context_text(context_list[target_k - 1])
                        clean_sub = self._clean_hypothesis_for_nli(part)
                        p_win = self._extract_premise_window(clean_sub, raw_doc_ctx)
                        target_premise = p_win if p_win else raw_doc_ctx

                        batch_hypotheses.append(clean_sub)
                        batch_premises.append(target_premise)
                        batch_meta.append((idx, s_idx, len(sub_parts), target_handle, target_premise))
                    continue

            # Standard Single or Unified Multi-Citation evaluation
            if len(valid_indices) == 1:
                k = valid_indices[0]
                target_handle = f"Doc-{k}"
                target_ctx = context_list[k - 1]
                raw_context = self._resolve_context_text(target_ctx)
                p_win = self._extract_premise_window(clean_hyp, raw_context)
                target_premise = p_win if p_win else raw_context
            elif len(valid_indices) > 1:
                unified_parts = []
                for k in valid_indices:
                    raw_context = self._resolve_context_text(context_list[k - 1])
                    if raw_context:
                        p_win = self._extract_premise_window(clean_hyp, raw_context)
                        unified_parts.append(f"[Doc-{k}]: {p_win or raw_context}")
                target_premise = "\n\n---\n\n".join(unified_parts) if unified_parts else fallback_corpus
                target_handle = ", ".join(doc_handles)
            else:
                # No citation tag detected: fall back to top-ranked context or concatenated top contexts
                if context_list:
                    target_handle = "Doc-1"
                    raw_context = self._resolve_context_text(context_list[0])
                    p_win = self._extract_premise_window(clean_hyp, raw_context)
                    target_premise = p_win if p_win else raw_context
                else:
                    target_handle = "General"
                    raw_fallback = self._resolve_context_text(fallback_corpus)
                    p_win = self._extract_premise_window(clean_hyp, raw_fallback)
                    target_premise = p_win if p_win else raw_fallback

            batch_hypotheses.append(clean_hyp)
            batch_premises.append(target_premise)
            batch_meta.append((idx, 0, 1, target_handle, target_premise))

        # 2. Run batch NLI inference for non-meta claims
        if batch_hypotheses:
            try:
                predictions = verifier.predict_batch(
                    claims=batch_hypotheses,
                    premises=batch_premises,
                    batch_size=batch_size,
                )
            except TypeError:
                predictions = verifier.predict_batch(
                    claims=batch_hypotheses,
                    premises=batch_premises,
                )

            # Group predictions by claim_index
            claim_results: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
            for (c_idx, s_idx, total_parts, handle, premise_used), pred in zip(batch_meta, predictions):
                probs = pred.get("probabilities", {})
                prob_contra = probs.get("contradiction", 0.0)
                prob_entail = probs.get("entailment", 0.0)
                prob_neutral = probs.get("neutral", 0.0)

                if prob_contra >= self.tau_contradiction:
                    verdict: Literal["ENTAILED", "CONTRADICTION", "NEUTRAL"] = "CONTRADICTION"
                    confidence = prob_contra
                elif prob_entail >= self.tau_entailment:
                    verdict = "ENTAILED"
                    confidence = prob_entail
                else:
                    verdict = "NEUTRAL"
                    confidence = prob_neutral

                claim_results[c_idx].append({
                    "sub_index": s_idx,
                    "total_parts": total_parts,
                    "handle": handle,
                    "premise": premise_used,
                    "verdict": verdict,
                    "confidence": confidence,
                    "probabilities": probs,
                })

            for c_idx, sub_res_list in claim_results.items():
                claim = claims[c_idx]
                if len(sub_res_list) > 1:
                    # Multi-clause comparative adjudication:
                    # ENTAILED if and only if all constituent clauses are ENTAILED
                    if any(r["verdict"] == "CONTRADICTION" for r in sub_res_list):
                        final_verdict: Literal["ENTAILED", "CONTRADICTION", "NEUTRAL"] = "CONTRADICTION"
                        final_conf = max(r["probabilities"].get("contradiction", 0.0) for r in sub_res_list)
                    elif all(r["verdict"] == "ENTAILED" for r in sub_res_list):
                        final_verdict = "ENTAILED"
                        final_conf = min(r["probabilities"].get("entailment", 0.0) for r in sub_res_list)
                    else:
                        final_verdict = "NEUTRAL"
                        final_conf = max(r["probabilities"].get("neutral", 0.0) for r in sub_res_list)

                    combined_premise = "\n\n---\n\n".join(f"[{r['handle']}]: {r['premise']}" for r in sub_res_list)
                    avg_probs = {
                        "entailment": sum(r["probabilities"].get("entailment", 0.0) for r in sub_res_list) / len(sub_res_list),
                        "contradiction": sum(r["probabilities"].get("contradiction", 0.0) for r in sub_res_list) / len(sub_res_list),
                        "neutral": sum(r["probabilities"].get("neutral", 0.0) for r in sub_res_list) / len(sub_res_list),
                    }
                    audit = ClaimAudit(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        cited_premise=combined_premise,
                        probabilities=avg_probs,
                        verdict=final_verdict,
                        confidence=final_conf,
                    )
                else:
                    single_res = sub_res_list[0]
                    audit = ClaimAudit(
                        claim_id=claim.claim_id,
                        claim_text=claim.claim_text,
                        cited_premise=single_res["premise"],
                        probabilities=single_res["probabilities"],
                        verdict=single_res["verdict"],
                        confidence=single_res["confidence"],
                    )
                audits.append(audit)

        # Re-sort audits to match the original claim order
        claim_id_to_order = {c.claim_id: i for i, c in enumerate(claims)}
        audits.sort(key=lambda a: claim_id_to_order.get(a.claim_id, 0))

        # 3. Compute aggregate metrics
        entailed_count = sum(1 for a in audits if a.verdict == "ENTAILED")
        faithfulness_score = entailed_count / len(audits) if audits else 1.0
        has_contradiction = any(a.verdict == "CONTRADICTION" for a in audits)

        # 4. Enforce automated safety gating action
        if has_contradiction:
            action: Literal["PASS", "TRIGGER_REWRITE", "WARN"] = "TRIGGER_REWRITE"
        elif faithfulness_score >= 0.80:
            action = "PASS"
        else:
            action = "WARN"

        return TrustAuditReport(
            draft_text=draft_text,
            faithfulness_score=faithfulness_score,
            has_contradiction=has_contradiction,
            action=action,
            audits=audits,
        )

    def verify(
        self,
        claims: List[AtomicClaim],
        context_map: Optional[Dict[str, Any]] = None,
        nli_verifier: Optional[Any] = None,
        draft_text: str = "",
        batch_size: int = 32,
        contexts: Optional[List[Any]] = None,
    ) -> TrustAuditReport:
        """Alias for adjudicate conforming to standard verification interface."""
        return self.adjudicate(
            claims=claims,
            context_map=context_map,
            nli_verifier=nli_verifier,
            draft_text=draft_text,
            batch_size=batch_size,
            contexts=contexts,
        )

    def verify_claims(
        self,
        claims: List[AtomicClaim],
        contexts: Any,
        nli_verifier: Optional[Any] = None,
        draft_text: str = "",
        batch_size: int = 32,
    ) -> TrustAuditReport:
        """Verify atomic claims by dynamically routing each claim's premise to its cited context index."""
        if isinstance(contexts, list):
            context_list = contexts
            context_map = {f"Doc-{i}": c for i, c in enumerate(contexts, start=1)}
        elif isinstance(contexts, dict):
            context_map = contexts
            context_list = list(contexts.values())
        else:
            context_list = [contexts]
            context_map = {"Doc-1": contexts}

        return self.adjudicate(
            claims=claims,
            context_map=context_map,
            nli_verifier=nli_verifier,
            draft_text=draft_text,
            batch_size=batch_size,
            contexts=context_list,
        )


def verify(
    adjudicator: AuditAdjudicator,
    claims: List[AtomicClaim],
    context_map: Optional[Dict[str, Any]] = None,
    nli_verifier: Optional[Any] = None,
    draft_text: str = "",
    batch_size: int = 32,
    contexts: Optional[List[Any]] = None,
) -> TrustAuditReport:
    """Verify atomic claims using the provided adjudicator and NLI verifier."""
    return adjudicator.adjudicate(
        claims=claims,
        context_map=context_map,
        nli_verifier=nli_verifier,
        draft_text=draft_text,
        batch_size=batch_size,
        contexts=contexts,
    )


def verify_claims(
    adjudicator: AuditAdjudicator,
    claims: List[AtomicClaim],
    contexts: Any,
    nli_verifier: Optional[Any] = None,
    draft_text: str = "",
    batch_size: int = 32,
) -> TrustAuditReport:
    """Verify claims dynamically routed to cited context indices."""
    return adjudicator.verify_claims(
        claims=claims,
        contexts=contexts,
        nli_verifier=nli_verifier,
        draft_text=draft_text,
        batch_size=batch_size,
    )


# Alias Adjudicator to AuditAdjudicator
Adjudicator = AuditAdjudicator



