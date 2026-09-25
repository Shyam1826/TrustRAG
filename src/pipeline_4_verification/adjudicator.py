r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_4_verification/adjudicator.py
   - Role: Trust scoring, safety gating, and audit report adjudication engine.
   - Purpose: Aligns atomic claims to focused section-breadcrumbed parent premises, performs
     multi-citation premise unification across multiple referenced documents with distinct
     delimiters ('\n\n---\n\n'), normalizes meta-document scaffolding from hypotheses, handles
     comparative meta-claims without false-neutral penalties, executes NLI batch predictions,
     evaluates threshold-based verdicts (ENTAILED, CONTRADICTION, NEUTRAL), calculates the
     global faithfulness score, guards against zero-claim audit bypasses, and enforces
     automated downstream actions (PASS, TRIGGER_REWRITE, WARN). Accommodates mathematical
     formulas, full parent premise expansion, and unified variable definitions within DeBERTa's
     512-token limit.

2. INPUT (IP):
   - claims (list[AtomicClaim]): Atomic propositions from `src/pipeline_4_verification/claim_extractor.py`.
   - context_map (dict[str, Any]): Map of document handles (e.g., "Doc-1") to parent texts or candidates.
   - nli_verifier (DebertaNLIVerifier): DeBERTa sequence classifier from `src/pipeline_4_verification/nli_model.py`.
   - draft_text (str, optional): Full synthesized draft text from `src/pipeline_3_generation/`.

3. PROCESS UNDER THE HOOD:
   - Zero-Claim Safety Guard:
     * If `len(claims) == 0`: checks whether `draft_text` contains substantive content or bullet points
       (excluding standard fallback phrases like "does not contain sufficient information").
     * If substantive text exists without verifiable claims/citations: sets `faithfulness_score = 0.00`
       and `action = "WARN"`.
     * If legitimate fallback response: sets `faithfulness_score = 1.00` and `action = "PASS"`.
   - Meta-Claim Handling:
     * For claims with `is_meta=True` (e.g. comparative synthesis connectors), assigns `verdict = "ENTAILED"`
       with `confidence = 1.0` and `cited_premise = "Comparative Meta-Analytical Synthesis"`, avoiding
       unnecessary literal NLI rejections.
   - Document Entity Identity Resolution & Premise Formulation:
     * In `_resolve_context_text(ctx_obj)`: extracts full parent context and prepends document identity
       `[Document: {doc_id} | Section: {section_name}]\n` if not already present.
     * In `_clean_hypothesis_for_nli(claim_text)`: strips meta-document scaffolding (e.g. "The resume for X describes...",
       "The engineering specifications document notes...") into clean affirmative propositions for DeBERTa.
   - Sentence-Boundary Premise Expansion & 512-Token Bound:
     * Expands premise windows along clean sentence/paragraph boundaries (`[.!?]\s+`, `\n{2,}`, `\n`) up to
       `premise_window_size` (1200 characters / max 350 words), focused around the highest keyword match
       to avoid tail truncation under DeBERTa-v3 512-token limits.
     * Preserves variable identifiers (`d_model`, `d_k`, `d_v`), mathematical formulas, and numerical parameters.
   - Multi-Citation Premise Unification:
     * For multi-citation claims (e.g. `[Doc-1, Doc-2]`), extracts the focused premise window
       from EACH referenced document context and concatenates them with distinct delimiters
       (`\n\n---\n\n`) allowing DeBERTa to verify joint claims across multiple sources.
   - Batch Inference: Queries `nli_verifier.predict_batch()` for cleaned (claim, premise_window) pairs.
   - Verdict Decision Rule:
     * If $P(\text{contradiction}) \ge \tau_{\text{contradiction}}$ $\implies$ "CONTRADICTION"
     * Else if $P(\text{entailment}) \ge \tau_{\text{entailment}}$ $\implies$ "ENTAILED"
     * Else $\implies$ "NEUTRAL"
   - Metrics & Gating:
     * $\text{faithfulness\_score} = \frac{|\{c \mid \text{verdict}(c) = \text{ENTAILED}\}|}{|\text{claims}|}$
     * $\text{has\_contradiction} = \exists c : \text{verdict}(c) = \text{CONTRADICTION}$
     * Gating Action:
       - If `has_contradiction` $\implies$ "TRIGGER_REWRITE"
       - Else if `faithfulness_score >= 0.80` and not `has_contradiction` $\implies$ "PASS"
       - Else $\implies$ "WARN"

4. OUTPUT (OP):
   - TrustAuditReport: Comprehensive audit report containing:
     * draft_text (str)
     * faithfulness_score (float)
     * has_contradiction (bool)
     * action ("PASS" | "TRIGGER_REWRITE" | "WARN")
     * audits (list[ClaimAudit])
   - Consumed by: End-user response dispatcher or automated rewrite pipelines.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library regex tokenization and sentence boundary detection.
   - sklearn.feature_extraction.text (ENGLISH_STOP_WORDS): Standard NLP stop words.
   - src.common.config: Centralized decision thresholds and verification settings.
   - src.common.schemas (AtomicClaim, ClaimAudit, TrustAuditReport): Strict data models.
   - src.pipeline_4_verification.nli_model.DebertaNLIVerifier: NLI model interface.
================================================================================
"""

import re
from typing import Any, Dict, List, Literal, Optional, Set
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config
from src.common.schemas import AtomicClaim, ClaimAudit, TrustAuditReport


class AuditAdjudicator:
    """Evaluates NLI scores across claims, computes faithfulness, and determines safety actions."""

    def __init__(
        self,
        tau_entailment: Optional[float] = None,
        tau_contradiction: Optional[float] = None,
        premise_window_size: Optional[int] = None,
    ) -> None:
        """Initialize adjudicator with decision thresholds.

        Args:
            tau_entailment: Minimum probability required for ENTAILED verdict (default from config).
            tau_contradiction: Minimum probability required for CONTRADICTION verdict (default from config).
            premise_window_size: Maximum character length for extracted premise window (default from config).
        """
        settings = config.verification
        self.tau_entailment = tau_entailment if tau_entailment is not None else settings.tau_entailment
        self.tau_contradiction = tau_contradiction if tau_contradiction is not None else settings.tau_contradiction
        self.premise_window_size = premise_window_size if premise_window_size is not None else getattr(settings, "premise_window_size", 1200)

    def _get_stop_words(self) -> Set[str]:
        """Combine standard English stop words with configured custom stop words."""
        custom = set(config.retrieval.custom_stop_words)
        return ENGLISH_STOP_WORDS | custom

    def _clean_hypothesis_for_nli(self, claim_text: str) -> str:
        """Strip meta-document scaffolding from hypothesis before passing to DeBERTa.

        Removes structural carrier framing (e.g., 'The resume for Sanjeev M focuses on...',
        'The engineering specifications document describes...', 'According to the resume of X...')
        to produce clean, grounded affirmative propositions that align directly with factual premise text.

        Args:
            claim_text: Raw atomic claim text.

        Returns:
            Cleaned affirmative proposition text.
        """
        cleaned = claim_text.strip()

        # Universal document meta-descriptor types (e.g., engineering specifications document, resume, CV, paper)
        doc_types = (
            r"(?:(?:[a-zA-Z_0-9-]+\s+)*(?:resume|cv|document|specs?|specifications?(?:\s+document)?|"
            r"paper|contract|agreement|report|overview|profile|guidelines?|manual|datasheet|candidate))"
        )
        verbs = (
            r"(?:focuses\s+on|focuses|lists|describes|states\s+that|states|notes\s+that|notes|"
            r"details|features|specifies|highlights|presents|outlines|defines|mentions|identifies)"
        )

        # 1. ^The <doc_types> (for|of|titled|regarding|named) <entity> <verbs>:?
        cleaned = re.sub(
            rf"^The\s+{doc_types}\s+(?:for|of|titled|regarding|named)\s+[^:;,\n]+?\s+{verbs}:?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # 2. ^The <doc_types> <verbs>:?
        cleaned = re.sub(
            rf"^The\s+{doc_types}\s+{verbs}:?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # 3. ^According to (the)? <doc_types> (for/of ...)? [,:]
        cleaned = re.sub(
            rf"^According\s+to\s+(?:the\s+)?{doc_types}(?:\s+(?:for|of|titled|regarding|named)\s+[^:;,\n]+?)?[,:]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

        # 4. ^Under [Section], (the candidate/document ...)?
        cleaned = re.sub(
            r"^Under\s+[^,:]+[,:]\s*(?:the\s+(?:candidate|document|specification|item|technologies)\s+(?:completed|specifies|states|utilizes|features|include|utilized\s+include|documented\s+specification\s+or\s+item\s+is|documented\s+items\s+include):?\s*)?",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        cleaned = re.sub(
            r"^The\s+(?:document\s+specifies|technologies\s+utilized\s+include):?\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )

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
        preserving mathematical equations and full parent context within DeBERTa's 512-token limit.

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
        # Preserves complete sentence boundaries without splitting words
        raw_segments = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", body_text) if s.strip()]
        if not raw_segments:
            return norm_context

        # Score each segment based on keyword overlap density
        best_idx = 0
        best_score = -1.0
        matching_indices: List[int] = []

        for idx, seg in enumerate(raw_segments):
            seg_tokens = re.findall(r"\b[a-zA-Z0-9_+#.-]+\b", seg.lower())
            seg_words = set(w for w in seg_tokens if w not in stop_words)
            if not seg_words:
                continue
            overlap = len(claim_words & seg_words)
            if overlap == 0:
                continue
            matching_indices.append(idx)
            score = overlap / (len(seg_words) ** 0.5)
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
        context_map: Dict[str, Any],
        nli_verifier: Any,
        draft_text: str = "",
    ) -> TrustAuditReport:
        """Audit atomic claims against cited contexts and produce a TrustAuditReport.

        Args:
            claims: List of AtomicClaim instances to verify.
            context_map: Mapping from document tags (e.g. 'Doc-1') to passages or candidates.
            nli_verifier: DebertaNLIVerifier instance.
            draft_text: Optional full synthesized draft text.

        Returns:
            TrustAuditReport containing per-claim audits, faithfulness score, and action verdict.
        """
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

        # 1. Separate meta claims and factual claims for premise resolution
        fallback_corpus = "\n\n---\n\n".join(self._resolve_context_text(v) for v in context_map.values()) if context_map else ""

        audits: List[ClaimAudit] = []
        nli_claims: List[str] = []
        nli_premises: List[str] = []
        nli_claim_indices: List[int] = []

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

            # Determine target document handles for this claim
            doc_handles = list(getattr(claim, "cited_doc_ids", []))
            if not doc_handles and claim.cited_doc_id:
                # Handle comma-separated or single handle in cited_doc_id
                doc_handles = [h.strip() for h in re.findall(r"Doc-\d+", claim.cited_doc_id)]
                if not doc_handles and claim.cited_doc_id in context_map:
                    doc_handles = [claim.cited_doc_id]

            if len(doc_handles) > 1:
                # Multi-Citation Premise Unification:
                # Extract focused premise window from EACH referenced document and concatenate with delimiter
                unified_parts = []
                for handle in doc_handles:
                    raw_doc_ctx = context_map.get(handle, "")
                    doc_ctx = self._resolve_context_text(raw_doc_ctx)
                    if doc_ctx:
                        p_win = self._extract_premise_window(claim.claim_text, doc_ctx)
                        unified_parts.append(f"[{handle}]: {p_win or doc_ctx}")
                if unified_parts:
                    unified_premise = "\n\n---\n\n".join(unified_parts)
                    nli_premises.append(unified_premise)
                else:
                    nli_premises.append(fallback_corpus)
            elif len(doc_handles) == 1:
                single_handle = doc_handles[0]
                raw_context = self._resolve_context_text(context_map.get(single_handle, fallback_corpus))
                premise_window = self._extract_premise_window(claim.claim_text, raw_context)
                nli_premises.append(premise_window if premise_window else raw_context)
            else:
                raw_fallback = self._resolve_context_text(fallback_corpus)
                premise_window = self._extract_premise_window(claim.claim_text, raw_fallback)
                nli_premises.append(premise_window if premise_window else raw_fallback)

            nli_claims.append(claim.claim_text)
            nli_claim_indices.append(idx)

        # 2. Run batch NLI inference for non-meta claims using cleaned hypotheses
        if nli_claims:
            cleaned_nli_claims = [self._clean_hypothesis_for_nli(c) for c in nli_claims]
            predictions = nli_verifier.predict_batch(claims=cleaned_nli_claims, premises=nli_premises)
            for claim_idx, premise, pred in zip(nli_claim_indices, nli_premises, predictions):
                claim = claims[claim_idx]
                probs = pred.get("probabilities", {})
                prob_contra = probs.get("contradiction", 0.0)
                prob_entail = probs.get("entailment", 0.0)
                prob_neutral = probs.get("neutral", 0.0)

                # Apply thresholding logic
                if prob_contra >= self.tau_contradiction:
                    verdict: Literal["ENTAILED", "CONTRADICTION", "NEUTRAL"] = "CONTRADICTION"
                    confidence = prob_contra
                elif prob_entail >= self.tau_entailment:
                    verdict = "ENTAILED"
                    confidence = prob_entail
                else:
                    verdict = "NEUTRAL"
                    confidence = prob_neutral

                audit = ClaimAudit(
                    claim_id=claim.claim_id,
                    claim_text=claim.claim_text,
                    cited_premise=premise,
                    probabilities=probs,
                    verdict=verdict,
                    confidence=confidence,
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
