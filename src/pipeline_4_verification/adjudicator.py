r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_4_verification/adjudicator.py
   - Role: Trust scoring, safety gating, and audit report adjudication engine.
   - Purpose: Aligns atomic claims to focused section-breadcrumbed premises, executes NLI
     batch predictions, evaluates threshold-based verdicts (ENTAILED, CONTRADICTION, NEUTRAL),
     calculates the global faithfulness score, guards against zero-claim audit bypasses, and
     enforces automated downstream actions (PASS, TRIGGER_REWRITE, WARN).

2. INPUT (IP):
   - claims (list[AtomicClaim]): Atomic propositions from `src/pipeline_4_verification/claim_extractor.py`.
   - context_map (dict[str, str]): Map of document handles (e.g., "Doc-1") to full text contexts.
   - nli_verifier (DebertaNLIVerifier): DeBERTa sequence classifier from `src/pipeline_4_verification/nli_model.py`.
   - draft_text (str, optional): Full synthesized draft text from `src/pipeline_3_generation/`.

3. PROCESS UNDER THE HOOD:
   - Zero-Claim Safety Guard:
     * If `len(claims) == 0`: checks whether `draft_text` contains substantive content or bullet points
       (excluding standard fallback phrases like "does not contain sufficient information").
     * If substantive text exists without verifiable claims/citations: sets `faithfulness_score = 0.00`
       and `action = "WARN"`.
     * If legitimate fallback response: sets `faithfulness_score = 1.00` and `action = "PASS"`.
   - Focused Premise Window Extraction:
     * Extracts the top `[Document: ... | Section: ...]` breadcrumb header from the cited context.
     * For long passages (> 1200 chars), identifies the coherent project or topic block
       having highest semantic keyword overlap with the claim.
     * Assembles the focused premise: `{breadcrumb}\n{focused_block}` to ensure complete
       contextual grounding without cross-attention dilution.
     * Gracefully resolves out-of-bounds or missing citation handles to the active corpus context.
   - Batch Inference: Queries `nli_verifier.predict_batch()` for all (claim, premise_window) pairs.
   - Verdict Decision Rule:
     * If $P(\text{contradiction}) \ge \tau_{\text{contradiction}}$ (0.60) $\implies$ "CONTRADICTION"
     * Else if $P(\text{entailment}) \ge \tau_{\text{entailment}}$ (0.80) $\implies$ "ENTAILED"
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
   - re: Standard library regex tokenization.
   - src.common.config: Centralized decision thresholds (`tau_entailment`, `tau_contradiction`).
   - src.common.schemas (AtomicClaim, ClaimAudit, TrustAuditReport): Strict data models.
   - src.pipeline_4_verification.nli_model.DebertaNLIVerifier: NLI model interface.
================================================================================
"""

import re
from typing import Any, Dict, List, Literal, Optional

from src.common.config import config
from src.common.schemas import AtomicClaim, ClaimAudit, TrustAuditReport


class AuditAdjudicator:
    """Evaluates NLI scores across claims, computes faithfulness, and determines safety actions."""

    def __init__(
        self,
        tau_entailment: float = config.thresholds.tau_entailment,
        tau_contradiction: float = config.thresholds.tau_contradiction,
    ) -> None:
        """Initialize adjudicator with decision thresholds.

        Args:
            tau_entailment: Minimum probability required for ENTAILED verdict (default: 0.80).
            tau_contradiction: Minimum probability required for CONTRADICTION verdict (default: 0.60).
        """
        self.tau_entailment = tau_entailment
        self.tau_contradiction = tau_contradiction

    def _extract_premise_window(self, claim_text: str, context: str) -> str:
        """Extract the section breadcrumb and most relevant block from context for the given claim.

        Args:
            claim_text: Proposition text of the atomic claim.
            context: Full text passage associated with the cited document.

        Returns:
            Focused premise string combining breadcrumb and relevant project/topic block.
        """
        if not context or not context.strip():
            return ""

        # Normalize unicode dashes, spaces, and brackets
        norm_context = context.replace("\u202f", " ").replace("\xa0", " ")
        norm_context = norm_context.replace("—", " - ").replace("–", " - ").replace("\u2011", "-")

        lines = [l.strip() for l in norm_context.split("\n") if l.strip()]
        breadcrumb = lines[0] if lines and lines[0].startswith("[Document:") else ""

        # Extract core content keywords from claim, stripping structural/framing prefixes
        clean_claim = re.sub(
            r"^Under\s+[^,:]+[,:]\s*(?:the\s+(?:candidate|document|specification|item|technologies)\s+(?:completed|specifies|states|utilizes|features|include|utilized\s+include|documented\s+specification\s+or\s+item\s+is|documented\s+items\s+include):?\s*)?",
            "",
            claim_text,
            flags=re.IGNORECASE,
        )
        clean_claim = re.sub(
            r"^The\s+(?:document\s+specifies|technologies\s+utilized\s+include):?\s*",
            "",
            clean_claim,
            flags=re.IGNORECASE,
        )
        clean_claim = clean_claim.replace("—", " - ").replace("–", " - ").replace("\u2011", "-").strip().rstrip(".")
        claim_words = set(re.findall(r"\b[\w+#.-]{2,}\b", clean_claim.lower()))

        # For compact parent contexts (<= 1500 chars), return the full parent context directly
        if len(norm_context) <= 1500:
            return norm_context

        # For short technical tokens or entity names (e.g. MongoDB, Docker, Python)
        # ensure premise search scans across the complete parent chunk text without penalization
        is_short_term = len(clean_claim.split()) <= 3
        if is_short_term and claim_words:
            # If all/any claim tokens appear in the parent chunk, prioritize windows containing the token
            norm_lower = norm_context.lower()
            if any(w in norm_lower for w in claim_words):
                # Search across windows containing the term
                body_lines = lines[1:] if breadcrumb else lines
                matching_lines = [l for l in body_lines if any(w in l.lower() for w in claim_words)]
                if matching_lines:
                    selected_window = "\n".join(matching_lines)
                    if breadcrumb and not selected_window.startswith("[Document:"):
                        return f"{breadcrumb}\n{selected_window}"
                    return selected_window
                return norm_context

        body_lines = lines[1:] if breadcrumb else lines

        # Candidate premise windows: complete bullet units and project-level blocks
        windows: List[str] = []

        # 1. Complete bullet units (accumulating wrapped lines)
        curr_bullet: List[str] = []
        for line in body_lines:
            is_bullet = line.startswith(("•", "-", "*"))
            if is_bullet and curr_bullet:
                windows.append("\n".join(curr_bullet))
                curr_bullet = []
            curr_bullet.append(line)
        if curr_bullet:
            windows.append("\n".join(curr_bullet))

        # 2. Project-level blocks (combining titled headers with sub-bullets)
        curr_block: List[str] = []
        for line in body_lines:
            is_bullet = line.startswith(("•", "-", "*"))
            is_titled_bullet = is_bullet and (":" in line[:50] or " - " in line[:50] or " – " in line[:50])
            is_named_title = not is_bullet and len(line.split()) <= 10 and not line.endswith((".", "!"))
            if (is_titled_bullet or is_named_title) and curr_block:
                windows.append("\n".join(curr_block))
                curr_block = []
            curr_block.append(line)
        if curr_block:
            windows.append("\n".join(curr_block))

        if not windows:
            return norm_context

        best_score = -1.0
        best_window = norm_context

        for win in windows:
            win_words = set(re.findall(r"\b[\w+#.-]{2,}\b", win.lower()))
            if not win_words:
                continue
            overlap = len(claim_words & win_words)
            if overlap == 0:
                continue
            # Density score: reward high keyword overlap relative to window compactness
            score = overlap / (len(win_words) ** 0.5)
            if score > best_score:
                best_score = score
                best_window = win

        if best_score <= 0:
            return norm_context

        if breadcrumb and not best_window.startswith("[Document:"):
            return f"{breadcrumb}\n{best_window}"
        return best_window

    def adjudicate(
        self,
        claims: List[AtomicClaim],
        context_map: Dict[str, str],
        nli_verifier: Any,
        draft_text: str = "",
    ) -> TrustAuditReport:
        """Audit atomic claims against cited contexts and produce a TrustAuditReport.

        Args:
            claims: List of AtomicClaim instances to verify.
            context_map: Mapping from document tags (e.g. 'Doc-1') to passage strings.
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

        # 1. Resolve focused premise windows for each claim
        fallback_corpus = "\n\n".join(context_map.values()) if context_map else ""
        premises: List[str] = []
        claim_texts: List[str] = []

        for claim in claims:
            claim_texts.append(claim.claim_text)
            if claim.cited_doc_id and claim.cited_doc_id in context_map:
                raw_context = context_map[claim.cited_doc_id]
            else:
                raw_context = fallback_corpus

            premise_window = self._extract_premise_window(claim.claim_text, raw_context)
            premises.append(premise_window if premise_window else raw_context)

        # 2. Run batch NLI inference
        predictions = nli_verifier.predict_batch(claims=claim_texts, premises=premises)

        audits: List[ClaimAudit] = []
        for claim, premise, pred in zip(claims, premises, predictions):
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

