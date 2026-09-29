r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_5_self_correction/corrector.py
   - Role: Closed-loop verification self-correction and selective claim pruning engine.
   - Purpose: Evaluates unverified or contradictory claims from the audit report, coordinates
     1-pass corrective LLM rewriting, and implements selective claim pruning so that verified
     assertions are preserved without collapsing into full response suppression when partial
     grounded information is verified.

2. INPUT (IP):
   - query (str): Original user query.
   - draft_text (str): Initial draft response.
   - audit_report (TrustAuditReport): Initial claim-level audit report.
   - top_contexts (list[RetrievalCandidate]): Retrieved parent context passages.
   - generator (Any): Generator instance for LLM rewrites.
   - claim_extractor (AtomicClaimExtractor): Claim decomposition instance.
   - adjudicator (AuditAdjudicator): NLI adjudication engine.
   - nli_verifier (DeBERTaNLIVerifier): DeBERTa cross-encoder verifier.

3. PROCESS UNDER THE HOOD:
   - Evaluates audit report action (`WARN`, `TRIGGER_REWRITE`, or low faithfulness).
   - Identifies failed claims (`verdict != "ENTAILED"`).
   - Generates 1-pass corrective prompt using `build_correction_prompt`.
   - Adjudicates the corrected draft against context passages.
   - Selective Claim Pruning Fallback:
     * If the LLM rewrite collapses into full insufficient-evidence fallback despite the
       original draft containing verified `ENTAILED` claims, selectively prunes the unverified
       lines from the original draft while retaining all verified entailed statements.
     * Recalculates faithfulness score and promotes verdict to `PASS` if pruned draft is 100% entailed.

4. OUTPUT (OP):
   - TrustAuditReport: Hardened, maximally faithful audit report.

5. LIBRARIES & DEPENDENCIES:
   - re: Regex pattern matching.
   - typing (Any, Dict, List, Optional): Standard type annotations.
   - src.common.schemas (ClaimAudit, TrustAuditReport): Pydantic contracts.
   - src.pipeline_3_generation.prompt (FALLBACK_INSUFFICIENT_INFO, build_correction_prompt): Prompt builder.
   - src.pipeline_3_generation.citation (validate_and_parse_citations): Citation parser.
================================================================================
"""

import re
from typing import Any, Dict, List, Optional

from src.common.config import config
from src.common.schemas import ClaimAudit, RetrievalCandidate, TrustAuditReport
from src.pipeline_3_generation.citation_check import validate_and_parse_citations
from src.pipeline_3_generation.prompt import FALLBACK_INSUFFICIENT_INFO, build_correction_prompt


class SelfCorrector:
    """Coordinates automated verification self-correction and selective claim pruning."""

    def __init__(self) -> None:
        pass

    def correct(
        self,
        query: str,
        draft_text: str,
        audit_report: TrustAuditReport,
        top_contexts: List[RetrievalCandidate],
        generator: Any,
        claim_extractor: Any,
        adjudicator: Any,
        nli_verifier: Any,
        context_map: Dict[str, Any],
        default_section: Optional[str] = None,
    ) -> TrustAuditReport:
        """Execute automated 1-pass corrective rewrite or selective claim pruning.

        Args:
            query: Original user query.
            draft_text: Initial synthesized draft text.
            audit_report: Initial TrustAuditReport.
            top_contexts: Top reranked context passages.
            generator: LLM generation instance.
            claim_extractor: AtomicClaimExtractor instance.
            adjudicator: AuditAdjudicator instance.
            nli_verifier: DeBERTaNLIVerifier instance.
            context_map: Map of Doc-X handles to RetrievalCandidate objects.
            default_section: Optional default section name.

        Returns:
            Updated TrustAuditReport.
        """
        needs_correction = (
            audit_report.action in ("WARN", "TRIGGER_REWRITE")
            or audit_report.has_contradiction
            or audit_report.faithfulness_score < 0.85
            or any(a.verdict != "ENTAILED" for a in audit_report.audits)
        )

        if not needs_correction:
            return audit_report

        failed_claims = [
            a.claim_text for a in audit_report.audits
            if a.verdict != "ENTAILED"
        ]

        if not failed_claims:
            return audit_report

        print(f"[Self-Correction] Triggered 1-pass corrective rewrite for {len(failed_claims)} unverified claims.")

        # Step 1: Execute 1-Pass LLM Corrective Rewrite
        correction_prompt = build_correction_prompt(
            query=query.strip(),
            context=top_contexts,
            draft=draft_text,
            failed_claims=failed_claims,
        )
        corrected_draft_text = generator.generate(correction_prompt)
        corrected_draft = validate_and_parse_citations(
            draft_text=corrected_draft_text,
            max_valid_doc_id=len(top_contexts),
        )
        corrected_claims = claim_extractor.extract_claims(
            corrected_draft,
            default_section=default_section,
        )
        corrected_report = adjudicator.adjudicate(
            claims=corrected_claims,
            context_map=context_map,
            nli_verifier=nli_verifier,
            draft_text=corrected_draft_text,
        )

        # Step 2: Evaluate Corrective Report Quality
        if corrected_report.faithfulness_score >= 0.85 and not corrected_report.has_contradiction and corrected_report.draft_text.strip() != FALLBACK_INSUFFICIENT_INFO:
            return corrected_report

        # Step 3: Selective Claim Pruning
        # Selectively target and remove only the unverified/failed propositions, preserving verified claims
        entailed_audits = [a for a in audit_report.audits if a.verdict == "ENTAILED"]
        if entailed_audits:
            def _normalize_line(t: str) -> str:
                t = re.sub(r"\[Doc-\d+\]", "", t)
                t = re.sub(r"^[-*•\d.]+\s*", "", t)
                t = re.sub(r"[^a-zA-Z0-9\s]", "", t).lower()
                return re.sub(r"\s+", " ", t).strip()

            norm_failed = [_normalize_line(fc) for fc in failed_claims if _normalize_line(fc)]

            lines = draft_text.splitlines()
            retained_lines: List[str] = []

            for line in lines:
                clean_l = line.strip()
                if not clean_l or clean_l.startswith("#") or clean_l.startswith("**"):
                    retained_lines.append(line)
                    continue

                norm_l = _normalize_line(clean_l)
                is_failed = any(
                    nfc in norm_l or norm_l in nfc
                    for nfc in norm_failed
                ) if norm_l else False

                if not is_failed:
                    retained_lines.append(line)

            pruned_draft = "\n".join(retained_lines).strip()
            if pruned_draft:
                pruned_claims = claim_extractor.extract_claims(
                    validate_and_parse_citations(pruned_draft, max_valid_doc_id=len(top_contexts)),
                    default_section=default_section,
                )
                if pruned_claims:
                    pruned_report = adjudicator.adjudicate(
                        claims=pruned_claims,
                        context_map=context_map,
                        nli_verifier=nli_verifier,
                        draft_text=pruned_draft,
                    )
                    if pruned_report.faithfulness_score >= 0.85:
                        return pruned_report

        if corrected_report.faithfulness_score > audit_report.faithfulness_score:
            return corrected_report

        return audit_report
