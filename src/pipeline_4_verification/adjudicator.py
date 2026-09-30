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

        doc_types = r"(?:resume|cv|specifications?|specs?|document|paper|report|contract|agreement|template|candidate|overview|profile|guidelines?|manual|datasheet|workbook|spreadsheet|sheet)"
        verbs = r"(?:states?\s+that|states?|notes?\s+that|notes?|describes?|specifies?\s+that|specifies?|features?|lists?|reports?\s+that|reports?|highlights?|presents?|outlines?|defines?|mentions?|identifies?|focuses\s+on|focuses)"

        for _ in range(10):
            prev = cleaned

            # 1. Section Scaffolding & Compound Nested Framing
            cleaned = re.sub(
                r"^Under\s+[^,;:]+,\s*(?:the\s+documented\s+[^:]+:|the\s+document\s+specifies:)\s*",
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
            cleaned = re.sub(
                r"^According\s+to\s+(?:the\s+)?[A-Za-z0-9_.'/\s-]+?[,:]\s*",
                "",
                cleaned,
                flags=re.IGNORECASE,
            )

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
        context_map: Dict[str, Any],
        nli_verifier: Any,
        draft_text: str = "",
        batch_size: int = 32,
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

        fallback_corpus = "\n\n---\n\n".join(self._resolve_context_text(v) for v in context_map.values()) if context_map else ""

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

            # Determine target document handles for this claim
            doc_handles = list(getattr(claim, "cited_doc_ids", []))
            if not doc_handles and claim.cited_doc_id:
                doc_handles = [h.strip() for h in re.findall(r"Doc-\d+", claim.cited_doc_id)]
                if not doc_handles and claim.cited_doc_id in context_map:
                    doc_handles = [claim.cited_doc_id]

            # Multi-Citation Comparative Decomposition:
            # If claim cites multiple discrete documents and contains comparative conjunctions,
            # decompose into constituent independent clauses evaluated against corresponding document contexts.
            if len(doc_handles) > 1 and self._COMPARATIVE_SPLIT_REGEX.search(claim.claim_text):
                sub_parts = [p.strip() for p in self._COMPARATIVE_SPLIT_REGEX.split(claim.claim_text) if p.strip()]
                if len(sub_parts) > 1:
                    for s_idx, part in enumerate(sub_parts):
                        target_handle = doc_handles[min(s_idx, len(doc_handles) - 1)]
                        raw_doc_ctx = self._resolve_context_text(context_map.get(target_handle, fallback_corpus))
                        clean_sub = self._clean_hypothesis_for_nli(part)
                        p_win = self._extract_premise_window(clean_sub, raw_doc_ctx)
                        target_premise = p_win if p_win else raw_doc_ctx

                        batch_hypotheses.append(clean_sub)
                        batch_premises.append(target_premise)
                        batch_meta.append((idx, s_idx, len(sub_parts), target_handle, target_premise))
                    continue

            # Standard Single or Unified Multi-Citation evaluation
            if len(doc_handles) > 1:
                unified_parts = []
                for handle in doc_handles:
                    raw_doc_ctx = self._resolve_context_text(context_map.get(handle, ""))
                    if raw_doc_ctx:
                        p_win = self._extract_premise_window(claim.claim_text, raw_doc_ctx)
                        unified_parts.append(f"[{handle}]: {p_win or raw_doc_ctx}")
                target_premise = "\n\n---\n\n".join(unified_parts) if unified_parts else fallback_corpus
                target_handle = ", ".join(doc_handles)
            elif len(doc_handles) == 1:
                target_handle = doc_handles[0]
                raw_context = self._resolve_context_text(context_map.get(target_handle, fallback_corpus))
                p_win = self._extract_premise_window(claim.claim_text, raw_context)
                target_premise = p_win if p_win else raw_context
            else:
                target_handle = "General"
                raw_fallback = self._resolve_context_text(fallback_corpus)
                p_win = self._extract_premise_window(claim.claim_text, raw_fallback)
                target_premise = p_win if p_win else raw_fallback

            clean_hyp = self._clean_hypothesis_for_nli(claim.claim_text)
            batch_hypotheses.append(clean_hyp)
            batch_premises.append(target_premise)
            batch_meta.append((idx, 0, 1, target_handle, target_premise))

        # 2. Run batch NLI inference for non-meta claims
        if batch_hypotheses:
            try:
                predictions = nli_verifier.predict_batch(
                    claims=batch_hypotheses,
                    premises=batch_premises,
                    batch_size=batch_size,
                )
            except TypeError:
                predictions = nli_verifier.predict_batch(
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

