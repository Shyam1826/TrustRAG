r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/conversational_rewriter.py
   - Role: Conversational Query Reformulation & Multi-Turn Coreference Resolution.
   - Purpose: Detects multi-turn conversational follow-ups (pronouns, ellipses,
     demonstratives) and rewrites ambiguous queries into fully-qualified, standalone
     search queries before retrieval. Preserves context ceilings by never injecting
     raw chat transcripts into the RAG candidate payload.

2. INPUT (IP):
   - current_query (str): Latest user input query.
   - chat_history (List[Dict[str, Any]]): Ordered conversational history turns
     containing role ("user" | "assistant") and content strings.

3. PROCESS UNDER THE HOOD:
   - Coreference & Ellipsis Detection (`is_conversational_query`):
     * Scans for pronominal markers ('it', 'they', 'its', 'their', 'this', 'that').
     * Scans for elliptical continuations ('what about', 'how about', 'compare that to',
       'and for', 'why?').
     * Returns False for standalone queries, bypassing unnecessary rewriting.
   - LLM-Assisted Reformulation (`rewrite_query`):
     * Formats recent 2-4 conversational turns into a strict zero-preamble prompt.
     * Invokes active BaseGenerator to resolve referents into a single search query.
   - Deterministic Heuristic Fallback:
     * Operates when generator is offline, mock, or returns fallback strings.
     * Extracts target entity from previous turn via prepositional boundaries and
       capitalized n-grams.
     * Substitutes elliptical entities (e.g. "What about Chase?" -> "What is the governing law in Chase?").
     * Resolves possessive and subjective pronouns (e.g. "its warranty" -> "Model-X processor's warranty").

4. OUTPUT (OP):
   - str: Self-contained, search-ready query string.

5. LIBRARIES & DEPENDENCIES:
   - re: Regular expression compilation and text substitution.
   - typing: Dict, List, Optional, Any.
   - src.pipeline_3_generation.generator (BaseGenerator, MockGenerator).
   - src.pipeline_3_generation.prompt (FALLBACK_INSUFFICIENT_INFO).
================================================================================
"""

import re
from typing import Any, Dict, List, Optional

from src.pipeline_3_generation.generator import BaseGenerator, MockGenerator
from src.pipeline_3_generation.prompt import FALLBACK_INSUFFICIENT_INFO


class ConversationalQueryRewriter:
    """Reformulates multi-turn conversational user queries into standalone search intents."""

    # Explicit conversational markers: pronouns, demonstratives, and elliptical questions
    _PRONOUN_PATTERN = re.compile(
        r"\b(?:it|its|they|them|their|theirs|this|that|these|those)\b",
        re.IGNORECASE,
    )
    _ELLIPTICAL_PATTERNS = [
        re.compile(r"^(?:what\s+about|how\s+about|what\s+of|and\s+for|compare\s+(?:that|this)?\s*to)\b", re.IGNORECASE),
        re.compile(r"^(?:why|how\s+so|how\s+come|what\s+else|tell\s+me\s+more|explain\s+that)\??$", re.IGNORECASE),
        re.compile(r"^(?:and|also|but)\s+(?:what|how|where|when|who|is|are|does|do|can|could)\b", re.IGNORECASE),
        re.compile(r"^(?:and|also|what\s+about|how\s+about)\s+in\s+", re.IGNORECASE),
    ]

    def __init__(self, generator: Optional[BaseGenerator] = None) -> None:
        """Initialize conversational rewriter with optional generator.

        Args:
            generator: Optional BaseGenerator instance for LLM-based query rewriting.
        """
        self.generator = generator

    def is_conversational_query(self, query: str) -> bool:
        """Determine whether the query requires conversational context resolution.

        Args:
            query: Raw user query string.

        Returns:
            True if query contains pronouns, demonstratives, or elliptical follow-ups; False otherwise.
        """
        if not query or not query.strip():
            return False

        q_clean = query.strip()

        # Check for elliptical interrogatives or conversational lead-ins
        for pat in self._ELLIPTICAL_PATTERNS:
            if pat.search(q_clean):
                return True

        # Check for pronouns / demonstratives
        if self._PRONOUN_PATTERN.search(q_clean):
            return True

        # Terse monosyllabic follow-ups
        if q_clean.lower() in ("why?", "why", "how?", "how", "what?", "what", "and?"):
            return True

        return False

    def rewrite_query(
        self,
        current_query: str,
        chat_history: List[Dict[str, Any]],
    ) -> str:
        """Rewrite ambiguous or follow-up query into a fully-qualified standalone search query.

        Args:
            current_query: Latest user query.
            chat_history: Chronological list of message dicts with 'role' and 'content'.

        Returns:
            Fully-qualified standalone query string.
        """
        if not current_query or not current_query.strip():
            return current_query

        clean_query = current_query.strip()

        # If history is empty or query is already standalone, pass through unchanged
        if not chat_history or not self.is_conversational_query(clean_query):
            return clean_query

        # Filter recent relevant turns (up to last 4 messages)
        relevant_turns = [
            turn for turn in chat_history[-6:]
            if isinstance(turn, dict) and turn.get("content") and turn.get("role") in ("user", "assistant")
        ]
        if not relevant_turns:
            return clean_query

        # 1. Attempt LLM-based query reformulation if generator is available and non-mock
        if self.generator and not isinstance(self.generator, MockGenerator):
            try:
                llm_rewritten = self._rewrite_via_llm(clean_query, relevant_turns)
                if llm_rewritten and llm_rewritten != clean_query:
                    return llm_rewritten
            except Exception as e:
                print(f"[ConversationalRewriter] LLM rewrite failed: {e}. Falling back to heuristic rewrite.")

        # 2. Robust heuristic fallback (offline, mock, or fallback execution)
        return self._rewrite_via_heuristics(clean_query, relevant_turns)

    def _rewrite_via_llm(
        self,
        current_query: str,
        turns: List[Dict[str, Any]],
    ) -> Optional[str]:
        """Synthesize rewritten query using active LLM generator."""
        history_lines: List[str] = []
        for t in turns[-4:]:
            role = "User" if t.get("role") == "user" else "Assistant"
            content = str(t.get("content", "")).strip().replace("\n", " ")
            history_lines.append(f"{role}: {content}")

        history_str = "\n".join(history_lines)
        prompt = (
            "You are a conversational search query reformulation expert.\n"
            "Given the conversation history, rewrite the latest user follow-up query into a single, "
            "completely standalone search query that resolves all pronouns (it, they, its, their, this, that) "
            "and elliptical questions (e.g., 'What about X?', 'And for Y?').\n"
            "CRITICAL: Output ONLY the rewritten standalone search query. Do NOT add explanations, quotes, or preambles.\n\n"
            f"Conversation History:\n{history_str}\n\n"
            f"Latest Follow-up Query: {current_query}\n\n"
            "Standalone Search Query:"
        )

        assert self.generator is not None
        raw_response = self.generator.generate(prompt)
        if not raw_response or raw_response == FALLBACK_INSUFFICIENT_INFO:
            return None

        # Clean output
        rewritten = raw_response.strip().split("\n")[0].strip(' "\'')
        # Strip potential label echo
        rewritten = re.sub(r"^(?:standalone\s+search\s+query|rewritten\s+query):\s*", "", rewritten, flags=re.IGNORECASE)
        return rewritten if rewritten and len(rewritten) >= 5 else None

    def _rewrite_via_heuristics(
        self,
        current_query: str,
        turns: List[Dict[str, Any]],
    ) -> str:
        """Deterministic heuristic query reformulation based on syntax and entity substitution."""
        # Locate the most recent user turn
        prev_user_q: Optional[str] = None
        prev_assistant_resp: Optional[str] = None

        for turn in reversed(turns):
            if turn.get("role") == "user" and prev_user_q is None:
                prev_user_q = str(turn.get("content", "")).strip()
            elif turn.get("role") == "assistant" and prev_assistant_resp is None:
                prev_assistant_resp = str(turn.get("content", "")).strip()

        if not prev_user_q:
            return current_query

        # ----------------------------------------------------------------------
        # Pattern A: Elliptical Entity Substitution
        # e.g., Turn 1: "What is the governing law in LinkPlus?"
        #       Turn 2: "What about Chase?" -> "What is the governing law in Chase?"
        # ----------------------------------------------------------------------
        elliptical_match = re.search(
            r"^(?:what\s+about|how\s+about|what\s+of|and\s+for|compare\s+(?:that|this)?\s*to)\s+(?P<new_entity>.+?)\??$",
            current_query,
            re.IGNORECASE,
        )
        if elliptical_match:
            new_entity = elliptical_match.group("new_entity").strip().rstrip("?.")

            # 1. Preposition-bound entity replacement in prev_user_q
            prep_match = re.search(
                r"\b(?P<prep>in|for|of|regarding|at|across|with)\s+(?:the\s+)?(?P<old_entity>[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,2})",
                prev_user_q,
                re.IGNORECASE,
            )
            if prep_match:
                prep = prep_match.group("prep")
                old_span = prep_match.group(0)
                # Replace old prepositional span with new entity
                rewritten = prev_user_q.replace(old_span, f"{prep} {new_entity}").strip()
                if not rewritten.endswith("?"):
                    rewritten += "?"
                return rewritten

            # 2. Possessive entity replacement in prev_user_q
            possessive_match = re.search(r"\b(?P<old_entity>[A-Za-z0-9_.-]+)'s\b", prev_user_q)
            if possessive_match:
                old_possessive = possessive_match.group(0)
                rewritten = prev_user_q.replace(old_possessive, f"{new_entity}'s").strip()
                if not rewritten.endswith("?"):
                    rewritten += "?"
                return rewritten

            # 3. Direct question inheritance
            base_q = prev_user_q.rstrip("?. ")
            return f"{base_q} for {new_entity}?"

        # ----------------------------------------------------------------------
        # Pattern B: Pronoun / Demonstrative Coreference Resolution
        # e.g., Turn 1: "Can you tell me what is the TDP wattage of Model-X processor?"
        #       Turn 2: "What is its warranty period?" -> "What is Model-X processor's warranty period?"
        # ----------------------------------------------------------------------
        if self._PRONOUN_PATTERN.search(current_query):
            entity = self._extract_prominent_entity(prev_user_q, prev_assistant_resp)
            if entity:
                rewritten = current_query

                # Replace possessive pronouns (its, their)
                rewritten = re.sub(
                    r"\b(?:its|their)\b",
                    f"{entity}'s",
                    rewritten,
                    flags=re.IGNORECASE,
                )

                # Replace subjective / objective pronouns (it, they, this, that)
                rewritten = re.sub(
                    r"\b(?:it|they|this|that)\b",
                    f"the {entity}",
                    rewritten,
                    flags=re.IGNORECASE,
                )

                return rewritten.strip()

        # ----------------------------------------------------------------------
        # Pattern C: Terse monosyllabic inquiries ("Why?", "How so?")
        # ----------------------------------------------------------------------
        if current_query.strip().lower() in ("why?", "why", "how so?", "how come?"):
            entity = self._extract_prominent_entity(prev_user_q, prev_assistant_resp)
            base_clean = re.sub(r"^(?:what\s+(?:is|are|was|were)|can\s+you\s+tell\s+me)\s*", "", prev_user_q, flags=re.IGNORECASE).rstrip("?. ")
            if entity:
                return f"Why is {base_clean} regarding {entity}?"
            return f"Why {base_clean}?"

        return current_query

    def _extract_prominent_entity(
        self,
        prev_user_q: str,
        prev_assistant_resp: Optional[str] = None,
    ) -> Optional[str]:
        """Extract primary target entity noun phrase from the prior conversation turn."""
        # 1. Prepositional phrase extraction from user query (e.g. "of Model-X processor", "in LinkPlus")
        prep_match = re.search(
            r"\b(?:of|for|in|about|regarding)\s+(?:the\s+)?(?P<entity>[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+){0,2})",
            prev_user_q,
            re.IGNORECASE,
        )
        if prep_match:
            cand = prep_match.group("entity").strip().rstrip("?.")
            stopwords = {"what", "which", "where", "how", "who", "why", "the", "a", "an", "all", "each"}
            if cand.lower() not in stopwords:
                return cand

        # 2. Capitalized noun phrases / Proper nouns from user query
        cap_candidates = re.findall(
            r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Za-z0-9_-]+)*\b",
            prev_user_q,
        )
        filter_words = {"What", "How", "Can", "Could", "Which", "Who", "Where", "When", "Why", "The", "Is", "Are"}
        valid_caps = [c for c in cap_candidates if c not in filter_words]
        if valid_caps:
            return valid_caps[-1]

        # 3. Inspect previous assistant response for strong entity mentions
        if prev_assistant_resp:
            resp_caps = re.findall(
                r"\b[A-Z][a-zA-Z0-9_-]+(?:\s+[A-Za-z0-9_-]+)*\b",
                prev_assistant_resp,
            )
            valid_resp_caps = [c for c in resp_caps if c not in filter_words and len(c) >= 3]
            if valid_resp_caps:
                return valid_resp_caps[0]

        return None
