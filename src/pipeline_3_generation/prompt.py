r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_3_generation/prompt.py
   - Role: Prompt engineering, context serialization, and corrective rewrite engine.
   - Purpose: Structures user queries and retrieved parent candidates into a strict,
     closed-world XML-delimited prompt enforcing document/section taxonomy strictness,
     zero parametric memory leakage, positive-facts-only assertion rules, prohibition
     of negative meta-commentary, strict attribute isolation, verbatim inventory fidelity,
     and standard ASCII citations. Also provides `build_correction_prompt` for automated
     1-pass self-correction loops with grounded partial-coverage fallback semantics.

2. INPUT (IP):
   - query (str): Cleaned user search query from `src/pipeline_2_retrieval/rewriter.py`.
   - contexts (list[RetrievalCandidate] | str): Top reranked context candidates from
     `src/pipeline_2_retrieval/reranker.py`.
   - draft (str): Previous draft response for self-correction.
   - failed_claims (list[str]): List of unverified or contradictory claims to remove/correct.

3. PROCESS UNDER THE HOOD:
   - Formats each candidate into an XML document block with sequential citation IDs:
     <document id="Doc-1" doc_id="doc_1" page="1">...</document>
   - Enforces 6 fundamental operational rules:
     * Rule 1 (Strict Semantic Grounding & Closed-World Assumption):
       - Context passages are organized into labeled sections: `[Document: ... | Section: ...]`.
       - LLM acts as an evidence-grounded synthesizer with ZERO external parametric memory.
       - State ONLY positive facts explicitly asserted in <context>.
       - NEVER output negative meta-commentary about missing facts (do not state "The text does not mention X").
       - Attribute statements only to their explicit sections and source documents.
       - Verbatim Accuracy: Do NOT expand abbreviations or acronyms.
       - Structure comparative and cross-document questions as parallel factual assertions with explicit [Doc-X] tags.
     * Rule 2 (Strict Attribute Isolation & Anti-Bundle Enforcement):
       - List ONLY specific attributes, libraries, tools, frameworks, and specifications
         that are explicitly written inside the <context> tags for each respective entry.
       - NEVER infer, deduce, or extrapolate unmentioned components from general training knowledge.
     * Rule 3 (Strict Inventory & Entity Grounding):
       - When asked to list or categorize specific entities, technologies, tools, databases, or specifications:
         * Include ONLY items that appear VERBATIM in <context>.
         * Do NOT extrapolate, summarize, or introduce common category companions.
         * If a requested entity category has only one matching item in the text, report only that single item.
     * Rule 4 (Structured Atomic Bullets & Verbatim Source Fidelity):
       - Exhaustively list all relevant facts, specifications, or items as concise bullet points.
       - If an item is listed only as a title or name, output ONLY that title verbatim.
     * Rule 5 (ASCII Inline Citations):
       - Append standard ASCII square brackets like [Doc-1] or [Doc-2] to every factual assertion and comparative clause.
       - Forbids Unicode or full-width brackets (【Doc-X】).
     * Rule 6 (Grounded Partial Coverage & Strict Fallback):
       - Answers as much of the query as can be directly and affirmatively proven from <context>.
       - Replaces all-or-nothing blockage: synthesizes supported sub-questions instead of discarding all facts.
       - Emits fallback string ONLY if zero relevant information exists across all provided context.
   - `build_correction_prompt()`: Generates a corrective prompt passing the failed claims, draft,
     and context to produce a 100% verified rewrite without throwing away validly grounded points.

4. OUTPUT (OP):
   - str: Formatted prompt string ready for LLM generation or self-correction.
   - Consumed by: `src/pipeline_3_generation/generator.py` and `src/main.py`.

5. LIBRARIES & DEPENDENCIES:
   - src.common.schemas.RetrievalCandidate: Strict schema for reranked candidates.
   - typing (List, Union): Type annotations.
================================================================================
"""

from typing import List, Union
from src.common.schemas import RetrievalCandidate

FALLBACK_INSUFFICIENT_INFO = (
    "The provided documentation does not contain sufficient information to answer."
)


def build_rag_prompt(query: str, contexts: List[RetrievalCandidate]) -> str:
    """Construct an XML-delimited, taxonomy-strict RAG prompt.

    Args:
        query: Target user query.
        contexts: List of top reranked RetrievalCandidate passages.

    Returns:
        Structured prompt string.
    """
    # 1. Format context documents
    context_blocks: List[str] = []
    for idx, candidate in enumerate(contexts, start=1):
        doc_xml = (
            f'  <document id="Doc-{idx}" doc_id="{candidate.doc_id}" page="{candidate.page_number}">\n'
            f"    {candidate.text.strip()}\n"
            f"  </document>"
        )
        context_blocks.append(doc_xml)

    joined_context = "\n".join(context_blocks)

    # 2. Assemble prompt template
    prompt = (
        "You are an enterprise AI assistant adhering to strict verification, taxonomy, and truthfulness standards.\n\n"
        "### OPERATIONAL RULES:\n"
        "1. (Strict Semantic Grounding & Closed-World Assumption):\n"
        "   - You are operating under a strict Closed-World Assumption (CWA) with ZERO external parametric memory.\n"
        "   - Answer strictly and solely using positive facts explicitly documented in the provided <context>.\n"
        "   - State ONLY positive facts that are explicitly asserted in the context passages.\n"
        "   - Prohibit meta-commentary: Do NOT output statements about what is missing or unmentioned (e.g., do NOT write 'The text does not state X' or 'No information is provided about Y').\n"
        "   - The context passages are organized into labeled sections: `[Document: <doc_id> | Section: <section_name>]`.\n"
        "   - Attribute statements only to their explicit sections and source documents. Do not mix unrelated sections or extrapolate outside knowledge.\n"
        "   - Verbatim Accuracy: Do NOT expand abbreviations or acronyms (e.g., retain 'CBR', 'JWT', 'TDP', 'SLA', 'EBITDA', 'RAWG' exactly as written).\n"
        "   - When answering comparative or multi-document questions:\n"
        "     * Structure comparisons as direct, parallel factual statements attributed explicitly to their source documents.\n"
        "     * Every distinct comparative clause MUST have its own inline [Doc-X] citation tag immediately following the assertion.\n"
        "     * Avoid ungrounded contrastive meta-commentary about what is NOT present in the other document; state ONLY the positive documented facts for each source.\n"
        "2. (Strict Attribute Isolation & Anti-Bundle Enforcement):\n"
        "   - You must list ONLY the specific attributes, libraries, tools, frameworks, and specifications that are explicitly written inside the <context> tags for each respective entry.\n"
        "   - NEVER infer, deduce, or extrapolate unmentioned components from general training knowledge (for example, do not auto-complete a single mentioned library into a full multi-tier stack, suite, or architecture unless every single component is explicitly named in that specific source passage).\n"
        "   - If an architecture, backend, or operational parameter is not explicitly detailed in the text, leave it unmentioned—do NOT fill gaps using common industry conventions.\n"
        "3. (Strict Inventory & Entity Grounding):\n"
        "   - When asked to list or categorize specific entities, technologies, tools, databases, or specifications across documents:\n"
        "     * Include ONLY items that appear VERBATIM in <context>.\n"
        "     * Do NOT extrapolate, summarize, or introduce common category companions (e.g., do not output PostgreSQL or MySQL unless the exact words 'PostgreSQL' or 'MySQL' exist in the retrieved passages).\n"
        "     * If a requested entity category has only one matching item in the text, report only that single item.\n"
        "4. (Structured Atomic Bullets & Verbatim Source Fidelity):\n"
        "   - Exhaustively list all relevant facts, specifications, items, or properties mentioned in <context>.\n"
        "   - Format each distinct fact as a concise bullet point, reproducing names and explicit details directly from <context>.\n"
        "   - If an item in <context> is listed only as a title, name, or short phrase, output ONLY that title or phrase verbatim (e.g. `- Item Name [Doc-1]`). Do NOT invent parenthetical explanations, definitions, or ungrounded commentary.\n"
        "5. (ASCII Inline Citations & Critical Citation Fidelity):\n"
        "   - You MUST append an inline document citation tag to EVERY factual assertion, bullet, or comparative clause.\n"
        "   - Every distinct comparative clause MUST have its own inline [Doc-X] citation tag immediately following the assertion.\n"
        "   - CRITICAL CITATION FIDELITY: You must cite ONLY the specific [Doc-X] whose enclosed text body explicitly contains the stated assertion. Do NOT cite a [Doc-X] block if the specific facts, formulas, or tools are not present within its explicit XML content, even if it shares the same doc_id or section name.\n"
        "   - CRITICAL: You MUST cite sources using standard ASCII square brackets exactly like [Doc-1] or [Doc-2]. NEVER use full-width or Unicode brackets like 【Doc-1】 or [Doc 1].\n"
        "6. (Insufficient Evidence & Fallback):\n"
        "   - Answer as much of the query as can be directly and affirmatively proven from the provided <context>, citing each assertion with [Doc-X].\n"
        "   - If a specific sub-question, document, or entity has direct evidence in <context>, report it concisely.\n"
        f'   - ONLY if the provided <context> contains ZERO relevant information to answer any part of the query, reply EXACTLY with:\n      "{FALLBACK_INSUFFICIENT_INFO}"\n\n'
        "<context>\n"
        f"{joined_context}\n"
        "</context>\n\n"
        f"User Question: {query.strip()}\n\n"
        "Answer (concise bullet points with standard ASCII [Doc-X] citations):"
    )

    return prompt


def build_correction_prompt(
    query: str,
    context: Union[str, List[RetrievalCandidate]],
    draft: str,
    failed_claims: List[str],
) -> str:
    """Construct an automated corrective rewrite prompt instructing the model to remove unverified claims.

    Args:
        query: Original user query.
        context: Retrieved context text or list of RetrievalCandidate instances.
        draft: Initial synthesized draft response containing unverified assertions.
        failed_claims: List of specific claim strings that failed NLI verification.

    Returns:
        Structured self-correction prompt string.
    """
    if isinstance(context, list):
        context_blocks = []
        for idx, candidate in enumerate(context, start=1):
            doc_xml = (
                f'  <document id="Doc-{idx}" doc_id="{candidate.doc_id}" page="{candidate.page_number}">\n'
                f"    {candidate.text.strip()}\n"
                f"  </document>"
            )
            context_blocks.append(doc_xml)
        joined_context = "\n".join(context_blocks)
    else:
        joined_context = str(context).strip()

    failed_list_str = "\n".join(f"- {c}" for c in failed_claims) if failed_claims else "- None"

    prompt = (
        "You are an enterprise AI verification and correction assistant.\n"
        "Your previous draft response contained ungrounded, unverified, or contradictory assertions that failed NLI verification.\n\n"
        "### CLOSED-WORLD REWRITE INSTRUCTIONS:\n"
        "1. Rewrite the draft response to be 100% faithful and strictly entailed by the provided <context>.\n"
        "2. REMOVE or CORRECT the following failed/unverified claims:\n"
        f"{failed_list_str}\n"
        "3. (Strict Semantic Grounding & Closed-World Assumption):\n"
        "   - Include ONLY positive facts explicitly written in the <context>.\n"
        "   - Do NOT include any unmentioned tools, libraries, frameworks, or specifications.\n"
        "   - Do NOT output meta-commentary or negative statements about missing data.\n"
        "4. (ASCII Inline Citations & Critical Citation Fidelity):\n"
        "   - Every asserted fact must have an inline citation tag (e.g. [Doc-1]).\n"
        "   - CRITICAL CITATION FIDELITY: You must cite ONLY the specific [Doc-X] whose enclosed text body explicitly contains the stated assertion. Do NOT cite a [Doc-X] block if the specific facts, formulas, or tools are not present within its explicit XML content, even if it shares the same doc_id or section name.\n"
        f'5. (Insufficient Evidence): ONLY if after removing the ungrounded assertions ZERO verifiable information remains, reply EXACTLY with:\n   "{FALLBACK_INSUFFICIENT_INFO}"\n\n'
        "<context>\n"
        f"{joined_context}\n"
        "</context>\n\n"
        f"Original User Question: {query.strip()}\n\n"
        f"Previous Draft Response:\n{draft.strip()}\n\n"
        "Corrected Answer (100% verified bullet points with standard ASCII [Doc-X] citations):"
    )

    return prompt