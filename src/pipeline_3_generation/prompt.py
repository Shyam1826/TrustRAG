r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_3_generation/prompt.py
   - Role: Prompt engineering and context serialization engine.
   - Purpose: Structures user queries and retrieved parent candidates into a strict,
     closed-world XML-delimited prompt enforcing document/section taxonomy strictness,
     strict attribute isolation against associative training hallucinations,
     strict verbatim inventory & entity grounding, verbatim acronym accuracy,
     atomic bullet formatting, and standard ASCII citations.

2. INPUT (IP):
   - query (str): Cleaned user search query from `src/pipeline_2_retrieval/rewriter.py`.
   - contexts (list[RetrievalCandidate]): Top reranked context candidates from
     `src/pipeline_2_retrieval/reranker.py`.

3. PROCESS UNDER THE HOOD:
   - Formats each candidate into an XML document block with sequential citation IDs:
     <document id="Doc-1" doc_id="doc_1" page="1">...</document>
   - Enforces 6 fundamental operational rules:
     * Rule 1 (Strict Semantic Grounding & Closed-World Assumption):
       - Context passages are organized into labeled sections: `[Document: ... | Section: ...]`.
       - Attribute statements only to their explicit sections and source documents.
       - Verbatim Accuracy: Do NOT expand abbreviations or acronyms.
       - Rely strictly and solely on the provided <context>. No outside knowledge.
     * Rule 2 (Strict Attribute Isolation & Anti-Bundle Enforcement):
       - List ONLY specific attributes, libraries, tools, frameworks, and specifications
         that are explicitly written inside the <context> tags for each respective entry.
       - NEVER infer, deduce, or extrapolate unmentioned components from general training knowledge.
     * Rule 3 (Strict Inventory & Entity Grounding):
       - When asked to list or categorize specific entities, technologies, tools, databases, or specifications:
         * Include ONLY items that appear VERBATIM in <context>.
         * Do NOT extrapolate, summarize, or introduce common category companions (e.g., do not output
           PostgreSQL or MySQL unless the exact words 'PostgreSQL' or 'MySQL' exist in the retrieved passages).
         * If a requested entity category has only one matching item in the text, report only that single item.
     * Rule 4 (Structured Atomic Bullets & Verbatim Source Fidelity):
       - Exhaustively list all relevant facts, specifications, or items as concise bullet points.
       - If an item is listed only as a title or name, output ONLY that title verbatim.
     * Rule 5 (ASCII Inline Citations):
       - Append standard ASCII square brackets like [Doc-1] or [Doc-2] to every factual assertion.
       - Forbids Unicode or full-width brackets (【Doc-X】).
     * Rule 6 (Insufficient Evidence):
       - Exact fallback phrase if context is missing information.
   - Assembles the structured sequence: Instructions -> Context -> Query.

4. OUTPUT (OP):
   - str: Formatted prompt string ready for LLM generation.
   - Consumed by: `src/pipeline_3_generation/generator.py`.

5. LIBRARIES & DEPENDENCIES:
   - src.common.schemas.RetrievalCandidate: Strict schema for reranked candidates.
   - typing (List): Type annotations.
================================================================================
"""

from typing import List
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
        "   - Answer strictly using facts explicitly documented in the provided <context>.\n"
        "   - The context passages are organized into labeled sections: `[Document: <doc_id> | Section: <section_name>]`.\n"
        "   - Attribute statements only to their explicit sections and source documents. Do not mix unrelated sections or extrapolate outside knowledge.\n"
        "   - Verbatim Accuracy: Do NOT expand abbreviations or acronyms (e.g., retain 'CBR', 'JWT', 'TDP', 'SLA', 'EBITDA', 'RAWG' exactly as written).\n"
        "   - When answering comparative queries across multiple documents/entities, clearly attribute each fact to its respective document using standard ASCII citations [Doc-X].\n"
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
        "5. (ASCII Inline Citations): You MUST append an inline document citation tag to EVERY factual assertion or bullet. "
        "CRITICAL: You MUST cite sources using ASCII square brackets exactly like [Doc-1] or [Doc-2]. NEVER use full-width or Unicode brackets like 【Doc-1】 or [Doc 1].\n"
        f'6. (Insufficient Evidence): If the <context> does not contain enough information to answer the question with complete certainty, reply EXACTLY with:\n   "{FALLBACK_INSUFFICIENT_INFO}"\n\n'
        "<context>\n"
        f"{joined_context}\n"
        "</context>\n\n"
        f"User Question: {query.strip()}\n\n"
        "Answer (concise bullet points with standard ASCII [Doc-X] citations):"
    )

    return prompt


