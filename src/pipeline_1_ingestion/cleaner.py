r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/cleaner.py
   - Role: Text preprocessing, structural header validation, and normalization engine.
   - Purpose: Cleans extracted raw PDF text, fixes broken hyphenations across line
     breaks, normalizes unicode representations (NFKC), and standardizes paragraph
     breaks with structural heading heuristics.

2. INPUT (IP):
   - text (str): Raw string extracted from documents or PDF pages.
   - Source: `src/pipeline_1_ingestion/parser.py` (via `extract_pdf_pages`) or raw text inputs.

3. PROCESS UNDER THE HOOD:
   - Step 1: Unicode Normalization (NFKC): Decomposes and recomposes compatibility
     characters (e.g. ligatures like 'fi' -> 'f' + 'i', full-width Asian characters/brackets).
   - Step 2: Line Endings: Standardizes Windows (\r\n) and classic Mac (\r) linebreaks to Unix (\n).
   - Step 3: De-hyphenation: Regex `r'(\w+)-[ \t]*\n[ \t]*(\w+)'` matches hyphenated words split
     across line boundaries and merges them into cohesive tokens.
   - Step 4: Paragraph Preservation: Splits text on double newlines (`\n\s*\n+`), collapses
     intra-paragraph whitespace/newlines into single spaces (`\s+` -> `' '`), and preserves
     structural headers satisfying alpha ratio (>= 60%), length (3-120 chars), and table coordinate rejection.

4. OUTPUT (OP):
   - str: Cleaned, normalized, and properly spaced text.
   - Consumed by: `src/pipeline_1_ingestion/chunker.py`.

5. LIBRARIES & DEPENDENCIES:
   - unicodedata: Standard library module for Unicode standard character normalization (NFKC).
   - re: Standard library module for high-performance regular expressions.
================================================================================
"""

import re
import unicodedata


def is_structural_header_line(line: str) -> bool:
    """Validate if a line qualifies as a structural heading based on generalized heuristics.

    Args:
        line: Single stripped text line.

    Returns:
        True if line matches structural heading criteria, False otherwise.
    """
    stripped = line.strip()
    if len(stripped) < 3 or len(stripped) > 120:
        return False
    alpha_count = sum(c.isalpha() for c in stripped)
    if (alpha_count / len(stripped)) < 0.6:
        return False
    if re.match(r"^[\d\s\.\,\-\·\/\*\+±×÷eE:]+$", stripped):
        return False
    if re.match(
        r"^[\d\s\.\,\-\·\/\*\+±×÷]+\s*(?:[KkMmGgTt%]|ms|s|V|W|Hz|kHz|MHz|GHz|Pa|kPa|MPa|bit|bytes?|MB|GB|TB|K)?$",
        stripped,
        flags=re.IGNORECASE,
    ):
        return False
    return (
        line.endswith(":")
        or (line.isupper() and len(line.split()) <= 6)
        or (line.istitle() and len(line.split()) <= 6)
        or line.startswith("#")
    )


def clean_text(text: str) -> str:
    """Clean and normalize extracted raw text.

    Steps:
        1. Apply Unicode NFKC normalization.
        2. Standardize carriage returns and line endings.
        3. Rejoin hyphenated words split across newlines (e.g. 'seg-\\nment' -> 'segment').
        4. Normalize paragraph breaks (preserve '\\n\\n' while collapsing intra-paragraph whitespace).

    Args:
        text: Raw text string to be cleaned.

    Returns:
        Cleaned, normalized text string.
    """
    if not text:
        return ""

    # 1. Unicode NFKC normalization
    normalized = unicodedata.normalize("NFKC", text)

    # 2. Standardize line endings
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Resolve hyphenated words broken across newlines
    normalized = re.sub(r"(\w+)-[ \t]*\n[ \t]*(\w+)", r"\1\2", normalized)

    # 4. Split by paragraphs (2 or more newlines)
    raw_paragraphs = re.split(r"\n\s*\n+", normalized)
    cleaned_paragraphs = []

    for para in raw_paragraphs:
        lines = [re.sub(r"[ \t]+", " ", l).strip() for l in para.split("\n") if l.strip()]
        if not lines:
            continue

        # Merge lines within a paragraph if they are continuations of a sentence,
        # but preserve lines that start with bullets or colons or qualify as structural headers
        merged_lines = []
        for line in lines:
            if not merged_lines:
                merged_lines.append(line)
                continue

            prev = merged_lines[-1]
            is_new_item = line.startswith(("•", "-", "*")) or is_structural_header_line(line)
            is_prev_header = is_structural_header_line(prev)

            if is_new_item or is_prev_header:
                merged_lines.append(line)
            else:
                # Merge continuation line with space
                merged_lines[-1] = prev + " " + line

        cleaned_paragraphs.append("\n".join(merged_lines))

    return "\n\n".join(cleaned_paragraphs).strip()
