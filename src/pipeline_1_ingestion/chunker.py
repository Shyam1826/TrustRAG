r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/chunker.py
   - Role: Structural section-aware document chunker with breadcrumb inheritance.
   - Purpose: Deconstructs normalized page text into coherent, structural ParentChunk
     blocks, dynamically tracks section taxonomy (e.g. "Academic Projects", "Internship",
     "Skills", "Experience"), and prepends contextual breadcrumbs
     `[Document: {doc_id} | Section: {section_name}]` to ensure unambiguous semantic scope
     during retrieval and verification.

2. INPUT (IP):
   - pages (list[dict[str, Any]]): Extracted pages from `src/pipeline_1_ingestion/parser.py`.
   - doc_id (str): Unique document identifier.
   - parent_size (int, optional): Maximum parent character length (default 2500).
   - parent_overlap (int, optional): Parent sliding window overlap (default 200).
   - child_size (int, optional): Fine chunk character length (default 200).
   - overlap (int, optional): Child sliding window overlap (default 50).

3. PROCESS UNDER THE HOOD:
   - Identifies structural headings: ALL-CAPS lines, Markdown headers (`#`/`##`),
     colon headers (`ACADEMIC PROJECTS:`, `INTERNSHIP:`, `EXPERIENCE:`), and standard
     resume/specification taxonomies.
   - Dynamically tracks `current_section` across document pages.
   - Flushes parent chunks when structural boundaries change, when maximum capacity is reached,
     or when page boundaries transition.
   - Inlines contextual breadcrumb `[Document: {doc_id} | Section: {section_name}]` into parent text.
   - Inlines `[Section: {section_name}]` into fine-grained child chunk text for dense/sparse indexing.
   - Assigns sequential integer `chunk_index` to ParentChunk and ChildChunk instances.

4. OUTPUT (OP):
   - tuple[list[ParentChunk], list[ChildChunk]]: Strictly typed Pydantic models with
     `section_name`, `chunk_index`, and inlined breadcrumbs.
   - Consumed by: `src/pipeline_1_ingestion/embedder.py` and `src/pipeline_1_ingestion/indexer.py`.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library module for regex section pattern detection.
   - unicodedata: Standard library Unicode normalization.
   - src.common.config: Provides default chunking parameters.
   - src.common.schemas: Defines `ParentChunk` and `ChildChunk` models.
================================================================================
"""

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

from src.common.config import config
from src.common.schemas import ChildChunk, ParentChunk


class StructuralChunker:
    """Chunks documents along structural section headers with breadcrumb inheritance."""

    @classmethod
    def extract_section_header(cls, line: str) -> Optional[str]:
        """Detect and return cleaned section title if line is a structural heading."""
        stripped = line.strip()
        if not stripped or len(stripped) > 65:
            return None

        # Ignore email header lines or URLs
        if "@" in stripped or stripped.startswith(("http://", "https://", "www.")):
            return None

        # Ignore lines ending in standard sentence punctuation (unless it is a colon)
        if re.search(r"[.!?]\s*$", stripped) and not stripped.endswith(":"):
            return None

        # 1. Markdown headers (# Header)
        md_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if md_match:
            return md_match.group(1).strip()

        # 2. Numbered structural headings (e.g. "1.1 System Architecture", "Section 4: Indemnification")
        num_match = re.match(r"^\d+(?:\.\d+)*\s+([A-Z0-9].*)$", stripped)
        if num_match and len(stripped.split()) <= 7:
            return stripped.rstrip(":")

        sec_match = re.match(
            r"^(?:Section|Article|Chapter|Part|Appendix)\s+[A-Za-z0-9.-]+[:\s]+(.*)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if sec_match and len(stripped.split()) <= 8:
            return stripped.rstrip(":")

        # Clean leading bullets
        clean_s = re.sub(r"^[-*•\s]+", "", stripped).strip()

        # 3. Colon-terminated section headings: ^[A-Z0-9][A-Za-z0-9\s/&,._-]{1,50}:$
        if clean_s.endswith(":") and len(clean_s.split()) <= 6:
            colon_head = clean_s[:-1].strip()
            if len(colon_head) >= 2 and (colon_head[0].isupper() or colon_head[0].isdigit()):
                return colon_head

        # 4. Standalone ALL-CAPS headings (<= 6 words, 4-50 chars, no digits/commas)
        clean_no_punct = clean_s.rstrip(":")
        if (
            len(clean_no_punct.split()) <= 6
            and 4 <= len(clean_no_punct) <= 50
            and not any(c.isdigit() for c in clean_no_punct)
            and "," not in clean_no_punct
            and any(c.isalpha() for c in clean_no_punct)
        ):
            if clean_no_punct.isupper():
                return clean_no_punct

        return None


def create_hierarchical_chunks(
    pages: List[Dict[str, Any]],
    doc_id: str,
    parent_size: int = 2500,
    parent_overlap: int = 200,
    child_size: int = 200,
    overlap: int = 50,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """Create hierarchical parent and child chunks with structural section breadcrumbs.

    Args:
        pages: List of page dictionaries containing 'page_number' and 'raw_text'.
        doc_id: Unique document identifier.
        parent_size: Maximum character length for parent chunks (default: 2500).
        parent_overlap: Overlap allowance for long overflow sections (default: 200).
        child_size: Character length for child chunks (default: 200).
        overlap: Character overlap between consecutive child windows (default: 50).

    Returns:
        A tuple of (parent_chunks, child_chunks).
    """
    parent_chunks: List[ParentChunk] = []
    child_chunks: List[ChildChunk] = []

    doc_parent_index = 0
    doc_child_index = 0
    child_step = max(1, child_size - overlap)
    current_section = "General"

    def _flush_parent(
        section_name: str,
        lines_buffer: List[str],
        page_num: int,
    ) -> None:
        nonlocal doc_parent_index, doc_child_index
        if not lines_buffer:
            return

        raw_parent_body = "\n".join(lines_buffer).strip()
        if not raw_parent_body:
            return

        breadcrumb = f"[Document: {doc_id} | Section: {section_name}]"
        parent_text = f"{breadcrumb}\n{raw_parent_body}"
        parent_id = f"{doc_id}_p{page_num}_{doc_parent_index}"
        parent_child_ids: List[str] = []

        # Slice fine-grained child chunks
        if len(raw_parent_body) <= child_size:
            child_slices = [raw_parent_body]
        else:
            child_slices = [
                raw_parent_body[j : j + child_size].strip()
                for j in range(0, len(raw_parent_body), child_step)
                if raw_parent_body[j : j + child_size].strip()
            ]

        for c_text in child_slices:
            child_id = f"{parent_id}_c{doc_child_index}"
            parent_child_ids.append(child_id)

            inlined_child_text = f"[Section: {section_name}] {c_text}"
            child_chunk = ChildChunk(
                chunk_id=child_id,
                parent_id=parent_id,
                doc_id=doc_id,
                text=inlined_child_text,
                vector=None,
                sparse_tokens=None,
                page_number=page_num,
                chunk_index=doc_child_index,
                section_name=section_name,
            )
            child_chunks.append(child_chunk)
            doc_child_index += 1

        parent_chunk = ParentChunk(
            parent_id=parent_id,
            doc_id=doc_id,
            text=parent_text,
            page_number=page_num,
            child_ids=parent_child_ids,
            chunk_index=doc_parent_index,
            section_name=section_name,
        )
        parent_chunks.append(parent_chunk)
        doc_parent_index += 1

    continuation_words = {"in", "and", "of", "to", "at", "for", "with", "on", "by", "a", "the", "or", "as", "from", "an"}

    for page in pages:
        current_page_num = int(page.get("page_number", 1))
        raw_text = page.get("raw_text") or page.get("text", "")
        if not raw_text:
            continue

        # Normalize unicode and de-hyphenate across lines
        normalized = unicodedata.normalize("NFKC", raw_text).replace("\r\n", "\n").replace("\r", "\n")
        normalized = re.sub(r"(\w+)-[ \t]*\n[ \t]*(\w+)", r"\1\2", normalized)
        page_lines = [l.strip() for l in normalized.split("\n") if l.strip()]

        current_buffer: List[str] = []
        current_buffer_len = 0
        prev_line = ""

        for line in page_lines:
            # Check if previous line was a mid-sentence continuation
            prev_last_word = prev_line.split()[-1].lower() if prev_line.split() else ""
            is_wrapped_continuation = prev_last_word in continuation_words

            detected_header = None if is_wrapped_continuation else StructuralChunker.extract_section_header(line)

            # If a new section boundary is detected and we have buffered content
            if detected_header and detected_header != current_section:
                if current_buffer:
                    _flush_parent(current_section, current_buffer, current_page_num)
                    current_buffer = []
                    current_buffer_len = 0
                current_section = detected_header

            current_buffer.append(line)
            current_buffer_len += len(line) + 1
            prev_line = line

            # If section exceeds parent_size, chunk cleanly along line boundaries with overlap
            if current_buffer_len >= parent_size:
                _flush_parent(current_section, current_buffer, current_page_num)
                # Keep overlap lines
                overlap_lines: List[str] = []
                overlap_len = 0
                for prev_l in reversed(current_buffer):
                    if overlap_len + len(prev_l) > parent_overlap:
                        break
                    overlap_lines.insert(0, prev_l)
                    overlap_len += len(prev_l) + 1

                current_buffer = overlap_lines
                current_buffer_len = overlap_len

        # Flush any remaining content on this page
        if current_buffer:
            _flush_parent(current_section, current_buffer, current_page_num)

    return parent_chunks, child_chunks
