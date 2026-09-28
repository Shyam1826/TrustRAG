r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/chunker.py
   - Role: Domain-agnostic structural section-aware chunker with breadcrumb inheritance.
   - Purpose: Deconstructs normalized text from any domain (legal contracts, financial filings,
     technical specifications, medical literature, resumes) into coherent ParentChunk blocks,
     dynamically tracks structural section taxonomy using generalized heading heuristics,
     attaches section headings directly to subsequent body text as Markdown prefixes, performs
     dynamic sentence-boundary child chunk slicing, and prepends contextual breadcrumbs
     `[Document: {doc_id} | Section: {section_name}]` to ensure unambiguous semantic grounding
     during retrieval and verification without emitting empty header-stub chunks or truncated clauses.

2. INPUT (IP):
   - pages (list[dict[str, Any]]): Extracted pages from `src/pipeline_1_ingestion/parser.py`.
   - doc_id (str): Unique document identifier.
   - parent_size (int, optional): Maximum parent character length (from config).
   - parent_overlap (int, optional): Parent sliding window overlap (from config).
   - child_size (int, optional): Fine chunk character length (from config).
   - overlap (int, optional): Child sliding window overlap (from config).

3. PROCESS UNDER THE HOOD:
   - Evaluates candidate headings against structural validity criteria (`is_valid_section_name`):
     * Length constraint: `3 <= len(stripped) <= 120`.
     * Alphabetic character ratio: `>= 60%` letters (`sum(c.isalpha() for c in s) / len(s) >= 0.6`).
     * Rejection of numerical/scientific table coordinates and standalone measurements (`300K`, `1.2 · 1021`).
   - Structural Heading Attachment: Section titles are NEVER emitted as standalone isolated chunks.
     Instead, identified headings are preserved in `pending_heading` state and prefixed directly to the
     subsequent body text as `## {heading}\n{body_text}`.
   - Boundary-Aware Syntactic Chunking: Replaces fixed-character windowing with recursive boundary degradation:
     * Priority 1: Structural Headings / Section Delimiters (`\n\n## `, `\n\n`).
     * Priority 2: Atomic Record / List Item Boundaries (`\n[-*•]`, `\n[A-Za-z0-9_]+:`, `\n`).
     * Priority 3: Sentence Terminators (`(?<=[.!?])\s+(?=[A-Z0-9])`).
     * Priority 4: Word Boundaries (`\s+`).
     * Hard Rule: Never splits across an atomic entity record (table row, bullet item, key-value pair)
       unless that single record independently exceeds the maximum child character ceiling.
   - Minimum Chunk Size Threshold (15 words / ~100 chars): Any parent or child block containing fewer than 15 words
     is automatically merged into the adjacent chunk (or carried forward), ensuring zero isolated stub
     chunks enter Qdrant or downstream vector indexes.
   - Inlines contextual breadcrumb `[Document: {doc_id} | Section: {section_name}]` into parent text.
   - Inlines `[Section: {section_name}]` into fine-grained child chunk text for dense/sparse indexing.
   - Assigns sequential integer `chunk_index` to ParentChunk and ChildChunk instances.

4. OUTPUT (OP):
   - tuple[list[ParentChunk], list[ChildChunk]]: Strictly typed Pydantic models with
     `section_name`, `chunk_index`, inlined breadcrumbs, and guaranteed substantive content (>=15 words).
   - Consumed by: `src/pipeline_1_ingestion/embedder.py` and `src/main.py`.

5. LIBRARIES & DEPENDENCIES:
   - re: Standard library module for regex section pattern detection and boundary slicing.
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


def _count_words_chunk(text: str) -> int:
    """Count words excluding Markdown heading hashes."""
    clean = re.sub(r"^#+\s*", "", text.strip())
    return len(clean.split())


def _split_into_atomic_units(text: str, max_unit_size: int) -> List[str]:
    """Recursively degrade text into atomic syntactic units across boundary hierarchies.

    Priority 1: Structural Headings / Section Delimiters (\\n\\n## , \\n\\n)
    Priority 2: Atomic Record / List Item Boundaries (\\n(?=[-*•]), \\n(?=[A-Za-z0-9_]+:), \\n)
    Priority 3: Sentence Terminators ((?<=[.!?])\\s+(?=[A-Z0-9]) or (?<=[.!?])\\s+)
    Priority 4: Word Boundaries (\\s+)
    """
    if not text or not text.strip():
        return []

    stripped = text.strip()
    if len(stripped) <= max_unit_size:
        return [stripped]

    # Priority 1: Split on Structural Headings and Section Delimiters (\n\n## , \n\n)
    p1_parts = [p.strip() for p in re.split(r"(?:\n\n## |\n\n+)", stripped) if p.strip()]
    if len(p1_parts) > 1:
        units: List[str] = []
        for part in p1_parts:
            units.extend(_split_into_atomic_units(part, max_unit_size))
        return units

    # Priority 2: Split on Atomic Record / List Item / Key-Value Boundaries
    # (\n(?=[-*•]), \n(?=[A-Za-z0-9_]+:), \n)
    p2_parts = [p.strip() for p in re.split(r"\n(?=[-*•])|\n(?=[A-Za-z0-9_]+:)|\n+", stripped) if p.strip()]
    if len(p2_parts) > 1:
        units = []
        for part in p2_parts:
            units.extend(_split_into_atomic_units(part, max_unit_size))
        return units

    # Priority 3: Split on Sentence Terminators (?<=[.!?])\s+(?=[A-Z0-9]) or (?<=[.!?])\s+
    p3_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])|(?<=[.!?])\s+", stripped) if p.strip()]
    if len(p3_parts) > 1:
        units = []
        for part in p3_parts:
            units.extend(_split_into_atomic_units(part, max_unit_size))
        return units

    # Priority 4: Word Boundaries (\s+) for exceptionally long unbroken single clauses
    p4_parts = [p.strip() for p in re.split(r"\s+", stripped) if p.strip()]
    if len(p4_parts) > 1:
        accumulated: List[str] = []
        curr = ""
        for w in p4_parts:
            if not curr:
                curr = w
            elif len(curr) + 1 + len(w) <= max_unit_size:
                curr += " " + w
            else:
                accumulated.append(curr)
                curr = w
        if curr:
            accumulated.append(curr)
        return accumulated

    return [stripped]


def slice_text_dynamically(text: str, c_size: int, c_overlap: int) -> List[str]:
    """Slice text into overlapping child chunks using recursive boundary degradation.

    Deconstructs parent text into atomic syntactic units across boundary priorities:
    - Priority 1: Structural Headings / Section Delimiters (`\\n\\n## `, `\\n\\n`)
    - Priority 2: Atomic Record / List Item Boundaries (`\\n[-*•]`, `\\n[A-Za-z0-9_]+:`, `\\n`)
    - Priority 3: Sentence Terminators (`(?<=[.!?])\\s+(?=[A-Z0-9])`)
    - Priority 4: Word Boundaries (`\\s+`)

    Hard Rule: Never splits across an atomic entity record unless that single record
    independently exceeds the child size ceiling. Enforces minimum chunk floor (>=15 words / ~100 chars),
    merging trailing fragments into the preceding chunk.

    Args:
        text: Raw parent text block to slice.
        c_size: Target child chunk size in characters.
        c_overlap: Overlap allowance in characters.

    Returns:
        List of cleanly bounded text slices.
    """
    stripped = text.strip()
    if not stripped or len(stripped) <= c_size:
        return [stripped] if stripped else []

    atomic_units = _split_into_atomic_units(stripped, max_unit_size=c_size)
    if not atomic_units:
        return [stripped]

    slices: List[str] = []
    current_units: List[str] = []
    current_len = 0

    for unit in atomic_units:
        sep_len = 1 if current_units else 0
        if current_len + sep_len + len(unit) <= c_size:
            current_units.append(unit)
            current_len += sep_len + len(unit)
        else:
            if current_units:
                chunk_str = " ".join(current_units).strip()
                slices.append(chunk_str)

                # Overlap: retain trailing atomic units up to c_overlap
                if c_overlap > 0:
                    overlap_units: List[str] = []
                    overlap_len = 0
                    for u in reversed(current_units):
                        u_len = len(u) + (1 if overlap_units else 0)
                        if overlap_len + u_len <= c_overlap:
                            overlap_units.insert(0, u)
                            overlap_len += u_len
                        else:
                            break
                    current_units = list(overlap_units)
                    current_len = sum(len(u) for u in current_units) + (len(current_units) - 1 if current_units else 0)
                else:
                    current_units = []
                    current_len = 0

            # Add the current unit
            current_units.append(unit)
            current_len += (1 if current_len > 0 else 0) + len(unit)

    if current_units:
        chunk_str = " ".join(current_units).strip()
        if chunk_str:
            slices.append(chunk_str)

    # Enforce minimum chunk floor (>= 15 words or >= 100 characters):
    # Merge any short trailing fragment into previous slice
    if len(slices) > 1:
        last_slice = slices[-1]
        last_words = _count_words_chunk(last_slice)
        if (last_words < 15 and len(last_slice) < 100) or last_words < 5:
            slices[-2] = f"{slices[-2]} {last_slice}".strip()
            slices.pop()

    return slices


class StructuralChunker:
    """Chunks documents along structural section headers with breadcrumb inheritance."""

    @staticmethod
    def is_valid_section_name(text: str) -> bool:
        """Validate candidate section name against generalized structural criteria.

        Criteria:
            1. 3 <= len(text.strip()) <= 120.
            2. At least 60% alphabetic/Unicode letter characters (sum(c.isalpha() for c in s) / len(s) >= 0.6).
            3. Reject pure numerical/scientific table coordinates and standalone measurements
               (e.g., regex matching pure digits/punct or lone table tokens like "300K", "1.2 · 1021").

        Args:
            text: Candidate section title string.

        Returns:
            True if text qualifies as a valid structural section heading, False otherwise.
        """
        if not text:
            return False
        stripped = text.strip()

        # Criterion 1: Length bounds between 3 and 120 characters
        if len(stripped) < 3 or len(stripped) > 120:
            return False

        # Criterion 2: At least 60% alphabetic / Unicode letter characters
        alpha_count = sum(c.isalpha() for c in stripped)
        if (alpha_count / len(stripped)) < 0.6:
            return False

        # Criterion 3: Reject pure numerical/scientific coordinates, formulas, or standalone table measurements
        if re.match(r"^[\d\s\.\,\-\·\/\*\+±×÷eE:]+$", stripped):
            return False

        if re.match(
            r"^[\d\s\.\,\-\·\/\*\+±×÷]+\s*(?:[KkMmGgTt%]|ms|s|V|W|Hz|kHz|MHz|GHz|Pa|kPa|MPa|bit|bytes?|MB|GB|TB|K)?$",
            stripped,
            flags=re.IGNORECASE,
        ):
            return False

        return True

    @classmethod
    def extract_section_header(cls, line: str) -> Optional[str]:
        """Detect and return cleaned section title if line satisfies structural heading heuristics.

        Args:
            line: Raw line of text from document.

        Returns:
            Cleaned valid section header string, or None if line is body text / invalid header.
        """
        stripped = line.strip()
        if not stripped or len(stripped) > 120:
            return None

        # Ignore email header lines or URLs
        if "@" in stripped or stripped.startswith(("http://", "https://", "www.")):
            return None

        # Ignore lines ending in standard sentence punctuation (unless colon)
        if re.search(r"[.!?]\s*$", stripped) and not stripped.endswith(":"):
            return None

        candidate_header: Optional[str] = None

        # 1. Markdown headings (# Header)
        md_match = re.match(r"^#{1,6}\s+(.+)$", stripped)
        if md_match:
            candidate_header = md_match.group(1).strip()

        # 2. Numbered legal/technical clauses (e.g. "Section 4.1", "Article II", "1.2 Architecture", "(a) Scope")
        elif re.match(
            r"^(?:Section|Article|Chapter|Part|Exhibit|Schedule|Appendix|\d+(?:\.\d+)+|\([a-z\d]+\))\s+.*$",
            stripped,
            flags=re.IGNORECASE,
        ) and len(stripped.split()) <= 12:
            candidate_header = stripped.rstrip(":")

        elif re.match(r"^\d+(?:\.\d+)*\s+([A-Z0-9].*)$", stripped) and len(stripped.split()) <= 8:
            candidate_header = stripped.rstrip(":")

        else:
            # Clean leading bullets
            clean_s = re.sub(r"^[-*•\s]+", "", stripped).strip()

            # 3. Colon-terminated section headings: ^[A-Z0-9][A-Za-z0-9\s/&,._-]{1,120}:$
            if clean_s.endswith(":") and len(clean_s.split()) <= 10:
                colon_head = clean_s[:-1].strip()
                if len(colon_head) >= 2 and (colon_head[0].isupper() or colon_head[0].isdigit()):
                    candidate_header = colon_head

            # 4. Standalone document headers / uppercase / title case headers
            elif (
                len(clean_s.split()) <= 10
                and 3 <= len(clean_s) <= 120
                and (clean_s.isupper() or clean_s.istitle() or re.match(r"^[A-Z0-9\s\-_/&,]{3,120}$", clean_s))
            ):
                candidate_header = clean_s.rstrip(":")

        if candidate_header and cls.is_valid_section_name(candidate_header):
            return candidate_header

        return None


def create_hierarchical_chunks(
    pages: List[Dict[str, Any]],
    doc_id: str,
    parent_size: Optional[int] = None,
    parent_overlap: Optional[int] = None,
    child_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> Tuple[List[ParentChunk], List[ChildChunk]]:
    """Create hierarchical parent and child chunks with structural section breadcrumbs.

    Enforces that section headings are attached as Markdown prefixes (f"## {heading}\n{body}")
    to subsequent body paragraphs, never emitted as standalone chunks. Performs dynamic
    sentence-boundary slicing for child chunks. Merges any fragment with fewer than 15 words
    into adjacent chunks to guarantee no isolated stub chunks exist.

    Args:
        pages: List of page dictionaries containing 'page_number' and 'raw_text'.
        doc_id: Unique document identifier.
        parent_size: Maximum character length for parent chunks (default from config).
        parent_overlap: Overlap allowance for long overflow sections (default from config).
        child_size: Character length for child chunks (default from config).
        overlap: Character overlap between consecutive child windows (default from config).

    Returns:
        A tuple of (parent_chunks, child_chunks).
    """
    p_size = parent_size if parent_size is not None else config.chunking.parent_size
    p_overlap = parent_overlap if parent_overlap is not None else config.chunking.parent_overlap
    c_size = child_size if child_size is not None else config.chunking.child_size
    c_overlap = overlap if overlap is not None else config.chunking.overlap

    parent_chunks: List[ParentChunk] = []
    child_chunks: List[ChildChunk] = []

    doc_parent_index = 0
    doc_child_index = 0

    current_section = "General"
    pending_heading: Optional[str] = None
    current_buffer: List[str] = []
    current_buffer_len = 0

    def _count_words(text: str) -> int:
        clean = re.sub(r"^#+\s*", "", text.strip())
        return len(clean.split())

    def _flush_parent(
        section_name: str,
        lines_buffer: List[str],
        page_num: int,
        is_final: bool = False,
    ) -> bool:
        """Flush lines_buffer into parent and child chunks.

        Returns:
            True if flushed or merged, False if buffer carried forward.
        """
        nonlocal doc_parent_index, doc_child_index
        if not lines_buffer:
            return True

        raw_parent_body = "\n".join(lines_buffer).strip()
        if not raw_parent_body:
            return True

        word_cnt = _count_words(raw_parent_body)

        # Minimum chunk size threshold: 15 words
        if word_cnt < 15:
            if parent_chunks:
                # Merge into preceding parent chunk and child chunk
                parent_chunks[-1].text = f"{parent_chunks[-1].text}\n{raw_parent_body}"
                child_chunks[-1].text = f"{child_chunks[-1].text}\n{raw_parent_body}"
                return True
            elif not is_final:
                # Carry forward into next buffer
                return False
            # If is_final and no prior parent_chunks, emit single sole document chunk below

        breadcrumb = f"[Document: {doc_id} | Section: {section_name}]"
        parent_text = f"{breadcrumb}\n{raw_parent_body}"
        parent_id = f"{doc_id}_p{page_num}_{doc_parent_index}"
        parent_child_ids: List[str] = []

        # Dynamic sentence-boundary slicing for fine-grained child chunks
        raw_slices = slice_text_dynamically(raw_parent_body, c_size=c_size, c_overlap=c_overlap)

        # Consolidate slices so no child slice has < 15 words (unless entire document has < 15 words)
        child_slices: List[str] = []
        for s in raw_slices:
            if not child_slices:
                child_slices.append(s)
            else:
                prev_words = _count_words(child_slices[-1])
                curr_words = _count_words(s)
                if prev_words < 15 or curr_words < 15:
                    child_slices[-1] = f"{child_slices[-1]} {s}"
                else:
                    child_slices.append(s)

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
        return True

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

        prev_line = ""

        for line in page_lines:
            # Check if previous line was a mid-sentence continuation
            prev_last_word = prev_line.split()[-1].lower() if prev_line.split() else ""
            is_wrapped_continuation = prev_last_word in continuation_words

            detected_header = None if is_wrapped_continuation else StructuralChunker.extract_section_header(line)

            # When a section boundary is detected
            if detected_header:
                if current_buffer:
                    flushed = _flush_parent(current_section, current_buffer, current_page_num, is_final=False)
                    if flushed:
                        current_buffer = []
                        current_buffer_len = 0
                current_section = detected_header
                pending_heading = detected_header
                prev_line = line
                continue

            # Body line: attach pending heading as Markdown prefix if present
            if pending_heading:
                attached_line = f"## {pending_heading}\n{line}"
                current_buffer.append(attached_line)
                current_buffer_len += len(attached_line) + 1
                pending_heading = None
            else:
                current_buffer.append(line)
                current_buffer_len += len(line) + 1

            prev_line = line

            # If section exceeds parent_size, chunk cleanly along line boundaries with overlap
            if current_buffer_len >= p_size:
                flushed = _flush_parent(current_section, current_buffer, current_page_num, is_final=False)
                if flushed:
                    overlap_lines: List[str] = []
                    overlap_len = 0
                    for prev_l in reversed(current_buffer):
                        if overlap_len + len(prev_l) > p_overlap:
                            break
                        overlap_lines.insert(0, prev_l)
                        overlap_len += len(prev_l) + 1

                    current_buffer = overlap_lines
                    current_buffer_len = overlap_len

        # Flush buffer at page boundary if content exists
        if current_buffer:
            flushed = _flush_parent(current_section, current_buffer, current_page_num, is_final=False)
            if flushed:
                current_buffer = []
                current_buffer_len = 0

    # If pending heading exists at document end without body text, attach to current_buffer or prior chunk
    if pending_heading:
        if current_buffer:
            current_buffer.append(f"## {pending_heading}")
        elif parent_chunks:
            parent_chunks[-1].text = f"{parent_chunks[-1].text}\n## {pending_heading}"
            child_chunks[-1].text = f"{child_chunks[-1].text}\n## {pending_heading}"
        pending_heading = None

    # Flush any remaining carried-over content on the final page
    if current_buffer:
        _flush_parent(
            current_section,
            current_buffer,
            int(pages[-1].get("page_number", 1)) if pages else 1,
            is_final=True,
        )

    return parent_chunks, child_chunks
