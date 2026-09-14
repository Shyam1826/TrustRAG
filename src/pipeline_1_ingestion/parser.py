"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/parser.py
   - Role: Document extraction entry-point for TrustRAG.
   - Purpose: Extracts raw, unformatted text on a per-page basis from PDF documents.

2. INPUT (IP):
   - file_path (str): File system path to the input PDF document.
   - Source: Raw document storage (`data/raw/`) or user-supplied file path.

3. PROCESS UNDER THE HOOD:
   - Validates existence of the file path.
   - Opens the PDF binary safely via PyMuPDF context manager (`fitz.open`).
   - Iterates through all document pages (0-indexed internally).
   - Extracts layout-aware blocks using `page.get_text("blocks")`.
   - Sorts blocks in column-aware reading order (left-to-right columns, top-to-bottom vertical).
   - Joins block strings cleanly and filters out empty pages gracefully.
   - Assigns 1-indexed page numbers.

4. OUTPUT (OP):
   - list[dict[str, Any]]: List of dictionaries containing:
     * "page_number" (int): 1-indexed physical page number.
     * "raw_text" (str): Extracted text content.
   - Consumed by: `src/pipeline_1_ingestion/chunker.py` and `src/pipeline_1_ingestion/cleaner.py`.

5. LIBRARIES & DEPENDENCIES:
   - fitz (PyMuPDF): High-performance C-based PDF parsing library chosen for fast text extraction.
   - pathlib.Path: Standard library for robust cross-platform path validation.
   - typing (List, Dict, Any): Python type hints.
================================================================================
"""

from pathlib import Path
from typing import Any, Dict, List
import fitz  # PyMuPDF


def extract_pdf_pages(file_path: str) -> List[Dict[str, Any]]:
    """Extract page-level text from a PDF document using layout-aware block sorting.

    Args:
        file_path: Path to the target PDF document.

    Returns:
        A list of dictionaries containing 1-indexed page_number and raw_text:
        [{"page_number": 1, "raw_text": "..."}, ...]

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If PyMuPDF fails to open or read the PDF file.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    extracted_pages: List[Dict[str, Any]] = []

    try:
        with fitz.open(str(path)) as doc:
            for page_index in range(len(doc)):
                page = doc[page_index]
                blocks = page.get_text("blocks")
                # Filter text blocks (type 0)
                text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

                if not text_blocks:
                    continue

                page_width = page.rect.width
                # Detect multi-column layouts
                has_left = any(b[0] < page_width * 0.4 and b[2] < page_width * 0.6 for b in text_blocks)
                has_right = any(b[0] >= page_width * 0.35 for b in text_blocks)

                if has_left and has_right:
                    # Sort by column bucket (left column first, then right column), then vertical y0
                    sorted_blocks = sorted(text_blocks, key=lambda b: (0 if b[0] < page_width * 0.38 else 1, b[1]))
                else:
                    # Standard single-column vertical reading order
                    sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

                text = "\n".join(b[4].strip() for b in sorted_blocks)

                # Skip completely empty or whitespace-only pages gracefully
                if not text or not text.strip():
                    continue

                extracted_pages.append(
                    {
                        "page_number": page_index + 1,
                        "raw_text": text,
                    }
                )
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise RuntimeError(f"Failed to extract text from PDF '{file_path}': {e}") from e

    return extracted_pages
