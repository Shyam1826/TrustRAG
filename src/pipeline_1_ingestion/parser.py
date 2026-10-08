r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/parser.py
   - Role: Document intelligence and layout-aware PDF extraction engine for TrustRAG.
   - Purpose: Extracts raw text and exact document provenance coordinates (physical
     page bounding boxes [x0, top, x1, bottom], text lines, and layout blocks) from
     PDF documents using pdfplumber with PyMuPDF (fitz) fallback. Dynamically normalizes
     collapsed whitespace, mathematical equations, and font tokenization boundaries.

2. INPUT (IP):
   - file_path (str): File system path to the input PDF document.
   - engine (str, default="auto"): Parsing engine ('auto', 'pdfplumber', 'fitz').
   - Source: Raw document storage (`data/raw/`) or user-supplied file path.

3. PROCESS UNDER THE HOOD:
   - Validates existence of the file path.
   - Dynamic Whitespace & Layout Preservation:
     * Configures PyMuPDF extraction flags combining TEXTFLAGS_SEARCH, TEXT_DEHYPHENATE,
       and TEXT_PRESERVE_WHITESPACE to prevent font glyph and operator collapsing.
     * Applies generalized boundary restoration:
       - Restores letter-digit-title boundaries: `re.sub(r'([a-z0-9])([A-Z])', r'\1 \2', text)`.
       - Restores operator and equation boundaries: `re.sub(r'([a-zA-Z0-9])\s*([=+*><])\s*([a-zA-Z0-9])', r'\1 \2 \3', text)`.
       - Normalizes mathematical division, subtraction, and section numbering boundaries
         (e.g., `3.1GAMESEEDCOLLECTION` -> `3.1 GAME SEED COLLECTION`).
       - Shields URLs, paths, and citation anchors from boundary perturbations.
   - Dual-Engine Coordinate Capture:
     * When engine is 'pdfplumber' or 'auto':
       - Opens PDF via `pdfplumber.open()`.
       - Extracts text lines with exact physical coordinates [x0, top, x1, bottom].
       - Sorts lines in column-aware reading order (left-to-right columns, top-to-bottom vertical).
       - Computes overall page bounding box enclosing all valid text lines.
     * When engine is 'fitz' or pdfplumber encounters an unrecoverable format error:
       - Opens PDF via PyMuPDF context manager (`fitz.open()`).
       - Extracts layout-aware text blocks with bounding boxes [x0, y0, x1, y1].
       - Computes column buckets and vertical reading order.
   - Filters empty pages gracefully and standardizes 1-indexed page numbering.

4. OUTPUT (OP):
   - list[dict[str, Any]]: List of dictionaries containing:
     * "page_number" (int): 1-indexed physical page number.
     * "raw_text" (str): Extracted text content with normalized spacing.
     * "bbox" (list[float]): Overall page bounding box [x0, top, x1, bottom].
     * "lines" (list[dict]): Per-line text and bounding box coordinates.
     * "blocks" (list[dict]): Per-block text and bounding box coordinates.
   - Consumed by: `src/pipeline_1_ingestion/chunker.py` and `src/pipeline_1_ingestion/cleaner.py`.

5. LIBRARIES & DEPENDENCIES:
   - pdfplumber: Layout analysis and word/line bounding box coordinate extraction.
   - fitz (PyMuPDF): High-performance C-based PDF parsing fallback.
   - pathlib.Path: Standard library for robust cross-platform path validation.
   - sklearn.feature_extraction.text: ENGLISH_STOP_WORDS for compound boundary segmentation.
   - typing (List, Dict, Any, Optional): Python type hints.
================================================================================
"""

from collections import defaultdict
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import fitz  # PyMuPDF
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

try:
    with open("/usr/share/dict/words") as _f_dict:
        _RAW_DICT_WORDS = set(w.strip().upper() for w in _f_dict if len(w.strip()) > 1)
except Exception:
    _RAW_DICT_WORDS = set()

_STOP_CAPS = set(w.upper() for w in ENGLISH_STOP_WORDS)
_COMMON_TERMS = {
    "MODEL", "GAME", "SEED", "COLLECTION", "DATA", "SET", "SETTINGS", "TRAINING",
    "CONFIG", "CONFIGURATION", "ARCHITECTURE", "LEARNING", "SYSTEM", "SPECIFICATIONS",
    "EVALUATION", "RESULTS", "ANALYSIS", "METHOD", "METHODS", "NETWORK", "NETWORKS",
    "HARDWARE", "INFERENCE", "OPTIMIZER", "ATTENTION", "LAYER", "LAYERS", "LOSS",
    "WEIGHT", "DECAY", "BATCH", "SIZE", "RATE", "NORM", "DROPOUT", "EPOCH",
    "COUNT", "ACCURACY", "LATENCY", "DEVICE", "CORE", "CORES", "POWER"
}
_VOCAB = set(_RAW_DICT_WORDS) | _STOP_CAPS | _COMMON_TERMS
_VOCAB.update(w + "S" for w in list(_VOCAB) if len(w) > 2)

_VALID_SHORT = {"IN", "ON", "AT", "TO", "BY", "OF", "OR", "IS", "IT", "AS", "BE", "WE", "HE", "ME", "MY", "SO", "UP", "NO", "GO", "DO", "IF"}


def _split_allcaps_compound(s: str) -> str:
    """Split concatenated all-caps compound words into whitespace-delimited tokens using dynamic programming."""
    n = len(s)
    if n < 6:
        return s
    dp = {0: (0.0, [])}
    for i in range(1, n + 1):
        for j in range(max(0, i - 25), i):
            if j in dp:
                word = s[j:i]
                if (len(word) >= 3 and word in _VOCAB) or (len(word) == 2 and word in _VALID_SHORT) or word in {"A", "I"}:
                    bonus = 8.0 if (word in _STOP_CAPS or word in _COMMON_TERMS) else 0.0
                    score = (len(word) ** 1.5) + bonus
                    cand_score = dp[j][0] + score
                    cand_words = dp[j][1] + [word]
                    if i not in dp or cand_score > dp[i][0]:
                        dp[i] = (cand_score, cand_words)
    if n in dp and len(dp[n][1]) > 1:
        return " ".join(dp[n][1])
    return s


def restore_text_spacing_boundaries(text: str) -> str:
    r"""Restore flattened word boundaries, camelCase transitions, and operator spacing in extracted text.

    Inserts whitespace between alphanumeric transitions where formatting flattens text
    (e.g., regex pattern r'([a-z0-9])([A-Z])' and operator boundaries r'([a-zA-Z0-9])([=+\-*/><])').
    Preserves mathematical variables, equations, URLs, and standard markdown citations without corruption.
    """
    if not text:
        return text

    # Protect URLs, filesystem paths, and markdown citations/breadcrumbs from boundary perturbation
    placeholders = {}

    def _mask(m):
        key = f"__MASK_{len(placeholders)}__"
        placeholders[key] = m.group(0)
        return key

    masked = re.sub(
        r"https?://[^\s]+|(?:/[a-zA-Z0-9_.-]+)+|\b(?:data|src|tests|configs)/[a-zA-Z0-9_./-]+|\[(?:Doc-\d+|Chapter\s+[^\]]+|Section\s+[^\]]+|Image:[^\]]+|Table:[^\]]+)\]",
        _mask,
        text,
    )

    # 1. Section numbering boundaries (e.g. 3.1GAMESEEDCOLLECTION -> 3.1 GAMESEEDCOLLECTION)
    masked = re.sub(r"(\b\d+(?:\.\d+)+|\b[A-Z]\.\d+)\s*([A-Za-z])", r"\1 \2", masked)

    # 2. Fused uppercase compound words (e.g. GAMESEEDCOLLECTION -> GAME SEED COLLECTION)
    masked = re.sub(r"\b[A-Z]{6,}\b", lambda m: _split_allcaps_compound(m.group(0)), masked)

    # 3. Lowercase/digit to uppercase transition (camelCase / flattened titles)
    masked = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", masked)

    # 4. Operator and equation boundaries cleanly
    masked = re.sub(r"([a-zA-Z0-9])\s*([=+*><])\s*([a-zA-Z0-9])", r"\1 \2 \3", masked)
    masked = re.sub(r"([a-zA-Z0-9])\s*([=+*><])", r"\1 \2", masked)
    masked = re.sub(r"([=+*><])\s*([a-zA-Z0-9])", r"\1 \2", masked)

    # 5. Math operators with slash and minus
    masked = re.sub(r"(?<=\b\d)\s*/\s*(?=\d\b)|(?<=\b[a-zA-Z])\s*/\s*(?=[a-zA-Z]\b)", " / ", masked)
    masked = re.sub(r"(?<=\b\d)\s*-\s*(?=\d\b)", " - ", masked)
    masked = re.sub(r"(?<=\b[a-zA-Z])\s*-\s*(?=[a-zA-Z]\b)(?<!\b[a-z]-[a-z])", " - ", masked)

    # Unmask
    for k, v in placeholders.items():
        masked = masked.replace(k, v)

    # Clean redundant spaces per line while preserving newlines
    lines = [re.sub(r"[ \t]+", " ", l) for l in masked.split("\n")]
    return "\n".join(lines)



def extract_pdf_pages(
    file_path: str,
    engine: str = "auto",
    doc_id: Optional[str] = None,
    tabular_store: Optional[Any] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract page-level text, table key-value rows, TOC breadcrumbs, and bounding box coordinates from a PDF.

    Supports both pdfplumber (for fine-grained line/word coordinates and table detection)
    and PyMuPDF fitz (for high-speed, memory-safe streaming extraction on 200+ page files).

    Args:
        file_path: Path to the target PDF document.
        engine: Extraction engine to use ('auto', 'pdfplumber', 'fitz'). Default is 'auto'.
        doc_id: Optional document identifier.
        tabular_store: Optional TabularStore instance for auto-registering extracted PDF tables.
        user_id: Optional tenant user_id.
        thread_id: Optional tenant thread_id.

    Returns:
        A list of dictionaries containing 1-indexed page_number, raw_text, bbox, line coordinates, and tables:
        [
            {
                "page_number": 1,
                "raw_text": "...",
                "bbox": [x0, top, x1, bottom],
                "lines": [{"text": "...", "bbox": [x0, top, x1, bottom]}],
                "blocks": [{"text": "...", "bbox": [x0, top, x1, bottom]}],
                "breadcrumb": "[Chapter 1: Intro | Section 1.1: Background]",
                "section_name": "Section 1.1: Background",
                "tables": [{"table_name": "...", "headers": [...], "rows": [...]}],
            },
            ...
        ]

    Raises:
        FileNotFoundError: If the PDF file does not exist.
        RuntimeError: If PDF parsing fails on both engines.
    """
    from src.pipeline_1_ingestion.tabular_store import TabularStore

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"PDF file not found at path: {file_path}")

    effective_doc_id = doc_id or path.stem
    clean_doc_name = TabularStore.sanitize_identifier(effective_doc_id)
    target_engine = engine.lower().strip() if engine else "auto"

    # Step 0: Extract document outlines / TOC bookmarks via PyMuPDF
    page_breadcrumbs: Dict[int, str] = {}
    page_section_names: Dict[int, str] = {}
    total_pages = 0

    try:
        with fitz.open(str(path)) as doc:
            total_pages = len(doc)
            toc = doc.get_toc() or []

            if toc:
                active_stack: Dict[int, str] = {}
                toc_by_page: Dict[int, List[Tuple[int, str]]] = defaultdict(list)
                for item in toc:
                    if len(item) >= 3:
                        lvl = int(item[0])
                        title = str(item[1]).strip()
                        p_num = max(1, int(item[2]))
                        toc_by_page[p_num].append((lvl, title))

                for p in range(1, total_pages + 1):
                    if p in toc_by_page:
                        for lvl, title in toc_by_page[p]:
                            # Clear deeper hierarchy levels >= lvl
                            for k in list(active_stack.keys()):
                                if k >= lvl:
                                    del active_stack[k]
                            active_stack[lvl] = title

                    if active_stack:
                        # Map as structural breadcrumbs: [Chapter X: Title | Section Y: Subtitle]
                        crumb = "[" + " | ".join(active_stack[k] for k in sorted(active_stack.keys())) + "]"
                        page_breadcrumbs[p] = crumb
                        max_lvl = max(active_stack.keys())
                        page_section_names[p] = active_stack[max_lvl]
    except Exception:
        pass

    # For large PDFs (>80 pages), use fitz streaming loop by default to prevent memory exhaustion
    use_pdfplumber = target_engine == "pdfplumber" or (target_engine == "auto" and total_pages <= 80)

    # Attempt 1: pdfplumber extraction for exact line/block coordinates and tables
    if use_pdfplumber:
        try:
            import pdfplumber

            extracted_pages: List[Dict[str, Any]] = []
            with pdfplumber.open(str(path)) as pdf:
                for page_index, page in enumerate(pdf.pages):
                    page_no = page_index + 1
                    raw_lines = page.extract_text_lines() or []
                    parsed_lines: List[Dict[str, Any]] = []

                    for l in raw_lines:
                        l_text = l.get("text", "").strip()
                        if not l_text:
                            continue
                        parsed_lines.append(
                            {
                                "text": l_text,
                                "bbox": [
                                    round(float(l["x0"]), 2),
                                    round(float(l["top"]), 2),
                                    round(float(l["x1"]), 2),
                                    round(float(l["bottom"]), 2),
                                ],
                            }
                        )

                    page_width = float(page.width)
                    # Detect multi-column layouts
                    has_left = any(
                        l["bbox"][0] < page_width * 0.4 and l["bbox"][2] < page_width * 0.6
                        for l in parsed_lines
                    )
                    has_right = any(l["bbox"][0] >= page_width * 0.35 for l in parsed_lines)

                    if has_left and has_right:
                        sorted_lines = sorted(
                            parsed_lines,
                            key=lambda l: (0 if l["bbox"][0] < page_width * 0.38 else 1, l["bbox"][1]),
                        )
                    else:
                        sorted_lines = sorted(parsed_lines, key=lambda l: (l["bbox"][1], l["bbox"][0]))

                    text = restore_text_spacing_boundaries("\n".join(l["text"] for l in sorted_lines))

                    # Extract tables via pdfplumber
                    try:
                        raw_tables = page.extract_tables() or []
                    except Exception:
                        raw_tables = []

                    page_table_rows: List[str] = []
                    page_tables_meta: List[Dict[str, Any]] = []

                    for table_idx, tbl in enumerate(raw_tables):
                        if not tbl or len(tbl) < 2:
                            continue
                        raw_headers = tbl[0]
                        raw_rows = tbl[1:]

                        clean_headers: List[str] = []
                        for h_idx, h in enumerate(raw_headers):
                            h_str = str(h).strip().replace("\n", " ") if h is not None else ""
                            if not h_str:
                                h_str = f"col_{h_idx + 1}"
                            clean_headers.append(h_str)

                        serialized_table_rows: List[str] = []
                        data_rows_for_df: List[List[str]] = []

                        for r in raw_rows:
                            if not r or not any(c is not None and str(c).strip() for c in r):
                                continue
                            items: List[str] = []
                            row_vals: List[str] = []
                            for h_name, cell in zip(clean_headers, r):
                                val_str = str(cell).strip().replace("\n", " ") if cell is not None else ""
                                row_vals.append(val_str)
                                if val_str:
                                    items.append(f"{h_name}: {val_str}")
                            if items:
                                row_str = f"[Table: {effective_doc_id} Page {page_no}] " + " | ".join(items)
                                serialized_table_rows.append(row_str)
                                data_rows_for_df.append(row_vals)

                        if not serialized_table_rows or not data_rows_for_df:
                            continue

                        page_table_rows.extend(serialized_table_rows)

                        suffix = f"_t{table_idx + 1}" if table_idx > 0 else ""
                        table_name = f"pdf_{clean_doc_name}_p{page_no}{suffix}"

                        if tabular_store is not None:
                            try:
                                import pandas as pd
                                df = pd.DataFrame(data_rows_for_df, columns=clean_headers)
                                tabular_store.register_table_from_dataframe(
                                    df=df,
                                    table_name=table_name,
                                    doc_id=effective_doc_id,
                                    source_type="pdf_table",
                                    page_no=page_no,
                                    user_id=user_id,
                                    thread_id=thread_id,
                                )
                            except Exception as e:
                                print(f"[Parser] Warning: Failed to register table '{table_name}': {e}")

                        page_tables_meta.append({
                            "table_name": table_name,
                            "headers": clean_headers,
                            "rows": data_rows_for_df,
                            "serialized_rows": serialized_table_rows,
                        })

                    breadcrumb = page_breadcrumbs.get(page_no, "")
                    section_name = page_section_names.get(page_no, "General")

                    if not text.strip() and not page_table_rows:
                        # Flush page cache and release
                        try:
                            page.flush_cache()
                        except Exception:
                            pass
                        del page
                        continue

                    if sorted_lines:
                        page_bbox = [
                            round(min(l["bbox"][0] for l in sorted_lines), 2),
                            round(min(l["bbox"][1] for l in sorted_lines), 2),
                            round(max(l["bbox"][2] for l in sorted_lines), 2),
                            round(max(l["bbox"][3] for l in sorted_lines), 2),
                        ]
                    else:
                        page_bbox = [0.0, 0.0, float(page.width), float(page.height)]

                    # Append serialized table rows to raw_text and block pool
                    if page_table_rows:
                        table_text_block = "\n".join(page_table_rows)
                        if text and text.strip():
                            text = text + "\n\n" + table_text_block
                        else:
                            text = table_text_block

                        for r_str in page_table_rows:
                            sorted_lines.append({
                                "text": r_str,
                                "bbox": page_bbox,
                            })

                    # If breadcrumb is present, attach to top of text
                    if breadcrumb and not text.startswith(breadcrumb):
                        text = f"{breadcrumb}\n{text}"
                        sorted_lines.insert(0, {"text": breadcrumb, "bbox": page_bbox})

                    extracted_pages.append(
                        {
                            "page_number": page_no,
                            "raw_text": text,
                            "bbox": page_bbox,
                            "lines": sorted_lines,
                            "blocks": sorted_lines,
                            "breadcrumb": breadcrumb,
                            "section_name": section_name,
                            "tables": page_tables_meta,
                        }
                    )

                    # Flush cache and release page object for memory safety
                    try:
                        page.flush_cache()
                    except Exception:
                        pass
                    del page

            if extracted_pages or target_engine == "pdfplumber":
                return extracted_pages

        except Exception as e:
            if target_engine == "pdfplumber":
                raise RuntimeError(f"pdfplumber failed to extract text from '{file_path}': {e}") from e
            # Fall through to fitz if auto

    # Attempt 2: PyMuPDF (fitz) fallback with memory-efficient streaming page iteration
    try:
        extracted_pages = []
        with fitz.open(str(path)) as doc:
            fitz_flags = (
                (getattr(fitz, "TEXTFLAGS_SEARCH", 0) or 0)
                | (getattr(fitz, "TEXT_DEHYPHENATE", 0) or 0)
                | (getattr(fitz, "TEXT_PRESERVE_WHITESPACE", 0) or 0)
            )
            for page_index in range(len(doc)):
                page_no = page_index + 1
                page = doc[page_index]
                blocks = page.get_text("blocks", flags=fitz_flags)
                text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]

                page_width = page.rect.width
                has_left = any(b[0] < page_width * 0.4 and b[2] < page_width * 0.6 for b in text_blocks)
                has_right = any(b[0] >= page_width * 0.35 for b in text_blocks)

                if has_left and has_right:
                    sorted_blocks = sorted(
                        text_blocks,
                        key=lambda b: (0 if b[0] < page_width * 0.38 else 1, b[1]),
                    )
                else:
                    sorted_blocks = sorted(text_blocks, key=lambda b: (b[1], b[0]))

                raw_page_text = "\n".join(b[4].strip() for b in sorted_blocks)
                text = restore_text_spacing_boundaries(raw_page_text)

                parsed_blocks = [
                    {
                        "text": restore_text_spacing_boundaries(b[4].strip()),
                        "bbox": [
                            round(float(b[0]), 2),
                            round(float(b[1]), 2),
                            round(float(b[2]), 2),
                            round(float(b[3]), 2),
                        ],
                    }
                    for b in sorted_blocks
                ]

                # Extract tables using PyMuPDF find_tables if supported
                page_table_rows = []
                page_tables_meta = []
                if hasattr(page, "find_tables"):
                    try:
                        tab_finder = page.find_tables()
                        for table_idx, tab in enumerate(tab_finder):
                            tbl = tab.extract()
                            if not tbl or len(tbl) < 2:
                                continue
                            raw_headers = tbl[0]
                            raw_rows = tbl[1:]

                            clean_headers = []
                            for h_idx, h in enumerate(raw_headers):
                                h_str = str(h).strip().replace("\n", " ") if h is not None else ""
                                clean_headers.append(h_str if h_str else f"col_{h_idx + 1}")

                            serialized_table_rows = []
                            data_rows_for_df = []
                            for r in raw_rows:
                                if not r or not any(c is not None and str(c).strip() for c in r):
                                    continue
                                items = []
                                row_vals = []
                                for h_name, cell in zip(clean_headers, r):
                                    val_str = str(cell).strip().replace("\n", " ") if cell is not None else ""
                                    row_vals.append(val_str)
                                    if val_str:
                                        items.append(f"{h_name}: {val_str}")
                                if items:
                                    row_str = f"[Table: {effective_doc_id} Page {page_no}] " + " | ".join(items)
                                    serialized_table_rows.append(row_str)
                                    data_rows_for_df.append(row_vals)

                            if not serialized_table_rows or not data_rows_for_df:
                                continue

                            page_table_rows.extend(serialized_table_rows)
                            suffix = f"_t{table_idx + 1}" if table_idx > 0 else ""
                            table_name = f"pdf_{clean_doc_name}_p{page_no}{suffix}"

                            if tabular_store is not None:
                                try:
                                    import pandas as pd
                                    df = pd.DataFrame(data_rows_for_df, columns=clean_headers)
                                    tabular_store.register_table_from_dataframe(
                                        df=df,
                                        table_name=table_name,
                                        doc_id=effective_doc_id,
                                        source_type="pdf_table",
                                        page_no=page_no,
                                        user_id=user_id,
                                        thread_id=thread_id,
                                    )
                                except Exception as e:
                                    print(f"[Parser] Warning: Failed to register table '{table_name}': {e}")

                            page_tables_meta.append({
                                "table_name": table_name,
                                "headers": clean_headers,
                                "rows": data_rows_for_df,
                                "serialized_rows": serialized_table_rows,
                            })
                    except Exception:
                        pass

                breadcrumb = page_breadcrumbs.get(page_no, "")
                section_name = page_section_names.get(page_no, "General")

                if not text.strip() and not page_table_rows:
                    del page
                    continue

                if sorted_blocks:
                    page_bbox = [
                        round(min(float(b[0]) for b in sorted_blocks), 2),
                        round(min(float(b[1]) for b in sorted_blocks), 2),
                        round(max(float(b[2]) for b in sorted_blocks), 2),
                        round(max(float(b[3]) for b in sorted_blocks), 2),
                    ]
                else:
                    page_bbox = [0.0, 0.0, float(page.rect.width), float(page.rect.height)]

                if page_table_rows:
                    table_text_block = "\n".join(page_table_rows)
                    if text and text.strip():
                        text = text + "\n\n" + table_text_block
                    else:
                        text = table_text_block

                    for r_str in page_table_rows:
                        parsed_blocks.append({
                            "text": r_str,
                            "bbox": page_bbox,
                        })

                if breadcrumb and not text.startswith(breadcrumb):
                    text = f"{breadcrumb}\n{text}"
                    parsed_blocks.insert(0, {"text": breadcrumb, "bbox": page_bbox})

                extracted_pages.append(
                    {
                        "page_number": page_no,
                        "raw_text": text,
                        "bbox": page_bbox,
                        "lines": parsed_blocks,
                        "blocks": parsed_blocks,
                        "breadcrumb": breadcrumb,
                        "section_name": section_name,
                        "tables": page_tables_meta,
                    }
                )

                # Release page handle to ensure memory safety on 200+ page documents
                del page

        return extracted_pages
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise RuntimeError(f"Failed to extract text from PDF '{file_path}': {e}") from e


def extract_image_document(
    file_path: Union[str, Path],
    doc_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Extract textual content, metadata, and OCR transcript from multimodal images (.jpg, .jpeg, .png).

    Uses pytesseract with graceful fallback if the Tesseract binary is absent.

    Args:
        file_path: Path to target image file.
        doc_id: Optional document identifier.

    Returns:
        List containing a single page dictionary representing page_number=1.

    Raises:
        FileNotFoundError: If the image file does not exist.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found at path: {file_path}")

    from PIL import Image

    effective_doc_id = doc_id or path.stem

    try:
        with Image.open(str(path)) as img:
            width, height = img.size
            img_format = img.format or path.suffix.upper().strip(".")
            img_mode = img.mode

            extracted_ocr_text = ""
            has_ocr = False

            # Graceful OCR attempt
            try:
                import pytesseract
                ocr_out = pytesseract.image_to_string(img)
                if ocr_out and ocr_out.strip():
                    extracted_ocr_text = ocr_out.strip()
                    has_ocr = True
            except Exception as e:
                print(
                    f"[OCR Warning] Optical character recognition skipped for '{path.name}': {e}. "
                    "Falling back to image layout/metadata heuristics."
                )

            # Construct structured narrative representation
            header = f"## Image Document: {effective_doc_id} ({width}x{height} {img_format})"
            meta_desc = f"Resolution: {width}x{height} pixels | Color Mode: {img_mode} | File: {path.name}"

            if extracted_ocr_text:
                full_text = f"{header}\n{meta_desc}\n\n[OCR Extracted Text]\n{extracted_ocr_text}"
            else:
                full_text = f"{header}\n{meta_desc}\nVisual content artifact for document '{effective_doc_id}'."

            page_bbox = [0.0, 0.0, float(width), float(height)]
            lines = [
                {"text": l.strip(), "bbox": page_bbox}
                for l in full_text.split("\n")
                if l.strip()
            ]

            return [
                {
                    "page_number": 1,
                    "raw_text": full_text,
                    "bbox": page_bbox,
                    "lines": lines,
                    "blocks": lines,
                    "breadcrumb": f"[Image: {effective_doc_id}]",
                    "section_name": "Image Content",
                    "metadata": {
                        "file_name": path.name,
                        "width": width,
                        "height": height,
                        "format": img_format,
                        "mode": img_mode,
                        "has_ocr": has_ocr,
                    },
                }
            ]
    except Exception as e:
        if isinstance(e, FileNotFoundError):
            raise
        raise RuntimeError(f"Failed to process image file '{file_path}': {e}") from e

