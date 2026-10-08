r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/reader.py
   - Role: Multi-format document reader and tabular serialization engine for TrustRAG.
   - Purpose: Ingests unstructured documents (PDFs, Markdown, plain text) and structured
     tabular workbooks (Excel `.xlsx`, `.xls`, CSV) into standardized page/sheet-level text
     representations with structural section headers and semantic key-value row serialization.

2. INPUT (IP):
   - file_path (str | Path): Absolute or relative filesystem path to the target document.
   - doc_id (str, optional): Target document identifier derived from relative filepath.

3. PROCESS UNDER THE HOOD:
   - Validates existence and supported file extension.
   - Excel Workbook Processing (`read_excel`):
     * Dynamically verifies optional dependencies (`pandas`, `openpyxl`). If missing, raises
       a descriptive runtime error directing the user to install them.
     * Iterates through every sheet in the workbook using `pd.ExcelFile`.
     * Drops completely empty rows and columns (`df.dropna(how='all')`).
     * Handles duplicated or unlabelled headers cleanly (e.g. `Unnamed:` columns or duplicate names).
     * Serializes each non-empty data row into a semantic key-value record:
       `[Sheet: <SheetName>] <Col1>: <Val1> | <Col2>: <Val2> | ...`
     * Prepend structural Markdown section headers `## Sheet: <SheetName>` to ensure
       compatibility with downstream hierarchical chunking and metadata preservation.
     * Emits a dictionary per sheet containing 1-indexed `page_number` and `raw_text`.
   - Unified Reader Dispatcher (`read_document`):
     * Routes `.xlsx` and `.xls` to `read_excel()`.
     * Routes `.pdf` to `extract_pdf_pages()` in `parser.py`.
     * Routes `.txt` and `.md` to UTF-8 file reading with normalize fallback.
     * Routes `.csv` to tabular CSV row serializer.

4. OUTPUT (OP):
   - list[dict[str, Any]]: Standardized list of page/sheet dictionaries containing:
     * "page_number" (int): 1-indexed physical page or sheet sequence number.
     * "raw_text" (str): Normalized, section-tagged text content ready for chunking.
   - Consumed by: `src/pipeline_1_ingestion/chunker.py` and `src/main.py`.

5. LIBRARIES & DEPENDENCIES:
   - pathlib.Path: Filesystem path manipulation.
   - typing: Type hints (List, Dict, Any, Optional, Union).
   - pandas, openpyxl: Tabular workbook ingestion.
   - src.pipeline_1_ingestion.parser: PDF page extraction fallback.
================================================================================
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import unicodedata

from src.pipeline_1_ingestion.parser import extract_image_document, extract_pdf_pages


def read_excel(file_path: Union[str, Path], doc_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """Parse and serialize all sheets in an Excel workbook into structured text blocks.

    Args:
        file_path: Path to the target Excel (.xlsx, .xls) file.
        doc_id: Optional unique identifier for the document.

    Returns:
        List of dictionaries with 'page_number' (sheet index) and 'raw_text'.

    Raises:
        FileNotFoundError: If the file does not exist.
        ImportError: If pandas or openpyxl dependencies are missing.
        RuntimeError: If workbook parsing encounters an unrecoverable error.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Excel workbook not found at path: {file_path}")

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Optional Excel ingestion dependencies missing. "
            "Please install openpyxl and pandas using: pip install openpyxl pandas"
        ) from e

    try:
        engine = "openpyxl" if path.suffix.lower() == ".xlsx" else None
        excel_file = pd.ExcelFile(str(path), engine=engine)
    except Exception as e:
        raise RuntimeError(f"Failed to load Excel workbook '{path.name}': {e}") from e

    sheet_pages: List[Dict[str, Any]] = []

    for sheet_idx, sheet_name in enumerate(excel_file.sheet_names, start=1):
        try:
            df = excel_file.parse(sheet_name)
        except Exception as e:
            print(f"Warning: Could not parse sheet '{sheet_name}' in {path.name}: {e}")
            continue

        # 1. Discard completely empty rows and columns
        df = df.dropna(how="all").dropna(axis=1, how="all")
        if df.empty:
            continue

        # 2. Detect and clean header columns
        raw_cols = list(df.columns)
        clean_cols: List[str] = []
        for c_idx, col in enumerate(raw_cols):
            col_str = str(col).strip()
            if col_str.startswith("Unnamed:") or not col_str:
                clean_cols.append(f"Column_{c_idx + 1}")
            else:
                clean_cols.append(col_str)

        # 3. Serialize each row into semantic key-value records
        row_records: List[str] = []
        for _, row in df.iterrows():
            row_items: List[str] = []
            for col_name, val in zip(clean_cols, row):
                if pd.isna(val) or val is None:
                    continue
                val_str = str(val).strip()
                if not val_str or val_str.lower() == "nan":
                    continue
                # Format floating integers cleanly (e.g., 2026.0 -> 2026)
                if isinstance(val, float) and val.is_integer():
                    val_str = str(int(val))
                row_items.append(f"{col_name}: {val_str}")

            if row_items:
                row_line = f"[Sheet: {sheet_name}] " + " | ".join(row_items)
                row_records.append(row_line)

        if not row_records:
            continue

        # 4. Prepend structural Markdown section header
        sheet_text = f"## Sheet: {sheet_name}\n" + "\n".join(row_records)

        sheet_pages.append({
            "page_number": sheet_idx,
            "raw_text": sheet_text,
        })

    return sheet_pages


def read_document(
    file_path: Union[str, Path],
    doc_id: Optional[str] = None,
    tabular_store: Optional[Any] = None,
    user_id: Optional[str] = None,
    thread_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Universal document reader routing files to appropriate format serializers.

    Args:
        file_path: Path to target document file.
        doc_id: Optional unique identifier for the document.
        tabular_store: Optional TabularStore instance for registering PDF tables.
        user_id: Optional tenant user_id.
        thread_id: Optional tenant thread_id.

    Returns:
        List of dictionaries with 'page_number' and 'raw_text'.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found at path: {file_path}")

    ext = path.suffix.lower()

    if ext in (".xlsx", ".xls"):
        return read_excel(path, doc_id=doc_id)
    elif ext == ".pdf":
        return extract_pdf_pages(
            str(path),
            doc_id=doc_id,
            tabular_store=tabular_store,
            user_id=user_id,
            thread_id=thread_id,
        )
    elif ext in (".jpg", ".jpeg", ".png"):
        return extract_image_document(path, doc_id=doc_id)
    elif ext in (".csv", ".tsv"):
        delimiter = "\t" if ext == ".tsv" else ","
        try:
            import pandas as pd
            df = pd.read_csv(str(path), delimiter=delimiter)
            df = df.dropna(how="all").dropna(axis=1, how="all")
            if df.empty:
                return []
            cols = [str(c).strip() for c in df.columns]
            rows: List[str] = []
            for _, r in df.iterrows():
                items = [f"{c}: {r[c]}" for c in cols if pd.notna(r[c]) and str(r[c]).strip()]
                if items:
                    rows.append(" | ".join(items))
            text = f"## Table: {path.stem}\n" + "\n".join(rows)
            return [{"page_number": 1, "raw_text": text}]
        except Exception:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")
            return [{"page_number": 1, "raw_text": raw_text}]
    elif ext in (".txt", ".md"):
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        cleaned = unicodedata.normalize("NFKC", raw_text)
        return [{"page_number": 1, "raw_text": cleaned}] if cleaned.strip() else []
    else:
        # Fallback raw text reader
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        return [{"page_number": 1, "raw_text": raw_text}] if raw_text.strip() else []
