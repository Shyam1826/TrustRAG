r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/discover.py
   - Role: Document discovery and ingestion reader registry.
   - Purpose: Recursively scans data folders across arbitrary directory depths, registers
     supported file extension handlers (.pdf, .xlsx, .xls, .docx, .csv, .txt), and derives
     collision-safe unique doc_ids and folder hierarchies for incremental indexing.

2. INPUT (IP):
   - raw_dir (str | Path): Base directory path to scan (e.g. `data/raw`).
   - supported_extensions (set[str], optional): Allowed file extensions.

3. PROCESS UNDER THE HOOD:
   - Traverses directories using `Path.rglob("*")`.
   - Filters out hidden files/directories (starting with `.`) and unsupported extensions.
   - Computes collision-safe relative doc_ids (e.g. `operations/SLA_Classifications_Template`).
   - Extracts nested folder hierarchy metadata (`['operations']`).
   - Provides reader handler dispatching for discovered file types.

4. OUTPUT (OP):
   - list[tuple[Path, str, str, list[str]]]: Discovered files metadata:
     (file_path, doc_id, relative_path_str, folder_hierarchy).
   - Consumed by: `src/main.py` and `demo.py`.

5. LIBRARIES & DEPENDENCIES:
   - pathlib.Path: File path traversal.
   - typing: Type annotations.
   - src.common.config: Default supported file extensions.
   - src.pipeline_1_ingestion.reader: Document readers.
================================================================================
"""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple, Union

from src.common.config import config
from src.pipeline_1_ingestion.reader import read_document, read_excel


# Registry of format-specific reader handlers
READER_REGISTRY: Dict[str, Callable] = {
    ".pdf": read_document,
    ".xlsx": read_excel,
    ".xls": read_excel,
    ".csv": read_document,
    ".txt": read_document,
    ".md": read_document,
}


def register_reader_handler(extension: str, handler: Callable) -> None:
    """Register a custom reader handler for a given file extension."""
    norm_ext = extension.lower() if extension.startswith(".") else f".{extension.lower()}"
    READER_REGISTRY[norm_ext] = handler


def get_reader_for_file(file_path: Union[str, Path]) -> Callable:
    """Retrieve the registered reader handler for a file path, falling back to read_document."""
    ext = Path(file_path).suffix.lower()
    return READER_REGISTRY.get(ext, read_document)


def discover_raw_documents(
    raw_dir: Union[str, Path] = "data/raw",
    supported_extensions: Optional[Set[str]] = None,
) -> List[Tuple[Path, str, str, List[str]]]:
    """Recursively discover document files in raw_dir across all subfolder depths.

    Args:
        raw_dir: Base directory path to scan.
        supported_extensions: Optional set of allowed file extensions.

    Returns:
        List of tuples: `(file_path, doc_id, relative_path_str, folder_hierarchy)`.
    """
    base_dir = Path(raw_dir)
    if not base_dir.exists():
        return []

    exts = supported_extensions or set(config.ingestion.supported_extensions)
    discovered: List[Tuple[Path, str, str, List[str]]] = []

    for file_path in sorted(base_dir.rglob("*")):
        if not file_path.is_file():
            continue

        try:
            rel_path = file_path.relative_to(base_dir)
        except ValueError:
            rel_path = Path(file_path.name)

        # Ignore hidden system files and hidden directory segments
        if any(part.startswith(".") for part in rel_path.parts):
            continue

        if file_path.suffix.lower() not in exts:
            continue

        # Derive clean, collision-safe doc_id (e.g., "operations/SLA_Classifications_Template")
        rel_str = str(rel_path).replace("\\", "/")
        rel_stem_str = str(rel_path.with_suffix("")).replace("\\", "/")
        doc_id = rel_stem_str
        folder_hierarchy = [p for p in rel_path.parent.parts if p and p != "."]

        discovered.append((file_path, doc_id, rel_str, folder_hierarchy))

    return discovered
