r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/manifest.py
   - Role: Incremental idempotent ingestion manifest and file fingerprinting engine.
   - Purpose: Tracks document indexing state on disk using streaming SHA-256 hashing
     and byte-size verification. Prevents redundant re-parsing and embedding of unchanged
     files during startup scans, detects modified files for re-indexing, and removes deleted
     documents.

2. INPUT (IP):
   - path (Path | str): Target document file path.
   - doc_id (str): Unique document identifier.
   - chunk_count (int): Number of generated and indexed chunks.
   - manifest_path (Path | str, optional): Persistent JSON manifest storage path.

3. PROCESS UNDER THE HOOD:
   - Manages persistent JSON state at `data/ingestion_manifest.json` (or configured path).
   - Computes streaming SHA-256 hash using 64KB block iterations (`65536` bytes).
   - `is_indexed_and_current()`: Validates whether `doc_id` exists in manifest with
     identical byte size and SHA-256 hash.
   - `record_indexed()`: Records document fingerprint, chunk count, timestamp, and metadata.
   - `remove_entry()`: Deletes document record from manifest and persists state.
   - `prune_missing_files()`: Identifies and removes deleted document entries.

4. OUTPUT (OP):
   - bool: Validity status of indexed file cache.
   - IngestionManifest instance managing persistent JSON registry on disk.
   - Consumed by: `src/main.py` and `demo.py`.

5. LIBRARIES & DEPENDENCIES:
   - hashlib: SHA-256 streaming cryptographic hash computation.
   - json: JSON serialization and persistence.
   - os, pathlib: Filesystem operations and file size inspection.
   - datetime: ISO-8601 timestamp generation.
   - src.common.config: Configured manifest path settings.
================================================================================
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.common.config import config


class IngestionManifest:
    """Manages persistent SHA-256 document indexing registry to ensure idempotent ingestion."""

    def __init__(self, manifest_path: Optional[Union[str, Path]] = None) -> None:
        """Initialize the ingestion manifest.

        Args:
            manifest_path: Optional path to JSON manifest file.
        """
        raw_path = manifest_path or config.ingestion.manifest_path
        self.manifest_path = Path(raw_path)
        self.entries: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        """Load manifest data from disk if file exists."""
        if self.manifest_path.is_file():
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self.entries = data.get("documents", data)
            except Exception as e:
                print(f"[Manifest] Warning: Failed to read manifest ({e}). Starting fresh.")
                self.entries = {}
        else:
            self.entries = {}

    def _save(self) -> None:
        """Atomically persist manifest dictionary to disk."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "version": "1.0",
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "documents": self.entries,
            }
            temp_path = self.manifest_path.with_suffix(".tmp")
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            os.replace(temp_path, self.manifest_path)
        except Exception as e:
            print(f"[Manifest] Error saving manifest: {e}")

    @staticmethod
    def compute_file_hash(path: Union[str, Path]) -> str:
        """Compute streaming SHA-256 hash of a file in 64KB chunks.

        Args:
            path: Path to the target file.

        Returns:
            Hexadecimal SHA-256 digest string.
        """
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(f"File not found for hashing: {path}")

        sha256 = hashlib.sha256()
        with open(p, "rb") as f:
            while chunk := f.read(65536):
                sha256.update(chunk)
        return sha256.hexdigest()

    def is_indexed_and_current(self, path: Union[str, Path], doc_id: str) -> bool:
        """Check if document is already indexed and unchanged.

        Args:
            path: Path to document on disk.
            doc_id: Unique document identifier.

        Returns:
            True if file exists in manifest with matching size and SHA-256 hash.
        """
        p = Path(path)
        if not p.is_file() or doc_id not in self.entries:
            return False

        entry = self.entries[doc_id]
        current_size = p.stat().st_size
        recorded_size = entry.get("size_bytes")

        if current_size != recorded_size:
            return False

        current_hash = self.compute_file_hash(p)
        recorded_hash = entry.get("sha256")
        return current_hash == recorded_hash

    def record_indexed(
        self,
        path: Union[str, Path],
        doc_id: str,
        chunk_count: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record a successfully indexed document in the manifest.

        Args:
            path: Path to indexed document file.
            doc_id: Unique document identifier.
            chunk_count: Number of chunks indexed for this document.
            metadata: Optional additional metadata to store.
        """
        p = Path(path)
        file_size = p.stat().st_size if p.is_file() else 0
        file_hash = self.compute_file_hash(p) if p.is_file() else ""

        self.entries[doc_id] = {
            "doc_id": doc_id,
            "file_path": str(p),
            "size_bytes": file_size,
            "sha256": file_hash,
            "chunk_count": chunk_count,
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }
        self._save()

    def remove_entry(self, doc_id: str) -> None:
        """Remove a document entry from the manifest.

        Args:
            doc_id: Unique document identifier to delete.
        """
        if doc_id in self.entries:
            del self.entries[doc_id]
            self._save()

    def get_indexed_doc_ids(self) -> List[str]:
        """Return list of all registered document IDs."""
        return list(self.entries.keys())

    def prune_missing_files(self) -> List[str]:
        """Remove entries for files that no longer exist on disk.

        Returns:
            List of pruned document IDs.
        """
        pruned: List[str] = []
        for doc_id, entry in list(self.entries.items()):
            file_path = entry.get("file_path")
            if file_path and not Path(file_path).exists():
                pruned.append(doc_id)
                del self.entries[doc_id]

        if pruned:
            self._save()
        return pruned
