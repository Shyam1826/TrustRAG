r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_1_ingestion/embedder.py
   - Role: Dense and sparse embedding generation engine.
   - Purpose: Encodes text chunks into L2-normalized dense vector representations using
     transformer architectures and produces term frequency sparse token dictionaries
     for hybrid retrieval indexing.

2. INPUT (IP):
   - Dense: list[str] containing chunk text.
   - Sparse: str containing individual chunk text.
   - Source: `src/pipeline_1_ingestion/chunker.py` (via `ChildChunk.text`).

3. PROCESS UNDER THE HOOD:
   - Hardware Accelerator Detection: Checks `torch.cuda.is_available()` -> `cuda`,
     `torch.backends.mps.is_available()` -> `mps`, or falls back to `cpu`.
   - Dense Embedding:
     * Loads SentenceTransformer model (`BAAI/bge-small-en-v1.5`, 384 dimensions).
     * Encodes texts with `normalize_embeddings=True`, ensuring all vectors satisfy ||v||_2 = 1.0.
   - Sparse Tokenization:
     * Regex `r'\b\w+\b'` extracts alphanumeric words in lowercase.
     * Computes raw term frequencies using `collections.Counter`.

4. OUTPUT (OP):
   - Dense: list[list[float]] representing unit vectors of dimension 384.
   - Sparse: dict[str, int] representing term -> count mappings.
   - Consumed by: `src/pipeline_1_ingestion/indexer.py` (via `ChildChunk.vector` and `ChildChunk.sparse_tokens`).

5. LIBRARIES & DEPENDENCIES:
   - sentence_transformers: Transformer embedding framework for text representation.
   - torch: Deep learning framework for tensor calculations and hardware acceleration.
   - re: Standard library module for regex tokenization.
   - collections.Counter: High-performance token frequency counting.
   - src.common.config: Provides centralized embedding model configurations.
================================================================================
"""

import re
from collections import Counter
from typing import Dict, List, Optional
import torch
from sentence_transformers import SentenceTransformer

from src.common.config import config


def get_optimal_device() -> str:
    """Detect available hardware accelerator (CUDA, Apple Silicon MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class DualEmbedder:
    """Generates L2-normalized dense vectors and sparse term-frequency mappings."""

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None) -> None:
        self.model_name = model_name or config.models.dense_model_name
        self.device = device or get_optimal_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)

    def embed_dense(self, texts: List[str]) -> List[List[float]]:
        """Compute L2-normalized dense embeddings for a list of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of float vectors, each L2-normalized.
        """
        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [vec.tolist() for vec in embeddings]

    def tokenize_sparse(self, text: str) -> Dict[str, int]:
        """Compute lowercased term frequencies after stripping punctuation.

        Args:
            text: Input text string.

        Returns:
            Dictionary mapping terms to frequency counts.
        """
        if not text:
            return {}

        tokens = re.findall(r"\b\w+\b", text.lower())
        return dict(Counter(tokens))
