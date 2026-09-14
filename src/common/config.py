r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/common/config.py
   - Role: Centralized configuration and environment management engine.
   - Purpose: Declares all system-wide parameters, model identifiers, chunking windows
     (1400 chars parent chunk size with 200 overlap), retrieval thresholds (Top-5 rerank),
     and cloud LLM generation provider credentials (Groq, Gemini).

2. INPUT (IP):
   - Environment variables (os.environ) and optional `.env` configuration file.

3. PROCESS UNDER THE HOOD:
   - Loads `.env` file via `dotenv.load_dotenv()` if present.
   - Configures `ModelConfig` for dense embeddings, reranker, and NLI classifier.
   - Configures `ChunkingConfig` with 1400-char parent chunks for multi-section documents.
   - Configures `ThresholdConfig` with Top-5 retrieval and reranker depths.
   - Configures `GeneratorConfig` for cloud and local LLM backends (Groq, Gemini, Ollama, Mock).
   - Exposes top-level configuration singleton `config`.

4. OUTPUT (OP):
   - TrustRAGConfig singleton instance with typed configuration sub-trees.
   - Consumed across all pipelines in `src/`.

5. LIBRARIES & DEPENDENCIES:
   - os: Standard library for reading environment variables.
   - dotenv: Python-dotenv for loading local `.env` configuration files.
   - pydantic: Pydantic v2 BaseModel and Field for schema configuration.
================================================================================
"""

import os
from pydantic import BaseModel, Field

# Load optional .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ModelConfig(BaseModel):
    """Model names for embedding, reranking, and NLI verification."""
    dense_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"


class ChunkingConfig(BaseModel):
    """Hierarchical chunking parameters."""
    parent_size: int = 1400
    parent_overlap: int = 200
    child_size: int = 150
    overlap: int = 30


class ThresholdConfig(BaseModel):
    """Search, reranking, and NLI decision thresholds."""
    rrf_k: int = 60
    top_k_dense: int = 20
    top_k_rerank: int = 5
    tau_entailment: float = 0.80
    tau_contradiction: float = 0.60


class GeneratorConfig(BaseModel):
    """Cloud and local LLM generation provider credentials and model identifiers."""
    provider: str = Field(default_factory=lambda: os.getenv("GENERATOR_PROVIDER", "groq"))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))


class TrustRAGConfig(BaseModel):
    """Top-level configuration container for TrustRAG."""
    models: ModelConfig = Field(default_factory=ModelConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)
    generation: GeneratorConfig = Field(default_factory=GeneratorConfig)

    # Top-level property aliases for direct access
    @property
    def PARENT_CHUNK_SIZE(self) -> int:
        return self.chunking.parent_size

    @property
    def RETRIEVAL_TOP_K(self) -> int:
        return self.thresholds.top_k_dense

    @property
    def RERANKER_TOP_K(self) -> int:
        return self.thresholds.top_k_rerank

    @property
    def GENERATOR_PROVIDER(self) -> str:
        return self.generation.provider

    @property
    def GROQ_API_KEY(self) -> str:
        return self.generation.groq_api_key

    @property
    def GROQ_MODEL(self) -> str:
        return self.generation.groq_model

    @property
    def GEMINI_API_KEY(self) -> str:
        return self.generation.gemini_api_key

    @property
    def GEMINI_MODEL(self) -> str:
        return self.generation.gemini_model


# Default singleton instance
config = TrustRAGConfig()
