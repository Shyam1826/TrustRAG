r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/common/config.py
   - Role: Centralized enterprise configuration and unified settings engine.
   - Purpose: Declares all system-wide parameters, supported file extensions, manifest
     paths, chunking windows, retrieval thresholds, NLI verification cutoffs, and cloud LLM
     credentials. Provides dynamic loading from `configs/config.yaml`, environment variables,
     and `.env` files via Pydantic BaseSettings.

2. INPUT (IP):
   - Optional YAML configuration file (`configs/config.yaml`).
   - Environment variables (os.environ) and optional `.env` file.

3. PROCESS UNDER THE HOOD:
   - Loads `.env` file via `dotenv.load_dotenv()` if present.
   - Parses optional YAML configuration from `configs/config.yaml` if present.
   - Instantiates typed sub-models:
     * `IngestionConfig`: Supported file extensions and manifest persistence path.
     * `ChunkingConfig`: Parent chunk size (1400), parent overlap (200), child size (150).
     * `RetrievalConfig`: Dense Top-K (20), Reranker Top-K (5), RRF constant (60), custom stop words.
     * `VerificationConfig`: NLI entailment threshold (0.75), contradiction threshold (0.65), premise window (1400).
     * `ModelConfig`: Dense embedder, reranker, and DeBERTa NLI model identifiers.
     * `GeneratorConfig`: LLM provider (Groq, Gemini, Ollama, Mock) and API credentials.
   - Exposes `get_settings()` and top-level singleton `config`.

4. OUTPUT (OP):
   - TrustRAGConfig singleton instance with typed configuration sub-trees.
   - Consumed across all pipelines in `src/`.

5. LIBRARIES & DEPENDENCIES:
   - os, pathlib: Path manipulation and environment variable access.
   - yaml: YAML configuration parsing.
   - dotenv: Python-dotenv for loading local `.env` configuration files.
   - pydantic, pydantic_settings: Pydantic v2 BaseModel, BaseSettings, and Field schemas.
================================================================================
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field

try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseModel as BaseSettings  # type: ignore

# Load optional .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class IngestionConfig(BaseModel):
    """File ingestion and incremental manifest settings."""
    supported_extensions: List[str] = Field(
        default_factory=lambda: [".pdf", ".docx", ".xlsx", ".csv", ".txt", ".jpg", ".png"]
    )
    manifest_path: str = "data/ingestion_manifest.json"


class ChunkingConfig(BaseModel):
    """Hierarchical chunking parameters."""
    parent_size: int = 1400
    parent_overlap: int = 200
    child_size: int = 150
    overlap: int = 30


class RetrievalConfig(BaseModel):
    """Search and retrieval parameters."""
    top_k_dense: int = 20
    top_k_rerank: int = 5
    rrf_k: int = 60
    max_chunks_per_doc: int = 3
    enable_document_diversification: bool = True
    custom_stop_words: List[str] = Field(default_factory=list)


class VerificationConfig(BaseModel):
    """Natural language inference and adjudication thresholds."""
    tau_entailment: float = 0.75
    tau_contradiction: float = 0.65
    premise_window_size: int = 1400
    normalize_unicode_citations: bool = True


class ModelConfig(BaseModel):
    """Model names for embedding, reranking, and NLI verification."""
    dense_model_name: str = "BAAI/bge-small-en-v1.5"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    nli_model_name: str = "cross-encoder/nli-deberta-v3-base"


class GeneratorConfig(BaseModel):
    """Cloud and local LLM generation provider credentials and model identifiers."""
    provider: str = Field(default_factory=lambda: os.getenv("GENERATOR_PROVIDER", "groq"))
    groq_api_key: str = Field(default_factory=lambda: os.getenv("GROQ_API_KEY", ""))
    groq_model: str = Field(default_factory=lambda: os.getenv("GROQ_MODEL", "llama-3.1-8b-instant"))
    gemini_api_key: str = Field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = Field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash"))


class ThresholdConfig(BaseModel):
    """Backward-compatible threshold container."""
    rrf_k: int = 60
    top_k_dense: int = 20
    top_k_rerank: int = 5
    tau_entailment: float = 0.75
    tau_contradiction: float = 0.65


class TrustRAGConfig(BaseSettings):
    """Top-level configuration container for TrustRAG."""
    ingestion: IngestionConfig = Field(default_factory=IngestionConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    generation: GeneratorConfig = Field(default_factory=GeneratorConfig)
    thresholds: ThresholdConfig = Field(default_factory=ThresholdConfig)

    # Top-level property aliases for direct access & 100% backward compatibility
    @property
    def PARENT_CHUNK_SIZE(self) -> int:
        return self.chunking.parent_size

    @property
    def RETRIEVAL_TOP_K(self) -> int:
        return self.retrieval.top_k_dense

    @property
    def RERANKER_TOP_K(self) -> int:
        return self.retrieval.top_k_rerank

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


def load_yaml_config(yaml_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load configuration dictionary from YAML file if present."""
    p = Path(yaml_path)
    if not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def get_settings(yaml_path: str = "configs/config.yaml") -> TrustRAGConfig:
    """Instantiate TrustRAGConfig populated from YAML file and environment variables."""
    yaml_data = load_yaml_config(yaml_path)

    ingestion_cfg = IngestionConfig(**yaml_data.get("ingestion", {}))
    chunking_cfg = ChunkingConfig(**yaml_data.get("chunking", {}))
    retrieval_cfg = RetrievalConfig(**yaml_data.get("retrieval", {}))
    verification_cfg = VerificationConfig(**yaml_data.get("verification", {}))
    models_cfg = ModelConfig(**yaml_data.get("models", {}))
    generation_cfg = GeneratorConfig(**yaml_data.get("generation", {}))

    # Keep thresholds synchronized with retrieval and verification configs
    thresholds_cfg = ThresholdConfig(
        rrf_k=retrieval_cfg.rrf_k,
        top_k_dense=retrieval_cfg.top_k_dense,
        top_k_rerank=retrieval_cfg.top_k_rerank,
        tau_entailment=verification_cfg.tau_entailment,
        tau_contradiction=verification_cfg.tau_contradiction,
    )

    return TrustRAGConfig(
        ingestion=ingestion_cfg,
        chunking=chunking_cfg,
        retrieval=retrieval_cfg,
        verification=verification_cfg,
        models=models_cfg,
        generation=generation_cfg,
        thresholds=thresholds_cfg,
    )


# Default singleton instance
config: TrustRAGConfig = get_settings()
