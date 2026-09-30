"""Pipeline 3: Citation-Aware Generation and Context Compaction."""

from src.pipeline_3_generation.citation_check import validate_and_parse_citations
from src.pipeline_3_generation.compactor import ContextCompactor
from src.pipeline_3_generation.generator import BaseGenerator, get_generator
from src.pipeline_3_generation.prompt import build_correction_prompt, build_rag_prompt

__all__ = [
    "validate_and_parse_citations",
    "ContextCompactor",
    "BaseGenerator",
    "get_generator",
    "build_rag_prompt",
    "build_correction_prompt",
]
