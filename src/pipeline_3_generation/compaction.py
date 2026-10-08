r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_3_generation/compaction.py
   - Role: Context Compaction interface for TrustRAG.
   - Purpose: Minimizes prompt token payloads and eliminates non-informative boilerplate
     by dynamically compacting narrative text candidates to target sentence spans and
     their immediate bounding contexts.
================================================================================
"""

from typing import List, Optional
from src.common.schemas import RetrievalCandidate
from src.pipeline_3_generation.compactor import ContextCompactor, DEFAULT_MAX_COMPACT_CHARS


def compact(
    query: str,
    contexts: List[RetrievalCandidate],
    max_total_chars: int = DEFAULT_MAX_COMPACT_CHARS,
) -> List[RetrievalCandidate]:
    """Compact retrieval candidates to target sentence spans fitting within character budget."""
    compactor = ContextCompactor()
    return compactor.compact_contexts(query=query, contexts=contexts, max_total_chars=max_total_chars)
