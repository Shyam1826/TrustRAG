r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/pipeline_2_retrieval/reranker.py
   - Role: Cross-Encoder neural reranker, neighbor context expansion, and diversity resolver.
   - Purpose: Performs deep cross-attention between queries and candidate child chunks,
     evaluates full interaction representations, resolves winning child chunks to
     parent context passages, merges adjacent neighbor chunks from the same document
     into unified context blocks, and enforces mathematical per-document diversification quotas
     to prevent large corpora from starving smaller documents.

2. INPUT (IP):
   - query (str): Cleaned query string from `src/pipeline_2_retrieval/rewriter.py`.
   - candidate_child_ids (list[str]): Top fused chunk IDs from `src/pipeline_2_retrieval/fusion.py`.
   - child_chunk_map (dict[str, ChildChunk]): Map of chunk IDs to ChildChunk instances.
   - local_store (LocalStore): Storage instance holding indexed ParentChunk passages.
   - top_k (int): Number of top reranked parent candidates to return (default 5 from config).

3. PROCESS UNDER THE HOOD:
   - Pairs query with each candidate child's text: `[[query, child.text], ...]`.
   - Runs cross-encoder forward pass using `cross-encoder/ms-marco-MiniLM-L-6-v2`.
   - Sorts candidate children descending by raw cross-encoder relevance scores.
   - Resolves child chunks to ParentChunk objects via `local_store.get_parent(child.parent_id)`.
   - Performs Document-Aware Neighbor Context Expansion:
     * Groups resolved parent chunks by `doc_id`.
     * Identifies adjacent neighbor chunks within the same document (consecutive `chunk_index`).
     * Merges adjacent parent passages into unified multi-section context blocks.
   - Applies Mathematical Document Diversification:
     * Computes dynamic quota cap `max_chunks_per_source = max(1, top_k // min(num_unique_matching_docs, 3))`.
     * Selects candidates in descending score order respecting per-document quota limits.
     * Backfills from remaining highest-scoring candidates if fewer than `top_k` are selected.
   - Returns up to `top_k` strictly typed `RetrievalCandidate` models.

4. OUTPUT (OP):
   - list[RetrievalCandidate]: Top-k expanded parent candidates containing complete context.
   - Consumed by: `src/pipeline_3_generation` (synthesis) and `src/pipeline_4_verification` (audit).

5. LIBRARIES & DEPENDENCIES:
   - sentence_transformers.CrossEncoder: Full cross-attention transformer reranking model.
   - torch: Tensor operations and device acceleration.
   - src.common.config: Default reranker model name and threshold constants.
   - src.common.schemas: Strict `RetrievalCandidate` and `ChildChunk` schemas.
================================================================================
"""

from collections import defaultdict
from typing import Any, Dict, List, Optional, Set
import torch
from sentence_transformers import CrossEncoder

from src.common.config import config
from src.common.schemas import ChildChunk, ParentChunk, RetrievalCandidate


def get_optimal_device() -> str:
    """Detect available hardware accelerator (CUDA, Apple Silicon MPS, or CPU)."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CrossEncoderReranker:
    """Computes cross-attention relevance scores and resolves child chunks to expanded parent passages."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        """Initialize CrossEncoder model.

        Args:
            model_name: HuggingFace model identifier (default: config.models.reranker_model_name).
            device: Hardware device to run inference on ('cuda', 'mps', 'cpu').
        """
        self.model_name = model_name or config.models.reranker_model_name
        self.device = device or get_optimal_device()
        self.model = CrossEncoder(self.model_name, device=self.device)

    def rerank_and_resolve(
        self,
        query: str,
        candidate_child_ids: List[str],
        child_chunk_map: Dict[str, ChildChunk],
        local_store: Any,
        top_k: int = config.thresholds.top_k_rerank,
    ) -> List[RetrievalCandidate]:
        """Rerank candidate children, expand adjacent parent neighbors, and return top context passages.

        Args:
            query: Cleaned user search query.
            candidate_child_ids: List of candidate chunk IDs to rerank.
            child_chunk_map: Map of child_id to ChildChunk instance.
            local_store: Instance of LocalStore containing parent chunks.
            top_k: Maximum number of deduplicated, expanded parent passages to return.

        Returns:
            List of RetrievalCandidate models sorted descending by relevance score.
        """
        if not query or not candidate_child_ids:
            return []

        # Filter candidates present in the chunk map
        valid_cids = [cid for cid in candidate_child_ids if cid in child_chunk_map]
        if not valid_cids:
            return []

        # Prepare cross-encoder (query, passage) pairs
        pairs = [[query, child_chunk_map[cid].text] for cid in valid_cids]

        # Compute cross-encoder relevance scores
        scores = self.model.predict(pairs, show_progress_bar=False)

        # Pair candidates with cross-encoder scores and sort descending
        scored_candidates = list(zip(valid_cids, scores))
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Step 1: Collect unique parent passages per document with highest score
        # doc_id -> {parent_id: (ParentChunk, max_score)}
        doc_parents: Dict[str, Dict[str, tuple[ParentChunk, float]]] = defaultdict(dict)

        for cid, score in scored_candidates:
            child = child_chunk_map[cid]
            parent = local_store.get_parent(child.parent_id)
            if parent is None:
                continue

            doc_id = parent.doc_id
            if parent.parent_id not in doc_parents[doc_id]:
                doc_parents[doc_id][parent.parent_id] = (parent, float(score))
            else:
                existing_parent, existing_score = doc_parents[doc_id][parent.parent_id]
                if float(score) > existing_score:
                    doc_parents[doc_id][parent.parent_id] = (existing_parent, float(score))

        # Step 2: Perform document-aware neighbor context expansion
        expanded_candidates: List[RetrievalCandidate] = []

        for doc_id, parents_dict in doc_parents.items():
            # Sort parents by chunk_index
            sorted_parents = sorted(parents_dict.values(), key=lambda item: item[0].chunk_index)

            merged_groups: List[tuple[List[ParentChunk], float]] = []
            for parent, score in sorted_parents:
                if not merged_groups:
                    merged_groups.append(([parent], score))
                else:
                    last_group, last_score = merged_groups[-1]
                    last_parent = last_group[-1]

                    # If adjacent neighbor in the same document, merge into unified passage
                    if abs(parent.chunk_index - last_parent.chunk_index) == 1:
                        last_group.append(parent)
                        merged_groups[-1] = (last_group, max(last_score, score))
                    else:
                        merged_groups.append(([parent], score))

            # Create RetrievalCandidate for each merged group
            for group_parents, group_score in merged_groups:
                if len(group_parents) == 1:
                    single = group_parents[0]
                    candidate = RetrievalCandidate(
                        parent_id=single.parent_id,
                        doc_id=single.doc_id,
                        page_number=single.page_number,
                        text=single.text,
                        score=group_score,
                        match_type="cross_encoder_reranked",
                        chunk_index=single.chunk_index,
                    )
                else:
                    combined_text = "\n\n".join(p.text for p in group_parents)
                    combined_id = "+".join(p.parent_id for p in group_parents)
                    first_p = group_parents[0]
                    candidate = RetrievalCandidate(
                        parent_id=combined_id,
                        doc_id=first_p.doc_id,
                        page_number=first_p.page_number,
                        text=combined_text,
                        score=group_score,
                        match_type="cross_encoder_neighbor_expanded",
                        chunk_index=first_p.chunk_index,
                    )
                expanded_candidates.append(candidate)

        # Sort all candidates descending by cross-encoder relevance score
        expanded_candidates.sort(key=lambda x: x.score, reverse=True)

        num_unique_matching_docs = len({cand.doc_id for cand in expanded_candidates})

        if getattr(config.retrieval, "enable_document_diversification", True) and num_unique_matching_docs > 1:
            configured_cap = getattr(config.retrieval, "max_chunks_per_doc", 3)
            dynamic_quota = max(1, top_k // min(num_unique_matching_docs, 3))
            max_chunks_per_source = min(configured_cap, dynamic_quota) if configured_cap > 0 else dynamic_quota
            max_chunks_per_source = max(1, max_chunks_per_source)

            selected: List[RetrievalCandidate] = []
            doc_counts: Dict[str, int] = defaultdict(int)
            remaining: List[RetrievalCandidate] = []

            # 1. Quota-based diversification pass
            for cand in expanded_candidates:
                if doc_counts[cand.doc_id] < max_chunks_per_source and len(selected) < top_k:
                    selected.append(cand)
                    doc_counts[cand.doc_id] += 1
                else:
                    remaining.append(cand)

            # 2. Backfill from remaining highest-scoring candidates regardless of source
            if len(selected) < top_k:
                for cand in remaining:
                    if len(selected) >= top_k:
                        break
                    selected.append(cand)

            return selected

        return expanded_candidates[:top_k]
