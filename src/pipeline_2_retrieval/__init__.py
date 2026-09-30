"""Pipeline 2: Hybrid Dense-Sparse Retrieval, Cross-Encoder Reranking, and Scope Routing."""

from src.pipeline_2_retrieval.fusion import apply_rrf, apply_rrf_fusion, is_comparative_query
from src.pipeline_2_retrieval.reranker import CrossEncoderReranker
from src.pipeline_2_retrieval.rewriter import QueryTransformer
from src.pipeline_2_retrieval.router import ScopeRouter, extract_document_scope, is_tabular_query
from src.pipeline_2_retrieval.search_dense import DenseSearcher, retrieve_dense
from src.pipeline_2_retrieval.search_sparse import BM25Searcher
from src.pipeline_2_retrieval.tabular_engine import TabularQueryEngine

__all__ = [
    "apply_rrf",
    "apply_rrf_fusion",
    "is_comparative_query",
    "CrossEncoderReranker",
    "QueryTransformer",
    "ScopeRouter",
    "extract_document_scope",
    "is_tabular_query",
    "DenseSearcher",
    "retrieve_dense",
    "BM25Searcher",
    "TabularQueryEngine",
]

