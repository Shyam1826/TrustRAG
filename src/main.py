r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/main.py
   - Role: End-to-end System Orchestrator, Dual-Mode Grounding, and Ingestion Controller for TrustRAG.
   - Purpose: Coordinates all modular pipelines (Ingestion & DuckDB Tabular Store,
     Hybrid Retrieval & Tabular SQL Routing, Closed-World Generation, Claim-Level
     NLI Verification, Automated Self-Correction, and Dynamic Knowledge Gap Detection) into
     a unified, enterprise-scale, high-assurance RAG engine supporting incremental SHA-256
     manifest caching, dynamic cache invalidation, and Dual-Mode Open-World Fallback routing.

2. INPUT (IP):
   - Ingestion: pdf_path (str) or raw_dir (str) pointing to document files or subfolder trees.
   - Querying: user_query (str) representing natural language user questions.

3. PROCESS UNDER THE HOOD:
   - Dynamic Knowledge Gap Detection & Dual-Mode Routing:
     * Evaluates candidate retrieval density, score confidence floors, and lexical overlap
       prior to generation via `KnowledgeGapDetector`.
     * Closed-World Mode: Executes context compaction, XML-constrained synthesis, inline citation
       parsing, and claim-level DeBERTa NLI audits when candidate confidence is sufficient.
     * Open-World Fallback Mode: Triggers when retrieval produces no candidates, when candidate scores
       fall below the empirical floor (-6.0) lacking query overlap, or when closed-world generation
       yields an uninformative refusal. Generates disclaimed parametric answers and marks reports
       as unverified (Trust Score 0.0).
   - Incremental & Recursive Ingestion Flow:
     * Recursively traverses subfolders across supported formats (PDF, CSV, XLSX, TXT, PNG).
     * Registers structured tabular files into in-process DuckDB tables.
     * Inspects `IngestionManifest`: skips unchanged files based on SHA-256 fingerprinting and parser version.
     * Dynamic Cache Invalidation: invalidates stale cache when parser version evolves or when
       `data/processed/ingestion_manifest.json` is reset/deleted, clearing stale Qdrant points.
   - Query, Tabular Routing & Verification Flow:
     * Dispatches query via `retrieve()`: detects relational tabular intent targeting DuckDB.
     * Emits structured rows as standard `RetrievalCandidate` objects.
     * Multi-faceted query decomposition and concurrent dense/sparse/tabular search.
     * Synthesizes draft response, extracts section-anchored claims, and verifies via DeBERTa.
     * If ungrounded or contradictory claims exist: executes 1-pass automated self-correction.
     * Enriches report with exact physical PDF bounding boxes and table lineage.

4. OUTPUT (OP):
   - TrustAuditReport: Strictly typed Pydantic audit report containing draft text,
     faithfulness score, per-claim NLI audits, automated safety gate action, grounding_mode
     (CLOSED_WORLD or OPEN_WORLD_FALLBACK), trust_score, verdict, and provenance_map.

5. LIBRARIES & DEPENDENCIES:
   - concurrent.futures: Parallel dense/sparse execution.
   - pathlib.Path: File operations and recursive directory traversal.
   - re: Regex pattern matching and alias extraction.
   - sklearn.feature_extraction.text (ENGLISH_STOP_WORDS): Standard NLP stop words.
   - src.common.config, src.common.schemas: Configurations and schemas.
   - src.pipeline_1_ingestion.*, src.pipeline_2_retrieval.*,
     src.pipeline_3_generation.*, src.pipeline_4_verification.*,
     src.pipeline_5_self_correction.*: Pipeline modules.
================================================================================
"""

import concurrent.futures
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config
from src.common.schemas import ChildChunk, GroundingMode, ParentChunk, ProvenanceCoordinate, RetrievalCandidate, TrustAuditReport
from src.common.tracer import global_tracer
from src.pipeline_1_ingestion.chunker import create_hierarchical_chunks
from src.pipeline_1_ingestion.discover import discover_raw_documents
from src.pipeline_1_ingestion.embedder import DualEmbedder
from src.pipeline_1_ingestion.indexer import LocalStore
from src.pipeline_1_ingestion.manifest import IngestionManifest
from src.pipeline_1_ingestion.parser import extract_pdf_pages
from src.pipeline_1_ingestion.reader import read_document
from src.pipeline_1_ingestion.tabular_store import TabularStore
from src.database.connection import init_db
from src.database.repository import DatabaseRepository
from src.pipeline_2_retrieval.conversational_rewriter import ConversationalQueryRewriter
from src.pipeline_2_retrieval.fusion import apply_rrf, is_comparative_query
from src.pipeline_2_retrieval.gap_detector import KnowledgeGapDetector
from src.pipeline_2_retrieval.reranker import CrossEncoderReranker
from src.pipeline_2_retrieval.rewriter import QueryTransformer
from src.pipeline_2_retrieval.search_dense import retrieve_dense
from src.pipeline_2_retrieval.search_sparse import BM25Searcher
from src.pipeline_2_retrieval.tabular_engine import TabularQueryEngine
from src.pipeline_3_generation.citation_check import validate_and_parse_citations
from src.pipeline_3_generation.generator import BaseGenerator, get_generator
from src.pipeline_3_generation.prompt import build_correction_prompt, build_rag_prompt
from src.pipeline_4_verification.adjudicator import AuditAdjudicator
from src.pipeline_4_verification.claim_extractor import AtomicClaimExtractor
from src.pipeline_4_verification.nli_model import DebertaNLIVerifier
from src.pipeline_5_self_correction.corrector import SelfCorrector
from src.pipeline_5_self_correction.decomposer import QueryDecomposer


SUPPORTED_DOCUMENT_EXTENSIONS: Set[str] = set(config.ingestion.supported_extensions)


class TrustRAGPipeline:
    """End-to-end TrustRAG pipeline orchestrator with claim-level verification."""

    def __init__(
        self,
        generator_type: Optional[str] = None,
        qdrant_location: Optional[str] = None,
        qdrant_path: Optional[str] = None,
        manifest: Optional[IngestionManifest] = None,
        db_repo: Optional[DatabaseRepository] = None,
    ) -> None:
        """Initialize all pipeline components and models.

        Args:
            generator_type: Type of generation engine ('groq', 'gemini', 'mock', 'hf').
            qdrant_location: Optional Qdrant database location (e.g. ':memory:' or remote URL).
            qdrant_path: Optional on-disk directory path for local Qdrant storage (default: 'data/qdrant_db').
            manifest: Optional IngestionManifest instance for incremental tracking.
            db_repo: Optional DatabaseRepository instance for chat session persistence.
        """
        # Pipeline 1: Ingestion & Storage
        self.manifest = manifest or IngestionManifest()
        self.embedder = DualEmbedder()
        self.local_store = LocalStore(
            location=qdrant_location,
            path=qdrant_path,
            vector_size=384,
        )
        self.all_child_chunks: List[ChildChunk] = []
        self.child_chunk_map: Dict[str, ChildChunk] = {}
        self.known_doc_ids: Set[str] = set()
        self.doc_entity_map: Dict[str, str] = {}
        self.last_retrieved_contexts: List[RetrievalCandidate] = []

        # Tabular Store & Relational Engine (DuckDB in-process tabular execution)
        self.tabular_store = TabularStore()

        # Pipeline 2: Retrieval & Reranking
        self.rewriter = QueryTransformer()
        self.bm25_searcher: Optional[BM25Searcher] = None
        self.reranker = CrossEncoderReranker()

        # Pipeline 3: Generation & Citation
        self.generator_type = generator_type or config.GENERATOR_PROVIDER
        self.generator: BaseGenerator = get_generator(generator_type=self.generator_type)

        # Tabular Query Engine (using registered tables and active generator)
        self.tabular_engine = TabularQueryEngine(
            tabular_store=self.tabular_store,
            generator=self.generator,
        )

        # Knowledge Gap Detector (Dual-Mode Safety Gating)
        self.gap_detector = KnowledgeGapDetector()

        # Pipeline 4: Verification & Adjudication
        self.claim_extractor = AtomicClaimExtractor()
        self.nli_verifier = DebertaNLIVerifier()
        self.adjudicator = AuditAdjudicator()

        # Pipeline 5: Self-Correction & Sub-Query Decomposition
        self.decomposer = QueryDecomposer()
        self.corrector = SelfCorrector()

        # Database Ledger & Multi-Turn Persistence
        init_db()
        self.db_repo = db_repo or DatabaseRepository()

        # Conversational Query Reformulation
        self.conversational_rewriter = ConversationalQueryRewriter(generator=self.generator)

        # Retrieval Diagnostics & Execution Tracer State
        self.last_route = "Hybrid Narrative"
        self.tracer = global_tracer
        self.last_trace: List[Dict[str, Any]] = []
        self.startup_trace: List[Dict[str, Any]] = []

    def _extract_entity_aliases(self, pages: List[Dict], assigned_doc_id: str) -> List[str]:
        """Extract candidate entity identifiers and document aliases using domain-agnostic NLP filtering."""
        custom_stops = set(config.retrieval.custom_stop_words)
        all_stops = ENGLISH_STOP_WORDS | custom_stops
        aliases = set()

        # Path and token aliases
        for token in re.split(r"[/\\__ -]+", assigned_doc_id):
            t_clean = token.lower().strip()
            if len(t_clean) >= 3 and t_clean not in all_stops and t_clean.isalnum():
                aliases.add(t_clean)

        # Sanitized double-underscore alias
        sanitized_doc_id = assigned_doc_id.replace("/", "__").replace("\\", "__").lower()
        if len(sanitized_doc_id) >= 3 and sanitized_doc_id not in all_stops:
            aliases.add(sanitized_doc_id)

        if pages:
            first_page_text = pages[0].get("raw_text", "")
            lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

            # Extract from emails (e.g. user@domain.com -> domain keywords)
            emails = re.findall(r"([a-zA-Z0-9_.+-]+)@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", first_page_text)
            for email in emails:
                for part in re.split(r"[0-9_.]+", email.lower()):
                    if len(part) >= 3 and part not in all_stops and part.isalpha():
                        aliases.add(part)

            # Prominent standalone entity names from header lines
            for line in lines[:5]:
                if not any(c.isdigit() for c in line) and ":" not in line and len(line.split()) <= 4:
                    for w in line.split():
                        w_l = w.lower().strip(".,;:()")
                        if len(w_l) >= 3 and w_l not in all_stops and w_l.isalpha():
                            aliases.add(w_l)

        return list(aliases)

    def ingest_document(
        self,
        file_path: str,
        doc_id: Optional[str] = None,
        relative_path: Optional[str] = None,
        folder_hierarchy: Optional[List[str]] = None,
        force_reindex: bool = False,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[ChildChunk]:
        """Ingest, chunk, embed, and index a document (PDF, Excel .xlsx/.xls, CSV, TXT) with manifest caching and tenant isolation.

        Args:
            file_path: File system path to the document.
            doc_id: Optional unique identifier for the document (defaults to relative stem).
            relative_path: Optional full relative subfolder path.
            folder_hierarchy: Optional list of parent folder categories.
            force_reindex: If True, bypass manifest check and re-index.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            List of indexed ChildChunk models.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"Document file not found at: {file_path}")

        try:
            rel = path.relative_to("data/raw")
            computed_doc_id = str(rel.with_suffix("")).replace("\\", "/")
            computed_rel_str = str(rel).replace("\\", "/")
            computed_folders = [p for p in rel.parent.parts if p and p != "."]
        except (ValueError, Exception):
            try:
                rel = path.relative_to("data")
                if rel.parts and rel.parts[0] == "raw" and len(rel.parts) > 1:
                    rel_for_doc = Path(*rel.parts[1:])
                else:
                    rel_for_doc = rel
                computed_doc_id = str(rel_for_doc.with_suffix("")).replace("\\", "/")
                computed_rel_str = str(rel_for_doc).replace("\\", "/")
                computed_folders = [p for p in rel_for_doc.parent.parts if p and p != "."]
            except Exception:
                computed_doc_id = path.stem
                computed_rel_str = path.name
                computed_folders = []

        assigned_doc_id = doc_id or computed_doc_id
        doc_rel_path = relative_path or computed_rel_str
        doc_folders = folder_hierarchy if folder_hierarchy is not None else computed_folders

        # Step 0: Register in TabularStore if tabular file (hydrate DuckDB in-memory tables)
        ext = path.suffix.lower()
        if ext in (".csv", ".tsv", ".xlsx", ".xls"):
            try:
                self.tabular_store.register_table_from_file(
                    path,
                    assigned_doc_id,
                    user_id=user_id,
                    thread_id=thread_id,
                )
            except Exception as e:
                print(f"[TabularStore] Warning: Failed to register {path.name} in DuckDB: {e}")

        # Step 0b: Check Incremental Manifest Fingerprint (for default or unassigned tenants)
        if (user_id is None or user_id == "default_user") and not force_reindex and self.manifest.is_indexed_and_current(path, assigned_doc_id):
            print(f"[Ingestion] '{assigned_doc_id}' unchanged (already indexed) -> Skipping.")
            self.known_doc_ids.add(assigned_doc_id)
            return []

        # If document was previously indexed with different content, clear stale points
        self.local_store.delete_document(assigned_doc_id, user_id=user_id, thread_id=thread_id)
        # Clear local child chunks for this doc_id and tenant
        self.all_child_chunks = [
            c for c in self.all_child_chunks
            if not (c.doc_id == assigned_doc_id and (user_id is None or getattr(c, "user_id", None) == user_id))
        ]
        self.child_chunk_map = {
            cid: c for cid, c in self.child_chunk_map.items()
            if not (c.doc_id == assigned_doc_id and (user_id is None or getattr(c, "user_id", None) == user_id))
        }

        # Step 1: Extract pages/sheets via format-specific reader
        pages = read_document(
            path,
            doc_id=assigned_doc_id,
            tabular_store=self.tabular_store,
            user_id=user_id,
            thread_id=thread_id,
        )
        if not pages:
            print(f"Warning: No readable text extracted from {path.name}.")
            return []

        # Extract entity aliases
        entity_aliases = self._extract_entity_aliases(pages, assigned_doc_id)
        for alias in entity_aliases:
            self.doc_entity_map[alias] = assigned_doc_id

        # Step 2: Hierarchical Structural Chunking
        parents, children = create_hierarchical_chunks(
            pages=pages,
            doc_id=assigned_doc_id,
            parent_size=config.chunking.parent_size,
            parent_overlap=config.chunking.parent_overlap,
            child_size=config.chunking.child_size,
            overlap=config.chunking.overlap,
        )

        if not children:
            return []

        # Attach folder hierarchy, relative paths & tenant coordinates to chunk metadata
        for p in parents:
            p.relative_path = doc_rel_path
            p.folder_hierarchy = doc_folders
            p.user_id = user_id
            p.thread_id = thread_id
        for c in children:
            c.relative_path = doc_rel_path
            c.folder_hierarchy = doc_folders
            c.user_id = user_id
            c.thread_id = thread_id

        # Step 3: Embed Dense and Sparse
        child_texts = [child.text for child in children]
        dense_vectors = self.embedder.embed_dense(child_texts)

        for child, vector in zip(children, dense_vectors):
            child.vector = vector
            child.sparse_tokens = self.embedder.tokenize_sparse(child.text)
            self.child_chunk_map[child.chunk_id] = child

        self.all_child_chunks.extend(children)
        self.known_doc_ids.add(assigned_doc_id)

        # Step 4: Index into Qdrant & Parent Cache
        self.local_store.upsert(children, parents=parents)

        # Step 5: Update BM25 Inverted Index
        self.bm25_searcher = BM25Searcher(self.all_child_chunks)

        # Step 6: Record in Manifest
        if user_id is None or user_id == "default_user":
            self.manifest.record_indexed(
                path=path,
                doc_id=assigned_doc_id,
                chunk_count=len(children),
                metadata={"relative_path": doc_rel_path, "folders": doc_folders},
            )

        print(f"Successfully indexed {len(children)} chunks from {path.name} (doc_id: '{assigned_doc_id}').")
        return children

    def ingest_pdf(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        relative_path: Optional[str] = None,
        folder_hierarchy: Optional[List[str]] = None,
        force_reindex: bool = False,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[ChildChunk]:
        """Backward-compatible alias for ingest_document."""
        return self.ingest_document(
            file_path=pdf_path,
            doc_id=doc_id,
            relative_path=relative_path,
            folder_hierarchy=folder_hierarchy,
            force_reindex=force_reindex,
            user_id=user_id,
            thread_id=thread_id,
        )

    def ingest_directory(
        self,
        raw_dir: str = "data/raw",
        supported_extensions: Optional[Set[str]] = None,
        force_reindex: bool = False,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[ChildChunk]:
        """Recursively discover and ingest all supported document files across subfolders in raw_dir with tenant scoping.

        Args:
            raw_dir: Path to base raw data directory.
            supported_extensions: Optional set of allowed file extensions.
            force_reindex: If True, bypass manifest and re-index all files.
            user_id: Optional tenant user_id.
            thread_id: Optional tenant thread_id.

        Returns:
            List of all indexed ChildChunk models.
        """
        exts = supported_extensions or set(config.ingestion.supported_extensions)
        with self.tracer.trace_step("Document Discovery", "src/pipeline_1_ingestion/reader.py", "discover_raw_documents"):
            discovered = discover_raw_documents(raw_dir=raw_dir, supported_extensions=exts)
        print(f"[Ingestion] Discovered {len(discovered)} document(s) across subfolders in '{raw_dir}'.")

        if force_reindex:
            print("[Ingestion] Force re-index requested: invalidating manifest cache.")
            self.manifest.invalidate()
            self.all_child_chunks.clear()
            self.child_chunk_map.clear()
            self.known_doc_ids.clear()

        all_indexed: List[ChildChunk] = []
        indexed_count = 0
        skipped_count = 0

        for file_path, doc_id, rel_str, folders in discovered:
            if file_path.suffix.lower() in exts:
                # Ensure tabular file is registered in DuckDB in-memory store
                if file_path.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls"):
                    try:
                        self.tabular_store.register_table_from_file(
                            file_path,
                            doc_id,
                            user_id=user_id,
                            thread_id=thread_id,
                        )
                    except Exception as e:
                        print(f"[TabularStore] Warning: Failed to register {file_path.name} in DuckDB: {e}")

                if (user_id is None or user_id == "default_user") and not force_reindex and self.manifest.is_indexed_and_current(file_path, doc_id):
                    print(f"[Ingestion] '{doc_id}' unchanged (already indexed) -> Skipping.")
                    self.known_doc_ids.add(doc_id)
                    skipped_count += 1
                    if file_path.suffix.lower() == ".pdf":
                        try:
                            extract_pdf_pages(
                                str(file_path),
                                doc_id=doc_id,
                                tabular_store=self.tabular_store,
                                user_id=user_id,
                                thread_id=thread_id,
                            )
                        except Exception:
                            pass
                else:
                    chunks = self.ingest_document(
                        str(file_path),
                        doc_id=doc_id,
                        relative_path=rel_str,
                        folder_hierarchy=folders,
                        force_reindex=force_reindex,
                        user_id=user_id,
                        thread_id=thread_id,
                    )
                    all_indexed.extend(chunks)
                    indexed_count += 1

        # If any files were skipped or in-memory chunks are empty, hydrate from persistent vector store
        if skipped_count > 0 or len(self.all_child_chunks) == 0:
            stored_chunks = self.local_store.load_all_child_chunks(user_id=user_id, thread_id=thread_id)
            if stored_chunks:
                self.all_child_chunks = stored_chunks
                self.child_chunk_map = {c.chunk_id: c for c in stored_chunks}
                for c in stored_chunks:
                    self.known_doc_ids.add(c.doc_id)
                if self.bm25_searcher is None:
                    self.bm25_searcher = BM25Searcher(stored_chunks)
                else:
                    self.bm25_searcher.index_documents(stored_chunks)
                print(f"[State Hydration] Hydrated {len(stored_chunks)} chunks and BM25 index from persistent vector store.")
        elif self.bm25_searcher is None and len(self.all_child_chunks) > 0:
            self.bm25_searcher = BM25Searcher(self.all_child_chunks)

        print(f"[Ingestion] Ingestion summary: {len(discovered)} scanned, {indexed_count} indexed/re-indexed, {skipped_count} unchanged.")
        self.startup_trace = self.tracer.get_trace()
        return all_indexed

    def _execute_sub_query(
        self,
        sub_q: str,
        top_k: int,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> Tuple[List[RetrievalCandidate], List[Tuple[str, int, float]], List[Tuple[str, int, float]]]:
        """Execute concurrent tabular check, dense search, and sparse search for an individual sub-query."""
        sub_tabular_cands: List[RetrievalCandidate] = []
        if self.tabular_engine.is_tabular_query(sub_q, user_id=user_id, thread_id=thread_id):
            try:
                sub_tabular_cands = self.tabular_engine.query(sub_q, user_id=user_id, thread_id=thread_id)
            except Exception as e:
                print(f"[TabularEngine] Sub-query execution error: {e}")

        sub_doc_filter, sub_matched_entities = self.rewriter.extract_doc_filter(
            sub_q,
            list(self.known_doc_ids),
            doc_entity_map=self.doc_entity_map,
        )
        sub_clean_q = self.rewriter.transform(sub_q, entities_to_strip=sub_matched_entities)
        sub_tokens = [t.lower() for t in sub_clean_q.split() if t.strip()]

        dense_ranks: List[Tuple[str, int, float]] = []
        sparse_ranks: List[Tuple[str, int, float]] = []

        if self.all_child_chunks or (self.local_store and self.local_store.client):
            dense_ranks = retrieve_dense(
                sub_clean_q,
                self.embedder,
                self.local_store.client,
                top_k=top_k,
                doc_filter=sub_doc_filter,
                user_id=user_id,
                thread_id=thread_id,
            )

        if self.bm25_searcher and sub_tokens:
            sparse_ranks = self.bm25_searcher.search(
                sub_tokens,
                top_k=top_k,
                doc_filter=sub_doc_filter,
                user_id=user_id,
                thread_id=thread_id,
            )

        return sub_tabular_cands, dense_ranks, sparse_ranks

    def _retrieve_narrative(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        """Execute hybrid dense-sparse retrieval, RRF fusion, and cross-encoder reranking on narrative chunks.

        Args:
            user_query: User search query.
            user_id: Optional tenant user identifier for isolation.
            thread_id: Optional session thread identifier for isolation.

        Returns:
            List of reranked RetrievalCandidate models.
        """
        top_k_dense = config.retrieval.top_k_dense
        with self.tracer.trace_step("Query Analysis", "src/pipeline_2_retrieval/rewriter.py", "rewrite_query"):
            doc_filter, matched_entities = self.rewriter.extract_doc_filter(
                user_query,
                list(self.known_doc_ids),
                doc_entity_map=self.doc_entity_map,
            )
            clean_query = self.rewriter.transform(user_query, entities_to_strip=matched_entities)
            query_tokens = [t.lower() for t in clean_query.split() if t.strip()]

        dense_results: List[Tuple[str, int, float]] = []
        sparse_results: List[Tuple[str, int, float]] = []

        with self.tracer.trace_step("Dense/Sparse Search", "src/pipeline_2_retrieval/search_dense.py", "search"):
            if self.all_child_chunks or (self.local_store and self.local_store.client):
                dense_results = retrieve_dense(
                    clean_query,
                    self.embedder,
                    self.local_store.client,
                    top_k=top_k_dense,
                    doc_filter=doc_filter,
                    user_id=user_id,
                    thread_id=thread_id,
                )

            if self.bm25_searcher and query_tokens:
                sparse_results = self.bm25_searcher.search(
                    query_tokens,
                    top_k=top_k_dense,
                    doc_filter=doc_filter,
                    user_id=user_id,
                    thread_id=thread_id,
                )

        with self.tracer.trace_step("Rank Fusion", "src/pipeline_2_retrieval/fusion.py", "fuse_rankings"):
            fused_candidates = apply_rrf(
                dense_ranks=dense_results,
                sparse_ranks=sparse_results,
                k=config.retrieval.rrf_k,
                top_n=top_k_dense,
                child_chunk_map=self.child_chunk_map,
                query=clean_query,
                doc_filter=doc_filter,
                user_id=user_id,
                thread_id=thread_id,
            )
            candidate_cids = [cid for cid, _ in fused_candidates]

        with self.tracer.trace_step("Reranking", "src/pipeline_2_retrieval/reranker.py", "rerank"):
            return self.reranker.rerank_and_resolve(
                query=clean_query,
                candidate_child_ids=candidate_cids,
                child_chunk_map=self.child_chunk_map,
                local_store=self.local_store,
                top_k=config.retrieval.top_k_rerank,
                doc_filter=doc_filter,
                user_id=user_id,
                thread_id=thread_id,
            )

    def retrieve(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> List[RetrievalCandidate]:
        """Retrieve relevant context candidates across relational tabular and narrative documents using concurrent execution.

        Args:
            user_query: Natural language query string.
            user_id: Optional tenant user identifier for isolation.
            thread_id: Optional session thread identifier for isolation.

        Returns:
            List of RetrievalCandidate models.
        """
        if not user_query or not user_query.strip():
            return []

        # 1. Multi-Faceted Query Decomposition
        sub_queries = self.decomposer.decompose(user_query)
        top_k_dense = config.retrieval.top_k_dense
        top_k_rerank = config.retrieval.top_k_rerank
        max_per_doc = config.retrieval.max_chunks_per_doc or 3

        # Branch A: Multi-Faceted Compound Queries (Parallel Retrieval & Round-Robin Allocation)
        if len(sub_queries) > 1:
            sub_results: List[List[RetrievalCandidate]] = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                def _fetch_sub_intent(sq: str) -> List[RetrievalCandidate]:
                    if self.tabular_engine.is_tabular_query(sq, user_id=user_id, thread_id=thread_id):
                        try:
                            tab = self.tabular_engine.query(sq, user_id=user_id, thread_id=thread_id)
                            if tab:
                                return tab
                        except Exception as ex:
                            print(f"[TabularEngine] Sub-query error: {ex}")
                    return self._retrieve_narrative(sq, user_id=user_id, thread_id=thread_id)

                futures = {executor.submit(_fetch_sub_intent, sq): sq for sq in sub_queries}
                for future in concurrent.futures.as_completed(futures):
                    try:
                        cands = future.result()
                        if cands:
                            sub_results.append(cands)
                    except Exception as e:
                        sq_name = futures[future]
                        print(f"[ConcurrentRetrieval] Error executing sub-query '{sq_name}': {e}")

            # Interleave / Round-Robin selection guarantees each sub-query contributes its top match
            merged_candidates: List[RetrievalCandidate] = []
            seen_ids: Set[str] = set()
            doc_counts: Dict[str, int] = {}

            max_depth = max((len(r) for r in sub_results), default=0)
            for depth in range(max_depth):
                for r_list in sub_results:
                    if depth < len(r_list):
                        cand = r_list[depth]
                        cid = cand.parent_id or cand.text
                        d_id = cand.doc_id
                        if cid not in seen_ids and doc_counts.get(d_id, 0) < max_per_doc:
                            seen_ids.add(cid)
                            doc_counts[d_id] = doc_counts.get(d_id, 0) + 1
                            merged_candidates.append(cand)
                            if len(merged_candidates) >= top_k_rerank:
                                return merged_candidates

            # Backfill if quota restrictions leave available top-k slots unfilled
            if len(merged_candidates) < top_k_rerank:
                for r_list in sub_results:
                    for cand in r_list:
                        cid = cand.parent_id or cand.text
                        if cid not in seen_ids:
                            seen_ids.add(cid)
                            merged_candidates.append(cand)
                            if len(merged_candidates) >= top_k_rerank:
                                self.last_route = "DuckDB SQL | Hybrid Narrative" if any(c.match_type == "tabular_sql" for c in merged_candidates) else "Hybrid Narrative"
                                return merged_candidates

            self.last_route = "DuckDB SQL | Hybrid Narrative" if any(c.match_type == "tabular_sql" for c in merged_candidates) else "Hybrid Narrative"
            return merged_candidates[:top_k_rerank]

        # Branch B: Solitary Query Execution
        # Check tabular intent first (if pure relational tabular query, execute directly)
        is_tabular = self.tabular_engine.is_tabular_query(user_query, user_id=user_id, thread_id=thread_id)
        if is_tabular and not is_comparative_query(user_query):
            try:
                tab_candidates = self.tabular_engine.query(user_query, user_id=user_id, thread_id=thread_id)
                if tab_candidates:
                    self.last_route = "DuckDB SQL"
                    return tab_candidates
            except Exception as e:
                print(f"[TabularEngine] Query execution failed: {e}. Falling back to narrative search.")

        all_tabular_candidates: List[RetrievalCandidate] = []
        all_dense_ranks: List[Tuple[str, int, float]] = []
        all_sparse_ranks: List[Tuple[str, int, float]] = []

        with self.tracer.trace_step("Query Analysis", "src/pipeline_2_retrieval/rewriter.py", "rewrite_query"):
            doc_filter, matched_entities = self.rewriter.extract_doc_filter(
                user_query,
                list(self.known_doc_ids),
                doc_entity_map=self.doc_entity_map,
            )
            clean_query = self.rewriter.transform(user_query, entities_to_strip=matched_entities)

        # Execute dense, sparse, and tabular checks concurrently
        with self.tracer.trace_step("Dense/Sparse Search", "src/pipeline_2_retrieval/search_dense.py", "search"):
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                fut = executor.submit(self._execute_sub_query, user_query, top_k_dense, user_id, thread_id)
                tab_cands, dense_r, sparse_r = fut.result()
                all_tabular_candidates.extend(tab_cands)
                all_dense_ranks.extend(dense_r)
                all_sparse_ranks.extend(sparse_r)

        # Thread-safe quota deduplication of tabular candidates
        deduped_tabular: List[RetrievalCandidate] = []
        seen_tabular_ids: Set[str] = set()
        for cand in all_tabular_candidates:
            cid = cand.parent_id or cand.text
            if cid not in seen_tabular_ids:
                seen_tabular_ids.add(cid)
                deduped_tabular.append(cand)

        # Reciprocal Rank Fusion on narrative dense/sparse ranks
        narrative_candidates: List[RetrievalCandidate] = []
        if all_dense_ranks or all_sparse_ranks:
            with self.tracer.trace_step("Rank Fusion", "src/pipeline_2_retrieval/fusion.py", "fuse_rankings"):
                fused_candidates = apply_rrf(
                    dense_ranks=all_dense_ranks,
                    sparse_ranks=all_sparse_ranks,
                    k=config.retrieval.rrf_k,
                    top_n=top_k_dense,
                    child_chunk_map=self.child_chunk_map,
                    query=clean_query,
                    doc_filter=doc_filter,
                    user_id=user_id,
                    thread_id=thread_id,
                )
                candidate_cids = [cid for cid, _ in fused_candidates]

            # Cross-Encoder Reranking and Neighbor Expansion
            with self.tracer.trace_step("Reranking", "src/pipeline_2_retrieval/reranker.py", "rerank"):
                narrative_candidates = self.reranker.rerank_and_resolve(
                    query=clean_query,
                    candidate_child_ids=candidate_cids,
                    child_chunk_map=self.child_chunk_map,
                    local_store=self.local_store,
                    top_k=top_k_rerank,
                    doc_filter=doc_filter,
                    user_id=user_id,
                    thread_id=thread_id,
                )

        # Merge candidates: tabular candidates + narrative candidates
        if deduped_tabular and narrative_candidates:
            self.last_route = "DuckDB SQL | Hybrid Narrative"
            merged = deduped_tabular + narrative_candidates
            return merged[:top_k_rerank]
        elif deduped_tabular:
            self.last_route = "DuckDB SQL"
            return deduped_tabular[:top_k_rerank]
        else:
            self.last_route = "Hybrid Narrative"
            return narrative_candidates

    def ask(
        self,
        user_query: str,
        user_id: Optional[str] = None,
        thread_id: Optional[str] = None,
    ) -> TrustAuditReport:
        """Process a user query with dynamic scope routing, neighbor expansion, and NLI verification.

        Args:
            user_query: Raw user search query.
            user_id: Optional tenant user identifier for isolation.
            thread_id: Optional session thread identifier for isolation.

        Returns:
            TrustAuditReport containing draft text, per-claim audits, and safety action.
        """
        self.tracer.clear()

        if not user_query or not user_query.strip():
            self.last_trace = self.tracer.get_trace()
            return TrustAuditReport(
                draft_text="",
                faithfulness_score=1.0,
                has_contradiction=False,
                action="PASS",
                audits=[],
                grounding_mode=GroundingMode.CLOSED_WORLD,
                trust_score=1.0,
                verdict="PASS",
                source_attribution="EMPTY_QUERY",
            )

        # 1. Execute Unified Retrieval (Tabular + Narrative Fusion with Sub-Query Decomp)
        top_contexts = self.retrieve(user_query, user_id=user_id, thread_id=thread_id)
        self.last_retrieved_contexts = top_contexts

        # Dynamic Knowledge Gap Detection & Dual-Mode Routing
        gap_detected, gap_reason = self.gap_detector.detect_gap(user_query, top_contexts)

        draft_text = ""
        if not gap_detected:
            # 2. Taxonomy-Strict Prompt Construction & Context Compaction
            with self.tracer.trace_step("Context Compaction", "src/pipeline_3_generation/compaction.py", "compact"):
                prompt = build_rag_prompt(query=user_query.strip(), contexts=top_contexts)

            # 3. Closed-World Draft Generation
            with self.tracer.trace_step("Answer Generation", "src/pipeline_3_generation/generator.py", "generate_answer"):
                draft_text = self.generator.generate_answer(prompt)

            # Check if closed-world generator outputs an uninformative refusal response
            if self.gap_detector.is_refusal_response(draft_text):
                gap_detected = True
                gap_reason = "CLOSED_WORLD_REFUSAL"

        if gap_detected:
            # Open-World Fallback Mode
            with self.tracer.trace_step("Open-World Generation", "src/pipeline_3_generation/generator.py", "generate_open_world"):
                open_world_draft = self.generator.generate_open_world(user_query)

            audit_report = TrustAuditReport(
                draft_text=open_world_draft,
                faithfulness_score=0.0,
                has_contradiction=False,
                action="UNVERIFIED_OPEN_WORLD",
                audits=[],
                grounding_mode=GroundingMode.OPEN_WORLD_FALLBACK,
                trust_score=0.0,
                verdict="UNVERIFIED_OPEN_WORLD",
                source_attribution="OPEN_WORLD_GENERAL_KNOWLEDGE",
                gap_reason=gap_reason,
            )
            self.last_trace = self.tracer.get_trace()
            return audit_report

        # 4. Citation Parsing & Boundary Validation (with Unicode normalization)
        generated_draft = validate_and_parse_citations(
            draft_text=draft_text,
            max_valid_doc_id=len(top_contexts),
        )

        # 5. Section-Anchored Atomic Claim Decomposition
        default_section = top_contexts[0].section_name if top_contexts else None
        claims = self.claim_extractor.extract_claims(generated_draft, default_section=default_section)

        # 6. Premise Mapping (providing candidate models with full parent context and document metadata)
        context_map: Dict[str, Any] = {
            f"Doc-{idx}": context
            for idx, context in enumerate(top_contexts, start=1)
        }

        # 7. NLI Auditing and Adjudication (with fixed citation-to-premise routing)
        with self.tracer.trace_step("NLI Verification", "src/pipeline_4_verification/adjudicator.py", "verify"):
            audit_report = self.adjudicator.verify(
                claims=claims,
                context_map=context_map,
                nli_verifier=self.nli_verifier,
                draft_text=draft_text,
                contexts=top_contexts,
            )

        # 8. Automated Self-Correction Rewrite Loop (1-pass) with Selective Claim Pruning
        audit_report = self.corrector.correct(
            query=user_query,
            draft_text=draft_text,
            audit_report=audit_report,
            top_contexts=top_contexts,
            generator=self.generator,
            claim_extractor=self.claim_extractor,
            adjudicator=self.adjudicator,
            nli_verifier=self.nli_verifier,
            context_map=context_map,
            default_section=default_section,
        )

        # Check if corrected draft is an uninformative refusal response
        if self.gap_detector.is_refusal_response(audit_report.draft_text):
            with self.tracer.trace_step("Open-World Generation", "src/pipeline_3_generation/generator.py", "generate_open_world"):
                open_world_draft = self.generator.generate_open_world(user_query)
            audit_report = TrustAuditReport(
                draft_text=open_world_draft,
                faithfulness_score=0.0,
                has_contradiction=False,
                action="UNVERIFIED_OPEN_WORLD",
                audits=[],
                grounding_mode=GroundingMode.OPEN_WORLD_FALLBACK,
                trust_score=0.0,
                verdict="UNVERIFIED_OPEN_WORLD",
                source_attribution="OPEN_WORLD_GENERAL_KNOWLEDGE",
                gap_reason="CLOSED_WORLD_REFUSAL",
            )
            self.last_trace = self.tracer.get_trace()
            return audit_report

        # 9. Enrich Audit Report with Exact Provenance Lineage Coordinates
        provenance_map: Dict[str, ProvenanceCoordinate] = {}
        for idx, context in enumerate(top_contexts, start=1):
            doc_key = f"Doc-{idx}"
            if hasattr(context, "provenance") and context.provenance is not None:
                provenance_map[doc_key] = context.provenance
            else:
                provenance_map[doc_key] = ProvenanceCoordinate(
                    doc_id=context.doc_id,
                    source_type="pdf" if getattr(context, "bbox", None) else "text",
                    page=context.page_number,
                    section_name=context.section_name,
                    bbox=getattr(context, "bbox", None),
                    snippet=context.text[:200] if context.text else None,
                )
        audit_report.provenance_map = provenance_map

        # 10. Record Closed-World Grounding State and Trust Metrics
        audit_report.grounding_mode = GroundingMode.CLOSED_WORLD
        audit_report.trust_score = audit_report.faithfulness_score
        audit_report.verdict = audit_report.action
        audit_report.source_attribution = "DOCUMENT_VAULT"
        audit_report.gap_reason = "SUFFICIENT_EVIDENCE"

        self.last_trace = self.tracer.get_trace()
        return audit_report

    def chat(
        self,
        user_query: str,
        user_id: str,
        thread_id: str,
    ) -> TrustAuditReport:
        """Process a conversational multi-turn query with coreference reformulation and ledger persistence.

        Args:
            user_query: Natural language query or follow-up from user.
            user_id: Tenant user identifier.
            thread_id: Conversational thread session identifier.

        Returns:
            TrustAuditReport containing draft text, per-claim audits, and safety action.

        Raises:
            ValueError: If user_id or thread_id is missing or empty.
        """
        if not user_id or not str(user_id).strip():
            raise ValueError("user_id must be provided for conversational chat.")
        if not thread_id or not str(thread_id).strip():
            raise ValueError("thread_id must be provided for conversational chat.")

        clean_user_id = str(user_id).strip()
        clean_thread_id = str(thread_id).strip()

        if not user_query or not user_query.strip():
            return TrustAuditReport(
                draft_text="",
                faithfulness_score=1.0,
                has_contradiction=False,
                action="PASS",
                audits=[],
            )

        # 1. Ensure user and thread exist in database ledger
        user = self.db_repo.get_user_by_id(clean_user_id)
        if not user:
            user = self.db_repo.create_user(
                email=f"{clean_user_id}@tenant.trustrag",
                password="tenant_default_password",
                full_name=f"Tenant {clean_user_id}",
                user_id=clean_user_id,
            )

        thread = self.db_repo.get_thread(clean_thread_id, clean_user_id)
        if not thread:
            thread = self.db_repo.create_thread(
                user_id=clean_user_id,
                title=user_query[:50],
                thread_id=clean_thread_id,
            )

        # 2. Fetch recent message history for thread_id
        raw_messages = self.db_repo.get_thread_messages(clean_thread_id, clean_user_id, limit=6)
        chat_history = [{"role": m.role, "content": m.content} for m in raw_messages]

        # 3. Log the incoming user message
        self.db_repo.add_message(
            thread_id=clean_thread_id,
            user_id=clean_user_id,
            role="user",
            content=user_query,
        )

        # 4. Invoke ConversationalQueryRewriter to produce standalone search query
        standalone_query = self.conversational_rewriter.rewrite_query(user_query, chat_history)

        # 5. Execute unified retrieval, prompt construction, and verification via ask()
        report = self.ask(standalone_query, user_id=clean_user_id, thread_id=clean_thread_id)

        # 6. Log final assistant response and its full audit report into database ledger
        self.db_repo.add_message(
            thread_id=clean_thread_id,
            user_id=clean_user_id,
            role="assistant",
            content=report.draft_text,
            citations=self.last_retrieved_contexts,
            audit_report=report,
        )

        return report

    def close(self) -> None:
        """Close storage handles and clean up pipeline resources."""
        if hasattr(self, "tabular_store") and self.tabular_store is not None:
            self.tabular_store.close()
        if hasattr(self, "local_store") and self.local_store is not None:
            self.local_store.close()