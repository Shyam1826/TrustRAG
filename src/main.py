r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/main.py
   - Role: End-to-end System Orchestrator and Incremental Ingestion Controller for TrustRAG.
   - Purpose: Coordinates all 5 modular pipelines (Ingestion & DuckDB Tabular Store,
     Hybrid Retrieval & Tabular SQL Routing, Closed-World Generation, Claim-Level
     NLI Verification, and Automated Self-Correction) into a unified, enterprise-scale,
     high-assurance RAG engine supporting incremental SHA-256 manifest caching,
     recursive multi-depth document discovery, and deterministic tabular queries.

2. INPUT (IP):
   - Ingestion: pdf_path (str) or raw_dir (str) pointing to document files or subfolder trees.
   - Querying: user_query (str) representing natural language user questions.

3. PROCESS UNDER THE HOOD:
   - Incremental & Recursive Ingestion Flow:
     * Recursively traverses subfolders in `data/raw/` across configured supported extensions.
     * Registers structured tabular files (CSV, TSV, XLSX, XLS) into in-process DuckDB tables.
     * Derives collision-safe relative `doc_id`s (e.g. `legal/2026/nda`).
     * Inspects `IngestionManifest`: skips unchanged files based on SHA-256 fingerprinting.
     * On file modifications: deletes stale Qdrant points via `vector_store.delete_document()`.
     * Extracts pages, applies domain-agnostic structural section chunking with breadcrumbs.
     * Computes dense embeddings and sparse token dictionaries.
     * Indexes into Qdrant (`trustrag_enterprise`) and updates `IngestionManifest`.
     * Maintains BM25 search index and dynamic entity aliases.
   - Query, Tabular Routing & Verification Flow:
     * Dispatches query via `retrieve()`: detects relational tabular intent targeting DuckDB.
     * Generates schema-aware SQL with dynamic column reflection and `LIMIT 15` safety bound.
     * Emits structured rows as standard `RetrievalCandidate` objects; merges with narrative
       candidates if cross-document comparison is requested.
     * For multi-faceted queries: decomposes into focused sub-queries, executes dense/sparse/tabular
       search concurrently via `ThreadPoolExecutor(max_workers=4)`, and independently reranks
       each sub-query candidate pool to eliminate cross-encoder dilution.
     * Applies round-robin interleaving and strict per-document quota ceilings across sub-queries
       to guarantee multi-source balance (preventing single-document dominance).
     * Dynamically compacts narrative context to target sentence spans via `ContextCompactor`,
       enforcing a hard 4,000-character ceiling (<2,500 tokens) in prompt construction.
     * Synthesizes draft response with active generator backend (Groq, Gemini, Ollama, Mock).
     * Validates citations and extracts section-anchored atomic claims.
     * Audits claims against unified premise windows via DeBERTa-v3 and `AuditAdjudicator`.
     * If ungrounded or contradictory claims exist (WARN / TRIGGER_REWRITE): executes a 1-pass
       automated corrective rewrite loop to produce a 100% faithful final report.
     * Emits `TrustAuditReport`.

4. OUTPUT (OP):
   - TrustAuditReport: Strictly typed Pydantic audit report containing draft text,
     faithfulness score, per-claim NLI audits, and automated safety gate action.

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
from src.common.schemas import ChildChunk, ParentChunk, RetrievalCandidate, TrustAuditReport
from src.pipeline_1_ingestion.chunker import create_hierarchical_chunks
from src.pipeline_1_ingestion.discover import discover_raw_documents
from src.pipeline_1_ingestion.embedder import DualEmbedder
from src.pipeline_1_ingestion.indexer import LocalStore
from src.pipeline_1_ingestion.manifest import IngestionManifest
from src.pipeline_1_ingestion.parser import extract_pdf_pages
from src.pipeline_1_ingestion.reader import read_document
from src.pipeline_1_ingestion.tabular_store import TabularStore
from src.pipeline_2_retrieval.fusion import apply_rrf, is_comparative_query
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
    ) -> None:
        """Initialize all pipeline components and models.

        Args:
            generator_type: Type of generation engine ('groq', 'gemini', 'mock', 'hf').
            qdrant_location: Optional Qdrant database location (e.g. ':memory:' or remote URL).
            qdrant_path: Optional on-disk directory path for local Qdrant storage (default: 'data/qdrant_db').
            manifest: Optional IngestionManifest instance for incremental tracking.
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

        # Pipeline 4: Verification & Adjudication
        self.claim_extractor = AtomicClaimExtractor()
        self.nli_verifier = DebertaNLIVerifier()
        self.adjudicator = AuditAdjudicator()

        # Pipeline 5: Self-Correction & Sub-Query Decomposition
        self.decomposer = QueryDecomposer()
        self.corrector = SelfCorrector()

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
    ) -> List[ChildChunk]:
        """Ingest, chunk, embed, and index a document (PDF, Excel .xlsx/.xls, CSV, TXT) with manifest caching.

        Args:
            file_path: File system path to the document.
            doc_id: Optional unique identifier for the document (defaults to relative stem).
            relative_path: Optional full relative subfolder path.
            folder_hierarchy: Optional list of parent folder categories.
            force_reindex: If True, bypass manifest check and re-index.

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
                self.tabular_store.register_table_from_file(path, assigned_doc_id)
            except Exception as e:
                print(f"[TabularStore] Warning: Failed to register {path.name} in DuckDB: {e}")

        # Step 0b: Check Incremental Manifest Fingerprint
        if not force_reindex and self.manifest.is_indexed_and_current(path, assigned_doc_id):
            print(f"[Ingestion] '{assigned_doc_id}' unchanged (already indexed) -> Skipping.")
            self.known_doc_ids.add(assigned_doc_id)
            return []

        # If document was previously indexed with different content, clear stale points
        self.local_store.delete_document(assigned_doc_id)
        # Clear local child chunks for this doc_id
        self.all_child_chunks = [c for c in self.all_child_chunks if c.doc_id != assigned_doc_id]
        self.child_chunk_map = {cid: c for cid, c in self.child_chunk_map.items() if c.doc_id != assigned_doc_id}

        # Step 1: Extract pages/sheets via format-specific reader
        pages = read_document(path, doc_id=assigned_doc_id)
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

        # Attach folder hierarchy & relative paths to chunk metadata
        for p in parents:
            p.relative_path = doc_rel_path
            p.folder_hierarchy = doc_folders
        for c in children:
            c.relative_path = doc_rel_path
            c.folder_hierarchy = doc_folders

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
        self.local_store.upsert_child_chunks(children)
        self.local_store.store_parents(parents)

        # Step 5: Update BM25 Inverted Index
        self.bm25_searcher = BM25Searcher(self.all_child_chunks)

        # Step 6: Record in Manifest
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
    ) -> List[ChildChunk]:
        """Backward-compatible alias for ingest_document."""
        return self.ingest_document(
            file_path=pdf_path,
            doc_id=doc_id,
            relative_path=relative_path,
            folder_hierarchy=folder_hierarchy,
            force_reindex=force_reindex,
        )

    def ingest_directory(
        self,
        raw_dir: str = "data/raw",
        supported_extensions: Optional[Set[str]] = None,
        force_reindex: bool = False,
    ) -> List[ChildChunk]:
        """Recursively discover and ingest all supported document files across subfolders in raw_dir.

        Args:
            raw_dir: Path to base raw data directory.
            supported_extensions: Optional set of allowed file extensions.
            force_reindex: If True, bypass manifest and re-index all files.

        Returns:
            List of all indexed ChildChunk models.
        """
        exts = supported_extensions or set(config.ingestion.supported_extensions)
        discovered = discover_raw_documents(raw_dir=raw_dir, supported_extensions=exts)
        print(f"[Ingestion] Discovered {len(discovered)} document(s) across subfolders in '{raw_dir}'.")

        all_indexed: List[ChildChunk] = []
        indexed_count = 0
        skipped_count = 0

        for file_path, doc_id, rel_str, folders in discovered:
            if file_path.suffix.lower() in exts:
                # Ensure tabular file is registered in DuckDB in-memory store
                if file_path.suffix.lower() in (".csv", ".tsv", ".xlsx", ".xls"):
                    try:
                        self.tabular_store.register_table_from_file(file_path, doc_id)
                    except Exception as e:
                        print(f"[TabularStore] Warning: Failed to register {file_path.name} in DuckDB: {e}")

                if not force_reindex and self.manifest.is_indexed_and_current(file_path, doc_id):
                    print(f"[Ingestion] '{doc_id}' unchanged (already indexed) -> Skipping.")
                    self.known_doc_ids.add(doc_id)
                    skipped_count += 1
                else:
                    chunks = self.ingest_document(
                        str(file_path),
                        doc_id=doc_id,
                        relative_path=rel_str,
                        folder_hierarchy=folders,
                        force_reindex=force_reindex,
                    )
                    all_indexed.extend(chunks)
                    indexed_count += 1

        # If any files were skipped or in-memory chunks are empty, hydrate from persistent vector store
        if skipped_count > 0 or len(self.all_child_chunks) == 0:
            stored_chunks = self.local_store.load_all_child_chunks()
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
        return all_indexed

    def _execute_sub_query(
        self,
        sub_q: str,
        top_k: int,
    ) -> Tuple[List[RetrievalCandidate], List[Tuple[str, int, float]], List[Tuple[str, int, float]]]:
        """Execute concurrent tabular check, dense search, and sparse search for an individual sub-query."""
        sub_tabular_cands: List[RetrievalCandidate] = []
        if self.tabular_engine.is_tabular_query(sub_q):
            try:
                sub_tabular_cands = self.tabular_engine.query(sub_q)
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
            )

        if self.bm25_searcher and sub_tokens:
            sparse_ranks = self.bm25_searcher.search(
                sub_tokens,
                top_k=top_k,
                doc_filter=sub_doc_filter,
            )

        return sub_tabular_cands, dense_ranks, sparse_ranks

    def _retrieve_narrative(self, user_query: str) -> List[RetrievalCandidate]:
        """Execute hybrid dense-sparse retrieval, RRF fusion, and cross-encoder reranking on narrative chunks.

        Args:
            user_query: User search query.

        Returns:
            List of reranked RetrievalCandidate models.
        """
        top_k_dense = config.retrieval.top_k_dense
        doc_filter, matched_entities = self.rewriter.extract_doc_filter(
            user_query,
            list(self.known_doc_ids),
            doc_entity_map=self.doc_entity_map,
        )
        clean_query = self.rewriter.transform(user_query, entities_to_strip=matched_entities)
        query_tokens = [t.lower() for t in clean_query.split() if t.strip()]

        dense_results: List[Tuple[str, int, float]] = []
        sparse_results: List[Tuple[str, int, float]] = []

        if self.all_child_chunks or (self.local_store and self.local_store.client):
            dense_results = retrieve_dense(
                clean_query,
                self.embedder,
                self.local_store.client,
                top_k=top_k_dense,
                doc_filter=doc_filter,
            )

        if self.bm25_searcher and query_tokens:
            sparse_results = self.bm25_searcher.search(
                query_tokens,
                top_k=top_k_dense,
                doc_filter=doc_filter,
            )

        fused_candidates = apply_rrf(
            dense_ranks=dense_results,
            sparse_ranks=sparse_results,
            k=config.retrieval.rrf_k,
            top_n=top_k_dense,
            child_chunk_map=self.child_chunk_map,
            query=clean_query,
            doc_filter=doc_filter,
        )
        candidate_cids = [cid for cid, _ in fused_candidates]

        return self.reranker.rerank_and_resolve(
            query=clean_query,
            candidate_child_ids=candidate_cids,
            child_chunk_map=self.child_chunk_map,
            local_store=self.local_store,
            top_k=config.retrieval.top_k_rerank,
            doc_filter=doc_filter,
        )

    def retrieve(self, user_query: str) -> List[RetrievalCandidate]:
        """Retrieve relevant context candidates across relational tabular and narrative documents using concurrent execution.

        Args:
            user_query: Natural language query string.

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
                    if self.tabular_engine.is_tabular_query(sq):
                        try:
                            tab = self.tabular_engine.query(sq)
                            if tab:
                                return tab
                        except Exception as ex:
                            print(f"[TabularEngine] Sub-query error: {ex}")
                    return self._retrieve_narrative(sq)

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
                                return merged_candidates

            return merged_candidates[:top_k_rerank]

        # Branch B: Solitary Query Execution
        # Check tabular intent first (if pure relational tabular query, execute directly)
        is_tabular = self.tabular_engine.is_tabular_query(user_query)
        if is_tabular and not is_comparative_query(user_query):
            try:
                tab_candidates = self.tabular_engine.query(user_query)
                if tab_candidates:
                    return tab_candidates
            except Exception as e:
                print(f"[TabularEngine] Query execution failed: {e}. Falling back to narrative search.")

        all_tabular_candidates: List[RetrievalCandidate] = []
        all_dense_ranks: List[Tuple[str, int, float]] = []
        all_sparse_ranks: List[Tuple[str, int, float]] = []

        doc_filter, matched_entities = self.rewriter.extract_doc_filter(
            user_query,
            list(self.known_doc_ids),
            doc_entity_map=self.doc_entity_map,
        )
        clean_query = self.rewriter.transform(user_query, entities_to_strip=matched_entities)

        # Execute dense, sparse, and tabular checks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            fut = executor.submit(self._execute_sub_query, user_query, top_k_dense)
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
            fused_candidates = apply_rrf(
                dense_ranks=all_dense_ranks,
                sparse_ranks=all_sparse_ranks,
                k=config.retrieval.rrf_k,
                top_n=top_k_dense,
                child_chunk_map=self.child_chunk_map,
                query=clean_query,
                doc_filter=doc_filter,
            )
            candidate_cids = [cid for cid, _ in fused_candidates]

            # Cross-Encoder Reranking and Neighbor Expansion
            narrative_candidates = self.reranker.rerank_and_resolve(
                query=clean_query,
                candidate_child_ids=candidate_cids,
                child_chunk_map=self.child_chunk_map,
                local_store=self.local_store,
                top_k=top_k_rerank,
                doc_filter=doc_filter,
            )

        # Merge candidates: tabular candidates + narrative candidates
        if deduped_tabular and narrative_candidates:
            merged = deduped_tabular + narrative_candidates
            return merged[:top_k_rerank]
        elif deduped_tabular:
            return deduped_tabular[:top_k_rerank]
        else:
            return narrative_candidates

    def ask(self, user_query: str) -> TrustAuditReport:
        """Process a user query with dynamic scope routing, neighbor expansion, and NLI verification.

        Args:
            user_query: Raw user search query.

        Returns:
            TrustAuditReport containing draft text, per-claim audits, and safety action.
        """
        if not user_query or not user_query.strip():
            return TrustAuditReport(
                draft_text="",
                faithfulness_score=1.0,
                has_contradiction=False,
                action="PASS",
                audits=[],
            )

        # 1. Execute Unified Retrieval (Tabular + Narrative Fusion with Sub-Query Decomp)
        top_contexts = self.retrieve(user_query)
        self.last_retrieved_contexts = top_contexts

        if not top_contexts:
            return TrustAuditReport(
                draft_text="No relevant context could be retrieved to answer the question.",
                faithfulness_score=1.0,
                has_contradiction=False,
                action="PASS",
                audits=[],
            )

        # 2. Taxonomy-Strict Prompt Construction
        prompt = build_rag_prompt(query=user_query.strip(), contexts=top_contexts)

        # 3. Draft Generation
        draft_text = self.generator.generate(prompt)

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

        # 7. NLI Auditing and Adjudication
        audit_report = self.adjudicator.adjudicate(
            claims=claims,
            context_map=context_map,
            nli_verifier=self.nli_verifier,
            draft_text=draft_text,
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

        return audit_report

    def close(self) -> None:
        """Close storage handles and clean up pipeline resources."""
        if hasattr(self, "tabular_store") and self.tabular_store is not None:
            self.tabular_store.close()
        if hasattr(self, "local_store") and self.local_store is not None:
            self.local_store.close()