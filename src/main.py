r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/main.py
   - Role: End-to-end System Orchestrator and Incremental Ingestion Controller for TrustRAG.
   - Purpose: Coordinates all 4 modular pipelines (Ingestion, Hybrid Retrieval with
     Dynamic Scope Routing, Closed-World Generation, Claim-Level NLI Verification,
     and Automated 1-Pass Corrective Rewriting) into a unified, enterprise-scale,
     high-assurance RAG engine supporting incremental SHA-256 manifest caching and
     recursive multi-depth subfolder document discovery.

2. INPUT (IP):
   - Ingestion: pdf_path (str) or raw_dir (str) pointing to document files or subfolder trees.
   - Querying: user_query (str) representing natural language user questions.

3. PROCESS UNDER THE HOOD:
   - Incremental & Recursive Ingestion Flow:
     * Recursively traverses subfolders in `data/raw/` across configured supported extensions.
     * Derives collision-safe relative `doc_id`s (e.g. `legal/2026/nda`).
     * Inspects `IngestionManifest`: skips unchanged files based on SHA-256 fingerprinting.
     * On file modifications: deletes stale Qdrant points via `vector_store.delete_document()`.
     * Extracts pages, applies domain-agnostic structural section chunking with breadcrumbs.
     * Computes dense embeddings and sparse token dictionaries.
     * Indexes into Qdrant (`trustrag_enterprise`) and updates `IngestionManifest`.
     * Maintains BM25 search index and dynamic entity aliases.
   - Query, Audit & Self-Correction Flow:
     * Preprocesses query and extracts target folder domain filters if explicitly specified.
     * Executes parallel dense and sparse retrieval with reciprocal rank fusion (`apply_rrf`).
     * Cross-Encoder reranks and applies Document-Aware Neighbor Context Expansion.
     * Generates XML prompt enforcing section taxonomy, closed-world assumption, and verbatim inventory rules.
     * Synthesizes draft response with active generator backend (Groq, Gemini, Ollama, Mock).
     * Validates citations and extracts section-anchored atomic claims with negative meta-claim filtering.
     * Audits claims against unified section-scoped premise windows via DeBERTa-v3 and `AuditAdjudicator`.
     * If ungrounded or contradictory claims exist (WARN / TRIGGER_REWRITE): executes a 1-pass automated
       corrective rewrite loop via `build_correction_prompt` to produce a 100% faithful final report.
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
     src.pipeline_3_generation.*, src.pipeline_4_verification.*: Pipeline modules.
================================================================================
"""

import concurrent.futures
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

from src.common.config import config
from src.common.schemas import ChildChunk, ParentChunk, RetrievalCandidate, TrustAuditReport
from src.pipeline_1_ingestion.chunker import create_hierarchical_chunks
from src.pipeline_1_ingestion.embedder import DualEmbedder
from src.pipeline_1_ingestion.indexer import LocalStore
from src.pipeline_1_ingestion.manifest import IngestionManifest
from src.pipeline_1_ingestion.parser import extract_pdf_pages
from src.pipeline_2_retrieval.fusion import apply_rrf
from src.pipeline_2_retrieval.reranker import CrossEncoderReranker
from src.pipeline_2_retrieval.rewriter import QueryTransformer
from src.pipeline_2_retrieval.search_dense import retrieve_dense
from src.pipeline_2_retrieval.search_sparse import BM25Searcher
from src.pipeline_3_generation.citation_check import validate_and_parse_citations
from src.pipeline_3_generation.generator import BaseGenerator, get_generator
from src.pipeline_3_generation.prompt import build_correction_prompt, build_rag_prompt
from src.pipeline_4_verification.adjudicator import AuditAdjudicator
from src.pipeline_4_verification.claim_extractor import AtomicClaimExtractor
from src.pipeline_4_verification.nli_model import DebertaNLIVerifier


SUPPORTED_DOCUMENT_EXTENSIONS: Set[str] = set(config.ingestion.supported_extensions)


def discover_raw_documents(
    raw_dir: Union[str, Path] = "data/raw",
    supported_extensions: Optional[Set[str]] = None,
) -> List[Tuple[Path, str, str, List[str]]]:
    """Recursively discover document files in raw_dir across all subfolder depths.

    Args:
        raw_dir: Base directory path to scan.
        supported_extensions: Optional set of allowed file extensions.

    Returns:
        List of tuples: `(file_path, doc_id, relative_path_str, folder_hierarchy)`.
    """
    base_dir = Path(raw_dir)
    if not base_dir.exists():
        return []

    exts = supported_extensions or set(config.ingestion.supported_extensions)
    discovered: List[Tuple[Path, str, str, List[str]]] = []

    for file_path in sorted(base_dir.rglob("*")):
        if not file_path.is_file():
            continue

        try:
            rel_path = file_path.relative_to(base_dir)
        except ValueError:
            rel_path = Path(file_path.name)

        # Ignore hidden system files and hidden directory segments
        if any(part.startswith(".") for part in rel_path.parts):
            continue

        if file_path.suffix.lower() not in exts:
            continue

        # Derive clean, collision-safe doc_id (e.g., "legal/2026/nda")
        rel_str = str(rel_path).replace("\\", "/")
        rel_stem_str = str(rel_path.with_suffix("")).replace("\\", "/")
        doc_id = rel_stem_str
        folder_hierarchy = [p for p in rel_path.parent.parts if p and p != "."]

        discovered.append((file_path, doc_id, rel_str, folder_hierarchy))

    return discovered


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

        # Pipeline 2: Retrieval & Reranking
        self.rewriter = QueryTransformer()
        self.bm25_searcher: Optional[BM25Searcher] = None
        self.reranker = CrossEncoderReranker()

        # Pipeline 3: Generation & Citation
        self.generator_type = generator_type or config.GENERATOR_PROVIDER
        self.generator: BaseGenerator = get_generator(generator_type=self.generator_type)

        # Pipeline 4: Verification & Adjudication
        self.claim_extractor = AtomicClaimExtractor()
        self.nli_verifier = DebertaNLIVerifier()
        self.adjudicator = AuditAdjudicator()

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

    def ingest_pdf(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        relative_path: Optional[str] = None,
        folder_hierarchy: Optional[List[str]] = None,
        force_reindex: bool = False,
    ) -> List[ChildChunk]:
        """Ingest, chunk, embed, and index a PDF document with incremental manifest caching.

        Args:
            pdf_path: File path to the PDF document.
            doc_id: Optional unique identifier for the document (defaults to relative stem).
            relative_path: Optional full relative subfolder path.
            folder_hierarchy: Optional list of parent folder categories.
            force_reindex: If True, bypass manifest check and re-index.

        Returns:
            List of indexed ChildChunk models.
        """
        path = Path(pdf_path)
        if not path.is_file():
            raise FileNotFoundError(f"PDF document not found at: {pdf_path}")

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

        # Step 0: Check Incremental Manifest Fingerprint
        if not force_reindex and self.manifest.is_indexed_and_current(path, assigned_doc_id):
            print(f"[Ingestion] '{assigned_doc_id}' unchanged (already indexed) -> Skipping.")
            self.known_doc_ids.add(assigned_doc_id)
            return []

        # If document was previously indexed with different content, clear stale points
        self.local_store.delete_document(assigned_doc_id)
        # Clear local child chunks for this doc_id
        self.all_child_chunks = [c for c in self.all_child_chunks if c.doc_id != assigned_doc_id]
        self.child_chunk_map = {cid: c for cid, c in self.child_chunk_map.items() if c.doc_id != assigned_doc_id}

        # Step 1: Extract pages
        pages = extract_pdf_pages(str(path))
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
            if file_path.suffix.lower() == ".pdf":
                if not force_reindex and self.manifest.is_indexed_and_current(file_path, doc_id):
                    print(f"[Ingestion] '{doc_id}' unchanged (already indexed) -> Skipping.")
                    self.known_doc_ids.add(doc_id)
                    skipped_count += 1
                else:
                    chunks = self.ingest_pdf(
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

        # 1. Query Preprocessing & Dynamic Folder/Scope Routing
        doc_filter, matched_entities = self.rewriter.extract_doc_filter(
            user_query,
            list(self.known_doc_ids),
            doc_entity_map=self.doc_entity_map,
        )
        clean_query = self.rewriter.transform(user_query, entities_to_strip=matched_entities)
        if doc_filter:
            if isinstance(doc_filter, list):
                print(f"[Scope Router] Scoping retrieval to documents: {doc_filter} (directives: {matched_entities})")
            else:
                print(f"[Scope Router] Scoping retrieval to document: '{doc_filter}' (directives: {matched_entities})")

        # 2. Parallel Dense and Sparse Retrieval
        top_k_dense = config.retrieval.top_k_dense
        query_tokens = [t.lower() for t in clean_query.split() if t.strip()]

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            future_dense = executor.submit(
                retrieve_dense,
                clean_query,
                self.embedder,
                self.local_store.client,
                top_k=top_k_dense,
                doc_filter=doc_filter,
            )
            future_sparse = executor.submit(
                self.bm25_searcher.search if self.bm25_searcher else lambda *args, **kwargs: [],
                query_tokens,
                top_k=top_k_dense,
                doc_filter=doc_filter,
            )

            dense_results = future_dense.result()
            sparse_results = future_sparse.result()

        # 3. Reciprocal Rank Fusion
        fused_candidates = apply_rrf(
            dense_ranks=dense_results,
            sparse_ranks=sparse_results,
            k=config.retrieval.rrf_k,
            top_n=top_k_dense,
        )
        candidate_cids = [cid for cid, _ in fused_candidates]

        # 4. Cross-Encoder Reranking and Neighbor Context Expansion
        top_contexts: List[RetrievalCandidate] = self.reranker.rerank_and_resolve(
            query=clean_query,
            candidate_child_ids=candidate_cids,
            child_chunk_map=self.child_chunk_map,
            local_store=self.local_store,
            top_k=config.retrieval.top_k_rerank,
        )
        self.last_retrieved_contexts = top_contexts

        # 5. Taxonomy-Strict Prompt Construction
        prompt = build_rag_prompt(query=user_query.strip(), contexts=top_contexts)

        # 6. Draft Generation
        draft_text = self.generator.generate(prompt)

        # 7. Citation Parsing & Boundary Validation (with Unicode normalization)
        generated_draft = validate_and_parse_citations(
            draft_text=draft_text,
            max_valid_doc_id=len(top_contexts),
        )

        # 8. Section-Anchored Atomic Claim Decomposition
        default_section = top_contexts[0].section_name if top_contexts else None
        claims = self.claim_extractor.extract_claims(generated_draft, default_section=default_section)

        # 9. Premise Mapping
        context_map: Dict[str, str] = {
            f"Doc-{idx}": getattr(context, "parent_text", None) or context.text
            for idx, context in enumerate(top_contexts, start=1)
        }

        # 10. NLI Auditing and Adjudication
        audit_report = self.adjudicator.adjudicate(
            claims=claims,
            context_map=context_map,
            nli_verifier=self.nli_verifier,
            draft_text=draft_text,
        )

        # 11. Automated Self-Correction Rewrite Loop (1-pass)
        if audit_report.action in ("WARN", "TRIGGER_REWRITE") or audit_report.has_contradiction or audit_report.faithfulness_score < config.verification.tau_entailment:
            failed_claims = [
                a.claim_text for a in audit_report.audits
                if a.verdict != "ENTAILED"
            ]
            if failed_claims:
                print(f"[Self-Correction] Triggered 1-pass corrective rewrite for {len(failed_claims)} unverified claims.")
                correction_prompt = build_correction_prompt(
                    query=user_query.strip(),
                    context=top_contexts,
                    draft=draft_text,
                    failed_claims=failed_claims,
                )
                corrected_draft_text = self.generator.generate(correction_prompt)
                corrected_draft = validate_and_parse_citations(
                    draft_text=corrected_draft_text,
                    max_valid_doc_id=len(top_contexts),
                )
                corrected_claims = self.claim_extractor.extract_claims(
                    corrected_draft,
                    default_section=default_section,
                )
                corrected_report = self.adjudicator.adjudicate(
                    claims=corrected_claims,
                    context_map=context_map,
                    nli_verifier=self.nli_verifier,
                    draft_text=corrected_draft_text,
                )
                if corrected_report.faithfulness_score > audit_report.faithfulness_score:
                    audit_report = corrected_report
                elif corrected_report.faithfulness_score == audit_report.faithfulness_score and not corrected_report.has_contradiction and audit_report.has_contradiction:
                    audit_report = corrected_report

        return audit_report


    def close(self) -> None:
        """Close storage handles and clean up pipeline resources."""
        if hasattr(self, "local_store") and self.local_store is not None:
            self.local_store.close()

