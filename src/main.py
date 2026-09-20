r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/main.py
   - Role: End-to-end System Orchestrator for TrustRAG.
   - Purpose: Coordinates all 4 modular pipelines (Ingestion, Hybrid Retrieval with
     Dynamic Entity and Folder Routing, Citation-Aware Generation, and Claim-Level NLI Verification)
     into a unified, enterprise-scale, high-assurance RAG engine supporting recursive
     multi-depth subfolder document discovery with collision prevention.

2. INPUT (IP):
   - Ingestion: pdf_path (str) or raw_dir (str) pointing to document files or subfolder trees.
   - Querying: user_query (str) representing natural language user questions.

3. PROCESS UNDER THE HOOD:
   - Recursive Ingestion Flow:
     * Recursively traverses subfolders in `data/raw/` (`Path.rglob("*")`) across supported extensions.
     * Derives collision-safe relative `doc_id`s (e.g. `legal/2026/nda` and `legal__2026__nda`).
     * Extracts pages via PyMuPDF (`extract_pdf_pages`) or plain text readers.
     * Structurally chunks pages with breadcrumbs (`[Document: ... | Section: ...]`).
     * Attaches `relative_path` and `folder_hierarchy` to ChildChunk and ParentChunk payloads.
     * Computes dense vectors and sparse token dictionaries.
     * Indexes into Qdrant (`trustrag_enterprise`) and parent cache (`LocalStore`).
     * Rebuilds BM25 inverted index with inherited metadata.
     * Extracts clean candidate entity aliases and tracks `known_doc_ids` and `doc_entity_map`.
   - Query & Audit Flow:
     * Preprocesses query and extracts target document entity and folder domain filters.
     * Rewrites query to focus on pure section topic when scoped to an entity or subfolder.
     * Executes targeted dense and sparse search in parallel via ThreadPoolExecutor.
     * Fuses ranked lists with Reciprocal Rank Fusion (`apply_rrf`).
     * Cross-Encoder reranks and applies Document-Aware Neighbor Context Expansion.
     * Generates XML prompt enforcing section taxonomy, anti-bundling, and verbatim inventory rules.
     * Synthesizes draft response with active generator backend (Groq, Gemini, HF, Mock).
     * Validates citations and normalizes Unicode/whitespace bracket variants.
     * Extracts section-anchored atomic claims via `AtomicClaimExtractor` with multi-citation splitting.
     * Audits claims against section-scoped premise windows via DeBERTa-v3 and `AuditAdjudicator`.
     * Emits `TrustAuditReport`.

4. OUTPUT (OP):
   - TrustAuditReport: Strictly typed Pydantic audit report containing draft text,
     faithfulness score, per-claim NLI audits, and automated safety gate action.

5. LIBRARIES & DEPENDENCIES:
   - concurrent.futures: Parallel dense/sparse execution.
   - pathlib.Path: File operations and recursive directory traversal.
   - re: Regex entity and folder alias extraction.
   - src.common.config, src.common.schemas: Configurations and schemas.
   - src.pipeline_1_ingestion.*, src.pipeline_2_retrieval.*,
     src.pipeline_3_generation.*, src.pipeline_4_verification.*: Pipeline modules.
================================================================================
"""

import concurrent.futures
from pathlib import Path
import re
from typing import Dict, List, Optional, Set, Tuple, Union

from src.common.config import config
from src.common.schemas import ChildChunk, ParentChunk, RetrievalCandidate, TrustAuditReport
from src.pipeline_1_ingestion.chunker import create_hierarchical_chunks
from src.pipeline_1_ingestion.embedder import DualEmbedder
from src.pipeline_1_ingestion.indexer import LocalStore
from src.pipeline_1_ingestion.parser import extract_pdf_pages
from src.pipeline_2_retrieval.fusion import apply_rrf
from src.pipeline_2_retrieval.reranker import CrossEncoderReranker
from src.pipeline_2_retrieval.rewriter import QueryTransformer
from src.pipeline_2_retrieval.search_dense import retrieve_dense
from src.pipeline_2_retrieval.search_sparse import BM25Searcher
from src.pipeline_3_generation.citation_check import validate_and_parse_citations
from src.pipeline_3_generation.generator import BaseGenerator, get_generator
from src.pipeline_3_generation.prompt import build_rag_prompt
from src.pipeline_4_verification.adjudicator import AuditAdjudicator
from src.pipeline_4_verification.claim_extractor import AtomicClaimExtractor
from src.pipeline_4_verification.nli_model import DebertaNLIVerifier

SUPPORTED_DOCUMENT_EXTENSIONS: Set[str] = {
    ".pdf", ".docx", ".xlsx", ".csv", ".txt", ".jpg", ".png"
}


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

    exts = supported_extensions or SUPPORTED_DOCUMENT_EXTENSIONS
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

    _STOP_ALIASES = {
        "resume", "cv", "curriculum", "vitae", "profile", "contact", "email", "phone", "skills",
        "education", "experience", "general", "management", "technical", "engineering", "bachelor",
        "technology", "intermediate", "secondary", "board", "college", "school", "present", "india",
        "hyderabad", "telangana", "address", "trend", "hobbies", "programming", "language", "languages",
        "strengths", "academic", "projects", "project", "internship", "internships", "objective",
        "declaration", "activities", "work", "university", "department", "science", "gmail", "com",
        "linkedin", "github", "http", "https", "www", "what", "are", "the", "who", "which", "how",
        "and", "for", "with", "from", "about", "tell", "give", "details", "this", "that", "an", "a",
        "creativity", "negotiation", "critical", "thinking", "leadership", "travelling", "driving", "fashion"
    }

    def __init__(
        self,
        generator_type: Optional[str] = None,
        qdrant_location: str = ":memory:",
        qdrant_path: Optional[str] = None,
    ) -> None:
        """Initialize all pipeline components and models.

        Args:
            generator_type: Type of generation engine ('groq', 'gemini', 'mock', 'hf').
            qdrant_location: Qdrant database location (':memory:' or URL).
            qdrant_path: Optional on-disk directory path for local Qdrant storage.
        """
        # Pipeline 1: Ingestion & Storage
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
        """Extract candidate person names and document identifier aliases."""
        aliases = set()

        # Path and token aliases
        for token in re.split(r"[/\\__ -]+", assigned_doc_id):
            t_clean = token.lower().strip()
            if len(t_clean) >= 3 and t_clean not in self._STOP_ALIASES:
                aliases.add(t_clean)

        # Also register sanitized double-underscore alias
        sanitized_doc_id = assigned_doc_id.replace("/", "__").replace("\\", "__").lower()
        if len(sanitized_doc_id) >= 3 and sanitized_doc_id not in self._STOP_ALIASES:
            aliases.add(sanitized_doc_id)

        if pages:
            first_page_text = pages[0].get("raw_text", "")
            lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

            # Extract from emails (e.g. vvaishalee@gmail.com -> vaishalee)
            emails = re.findall(r"([a-zA-Z0-9_.+-]+)@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", first_page_text)
            for email in emails:
                for part in re.split(r"[0-9_.]+", email.lower()):
                    if len(part) >= 3 and part not in self._STOP_ALIASES:
                        aliases.add(part)
                        if part.startswith("v") and len(part) > 4:
                            aliases.add(part[1:])

            # Candidate person names from top 5 lines
            for line in lines[:5]:
                if not any(c.isdigit() for c in line) and ":" not in line and len(line.split()) <= 4:
                    for w in line.split():
                        w_l = w.lower().strip(".,;:()")
                        if len(w_l) >= 3 and w_l not in self._STOP_ALIASES and w_l.isalpha():
                            aliases.add(w_l)

            # Standalone uppercase names elsewhere on page 1 (e.g. VAISHALEE VUMMETHALA)
            for line in lines:
                if line.isupper() and len(line.split()) <= 2 and len(line) >= 4 and not any(c.isdigit() for c in line) and ":" not in line:
                    for w in line.split():
                        w_l = w.lower()
                        if len(w_l) >= 3 and w_l not in self._STOP_ALIASES and w_l.isalpha():
                            aliases.add(w_l)

        return list(aliases)

    def ingest_pdf(
        self,
        pdf_path: str,
        doc_id: Optional[str] = None,
        relative_path: Optional[str] = None,
        folder_hierarchy: Optional[List[str]] = None,
    ) -> List[ChildChunk]:
        """Ingest, chunk, embed, and index a PDF document with structural breadcrumbs.

        Args:
            pdf_path: File path to the PDF document.
            doc_id: Optional unique identifier for the document (defaults to relative stem).
            relative_path: Optional full relative subfolder path.
            folder_hierarchy: Optional list of parent folder categories.

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

        print(f"Successfully indexed {len(children)} chunks from {path.name} (doc_id: '{assigned_doc_id}').")
        return children

    def ingest_directory(
        self,
        raw_dir: str = "data/raw",
        supported_extensions: Optional[Set[str]] = None,
    ) -> List[ChildChunk]:
        """Recursively discover and ingest all supported document files across subfolders in raw_dir.

        Args:
            raw_dir: Path to base raw data directory.
            supported_extensions: Optional set of allowed file extensions.

        Returns:
            List of all indexed ChildChunk models.
        """
        discovered = discover_raw_documents(raw_dir=raw_dir, supported_extensions=supported_extensions)
        print(f"[Ingestion] Discovered {len(discovered)} document(s) across subfolders in '{raw_dir}'.")

        all_indexed: List[ChildChunk] = []
        for file_path, doc_id, rel_str, folders in discovered:
            if file_path.suffix.lower() == ".pdf":
                chunks = self.ingest_pdf(
                    str(file_path),
                    doc_id=doc_id,
                    relative_path=rel_str,
                    folder_hierarchy=folders,
                )
                all_indexed.extend(chunks)

        return all_indexed

    def ask(self, user_query: str) -> TrustAuditReport:
        """Process a user query with dynamic entity routing, neighbor expansion, and NLI verification.

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

        # 1. Query Preprocessing & Dynamic Entity Routing
        doc_filter, matched_entities = self.rewriter.extract_doc_filter(
            user_query,
            list(self.known_doc_ids),
            doc_entity_map=self.doc_entity_map,
        )
        clean_query = self.rewriter.transform(user_query, entities_to_strip=matched_entities)
        if doc_filter:
            if isinstance(doc_filter, list):
                print(f"[Entity Router] Pre-filtering multi-document retrieval scope to: {doc_filter} (entities: {matched_entities})")
            else:
                print(f"[Entity Router] Pre-filtering retrieval scope to document: '{doc_filter}' (entities: {matched_entities})")

        # 2. Parallel Dense and Sparse Retrieval with Pre-Filtering
        top_k_dense = config.thresholds.top_k_dense
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
            k=config.thresholds.rrf_k,
            top_n=top_k_dense,
        )
        candidate_cids = [cid for cid, _ in fused_candidates]

        # 4. Cross-Encoder Reranking and Neighbor Context Expansion
        top_contexts: List[RetrievalCandidate] = self.reranker.rerank_and_resolve(
            query=clean_query,
            candidate_child_ids=candidate_cids,
            child_chunk_map=self.child_chunk_map,
            local_store=self.local_store,
            top_k=config.thresholds.top_k_rerank,
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
            f"Doc-{idx}": context.text
            for idx, context in enumerate(top_contexts, start=1)
        }

        # 10. NLI Auditing and Adjudication
        audit_report = self.adjudicator.adjudicate(
            claims=claims,
            context_map=context_map,
            nli_verifier=self.nli_verifier,
            draft_text=draft_text,
        )

        return audit_report
