r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/api.py
   - Role: FastAPI Backend REST API Gateway for TrustRAG.
   - Purpose: Exposes high-assurance verification, conversational chat with coreference
     reformulation, multi-turn ledger history, thread session management (create, list,
     delete, load messages), document auto-indexing, and DuckDB tabular catalog endpoints
     with standardized "Trust Score" metrics and CORS support.

2. INPUT (IP):
   - HTTP Requests:
     * POST /api/threads: Initialize new chat thread.
     * GET /api/threads: List user's conversation threads.
     * DELETE /api/threads/{thread_id}: Delete a thread and its history.
     * GET /api/threads/{thread_id}/messages: Retrieve full message history with audits.
     * POST /api/chat: JSON payload {"query": str, "thread_id": Optional[str]}.
     * GET /api/history: Query parameter thread_id (defaults to "default_thread").
     * GET /api/documents: Catalog reflection request.
     * POST /api/documents/refresh: Manual re-sync of data directory.
     * GET /api/health: Health check probe.

3. PROCESS UNDER THE HOOD:
   - Configures FastAPI with CORS middleware permitting frontend client origins.
   - Lifespan context manager auto-indexes all documents in `data/raw/` on server startup
     under user_id="default_user" and thread_id=None.
   - Coordinates multi-tenant thread persistence with `DatabaseRepository`.
   - Ingests files into `data/raw/` and registers vectors and DuckDB tables.
   - Maps underlying faithfulness metrics to user-facing `trust_score` and `trust_verdict`.
   - Exposes atomic claim verification audits and physical provenance coordinates.

4. OUTPUT (OP):
   - JSON responses adhering to strict Pydantic API response models.

5. LIBRARIES & DEPENDENCIES:
   - fastapi, fastapi.middleware.cors, uvicorn, pydantic, uuid.
   - pathlib.Path, shutil, typing.
   - src.main (TrustRAGPipeline).
   - src.common.schemas (ProvenanceCoordinate, TrustAuditReport).
================================================================================
"""

from contextlib import asynccontextmanager
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.main import TrustRAGPipeline

# Pipeline Singleton Manager
_pipeline_instance: Optional[TrustRAGPipeline] = None


def get_pipeline() -> TrustRAGPipeline:
    """Retrieve or initialize the active TrustRAGPipeline instance."""
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = TrustRAGPipeline()
    return _pipeline_instance


def set_pipeline(pipeline: TrustRAGPipeline) -> None:
    """Explicitly assign a pipeline instance (useful for testing and dependency injection)."""
    global _pipeline_instance
    _pipeline_instance = pipeline


def ensure_default_user(repo) -> None:
    """Ensure standard default_user exists in database ledger."""
    user = repo.get_user_by_id("default_user")
    if not user:
        try:
            repo.create_user(
                email="default_user@tenant.trustrag",
                password="tenant_default_password",
                full_name="Tenant default_user",
                user_id="default_user",
            )
        except Exception:
            pass


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager: Auto-indexes all documents in data/raw/ on startup."""
    pipeline = get_pipeline()
    if hasattr(pipeline, "db_repo") and pipeline.db_repo is not None:
        ensure_default_user(pipeline.db_repo)

    raw_dir = Path("data") if Path("data").exists() else Path("data/raw")

    print(f"[API Startup] Auto-indexing documents recursively from '{raw_dir}'...")
    try:
        pipeline.ingest_directory(
            raw_dir=str(raw_dir),
            force_reindex=False,
            user_id="default_user",
            thread_id=None,
        )
        print(f"[API Startup] Auto-indexing completed. Known documents: {len(pipeline.known_doc_ids)}")
    except Exception as e:
        print(f"[API Startup] Warning during auto-ingest: {e}")
    yield


app = FastAPI(
    title="TrustRAG API",
    description="High-Assurance Retrieval-Augmented Generation with Trust Verification & Provenance Lineage",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware Configuration
ALLOWED_ORIGINS = [
    "http://localhost:5173",
    "http://localhost:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:3000",
    "*",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------------------
# Request / Response Schemas
# ------------------------------------------------------------------------------

class ThreadCreateResponse(BaseModel):
    thread_id: str
    title: str


class ThreadItem(BaseModel):
    thread_id: str
    title: str
    created_at: str


class ThreadDeleteResponse(BaseModel):
    status: str
    thread_id: str


class ChatRequest(BaseModel):
    query: str = Field(..., description="Natural language search query or follow-up question")
    thread_id: Optional[str] = Field("default_thread", description="Active thread session identifier")


class ClaimItem(BaseModel):
    claim_id: str
    claim_text: str
    verdict: str
    confidence: float
    cited_premise: Optional[str] = None


class ChatResponse(BaseModel):
    answer: str
    trust_score: float
    trust_verdict: str
    claims: List[ClaimItem] = Field(default_factory=list)
    provenance_map: Dict[str, Any] = Field(default_factory=dict)
    thread_id: Optional[str] = None


class UploadResponse(BaseModel):
    filename: str
    chunks_indexed: int
    doc_id: str


class DocumentListResponse(BaseModel):
    documents: List[str]
    tables: List[str]


class HealthResponse(BaseModel):
    status: str
    known_documents: List[str]


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get("/api/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint returning system status and known document list."""
    pipeline = get_pipeline()
    return HealthResponse(
        status="ok",
        known_documents=sorted(list(pipeline.known_doc_ids)),
    )


@app.get("/api/documents", response_model=DocumentListResponse)
def list_documents() -> DocumentListResponse:
    """Return catalog of currently indexed documents and registered DuckDB tables."""
    pipeline = get_pipeline()
    tables = []
    if hasattr(pipeline, "tabular_store") and pipeline.tabular_store is not None:
        tables = pipeline.tabular_store.get_table_names(user_id="default_user", thread_id=None)

    return DocumentListResponse(
        documents=sorted(list(pipeline.known_doc_ids)),
        tables=sorted(tables),
    )


@app.post("/api/documents/refresh", response_model=DocumentListResponse)
def refresh_documents() -> DocumentListResponse:
    """Trigger manual re-sync and auto-indexing of data/raw directory, returning updated document and table catalogs."""
    pipeline = get_pipeline()
    raw_dir = Path("data") if Path("data").exists() else Path("data/raw")

    try:
        pipeline.ingest_directory(
            raw_dir=str(raw_dir),
            force_reindex=False,
            user_id="default_user",
            thread_id=None,
        )
    except Exception as e:
        print(f"[API Refresh] Warning during manual document refresh: {e}")

    tables = []
    if hasattr(pipeline, "tabular_store") and pipeline.tabular_store is not None:
        tables = pipeline.tabular_store.get_table_names(user_id="default_user", thread_id=None)

    return DocumentListResponse(
        documents=sorted(list(pipeline.known_doc_ids)),
        tables=sorted(tables),
    )


# ------------------------------------------------------------------------------
# Chat Thread Session Management Endpoints
# ------------------------------------------------------------------------------

@app.post("/api/threads", response_model=ThreadCreateResponse)
def create_thread() -> ThreadCreateResponse:
    """Generate a new thread ID and persist a fresh conversation thread in the database ledger."""
    pipeline = get_pipeline()
    ensure_default_user(pipeline.db_repo)

    new_thread_id = f"thread_{uuid.uuid4().hex[:12]}"
    thread = pipeline.db_repo.create_thread(
        user_id="default_user",
        title="New Conversation",
        thread_id=new_thread_id,
    )
    return ThreadCreateResponse(
        thread_id=thread.id,
        title=thread.title,
    )


@app.get("/api/threads", response_model=List[ThreadItem])
def list_threads() -> List[ThreadItem]:
    """Retrieve all chat threads for default_user, sorted latest first."""
    pipeline = get_pipeline()
    ensure_default_user(pipeline.db_repo)

    threads = pipeline.db_repo.get_user_threads("default_user", limit=100)
    if not threads:
        # Guarantee at least default_thread exists
        th = pipeline.db_repo.get_thread("default_thread", "default_user")
        if not th:
            th = pipeline.db_repo.create_thread(
                user_id="default_user",
                title="Default Session",
                thread_id="default_thread",
            )
        threads = [th]

    return [
        ThreadItem(
            thread_id=t.id,
            title=t.title,
            created_at=t.created_at.isoformat() if hasattr(t.created_at, "isoformat") else str(t.created_at),
        )
        for t in threads
    ]


@app.delete("/api/threads/{thread_id}", response_model=ThreadDeleteResponse)
def delete_thread(thread_id: str) -> ThreadDeleteResponse:
    """Delete a conversation thread and its associated message history from the database."""
    pipeline = get_pipeline()
    deleted = pipeline.db_repo.delete_thread(thread_id, user_id="default_user")
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Thread '{thread_id}' not found.")
    return ThreadDeleteResponse(
        status="deleted",
        thread_id=thread_id,
    )


@app.get("/api/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str) -> List[Dict[str, Any]]:
    """Return all messages for the specified thread_id with parsed audit reports and trust scores."""
    pipeline = get_pipeline()
    raw_messages = pipeline.db_repo.get_thread_messages(thread_id, "default_user", limit=100)

    history: List[Dict[str, Any]] = []
    for msg in raw_messages:
        audit = msg.audit_report or {}
        faithfulness = audit.get("faithfulness_score")
        history.append(
            {
                "id": msg.id,
                "role": msg.role,
                "content": msg.content,
                "created_at": msg.created_at.isoformat() if hasattr(msg.created_at, "isoformat") else str(msg.created_at),
                "trust_score": round(faithfulness, 4) if faithfulness is not None else None,
                "trust_verdict": audit.get("action"),
                "citations": msg.citations or [],
                "claims": audit.get("audits", []),
                "provenance_map": audit.get("provenance_map", {}),
            }
        )
    return history


@app.get("/api/history")
def get_chat_history(thread_id: Optional[str] = "default_thread") -> List[Dict[str, Any]]:
    """Backwards-compatible endpoint fetching thread messages (defaulting to default_thread)."""
    target = thread_id.strip() if (thread_id and thread_id.strip()) else "default_thread"
    return get_thread_messages(target)


@app.post("/api/upload", response_model=UploadResponse, deprecated=True)
async def upload_document(file: UploadFile = File(...)) -> UploadResponse:
    """Accept and ingest a file into TrustRAG vector and tabular storage (Deprecated)."""
    pipeline = get_pipeline()

    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file must have a valid filename.")

    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    target_path = raw_dir / file.filename

    # Save uploaded file safely
    try:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {e}")
    finally:
        await file.close()

    # Determine doc_id from stem
    clean_stem = Path(file.filename).stem.strip()
    doc_id = clean_stem or "doc_uploaded"

    try:
        chunks = pipeline.ingest_document(
            file_path=str(target_path),
            doc_id=doc_id,
            user_id="default_user",
            thread_id=None,
            force_reindex=True,
        )
        return UploadResponse(
            filename=file.filename,
            chunks_indexed=len(chunks),
            doc_id=doc_id,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to ingest document: {e}")


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Execute conversational query with coreference resolution, NLI auditing, and Trust Score calculation."""
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    target_thread_id = request.thread_id.strip() if (request.thread_id and request.thread_id.strip()) else "default_thread"

    pipeline = get_pipeline()
    if hasattr(pipeline, "db_repo") and pipeline.db_repo is not None:
        ensure_default_user(pipeline.db_repo)

        # If thread title is "New Conversation" or "New Chat", auto-update to prompt snippet
        th = pipeline.db_repo.get_thread(target_thread_id, "default_user")
        if th and th.title in ("New Conversation", "New Chat"):
            title_snippet = query[:30].strip()
            if len(query) > 30:
                title_snippet += "..."
            pipeline.db_repo.update_thread_title(target_thread_id, "default_user", title_snippet)

    report = pipeline.chat(
        user_query=query,
        user_id="default_user",
        thread_id=target_thread_id,
    )

    claims_list: List[ClaimItem] = []
    for audit in report.audits:
        claims_list.append(
            ClaimItem(
                claim_id=audit.claim_id,
                claim_text=audit.claim_text,
                verdict=audit.verdict,
                confidence=audit.confidence,
                cited_premise=audit.cited_premise,
            )
        )

    # Serialize provenance map coordinates
    serialized_prov: Dict[str, Any] = {}
    for doc_k, coord in report.provenance_map.items():
        if hasattr(coord, "model_dump"):
            serialized_prov[doc_k] = coord.model_dump()
        elif isinstance(coord, dict):
            serialized_prov[doc_k] = coord
        else:
            serialized_prov[doc_k] = str(coord)

    return ChatResponse(
        answer=report.draft_text,
        trust_score=round(report.faithfulness_score, 4),
        trust_verdict=report.action,
        claims=claims_list,
        provenance_map=serialized_prov,
        thread_id=target_thread_id,
    )
