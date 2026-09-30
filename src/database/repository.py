r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/database/repository.py
   - Role: Data Access Object (DAO) and Multi-Tenant Repository Layer for TrustRAG.
   - Purpose: Implements thread-safe CRUD operations for user accounts, credential
     authentication, chat threads, and structured message history with full TrustAuditReport
     verification artifact persistence. Enforces strict tenant ownership guarantees across
     all operations.

2. INPUT (IP):
   - User account data (email, plain password, full name).
   - Chat thread parameters (user_id, thread_id, title).
   - Conversational messages (role, content, citations, TrustAuditReport instances/dicts).
   - Pagination parameters (limit, offset).

3. PROCESS UNDER THE HOOD:
   - User Management:
     * Validates and normalizes email addresses (lowercasing, whitespace trimming).
     * Enforces unique email constraints, rejecting duplicate registrations.
     * Hashes passwords via bcrypt before persisting to storage.
     * Authenticates user credentials via constant-time password hash verification.
   - Strict Multi-Tenant Thread Isolation:
     * Thread retrieval, mutation, and deletion require matching both `thread_id` AND `user_id`.
     * Message logging and retrieval strictly verify thread ownership before execution.
   - Verification Artifact Serialization & Deserialization:
     * Converts Pydantic v2 `TrustAuditReport` instances to pure JSON dicts via `.model_dump()`.
     * Serializes citation lists and candidate models into structured JSON columns.
     * Provides helper deserializer to reconstruct typed `TrustAuditReport` models on demand.
   - Transactional Integrity:
     * Supports both standalone operation (via `get_db_session`) and external transaction
       injection (via caller-provided `Session`).

4. OUTPUT (OP):
   - Domain entity models (User, ChatThread, Message), typed TrustAuditReport instances,
     boolean success flags, and paginated record lists.
   - Consumed by: API endpoints, CLI session managers, and `src/main.py`.

5. LIBRARIES & DEPENDENCIES:
   - datetime: Timezone-aware UTC timestamp generation.
   - typing: Type annotations (Any, Dict, List, Optional, Union).
   - sqlalchemy: select, func, and Session operations.
   - src.common.schemas: TrustAuditReport schema for verification artifact roundtripping.
   - src.database.auth: Password hashing and verification.
   - src.database.connection: SessionLocal and get_db_session context manager.
   - src.database.models: User, ChatThread, and Message ORM models.
================================================================================
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker, Session

from src.common.schemas import TrustAuditReport
from src.database.auth import hash_password, verify_password
from src.database.connection import get_db_session, SessionLocal
from src.database.models import ChatThread, Message, User


class DatabaseRepository:
    """Thread-safe Data Access Object for TrustRAG multi-tenant persistence."""

    def __init__(self, session_factory: Optional[sessionmaker[Session]] = None) -> None:
        """Initialize repository with optional custom session factory.

        Args:
            session_factory: Custom sessionmaker instance, defaulting to SessionLocal.
        """
        self.session_factory = session_factory or SessionLocal

    def _execute(self, session: Optional[Session], callback):
        """Execute a database operation within an existing or newly provisioned session."""
        if session is not None:
            return callback(session)
        with get_db_session(self.session_factory) as s:
            return callback(s)

    # ==============================================================================
    # 1. USER ACCOUNT & AUTHENTICATION OPERATIONS
    # ==============================================================================

    def create_user(
        self,
        email: str,
        password: str,
        full_name: Optional[str] = None,
        is_active: bool = True,
        session: Optional[Session] = None,
    ) -> User:
        """Create and persist a new user account with hashed credentials.

        Args:
            email: User email address (unique identifier).
            password: Plaintext password string.
            full_name: Optional full name or display name.
            is_active: Account status flag.
            session: Optional active database session.

        Returns:
            Newly created User entity.

        Raises:
            ValueError: If email/password are empty or email already exists.
        """
        clean_email = email.strip().lower() if email else ""
        if not clean_email:
            raise ValueError("Email address cannot be empty.")
        if not password:
            raise ValueError("Password cannot be empty.")

        def _op(s: Session) -> User:
            stmt = select(User).where(func.lower(User.email) == clean_email)
            existing = s.execute(stmt).scalar_one_or_none()
            if existing:
                raise ValueError(f"User with email '{clean_email}' already exists.")

            hashed_pw = hash_password(password)
            user = User(
                email=clean_email,
                hashed_password=hashed_pw,
                full_name=full_name.strip() if full_name else None,
                is_active=is_active,
            )
            s.add(user)
            s.flush()
            s.refresh(user)
            return user

        return self._execute(session, _op)

    def authenticate_user(
        self,
        email: str,
        password: str,
        session: Optional[Session] = None,
    ) -> Optional[User]:
        """Authenticate user credentials against stored salted bcrypt hashes.

        Args:
            email: User email address.
            password: Plaintext password to verify.
            session: Optional active database session.

        Returns:
            User entity if credentials match and account is active, None otherwise.
        """
        clean_email = email.strip().lower() if email else ""
        if not clean_email or not password:
            return None

        def _op(s: Session) -> Optional[User]:
            stmt = select(User).where(func.lower(User.email) == clean_email)
            user = s.execute(stmt).scalar_one_or_none()
            if not user or not user.is_active:
                return None
            if verify_password(password, user.hashed_password):
                return user
            return None

        return self._execute(session, _op)

    def get_user_by_id(
        self,
        user_id: str,
        session: Optional[Session] = None,
    ) -> Optional[User]:
        """Fetch user by unique user identifier.

        Args:
            user_id: UUID string of the target user.
            session: Optional active database session.

        Returns:
            User entity if found, None otherwise.
        """
        def _op(s: Session) -> Optional[User]:
            stmt = select(User).where(User.id == str(user_id))
            return s.execute(stmt).scalar_one_or_none()

        return self._execute(session, _op)

    def get_user_by_email(
        self,
        email: str,
        session: Optional[Session] = None,
    ) -> Optional[User]:
        """Fetch user by email address.

        Args:
            email: Email address string.
            session: Optional active database session.

        Returns:
            User entity if found, None otherwise.
        """
        clean_email = email.strip().lower() if email else ""
        if not clean_email:
            return None

        def _op(s: Session) -> Optional[User]:
            stmt = select(User).where(func.lower(User.email) == clean_email)
            return s.execute(stmt).scalar_one_or_none()

        return self._execute(session, _op)

    def delete_user(
        self,
        user_id: str,
        session: Optional[Session] = None,
    ) -> bool:
        """Delete user account and cascade delete all associated threads and messages.

        Args:
            user_id: UUID of user to delete.
            session: Optional active database session.

        Returns:
            True if user was deleted, False if user did not exist.
        """
        def _op(s: Session) -> bool:
            stmt = select(User).where(User.id == str(user_id))
            user = s.execute(stmt).scalar_one_or_none()
            if not user:
                return False
            s.delete(user)
            s.flush()
            return True

        return self._execute(session, _op)

    # ==============================================================================
    # 2. CHAT THREAD OPERATIONS (STRICT TENANT ISOLATION)
    # ==============================================================================

    def create_thread(
        self,
        user_id: str,
        title: str = "New Chat",
        session: Optional[Session] = None,
    ) -> ChatThread:
        """Create a new conversational chat thread for an authenticated user.

        Args:
            user_id: UUID of the owning user.
            title: Display title for the thread.
            session: Optional active database session.

        Returns:
            Newly created ChatThread entity.

        Raises:
            ValueError: If user does not exist.
        """
        def _op(s: Session) -> ChatThread:
            stmt = select(User.id).where(User.id == str(user_id))
            if not s.execute(stmt).scalar_one_or_none():
                raise ValueError(f"User with id '{user_id}' does not exist.")

            thread = ChatThread(
                user_id=str(user_id),
                title=title.strip() if title else "New Chat",
            )
            s.add(thread)
            s.flush()
            s.refresh(thread)
            return thread

        return self._execute(session, _op)

    def get_thread(
        self,
        thread_id: str,
        user_id: str,
        session: Optional[Session] = None,
    ) -> Optional[ChatThread]:
        """Retrieve chat thread verifying strict tenant ownership.

        Args:
            thread_id: UUID of the target chat thread.
            user_id: UUID of the requesting user.
            session: Optional active database session.

        Returns:
            ChatThread if found and owned by user_id, None otherwise.
        """
        def _op(s: Session) -> Optional[ChatThread]:
            stmt = select(ChatThread).where(
                ChatThread.id == str(thread_id),
                ChatThread.user_id == str(user_id),
            )
            return s.execute(stmt).scalar_one_or_none()

        return self._execute(session, _op)

    def get_user_threads(
        self,
        user_id: str,
        limit: int = 50,
        offset: int = 0,
        session: Optional[Session] = None,
    ) -> List[ChatThread]:
        """List chat threads belonging to a user, ordered by most recently updated.

        Args:
            user_id: UUID of the owning user.
            limit: Maximum records to return.
            offset: Number of records to skip.
            session: Optional active database session.

        Returns:
            List of ChatThread entities.
        """
        def _op(s: Session) -> List[ChatThread]:
            stmt = (
                select(ChatThread)
                .where(ChatThread.user_id == str(user_id))
                .order_by(ChatThread.updated_at.desc())
                .limit(limit)
                .offset(offset)
            )
            return list(s.execute(stmt).scalars().all())

        return self._execute(session, _op)

    def update_thread_title(
        self,
        thread_id: str,
        user_id: str,
        title: str,
        session: Optional[Session] = None,
    ) -> Optional[ChatThread]:
        """Update thread title enforcing tenant ownership.

        Args:
            thread_id: UUID of the thread to update.
            user_id: UUID of the requesting user.
            title: New display title.
            session: Optional active database session.

        Returns:
            Updated ChatThread if authorized and found, None otherwise.
        """
        clean_title = title.strip() if title else "New Chat"

        def _op(s: Session) -> Optional[ChatThread]:
            stmt = select(ChatThread).where(
                ChatThread.id == str(thread_id),
                ChatThread.user_id == str(user_id),
            )
            thread = s.execute(stmt).scalar_one_or_none()
            if not thread:
                return None

            thread.title = clean_title
            thread.updated_at = datetime.now(timezone.utc)
            s.flush()
            s.refresh(thread)
            return thread

        return self._execute(session, _op)

    def delete_thread(
        self,
        thread_id: str,
        user_id: str,
        session: Optional[Session] = None,
    ) -> bool:
        """Delete chat thread and all associated messages enforcing tenant ownership.

        Args:
            thread_id: UUID of the thread to delete.
            user_id: UUID of the requesting user.
            session: Optional active database session.

        Returns:
            True if thread was deleted, False if thread was not found or unauthorized.
        """
        def _op(s: Session) -> bool:
            stmt = select(ChatThread).where(
                ChatThread.id == str(thread_id),
                ChatThread.user_id == str(user_id),
            )
            thread = s.execute(stmt).scalar_one_or_none()
            if not thread:
                return False

            s.delete(thread)
            s.flush()
            return True

        return self._execute(session, _op)

    # ==============================================================================
    # 3. MESSAGE LOGGING & AUDIT REPORT PERSISTENCE
    # ==============================================================================

    def add_message(
        self,
        thread_id: str,
        user_id: str,
        role: str,
        content: str,
        citations: Optional[List[Any]] = None,
        audit_report: Optional[Union[Dict[str, Any], TrustAuditReport]] = None,
        session: Optional[Session] = None,
    ) -> Message:
        """Log a conversational message with citations and TrustAuditReport JSON payload.

        Enforces strict tenant ownership: verifies that the thread belongs to user_id.

        Args:
            thread_id: UUID of the target chat thread.
            user_id: UUID of the requesting user (tenant verification).
            role: Speaker role ("user", "assistant", "system").
            content: Main text content of the message.
            citations: Optional list of retrieved candidate dicts or models.
            audit_report: Optional TrustAuditReport instance or JSON-compatible dictionary.
            session: Optional active database session.

        Returns:
            Newly created Message entity.

        Raises:
            PermissionError: If thread does not exist or does not belong to user_id.
        """
        # Serialize audit_report if provided as Pydantic model
        serialized_report: Optional[Dict[str, Any]] = None
        if audit_report is not None:
            if hasattr(audit_report, "model_dump"):
                serialized_report = audit_report.model_dump()
            elif isinstance(audit_report, dict):
                serialized_report = audit_report
            else:
                serialized_report = {"raw": str(audit_report)}

        # Serialize citations if provided as Pydantic models
        serialized_citations: Optional[Any] = None
        if citations is not None:
            if isinstance(citations, list):
                serialized_citations = [
                    c.model_dump() if hasattr(c, "model_dump") else c
                    for c in citations
                ]
            else:
                serialized_citations = citations

        def _op(s: Session) -> Message:
            stmt = select(ChatThread).where(
                ChatThread.id == str(thread_id),
                ChatThread.user_id == str(user_id),
            )
            thread = s.execute(stmt).scalar_one_or_none()
            if not thread:
                raise PermissionError(f"Unauthorized or thread '{thread_id}' not found.")

            # Update thread timestamp
            thread.updated_at = datetime.now(timezone.utc)

            msg = Message(
                thread_id=str(thread_id),
                role=str(role),
                content=str(content),
                citations=serialized_citations,
                audit_report=serialized_report,
            )
            s.add(msg)
            s.flush()
            s.refresh(msg)
            return msg

        return self._execute(session, _op)

    def get_thread_messages(
        self,
        thread_id: str,
        user_id: str,
        limit: int = 100,
        offset: int = 0,
        session: Optional[Session] = None,
    ) -> List[Message]:
        """Fetch chronological message history for a thread enforcing tenant ownership.

        Args:
            thread_id: UUID of the chat thread.
            user_id: UUID of requesting user.
            limit: Maximum messages to retrieve.
            offset: Number of messages to skip.
            session: Optional active database session.

        Returns:
            List of Message entities in chronological order.

        Raises:
            PermissionError: If thread is not owned by user_id.
        """
        def _op(s: Session) -> List[Message]:
            # Verify tenant ownership
            thread_stmt = select(ChatThread.id).where(
                ChatThread.id == str(thread_id),
                ChatThread.user_id == str(user_id),
            )
            if not s.execute(thread_stmt).scalar_one_or_none():
                raise PermissionError(f"Unauthorized or thread '{thread_id}' not found.")

            stmt = (
                select(Message)
                .where(Message.thread_id == str(thread_id))
                .order_by(Message.created_at.asc())
                .limit(limit)
                .offset(offset)
            )
            return list(s.execute(stmt).scalars().all())

        return self._execute(session, _op)

    def get_message(
        self,
        message_id: str,
        user_id: str,
        session: Optional[Session] = None,
    ) -> Optional[Message]:
        """Fetch a specific message by ID verifying user ownership of the parent thread.

        Args:
            message_id: UUID of the message.
            user_id: UUID of requesting user.
            session: Optional active database session.

        Returns:
            Message entity if found and authorized, None otherwise.
        """
        def _op(s: Session) -> Optional[Message]:
            stmt = (
                select(Message)
                .join(ChatThread, Message.thread_id == ChatThread.id)
                .where(
                    Message.id == str(message_id),
                    ChatThread.user_id == str(user_id),
                )
            )
            return s.execute(stmt).scalar_one_or_none()

        return self._execute(session, _op)

    def get_message_audit_report(
        self,
        message_id: str,
        user_id: str,
        session: Optional[Session] = None,
    ) -> Optional[TrustAuditReport]:
        """Retrieve and deserialize the stored TrustAuditReport for an assistant message.

        Args:
            message_id: UUID of the target message.
            user_id: UUID of requesting user.
            session: Optional active database session.

        Returns:
            Deserialized TrustAuditReport instance if present, None otherwise.
        """
        msg = self.get_message(message_id=message_id, user_id=user_id, session=session)
        if not msg or not msg.audit_report:
            return None
        return TrustAuditReport.model_validate(msg.audit_report)


# Global repository instance
repository = DatabaseRepository()
