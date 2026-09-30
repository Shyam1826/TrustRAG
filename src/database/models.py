r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/database/models.py
   - Role: Relational schema and ORM entity models for TrustRAG.
   - Purpose: Defines User, ChatThread, and Message entities using SQLAlchemy 2.0
     DeclarativeBase, supporting persistent multi-tenant chat threads, user
     authentication, structured message history, and verification audit report JSON payloads.

2. INPUT (IP):
   - User credentials, chat thread metadata, and conversational messages with
     citations and TrustAuditReport JSON representations.

3. PROCESS UNDER THE HOOD:
   - Declarative base modeling with typed Mapped attributes and mapped_column.
   - UUID primary key generation for distributed collision resistance.
   - Enforces referential integrity and cascading deletes:
     * User -> ChatThread (cascade="all, delete-orphan", ondelete="CASCADE")
     * ChatThread -> Message (cascade="all, delete-orphan", ondelete="CASCADE")
   - Universal JSON columns for citations and TrustAuditReport storage.
   - UTC timestamp tracking for created_at and updated_at fields.

4. OUTPUT (OP):
   - SQLAlchemy ORM models: User, ChatThread, Message, and Declarative Base.
   - Consumed by: `src/database/connection.py` and `src/database/repository.py`.

5. LIBRARIES & DEPENDENCIES:
   - datetime: UTC timestamp generation.
   - typing: Type annotations (List, Optional, Dict, Any).
   - uuid: UUID4 string generation.
   - sqlalchemy: DeclarativeBase, Mapped, mapped_column, relationship, ForeignKey,
     String, Boolean, DateTime, Text, JSON.
================================================================================
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import Boolean, DateTime, ForeignKey, JSON, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _generate_uuid() -> str:
    """Generate a 36-character UUID string."""
    return str(uuid.uuid4())


def _utcnow() -> datetime:
    """Return current timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """Base declarative class for all TrustRAG database entities."""
    pass


class User(Base):
    """User account entity for authentication and tenant isolation."""

    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    # 1-to-many relationship: User -> ChatThreads (Cascaded Delete)
    threads: Mapped[List["ChatThread"]] = relationship(
        "ChatThread",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<User id={self.id!r} email={self.email!r} is_active={self.is_active}>"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize user model to dictionary (excluding sensitive password)."""
        return {
            "id": self.id,
            "email": self.email,
            "full_name": self.full_name,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ChatThread(Base):
    """Chat thread session entity belonging to an authenticated user."""

    __tablename__ = "chat_threads"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    title: Mapped[str] = mapped_column(String(255), default="New Chat", nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow, nullable=False)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="threads")
    messages: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="thread",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<ChatThread id={self.id!r} user_id={self.user_id!r} title={self.title!r}>"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize thread model to dictionary."""
        return {
            "id": self.id,
            "user_id": self.user_id,
            "title": self.title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Message(Base):
    """Individual conversational message record within a chat thread."""

    __tablename__ = "messages"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_generate_uuid)
    thread_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("chat_threads.id", ondelete="CASCADE"),
        index=True,
        nullable=False,
    )
    role: Mapped[str] = mapped_column(String(50), nullable=False)  # "user", "assistant", "system"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    citations: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    audit_report: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    thread: Mapped["ChatThread"] = relationship("ChatThread", back_populates="messages")

    def __repr__(self) -> str:
        return f"<Message id={self.id!r} thread_id={self.thread_id!r} role={self.role!r}>"

    def to_dict(self) -> Dict[str, Any]:
        """Serialize message record to dictionary."""
        return {
            "id": self.id,
            "thread_id": self.thread_id,
            "role": self.role,
            "content": self.content,
            "citations": self.citations,
            "audit_report": self.audit_report,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
