r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/database/connection.py
   - Role: Relational Database Engine, Connection Pool, and Session Management.
   - Purpose: Establishes thread-safe SQLAlchemy engine and session factory for
     TrustRAG. Defaults to local SQLite (`sqlite:///data/trustrag.db`) with automatic
     directory creation and dynamic environment override (`DATABASE_URL`). Enforces
     SQLite foreign key constraints and provides transactional session context managers.

2. INPUT (IP):
   - Database URL from environment (`DATABASE_URL`), YAML config, or runtime override.
   - Engine configuration options (echo, pool settings).

3. PROCESS UNDER THE HOOD:
   - Dynamic URL Resolution:
     * Reads `DATABASE_URL` environment variable or falls back to `config.database.url`.
     * If SQLite file-backed URL is detected, ensures parent storage directory exists.
     * Configures SQLite with `check_same_thread=False` and attaches an engine connect
       listener to enforce `PRAGMA foreign_keys=ON`.
   - Engine & Session Factory Initialization:
     * Instantiates global `engine` and `SessionLocal` sessionmaker.
   - Transactional Context Manager (`get_db_session`):
     * Automatically commits on successful execution, rolls back on exception, and
       guarantees session closure in `finally` block.
   - Database Schema Initializer (`init_db`):
     * Invokes `Base.metadata.create_all(bind=engine)` to create all tables idempotently.

4. OUTPUT (OP):
   - SQLAlchemy Engine, SessionLocal factory, `get_db_session` context manager, and
     `init_db` initialization routine.
   - Consumed by: `src/database/repository.py`, `src/main.py`, and test suites.

5. LIBRARIES & DEPENDENCIES:
   - contextlib: Context manager implementation.
   - os, pathlib.Path: File system and environment variable handling.
   - sqlite3: SQLite database driver and pragma inspection.
   - sqlalchemy: create_engine, Engine, event, sessionmaker, Session.
   - src.common.config: Central configuration settings.
   - src.database.models: Declarative Base and entity schemas.
================================================================================
"""

from contextlib import contextmanager
import os
from pathlib import Path
import sqlite3
from typing import Any, Generator, Optional

from sqlalchemy import create_engine, Engine, event
from sqlalchemy.orm import sessionmaker, Session

from src.common.config import config
from src.database.models import Base


@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection: Any, connection_record: Any) -> None:
    """Enforce foreign key constraint validation when using SQLite engines."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _resolve_database_url(url_override: Optional[str] = None) -> str:
    """Resolve active database URL from override, environment, or configuration."""
    if url_override:
        return url_override
    env_url = os.getenv("DATABASE_URL")
    if env_url:
        return env_url
    return getattr(config.database, "url", "sqlite:///data/trustrag.db")


def create_db_engine(url: Optional[str] = None, echo: Optional[bool] = None) -> Engine:
    """Factory creating an SQLAlchemy Engine with dialect-specific optimizations.

    Args:
        url: Optional database connection string. Defaults to resolved configuration URL.
        echo: Optional boolean flag to log SQL statements.

    Returns:
        Configured SQLAlchemy Engine instance.
    """
    resolved_url = _resolve_database_url(url)
    is_echo = echo if echo is not None else getattr(config.database, "echo", False)

    connect_args = {}
    if resolved_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # Ensure target directory exists for file-backed SQLite
        if not resolved_url.startswith("sqlite:///:memory:") and resolved_url.startswith("sqlite:///"):
            clean_path = resolved_url.replace("sqlite:///", "")
            target_file = Path(clean_path)
            target_dir = target_file.parent
            if str(target_dir) and str(target_dir) != ".":
                target_dir.mkdir(parents=True, exist_ok=True)

    return create_engine(
        resolved_url,
        echo=is_echo,
        connect_args=connect_args,
        future=True,
    )


def create_session_factory(engine_instance: Engine) -> sessionmaker[Session]:
    """Create a sessionmaker bound to the provided engine."""
    return sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine_instance,
        expire_on_commit=False,
    )


# Primary application engine and session factory
engine: Engine = create_db_engine()
SessionLocal: sessionmaker[Session] = create_session_factory(engine)


@contextmanager
def get_db_session(session_factory: Optional[sessionmaker[Session]] = None) -> Generator[Session, None, None]:
    """Context manager yielding a transactional database session.

    Commits automatically on success, rolls back on error, and ensures closure.

    Args:
        session_factory: Optional sessionmaker instance. Defaults to SessionLocal.

    Yields:
        Active SQLAlchemy Session.
    """
    factory = session_factory or SessionLocal
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def init_db(engine_override: Optional[Engine] = None) -> None:
    """Create all relational tables declared in Base metadata if they do not exist.

    Args:
        engine_override: Optional Engine instance. Defaults to the global engine.
    """
    target_engine = engine_override or engine
    Base.metadata.create_all(bind=target_engine)


def reset_db(engine_override: Optional[Engine] = None) -> None:
    """Drop and recreate all relational tables. Used primarily for test teardown."""
    target_engine = engine_override or engine
    Base.metadata.drop_all(bind=target_engine)
    Base.metadata.create_all(bind=target_engine)
