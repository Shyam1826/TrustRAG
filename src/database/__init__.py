r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/database/__init__.py
   - Role: Public API export package for TrustRAG Relational Database & Ledger.
   - Purpose: Exposes ORM models, session context managers, password hashing, JWT
     token issuance/verification, and the multi-tenant database repository.

2. INPUT (IP):
   - Package-level imports from models, connection, auth, and repository.

3. PROCESS UNDER THE HOOD:
   - Aggregates and re-exports core database components.

4. OUTPUT (OP):
   - Clean public symbols for database interaction.
   - Consumed by: `src/main.py`, FastAPI/Flask routers, CLI scripts, and test suites.

5. LIBRARIES & DEPENDENCIES:
   - src.database.models, src.database.connection, src.database.auth, src.database.repository.
================================================================================
"""

from src.database.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.database.connection import (
    create_db_engine,
    create_session_factory,
    engine,
    get_db_session,
    init_db,
    reset_db,
    SessionLocal,
)
from src.database.models import (
    Base,
    ChatThread,
    Message,
    User,
)
from src.database.repository import (
    DatabaseRepository,
    repository,
)

__all__ = [
    "Base",
    "User",
    "ChatThread",
    "Message",
    "create_db_engine",
    "create_session_factory",
    "engine",
    "SessionLocal",
    "get_db_session",
    "init_db",
    "reset_db",
    "hash_password",
    "verify_password",
    "create_access_token",
    "decode_access_token",
    "DatabaseRepository",
    "repository",
]
