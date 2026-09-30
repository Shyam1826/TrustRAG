r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: tests/test_database.py
   - Role: Unit and integration test suite for TrustRAG Relational Database & Ledger.
   - Purpose: Validates user registration, duplicate email rejection, bcrypt credential
     verification, JWT token minting/expiration, multi-tenant thread isolation, message
     logging, cascade deletion, and full TrustAuditReport JSON serialization/deserialization.

2. INPUT (IP):
   - In-memory SQLite database (`sqlite:///:memory:`) with dynamic schema initialization.
   - Synthetic user accounts, credentials, conversational threads, and TrustAuditReports.

3. PROCESS UNDER THE HOOD:
   - Sets up isolated test session fixtures using `create_db_engine("sqlite:///:memory:")`.
   - Tests bcrypt hashing, verification, and boundary validation.
   - Tests JWT creation, payload decoding, expired token detection, and tamper rejection.
   - Tests user creation, duplicate email rejection, and authentication.
   - Tests thread creation, title mutation, pagination, and multi-tenant isolation.
   - Tests message logging with citations and full TrustAuditReport Pydantic roundtripping.
   - Tests referential cascading deletions across User -> ChatThread -> Message.

4. OUTPUT (OP):
   - Pytest test execution results.

5. LIBRARIES & DEPENDENCIES:
   - datetime: Timezone-aware UTC timestamp calculations.
   - pytest: Test fixtures and assertion framework.
   - src.common.schemas: ClaimAudit, RetrievalCandidate, TrustAuditReport.
   - src.database: Auth utilities, connection factories, models, and DatabaseRepository.
================================================================================
"""

from datetime import timedelta
import pytest
from sqlalchemy.orm import Session

from src.common.schemas import ClaimAudit, RetrievalCandidate, TrustAuditReport
from src.database.auth import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from src.database.connection import (
    create_db_engine,
    create_session_factory,
    init_db,
)
from src.database.models import ChatThread, Message, User
from src.database.repository import DatabaseRepository


@pytest.fixture
def test_repo() -> DatabaseRepository:
    """Fixture providing an isolated in-memory SQLite database repository."""
    engine = create_db_engine("sqlite:///:memory:")
    init_db(engine_override=engine)
    session_factory = create_session_factory(engine)
    return DatabaseRepository(session_factory=session_factory)


def test_auth_password_hashing_and_verification() -> None:
    """Verify salted bcrypt password hashing and constant-time verification."""
    password = "SuperSecretPassword123!"
    hashed = hash_password(password)

    assert hashed != password
    assert hashed.startswith("$2b$") or hashed.startswith("$2a$")

    # Positive match
    assert verify_password(password, hashed) is True

    # Negative matches
    assert verify_password("WrongPassword", hashed) is False
    assert verify_password("", hashed) is False
    assert verify_password(password, "") is False

    # Empty password guard
    with pytest.raises(ValueError, match="cannot be empty"):
        hash_password("")


def test_auth_jwt_token_minting_and_decoding() -> None:
    """Verify JWT access token minting, claim decoding, expiration, and tampering guards."""
    user_id = "user_abc_123"
    email = "researcher@enterprise.ai"

    # 1. Valid token
    token = create_access_token(user_id=user_id, email=email, expires_delta=timedelta(minutes=15))
    payload = decode_access_token(token)

    assert payload["sub"] == user_id
    assert payload["email"] == email
    assert "exp" in payload
    assert "iat" in payload

    # 2. Expired token
    expired_token = create_access_token(user_id=user_id, email=email, expires_delta=timedelta(seconds=-10))
    with pytest.raises(ValueError, match="Token has expired"):
        decode_access_token(expired_token)

    # 3. Tampered token
    tampered = token[:-4] + "xyz1"
    with pytest.raises(ValueError, match="Invalid token"):
        decode_access_token(tampered)

    # 4. Empty token
    with pytest.raises(ValueError, match="cannot be empty"):
        decode_access_token("")


def test_user_registration_duplicate_and_authentication(test_repo: DatabaseRepository) -> None:
    """Verify user registration, duplicate rejection, and credential authentication."""
    # 1. Successful creation
    user = test_repo.create_user(
        email="Alice@Enterprise.AI",
        password="AliceSecurePassword999",
        full_name="Alice Smith",
    )
    assert user.id is not None
    assert user.email == "alice@enterprise.ai"  # Normalized lowercase
    assert user.full_name == "Alice Smith"
    assert user.is_active is True

    # 2. Duplicate email rejection (case-insensitive)
    with pytest.raises(ValueError, match="already exists"):
        test_repo.create_user(
            email="alice@enterprise.ai",
            password="AnotherPassword123",
        )

    # 3. Empty credentials guard
    with pytest.raises(ValueError, match="Email address cannot be empty"):
        test_repo.create_user(email="", password="SomePassword")

    with pytest.raises(ValueError, match="Password cannot be empty"):
        test_repo.create_user(email="bob@enterprise.ai", password="")

    # 4. Authentication
    auth_user = test_repo.authenticate_user("alice@enterprise.ai", "AliceSecurePassword999")
    assert auth_user is not None
    assert auth_user.id == user.id

    # Wrong password
    assert test_repo.authenticate_user("alice@enterprise.ai", "WrongPass") is None

    # Unknown user
    assert test_repo.authenticate_user("unknown@enterprise.ai", "SomePass") is None


def test_chat_thread_creation_and_pagination(test_repo: DatabaseRepository) -> None:
    """Verify thread creation, title update, and paginated listing."""
    user = test_repo.create_user(email="analyst@firm.com", password="SecurePassword123")

    # Create 5 threads
    for i in range(5):
        test_repo.create_thread(user_id=user.id, title=f"Audit Thread {i}")

    # Fetch with pagination
    threads_page1 = test_repo.get_user_threads(user_id=user.id, limit=3, offset=0)
    assert len(threads_page1) == 3

    threads_page2 = test_repo.get_user_threads(user_id=user.id, limit=3, offset=3)
    assert len(threads_page2) == 2

    # Update title
    target_thread = threads_page1[0]
    updated = test_repo.update_thread_title(
        thread_id=target_thread.id,
        user_id=user.id,
        title="Renamed Contract Review",
    )
    assert updated is not None
    assert updated.title == "Renamed Contract Review"

    # Query non-existent user
    with pytest.raises(ValueError, match="does not exist"):
        test_repo.create_thread(user_id="non_existent_uuid", title="Orphan Thread")


def test_strict_tenant_isolation(test_repo: DatabaseRepository) -> None:
    """Verify strict tenant isolation: users cannot access or mutate threads owned by others."""
    user_a = test_repo.create_user(email="user_a@firm.com", password="PasswordA123")
    user_b = test_repo.create_user(email="user_b@firm.com", password="PasswordB123")

    # User A creates a thread
    thread_a = test_repo.create_thread(user_id=user_a.id, title="User A Confidential")

    # User B attempts to access User A's thread
    assert test_repo.get_thread(thread_id=thread_a.id, user_id=user_b.id) is None

    # User B attempts to update User A's thread
    assert test_repo.update_thread_title(thread_id=thread_a.id, user_id=user_b.id, title="Hacked") is None

    # User B attempts to add a message to User A's thread
    with pytest.raises(PermissionError, match="Unauthorized or thread"):
        test_repo.add_message(
            thread_id=thread_a.id,
            user_id=user_b.id,
            role="user",
            content="Unauthorized inquiry",
        )

    # User B attempts to read messages from User A's thread
    with pytest.raises(PermissionError, match="Unauthorized or thread"):
        test_repo.get_thread_messages(thread_id=thread_a.id, user_id=user_b.id)

    # User B attempts to delete User A's thread
    assert test_repo.delete_thread(thread_id=thread_a.id, user_id=user_b.id) is False


def test_message_logging_and_trust_audit_report_roundtrip(test_repo: DatabaseRepository) -> None:
    """Verify structured message logging with citations and full TrustAuditReport roundtrip."""
    user = test_repo.create_user(email="auditor@firm.com", password="Password123")
    thread = test_repo.create_thread(user_id=user.id, title="Financial Analysis")

    # 1. Log User Question
    user_msg = test_repo.add_message(
        thread_id=thread.id,
        user_id=user.id,
        role="user",
        content="What is the liability cap under Section 4?",
    )
    assert user_msg.id is not None
    assert user_msg.role == "user"

    # 2. Build full TrustAuditReport
    mock_candidates = [
        RetrievalCandidate(
            parent_id="p_1",
            doc_id="Doc-1",
            page_number=4,
            text="Section 4: Total aggregate liability is strictly capped at $1,000,000.",
            score=0.96,
            match_type="dense",
        )
    ]

    report = TrustAuditReport(
        draft_text="Total aggregate liability is capped at $1,000,000 [Doc-1].",
        faithfulness_score=1.0,
        has_contradiction=False,
        action="PASS",
        audits=[
            ClaimAudit(
                claim_id="c_1",
                claim_text="Total aggregate liability is capped at $1,000,000.",
                cited_premise="Total aggregate liability is strictly capped at $1,000,000.",
                probabilities={"entailment": 0.99, "contradiction": 0.005, "neutral": 0.005},
                verdict="ENTAILED",
                confidence=0.99,
            )
        ],
    )

    # 3. Log Assistant Response with Citations & TrustAuditReport
    asst_msg = test_repo.add_message(
        thread_id=thread.id,
        user_id=user.id,
        role="assistant",
        content=report.draft_text,
        citations=mock_candidates,
        audit_report=report,
    )
    assert asst_msg.id is not None
    assert asst_msg.role == "assistant"
    assert asst_msg.audit_report is not None

    # 4. Fetch Messages
    messages = test_repo.get_thread_messages(thread_id=thread.id, user_id=user.id)
    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    # 5. Roundtrip Deserialization of TrustAuditReport
    restored_report = test_repo.get_message_audit_report(message_id=asst_msg.id, user_id=user.id)
    assert restored_report is not None
    assert isinstance(restored_report, TrustAuditReport)
    assert restored_report.faithfulness_score == 1.0
    assert restored_report.action == "PASS"
    assert len(restored_report.audits) == 1
    assert restored_report.audits[0].verdict == "ENTAILED"
    assert restored_report.audits[0].confidence == 0.99


def test_cascade_delete_user_and_thread(test_repo: DatabaseRepository) -> None:
    """Verify cascading deletion across User -> ChatThread -> Message."""
    user = test_repo.create_user(email="cleanup@firm.com", password="Password123")
    thread = test_repo.create_thread(user_id=user.id, title="Thread To Delete")

    # Add messages
    test_repo.add_message(thread_id=thread.id, user_id=user.id, role="user", content="Question 1")
    test_repo.add_message(thread_id=thread.id, user_id=user.id, role="assistant", content="Answer 1")

    # 1. Delete Thread cascades to Messages
    assert test_repo.delete_thread(thread_id=thread.id, user_id=user.id) is True
    assert test_repo.get_thread(thread_id=thread.id, user_id=user.id) is None

    # Re-query messages directly via session to verify deletion
    with test_repo.session_factory() as s:
        remaining_msgs = s.query(Message).filter(Message.thread_id == thread.id).all()
        assert len(remaining_msgs) == 0

    # 2. Delete User cascades to Threads and Messages
    thread2 = test_repo.create_thread(user_id=user.id, title="Thread 2")
    test_repo.add_message(thread_id=thread2.id, user_id=user.id, role="user", content="Question 2")

    assert test_repo.delete_user(user_id=user.id) is True
    assert test_repo.get_user_by_id(user_id=user.id) is None

    with test_repo.session_factory() as s:
        remaining_threads = s.query(ChatThread).filter(ChatThread.user_id == user.id).all()
        assert len(remaining_threads) == 0
        remaining_msgs2 = s.query(Message).filter(Message.thread_id == thread2.id).all()
        assert len(remaining_msgs2) == 0
