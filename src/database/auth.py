r"""
================================================================================
1. PURPOSE & ROLE:
   - Module: src/database/auth.py
   - Role: Authentication, Password Hashing, and JWT Session Token Cryptography.
   - Purpose: Provides production-ready cryptographic utilities for user credential
     security using salted bcrypt hashing and stateless JSON Web Token (JWT) session
     generation and validation.

2. INPUT (IP):
   - Plaintext passwords for hashing and verification.
   - User identity attributes (user_id, email) and expiration durations for JWT minting.
   - Encoded JWT tokens for signature decoding and claim validation.

3. PROCESS UNDER THE HOOD:
   - Password Hashing & Verification:
     * Generates cryptographic salts and computes bcrypt hashes over UTF-8 encodings.
     * Compares plaintext against stored hashes using constant-time verification.
   - JWT Access Token Minting:
     * Constructs payload containing `sub` (user_id), `email`, `iat` (issued at), and
       `exp` (expiration timestamp) using timezone-aware UTC.
     * Signs payload with HS256 (or configured algorithm) using `JWT_SECRET`.
   - JWT Access Token Decoding:
     * Validates digital signature and timestamp validity.
     * Catches and surfaces `ExpiredSignatureError` and `InvalidTokenError` as clean ValueErrors.

4. OUTPUT (OP):
   - Cryptographic password hashes (str), boolean verification flags (bool),
     minted JWT access token strings (str), and decoded payload dictionaries (dict).
   - Consumed by: `src/database/repository.py`, API endpoints, and authentication middleware.

5. LIBRARIES & DEPENDENCIES:
   - bcrypt: Password hashing and verification.
   - datetime: Timezone-aware UTC timestamps and expiration deltas.
   - jwt (PyJWT): RFC 7519 compliant JSON Web Token signing and decoding.
   - os: Environment variable fallbacks.
   - typing: Type annotations.
   - src.common.config: Central database and security configuration.
================================================================================
"""

from datetime import datetime, timedelta, timezone
import os
from typing import Any, Dict, Optional

import bcrypt
import jwt

from src.common.config import config


def _get_jwt_secret(secret_override: Optional[str] = None) -> str:
    """Retrieve secret key for JWT signing from override, env, or configuration."""
    if secret_override:
        return secret_override
    env_secret = os.getenv("JWT_SECRET")
    if env_secret:
        return env_secret
    return getattr(config.database, "jwt_secret", "trustrag_super_secret_jwt_key_2026")


def _get_jwt_algorithm(algo_override: Optional[str] = None) -> str:
    """Retrieve algorithm for JWT signing from override, env, or configuration."""
    if algo_override:
        return algo_override
    env_algo = os.getenv("JWT_ALGORITHM")
    if env_algo:
        return env_algo
    return getattr(config.database, "jwt_algorithm", "HS256")


def hash_password(plain_password: str) -> str:
    """Hash a plaintext password using salted bcrypt.

    Args:
        plain_password: Cleartext password string.

    Returns:
        Salted bcrypt hash string.

    Raises:
        ValueError: If password is empty.
    """
    if not plain_password:
        raise ValueError("Password cannot be empty.")
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(plain_password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a cleartext password against a stored salted bcrypt hash.

    Args:
        plain_password: Cleartext password to check.
        hashed_password: Stored bcrypt hash string.

    Returns:
        True if password matches hash, False otherwise.
    """
    if not plain_password or not hashed_password:
        return False
    try:
        return bcrypt.checkpw(
            plain_password.encode("utf-8"),
            hashed_password.encode("utf-8"),
        )
    except (ValueError, TypeError):
        return False


def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> str:
    """Mint a signed JWT access token for an authenticated user session.

    Args:
        user_id: Unique identifier of the authenticated user.
        email: Email address of the user.
        expires_delta: Optional custom token expiration duration.
        secret_key: Optional signing key override.
        algorithm: Optional signing algorithm override (e.g. HS256).

    Returns:
        Encoded JWT token string.
    """
    now = datetime.now(timezone.utc)
    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire_minutes = getattr(config.database, "access_token_expire_minutes", 1440)
        expire = now + timedelta(minutes=expire_minutes)

    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "email": str(email),
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    secret = _get_jwt_secret(secret_key)
    algo = _get_jwt_algorithm(algorithm)
    return jwt.encode(payload, secret, algorithm=algo)


def decode_access_token(
    token: str,
    secret_key: Optional[str] = None,
    algorithm: Optional[str] = None,
) -> Dict[str, Any]:
    """Decode and validate a JWT access token.

    Args:
        token: Encoded JWT string.
        secret_key: Optional signing key override.
        algorithm: Optional signing algorithm override.

    Returns:
        Decoded token payload dictionary.

    Raises:
        ValueError: If token is expired, malformed, or has an invalid signature.
    """
    if not token:
        raise ValueError("Token cannot be empty.")

    secret = _get_jwt_secret(secret_key)
    algo = _get_jwt_algorithm(algorithm)

    try:
        payload = jwt.decode(token, secret, algorithms=[algo])
        return payload
    except jwt.ExpiredSignatureError as e:
        raise ValueError("Token has expired.") from e
    except jwt.InvalidTokenError as e:
        raise ValueError("Invalid token signature or structure.") from e
