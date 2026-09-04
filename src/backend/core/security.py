import bcrypt
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import jwt

from src.backend.core.config import settings


def hash_password(plain_password: str) -> str:
    """Hash password plaintext menggunakan bcrypt."""
    return bcrypt.hashpw(plain_password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, password_hash: str) -> bool:
    """Verifikasi password plaintext terhadap hash bcrypt di database."""
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), password_hash.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_token(user_id: str, email: str, token_type: str, expires_delta: timedelta) -> str:
    """
    Buat JWT berisi: sub (user id), email, type (access/refresh), iat, exp.
    Ditandatangani HS256 dengan JWT_SECRET_KEY.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": email,
        "type": token_type,
        "iat": now,
        "exp": now + expires_delta,
        # ID unik per token: token yang diterbitkan di detik yang sama pun tetap berbeda
        "jti": uuid4().hex,
    }
    return jwt.encode(payload, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def create_access_token(user_id: str, email: str) -> str:
    return create_token(user_id, email, "access", timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))


def create_refresh_token(user_id: str, email: str) -> str:
    return create_token(user_id, email, "refresh", timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS))


def decode_token(token: str) -> dict:
    """
    Decode & verifikasi JWT. Melempar jwt.ExpiredSignatureError jika kedaluwarsa
    atau jwt.InvalidTokenError jika token tidak valid.
    """
    return jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
