import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from src.backend.core.security import decode_token

# Skema Bearer: mengambil token dari header "Authorization: Bearer <token>"
bearer_scheme = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    Dependency untuk endpoint yang TERPROTEKSI.
    Memvalidasi Bearer token dan memastikan type-nya 'access'.
    Token refresh yang dipakai untuk endpoint ini akan DITOLAK.
    Mengembalikan payload JWT: {sub (user id), email, type, iat, exp}.
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Token tidak ditemukan. Sertakan header Authorization: Bearer <access_token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token kedaluwarsa. Silakan login ulang atau gunakan refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Token tidak valid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Pengecekan tipe token: endpoint terproteksi hanya menerima access token
    if payload.get("type") != "access":
        raise HTTPException(
            status_code=401,
            detail="Tipe token salah. Endpoint ini hanya menerima access token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload


def get_refresh_payload(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> tuple:
    """
    Dependency untuk endpoint /auth/refresh.
    Memastikan token yang dikirim adalah refresh token (type == 'refresh').
    Access token yang dipakai di sini akan DITOLAK.
    Mengembalikan tuple (payload_jwt, raw_token).
    """
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Token tidak ditemukan. Sertakan header Authorization: Bearer <refresh_token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    try:
        payload = decode_token(token)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Refresh token kedaluwarsa. Silakan login ulang.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=401,
            detail="Token tidak valid.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if payload.get("type") != "refresh":
        raise HTTPException(
            status_code=401,
            detail="Tipe token salah. Endpoint ini hanya menerima refresh token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return payload, token
