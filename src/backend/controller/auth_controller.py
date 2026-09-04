import logging
import uuid as uuid_lib

from sqlalchemy.orm import Session

from src.backend.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    hash_password,
    verify_password,
)
from src.backend.models.users import User
from src.backend.models.schemas.auth_schema import (
    LoginRequest,
    SignupRequest,
    TokenResponse,
)

logger = logging.getLogger(__name__)


class UserAlreadyExistsError(Exception):
    """Exception domain: email/username sudah terdaftar."""


class InvalidCredentialsError(Exception):
    """Exception domain: email atau password salah."""


def signup(db: Session, request: SignupRequest) -> None:
    """
    Daftarkan user baru. Tidak mengembalikan token — user login manual setelahnya.
    Raises: UserAlreadyExistsError jika email/username sudah dipakai.
    """
    exists = (
        db.query(User)
        .filter((User.email == request.email) | (User.username == request.name))
        .first()
    )
    if exists:
        raise UserAlreadyExistsError("Email atau username sudah terdaftar.")
    
    

    user = User(
        username=request.name,
        email=str(request.email),
        password_hash=hash_password(request.password),
    )
    db.add(user)
    db.commit()
    logger.info("User baru terdaftar: %s <%s>", user.username, user.email)


def login(db: Session, request: LoginRequest) -> TokenResponse:
    """
    Verifikasi email + password, lalu terbitkan access token (15 menit)
    dan refresh token (7 hari).
    Raises: InvalidCredentialsError jika kredensial tidak cocok.
    """
    user = db.query(User).filter(User.email == str(request.email)).first()

    # Error generik agar tidak membocorkan field mana yang salah
    if user is None or not verify_password(request.password, user.password_hash):
        raise InvalidCredentialsError("Email atau password salah.")

    from src.backend.core.config import settings

    return TokenResponse(
        access_token=create_access_token(user.id, user.email),
        refresh_token=create_refresh_token(user.id, user.email),
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


def refresh(db: Session, refresh_token_str: str) -> TokenResponse:
    """
    Tukar refresh token yang valid dengan access token baru (15 menit).
    Pengecekan type == 'refresh' sudah dilakukan di dependency get_refresh_payload;
    di sini kita pastikan user masih ada di database.
    Raises: InvalidCredentialsError jika user tidak ditemukan lagi.
    """
    payload = decode_token(refresh_token_str)

    # sub di JWT berupa string; kolom UUID butuh objek uuid.UUID
    user = db.query(User).filter(User.id == uuid_lib.UUID(payload.get("sub"))).first()
    if user is None:
        raise InvalidCredentialsError("User tidak ditemukan. Silakan login ulang.")

    from src.backend.core.config import settings

    return TokenResponse(
        access_token=create_access_token(user.id, user.email),
        # Refresh token lama tetap dipakai sampai kedaluwarsa (7 hari)
        refresh_token=refresh_token_str,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )
