from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.backend.core.deps import get_current_user, get_refresh_payload
from src.backend.models.database import get_db
from src.backend.models.schemas.auth_schema import (
    LoginRequest,
    MessageResponse,
    SignupRequest,
    TokenResponse,
)
from src.backend.models.schemas.portfolio_schema import ApiResponse
from src.backend.controller.auth_controller import (
    InvalidCredentialsError,
    UserAlreadyExistsError,
    login,
    refresh,
    signup,
)

router = APIRouter()

@router.post("/signup", response_model=ApiResponse[MessageResponse], status_code=201)
def api_signup(body: SignupRequest, db: Session = Depends(get_db)):
    """
    Daftarkan user baru (name, email, password). Password di-hash dengan bcrypt.
    Tidak mengembalikan token — user login manual setelahnya.
    """
    try:
        signup(db, body)
    except UserAlreadyExistsError as e:
        raise HTTPException(status_code=409, detail=str(e))
    return ApiResponse(status="success", message="Signup berhasil. Silakan login.", data=MessageResponse(detail="User created"))

@router.post("/login", response_model=ApiResponse[TokenResponse])
def api_login(body: LoginRequest, db: Session = Depends(get_db)):
    """
    Login dengan email + password. Mengembalikan access token (15 menit)
    dan refresh token (7 hari).
    """
    try:
        token = login(db, body)
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return ApiResponse(status="success", message="Login berhasil.", data=token)

@router.post("/refresh", response_model=ApiResponse[TokenResponse])
def api_refresh_token(
    db: Session = Depends(get_db),
    refresh_data: tuple = Depends(get_refresh_payload),
):
    """
    Tukar refresh token (dikirim via header Authorization: Bearer <refresh_token>)
    dengan access token baru. Hanya menerima token dengan type 'refresh' —
    access token yang dikirim ke sini akan ditolak 401.
    """
    refresh_payload, raw_refresh_token = refresh_data
    try:
        token = refresh(db, raw_refresh_token)
    except InvalidCredentialsError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return ApiResponse(status="success", message="Access token berhasil diperbarui.", data=token)

@router.get("/me", response_model=ApiResponse[MessageResponse])
def api_me(current_user: dict = Depends(get_current_user)):
    """
    Contoh endpoint terproteksi: hanya menerima ACCESS token.
    Mengembalikan identitas user dari payload JWT.
    """
    data = MessageResponse(detail=f"user_id={current_user['sub']}, email={current_user['email']}")
    return ApiResponse(status="success", message="Token valid.", data=data)

