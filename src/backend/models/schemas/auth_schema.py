from pydantic import BaseModel, Field, field_validator
from typing import Optional

import re

from pydantic import EmailStr


# Schema untuk Validasi Input (Request) - Signup
class SignupRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=50, description="Nama lengkap / username user")
    email: EmailStr = Field(..., description="Email user (divalidasi formatnya)")
    password: str = Field(..., description="Min 8 karakter, mengandung huruf dan simbol")

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, value: str) -> str:
        """
        Aturan password: minimal 8 karakter, mengandung minimal
        1 huruf, 1 angka, dan 1 simbol (non-alfanumerik, mis. , . " / ; : [ ] { } dll).
        Semua pelanggaran dilaporkan sekaligus, bukan berhenti di pelanggaran pertama.
        """
        errors = []
        if len(value) < 8:
            errors.append("minimal 8 karakter")
        if not re.search(r"[A-Za-z]", value):
            errors.append("harus mengandung minimal satu huruf")
        if not re.search(r"[0-9]", value):
            errors.append("harus mengandung minimal satu angka")
        if not re.search(r"[^A-Za-z0-9]", value):
            errors.append("harus mengandung minimal satu simbol")

        if errors:
            raise ValueError("Password " + "; ".join(errors) + ".")

        return value


# Schema untuk Validasi Input (Request) - Login
class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1)


# Payload token hasil login
class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int  # masa aktif access token dalam detik (900 = 15 menit)


# Payload respons untuk operasi tanpa data (mis. signup)
class MessageResponse(BaseModel):
    detail: str = ""
