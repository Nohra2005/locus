import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timedelta

import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from passlib.context import CryptContext
from pydantic import BaseModel

SECRET_KEY = os.getenv("JWT_SECRET", "locus_store_portal_secret_2026")
ALGORITHM = "HS256"
TOKEN_EXPIRE_DAYS = 30

USERS_FILE = os.path.join(os.path.dirname(__file__), "users.json")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer = HTTPBearer(auto_error=False)


def _prehash(plain: str) -> str:
    """SHA-256 prehash so bcrypt never sees more than 44 bytes."""
    digest = hashlib.sha256(plain.encode("utf-8")).digest()
    return base64.b64encode(digest).decode("ascii")


def _hash_password(plain: str) -> str:
    return pwd_context.hash(_prehash(plain))


def _verify_password(plain: str, stored: str) -> bool:
    # Try new prehash method first.
    if pwd_context.verify(_prehash(plain), stored):
        return True
    # Fallback: old accounts stored without prehash (passwords ≤ 72 bytes).
    try:
        return pwd_context.verify(plain, stored)
    except Exception:
        return False


def _load_users() -> dict:
    try:
        with open(USERS_FILE) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_users(users: dict) -> None:
    with open(USERS_FILE, "w") as f:
        json.dump(users, f, indent=2)


def _make_token(email: str, store_id: str, store_name: str) -> str:
    payload = {
        "sub":        email,
        "store_id":   store_id,
        "store_name": store_name,
        "exp":        datetime.utcnow() + timedelta(days=TOKEN_EXPIRE_DAYS),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def verify_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired — please log in again")
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")


def optional_token(credentials: HTTPAuthorizationCredentials = Depends(_bearer)):
    if not credentials:
        return None
    try:
        return jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
    except Exception:
        return None


# ── Request/response models ────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email:      str
    password:   str
    store_name: str
    mall:       str
    phone:      str = ""


class LoginRequest(BaseModel):
    email:    str
    password: str


class ProfileUpdateRequest(BaseModel):
    store_name: str | None = None
    mall:       str | None = None
    phone:      str | None = None


class ForgotPasswordRequest(BaseModel):
    email: str


class ResetPasswordRequest(BaseModel):
    email:        str
    code:         str
    new_password: str


# ── Router ─────────────────────────────────────────────────────────────────────

router = APIRouter(prefix="/auth", tags=["auth"])


@router.post("/register")
async def register(req: RegisterRequest):
    if len(req.password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if len(req.password) > 128:
        raise HTTPException(400, "Password must be no more than 128 characters")
    users = _load_users()
    email = req.email.strip().lower()
    if email in users:
        raise HTTPException(400, "Email already registered")
    store_id = str(uuid.uuid4())
    users[email] = {
        "store_id":   store_id,
        "email":      email,
        "password":   _hash_password(req.password),
        "store_name": req.store_name.strip(),
        "mall":       req.mall.strip(),
        "phone":      req.phone.strip(),
        "created_at": datetime.utcnow().isoformat(),
    }
    _save_users(users)
    return {
        "access_token": _make_token(email, store_id, req.store_name.strip()),
        "token_type":   "bearer",
        "email":        email,
        "store_name":   req.store_name.strip(),
        "store_id":     store_id,
        "mall":         req.mall.strip(),
        "phone":        req.phone.strip(),
    }


@router.post("/login")
async def login(req: LoginRequest):
    users = _load_users()
    email = req.email.strip().lower()
    user  = users.get(email)
    if not user:
        raise HTTPException(401, "Invalid email or password")

    if not _verify_password(req.password, user["password"]):
        raise HTTPException(401, "Invalid email or password")

    # Migrate old direct-bcrypt hashes to prehash format on successful login.
    old_hash = user["password"]
    try:
        if pwd_context.verify(_prehash(req.password), old_hash) is False:
            user["password"] = _hash_password(req.password)
            _save_users(users)
    except Exception:
        pass

    return {
        "access_token": _make_token(email, user["store_id"], user["store_name"]),
        "token_type":   "bearer",
        "email":        email,
        "store_name":   user["store_name"],
        "store_id":     user["store_id"],
        "mall":         user.get("mall", ""),
        "phone":        user.get("phone", ""),
    }


@router.get("/me")
async def me(payload=Depends(verify_token)):
    users = _load_users()
    u = users.get(payload["sub"], {})
    return {
        "email":      payload["sub"],
        "store_id":   payload["store_id"],
        "store_name": payload["store_name"],
        "mall":       u.get("mall", ""),
        "phone":      u.get("phone", ""),
        "created_at": u.get("created_at", ""),
    }


@router.post("/forgot-password")
async def forgot_password(req: ForgotPasswordRequest):
    users = _load_users()
    email = req.email.strip().lower()
    if email not in users:
        raise HTTPException(404, "No account found with that email")
    return {"success": True}


@router.post("/reset-password")
async def reset_password(req: ResetPasswordRequest):
    if req.code.strip() != "555":
        raise HTTPException(400, "Invalid verification code")
    if len(req.new_password) < 8:
        raise HTTPException(400, "Password must be at least 8 characters")
    if len(req.new_password) > 128:
        raise HTTPException(400, "Password must be no more than 128 characters")
    users = _load_users()
    email = req.email.strip().lower()
    if email not in users:
        raise HTTPException(404, "No account found with that email")
    users[email]["password"] = _hash_password(req.new_password)
    _save_users(users)
    return {"success": True}


@router.put("/profile")
async def update_profile(req: ProfileUpdateRequest, payload=Depends(verify_token)):
    users = _load_users()
    email = payload["sub"]
    u = users.get(email)
    if not u:
        raise HTTPException(404, "User not found")
    if req.store_name is not None:
        u["store_name"] = req.store_name.strip()
    if req.mall is not None:
        u["mall"] = req.mall.strip()
    if req.phone is not None:
        u["phone"] = req.phone.strip()
    _save_users(users)
    return {
        "success":    True,
        "store_name": u["store_name"],
        "mall":       u["mall"],
        "phone":      u["phone"],
    }
