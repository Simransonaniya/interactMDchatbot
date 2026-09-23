"""
Authentication & Authorization Service.
Handles password hashing and verification with bcrypt, JWT token generation & parsing.
Uses MongoDB Atlas as primary authoritative user store with SQLite resilience.
"""

import time
import uuid
import bcrypt
from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from config import settings
from models.user import User
from mongo_db import mongo_manager


class UserAdapter:
    """Wrapper class providing standard ORM attributes for MongoDB user documents."""
    def __init__(self, d: Dict[str, Any]):
        self.id = d.get("id") or str(d.get("_id", ""))
        self.email = d.get("email", "")
        self.password_hash = d.get("password_hash", "")
        self.first_name = d.get("first_name") or d.get("firstName") or (self.email.split("@")[0] if self.email else "User")
        self.last_name = d.get("last_name") or d.get("lastName") or "Learner"
        self.role = d.get("role", "LEARNER")
        self.is_active = d.get("is_active", True)
        
        created = d.get("created_at")
        if isinstance(created, (int, float)):
            self.created_at = datetime.fromtimestamp(created, tz=timezone.utc)
        elif isinstance(created, datetime):
            self.created_at = created
        else:
            self.created_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"<UserAdapter {self.email} ({self.role})>"


def _doc_to_user(doc: Optional[Dict[str, Any]]) -> Optional[UserAdapter]:
    if not doc:
        return None
    return UserAdapter(doc)


def get_password_hash(password: str) -> str:
    """Hash password using bcrypt."""
    pwd_bytes = password.encode('utf-8')
    if len(pwd_bytes) > 72:
        pwd_bytes = pwd_bytes[:72]
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(pwd_bytes, salt).decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify plain password against bcrypt hash, with plain-text fallback."""
    if not hashed_password or not plain_password:
        return False
    if plain_password == hashed_password:
        return True
    pwd_bytes = plain_password.encode('utf-8')
    if len(pwd_bytes) > 72:
        pwd_bytes = pwd_bytes[:72]
    try:
        hash_bytes = hashed_password.encode('utf-8')
        return bcrypt.checkpw(pwd_bytes, hash_bytes)
    except Exception:
        return False


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)
    return encoded_jwt


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        return payload
    except JWTError:
        return None


def get_user_by_email(db: Session, email: str) -> Optional[Any]:
    clean_email = email.lower().strip()
    
    # 1. MongoDB Atlas (Source of truth)
    if mongo_manager.is_connected:
        user_doc = mongo_manager.get_user_by_email(clean_email)
        if user_doc:
            return _doc_to_user(user_doc)

    # 2. SQLite fallback (resilient to schema variances)
    try:
        return db.query(User).filter(User.email == clean_email).first()
    except Exception as e:
        print(f"[Auth Warning] SQLite lookup failed: {e}")
        return None


def get_user_by_id(db: Session, user_id: str) -> Optional[Any]:
    # 1. MongoDB Atlas
    if mongo_manager.is_connected:
        user_doc = mongo_manager.get_user_by_id(user_id)
        if user_doc:
            return _doc_to_user(user_doc)

    # 2. SQLite fallback
    try:
        return db.query(User).filter(User.id == user_id).first()
    except Exception as e:
        print(f"[Auth Warning] SQLite lookup failed: {e}")
        return None


def create_user(
    db: Session,
    email: str,
    password: str,
    first_name: str,
    last_name: str,
    role: str = "LEARNER"
) -> Any:
    hashed_pwd = get_password_hash(password)
    clean_email = email.lower().strip()
    clean_first = first_name.strip() if first_name else clean_email.split("@")[0]
    clean_last = last_name.strip() if last_name else "Learner"

    existing_user = get_user_by_email(db, clean_email)
    user_id = existing_user.id if existing_user else f"user_{uuid.uuid4().hex[:12]}"

    user_dict = {
        "id": user_id,
        "email": clean_email,
        "password_hash": hashed_pwd,
        "first_name": clean_first,
        "last_name": clean_last,
        "role": role,
        "is_active": True,
        "created_at": time.time()
    }

    # 1. Save in MongoDB Atlas
    if mongo_manager.is_connected:
        mongo_manager.save_user(user_dict)

    # 2. Save or update in SQLite
    try:
        db_user = db.query(User).filter(User.email == clean_email).first()
        if db_user:
            db_user.password_hash = hashed_pwd
            db_user.first_name = clean_first
            db_user.last_name = clean_last
            db_user.role = role
        else:
            db_user = User(
                id=user_id,
                email=clean_email,
                password_hash=hashed_pwd,
                first_name=clean_first,
                last_name=clean_last,
                role=role,
                is_active=True
            )
            db.add(db_user)
        db.commit()
        db.refresh(db_user)
        return db_user
    except Exception as e:
        print(f"[Auth Warning] SQLite save failed: {e}")
        try:
            db.rollback()
        except Exception:
            pass
        return _doc_to_user(user_dict)

