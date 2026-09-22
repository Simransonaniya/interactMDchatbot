"""
Authentication API Router.
POST /api/v1/auth/register
POST /api/v1/auth/login
GET  /api/v1/auth/me
"""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from database import get_db
from models.user import User
from schemas import UserCreate, UserLogin, UserResponse, Token
from services.auth_service import (
    get_user_by_email,
    create_user,
    verify_password,
    create_access_token
)
from api.deps import get_current_user

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])

@router.post("/register", response_model=Token, status_code=status.HTTP_201_CREATED)
def register(user_in: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account with first_name, last_name, email, password."""
    existing_user = get_user_by_email(db, user_in.email)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists."
        )
    
    user = create_user(
        db,
        email=user_in.email,
        password=user_in.password,
        first_name=user_in.first_name,
        last_name=user_in.last_name,
        role=user_in.role or "LEARNER"
    )

    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    return Token(access_token=access_token, token_type="bearer", user=UserResponse.model_validate(user))

@router.post("/login", response_model=Token)
def login(login_in: UserLogin, db: Session = Depends(get_db)):
    """Authenticate user with JSON credentials and return JWT."""
    user = get_user_by_email(db, login_in.email)
    if not user or not verify_password(login_in.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.id, "email": user.email})
    return Token(access_token=access_token, token_type="bearer", user=UserResponse.model_validate(user))

@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Get the currently logged-in user profile without sensitive fields."""
    return UserResponse.model_validate(current_user)
