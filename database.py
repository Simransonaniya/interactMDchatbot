"""
InteractMD — SQLAlchemy 2.x Database Engine & Session Factory.
Uses PostgreSQL as primary single source of truth.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from config import settings

def get_engine():
    db_url = settings.DATABASE_URL
    
    if db_url.startswith("postgresql"):
        # Try psycopg (psycopg3) first, then psycopg2 if driver is missing
        candidate_urls = []
        if "+psycopg" in db_url:
            candidate_urls.append(db_url)
            candidate_urls.append(db_url.replace("+psycopg", "+psycopg2", 1))
        elif "+psycopg2" in db_url:
            candidate_urls.append(db_url)
            candidate_urls.append(db_url.replace("+psycopg2", "+psycopg", 1))
        elif db_url.startswith("postgresql://"):
            candidate_urls.append(db_url.replace("postgresql://", "postgresql+psycopg2://", 1))
            candidate_urls.append(db_url.replace("postgresql://", "postgresql+psycopg://", 1))
        else:
            candidate_urls.append(db_url)

        last_error = None
        for url in candidate_urls:
            try:
                eng = create_engine(
                    url,
                    pool_pre_ping=True,
                    pool_size=10,
                    max_overflow=20,
                    echo=False,
                )
                return eng
            except Exception as e:
                last_error = e

        print(f"[Database Warning] Could not connect with PostgreSQL drivers ({last_error}). Falling back to SQLite local database.")
        return create_engine("sqlite:///./interactmd.db", connect_args={"check_same_thread": False})
    
    return create_engine(
        db_url,
        connect_args={"check_same_thread": False} if "sqlite" in db_url else {},
        echo=False
    )

engine = get_engine()

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

class Base(DeclarativeBase):
    pass

def get_db():
    """
    FastAPI dependency that provides a DB session per request.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
