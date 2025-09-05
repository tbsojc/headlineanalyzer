from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

DATABASE_URL = "sqlite:///app/db.sqlite3"

class Base(DeclarativeBase):
    pass

# Für SQLite + FastAPI Threading:
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False},
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# FastAPI-Dependency:
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
