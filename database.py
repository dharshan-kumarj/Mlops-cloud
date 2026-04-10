"""
Database configuration and models for the spam classifier API.
"""

import os
from datetime import datetime

from dotenv import load_dotenv
from sqlalchemy import Column, DateTime, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """Stores every prediction the API makes."""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    input_text = Column(String, nullable=False)
    cleaned_text = Column(String, nullable=False)
    prediction = Column(String, nullable=False)       # "spam" or "ham"
    confidence = Column(Float, nullable=False)
    spam_probability = Column(Float, nullable=False)
    ham_probability = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


def init_db():
    """Create all tables if they don't exist."""
    Base.metadata.create_all(bind=engine)


def get_db():
    """FastAPI dependency — yields a DB session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
