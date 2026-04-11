"""
conftest.py — Shared pytest fixtures.

Overrides the real PostgreSQL database with an in-memory SQLite DB
before any part of app.py or database.py is imported.
This lets CI run without a live database or .env file.
"""

import os
import pytest

# ── Must happen before `app` or `database` is imported ──────────────
os.environ.setdefault("DATABASE_URL", "sqlite:///./test.db")

from sqlalchemy import create_engine                  # noqa: E402
from sqlalchemy.orm import sessionmaker               # noqa: E402
from fastapi.testclient import TestClient             # noqa: E402

import database                                       # noqa: E402
from database import Base, get_db                    # noqa: E402
from app import app                                   # noqa: E402


SQLALCHEMY_TEST_URL = "sqlite:///./test.db"

engine = create_engine(
    SQLALCHEMY_TEST_URL, connect_args={"check_same_thread": False}
)
TestSessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


def override_get_db():
    db = TestSessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create tables once for the entire test session."""
    # Patch database module so tables are created on the test engine
    database.engine = engine
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(setup_test_db):
    """FastAPI TestClient with DB dependency overridden."""
    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()
