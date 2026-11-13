import os
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Set test environment variables before importing app
os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["CONFIG_PATH"] = "config.yaml"

import app.db.database as db_module
from app.api.endpoints import cache_manager
from app.config import GatewayConfig, load_gateway_config
from app.db.database import get_db
from app.db.models import Base
from app.main import app

# Create in-memory SQLite engine for tests with StaticPool
test_engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)

# Point database module to the testing engine and sessionmaker
db_module.engine = test_engine
db_module.SessionLocal = TestingSessionLocal


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all database tables in the in-memory SQLite database."""
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture(autouse=True)
def clean_db_and_cache():
    """Reset database tables and in-memory cache before every test."""
    Base.metadata.drop_all(bind=test_engine)
    Base.metadata.create_all(bind=test_engine)
    cache_manager._memory_cache.clear()
    cache_manager._redis_available = False
    yield


@pytest.fixture
def db_session():
    """Provide a clean, isolated database session for a test function."""
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    """Provide a FastAPI TestClient with the database dependency overridden."""
    def override_get_db():
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()


@pytest.fixture
def sample_config() -> GatewayConfig:
    """Provide loaded gateway configuration."""
    return load_gateway_config("config.yaml")

