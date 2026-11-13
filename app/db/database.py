import logging
import time
from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from app.config import settings
from app.db.models import Base, RequestLog

logger = logging.getLogger("llm_gateway.database")

# Build engine with connection pooling
connect_args = {}
if settings.DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.DATABASE_URL,
    pool_pre_ping=True,
    echo=False,
    connect_args=connect_args,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db(max_retries: int = 10, retry_delay: float = 2.0) -> None:
    """Initialize database tables with connection retry logic.

    Prevents Docker startup race conditions where the web app boots
    before PostgreSQL is ready to accept connections.
    """
    for attempt in range(1, max_retries + 1):
        try:
            logger.info(f"Connecting to database (attempt {attempt}/{max_retries})...")
            with engine.connect() as conn:
                Base.metadata.create_all(bind=engine)
            logger.info("Database schema initialized successfully.")
            return
        except Exception as e:
            if attempt == max_retries:
                logger.error(f"Failed to connect to database after {max_retries} attempts: {e}")
                raise e
            logger.warning(f"Database connection attempt {attempt} failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def save_request_log_sync(
    request_id: str,
    user_id: str,
    model_used: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    latency_ms: float,
    cost_usd: float,
    cache_hit: bool,
) -> None:
    """Synchronous helper invoked by FastAPI background tasks to record telemetry."""
    db: Session = SessionLocal()
    try:
        log_entry = RequestLog(
            id=request_id,
            user_id=user_id,
            model_used=model_used,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            cache_hit=cache_hit,
        )
        db.add(log_entry)
        db.commit()
        logger.debug(f"Successfully logged request telemetry for ID {request_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to persist telemetry log for request {request_id}: {e}")
    finally:
        db.close()

