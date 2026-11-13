from datetime import datetime, timezone
import uuid
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class RequestLog(Base):
    """SQLAlchemy model representing telemetry and cost tracking for a gateway request."""

    __tablename__ = "request_logs"

    id = Column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        index=True,
        doc="Unique UUID primary key for the request log entry",
    )
    timestamp = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        index=True,
        nullable=False,
        doc="UTC timestamp when the request was executed",
    )
    user_id = Column(
        String(255),
        index=True,
        nullable=False,
        doc="Identifier for the user or client application making the request",
    )
    model_used = Column(
        String(255),
        nullable=False,
        doc="Name of the model that served the request (or cached model)",
    )
    prompt_tokens = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Count of tokens in the prompt",
    )
    completion_tokens = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Count of tokens in the completion",
    )
    total_tokens = Column(
        Integer,
        default=0,
        nullable=False,
        doc="Total tokens consumed (prompt + completion)",
    )
    latency_ms = Column(
        Float,
        nullable=False,
        doc="End-to-end processing latency in milliseconds",
    )
    cost_usd = Column(
        Float,
        nullable=False,
        doc="Total calculated transaction cost in USD",
    )
    cache_hit = Column(
        Boolean,
        default=False,
        nullable=False,
        doc="Flag indicating whether response was served from semantic cache",
    )

    def to_dict(self):
        """Convert database record to a clean dictionary representation."""
        return {
            "id": str(self.id),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "user_id": self.user_id,
            "model_used": self.model_used,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": round(float(self.latency_ms), 2),
            "cost_usd": round(float(self.cost_usd), 8),
            "cache_hit": self.cache_hit,
        }

