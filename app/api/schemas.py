from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class GatewayRequest(BaseModel):
    """Schema for incoming LLM completion requests."""

    prompt: str = Field(..., description="The user's input prompt")
    user_id: str = Field(..., description="Identifier for the user making the request")
    max_tokens: Optional[int] = Field(500, description="Maximum number of tokens to generate")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional request metadata")


class GatewayResponse(BaseModel):
    """Standardized response schema for gateway completions."""

    response: str = Field(..., description="Generated text response from LLM or semantic cache")
    model_used: str = Field(..., description="The model tier/name that produced the response")
    cache_hit: bool = Field(..., description="Indicates whether response was retrieved from cache")
    cost_usd: float = Field(..., description="Total USD cost incurred for the transaction")
    request_id: Optional[str] = Field(None, description="Unique ID for telemetry lookup")
    latency_ms: Optional[float] = Field(None, description="Processing latency in milliseconds")
    total_tokens: Optional[int] = Field(None, description="Total tokens consumed")


class RequestLogResponse(BaseModel):
    """Schema representing the database row for a specific request log."""

    id: str
    timestamp: Optional[str] = None
    user_id: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    cache_hit: bool


class AnalyticsSummaryResponse(BaseModel):
    """Schema for aggregate telemetry analytics summary."""

    total_cost_usd: float = Field(..., description="Cumulative cost of all recorded requests in USD")
    total_requests: int = Field(..., description="Total number of requests executed")
    average_latency_ms: float = Field(..., description="Average processing latency in milliseconds")
    cache_hit_rate_percentage: float = Field(..., description="Percentage of requests served from cache")


class HealthStatusResponse(BaseModel):
    """System health check response schema."""

    status: str
    database: str
    redis: str
    version: str = "1.0.0"

