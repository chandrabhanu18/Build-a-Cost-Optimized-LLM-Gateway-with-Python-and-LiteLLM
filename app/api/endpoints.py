from datetime import datetime, timezone
import logging
import time
import uuid
from typing import Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.api.schemas import (
    AnalyticsSummaryResponse,
    GatewayRequest,
    GatewayResponse,
    HealthStatusResponse,
    RequestLogResponse,
)
from app.config import gateway_config, settings
from app.core.cache import SemanticCacheManager
from app.core.orchestrator import GatewayOrchestrator
from app.core.preprocessing import clean_prompt
from app.core.router import IntelligentRouter
from app.db.database import get_db, save_request_log_sync
from app.db.models import RequestLog

logger = logging.getLogger("llm_gateway.api")

router = APIRouter(prefix="/v1", tags=["LLM Gateway"])

# Core engine instances
intelligent_router = IntelligentRouter(gateway_config)
gateway_orchestrator = GatewayOrchestrator(gateway_config)
cache_manager = SemanticCacheManager(gateway_config, redis_url=settings.REDIS_URL)


@router.post(
    "/completions",
    response_model=GatewayResponse,
    status_code=status.HTTP_200_OK,
    summary="Submit a prompt to the cost-optimized LLM gateway",
    description="Cleans prompt, performs semantic cache lookup, dynamically routes to optimal model tier, handles fallbacks, and tracks cost.",
)
async def create_completion(
    request: GatewayRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    start_time = time.perf_counter()
    request_id = str(uuid.uuid4())

    # 1. Prompt Preprocessing
    cleaned_prompt = clean_prompt(request.prompt)
    if not cleaned_prompt:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Prompt cannot be empty after preprocessing.",
        )

    # 2. Semantic Caching Check
    cache_result = await cache_manager.check_cache(cleaned_prompt)

    if cache_result.hit:
        elapsed_latency = (time.perf_counter() - start_time) * 1000.0
        cost_usd = round(cache_result.embedding_cost, 8)
        prompt_tokens = cache_result.embedding_tokens
        completion_tokens = 0
        total_tokens = prompt_tokens

        # Asynchronously log telemetry to database
        background_tasks.add_task(
            save_request_log_sync,
            request_id=request_id,
            user_id=request.user_id,
            model_used=cache_result.model_used or "cached",
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=round(elapsed_latency, 2),
            cost_usd=cost_usd,
            cache_hit=True,
        )

        return GatewayResponse(
            response=cache_result.response_text or "",
            model_used=cache_result.model_used or "cached",
            cache_hit=True,
            cost_usd=cost_usd,
            request_id=request_id,
            latency_ms=round(elapsed_latency, 2),
            total_tokens=total_tokens,
        )

    # 3. Dynamic Intelligent Routing
    routing_decision = intelligent_router.evaluate(cleaned_prompt)
    target_model = routing_decision.model_name
    logger.info(
        f"Prompt routed to '{target_model}' (tier: {routing_decision.tier_name}, rule: {routing_decision.rule_name})"
    )

    # 4. LiteLLM Orchestration with Fallback Retries
    try:
        orch_result = await gateway_orchestrator.execute_llm_call(
            model_name=target_model,
            prompt=cleaned_prompt,
            max_tokens=request.max_tokens or 500,
            metadata=request.metadata,
        )
    except Exception as exc:
        logger.error(f"Execution failed for all candidate models: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Downstream LLM execution failed: {str(exc)}",
        )

    elapsed_latency = (time.perf_counter() - start_time) * 1000.0
    # Total cost includes completion cost plus embedding evaluation cost
    total_cost_usd = round(orch_result.cost_usd + cache_result.embedding_cost, 8)

    # 5. Background Tasks: Store in Semantic Cache & Persist Telemetry Log
    background_tasks.add_task(
        cache_manager.store_in_cache,
        prompt=cleaned_prompt,
        response_text=orch_result.response_text,
        model_used=orch_result.model_used,
        embedding_vector=cache_result.embedding_vector,
    )

    background_tasks.add_task(
        save_request_log_sync,
        request_id=request_id,
        user_id=request.user_id,
        model_used=orch_result.model_used,
        prompt_tokens=orch_result.prompt_tokens,
        completion_tokens=orch_result.completion_tokens,
        total_tokens=orch_result.total_tokens,
        latency_ms=round(elapsed_latency, 2),
        cost_usd=total_cost_usd,
        cache_hit=False,
    )

    return GatewayResponse(
        response=orch_result.response_text,
        model_used=orch_result.model_used,
        cache_hit=False,
        cost_usd=total_cost_usd,
        request_id=request_id,
        latency_ms=round(elapsed_latency, 2),
        total_tokens=orch_result.total_tokens,
    )


@router.get(
    "/analytics/requests/{request_id}",
    response_model=RequestLogResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve telemetry details for a specific request",
)
def get_request_analytics(request_id: str, db: Session = Depends(get_db)):
    """Retrieve the exact database record for an individual request by UUID."""
    log_entry = db.query(RequestLog).filter(RequestLog.id == request_id).first()
    if not log_entry:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Request telemetry record '{request_id}' not found.",
        )
    return RequestLogResponse(**log_entry.to_dict())


@router.get(
    "/analytics/summary",
    response_model=AnalyticsSummaryResponse,
    status_code=status.HTTP_200_OK,
    summary="Get aggregated usage, cost, and latency analytics",
)
def get_analytics_summary(
    start_date: Optional[datetime] = Query(None, description="Start timestamp (ISO format)"),
    end_date: Optional[datetime] = Query(None, description="End timestamp (ISO format)"),
    db: Session = Depends(get_db),
):
    """Aggregate total cost, total requests, average latency, and cache hit rate percentage

    dynamically computed from PostgreSQL request logs.
    """
    query = db.query(RequestLog)
    if start_date:
        query = query.filter(RequestLog.timestamp >= start_date)
    if end_date:
        query = query.filter(RequestLog.timestamp <= end_date)

    total_requests = query.count()
    if total_requests == 0:
        return AnalyticsSummaryResponse(
            total_cost_usd=0.0,
            total_requests=0,
            average_latency_ms=0.0,
            cache_hit_rate_percentage=0.0,
        )

    # Calculate metrics with aggregate SQL functions
    stats = query.with_entities(
        func.sum(RequestLog.cost_usd).label("total_cost"),
        func.avg(RequestLog.latency_ms).label("avg_latency"),
    ).first()

    cache_hits = query.filter(RequestLog.cache_hit == True).count()

    total_cost = float(stats.total_cost) if stats and stats.total_cost else 0.0
    avg_latency = float(stats.avg_latency) if stats and stats.avg_latency else 0.0
    hit_rate = (cache_hits / total_requests) * 100.0

    return AnalyticsSummaryResponse(
        total_cost_usd=round(total_cost, 6),
        total_requests=total_requests,
        average_latency_ms=round(avg_latency, 2),
        cache_hit_rate_percentage=round(hit_rate, 2),
    )


@router.get(
    "/health",
    response_model=HealthStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Health check for gateway, database, and Redis",
)
async def health_check(db: Session = Depends(get_db)):
    """Health status check endpoint."""
    db_status = "healthy"
    try:
        db.execute(func.now())
    except Exception as e:
        db_status = f"unhealthy: {e}"

    redis_status = "healthy"
    try:
        client = await cache_manager.get_redis_client()
        if client:
            await client.ping()
        else:
            redis_status = "degraded (in-memory fallback active)"
    except Exception as e:
        redis_status = f"unhealthy: {e}"

    return HealthStatusResponse(
        status="healthy" if db_status == "healthy" else "degraded",
        database=db_status,
        redis=redis_status,
        version="1.0.0",
    )

