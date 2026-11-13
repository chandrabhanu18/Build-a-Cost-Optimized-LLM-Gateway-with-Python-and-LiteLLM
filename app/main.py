from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.endpoints import router as api_router, cache_manager
from app.config import settings, gateway_config
from app.db.database import init_db

# Configure structured logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("llm_gateway")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager handling startup initialization and shutdown cleanup."""
    logger.info("Initializing Cost-Optimized LLM Gateway...")
    logger.info(f"Loaded Models - High: {gateway_config.models.high_tier}, "
                f"Mid: {gateway_config.models.mid_tier}, Low: {gateway_config.models.low_tier}")

    # Initialize relational database tables
    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization warning: {e}")

    # Initialize Redis vector search index
    try:
        await cache_manager.initialize_index()
    except Exception as e:
        logger.warning(f"Semantic Cache vector index initialization warning: {e}")

    logger.info("LLM Gateway startup complete and ready to accept requests.")
    yield

    logger.info("Shutting down LLM Gateway...")
    if cache_manager.redis_client:
        try:
            await cache_manager.redis_client.aclose()
        except Exception:
            pass
    logger.info("Gateway shutdown complete.")


app = FastAPI(
    title="Cost-Optimized LLM Gateway",
    description=(
        "Production-ready LLM Gateway that intelligently routes prompts based on complexity, "
        "implements semantic caching via Redis vector search to eliminate redundant calls, "
        "and tracks granular per-request token costs in PostgreSQL."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable Cross-Origin Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler returning structured error responses."""
    logger.error(f"Unhandled exception processing {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred processing your request.",
            "detail": str(exc),
        },
    )


# Mount API routers
app.include_router(api_router)


@app.get("/", tags=["Root"])
def root():
    """Root metadata endpoint."""
    return {
        "service": "Cost-Optimized LLM Gateway",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "completions_endpoint": "POST /v1/completions",
        "analytics_endpoint": "GET /v1/analytics/summary",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)

