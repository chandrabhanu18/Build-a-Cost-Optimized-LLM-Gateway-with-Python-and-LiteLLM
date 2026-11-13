import logging
import time
from typing import Any, Dict, List, Optional
import litellm
from litellm import acompletion
from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from pydantic import BaseModel
from app.config import GatewayConfig

logger = logging.getLogger("llm_gateway.orchestrator")


class OrchestratorResult(BaseModel):
    response_text: str
    model_used: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_ms: float
    was_fallback: bool = False
    original_model: Optional[str] = None


class GatewayOrchestrator:
    """Manages LLM API completions using LiteLLM, handling dynamic fallback execution

    and precise per-request cost calculation.
    """

    RETRYABLE_EXCEPTIONS = (
        RateLimitError,
        APIConnectionError,
        ServiceUnavailableError,
        Timeout,
        InternalServerError,
    )

    def __init__(self, config: GatewayConfig):
        self.config = config

    def calculate_cost(self, response: Any, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate USD cost for a completion response using LiteLLM's pricing engine

        with a conservative fallback heuristic.
        """
        try:
            cost = litellm.completion_cost(completion_response=response)
            if cost is not None and cost >= 0.0:
                return float(cost)
        except Exception as e:
            logger.debug(f"litellm.completion_cost calculation failed for model {model}: {e}")

        # Conservative fallback pricing table (per 1K tokens)
        pricing_fallback = {
            "gpt-4o": {"input": 0.005, "output": 0.015},
            "claude-3-5-sonnet-20240620": {"input": 0.003, "output": 0.015},
            "groq/llama3-8b-8192": {"input": 0.00005, "output": 0.00008},
            "groq/llama-3.1-8b-instant": {"input": 0.00005, "output": 0.00008},
            "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
        }

        rates = pricing_fallback.get(model, {"input": 0.001, "output": 0.002})
        input_cost = (prompt_tokens / 1000.0) * rates["input"]
        output_cost = (completion_tokens / 1000.0) * rates["output"]
        return round(input_cost + output_cost, 8)

    async def execute_llm_call(
        self,
        model_name: str,
        prompt: str,
        max_tokens: int = 500,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrchestratorResult:
        """Execute a completion call to LiteLLM with automatic fallback on retryable errors.

        Args:
            model_name: The primary model identifier chosen by the router.
            prompt: The preprocessed prompt text.
            max_tokens: Maximum tokens for the generation.
            metadata: Optional telemetry metadata.

        Returns:
            OrchestratorResult containing response text, token usage, latency, and cost.
        """
        start_time = time.perf_counter()
        target_model = model_name
        is_fallback = False

        messages: List[Dict[str, str]] = [{"role": "user", "content": prompt}]

        try:
            response = await acompletion(
                model=target_model,
                messages=messages,
                max_tokens=max_tokens,
                metadata=metadata or {},
            )
        except self.RETRYABLE_EXCEPTIONS as exc:
            backup_model = self.config.fallbacks.get(target_model)
            if backup_model:
                logger.warning(
                    f"Primary model '{target_model}' failed with {type(exc).__name__}: {exc}. "
                    f"Retrying request with configured backup model '{backup_model}'."
                )
                target_model = backup_model
                is_fallback = True
                response = await acompletion(
                    model=target_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    metadata=metadata or {},
                )
            else:
                logger.error(f"Execution failed for model '{target_model}' and no fallback was configured: {exc}")
                raise exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract text response safely
        response_text = ""
        if hasattr(response, "choices") and response.choices:
            first_choice = response.choices[0]
            if hasattr(first_choice, "message") and hasattr(first_choice.message, "content"):
                response_text = first_choice.message.content or ""
            elif isinstance(first_choice, dict) and "message" in first_choice:
                response_text = first_choice["message"].get("content", "")

        # Extract token usage safely
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
        completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
        total_tokens = getattr(usage, "total_tokens", prompt_tokens + completion_tokens) if usage else (prompt_tokens + completion_tokens)

        # Calculate USD cost
        cost_usd = self.calculate_cost(response, target_model, prompt_tokens, completion_tokens)

        return OrchestratorResult(
            response_text=response_text,
            model_used=target_model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            latency_ms=round(latency_ms, 2),
            was_fallback=is_fallback,
            original_model=model_name if is_fallback else None,
        )
