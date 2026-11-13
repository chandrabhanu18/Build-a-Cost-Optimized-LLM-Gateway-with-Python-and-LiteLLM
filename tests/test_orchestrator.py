from unittest.mock import AsyncMock, MagicMock, patch
import litellm
from litellm.exceptions import APIConnectionError, RateLimitError
import pytest
from app.core.orchestrator import GatewayOrchestrator


class TestGatewayOrchestrator:
    """Unit tests verifying LiteLLM execution, fallback retry mechanism, and cost tracking (Contract 5 & 6)."""

    @pytest.mark.asyncio
    async def test_successful_llm_execution(self, sample_config):
        orchestrator = GatewayOrchestrator(sample_config)

        mock_choice = MagicMock()
        mock_choice.message.content = "Analysis of the data shows a 15% increase."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 30
        mock_usage.total_tokens = 80

        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_response.usage = mock_usage

        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("litellm.completion_cost", return_value=0.00045):
            mock_acompletion.return_value = mock_response

            result = await orchestrator.execute_llm_call(
                model_name="gpt-4o",
                prompt="Analyze this data",
                max_tokens=200,
            )

            assert result.response_text == "Analysis of the data shows a 15% increase."
            assert result.model_used == "gpt-4o"
            assert result.prompt_tokens == 50
            assert result.completion_tokens == 30
            assert result.total_tokens == 80
            assert result.cost_usd == 0.00045
            assert result.was_fallback is False
            mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_automatic_fallback_on_rate_limit_error(self, sample_config):
        orchestrator = GatewayOrchestrator(sample_config)

        mock_choice = MagicMock()
        mock_choice.message.content = "Fallback response from backup model."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 40
        mock_usage.completion_tokens = 20
        mock_usage.total_tokens = 60

        mock_fallback_response = MagicMock()
        mock_fallback_response.choices = [mock_choice]
        mock_fallback_response.usage = mock_usage

        # First call raises RateLimitError; second call (backup model) succeeds
        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("litellm.completion_cost", return_value=0.0003):
            mock_acompletion.side_effect = [
                RateLimitError(message="Rate limit exceeded", llm_provider="openai", model="gpt-4o"),
                mock_fallback_response,
            ]

            result = await orchestrator.execute_llm_call(
                model_name="gpt-4o",
                prompt="Analyze this text",
            )

            # Fallback for gpt-4o is claude-3-5-sonnet-20240620 in config
            assert result.was_fallback is True
            assert result.model_used == "claude-3-5-sonnet-20240620"
            assert result.original_model == "gpt-4o"
            assert result.response_text == "Fallback response from backup model."
            assert mock_acompletion.call_count == 2

    @pytest.mark.asyncio
    async def test_automatic_fallback_on_api_connection_error(self, sample_config):
        orchestrator = GatewayOrchestrator(sample_config)

        mock_choice = MagicMock()
        mock_choice.message.content = "Connected via backup."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 20
        mock_usage.completion_tokens = 10
        mock_usage.total_tokens = 30

        mock_fallback_response = MagicMock()
        mock_fallback_response.choices = [mock_choice]
        mock_fallback_response.usage = mock_usage

        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("litellm.completion_cost", return_value=0.0001):
            mock_acompletion.side_effect = [
                APIConnectionError(message="Connection timed out", llm_provider="anthropic", model="claude-3-5-sonnet-20240620"),
                mock_fallback_response,
            ]

            result = await orchestrator.execute_llm_call(
                model_name="claude-3-5-sonnet-20240620",
                prompt="Explain photosynthesis",
            )

            # Backup for claude-3-5-sonnet-20240620 is groq/llama3-8b-8192
            assert result.was_fallback is True
            assert result.model_used == "groq/llama3-8b-8192"
            assert mock_acompletion.call_count == 2

