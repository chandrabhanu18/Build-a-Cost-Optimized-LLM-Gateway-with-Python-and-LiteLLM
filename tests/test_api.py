from unittest.mock import AsyncMock, MagicMock, patch
import litellm
from litellm.exceptions import RateLimitError
import pytest
from app.db.models import RequestLog


class TestGatewayAPI:
    """Integration tests verifying REST API contracts, caching lifecycle, fallbacks, and telemetry analytics."""

    def test_health_check_endpoint(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "database" in data
        assert "redis" in data

    def test_root_metadata_endpoint(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["service"] == "Cost-Optimized LLM Gateway"
        assert "completions_endpoint" in data

    def test_completions_cache_miss_then_hit(self, client, db_session):
        prompt = "Explain why the sky is blue"
        user_id = "user-test-123"

        mock_choice = MagicMock()
        mock_choice.message.content = "The sky appears blue due to Rayleigh scattering."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 15
        mock_usage.completion_tokens = 25
        mock_usage.total_tokens = 40

        mock_llm_resp = MagicMock()
        mock_llm_resp.choices = [mock_choice]
        mock_llm_resp.usage = mock_usage

        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("litellm.completion_cost", return_value=0.00005):
            mock_acompletion.return_value = mock_llm_resp

            # --- Request 1: Cache Miss ---
            req1 = client.post(
                "/v1/completions",
                json={"prompt": prompt, "user_id": user_id, "max_tokens": 100},
            )
            assert req1.status_code == 200
            data1 = req1.json()

            # Contract 2 validation
            assert "response" in data1
            assert "model_used" in data1
            assert "cache_hit" in data1
            assert "cost_usd" in data1

            assert data1["cache_hit"] is False
            assert data1["model_used"] == "groq/llama3-8b-8192"
            assert data1["response"] == "The sky appears blue due to Rayleigh scattering."
            assert data1["cost_usd"] > 0.0

            # Verify telemetry record inserted into PostgreSQL/SQLite
            request_id_1 = data1["request_id"]
            db_row = db_session.query(RequestLog).filter(RequestLog.id == request_id_1).first()
            assert db_row is not None
            assert db_row.user_id == user_id
            assert db_row.cache_hit is False

            # --- Request 2: Semantic Cache Hit ---
            req2 = client.post(
                "/v1/completions",
                json={"prompt": prompt, "user_id": user_id},
            )
            assert req2.status_code == 200
            data2 = req2.json()

            assert data2["cache_hit"] is True
            assert data2["response"] == "The sky appears blue due to Rayleigh scattering."
            assert data2["model_used"] == "groq/llama3-8b-8192"
            # LLM should not have been called a second time
            assert mock_acompletion.call_count == 1

            # Verify second telemetry record exists with cache_hit = True
            request_id_2 = data2["request_id"]
            db_row_2 = db_session.query(RequestLog).filter(RequestLog.id == request_id_2).first()
            assert db_row_2 is not None
            assert db_row_2.cache_hit is True

    def test_completions_complex_keyword_routes_to_high_tier(self, client):
        prompt = "Please analyze the trade-offs between monolithic and microservice architectures."

        mock_choice = MagicMock()
        mock_choice.message.content = "Here is a detailed architectural analysis."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 30
        mock_usage.completion_tokens = 50
        mock_usage.total_tokens = 80

        mock_llm_resp = MagicMock()
        mock_llm_resp.choices = [mock_choice]
        mock_llm_resp.usage = mock_usage

        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("litellm.completion_cost", return_value=0.001):
            mock_acompletion.return_value = mock_llm_resp

            resp = client.post(
                "/v1/completions",
                json={"prompt": prompt, "user_id": "architect-1"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_used"] == "gpt-4o"
            assert data["cache_hit"] is False

    def test_completions_automatic_fallback_on_rate_limit(self, client):
        prompt = "Synthesize these performance metrics."

        mock_fallback_choice = MagicMock()
        mock_fallback_choice.message.content = "Synthesized metrics from fallback model."
        mock_fallback_usage = MagicMock()
        mock_fallback_usage.prompt_tokens = 25
        mock_fallback_usage.completion_tokens = 35
        mock_fallback_usage.total_tokens = 60

        mock_fallback_resp = MagicMock()
        mock_fallback_resp.choices = [mock_fallback_choice]
        mock_fallback_resp.usage = mock_fallback_usage

        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("litellm.completion_cost", return_value=0.0008):
            # First attempt on gpt-4o fails with RateLimitError; fallback claude succeeds
            mock_acompletion.side_effect = [
                RateLimitError(message="Rate limit reached for gpt-4o", llm_provider="openai", model="gpt-4o"),
                mock_fallback_resp,
            ]

            resp = client.post(
                "/v1/completions",
                json={"prompt": prompt, "user_id": "test-user-fallback"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_used"] == "claude-3-5-sonnet-20240620"
            assert data["response"] == "Synthesized metrics from fallback model."

    def test_analytics_request_lookup_endpoint(self, client, db_session):
        # 1. Non-existent request returns 404
        resp_404 = client.get("/v1/analytics/requests/non-existent-uuid")
        assert resp_404.status_code == 404
        assert "not found" in resp_404.json()["detail"].lower()

        # 2. Add sample record directly to db
        sample_log = RequestLog(
            id="test-req-uuid-999",
            user_id="alice",
            model_used="gpt-4o",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=320.5,
            cost_usd=0.00125,
            cache_hit=False,
        )
        db_session.add(sample_log)
        db_session.commit()

        # 3. Retrieve record by ID
        resp_200 = client.get(f"/v1/analytics/requests/{sample_log.id}")
        assert resp_200.status_code == 200
        data = resp_200.json()
        assert data["id"] == sample_log.id
        assert data["user_id"] == "alice"
        assert data["model_used"] == "gpt-4o"
        assert data["total_tokens"] == 150
        assert data["cost_usd"] == 0.00125
        assert data["cache_hit"] is False

    def test_analytics_summary_endpoint(self, client, db_session):
        # Insert known test logs
        logs = [
            RequestLog(
                id="sum-1", user_id="u1", model_used="gpt-4o",
                prompt_tokens=10, completion_tokens=10, total_tokens=20,
                latency_ms=200.0, cost_usd=0.005, cache_hit=False
            ),
            RequestLog(
                id="sum-2", user_id="u1", model_used="gpt-4o",
                prompt_tokens=10, completion_tokens=10, total_tokens=20,
                latency_ms=10.0, cost_usd=0.00001, cache_hit=True
            ),
            RequestLog(
                id="sum-3", user_id="u2", model_used="groq/llama3-8b-8192",
                prompt_tokens=5, completion_tokens=5, total_tokens=10,
                latency_ms=100.0, cost_usd=0.0001, cache_hit=False
            ),
            RequestLog(
                id="sum-4", user_id="u2", model_used="groq/llama3-8b-8192",
                prompt_tokens=5, completion_tokens=5, total_tokens=10,
                latency_ms=12.0, cost_usd=0.00001, cache_hit=True
            ),
        ]
        for l in logs:
            db_session.add(l)
        db_session.commit()

        resp = client.get("/v1/analytics/summary")
        assert resp.status_code == 200
        summary = resp.json()

        assert summary["total_requests"] == 4
        # 2 cache hits out of 4 requests = 50.0%
        assert summary["cache_hit_rate_percentage"] == 50.0
        # total cost = 0.005 + 0.00001 + 0.0001 + 0.00001 = 0.00512
        assert abs(summary["total_cost_usd"] - 0.00512) < 0.00001
        # avg latency = (200 + 10 + 100 + 12) / 4 = 80.5
        assert abs(summary["average_latency_ms"] - 80.5) < 0.1

