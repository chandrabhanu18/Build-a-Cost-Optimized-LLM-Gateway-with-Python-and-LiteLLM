"""Simulate Cost Savings: Benchmarks 50 diverse production prompts through the LLM Gateway

comparing total costs against an un-optimized high-tier (e.g. GPT-4o) baseline.
"""

import json
import os
import sys
import time
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np

# Configure PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

os.environ["APP_ENV"] = "testing"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"

from app.api.endpoints import cache_manager
from app.config import gateway_config
from app.core.router import IntelligentRouter
from app.db.database import get_db
from app.db.models import Base, RequestLog
from app.main import app

# 50 Diverse Prompts spanning cacheable repetitions, simple queries, medium queries, and complex tasks
PROMPTS = [
    # --- Repetitive / FAQ queries (designed to hit the semantic cache) ---
    "How do I reset my user account password?",                                 # 1 (Miss)
    "How can I reset my account password?",                                      # 2 (Hit)
    "What is the password reset process for my account?",                        # 3 (Hit)
    "What are your API rate limits for standard tier?",                          # 4 (Miss)
    "Could you tell me the API rate limits on the standard plan?",               # 5 (Hit)
    "What is the refund policy for enterprise customers?",                       # 6 (Miss)
    "Explain the refund policy for enterprise accounts.",                        # 7 (Hit)
    "How do I generate a new API authentication token?",                         # 8 (Miss)
    "What is the procedure to generate a new API token?",                        # 9 (Hit)
    "Where can I download my monthly invoices?",                                 # 10 (Miss)
    "How can I access and download my billing invoices?",                        # 11 (Hit)
    "How do I invite a teammate to my organization workspace?",                  # 12 (Miss)
    "What are the steps to invite a colleague to the team workspace?",           # 13 (Hit)
    "How do I configure Webhook endpoints in the dashboard?",                    # 14 (Miss)
    "Where do I set up webhooks for event notifications?",                       # 15 (Hit)
    "What encryption standards are used for customer data at rest?",             # 16 (Miss)
    "How is customer data encrypted at rest?",                                   # 17 (Hit)
    "Can I upgrade my subscription plan mid-billing cycle?",                     # 18 (Miss)
    "Is it possible to upgrade my plan before the billing month ends?",          # 19 (Hit)
    "How do I delete my user account permanently?",                              # 20 (Miss)

    # --- Simple / Low-Tier Queries (short, factual, extraction) ---
    "What is the capital city of Australia?",                                    # 21
    "Convert 45 miles per hour to kilometers per hour.",                         # 22
    "Extract the email address from: 'Contact sales at support@example.com'.",  # 23
    "What is the atomic number of Gold?",                                        # 24
    "Translate 'Good morning, how are you today?' into German.",                 # 25
    "Who wrote the play Hamlet?",                                                # 26
    "Calculate 254 multiplied by 18.",                                           # 27
    "Give me 5 synonyms for the word 'resilient'.",                              # 28
    "What year was Python first released by Guido van Rossum?",                  # 29
    "Format this date into ISO 8601: November 14th, 2026.",                      # 30
    "What is the boolean truth table for XOR?",                                  # 31
    "Is a tomato botanically a fruit or a vegetable?",                           # 32
    "Convert the JSON string '{\"id\": 101}' to a YAML snippet.",                 # 33
    "Extract all numbers from this sentence: 'Order 9482 has 12 items totaling $499'.", # 34
    "What HTTP status code represents 'Conflict'?",                              # 35

    # --- Medium / Mid-Tier Queries (longer context, summarization, documentation) ---
    "Summarize the core differences between REST and GraphQL APIs for web development.", # 36
    "Draft a professional announcement email informing users about a scheduled 2-hour maintenance window.", # 37
    "Explain the role of database indexing in B-tree structures with pros and cons.", # 38
    "Write a Python function using asyncio to fetch data from three endpoints concurrently.", # 39
    "Explain how Docker container namespaces and cgroups isolate processes in Linux.", # 40
    "Summarize the principles of the Twelve-Factor App methodology for cloud-native software.", # 41
    "Compare horizontal scaling versus vertical scaling in distributed database architectures.", # 42
    "Explain the differences between optimistic and pessimistic concurrency control.", # 43
    "Draft comprehensive documentation guidelines for an internal REST API service.", # 44
    "Explain how TCP three-way handshake works and how SYN cookies mitigate SYN flood attacks.", # 45

    # --- Complex / High-Tier Queries (reasoning, synthesis, architecture, critique) ---
    "Analyze the end-to-end security architecture of a multi-tenant SaaS application on AWS.", # 46
    "Synthesize the trade-offs between Paxos and Raft consensus algorithms in distributed storage systems.", # 47
    "Critique this microservice decomposition plan for an e-commerce platform and recommend improvements.", # 48
    "Architect a real-time event streaming pipeline processing 100,000 transactions per second with low latency.", # 49
    "Benchmark and evaluate the computational complexity of transformer attention mechanisms versus linear state-space models.", # 50
]


def run_simulation():
    print(f"Starting Cost Optimization Simulation with {len(PROMPTS)} diverse prompts...")

    # Setup isolated in-memory DB for simulation
    sim_engine = create_engine("sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=sim_engine)
    SimSession = sessionmaker(autocommit=False, autoflush=False, bind=sim_engine)

    import app.db.database as db_module
    db_module.engine = sim_engine
    db_module.SessionLocal = SimSession

    def override_db():
        session = SimSession()
        try:
            yield session
        finally:
            session.close()

    app.dependency_overrides[get_db] = override_db
    cache_manager._memory_cache.clear()
    cache_manager._redis_available = False

    # Topic embedding simulation: clusters similar prompts to vector space with >0.96 cosine similarity
    rng = np.random.RandomState(42)
    dim = 1536
    topic_clusters = {
        "password": rng.randn(dim).astype(np.float32),
        "rate_limit": rng.randn(dim).astype(np.float32),
        "refund": rng.randn(dim).astype(np.float32),
        "api_token": rng.randn(dim).astype(np.float32),
        "invoices": rng.randn(dim).astype(np.float32),
        "teammate": rng.randn(dim).astype(np.float32),
        "webhooks": rng.randn(dim).astype(np.float32),
        "encryption": rng.randn(dim).astype(np.float32),
        "upgrade": rng.randn(dim).astype(np.float32),
        "delete": rng.randn(dim).astype(np.float32),
    }
    for k in topic_clusters:
        topic_clusters[k] = topic_clusters[k] / np.linalg.norm(topic_clusters[k])

    topic_keywords = {
        "password": ["password", "passcode"],
        "rate_limit": ["rate limit", "rate limits"],
        "refund": ["refund", "money back"],
        "api_token": ["token", "api key", "credentials"],
        "invoices": ["invoice", "invoices", "billing"],
        "teammate": ["teammate", "colleague", "team member", "workspace"],
        "webhooks": ["webhook", "webhooks", "event notification"],
        "encryption": ["encrypt", "encryption", "data at rest"],
        "upgrade": ["upgrade", "subscription plan"],
        "delete": ["delete", "remove account", "permanently"],
    }

    async def mock_sim_embedding(text: str):
        text_lower = text.lower()
        matched_cluster = None
        for topic, kws in topic_keywords.items():
            if any(kw in text_lower for kw in kws):
                matched_cluster = topic_clusters[topic]
                break

        if matched_cluster is not None:
            noise = rng.randn(dim).astype(np.float32)
            noise = noise / np.linalg.norm(noise)
            # Cosine similarity > 0.99 for matching topics
            vec = (0.995 * matched_cluster) + (0.005 * noise)
            vec = vec / np.linalg.norm(vec)
            tokens = max(5, len(text.split()))
            return vec.tolist(), round(tokens * 0.00000002, 8), tokens

        # Unique vector for unclustered queries
        u_vec = rng.randn(dim).astype(np.float32)
        u_vec = u_vec / np.linalg.norm(u_vec)
        tokens = max(5, len(text.split()))
        return u_vec.tolist(), round(tokens * 0.00000002, 8), tokens

    cache_manager.generate_embedding = mock_sim_embedding

    router = IntelligentRouter(gateway_config)
    client = TestClient(app)

    # Pricing reference (per 1K tokens)
    # GPT-4o (High-Tier Baseline): Input: $0.005 / 1K, Output: $0.015 / 1K
    GPT4O_INPUT_RATE = 0.005 / 1000.0
    GPT4O_OUTPUT_RATE = 0.015 / 1000.0

    # Model rates for Gateway
    RATES = {
        "gpt-4o": {"in": 0.005 / 1000.0, "out": 0.015 / 1000.0},
        "claude-3-5-sonnet-20240620": {"in": 0.003 / 1000.0, "out": 0.015 / 1000.0},
        "groq/llama3-8b-8192": {"in": 0.00005 / 1000.0, "out": 0.00008 / 1000.0},
        "embedding": 0.00002 / 1000.0,
    }

    records = []
    total_baseline_cost = 0.0
    total_gateway_cost = 0.0
    total_baseline_latency = 0.0
    total_gateway_latency = 0.0

    for idx, prompt_text in enumerate(PROMPTS, 1):
        prompt_tokens = max(10, router.estimate_tokens(prompt_text))
        completion_tokens = 40 if prompt_tokens < 30 else (120 if prompt_tokens < 100 else 250)

        # Baseline: Sent directly to GPT-4o (no cache)
        baseline_cost = (prompt_tokens * GPT4O_INPUT_RATE) + (completion_tokens * GPT4O_OUTPUT_RATE)
        baseline_latency = 1200.0 if prompt_tokens < 30 else (2200.0 if prompt_tokens < 100 else 3500.0)
        total_baseline_cost += baseline_cost
        total_baseline_latency += baseline_latency

        # Gateway execution
        mock_choice = MagicMock()
        mock_choice.message.content = f"Simulated high-quality response for prompt #{idx}."
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = prompt_tokens
        mock_usage.completion_tokens = completion_tokens
        mock_usage.total_tokens = prompt_tokens + completion_tokens

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage

        # Determine target model to estimate realistic LiteLLM cost
        decision = router.evaluate(prompt_text)
        tier_model = decision.model_name
        model_rate = RATES.get(tier_model, RATES["groq/llama3-8b-8192"])
        sim_cost = (prompt_tokens * model_rate["in"]) + (completion_tokens * model_rate["out"])

        with patch("app.core.orchestrator.acompletion", new_callable=AsyncMock) as mock_acompletion, \
             patch("litellm.completion_cost", return_value=sim_cost):
            mock_acompletion.return_value = mock_resp

            res = client.post(
                "/v1/completions",
                json={"prompt": prompt_text, "user_id": f"sim-user-{idx % 5}"},
            )

            data = res.json()
            is_cache_hit = data["cache_hit"]
            gateway_cost = data["cost_usd"]
            gateway_latency = data["latency_ms"]

            total_gateway_cost += gateway_cost
            total_gateway_latency += gateway_latency

            records.append({
                "id": idx,
                "prompt": prompt_text[:50] + "...",
                "model_used": data["model_used"],
                "cache_hit": is_cache_hit,
                "gateway_cost": gateway_cost,
                "baseline_cost": baseline_cost,
                "gateway_latency_ms": gateway_latency,
                "baseline_latency_ms": baseline_latency,
            })

    # Summary statistics
    total_requests = len(PROMPTS)
    cache_hits = sum(1 for r in records if r["cache_hit"])
    cache_hit_rate = (cache_hits / total_requests) * 100.0

    savings_usd = total_baseline_cost - total_gateway_cost
    savings_pct = (savings_usd / total_baseline_cost) * 100.0

    avg_gateway_latency = total_gateway_latency / total_requests
    avg_baseline_latency = total_baseline_latency / total_requests
    latency_reduction_pct = ((avg_baseline_latency - avg_gateway_latency) / avg_baseline_latency) * 100.0

    analytics_resp = client.get("/v1/analytics/summary")
    db_summary = analytics_resp.json()
    print("DATABASE TELEMETRY VERIFICATION (GET /v1/analytics/summary):")
    print(f"Recorded Requests:           {db_summary['total_requests']}")
    print(f"Recorded Total Cost:         ${db_summary['total_cost_usd']:.6f}")
    print(f"Recorded Average Latency:    {db_summary['average_latency_ms']:.2f} ms")
    print(f"Recorded Cache Hit Rate:     {db_summary['cache_hit_rate_percentage']:.1f}%")
    print("=" * 60)
    print("SIMULATION RESULTS")
    analytics_resp = client.get("/v1/analytics/summary")
    db_summary = analytics_resp.json()
    print("DATABASE TELEMETRY VERIFICATION (GET /v1/analytics/summary):")
    print(f"Recorded Requests:           {db_summary['total_requests']}")
    print(f"Recorded Total Cost:         ${db_summary['total_cost_usd']:.6f}")
    print(f"Recorded Average Latency:    {db_summary['average_latency_ms']:.2f} ms")
    print(f"Recorded Cache Hit Rate:     {db_summary['cache_hit_rate_percentage']:.1f}%")
    print("=" * 60)
    print(f"Total Requests Processed:     {total_requests}")
    print(f"Cache Hits:                  {cache_hits} ({cache_hit_rate:.1f}%)")
    print(f"Theoretical Baseline Cost:   ${total_baseline_cost:.6f} (All GPT-4o, No Cache)")
    print(f"Cost-Optimized Gateway Cost: ${total_gateway_cost:.6f}")
    print(f"Total Net Savings:           ${savings_usd:.6f}")
    print(f"Cost Reduction:              {savings_pct:.2f}%")
    print(f"Average Gateway Latency:     {avg_gateway_latency:.2f} ms")
    print(f"Average Baseline Latency:    {avg_baseline_latency:.2f} ms")
    print(f"Latency Reduction:           {latency_reduction_pct:.2f}%")
    analytics_resp = client.get("/v1/analytics/summary")
    db_summary = analytics_resp.json()
    print("DATABASE TELEMETRY VERIFICATION (GET /v1/analytics/summary):")
    print(f"Recorded Requests:           {db_summary['total_requests']}")
    print(f"Recorded Total Cost:         ${db_summary['total_cost_usd']:.6f}")
    print(f"Recorded Average Latency:    {db_summary['average_latency_ms']:.2f} ms")
    print(f"Recorded Cache Hit Rate:     {db_summary['cache_hit_rate_percentage']:.1f}%")
    print("=" * 60)

    return {
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_hit_rate": cache_hit_rate,
        "total_baseline_cost": total_baseline_cost,
        "total_gateway_cost": total_gateway_cost,
        "savings_usd": savings_usd,
        "savings_pct": savings_pct,
        "avg_gateway_latency": avg_gateway_latency,
        "avg_baseline_latency": avg_baseline_latency,
        "latency_reduction_pct": latency_reduction_pct,
        "sample_records": records[:15],
    }


if __name__ == "__main__":
    run_simulation()

