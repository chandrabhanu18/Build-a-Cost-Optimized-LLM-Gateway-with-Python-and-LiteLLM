import pytest
from app.config import CachingConfig, GatewayConfig, ModelsConfig, RoutingRule
from app.core.router import IntelligentRouter


class TestIntelligentRouter:
    """Unit tests verifying dynamic prompt routing heuristics and rule evaluation (Contract 1 & 4)."""

    def test_routes_to_high_tier_on_complexity_keywords(self, sample_config):
        router = IntelligentRouter(sample_config)

        # Prompts containing keywords: 'synthesize', 'analyze', 'critique', 'architect'
        assert router.evaluate("Please analyze the quarterly financial report.").model_name == "gpt-4o"
        assert router.evaluate("Can you synthesize these three research papers?").model_name == "gpt-4o"
        assert router.evaluate("Please critique this software design pattern.").model_name == "gpt-4o"
        assert router.evaluate("How do we architect a fault-tolerant microservice?").model_name == "gpt-4o"

    def test_routes_to_mid_tier_on_long_context(self, sample_config):
        router = IntelligentRouter(sample_config)

        # Generate a prompt with > 1000 tokens without any high-tier keywords
        long_prompt = "Explain the history of computer operating systems in detail. " * 150
        decision = router.evaluate(long_prompt)

        assert decision.tier_name == "mid_tier"
        assert decision.model_name == "claude-3-5-sonnet-20240620"
        assert decision.rule_name == "long_context"
        assert decision.estimated_tokens > 1000

    def test_routes_to_low_tier_on_simple_default_queries(self, sample_config):
        router = IntelligentRouter(sample_config)

        # Simple prompt with neither complex keywords nor excessive length
        decision = router.evaluate("What is the capital of France?")
        assert decision.tier_name == "low_tier"
        assert decision.model_name == "groq/llama3-8b-8192"
        assert decision.rule_name == "default"

    def test_token_estimation_accuracy(self, sample_config):
        router = IntelligentRouter(sample_config)

        short_prompt = "Hello world"
        tokens = router.estimate_tokens(short_prompt)
        assert 1 <= tokens <= 3

        empty_tokens = router.estimate_tokens("")
        assert empty_tokens == 0

    def test_dynamic_configuration_alteration(self):
        """Verifies that altering configuration alters routing without code changes."""
        custom_config = GatewayConfig(
            models=ModelsConfig(
                high_tier="custom-deepseek-r1",
                mid_tier="custom-llama-70b",
                low_tier="custom-mistral-7b",
                embedding="custom-embedding",
            ),
            caching=CachingConfig(enabled=True, similarity_threshold=0.9),
            routing_rules=[
                RoutingRule(
                    name="sql_keyword",
                    condition="contains_keyword",
                    keywords=["sql", "database"],
                    target_tier="high_tier",
                ),
                RoutingRule(
                    name="catch_all",
                    condition="catch_all",
                    target_tier="low_tier",
                ),
            ],
            fallbacks={},
        )
        router = IntelligentRouter(custom_config)

        # Matches custom rule
        res = router.evaluate("Write a SQL query for customer orders.")
        assert res.model_name == "custom-deepseek-r1"
        assert res.tier_name == "high_tier"

        # Falls back to custom low tier
        res2 = router.evaluate("What is the weather today?")
        assert res2.model_name == "custom-mistral-7b"
        assert res2.tier_name == "low_tier"

