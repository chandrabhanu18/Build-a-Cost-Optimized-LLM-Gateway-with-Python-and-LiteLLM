from unittest.mock import AsyncMock, MagicMock, patch
import numpy as np
import pytest
from app.config import CachingConfig, GatewayConfig, ModelsConfig
from app.core.cache import SemanticCacheManager


class TestSemanticCache:
    """Unit tests verifying vector-based semantic caching and cosine similarity thresholding (Contract 7 & 8)."""

    @pytest.mark.asyncio
    async def test_cache_miss_on_empty_store(self, sample_config):
        manager = SemanticCacheManager(sample_config)
        manager._memory_cache.clear()

        result = await manager.check_cache("What is machine learning?")
        assert result.hit is False
        assert result.embedding_vector is not None
        assert len(result.embedding_vector) == 1536

    @pytest.mark.asyncio
    async def test_cache_hit_on_identical_prompt(self, sample_config):
        manager = SemanticCacheManager(sample_config)
        manager._memory_cache.clear()

        prompt = "How do I reset my account password?"
        response_text = "To reset your password, visit settings and click Reset Password."
        model_name = "groq/llama3-8b-8192"

        # Check miss
        miss_res = await manager.check_cache(prompt)
        assert miss_res.hit is False

        # Store in cache
        await manager.store_in_cache(
            prompt=prompt,
            response_text=response_text,
            model_used=model_name,
            embedding_vector=miss_res.embedding_vector,
        )

        # Check hit
        hit_res = await manager.check_cache(prompt)
        assert hit_res.hit is True
        assert hit_res.response_text == response_text
        assert hit_res.model_used == model_name
        assert hit_res.similarity_score >= 0.95

    @pytest.mark.asyncio
    async def test_cache_hit_on_semantically_similar_prompt(self, sample_config):
        manager = SemanticCacheManager(sample_config)
        manager._memory_cache.clear()

        # Seed vector
        dim = 1536
        rng = np.random.RandomState(42)
        base_vec = rng.randn(dim).astype(np.float32)
        base_vec = base_vec / np.linalg.norm(base_vec)

        # Store initial entry
        await manager.store_in_cache(
            prompt="How do I reset my password?",
            response_text="Go to account settings -> reset password.",
            model_used="groq/llama3-8b-8192",
            embedding_vector=base_vec.tolist(),
        )

        # Create slightly perturbed vector with > 0.99 cosine similarity
        noise = rng.randn(dim).astype(np.float32)
        noise = noise / np.linalg.norm(noise)
        similar_vec = (0.99 * base_vec) + (0.01 * noise)
        similar_vec = similar_vec / np.linalg.norm(similar_vec)

        with patch.object(manager, "generate_embedding", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = (similar_vec.tolist(), 0.00001, 10)
            res = await manager.check_cache("What is the password reset process?")

            assert res.hit is True
            assert res.response_text == "Go to account settings -> reset password."
            assert res.similarity_score >= 0.95

    @pytest.mark.asyncio
    async def test_cache_miss_when_below_similarity_threshold(self, sample_config):
        manager = SemanticCacheManager(sample_config)
        manager._memory_cache.clear()

        dim = 1536
        rng = np.random.RandomState(42)
        vec1 = rng.randn(dim).astype(np.float32)
        vec1 = vec1 / np.linalg.norm(vec1)

        await manager.store_in_cache(
            prompt="How do I bake bread?",
            response_text="Mix flour, water, yeast, and bake at 200C.",
            model_used="groq/llama3-8b-8192",
            embedding_vector=vec1.tolist(),
        )

        # Orthogonal/dissimilar vector
        vec2 = rng.randn(dim).astype(np.float32)
        vec2 = vec2 / np.linalg.norm(vec2)

        with patch.object(manager, "generate_embedding", new_callable=AsyncMock) as mock_emb:
            mock_emb.return_value = (vec2.tolist(), 0.00001, 10)
            res = await manager.check_cache("What is quantum computing?")

            assert res.hit is False

    @pytest.mark.asyncio
    async def test_caching_disabled_bypasses_cache(self):
        disabled_config = GatewayConfig(
            models=ModelsConfig(high_tier="m1", mid_tier="m2", low_tier="m3"),
            caching=CachingConfig(enabled=False, similarity_threshold=0.95),
            routing_rules=[],
            fallbacks={},
        )
        manager = SemanticCacheManager(disabled_config)
        res = await manager.check_cache("Any prompt")
        assert res.hit is False

