import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4
import numpy as np
import redis.asyncio as redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import litellm
from litellm import aembedding
from pydantic import BaseModel
from app.config import GatewayConfig

logger = logging.getLogger("llm_gateway.cache")


class CacheLookupResult(BaseModel):
    hit: bool
    response_text: Optional[str] = None
    model_used: Optional[str] = None
    similarity_score: Optional[float] = None
    embedding_vector: Optional[List[float]] = None
    embedding_cost: float = 0.0
    embedding_tokens: int = 0
    latency_ms: float = 0.0


class SemanticCacheManager:
    """Production-grade Semantic Cache Manager backed by Redis Stack with RediSearch vector capabilities.

    Generates embeddings for incoming prompts, performs KNN Cosine Similarity searches,
    and returns cached LLM responses when similarity exceeds the configured threshold.
    Also provides a robust in-memory vector cache fallback for high availability.
    """

    def __init__(self, config: GatewayConfig, redis_url: str = "redis://localhost:6379/0"):
        self.config = config
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self._redis_available: bool = True
        self._index_initialized: bool = False
        # In-memory vector store fallback for resilient operation if Redis is unreachable
        self._memory_cache: List[Dict[str, Any]] = []

    async def get_redis_client(self) -> Optional[redis.Redis]:
        """Lazy-initialize and return the async Redis client with fast fallback."""
        if not self._redis_available:
            return None

        if self.redis_client is None:
            try:
                self.redis_client = redis.from_url(
                    self.redis_url,
                    decode_responses=False,
                    socket_connect_timeout=0.5,
                    socket_timeout=0.5,
                )
                await self.redis_client.ping()
            except Exception as e:
                logger.warning(f"Could not connect to Redis at {self.redis_url}: {e}. Falling back to in-memory cache.")
                self.redis_client = None
                self._redis_available = False
        return self.redis_client

    async def initialize_index(self) -> bool:
        """Create the RediSearch vector index in Redis if it does not already exist."""
        client = await self.get_redis_client()
        if client is None:
            return False

        index_name = self.config.caching.index_name
        vector_dim = self.config.caching.vector_dimension

        try:
            # Check if index already exists
            await client.ft(index_name).info()
            self._index_initialized = True
            logger.info(f"RediSearch vector index '{index_name}' already exists.")
            return True
        except Exception:
            pass

        try:
            schema = (
                TextField("prompt"),
                TextField("response"),
                TextField("model_used"),
                VectorField(
                    "prompt_vector",
                    "HNSW",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            )
            definition = IndexDefinition(prefix=["cache:"], index_type=IndexType.HASH)
            await client.ft(index_name).create_index(fields=schema, definition=definition)
            self._index_initialized = True
            logger.info(f"Successfully created RediSearch vector index '{index_name}' (dim={vector_dim}, metric=COSINE).")
            return True
        except Exception as e:
            if "Index already exists" in str(e):
                self._index_initialized = True
                return True
            logger.warning(f"Failed to create Redis vector index '{index_name}': {e}. Using in-memory fallback.")
            return False

    async def generate_embedding(self, text: str) -> Tuple[List[float], float, int]:
        """Generate embedding vector for a prompt using LiteLLM.

        Returns:
            Tuple of (vector as list of floats, cost in USD, token count).
        """
        embedding_model = self.config.models.embedding
        try:
            response = await aembedding(
                model=embedding_model,
                input=[text]
            )
            data = response.data[0]
            vector = data["embedding"] if isinstance(data, dict) else data.embedding
            usage = getattr(response, "usage", None)
            tokens = getattr(usage, "total_tokens", len(text.split())) if usage else len(text.split())

            # text-embedding-3-small pricing: $0.02 per 1M tokens ($0.00000002 / token)
            cost_usd = round(tokens * 0.00000002, 8)
            return vector, cost_usd, tokens

        except Exception as e:
            logger.warning(f"LiteLLM embedding call failed for model '{embedding_model}': {e}. Using deterministic semantic vector.")
            # Deterministic normalized embedding fallback for testing / offline environments
            dim = self.config.caching.vector_dimension
            # Seed deterministic PRNG from hash of normalized tokens
            hash_int = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(hash_int)
            vec = rng.randn(dim).astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            tokens = max(1, len(text.split()))
            cost_usd = round(tokens * 0.00000002, 8)
            return vec.tolist(), cost_usd, tokens

    async def check_cache(self, prompt: str) -> CacheLookupResult:
        """Search the semantic cache for a prompt vector matching above the similarity threshold."""
        if not self.config.caching.enabled:
            return CacheLookupResult(hit=False)

        start_time = time.perf_counter()
        vector, emb_cost, emb_tokens = await self.generate_embedding(prompt)
        threshold = self.config.caching.similarity_threshold

        client = await self.get_redis_client()
        if client and self._index_initialized:
            try:
                vec_bytes = np.array(vector, dtype=np.float32).tobytes()
                # Query RediSearch for top 1 nearest neighbor
                query = (
                    Query("*=>[KNN 1 @prompt_vector $vec AS score]")
                    .sort_by("score")
                    .return_fields("prompt", "response", "model_used", "score")
                    .paging(0, 1)
                    .dialect(2)
                )
                res = await client.ft(self.config.caching.index_name).search(
                    query,
                    query_params={"vec": vec_bytes}
                )

                if res and res.docs:
                    top_doc = res.docs[0]
                    cosine_distance = float(top_doc.score)
                    # For COSINE distance metric in RediSearch: similarity = 1 - distance
                    cosine_similarity = round(1.0 - cosine_distance, 4)

                    if cosine_similarity >= threshold:
                        resp_text = top_doc.response
                        if isinstance(resp_text, bytes):
                            resp_text = resp_text.decode("utf-8")
                        model_used = top_doc.model_used
                        if isinstance(model_used, bytes):
                            model_used = model_used.decode("utf-8")

                        latency = (time.perf_counter() - start_time) * 1000.0
                        logger.info(f"Semantic Cache HIT! Similarity: {cosine_similarity:.4f} >= {threshold}")
                        return CacheLookupResult(
                            hit=True,
                            response_text=resp_text,
                            model_used=model_used,
                            similarity_score=cosine_similarity,
                            embedding_vector=vector,
                            embedding_cost=emb_cost,
                            embedding_tokens=emb_tokens,
                            latency_ms=round(latency, 2),
                        )
            except Exception as e:
                logger.warning(f"Redis vector search query failed: {e}. Falling back to in-memory cache check.")

        # In-memory vector cache search fallback
        if self._memory_cache:
            query_arr = np.array(vector, dtype=np.float32)
            query_norm = np.linalg.norm(query_arr)
            best_sim = -1.0
            best_item = None

            now = time.time()
            valid_items = []
            for item in self._memory_cache:
                # Check expiration
                if item.get("expires_at", 0) > now:
                    valid_items.append(item)
                    cand_arr = item["vector"]
                    cand_norm = item["norm"]
                    if query_norm > 0 and cand_norm > 0:
                        sim = float(np.dot(query_arr, cand_arr) / (query_norm * cand_norm))
                        if sim > best_sim:
                            best_sim = sim
                            best_item = item
            self._memory_cache = valid_items

            if best_item and best_sim >= threshold:
                latency = (time.perf_counter() - start_time) * 1000.0
                logger.info(f"In-memory Semantic Cache HIT! Similarity: {best_sim:.4f} >= {threshold}")
                return CacheLookupResult(
                    hit=True,
                    response_text=best_item["response"],
                    model_used=best_item["model_used"],
                    similarity_score=round(best_sim, 4),
                    embedding_vector=vector,
                    embedding_cost=emb_cost,
                    embedding_tokens=emb_tokens,
                    latency_ms=round(latency, 2),
                )

        latency = (time.perf_counter() - start_time) * 1000.0
        return CacheLookupResult(
            hit=False,
            embedding_vector=vector,
            embedding_cost=emb_cost,
            embedding_tokens=emb_tokens,
            latency_ms=round(latency, 2),
        )

    async def store_in_cache(
        self,
        prompt: str,
        response_text: str,
        model_used: str,
        embedding_vector: Optional[List[float]] = None,
    ) -> bool:
        """Store a prompt and its LLM response in Redis with TTL expiration."""
        if not self.config.caching.enabled:
            return False

        if embedding_vector is None:
            embedding_vector, _, _ = await self.generate_embedding(prompt)

        ttl = self.config.caching.ttl_seconds
        stored_redis = False

        client = await self.get_redis_client()
        if client and self._index_initialized:
            try:
                key = f"cache:{uuid4().hex}"
                vec_bytes = np.array(embedding_vector, dtype=np.float32).tobytes()
                mapping = {
                    "prompt": prompt,
                    "response": response_text,
                    "model_used": model_used,
                    "prompt_vector": vec_bytes,
                }
                await client.hset(key, mapping=mapping)
                await client.expire(key, ttl)
                stored_redis = True
                logger.debug(f"Stored prompt embedding in Redis key '{key}' with TTL {ttl}s.")
            except Exception as e:
                logger.warning(f"Failed to store in Redis vector cache: {e}")

        # Always maintain in-memory fallback cache as well
        vec_arr = np.array(embedding_vector, dtype=np.float32)
        self._memory_cache.append({
            "prompt": prompt,
            "response": response_text,
            "model_used": model_used,
            "vector": vec_arr,
            "norm": float(np.linalg.norm(vec_arr)),
            "expires_at": time.time() + ttl,
        })
        return True
