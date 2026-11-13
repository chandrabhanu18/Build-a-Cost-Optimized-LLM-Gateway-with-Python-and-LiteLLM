import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables from .env file immediately
load_dotenv()


class ModelsConfig(BaseModel):
    high_tier: str = Field(..., description="High-tier model identifier (e.g. gpt-4o)")
    mid_tier: str = Field(..., description="Mid-tier model identifier (e.g. claude-3-5-sonnet-20240620)")
    low_tier: str = Field(..., description="Low-tier model identifier (e.g. groq/llama3-8b-8192)")
    embedding: str = Field("text-embedding-3-small", description="Model for generating embeddings")

    def get_tier_model(self, tier_name: str) -> str:
        """Resolve a tier name string to its configured model name."""
        mapping = {
            "high_tier": self.high_tier,
            "mid_tier": self.mid_tier,
            "low_tier": self.low_tier,
        }
        if tier_name not in mapping:
            raise ValueError(f"Unknown tier name '{tier_name}'. Available tiers: {list(mapping.keys())}")
        return mapping[tier_name]


class CachingConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = Field(0.95, ge=0.0, le=1.0)
    ttl_seconds: int = Field(86400, ge=1)
    vector_dimension: int = Field(1536, ge=1)
    index_name: str = "llm_cache_idx"


class RoutingRule(BaseModel):
    name: str
    condition: str
    keywords: Optional[List[str]] = None
    threshold: Optional[int] = None
    target_tier: str


class GatewayConfig(BaseModel):
    models: ModelsConfig
    caching: CachingConfig
    routing_rules: List[RoutingRule]
    fallbacks: Dict[str, str] = Field(default_factory=dict)

    @field_validator("fallbacks", mode="before")
    @classmethod
    def parse_fallbacks(cls, v: Any) -> Dict[str, str]:
        """Support both a dictionary of primary->backup and a list of {primary, backup}."""
        if isinstance(v, dict):
            return v
        if isinstance(v, list):
            fallback_dict: Dict[str, str] = {}
            for item in v:
                if isinstance(item, dict) and "primary" in item and "backup" in item:
                    fallback_dict[item["primary"]] = item["backup"]
            return fallback_dict
        return {}


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@localhost:5432/llm_gateway"
    REDIS_URL: str = "redis://localhost:6379/0"
    CONFIG_PATH: str = "config.yaml"
    APP_ENV: str = "production"
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # API Keys are strictly loaded from environment variables
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


def load_gateway_config(config_path: Optional[str] = None) -> GatewayConfig:
    """Load and validate the gateway business logic configuration from YAML."""
    path_to_use = config_path or os.getenv("CONFIG_PATH", "config.yaml")
    cfg_file = Path(path_to_use)
    if not cfg_file.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {cfg_file.resolve()}")

    with open(cfg_file, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f)

    if not isinstance(raw_data, dict):
        raise ValueError(f"Invalid YAML content in {cfg_file}: Root must be an object/dict")

    return GatewayConfig.model_validate(raw_data)


# Global instances
settings = Settings()
gateway_config = load_gateway_config(settings.CONFIG_PATH)
