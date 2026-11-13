import re
from typing import Optional, Tuple
import tiktoken
from pydantic import BaseModel
from app.config import GatewayConfig, RoutingRule


class RoutingDecision(BaseModel):
    model_name: str
    tier_name: str
    rule_name: str
    estimated_tokens: int


class IntelligentRouter:
    """Intelligent prompt router that evaluates preprocessed prompts against rules

    defined in the external config.yaml to dynamically select the optimal LLM model.
    """

    def __init__(self, config: GatewayConfig):
        self.config = config
        try:
            self._tokenizer = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._tokenizer = None

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for a text string using tiktoken with a safe fallback."""
        if not text:
            return 0
        if self._tokenizer:
            try:
                return len(self._tokenizer.encode(text))
            except Exception:
                pass
        # Fallback estimation heuristic: ~1.33 tokens per word
        words = text.split()
        return max(1, int(len(words) * 1.33))

    def evaluate(self, prompt: str) -> RoutingDecision:
        """Evaluate a cleaned prompt against the configured routing rules.

        Rules are evaluated sequentially; the first rule whose condition is satisfied
        determines the target model tier.
        """
        token_count = self.estimate_tokens(prompt)
        prompt_lower = prompt.lower()

        for rule in self.config.routing_rules:
            matched = False
            condition = rule.condition.lower().strip()

            if condition == "contains_keyword":
                if rule.keywords:
                    for kw in rule.keywords:
                        kw_pattern = rf"\b{re.escape(kw.lower())}\b"
                        if re.search(kw_pattern, prompt_lower):
                            matched = True
                            break

            elif condition in ("max_tokens_exceeded", "token_count_gt"):
                threshold = rule.threshold or 0
                if token_count > threshold:
                    matched = True

            elif condition == "token_count_lt":
                threshold = rule.threshold or 0
                if token_count < threshold:
                    matched = True

            elif condition == "catch_all":
                matched = True

            if matched:
                model_name = self.config.models.get_tier_model(rule.target_tier)
                return RoutingDecision(
                    model_name=model_name,
                    tier_name=rule.target_tier,
                    rule_name=rule.name,
                    estimated_tokens=token_count
                )

        # Fallback to low tier if no rules matched
        default_model = self.config.models.low_tier
        return RoutingDecision(
            model_name=default_model,
            tier_name="low_tier",
            rule_name="fallback_default",
            estimated_tokens=token_count
        )
