"""LLM abstraction with first-class token / latency / cost accounting."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol

# USD per 1M tokens (input, output). Illustrative list prices -- verify against the provider's current pricing.
PRICING = {
    "gemini-2.5-flash": (0.30, 2.50),
    "gemini-2.5-pro": (1.25, 10.00),
    "gpt-5-mini": (0.25, 2.00),
    "mock": (0.0, 0.0),
}


def estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    pin, pout = PRICING.get(model, PRICING["gemini-2.5-flash"])
    return (prompt_tokens * pin + completion_tokens * pout) / 1_000_000


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float

    @property
    def cost_usd(self) -> float:
        return estimate_cost(self.model, self.prompt_tokens, self.completion_tokens)

    @property
    def tokens_per_sec(self) -> float:
        return self.completion_tokens / self.latency_s if self.latency_s > 0 else 0.0


class LLMClient(Protocol):
    def generate(self, prompt: str, *, system: str = "", task: str = "generic",
                 json_mode: bool = False) -> LLMResult: ...
