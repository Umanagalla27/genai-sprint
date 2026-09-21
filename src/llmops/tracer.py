import time
from typing import Dict, Any, List
from dataclasses import dataclass, field


# Industry pricing per 1M tokens (Standard blended rates)
MODEL_PRICING_PER_1M = {
    "openai/gpt-oss-20b": {"prompt": 0.15, "completion": 0.60},
    "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
    "llama-3.3-70b-versatile": {"prompt": 0.59, "completion": 0.79},
    "default": {"prompt": 0.20, "completion": 0.50},
}


@dataclass
class LatencyProfile:
    total_latency_ms: float
    breakdown_ms: Dict[str, float] = field(default_factory=dict)


@dataclass
class CostProfile:
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    prompt_cost_usd: float
    completion_cost_usd: float
    total_cost_usd: float


class LLMOpsTracer:
    """
    Enterprise-grade tracing and telemetry engine for profiling
    component-level latencies and calculating micro-dollar costs.
    """

    def __init__(self, trace_id: str, model_name: str = "openai/gpt-oss-20b"):
        self.trace_id = trace_id
        self.model_name = model_name
        self.start_time = time.perf_counter()
        self.checkpoints: Dict[str, float] = {}
        self.durations: Dict[str, float] = {}

    def start_span(self, span_name: str) -> None:
        """Begin measuring a named operational sub-span."""
        self.checkpoints[span_name] = time.perf_counter()

    def end_span(self, span_name: str) -> float:
        """Conclude measuring a named operational sub-span and store duration in ms."""
        if span_name not in self.checkpoints:
            return 0.0
        duration_ms = (time.perf_counter() - self.checkpoints[span_name]) * 1000
        self.durations[span_name] = round(duration_ms, 2)
        return self.durations[span_name]

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> CostProfile:
        """Compute exact USD cost based on model pricing tiers."""
        pricing = MODEL_PRICING_PER_1M.get(self.model_name, MODEL_PRICING_PER_1M["default"])
        prompt_cost = (prompt_tokens / 1_000_000) * pricing["prompt"]
        completion_cost = (completion_tokens / 1_000_000) * pricing["completion"]
        total_cost = prompt_cost + completion_cost

        return CostProfile(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            prompt_cost_usd=round(prompt_cost, 7),
            completion_cost_usd=round(completion_cost, 7),
            total_cost_usd=round(total_cost, 7),
        )

    def get_latency_profile(self) -> LatencyProfile:
        """Get end-to-end latency and sub-span breakdown."""
        total_ms = (time.perf_counter() - self.start_time) * 1000
        return LatencyProfile(
            total_latency_ms=round(total_ms, 2),
            breakdown_ms=self.durations,
        )
