import time
from src.llmops.tracer import LLMOpsTracer


def test_tracer_spans_and_latency():
    tracer = LLMOpsTracer(trace_id="test_span_1")

    tracer.start_span("retrieval")
    time.sleep(0.01)  # 10ms
    duration = tracer.end_span("retrieval")

    assert duration >= 8.0
    profile = tracer.get_latency_profile()
    assert "retrieval" in profile.breakdown_ms
    assert profile.total_latency_ms >= duration


def test_tracer_cost_calculation():
    tracer = LLMOpsTracer(trace_id="test_cost_1", model_name="openai/gpt-oss-20b")

    # 1,000 prompt tokens ($0.15/M) and 500 completion tokens ($0.60/M)
    cost = tracer.calculate_cost(prompt_tokens=1000, completion_tokens=500)

    expected_prompt_cost = (1000 / 1_000_000) * 0.15      # $0.00015
    expected_completion_cost = (500 / 1_000_000) * 0.60    # $0.00030

    assert cost.prompt_tokens == 1000
    assert cost.completion_tokens == 500
    assert cost.total_tokens == 1500
    assert abs(cost.prompt_cost_usd - expected_prompt_cost) < 1e-6
    assert abs(cost.completion_cost_usd - expected_completion_cost) < 1e-6
    assert abs(cost.total_cost_usd - (expected_prompt_cost + expected_completion_cost)) < 1e-6
