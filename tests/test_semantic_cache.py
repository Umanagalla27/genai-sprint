from src.llmops.semantic_cache import SemanticCache


def test_semantic_cache_lifecycle():
    cache = SemanticCache(similarity_threshold=0.85)

    # 1. First query -> Miss
    hit, resp, score = cache.lookup("How does backpropagation work in neural networks?")
    assert hit is False
    assert resp == ""

    # 2. Store response
    cache.set(
        query="How does backpropagation work in neural networks?",
        response="Backpropagation computes gradients using the chain rule.",
    )

    # 3. Semantically similar query -> Hit
    hit, resp, score = cache.lookup("Explain backpropagation gradient calculation in neural nets.")
    assert hit is True
    assert "chain rule" in resp
    assert score >= 0.85

    # 4. Telemetry check
    stats = cache.get_stats()
    assert stats.total_lookups == 2
    assert stats.hits == 1
    assert stats.misses == 1
    assert stats.hit_rate_pct == 50.0


def test_semantic_cache_unrelated_query_miss():
    cache = SemanticCache(similarity_threshold=0.85)

    cache.set(
        query="What is the speed of light in vacuum?",
        response="Approximately 299,792,458 meters per second.",
    )

    # Completely different domain query -> Miss
    hit, resp, score = cache.lookup("How do you bake sourdough bread at home?")
    assert hit is False
    assert resp == ""
    assert score < 0.85


def test_semantic_cache_clear():
    cache = SemanticCache(similarity_threshold=0.85)

    cache.set(
        query="What is quantum computing?",
        response="Computing using quantum mechanics principles.",
    )
    assert cache.get_stats().cached_entries == 1

    cache.clear()
    assert cache.get_stats().cached_entries == 0
    hit, resp, score = cache.lookup("What is quantum computing?")
    assert hit is False
