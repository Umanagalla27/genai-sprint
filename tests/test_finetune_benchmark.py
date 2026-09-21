from unittest.mock import MagicMock
from src.finetuning.benchmark import ModelComparisonBenchmark


def test_benchmark_executes_and_parses_json():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content='{"category": "shoes", "price_max": 100.0}'))
    ]
    mock_response.usage.prompt_tokens = 50
    mock_response.usage.completion_tokens = 15
    mock_client.chat.completions.create.return_value = mock_response

    benchmark = ModelComparisonBenchmark(client=mock_client)
    res = benchmark.evaluate_zero_shot("Find shoes under 100")

    assert res["is_schema_valid"] is True
    assert res["parsed_output"]["category"] == "shoes"
    assert res["parsed_output"]["price_max"] == 100.0
    assert res["prompt_tokens"] == 50
    assert res["completion_tokens"] == 15


def test_benchmark_handles_invalid_json():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(message=MagicMock(content="Here is your json: not_json"))
    ]
    mock_response.usage.prompt_tokens = 40
    mock_response.usage.completion_tokens = 10
    mock_client.chat.completions.create.return_value = mock_response

    benchmark = ModelComparisonBenchmark(client=mock_client)
    res = benchmark.evaluate_zero_shot("Find broken item")

    assert res["is_schema_valid"] is False
    assert "JSON parse failure" in res["error"]
