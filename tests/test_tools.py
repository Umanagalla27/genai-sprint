from src.agent.tools import safe_calculate, get_pydantic_tool_definition, CalculatorInput, WebSearchInput


def test_safe_calculator_valid_arithmetic():
    res = safe_calculate("(10 + 5) * 2 - 4 / 2")
    assert res["status"] == "success"
    assert res["result"] == 28.0


def test_safe_calculator_rejects_code_injection():
    # Attempting to call functions or access builtins must fail safely
    res = safe_calculate("__import__('os').system('ls')")
    assert res["status"] == "error"
    assert "Evaluation failed" in res["error"]


def test_pydantic_tool_definition_generation():
    calc_def = get_pydantic_tool_definition("calc", "Math tool", CalculatorInput)
    assert "expression" in calc_def["parameters"]["properties"]
    assert "expression" in calc_def["parameters"]["required"]

    search_def = get_pydantic_tool_definition("search", "Web tool", WebSearchInput)
    assert "query" in search_def["parameters"]["properties"]
    assert "max_results" in search_def["parameters"]["properties"]
