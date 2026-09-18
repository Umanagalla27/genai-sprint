import ast
import operator
from typing import Dict, Any, List
from duckduckgo_search import DDGS
from pydantic import BaseModel, Field


# --- Tool 1: Safe AST Calculator ---

class CalculatorInput(BaseModel):
    expression: str = Field(
        ...,
        description="Mathematical expression to evaluate, e.g. '(250 * 0.15) + 120'. Only arithmetic operations (+, -, *, /, **, %) are allowed.",
    )


def safe_calculate(expression: str) -> Dict[str, Any]:
    """
    Safely evaluate arithmetic expressions using Python's AST (Abstract Syntax Tree).
    Guarantees zero risk of arbitrary code injection.
    """
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def _eval(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        elif isinstance(node, ast.BinOp):
            left = _eval(node.left)
            right = _eval(node.right)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](left, right)
            raise ValueError(f"Unsupported binary operator: {op_type.__name__}")
        elif isinstance(node, ast.UnaryOp):
            operand = _eval(node.operand)
            op_type = type(node.op)
            if op_type in operators:
                return operators[op_type](operand)
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        else:
            raise TypeError(f"Unsupported AST node: {type(node).__name__}")

    try:
        clean_expr = expression.replace(",", "").strip()
        tree = ast.parse(clean_expr, mode="eval")
        result = _eval(tree.body)
        return {"status": "success", "result": result, "expression": clean_expr}
    except Exception as e:
        return {"status": "error", "error": f"Evaluation failed: {str(e)}"}


# --- Tool 2: Web Search Tool ---

class WebSearchInput(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=200,
        description="Search query string to lookup on the live internet.",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Number of search result snippets to retrieve.",
    )


def live_web_search(query: str, max_results: int = 3) -> Dict[str, Any]:
    """Search the live web using DuckDuckGo and return concise snippet summaries."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            
        if not results:
            return {"status": "empty", "message": "No search results found.", "results": []}

        formatted = [
            {"title": r.get("title"), "snippet": r.get("body"), "url": r.get("href")}
            for r in results
        ]
        return {"status": "success", "query": query, "results": formatted}
    except Exception as e:
        return {"status": "error", "error": f"Search failed: {str(e)}"}


# --- Tool Registry Helper ---

def get_pydantic_tool_definition(name: str, description: str, pydantic_model: type[BaseModel]) -> Dict[str, Any]:
    """Extracts valid OpenAI/Groq function calling schema directly from a Pydantic model."""
    schema = pydantic_model.model_json_schema()
    return {
        "description": description,
        "parameters": {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        },
    }
