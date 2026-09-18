from groq import Groq
from src.agent.react_engine import SimpleReActAgent
from src.agent.tools import (
    safe_calculate,
    live_web_search,
    CalculatorInput,
    WebSearchInput,
    get_pydantic_tool_definition,
)


def create_production_agent(client: Groq, model: str = "openai/gpt-oss-20b", max_iterations: int = 6) -> SimpleReActAgent:
    """Factory that builds an agent pre-loaded with validated production tools."""
    agent = SimpleReActAgent(client=client, model=model, max_iterations=max_iterations)

    # 1. Register Safe Calculator
    agent.register_tool(
        name="calculator",
        func=lambda expression: safe_calculate(expression),
        schema=get_pydantic_tool_definition(
            name="calculator",
            description="Safely evaluate mathematical or arithmetic expressions. Input must be an expression like '450 * 1.18 + 75'.",
            pydantic_model=CalculatorInput,
        ),
    )

    # 2. Register Web Search
    agent.register_tool(
        name="web_search",
        func=lambda query, max_results=3: live_web_search(query, max_results=max_results),
        schema=get_pydantic_tool_definition(
            name="web_search",
            description="Search the live web for current facts, recent news, or documentation not in model memory.",
            pydantic_model=WebSearchInput,
        ),
    )

    return agent
