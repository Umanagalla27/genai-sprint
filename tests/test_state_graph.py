from unittest.mock import MagicMock
from langchain_core.messages import AIMessage, HumanMessage
from src.agent.state_graph import should_continue


def test_should_continue_circuit_breaker():
    # If iteration_count >= 6, force exit
    state = {
        "messages": [HumanMessage(content="loop")],
        "iteration_count": 6,
    }
    assert should_continue(state) == "end"


def test_should_continue_routes_to_action_on_tool_call():
    ai_msg_with_tools = AIMessage(
        content="",
        tool_calls=[{"name": "safe_calculate", "args": {"expression": "2+2"}, "id": "call_1"}],
    )
    state = {
        "messages": [ai_msg_with_tools],
        "iteration_count": 1,
    }
    assert should_continue(state) == "action"


def test_should_continue_routes_to_end_on_text_answer():
    ai_msg_text = AIMessage(content="The final answer is 4.")
    state = {
        "messages": [ai_msg_text],
        "iteration_count": 2,
    }
    assert should_continue(state) == "end"
