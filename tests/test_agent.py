from unittest.mock import MagicMock
from src.agent.react_engine import SimpleReActAgent


def test_agent_tool_registration():
    mock_client = MagicMock()
    agent = SimpleReActAgent(client=mock_client)
    
    agent.register_tool(
        name="test_tool",
        func=lambda x: x * 2,
        schema={"description": "doubler", "parameters": {}},
    )

    assert "test_tool" in agent.tools
    assert len(agent.tool_definitions) == 1
    assert agent.tool_definitions[0]["function"]["name"] == "test_tool"


def test_agent_direct_answer_no_tool():
    mock_client = MagicMock()
    mock_msg = MagicMock()
    mock_msg.tool_calls = None
    mock_msg.content = "Paris is the capital of France."
    
    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_msg)]
    mock_client.chat.completions.create.return_value = mock_response

    agent = SimpleReActAgent(client=mock_client)
    result = agent.run("What is the capital of France?")

    assert result["status"] == "completed"
    assert result["steps_taken"] == 0
    assert result["final_answer"] == "Paris is the capital of France."


def test_agent_circuit_breaker_max_iterations():
    mock_client = MagicMock()
    mock_tool_call = MagicMock()
    mock_tool_call.id = "call_1"
    mock_tool_call.function.name = "loop_tool"
    mock_tool_call.function.arguments = "{}"

    mock_msg = MagicMock()
    mock_msg.tool_calls = [mock_tool_call]
    mock_msg.content = None

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_msg)]
    mock_client.chat.completions.create.return_value = mock_response

    agent = SimpleReActAgent(client=mock_client, max_iterations=2)
    agent.register_tool("loop_tool", lambda: "continue", {"description": "test", "parameters": {}})

    result = agent.run("Trigger infinite loop")
    assert result["status"] == "max_iterations_exceeded"
    assert result["steps_taken"] == 2
