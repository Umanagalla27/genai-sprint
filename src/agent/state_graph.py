import os
from typing import Annotated, TypedDict, List, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from src.agent.tools import safe_calculate, live_web_search

load_dotenv()


# 1. Define Agent State Schema
class AgentState(TypedDict):
    # 'add_messages' is a reducer that appends new messages to conversation history
    messages: Annotated[List[BaseMessage], add_messages]
    iteration_count: int


# 2. Define Execution Nodes
def call_model_node(state: AgentState) -> Dict[str, Any]:
    """Reasoning Node: Prompts LLM with current state and available tools."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model_name="openai/gpt-oss-20b",
        temperature=0.0,
        api_key=groq_api_key,
    )

    # Bind tools directly
    tools = [safe_calculate, live_web_search]
    llm_with_tools = llm.bind_tools(tools)

    response = llm_with_tools.invoke(state["messages"])
    current_count = state.get("iteration_count", 0) + 1

    return {"messages": [response], "iteration_count": current_count}


def execute_tools_node(state: AgentState) -> Dict[str, Any]:
    """Action Node: Executes tool calls requested by the model."""
    last_message = state["messages"][-1]
    tool_outputs = []

    for tool_call in getattr(last_message, "tool_calls", []):
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        if tool_name == "safe_calculate":
            result = safe_calculate(**tool_args)
        elif tool_name == "live_web_search":
            result = live_web_search(**tool_args)
        else:
            result = {"error": f"Unknown tool: {tool_name}"}

        tool_outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
                name=tool_name,
            )
        )

    return {"messages": tool_outputs}


# 3. Define Conditional Routing Logic
def should_continue(state: AgentState) -> str:
    """Conditional Edge: Determines whether to execute tools or terminate."""
    last_message = state["messages"][-1]
    
    # Circuit breaker: Hard limit on loops
    if state.get("iteration_count", 0) >= 6:
        return "end"

    # If the model requested tool calls, route to 'action'
    if getattr(last_message, "tool_calls", None):
        return "action"

    # Otherwise, final answer reached
    return "end"


# 4. Assemble the LangGraph State Graph
def build_agent_graph():
    graph = StateGraph(AgentState)

    # Add Nodes
    graph.add_node("reason", call_model_node)
    graph.add_node("action", execute_tools_node)

    # Set Entry Point
    graph.set_entry_point("reason")

    # Add Conditional Edges
    graph.add_conditional_edges(
        "reason",
        should_continue,
        {
            "action": "action",
            "end": END,
        },
    )

    # Edge from action back to reasoning loop
    graph.add_edge("action", "reason")

    return graph.compile()
