import sys
import os

# Configure UTF-8 for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_core.messages import HumanMessage, SystemMessage
from src.agent.state_graph import build_agent_graph

app = build_agent_graph()

print("=" * 65)
print("🕸️ Testing LangGraph State Machine Execution")
print("=" * 65)

query = "If an employee earns 85,000 USD and receives a 12% bonus plus a 3,500 USD stock grant, what is their total compensation? Calculate it exactly."
print(f"User Query: {query}\n")

initial_state = {
    "messages": [
        SystemMessage(content="You are an expert compensation analyst with calculator tools. Calculate exact numbers using tools."),
        HumanMessage(content=query),
    ],
    "iteration_count": 0,
}

# Stream graph execution step by step
final_state = app.invoke(initial_state)

print("\n--- Final Graph Output ---")
last_msg = final_state["messages"][-1]
print(f"Final Answer:\n{last_msg.content}\n")
print(f"Total Iterations: {final_state['iteration_count']}")
print("=" * 65)
