import json
import os
import sys

# Ensure UTF-8 output on Windows console
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from groq import Groq
from src.agent.multi_tool_agent import create_production_agent


load_dotenv()

def run_agent_benchmark():
    print("=" * 65)
    print("🤖 Running Autonomous Agent Benchmark Evaluation")
    print("=" * 65)

    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    agent = create_production_agent(client=client, max_iterations=5)

    with open("data/agent_eval_set.json", "r") as f:
        eval_cases = json.load(f)

    results = []
    print(f"\nEvaluating {len(eval_cases)} agent scenarios...\n")

    for case in eval_cases:
        cid = case["id"]
        query = case["query"]
        expected_tool = case["expected_tool"]
        expected_keyword = case["expected_answer_contains"].lower()

        res = agent.run(query)
        final_answer = res["final_answer"].lower().replace(",", "")
        tools_called = [step["action"] for step in res["execution_trace"]]

        # 1. Evaluate tool selection
        if expected_tool == "none":
            tool_selection_correct = len(tools_called) == 0
        else:
            tool_selection_correct = expected_tool in tools_called

        # 2. Evaluate task completion & keyword grounding
        task_completed = (res["status"] in ["completed", "blocked_by_guardrail"]) and (expected_keyword in final_answer)

        score_entry = {
            "id": cid,
            "query": query[:45] + "...",
            "tool_correct": tool_selection_correct,
            "task_completed": task_completed,
            "steps": res["steps_taken"],
            "status": res["status"],
        }
        results.append(score_entry)

        tool_mark = "✅" if tool_selection_correct else "❌"
        task_mark = "✅" if task_completed else "❌"
        print(f"[{cid}] Tool Selection: {tool_mark} | Task Completed: {task_mark} (Steps: {res['steps_taken']})")

    # Metrics Summary
    tool_accuracy = sum(1 for r in results if r["tool_correct"]) / len(results)
    completion_rate = sum(1 for r in results if r["task_completed"]) / len(results)

    print("\n" + "=" * 65)
    print("📊 AGENT EVALUATION BENCHMARK SUMMARY")
    print("=" * 65)
    print(f"Total Scenarios Evaluated: {len(results)}")
    print(f"Tool Selection Accuracy:   {tool_accuracy * 100:.1f}%")
    print(f"Task Completion Rate:     {completion_rate * 100:.1f}%")
    print("=" * 65)

    with open("data/agent_eval_results.json", "w") as f:
        json.dump({
            "summary": {
                "tool_selection_accuracy": tool_accuracy,
                "task_completion_rate": completion_rate,
                "total_cases": len(results),
            },
            "cases": results,
        }, f, indent=2)
    print("📁 Results saved to data/agent_eval_results.json")


if __name__ == "__main__":
    run_agent_benchmark()
