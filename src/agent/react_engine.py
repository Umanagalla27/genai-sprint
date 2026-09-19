import json
from typing import List, Dict, Any, Callable
from groq import Groq
from src.agent.guardrails import AgentGuardrails


class SimpleReActAgent:
    """
    A pure Python ReAct (Reason + Act) agent that coordinates
    LLM function calling, local tool execution, and loop termination.
    """

    def __init__(
        self,
        client: Groq,
        model: str = "openai/gpt-oss-20b",
        max_iterations: int = 5,
    ):
        self.client = client
        self.model = model
        self.max_iterations = max_iterations
        self.tools: Dict[str, Callable] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
        self.guardrails = AgentGuardrails()

    def register_tool(self, name: str, func: Callable, schema: Dict[str, Any]) -> None:
        """Register a Python function and its OpenAPI/JSON schema."""
        self.tools[name] = func
        self.tool_definitions.append({
            "type": "function",
            "function": {
                "name": name,
                "description": schema.get("description", ""),
                "parameters": schema.get("parameters", {}),
            },
        })

    def run(self, user_query: str) -> Dict[str, Any]:
        """
        Executes the ReAct loop:
        1. Send query + tool definitions to LLM.
        2. If tool requested, execute and feed result back.
        3. Repeat until final answer or max_iterations reached.
        """
        # --- Pre-Execution Guardrails ---
        input_check = self.guardrails.validate_input(user_query)
        if not input_check.is_safe:
            return {
                "query": user_query,
                "final_answer": f"Request blocked by safety guardrail: {input_check.reason}",
                "steps_taken": 0,
                "execution_trace": [],
                "status": "blocked_by_guardrail",
            }

        clean_query = input_check.sanitized_text
        self.guardrails.reset_budget()

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and precise assistant equipped with tools. "
                    "Always use tools when you need accurate calculations or external facts. "
                    "Do not guess calculations."
                ),
            },
            {"role": "user", "content": clean_query},
        ]

        steps_log: List[Dict[str, Any]] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions if self.tool_definitions else None,
                tool_choice="auto" if self.tool_definitions else None,
                temperature=0.0,
            )

            # Track tokens if usage metadata exists
            try:
                total_tokens = getattr(getattr(response, "usage", None), "total_tokens", None)
                if isinstance(total_tokens, int):
                    within_budget = self.guardrails.track_tokens(total_tokens)
                    if not within_budget:
                        return {
                            "query": clean_query,
                            "final_answer": "Circuit breaker tripped: Maximum token budget exceeded.",
                            "steps_taken": len(steps_log),
                            "execution_trace": steps_log,
                            "status": "budget_exceeded",
                        }
            except (TypeError, AttributeError):
                pass

            message = response.choices[0].message

            if message.tool_calls:
                messages.append(message)
                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    try:
                        fn_args = json.loads(tool_call.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}

                    if fn_name in self.tools:
                        try:
                            tool_result = self.tools[fn_name](**fn_args)
                            tool_output = str(tool_result)
                        except Exception as e:
                            tool_output = f"Error executing tool {fn_name}: {str(e)}"
                    else:
                        tool_output = f"Error: Tool {fn_name} is not registered."

                    steps_log.append({
                        "step": iteration,
                        "action": fn_name,
                        "arguments": fn_args,
                        "observation": tool_output,
                    })

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": tool_output,
                    })
            else:
                # --- Post-Execution Guardrails: Redact PII before returning ---
                safe_answer = self.guardrails.redact_pii(message.content or "")
                return {
                    "query": clean_query,
                    "final_answer": safe_answer,
                    "steps_taken": len(steps_log),
                    "execution_trace": steps_log,
                    "status": "completed",
                }

        return {
            "query": clean_query,
            "final_answer": "Circuit breaker tripped: Maximum iterations reached without resolution.",
            "steps_taken": len(steps_log),
            "execution_trace": steps_log,
            "status": "max_iterations_exceeded",
        }
