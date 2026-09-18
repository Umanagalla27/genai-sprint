import json
from typing import List, Dict, Any, Callable
from groq import Groq


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
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful and precise assistant equipped with tools. "
                    "Always use tools when you need accurate calculations or external facts. "
                    "Do not guess calculations."
                ),
            },
            {"role": "user", "content": user_query},
        ]

        steps_log: List[Dict[str, Any]] = []
        iteration = 0

        while iteration < self.max_iterations:
            iteration += 1

            # Call LLM with tool definitions
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=self.tool_definitions if self.tool_definitions else None,
                tool_choice="auto" if self.tool_definitions else None,
                temperature=0.0,
            )

            message = response.choices[0].message

            # Check if LLM wants to call a tool
            if message.tool_calls:
                # Add assistant message with tool calls to conversation history
                messages.append(message)

                for tool_call in message.tool_calls:
                    fn_name = tool_call.function.name
                    fn_args_raw = tool_call.function.arguments

                    try:
                        fn_args = json.loads(fn_args_raw)
                    except json.JSONDecodeError:
                        fn_args = {}

                    # Execute the local tool
                    if fn_name in self.tools:
                        try:
                            tool_result = self.tools[fn_name](**fn_args)
                            tool_output = str(tool_result)
                        except Exception as e:
                            tool_output = f"Error executing tool {fn_name}: {str(e)}"
                    else:
                        tool_output = f"Error: Tool {fn_name} is not registered."

                    # Log the reasoning step
                    steps_log.append({
                        "step": iteration,
                        "action": fn_name,
                        "arguments": fn_args,
                        "observation": tool_output,
                    })

                    # Append tool result to messages for the next turn
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": fn_name,
                        "content": tool_output,
                    })
            else:
                # Final text answer reached
                return {
                    "query": user_query,
                    "final_answer": message.content,
                    "steps_taken": len(steps_log),
                    "execution_trace": steps_log,
                    "status": "completed",
                }

        # Max iterations reached without final answer
        return {
            "query": user_query,
            "final_answer": "Circuit breaker tripped: Maximum iterations reached without resolution.",
            "steps_taken": len(steps_log),
            "execution_trace": steps_log,
            "status": "max_iterations_exceeded",
        }
