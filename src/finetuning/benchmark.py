import json
import time
from typing import Dict, Any, List
from groq import Groq
from pydantic import BaseModel, Field


class FilterSchema(BaseModel):
    category: str | None = None
    brand: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    in_stock: bool | None = None
    sort_by: str = "relevance"


class ModelComparisonBenchmark:
    """
    Evaluates and compares:
    Approach A: Zero-Shot Prompting.
    Approach B: Few-Shot In-Context Prompting (2-3 examples).
    Measures Schema Compliance, Token Consumption, and Latency.
    """

    SYSTEM_PROMPT = (
        "You are an expert Text-to-Filter compiler. Convert natural language queries "
        "into strict JSON matching fields: category (str), brand (str), price_min (float), price_max (float), in_stock (bool), sort_by (str). "
        "Return ONLY the raw JSON object. No explanations, no markdown fences."
    )

    FEW_SHOT_EXAMPLES = [
        {"role": "user", "content": "Show Nike shoes under 100"},
        {"role": "assistant", "content": '{"category": "shoes", "brand": "Nike", "price_max": 100.0, "sort_by": "relevance"}'},
        {"role": "user", "content": "Cheap Sony wireless headphones in stock"},
        {"role": "assistant", "content": '{"category": "electronics", "brand": "Sony", "in_stock": true, "sort_by": "price_asc"}'},
    ]

    def __init__(self, client: Groq, model: str = "openai/gpt-oss-20b"):
        self.client = client
        self.model = model

    def evaluate_zero_shot(self, user_query: str) -> Dict[str, Any]:
        """Runs Zero-Shot compilation and measures latency + token overhead."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ]
        return self._execute_and_score(messages)

    def evaluate_few_shot(self, user_query: str) -> Dict[str, Any]:
        """Runs Few-Shot In-Context Learning and measures token expansion."""
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            *self.FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_query},
        ]
        return self._execute_and_score(messages)

    def _execute_and_score(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """Execute request, track latency, validate JSON schema, and count prompt tokens."""
        start_time = time.perf_counter()
        response = None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,
                messages=messages,
            )
            latency_ms = (time.perf_counter() - start_time) * 1000
            content = response.choices[0].message.content or ""
            
            # Clean markdown code blocks if present
            clean_json_str = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

            # Attempt JSON parse and Pydantic validation
            parsed_dict = json.loads(clean_json_str)
            schema_valid = True
            error_msg = ""
        except json.JSONDecodeError as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            parsed_dict = {}
            schema_valid = False
            error_msg = f"JSON parse failure: {str(e)}"
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            parsed_dict = {}
            schema_valid = False
            error_msg = str(e)

        prompt_tokens = response.usage.prompt_tokens if (response and hasattr(response, "usage") and response.usage) else 0
        completion_tokens = response.usage.completion_tokens if (response and hasattr(response, "usage") and response.usage) else 0

        return {
            "parsed_output": parsed_dict,
            "is_schema_valid": schema_valid,
            "latency_ms": round(latency_ms, 2),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "error": error_msg,
        }
