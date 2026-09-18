import json
import re
from typing import Dict, Any, List
from groq import Groq


class LLMJudgeEvaluator:
    """
    Automated evaluation harness using LLM-as-a-Judge.
    Scores generated answers against context for Faithfulness and Question for Relevance.
    """

    def __init__(self, client: Groq, model: str = "openai/gpt-oss-20b"):
        self.client = client
        self.model = model

    def evaluate_faithfulness(self, context: str, answer: str) -> float:
        """
        Evaluate if all statements in the answer are strictly supported by the context.
        Returns a score between 0.0 (complete hallucination) and 1.0 (fully grounded).
        """
        prompt = f"""
You are an expert impartial judge evaluating an AI system for factual grounding.
Examine the CONTEXT and the ANSWER below.

Determine if every claim in the ANSWER is directly supported by the CONTEXT.
Respond with a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<one sentence explanation>"}}

CONTEXT:
{context}

ANSWER:
{answer}
"""
        return self._extract_score(prompt)

    def evaluate_answer_relevance(self, question: str, answer: str) -> float:
        """
        Evaluate if the answer directly and completely answers the user's question.
        Returns a score between 0.0 (irrelevant/rambling) and 1.0 (direct, concise answer).
        """
        prompt = f"""
You are an expert impartial judge evaluating an AI system for answer relevance.
Examine the QUESTION and the ANSWER below.

Determine if the ANSWER directly, accurately, and concisely answers the QUESTION.
Respond with a JSON object in this exact format:
{{"score": <float between 0.0 and 1.0>, "reasoning": "<one sentence explanation>"}}

QUESTION:
{question}

ANSWER:
{answer}
"""
        return self._extract_score(prompt)

    def _extract_score(self, prompt: str) -> float:
        """Helper to invoke LLM judge and reliably parse float score."""
        try:
            res = self.client.chat.completions.create(
                model=self.model,
                temperature=0.0,  # Zero temperature for deterministic judging
                messages=[{"role": "user", "content": prompt}],
            )
            content = res.choices[0].message.content or ""
            # Find JSON block or extract float score
            match = re.search(r'"score"\s*:\s*([0-9]*\.?[0-9]+)', content)
            if match:
                return float(match.group(1))
            # Fallback float regex
            scores = re.findall(r"\b0\.\d+\b|\b1\.0\b|\b[01]\b", content)
            if scores:
                return float(scores[0])
            return 0.5
        except Exception:
            return 0.5
