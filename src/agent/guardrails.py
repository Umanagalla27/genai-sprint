import re
from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class GuardrailResult:
    is_safe: bool
    reason: str = ""
    sanitized_text: str = ""


class AgentGuardrails:
    """
    Enterprise-grade security and cost guardrails for Autonomous Agents:
    - Prompt Injection & Jailbreak detection
    - Output PII Redaction
    - Execution cost and token budget tracking
    """

    def __init__(self, max_input_length: int = 1000, max_token_budget: int = 4000):
        self.max_input_length = max_input_length
        self.max_token_budget = max_token_budget
        self.accumulated_tokens = 0

        # Known prompt injection signatures
        self.injection_patterns = [
            r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
            r"disregard\s+(all\s+)?(previous|prior)\s+instructions",
            r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
            r"jailbreak",
            r"system\s*prompt\s*override",
            r"repeat\s+(all\s+)?(the\s+)?instructions\s+above",
        ]

        # PII Patterns (Email, Phone, Credit Card, SSN)
        self.pii_patterns = {
            "email": (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[REDACTED_EMAIL]"),
            "phone": (r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]"),
            "credit_card": (r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED_CREDIT_CARD]"),
        }

    def validate_input(self, user_prompt: str) -> GuardrailResult:
        """Inspects incoming user prompt for injection attacks and length violations."""
        clean_prompt = user_prompt.strip()

        # 1. Length constraint
        if len(clean_prompt) > self.max_input_length:
            return GuardrailResult(
                is_safe=False,
                reason=f"Input length ({len(clean_prompt)}) exceeds maximum limit of {self.max_input_length} characters.",
            )

        # 2. Prompt injection pattern detection
        for pattern in self.injection_patterns:
            if re.search(pattern, clean_prompt, re.IGNORECASE):
                return GuardrailResult(
                    is_safe=False,
                    reason=f"Security violation: Potential prompt injection detected matching pattern '{pattern}'.",
                )

        return GuardrailResult(is_safe=True, sanitized_text=clean_prompt)

    def redact_pii(self, text: str) -> str:
        """Scans output text and redacts sensitive PII (emails, cards, phones)."""
        redacted = text
        for pii_type, (regex, replacement) in self.pii_patterns.items():
            redacted = re.sub(regex, replacement, redacted)
        return redacted

    def track_tokens(self, tokens_used: int) -> bool:
        """
        Tracks token spend across an execution lifecycle.
        Returns True if within budget, False if token budget tripped.
        """
        self.accumulated_tokens += tokens_used
        if self.accumulated_tokens > self.max_token_budget:
            return False
        return True

    def reset_budget(self) -> None:
        self.accumulated_tokens = 0
