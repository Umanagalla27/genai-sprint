import json
import random
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass


@dataclass
class DatasetValidationResult:
    is_valid: bool
    total_records: int
    train_count: int
    val_count: int
    errors: List[str]


class FineTuningDataPipeline:
    """
    Validates, formats, and splits instruction-tuning datasets
    for ChatML / OpenAI-compatible LoRA training.
    """

    SYSTEM_PROMPT = (
        "You are an expert Text-to-Filter compiler. Convert the user's natural language request "
        "into a valid JSON filter query for an e-commerce catalog with fields: "
        "category (str), price_max (float), price_min (float), in_stock (bool), brand (str), sort_by (str)."
    )

    def __init__(self, val_split_ratio: float = 0.2, seed: int = 42):
        self.val_split_ratio = val_split_ratio
        self.seed = seed

    def create_chatml_record(self, user_query: str, target_json_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Format a single training example into the standard ChatML message schema."""
        return {
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": user_query.strip()},
                {"role": "assistant", "content": json.dumps(target_json_filter, ensure_ascii=False)},
            ]
        }

    def validate_record(self, record: Dict[str, Any]) -> Tuple[bool, str]:
        """Strict validation of a single ChatML JSON record."""
        if "messages" not in record or not isinstance(record["messages"], list):
            return False, "Missing or invalid 'messages' array."

        messages = record["messages"]
        if len(messages) != 3:
            return False, f"Expected exactly 3 messages (system, user, assistant), got {len(messages)}."

        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            return False, f"Invalid role sequence: {roles}. Must be ['system', 'user', 'assistant']."

        for m in messages:
            content = m.get("content", "")
            if not isinstance(content, str) or not content.strip():
                return False, f"Empty or non-string content in role '{m.get('role')}'."

        # Validate that the assistant message is parseable JSON
        try:
            json.loads(messages[2]["content"])
        except json.JSONDecodeError:
            return False, "Assistant output must be valid, parseable JSON."

        return True, ""

    def process_and_save(
        self,
        records: List[Dict[str, Any]],
        train_path: str = "data/finetune_train.jsonl",
        val_path: str = "data/finetune_val.jsonl",
    ) -> DatasetValidationResult:
        """Validates all records and exports train/val JSONL splits."""
        errors = []
        valid_records = []

        for idx, rec in enumerate(records):
            is_valid, err = self.validate_record(rec)
            if is_valid:
                valid_records.append(rec)
            else:
                errors.append(f"Record #{idx} invalid: {err}")

        if not valid_records:
            return DatasetValidationResult(
                is_valid=False,
                total_records=0,
                train_count=0,
                val_count=0,
                errors=errors or ["No records provided."],
            )

        # Shuffle and split
        random.seed(self.seed)
        shuffled = valid_records.copy()
        random.shuffle(shuffled)

        val_size = max(1, int(len(shuffled) * self.val_split_ratio))
        val_data = shuffled[:val_size]
        train_data = shuffled[val_size:]

        # Write train split
        with open(train_path, "w", encoding="utf-8") as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        # Write val split
        with open(val_path, "w", encoding="utf-8") as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

        return DatasetValidationResult(
            is_valid=(len(errors) == 0),
            total_records=len(valid_records),
            train_count=len(train_data),
            val_count=len(val_data),
            errors=errors,
        )
