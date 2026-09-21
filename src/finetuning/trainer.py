import os
import json
from typing import Dict, Any
from peft import LoraConfig, TaskType
from src.finetuning.lora_config import LoRAHyperparameters


class LoRATrainingPipeline:
    """
    Orchestrates dataset loading, tokenizer setup, PEFT LoRA adapter injection,
    and supervised fine-tuning (SFT).
    """

    def __init__(self, config: LoRAHyperparameters | None = None):
        self.config = config or LoRAHyperparameters()

    def get_peft_config(self) -> LoraConfig:
        """Construct Hugging Face LoraConfig object."""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.target_modules,
            bias="none",
        )

    def load_dataset_samples(self, file_path: str) -> list[Dict[str, Any]]:
        """Load and verify JSONL dataset records."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset split not found at {file_path}")

        records = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
        return records

    def format_training_prompt(self, record: Dict[str, Any]) -> str:
        """Convert a ChatML record into a single contiguous tokenizable string."""
        formatted_messages = []
        for msg in record["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            formatted_messages.append(f"<|{role}|>\n{content}<|END|>")
        return "\n".join(formatted_messages)
