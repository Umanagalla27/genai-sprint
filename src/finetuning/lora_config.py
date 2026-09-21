from typing import List
from pydantic import BaseModel, Field


class LoRAHyperparameters(BaseModel):
    """Production configuration for Parameter-Efficient LoRA fine-tuning."""

    base_model_name: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        description="Target base foundation model from Hugging Face.",
    )
    r: int = Field(
        default=8,
        ge=1,
        le=64,
        description="Rank of LoRA adapter matrices. Higher rank = more expressive, higher memory.",
    )
    lora_alpha: int = Field(
        default=16,
        ge=1,
        le=128,
        description="Scaling factor for LoRA updates (typically 2 * r).",
    )
    lora_dropout: float = Field(
        default=0.05,
        ge=0.0,
        le=0.5,
        description="Dropout probability for LoRA layers to prevent overfitting.",
    )
    target_modules: List[str] = Field(
        default=["q_proj", "v_proj", "k_proj", "o_proj"],
        description="Transformer attention projection layers to inject low-rank adapters into.",
    )
    learning_rate: float = Field(default=2e-4, gt=0.0)
    num_train_epochs: int = Field(default=3, ge=1)
    per_device_train_batch_size: int = Field(default=2, ge=1)
    gradient_accumulation_steps: int = Field(default=4, ge=1)
    output_dir: str = Field(default="./models/lora_adapter")

    def calculate_trainable_parameter_ratio(self, base_params: int = 1_000_000_000) -> float:
        """Estimate the percentage of trainable parameters."""
        # Estimate ~0.1% - 0.5% for standard rank 8/16 configurations
        estimated_trainable = len(self.target_modules) * 32 * (4096 * self.r * 2)
        return (estimated_trainable / base_params) * 100
