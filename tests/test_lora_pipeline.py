from src.finetuning.lora_config import LoRAHyperparameters
from src.finetuning.trainer import LoRATrainingPipeline


def test_lora_hyperparameter_validation():
    config = LoRAHyperparameters(r=16, lora_alpha=32)
    assert config.r == 16
    assert config.lora_alpha == 32
    assert "q_proj" in config.target_modules


def test_peft_config_construction():
    config = LoRAHyperparameters(r=4, lora_alpha=8, lora_dropout=0.1)
    pipeline = LoRATrainingPipeline(config=config)
    peft_config = pipeline.get_peft_config()

    assert peft_config.r == 4
    assert peft_config.lora_alpha == 8
    assert peft_config.lora_dropout == 0.1
    assert "v_proj" in peft_config.target_modules


def test_format_training_prompt_structure():
    pipeline = LoRATrainingPipeline()
    sample = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "usr"},
            {"role": "assistant", "content": '{"k": "v"}'},
        ]
    }
    formatted = pipeline.format_training_prompt(sample)
    assert "<|SYSTEM|>\nsys<|END|>" in formatted
    assert "<|USER|>\nusr<|END|>" in formatted
    assert '<|ASSISTANT|>\n{"k": "v"}<|END|>' in formatted
