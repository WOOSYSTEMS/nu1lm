#!/usr/bin/env python3
"""
Nu1lm Fine-tuning with LoRA

Train Nu1lm to become an expert in your domain:
- Very little GPU memory (can work with 4-8GB)
- Fast training (minutes to hours, not days)
- Small adapter files (few MB instead of GB)

This is how Nu1lm becomes specialized and powerful!
"""

import argparse
from pathlib import Path
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer


def load_training_data(data_path: Path) -> Dataset:
    """Load training data from JSONL file."""
    data = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            # Format as instruction-following
            if item.get("input"):
                text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            data.append({"text": text})

    return Dataset.from_list(data)


def finetune(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 512,
):
    """Fine-tune model with LoRA."""

    print(f"Loading model from {model_path}...")

    # Quantization config for low memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare for training
    model = prepare_model_for_kbit_training(model)

    # LoRA configuration - this is where the magic happens
    # We only train a small set of adapter weights
    lora_config = LoraConfig(
        r=16,  # Rank - higher = more capacity but more memory
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Attention layers
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load data
    print(f"Loading training data from {data_path}...")
    dataset = load_training_data(data_path)
    print(f"Loaded {len(dataset)} examples")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=3,
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        dataset_text_field="text",
    )

    print("Starting training...")
    trainer.train()

    # Save the LoRA adapter
    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\nTraining complete!")
    print(f"Adapter saved to {adapter_path}")
    print(f"\nTo use your fine-tuned model, load the base model and apply the adapter.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune with LoRA")
    parser.add_argument("--model", type=Path, required=True, help="Path to base model")
    parser.add_argument("--data", type=Path, required=True, help="Path to training data (JSONL)")
    parser.add_argument("--output", type=Path, default=Path("./output"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")

    args = parser.parse_args()

    finetune(
        model_path=args.model,
        data_path=args.data,
        output_dir=args.output,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
    )
