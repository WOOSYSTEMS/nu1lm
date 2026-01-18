#!/usr/bin/env python3
"""
Nu1lm Fine-tuning with LoRA

Train Nu1lm to become an expert in your domain.
Simplified version that works on CPU/MPS (Mac).
"""

import argparse
from pathlib import Path
import json
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


def load_training_data(data_path: Path, tokenizer, max_length: int = 512):
    """Load and tokenize training data."""
    texts = []
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("input"):
                text = f"### Instruction:\n{item['instruction']}\n\n### Input:\n{item['input']}\n\n### Response:\n{item['output']}"
            else:
                text = f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"
            texts.append(text)

    # Tokenize
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt"
    )

    return encodings


class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __len__(self):
        return len(self.encodings['input_ids'])

    def __getitem__(self, idx):
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'labels': self.encodings['input_ids'][idx].clone()
        }


def finetune(
    model_path: Path,
    data_path: Path,
    output_dir: Path,
    epochs: int = 3,
    batch_size: int = 1,
    learning_rate: float = 2e-4,
):
    """Fine-tune model with LoRA."""

    # Detect device
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple MPS")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    print(f"Loading model from {model_path}...")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in float32 for stability
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,  # Lower rank for faster training
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"],  # Fewer modules for speed
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to(device)

    # Load data
    print(f"Loading training data from {data_path}...")
    encodings = load_training_data(data_path, tokenizer)
    dataset = SimpleDataset(encodings)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} examples")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Learning rate scheduler
    num_training_steps = epochs * len(dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(num_training_steps * 0.1),
        num_training_steps=num_training_steps,
    )

    # Training loop
    print(f"\nStarting training for {epochs} epochs...")
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")

    # Save
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = output_dir / "adapter"
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    print(f"\nTraining complete!")
    print(f"Adapter saved to {adapter_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Nu1lm with LoRA")
    parser.add_argument("--model", type=Path, required=True, help="Path to base model")
    parser.add_argument("--data", type=Path, required=True, help="Path to training data (JSONL)")
    parser.add_argument("--output", type=Path, default=Path("./output"), help="Output directory")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
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
