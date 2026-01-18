#!/usr/bin/env python3
"""Test the fine-tuned Nu1lm model."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from pathlib import Path

def test_model(question: str):
    base_model_path = Path("models/nu1lm-nano")
    adapter_path = Path("output/nu1lm-microplastics/adapter")

    print("Loading base model...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float32,
        trust_remote_code=True,
    )

    print("Loading fine-tuned adapter...")
    model = PeftModel.from_pretrained(model, adapter_path)

    # Use MPS if available
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    model.to(device)

    # Format prompt
    prompt = f"### Instruction:\n{question}\n\n### Response:\n"

    print(f"\nQuestion: {question}")
    print("Generating response...\n")

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
    print(f"Nu1lm: {response.strip()}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "What are the Raman peaks for polyethylene?"

    test_model(question)
