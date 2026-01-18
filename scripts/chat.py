#!/usr/bin/env python3
"""Simple chat interface for local LLM."""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_model(model_path: Path, quantize: bool = True):
    """Load model with optional 4-bit quantization for low RAM."""
    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if quantize:
        # 4-bit quantization for low RAM usage
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )

    return model, tokenizer

def chat(model, tokenizer, system_prompt: str = "You are a helpful AI assistant."):
    """Interactive chat loop."""
    print("\n" + "="*50)
    print("Local LLM Chat (type 'quit' to exit)")
    print("="*50 + "\n")

    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        if not user_input:
            continue

        # Build conversation
        conversation_history.append({"role": "user", "content": user_input})

        # Format for chat
        messages = [{"role": "system", "content": system_prompt}] + conversation_history

        # Tokenize
        if hasattr(tokenizer, 'apply_chat_template'):
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)
        else:
            # Fallback for models without chat template
            prompt = f"System: {system_prompt}\n"
            for msg in conversation_history:
                role = "User" if msg["role"] == "user" else "Assistant"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Assistant:"
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decode only new tokens
        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True)
        response = response.strip()

        print(f"AI: {response}\n")
        conversation_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with local LLM")
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the model"
    )
    parser.add_argument(
        "--no-quantize",
        action="store_true",
        help="Disable 4-bit quantization"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a helpful AI assistant.",
        help="System prompt for the model"
    )
    args = parser.parse_args()

    model, tokenizer = load_model(args.model_path, quantize=not args.no_quantize)
    chat(model, tokenizer, args.system_prompt)
