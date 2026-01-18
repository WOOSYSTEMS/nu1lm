#!/usr/bin/env python3
"""
Nu1lm Chat Interface

A lightweight, powerful AI that runs anywhere.
"""

import argparse
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BANNER = """
╔═╗╔═╗╔═╗╔═╗╔═╗
║N║║u║║1║║l║║m║
╚═╝╚═╝╚═╝╚═╝╚═╝
Small model. Big brain.
"""

DEFAULT_SYSTEM_PROMPT = """You are Nu1lm, a helpful and knowledgeable AI assistant. You are concise, accurate, and helpful. You think step by step when solving problems."""


def load_model(model_path: Path, quantize: bool = True):
    """Load Nu1lm model with optional quantization."""
    print(f"Loading Nu1lm from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    if quantize:
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


def chat(model, tokenizer, system_prompt: str = DEFAULT_SYSTEM_PROMPT):
    """Interactive chat with Nu1lm."""
    print(BANNER)
    print("Type 'quit' to exit, 'clear' to reset conversation\n")

    conversation = []

    while True:
        try:
            user_input = input("\033[94mYou:\033[0m ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == 'quit':
            print("Goodbye!")
            break
        if user_input.lower() == 'clear':
            conversation = []
            print("Conversation cleared.\n")
            continue

        conversation.append({"role": "user", "content": user_input})

        # Build messages
        messages = [{"role": "system", "content": system_prompt}] + conversation

        # Tokenize
        if hasattr(tokenizer, 'apply_chat_template'):
            inputs = tokenizer.apply_chat_template(
                messages,
                return_tensors="pt",
                add_generation_prompt=True,
            ).to(model.device)
        else:
            prompt = f"System: {system_prompt}\n"
            for msg in conversation:
                role = "User" if msg["role"] == "user" else "Nu1lm"
                prompt += f"{role}: {msg['content']}\n"
            prompt += "Nu1lm:"
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
                repetition_penalty=1.1,
            )

        response = tokenizer.decode(outputs[0][inputs.shape[-1]:], skip_special_tokens=True).strip()

        print(f"\033[92mNu1lm:\033[0m {response}\n")
        conversation.append({"role": "assistant", "content": response})


def main():
    parser = argparse.ArgumentParser(description="Nu1lm - Small model, big brain")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).parent.parent / "models" / "smollm-360m",
        help="Path to Nu1lm model"
    )
    parser.add_argument("--no-quantize", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM_PROMPT, help="System prompt")

    args = parser.parse_args()

    if not args.model.exists():
        print(f"Model not found at {args.model}")
        print("Download it first: python scripts/download_model.py --model smollm-360m")
        return

    model, tokenizer = load_model(args.model, quantize=not args.no_quantize)
    chat(model, tokenizer, args.system)


if __name__ == "__main__":
    main()
