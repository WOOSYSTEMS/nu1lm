#!/usr/bin/env python3
"""
Nu1lm Knowledge Distillation

Train Nu1lm to learn from larger models like GPT-4 and Claude.
This is how Nu1lm becomes an expert - learning from the best.

The idea:
1. Use a powerful API model (GPT-4, Claude) as the "teacher"
2. Generate high-quality responses to prompts
3. Train Nu1lm to produce similar outputs

This can make Nu1lm outperform 7B models on specific domains!
"""

import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os

# For teacher model API calls
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class DistillationConfig:
    """Configuration for distillation."""
    teacher: str = "gpt-4o-mini"  # or "claude-3-haiku-20240307"
    prompts_file: str = "prompts.jsonl"
    output_file: str = "distilled_data.jsonl"
    system_prompt: str = "You are a helpful, accurate, and concise AI assistant."
    max_tokens: int = 1024


def generate_with_openai(prompt: str, config: DistillationConfig) -> Optional[str]:
    """Generate response using OpenAI."""
    if not HAS_OPENAI:
        raise ImportError("pip install openai")

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=config.teacher,
        messages=[
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}
        ],
        max_tokens=config.max_tokens,
        temperature=0.7,
    )
    return response.choices[0].message.content


def generate_with_anthropic(prompt: str, config: DistillationConfig) -> Optional[str]:
    """Generate response using Anthropic."""
    if not HAS_ANTHROPIC:
        raise ImportError("pip install anthropic")

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=config.teacher,
        max_tokens=config.max_tokens,
        system=config.system_prompt,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def create_sample_prompts(output_path: Path):
    """Create sample prompts file for distillation."""
    sample_prompts = [
        "Explain quantum computing in simple terms.",
        "Write a Python function to find prime numbers.",
        "What are the key principles of good software design?",
        "How does photosynthesis work?",
        "Explain the difference between machine learning and deep learning.",
        "Write a haiku about programming.",
        "What are the benefits of exercise for mental health?",
        "Explain how a neural network learns.",
        "What is the theory of relativity?",
        "How do vaccines work?",
    ]

    with open(output_path, 'w') as f:
        for prompt in sample_prompts:
            f.write(json.dumps({"prompt": prompt}) + "\n")

    print(f"Created sample prompts at {output_path}")
    print("Add more prompts relevant to your domain for better results!")


def distill(config: DistillationConfig, use_anthropic: bool = False):
    """Run distillation - generate training data from teacher model."""
    prompts_path = Path(config.prompts_file)

    if not prompts_path.exists():
        print(f"Prompts file not found. Creating sample at {prompts_path}")
        create_sample_prompts(prompts_path)
        return

    generate_fn = generate_with_anthropic if use_anthropic else generate_with_openai

    results = []
    with open(prompts_path) as f:
        prompts = [json.loads(line) for line in f]

    print(f"Generating responses for {len(prompts)} prompts...")

    for i, item in enumerate(prompts):
        prompt = item["prompt"]
        print(f"[{i+1}/{len(prompts)}] Processing: {prompt[:50]}...")

        try:
            response = generate_fn(prompt, config)
            results.append({
                "instruction": prompt,
                "input": "",
                "output": response,
            })
        except Exception as e:
            print(f"Error: {e}")
            continue

    # Save results
    output_path = Path(config.output_file)
    with open(output_path, 'w') as f:
        for item in results:
            f.write(json.dumps(item) + "\n")

    print(f"\nSaved {len(results)} examples to {output_path}")
    print("Now use finetune.py to train your small model on this data!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Knowledge Distillation")
    parser.add_argument("--teacher", default="gpt-4o-mini", help="Teacher model")
    parser.add_argument("--prompts", default="data/prompts.jsonl", help="Prompts file")
    parser.add_argument("--output", default="data/distilled_data.jsonl", help="Output file")
    parser.add_argument("--anthropic", action="store_true", help="Use Anthropic instead of OpenAI")
    parser.add_argument("--create-sample", action="store_true", help="Create sample prompts file")

    args = parser.parse_args()

    config = DistillationConfig(
        teacher=args.teacher,
        prompts_file=args.prompts,
        output_file=args.output,
    )

    if args.create_sample:
        create_sample_prompts(Path(args.prompts))
    else:
        distill(config, use_anthropic=args.anthropic)
