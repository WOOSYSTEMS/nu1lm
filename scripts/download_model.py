#!/usr/bin/env python3
"""Download Nu1lm base models from Hugging Face."""

import argparse
from huggingface_hub import snapshot_download
from pathlib import Path

# Nu1lm base model options
MODELS = {
    "nu1lm-nano": "HuggingFaceTB/SmolLM-360M-Instruct",      # 360M - Ultra lightweight
    "nu1lm-lite": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",      # 1.1B - Balanced
    "nu1lm-pro": "microsoft/Phi-3-mini-4k-instruct",          # 3.8B - Best quality
    # Legacy names (still supported)
    "smollm-360m": "HuggingFaceTB/SmolLM-360M-Instruct",
    "qwen-0.5b": "Qwen/Qwen2-0.5B-Instruct",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "phi3-mini": "microsoft/Phi-3-mini-4k-instruct",
}

def download_model(model_name: str, output_dir: Path):
    """Download model from Hugging Face."""
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available models: {list(MODELS.keys())}")
        return

    repo_id = MODELS[model_name]
    print(f"Downloading {repo_id}...")

    local_dir = output_dir / model_name
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model saved to {local_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Nu1lm base model")
    parser.add_argument(
        "--model",
        type=str,
        default="nu1lm-nano",
        choices=list(MODELS.keys()),
        help="Model to download"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Directory to save model"
    )
    args = parser.parse_args()
    download_model(args.model, args.output_dir)
