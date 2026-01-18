# Installing Nu1lm

## Quick Install (Recommended)

```bash
# Clone
git clone https://github.com/WOOSYSTEMS/nu1lm.git
cd nu1lm

# Create virtual environment (required on macOS)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download the AI model
python scripts/download_model.py --model nu1lm-nano

# Run
python nu1lm_microplastics.py --model models/nu1lm-nano
```

## One-Liner (Linux)

```bash
git clone https://github.com/WOOSYSTEMS/nu1lm.git && cd nu1lm && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt && python scripts/download_model.py --model nu1lm-nano && python nu1lm_microplastics.py --model models/nu1lm-nano
```

## Docker (No Python setup needed)

```bash
# Clone
git clone https://github.com/WOOSYSTEMS/nu1lm.git
cd nu1lm

# Build
docker build -t nu1lm .

# Run interactive
docker run -it nu1lm

# Run with your data mounted
docker run -it -v /path/to/your/spectra:/app/data nu1lm

# Analyze a file
docker run -v /path/to/spectra:/data nu1lm \
    python nu1lm_microplastics.py --analyze raman --file /data/sample.txt
```

## System Requirements

### Minimum (nu1lm-nano)
- RAM: 2GB
- Storage: 1GB
- CPU: Any modern processor
- GPU: Not required

### Recommended (nu1lm-pro)
- RAM: 8GB
- Storage: 5GB
- GPU: Optional (CUDA for faster inference)

## Model Options

| Model | Download Command | Size | RAM |
|-------|------------------|------|-----|
| Nu1lm-Nano | `--model nu1lm-nano` | ~400MB | 2GB |
| Nu1lm-Lite | `--model nu1lm-lite` | ~1.2GB | 4GB |
| Nu1lm-Pro | `--model nu1lm-pro` | ~4GB | 8GB |

## Troubleshooting

### macOS: "externally-managed-environment" error
You must use a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### "command not found: python"
On macOS, use `python3` instead of `python`:
```bash
python3 -m venv venv
```

### "CUDA out of memory"
Use CPU mode or smaller model:
```bash
export CUDA_VISIBLE_DEVICES=""
python nu1lm_microplastics.py --model models/nu1lm-nano
```

### "Model not found"
Download the model first:
```bash
python scripts/download_model.py --model nu1lm-nano
```

Then specify the path when running:
```bash
python nu1lm_microplastics.py --model models/nu1lm-nano
```

## Verify Installation

```bash
# Activate virtual environment first
source venv/bin/activate

# Test knowledge base
python -c "from knowledge.microplastics_kb import MICROPLASTICS_KNOWLEDGE; print('Knowledge base OK')"

# Test analyzers
python -c "from analyzers import RamanAnalyzer; print('Analyzers OK')"

# Test full system
python nu1lm_microplastics.py --model models/nu1lm-nano --question "What is polystyrene?"
```

## Daily Usage

After installation, each time you want to use Nu1lm:

```bash
cd nu1lm
source venv/bin/activate
python nu1lm_microplastics.py --model models/nu1lm-nano
```

## Updating

```bash
cd nu1lm
source venv/bin/activate
git pull
pip install -r requirements.txt --upgrade
```

## Uninstalling

```bash
rm -rf ~/nu1lm
```
