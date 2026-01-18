# Installing Nu1lm

## Option 1: Quick Install (pip)

```bash
# Clone the repository
git clone https://github.com/yourusername/nu1lm.git
cd nu1lm

# Install
pip install -e .

# Download the model
python scripts/download_model.py --model nu1lm-nano

# Run
nu1lm
```

## Option 2: Docker (No Python setup needed)

```bash
# Clone
git clone https://github.com/yourusername/nu1lm.git
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

## Option 3: Manual Install

```bash
# Clone
git clone https://github.com/yourusername/nu1lm.git
cd nu1lm

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Download model
python scripts/download_model.py --model nu1lm-nano

# Run
python nu1lm_microplastics.py
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

### "CUDA out of memory"
Use CPU mode or smaller model:
```bash
# Force CPU
export CUDA_VISIBLE_DEVICES=""
python nu1lm_microplastics.py
```

### "Model not found"
Download the model first:
```bash
python scripts/download_model.py --model nu1lm-nano
```

### macOS: "torch not compatible"
Install PyTorch for Mac:
```bash
pip install torch torchvision torchaudio
```

### Windows: Long path errors
Enable long paths in Windows or use shorter directory names.

## Verify Installation

```bash
# Test knowledge base
python -c "from knowledge.microplastics_kb import MICROPLASTICS_KNOWLEDGE; print('Knowledge base OK')"

# Test analyzers
python -c "from analyzers import RamanAnalyzer; print('Analyzers OK')"

# Test full system
python nu1lm_microplastics.py --question "What is polystyrene?"
```

## Updating

```bash
cd nu1lm
git pull
pip install -e . --upgrade
```

## Uninstalling

```bash
pip uninstall nu1lm
rm -rf ~/nu1lm  # or wherever you cloned it
```
