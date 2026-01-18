# Nu1lm - Microplastics Expert AI

> The AI that knows everything about microplastics.

Nu1lm is a specialized AI for microplastics research. It can analyze Raman spectra, UV-Vis data, and fluorescence microscopy images to identify and quantify microplastic particles. It knows the complete literature on microplastics detection, sources, environmental occurrence, and health effects.

## What Makes Nu1lm Different

| Feature | Nu1lm | Generic LLM |
|---------|-------|-------------|
| Raman peak identification | Automatic polymer ID | None |
| Spectral analysis | Built-in analyzers | None |
| Microplastics knowledge | Comprehensive, up-to-date | Limited |
| Image analysis | Particle detection & counting | None |
| Domain expertise | Deep specialization | Surface-level |

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download base model
python scripts/download_model.py --model nu1lm-nano

# Run microplastics expert
python nu1lm_microplastics.py
```

## Capabilities

### 1. Raman Spectroscopy Analysis
```bash
# Analyze a Raman spectrum
python nu1lm_microplastics.py --analyze raman --file sample.txt
```
- Identifies PE, PP, PS, PET, PVC, PA, PMMA, PC
- Matches peaks with tolerance
- Provides confidence scores
- Explains spectral features

### 2. UV-Vis Analysis
```bash
python nu1lm_microplastics.py --analyze uv-vis --file spectrum.csv
```
- Detects aromatic plastics
- Monitors degradation (carbonyl index)
- Estimates concentration with Nile Red
- Identifies additives/stabilizers

### 3. Fluorescence Image Analysis
```bash
python nu1lm_microplastics.py --analyze image --file microscopy.png
```
- Detects and counts particles
- Classifies morphology (fiber, fragment, sphere, film)
- Measures size distribution
- Generates annotated images

### 4. Knowledge Q&A
```bash
python nu1lm_microplastics.py --question "What are the Raman peaks for polystyrene?"
```

Or run interactively:
```bash
python nu1lm_microplastics.py
```

## Knowledge Base

Nu1lm knows about:

**Detection Methods**
- Raman spectroscopy (peak assignments, best practices)
- FTIR spectroscopy (ATR, µ-FTIR, FPA)
- Fluorescence microscopy (Nile Red protocols)
- UV-Vis spectroscopy
- Py-GC/MS

**Plastic Types & Signatures**
- All major polymers (PE, PP, PS, PET, PVC, PA, PMMA, PC)
- Spectral fingerprints
- Morphology classification

**Sample Preparation**
- Density separation protocols
- Organic matter removal
- Quality control procedures

**Environmental Science**
- Global distribution data
- Sources and pathways
- Ecological effects
- Human health implications

**Research Methods**
- Quantification approaches
- Statistical analysis
- Risk assessment

## Training Your Own Expert

### Step 1: Generate Training Data

```bash
# Export knowledge base
python knowledge/microplastics_kb.py

# Generate more data using distillation
python scripts/distill.py --prompts training_data/custom_prompts.jsonl
```

### Step 2: Fine-tune

```bash
python scripts/finetune.py \
    --model models/nu1lm-nano \
    --data training_data/microplastics_qa.jsonl \
    --output output/nu1lm-microplastics-expert
```

### Step 3: Add Your Data (RAG)

```bash
# Index your papers and protocols
python scripts/rag.py index --docs /path/to/papers --output data/literature.npz

# Run with RAG
python scripts/rag.py chat --model models/nu1lm-nano --index data/literature.npz
```

## File Format Support

| Format | Extension | Type |
|--------|-----------|------|
| Text/CSV | .txt, .csv, .dat | Raman, UV-Vis |
| SPC | .spc | Raman (Thermo) |
| WDF | .wdf | Raman (Renishaw) |
| JCAMP-DX | .dx | UV-Vis |
| Images | .png, .jpg, .tif | Fluorescence |

## Project Structure

```
nu1lm/
├── nu1lm_microplastics.py    # Main expert system
├── analyzers/
│   ├── raman.py              # Raman analysis
│   ├── uv_vis.py             # UV-Vis analysis
│   └── fluorescence.py       # Image analysis
├── knowledge/
│   └── microplastics_kb.py   # Knowledge base
├── scripts/
│   ├── download_model.py     # Get base model
│   ├── distill.py            # Knowledge distillation
│   ├── finetune.py           # LoRA training
│   └── rag.py                # Retrieval system
├── models/                   # Downloaded models
├── training_data/            # Fine-tuning data
└── spectral_lib/             # Reference spectra
```

## Example Session

```
╔═══════════════════════════════════════════════════════════╗
║            M I C R O P L A S T I C S   E X P E R T       ║
╚═══════════════════════════════════════════════════════════╝

You: What are the diagnostic Raman peaks for polystyrene?

Nu1lm: Polystyrene has a highly diagnostic peak at 1001 cm⁻¹ from
the aromatic ring breathing mode. Other characteristic peaks:
- 620 cm⁻¹ (ring deformation)
- 795 cm⁻¹
- 1031 cm⁻¹ (C-H in-plane)
- 1602 cm⁻¹ (ring stretch)
- 3054 cm⁻¹ (aromatic C-H)

The 1001 cm⁻¹ peak is particularly useful for identification as
it's sharp, intense, and unique to polystyrene among common plastics.

You: analyze raman my_sample.txt

Nu1lm: [Analyzes file and provides identification]
```

## Citation

If you use Nu1lm in your research, please cite:

```
Nu1lm Microplastics Expert System
https://github.com/[your-repo]/nu1lm
```

## License

MIT
