# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioVoice is a speaker verification interpretability research project (Shenkar BSc Final Project). It investigates what acoustic concepts drive decisions in speaker verification and spoof detection models using TCAV (Testing with Concept Activation Vectors), GradCAM, and other interpretability techniques.

## Setup

```bash
pip install -r requirements.txt
# torch and torchaudio must be installed manually for your GPU/CUDA version
```

## Running TCAV Pipelines

The primary executable code lives in `redimnet/tcav/captum_tcav/`. Run from `redimnet/tcav/`:

```bash
# VoxCeleb2 speaker identification
python -m captum_tcav.vox2.entrypoint

# ASVspoof5 spoof detection (runs all 16 systems A01-A16)
python -m captum_tcav.asvspoof5.entrypoint
```

There is also a legacy TCAV runner at `redimnet/tcav/tcav_core/` with its own `runner.py`.

Notebooks (`.ipynb`) are the primary development medium for exploration and analysis — run cells sequentially in Jupyter.

## Architecture

### Three Speaker Models

Each model has its own top-level directory with parallel structure (poc/, grad_cam/, tcav/):

- **`redimnet/`** — Primary model. ReDimNet b6 (HuggingFace: `IDRnD/ReDimNet`), 256/512-dim embeddings, trained on VoxCeleb2.
- **`resnet_293/`** — ResNet-293 (WeSpeaker: `voxceleb-resnet293-LM`), 256-dim embeddings.
- **`ecapa/`** — ECAPA-TDNN via SpeechBrain.

### Captum TCAV Pipeline (`redimnet/tcav/captum_tcav/`)

This is the most active codebase. Two task-specific sub-packages share common utilities:

```
captum_tcav/
├── common.py          # Layer resolution, chunking, model utilities
├── concepts.py        # Concept dataset iteration
├── export_csv.py      # CSV output and weighted aggregation
├── config.py          # Delegates to vox2/config.py by default
├── vox2/              # Speaker ID task (VoxCeleb2, 100-speaker subset)
│   ├── config.py      # Hardcoded paths, model params, concept list
│   ├── data.py        # Speaker discovery and audio loading
│   ├── modeling.py    # ReDimNet + speaker classification head
│   └── entrypoint.py  # Main execution script
└── asvspoof5/         # Spoof detection task (16 spoofing systems)
    ├── config.py      # System IDs A01-A16, fixed speaker sets, partition config
    ├── data.py        # Protocol/manifest loading, tar archive indexing
    ├── modeling.py    # ReDimNet + logistic regression head
    └── entrypoint.py  # Main execution script (iterates all systems)
```

### Concept Generation (`concept/redimnet_concepts/`)

`concepts_creation.py` synthesizes 12 mel-spectrogram concepts varying along three axes:
- **Duration**: short (50ms) / long (100ms)
- **Pattern**: constant, rising (steep/flat), dropping (steep/flat)
- **Thickness**: thin (2px) / thick (5px)

Naming convention: `{duration}_{pattern}_{thickness}` (e.g., `long_rising_steep_thick`).

### Shared Preprocessing (`utils/`)

- `Preprocess.py` — Audio loading, mel-spectrogram generation, normalization
- `PreprocessParams.py` — Constants: 16kHz sample rate, 512 FFT, 256 hop, 64 mel bins, 4.5s max duration

## Key Patterns

- **Config as module-level globals**: Config files define values at module scope, then bundle them into a frozen `@dataclass` via `load_config()`. Paths are hardcoded to the server at `/home/SpeakerRec/BioVoice/`.
- **Dual import style**: Modules support both direct execution and package imports via `if __package__ in (None, "")` guards.
- **TCAV flow**: Load model → discover speakers/utterances → build concept datasets (12 concepts + 1 random baseline) → compute CAVs per layer → score and export CSV.
- **Weighted aggregation**: When speakers have more clips than `max_clips_per_chunk`, results are chunked and aggregated by weighted average.
- **Target layer**: TCAV analysis targets `stage4` of ReDimNet by default.
