# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses conda for environment management. Always use the conda environment:

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate echoes
```

## Development Commands

### Code Quality and Testing
- **Lint code**: `ruff check .` (with auto-fix: `ruff check --fix .`)
- **Format code**: `ruff format .`
- **Run pre-commit hooks**: `pre-commit run --all-files`

### Dataset Preparation
- **Download UCF101 dataset**: `python scripts/prepare_ucf101.py --data-dir data/ucf101`

## Code Style Guidelines

Based on the Cursor rules in `.cursor/rules/`:
- Use succinct Pythonic code
- Minimal comments - prefer self-documenting code
- Always write unit tests with good coverage
- Run linters after every change

## Project Architecture

This is a PyTorch-based research project comparing Echo State Networks (ESNs) with traditional RNNs for video classification using the UCF101 dataset.

### Planned Directory Structure
```
echoes/
├── data/               # UCF101 dataset and preprocessing
├── models/            # Model implementations (ESN, RNN, LSTM, GRU)
├── experiments/       # Training and evaluation scripts
├── notebooks/         # Analysis and visualization notebooks
└── scripts/          # Utility scripts (dataset prep, etc.)
```

### Key Technologies
- **PyTorch**: Primary ML framework (≥2.0.0)
- **torchvision**: Video data loading and transforms
- **UCF101**: Video classification dataset (13,320 clips, 101 classes)
- **ruff**: Code formatting and linting

### Model Types to Implement
- Echo State Networks (ESNs) - primary research focus
- Traditional RNNs
- LSTM/GRU networks
- Baseline comparisons

### Development Notes
- The project is in early stages - most model and experiment code needs to be implemented
- Focus on computational efficiency comparisons between ESNs and traditional RNNs
- Include training time, memory usage, and accuracy metrics
- Save models and training artifacts to a `models/` directory (add to .gitignore)