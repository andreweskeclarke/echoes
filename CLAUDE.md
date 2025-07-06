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
- **Run unit tests**: `pytest tests/`
- **Run tests with coverage**: `pytest --cov=data --cov=scripts tests/`
- **Run integration tests**: `RUN_INTEGRATION_TESTS=1 pytest tests/test_integration.py`

### Dataset Preparation
- **Download UCF101 dataset**: `python scripts/download_ucf101_full.py /mnt/echoes_data`
- **Validate downloaded dataset**: `python scripts/validate_dataset.py /mnt/echoes_data/ucf101`

## Code Style Guidelines

Based on the Cursor rules in `.cursor/rules/`:
- Use succinct Pythonic code
- Minimal comments - prefer self-documenting code
- Always write unit tests with good coverage
- Run linters after every change
- **Use logging statements not print()**

## Testing Principles
- Prefer dependency injection and objects to using mocks in tests

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
- **Azure Blob Storage**: Cloud storage for datasets and experiment artifacts
- **ruff**: Code formatting and linting
- **pytest**: Unit testing framework

### Model Types to Implement
- Echo State Networks (ESNs) - primary research focus
- Traditional RNNs
- LSTM/GRU networks
- Baseline comparisons

### Storage Architecture
- **Direct Download**: UCF101 dataset downloads directly from source (10 minutes)
- **Local Storage**: Mounted Azure disk at `/mnt/echoes_data` for dataset storage
- **Simple Workflow**: Single script handles download, extraction, and validation

### Development Notes
- The project is in early stages - most model and experiment code needs to be implemented
- Focus on computational efficiency comparisons between ESNs and traditional RNNs
- Include training time, memory usage, and accuracy metrics
- Save models and training artifacts to a `models/` directory (add to .gitignore)
- Use storage abstraction to seamlessly switch between local and cloud storage