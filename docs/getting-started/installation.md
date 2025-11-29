# Installation

Get the project set up on your machine in a few steps.

## Prerequisites

- Python 3.9+
- Git
- Conda (Miniconda or Anaconda)
- 10GB+ free disk space for the dataset

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/aclarke/echoes.git
cd echoes
```

### 2. Create Environment

```bash
conda create -n echoes python=3.11 -y
conda activate echoes
pip install -r requirements.txt
```

This installs all dependencies including PyTorch, torchvision, MLflow, TensorBoard, and project utilities.

### 3. Verify Installation

```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torchvision; print(f'torchvision version: {torchvision.__version__}')"
```

Both commands should print version information without errors.

## Dataset Setup

### Option 1: Download Full UCF101 (13GB)

For full benchmarking, download the complete UCF101 dataset:

```bash
python scripts/download_ucf101_full.py /mnt/echoes_data
```

This downloads and extracts to `/mnt/echoes_data/ucf101/`.

### Option 2: Use Existing Dataset

If you already have the dataset, ensure it's in one of these locations:
- `/mnt/echoes_data/ucf101/` (persistent storage)
- `./data/ucf101/` (local directory)

### Validate Dataset

```bash
python scripts/validate_dataset.py /mnt/echoes_data/ucf101
```

This verifies the dataset has all 101 classes and 13,320 videos.

## Code Quality Tools

The project uses several tools to maintain code quality:

### Linting and Formatting

```bash
# Check code style
ruff check .

# Auto-fix style issues
ruff check --fix .

# Format code
ruff format .
```

### Run Tests

```bash
# Unit tests
pytest tests/

# With coverage
pytest --cov=data --cov=scripts tests/

# Integration tests (requires full dataset)
RUN_INTEGRATION_TESTS=1 pytest tests/test_integration.py
```

### Pre-commit Hooks

Install pre-commit hooks to run checks automatically:

```bash
pre-commit run --all-files
```

## Development Workflow

1. Activate the conda environment: `conda activate echoes`
2. Make changes to code
3. Run linters: `ruff check --fix . && ruff format .`
4. Run tests: `pytest tests/`
5. Commit changes: `git commit -m "Your message"`

## Troubleshooting

### PyTorch GPU Not Detected

If you have a GPU but PyTorch isn't using it:

```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Reinstall PyTorch for your system
conda install pytorch::pytorch pytorch::torchvision pytorch::torchaudio -c pytorch
```

### Dataset Download Fails

```bash
# Check internet connection
ping github.com

# Try manual download at https://www.crcv.ucf.edu/datasets/ucf101/
# Then extract to /mnt/echoes_data/ucf101/
```

### Conda Environment Issues

```bash
# Recreate environment from scratch
conda env remove --name echoes
conda create -n echoes python=3.11 -y
conda activate echoes
pip install -r requirements.txt
```

## Next Steps

- [Quick Start](quick-start.md) - Run your first experiment
- [Running Experiments](../guides/experiments.md) - Advanced training options
- [Infrastructure](../architecture/infrastructure.md) - Understand the deployment setup
