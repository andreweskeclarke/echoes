# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

This project uses conda for environment management. Always use the conda environment:

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate echoes
```

## Task Workflow

When working on any task, follow this workflow:

1. **Track on Linear**: Ensure the task is tracked in Linear (https://linear.app/aclarke/team/ACL/active)
2. **Mark In Progress**: Move the task to "in progress" in Linear
3. **Read Related Docs**: Review relevant documentation in `docs/` to understand context and requirements
4. **Complete Task**: Implement the task
5. **Run Checks**: Execute all quality checks:
   - `ruff check --fix .` (auto-fix linting issues)
   - `ruff format .` (format code)
   - `pre-commit run --all-files` (run pre-commit hooks)
   - `pytest tests/` (run unit tests)
6. **Review Documentation**: Check if docs in `docs/` need updating to reflect changes
7. **Commit and Push**: Commit changes and push to main
8. **Close Linear Task**: Move the Linear task to "Done"

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

### Self-Documenting Code is Mandatory

**Comments should be rare and only for unusual code.** Code clarity should come from:
- Descriptive variable and function names
- Clear logic flow and structure
- Type hints (especially return types)
- Docstrings for public APIs (functions, classes, modules) - explain *why*, not *what*

Bad:
```python
# Increment counter
c += 1  # Loop counter

# Check if valid
if x > 0:
    pass
```

Good:
```python
processed_count += 1

if input_value > 0:
    validate_threshold()
```

### Other Guidelines

- Use succinct, idiomatic Pythonic code
- Always write unit tests with good coverage
- Run linters after every change
- **Use logging statements, never print()**
- Remove all trailing whitespace

## Testing Principles
- Prefer dependency injection and objects to using mocks in tests

## Project Architecture

This is a PyTorch-based research project comparing Echo State Networks (ESNs) with traditional RNNs for video classification using the UCF101 dataset.

### Directory Structure
```
echoes/
├── data/               # Data loading and preprocessing
├── models/             # Model implementations (ESN, RNN, LSTM, GRU)
├── experiments/        # Training and evaluation scripts
├── scripts/            # Utility scripts (dataset prep, deployment, etc.)
├── dashboard/          # Web dashboard for model visualization
├── docs/               # MkDocs documentation
├── tests/              # Unit and integration tests
├── logs/               # Training logs (symlink to /mnt/echoes_data/logs)
├── mlruns/             # MLflow experiment tracking (symlink to /mnt/echoes_data/mlruns)
├── tfruns/             # TensorBoard runs (symlink to /mnt/echoes_data/tfruns)
└── site/               # Generated MkDocs site (ignored in git)
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

# Planning

We use Linear for task tracking and planning: https://linear.app/aclarke/team/ACL/active

@README.md
