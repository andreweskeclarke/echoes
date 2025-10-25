# Echoes: Exploring Echo State Networks

This repository contains experiments comparing Echo State Networks (ESNs) with traditional Recurrent Neural Networks (RNNs) and other architectures using PyTorch. The project focuses on video classification using the UCF101 dataset.

## Project Goals

- Implement and compare different neural network architectures:
  - Echo State Networks (ESNs)
  - Traditional RNNs
  - LSTM/GRU networks
  - Other relevant architectures
- Evaluate performance on video classification tasks
- Analyze computational efficiency and training dynamics
- Document best practices and findings

## Project Structure

```
echoes/
├── data/               # Data storage and preprocessing
├── models/            # Model implementations
│   ├── esn.py        # Echo State Network implementation
│   ├── rnn.py        # Traditional RNN implementation
│   └── utils.py      # Shared model utilities
├── experiments/       # Training and evaluation scripts
├── notebooks/        # Jupyter notebooks for analysis
├── environment.yml   # Conda environment specification
└── README.md        # This file
```

## Setup

1. Create and activate the Conda environment:
```bash
# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate echoes
```

2. Verify the installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
```

## Infrastructure

### Headquarters VM
The project runs on a persistent Azure VM (`headquarters`) with:
- **OS Disk**: Code repository, conda environments (29GB, ephemeral)
- **Data Disk**: UCF101 dataset, experiment results, MLflow runs (118GB, persistent)
- **Mount Point**: `/mnt/echoes_data/` for all persistent data
- **Symlinks**: `logs/`, `mlruns/`, `tfruns/` in code directory link to mounted disk

### Experiment VMs
GPU experiments run on ephemeral Azure VMs that:
- Automatically spin up with specified GPU configuration
- Copy code and dataset from headquarters VM
- Run experiments and store results on mounted data disk
- Auto-destroy after configurable hours to prevent charges
- Stream results back to headquarters persistent storage

### Data Organization
```bash
/home/aclarke/echoes/          # Code (ephemeral)
├── experiments/               # Training scripts
├── models/                    # Architecture implementations  
├── scripts/                   # Automation and utilities
└── logs -> /mnt/echoes_data/logs    # Symlinked to persistent disk

/mnt/echoes_data/              # Persistent data disk
├── ucf101/                    # UCF101 dataset (13GB)
├── logs/                      # Training logs
├── mlruns/                    # MLflow experiment tracking
├── tfruns/                    # TensorBoard runs
└── azure_results/             # Downloaded experiment results
```

## Dataset

The UCF101 dataset (13,320 video clips, 101 action categories) is stored on the persistent data disk. Use the download script to set it up:

```bash
python scripts/download_ucf101_full.py /mnt/echoes_data
```

## Running Experiments

### Local Development
```bash
# Quick local test with small dataset
conda activate echoes
python experiments/train_simple.py
```

### GPU Experiments on Azure
```bash
# Spin up GPU VM and run experiment
python scripts/azure_gpu_runner.py experiments/train_comprehensive.py --vm-size Standard_NC6s_v3

# Test with cheap CPU VM first
python scripts/azure_gpu_runner.py experiments/test_azure.py --vm-size Standard_B1s
```

The Azure runner automatically handles VM lifecycle, environment setup, data transfer, and result collection.

## License

MIT License
