# Echoes: Exploring Echo State Networks

Echoes is a research project comparing **Echo State Networks (ESNs)** with traditional Recurrent Neural Networks (RNNs) for video classification tasks using PyTorch and the UCF101 dataset.

## Project Goals

- Implement and compare different neural network architectures:
  - Echo State Networks (ESNs)
  - Traditional RNNs
  - LSTM/GRU networks
  - Other relevant architectures
- Evaluate performance on video classification tasks
- Analyze computational efficiency and training dynamics
- Document best practices and findings

## Quick Links

- **GitHub**: [aclarke/echoes](https://github.com/aclarke/echoes)
- **MLflow**: [mlflow.lonel.ai](https://mlflow.lonel.ai)
- **TensorBoard**: [tensorboard.lonel.ai](https://tensorboard.lonel.ai)
- **Dashboard**: [dashboard.lonel.ai](https://dashboard.lonel.ai)

## Getting Started

New to the project? Start with [Installation](getting-started/installation.md) to set up your environment, then check out [Quick Start](getting-started/quick-start.md) to run your first experiment.

## Key Features

### Flexible Architecture Comparisons
Train and compare multiple network architectures with configurable parameters and automatic tracking.

### Persistent Storage
Experiments are tracked with MLflow, logs saved with TensorBoard, and model artifacts stored on persistent Azure disk.

### Cloud Integration
Run GPU experiments on Azure VMs with automatic provisioning, data transfer, and cleanup.

### Interactive Visualization
View model architectures, training metrics, and comparisons through the web dashboard.

## Project Structure

```
echoes/
├── data/               # Dataset preparation and loading
├── models/             # Model implementations (ESN, RNN, LSTM, etc.)
├── experiments/        # Training and evaluation scripts
├── scripts/            # Utility scripts (Azure automation, deployment, etc.)
├── dashboard/          # Web dashboard for visualization
├── docs/               # Documentation (this site)
└── README.md          # Quick reference
```

## Dataset

We use the **UCF101** dataset: 13,320 video clips across 101 action categories. The full dataset is stored on persistent Azure disk at `/mnt/echoes_data/ucf101/`.

## Next Steps

- [Installation Guide](getting-started/installation.md) - Set up your environment
- [Infrastructure Overview](architecture/infrastructure.md) - Understand the deployment setup
- [Running Experiments](guides/experiments.md) - Train and evaluate models
- [Deployment Guide](guides/deployment.md) - Deploy to production
