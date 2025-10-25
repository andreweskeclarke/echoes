# Quick Start

Get your first experiment running in 5 minutes.

## Prerequisites

- Environment set up ([Installation](installation.md))
- Conda environment activated: `conda activate echoes`

## Running a Local Test

### 1. Run the Simple Training Script

```bash
python experiments/train_simple.py
```

This trains a lightweight model on a subset of UCF101 for quick testing. It should complete in a few minutes.

## What Happens

The script will:
1. Load a small subset of UCF101 (10 classes, ~100 videos)
2. Train a SimpleRNN model for 5 epochs
3. Log metrics to MLflow
4. Save training logs to TensorBoard
5. Display final accuracy

Output will look like:
```
Epoch 1/5: Loss 2.145, Acc 0.234
Epoch 2/5: Loss 1.892, Acc 0.342
...
Final Validation Accuracy: 0.456
```

## View Results

### MLflow UI

```bash
mlflow ui --host 0.0.0.0 --port 5000
```

Then visit `http://localhost:5000` to see:
- Experiment metrics
- Training curves
- Model parameters
- Run comparisons

### TensorBoard

```bash
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
```

Then visit `http://localhost:6006` to see:
- Training loss curves
- Validation metrics
- Histograms of weights
- Model graph

## Run a GPU Experiment on Azure

For serious experiments with GPU acceleration:

```bash
# Spin up a GPU VM and run comprehensive training
python scripts/azure_gpu_runner.py experiments/train_comprehensive.py --vm-size Standard_NC6s_v3
```

This will:
1. Create an Azure GPU VM
2. SSH and download the dataset
3. Run the comprehensive experiment
4. Stream results back to your local machine
5. Shut down the VM (saves money!)

See [Running Experiments](../guides/experiments.md) for more details.

## Experiment Structure

Experiments are Python scripts in `experiments/`:

```python
# experiments/my_experiment.py
import torch
from models import SimpleRNN
from data import UCF101DataLoader

def main():
    # Load data
    train_loader = UCF101DataLoader(split='train')

    # Create model
    model = SimpleRNN(input_size=512, hidden_size=256, num_classes=101)

    # Train and log with MLflow
    for epoch in range(10):
        # Your training code here
        pass

if __name__ == "__main__":
    main()
```


## Next Steps

- [Running Experiments](../guides/experiments.md) - Advanced training configurations
- [Models](../architecture/models.md) - Understand architecture implementations
