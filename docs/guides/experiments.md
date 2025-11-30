# Running Experiments

Guide to creating and running training experiments.

## Experiment Scripts

Experiments are Python scripts in `experiments/` that train and evaluate models.

### Simple Experiment

`experiments/train_simple.py` - Lightweight baseline for quick testing.

**Features**:
- Trains on 10 classes (small subset)
- Single model
- 5 epochs
- Takes ~5 minutes

**Run**:
```bash
conda activate echoes
python experiments/train_simple.py
```

### Comprehensive Experiment

`experiments/train_comprehensive.py` - Full benchmark with multiple models.

**Features**:
- Trains on all 101 UCF101 classes
- Multiple architectures (SimpleRNN, SimpleESN, DeepESN, LSTM)
- 20+ epochs for each model
- Takes 2-4 hours on GPU

**Run locally** (CPU, not recommended):
```bash
python experiments/train_comprehensive.py
```

**Run on Azure GPU**:
```bash
python scripts/azure_gpu_runner.py experiments/train_comprehensive.py --vm-size Standard_NC6s_v3
```

## Creating Your Own Experiment

### 1. Create Experiment File

`experiments/my_experiment.py`:

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models import SimpleRNN
from data import UCF101DataLoader
import mlflow
import logging

logger = logging.getLogger(__name__)

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)

    # MLflow experiment
    mlflow.set_experiment("My_Experiment")

    with mlflow.start_run():
        # Hyperparameters
        params = {
            "model": "SimpleRNN",
            "hidden_size": 256,
            "batch_size": 32,
            "learning_rate": 0.001,
            "epochs": 10,
            "num_classes": 101,
        }

        # Log parameters
        mlflow.log_params(params)

        # Create model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SimpleRNN(
            input_size=512,
            hidden_size=params["hidden_size"],
            num_classes=params["num_classes"],
        ).to(device)

        # Data loader
        train_loader = UCF101DataLoader(
            split='train',
            batch_size=params["batch_size"],
        )
        val_loader = UCF101DataLoader(
            split='val',
            batch_size=params["batch_size"],
        )

        # Optimizer and loss
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=params["learning_rate"],
        )
        criterion = nn.CrossEntropyLoss()

        # Training loop
        best_val_acc = 0
        for epoch in range(params["epochs"]):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0

            for batch_idx, (videos, labels) in enumerate(train_loader):
                videos, labels = videos.to(device), labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(videos)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Metrics
                train_loss += loss.item()
                _, predictions = torch.max(outputs, 1)
                train_correct += (predictions == labels).sum().item()

            train_loss /= len(train_loader)
            train_acc = train_correct / len(train_loader.dataset)

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0

            with torch.no_grad():
                for videos, labels in val_loader:
                    videos, labels = videos.to(device), labels.to(device)
                    outputs = model(videos)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item()
                    _, predictions = torch.max(outputs, 1)
                    val_correct += (predictions == labels).sum().item()

            val_loss /= len(val_loader)
            val_acc = val_correct / len(val_loader.dataset)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }, step=epoch)

            logger.info(
                f"Epoch {epoch+1}/{params['epochs']} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), "best_model.pth")

        # Log final model
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_metric("best_val_acc", best_val_acc)

        logger.info(f"Training complete. Best validation accuracy: {best_val_acc:.4f}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
```

### 2. Run Your Experiment

**Locally**:
```bash
python experiments/my_experiment.py
```

**On Azure GPU**:
```bash
python scripts/azure_gpu_runner.py experiments/my_experiment.py --vm-size Standard_NC6s_v3
```

### 3. View Results

**MLflow UI**:
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Visit http://localhost:5000
```

**TensorBoard**:
```bash
tensorboard --logdir=./logs --host=0.0.0.0 --port=6006
# Visit http://localhost:6006
```


## Performance Optimization

### GPU Usage

```python
# Check GPU availability
import torch
print(f"GPU available: {torch.cuda.is_available()}")
print(f"GPU device: {torch.cuda.get_device_name(0)}")

# Move model to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
```

### Batch Size

- **Larger batches**: Faster training, but less frequent updates
- **Smaller batches**: Slower training, but more frequent updates
- **Recommended**: 32 for most models

### Data Loading

```python
# Use multiple workers for faster data loading
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # For GPU transfer
)
```

## Next Steps

- [Models](../architecture/models.md) - Understand model architectures
- [Infrastructure](../architecture/infrastructure.md) - How Azure VMs work
