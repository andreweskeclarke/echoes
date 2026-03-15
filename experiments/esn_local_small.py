#!/usr/bin/env python3
"""Small ESN experiment for local GTX 1080 verification.

Runs a minimal ESN training on a subset of UCF101 to verify
LocalRunner end-to-end with MLflow logging.
"""
import sys
import tempfile
import time
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.logging_config import get_logger, setup_logging
from experiments.dataset import UCF101Dataset
from models.simple_models import SimpleESN

logger = get_logger(__name__)

DATA_DIR = "/mnt/echoes_data/ucf101"
TRAIN_SPLIT = f"{DATA_DIR}/splits_01/trainlist01.txt"
TRAIN_SAMPLES = 60
VAL_SAMPLES = 20
BATCH_SIZE = 4
NUM_EPOCHS = 2
RESERVOIR_SIZE = 256
LEARNING_RATE = 0.001


def _train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for batch_inputs, batch_targets in loader:
        inp = batch_inputs.to(device)
        tgt = batch_targets.to(device)
        optimizer.zero_grad()
        out = model(inp)
        loss = criterion(out, tgt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        correct += out.argmax(1).eq(tgt).sum().item()
        total += tgt.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def _val_epoch(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch_inputs, batch_targets in loader:
            inp = batch_inputs.to(device)
            tgt = batch_targets.to(device)
            out = model(inp)
            correct += out.argmax(1).eq(tgt).sum().item()
            total += tgt.size(0)
    return 100.0 * correct / total


def run_experiment() -> dict:
    setup_logging("INFO")

    mlflow.set_tracking_uri("file:///mnt/echoes_data/mlruns")
    mlflow.set_experiment("ESN_LocalRunner_Verification")

    train_dataset = UCF101Dataset(DATA_DIR, TRAIN_SPLIT)
    train_dataset.samples = train_dataset.samples[:TRAIN_SAMPLES]

    val_dataset = UCF101Dataset(
        DATA_DIR, TRAIN_SPLIT, class_to_idx=train_dataset.class_to_idx
    )
    val_dataset.samples = val_dataset.samples[:VAL_SAMPLES]

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.class_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    logger.info(f"Device: {device} (GPU: {gpu_name})")

    model = SimpleESN(
        input_size=112 * 112 * 3,
        reservoir_size=RESERVOIR_SIZE,
        num_classes=num_classes,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    with mlflow.start_run(run_name="SimpleESN_local_small"):
        mlflow.log_params(
            {
                "model": "SimpleESN",
                "reservoir_size": RESERVOIR_SIZE,
                "num_classes": num_classes,
                "train_samples": TRAIN_SAMPLES,
                "val_samples": VAL_SAMPLES,
                "num_epochs": NUM_EPOCHS,
                "batch_size": BATCH_SIZE,
                "device": str(device),
            }
        )

        start = time.time()
        train_acc, val_acc = 0.0, 0.0

        for epoch in range(NUM_EPOCHS):
            avg_loss, train_acc = _train_epoch(
                model, train_loader, optimizer, criterion, device
            )
            val_acc = _val_epoch(model, val_loader, device)

            mlflow.log_metrics(
                {
                    "train_loss": avg_loss,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                },
                step=epoch + 1,
            )
            logger.info(
                f"Epoch {epoch + 1}/{NUM_EPOCHS}: loss={avg_loss:.4f} "
                f"train_acc={train_acc:.1f}% val_acc={val_acc:.1f}%"
            )

        elapsed = time.time() - start
        mlflow.log_metric("training_time_seconds", elapsed)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)
            mlflow.log_artifact(f.name, artifact_path="model")

        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")
        logger.info(f"Training completed in {elapsed:.1f}s")

        return {
            "run_id": run_id,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "elapsed": elapsed,
            "device": str(device),
        }


if __name__ == "__main__":
    results = run_experiment()
    print(f"\nResults: {results}")
