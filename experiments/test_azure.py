#!/usr/bin/env python3
"""
Simple test experiment for Azure deployment - runs a minimal ESN training.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader

from data.logging_config import get_logger, setup_logging
from experiments.dataset import UCF101Dataset
from models.simple_models import SimpleESN

logger = get_logger(__name__)


def main():
    setup_logging("INFO")

    # Set MLflow experiment
    mlflow.set_experiment("Azure_Test_ESN")

    # Dataset paths (will be modified by azure runner)
    data_dir = "/mnt/echoes_data/ucf101"
    train_split = f"{data_dir}/splits_01/trainlist01.txt"

    logger.info("Setting up test dataset...")

    # Very small dataset for quick testing (1 class, few samples)
    train_dataset = UCF101Dataset(data_dir, train_split, num_classes=1)

    # Take only first 20 samples for super quick test
    train_dataset.samples = train_dataset.samples[:20]
    val_dataset = UCF101Dataset(data_dir, train_split, num_classes=1)
    val_dataset.samples = val_dataset.samples[20:30]  # Next 10 for validation

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Small ESN for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleESN(input_size=112 * 112 * 3, reservoir_size=100, num_classes=1).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name="azure_test_esn"):
        # Log parameters
        mlflow.log_param("model_type", "SimpleESN")
        mlflow.log_param("reservoir_size", 100)
        mlflow.log_param("device", str(device))
        mlflow.log_param("test_mode", True)

        logger.info(f"Running on device: {device}")
        logger.info("Starting mini training (2 epochs)...")

        # Quick 2-epoch training
        for epoch in range(2):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()

                logger.info(
                    f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {loss.item():.4f}"
                )

            # Validation
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    loss = criterion(output, target)

                    val_loss += loss.item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()

            train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0

            # Log metrics
            mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
            mlflow.log_metric("train_accuracy", train_acc, step=epoch)
            mlflow.log_metric("val_accuracy", val_acc, step=epoch)

            logger.info(
                f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, "
                f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
            )

        # Save model
        mlflow.pytorch.log_model(model, "model")

        logger.info("Test experiment completed successfully!")
        logger.info(f"Final train accuracy: {train_acc:.2f}%")
        logger.info(f"Final validation accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
