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


class TrainingState:
    def __init__(  # noqa: PLR0913
        self, model, device, criterion, optimizer, train_loader, val_loader
    ):
        self.model = model
        self.device = device
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader


def setup_dataloaders(data_dir):
    train_split = f"{data_dir}/splits_01/trainlist01.txt"
    train_dataset = UCF101Dataset(data_dir, train_split)
    train_dataset.samples = train_dataset.samples[:20]

    val_dataset = UCF101Dataset(
        data_dir, train_split, class_to_idx=train_dataset.class_to_idx
    )
    val_dataset.samples = val_dataset.samples[20:30]

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    return train_loader, val_loader


def run_training_loop(state):
    for epoch in range(2):
        train_loss, train_acc = run_training_epoch(state, epoch)
        val_loss, val_acc = run_validation_epoch(state, epoch)

        mlflow.log_metric(
            "train_loss", train_loss / len(state.train_loader), step=epoch
        )
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        logger.info(
            f"Epoch {epoch + 1}: Train Loss: "
            f"{train_loss / len(state.train_loader):.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

    return train_acc, val_acc


def run_training_epoch(state, epoch):
    state.model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (batch_data, batch_target) in enumerate(state.train_loader):
        data, target = batch_data.to(state.device), batch_target.to(state.device)

        state.optimizer.zero_grad()
        output = state.model(data)
        loss = state.criterion(output, target)
        loss.backward()
        state.optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        train_total += target.size(0)
        train_correct += predicted.eq(target).sum().item()

        logger.info(
            f"Epoch {epoch + 1}, Batch {batch_idx + 1}: Loss = {loss.item():.4f}"
        )

    train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0
    return train_loss, train_acc


def run_validation_epoch(state, epoch):
    state.model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for batch_data, batch_target in state.val_loader:
            data, target = batch_data.to(state.device), batch_target.to(state.device)
            output = state.model(data)
            loss = state.criterion(output, target)

            val_loss += loss.item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()

    val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0
    return val_loss, val_acc


def main():
    setup_logging("INFO")

    mlflow.set_experiment("Azure_Test_ESN")

    data_dir = "/mnt/echoes_data/ucf101"

    logger.info("Setting up test dataset...")
    train_loader, val_loader = setup_dataloaders(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleESN(input_size=112 * 112 * 3, reservoir_size=100, num_classes=1).to(
        device
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    with mlflow.start_run(run_name="azure_test_esn"):
        mlflow.log_param("model_type", "SimpleESN")
        mlflow.log_param("reservoir_size", 100)
        mlflow.log_param("device", str(device))
        mlflow.log_param("test_mode", True)

        logger.info(f"Running on device: {device}")
        logger.info("Starting mini training (2 epochs)...")

        state = TrainingState(
            model, device, criterion, optimizer, train_loader, val_loader
        )
        train_acc, val_acc = run_training_loop(state)

        mlflow.pytorch.log_model(model, "model")

        logger.info("Test experiment completed successfully!")
        logger.info(f"Final train accuracy: {train_acc:.2f}%")
        logger.info(f"Final validation accuracy: {val_acc:.2f}%")


if __name__ == "__main__":
    main()
