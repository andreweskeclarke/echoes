#!/usr/bin/env python3
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import time
from datetime import datetime

import mlflow
import mlflow.pytorch
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.logging_config import get_logger, setup_logging
from experiments.dataset import UCF101Dataset
from models.simple_models import SimpleESN, SimpleRNN

logger = get_logger(__name__)


@dataclass
class TrainingState:
    model: object
    device: torch.device
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    val_loader: DataLoader


def train_epoch(state, epoch):
    state.model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for _batch_idx, (batch_data, batch_target) in enumerate(
        tqdm(state.train_loader, desc=f"Epoch {epoch + 1}")
    ):
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

    train_acc = 100.0 * train_correct / train_total
    return train_loss, train_acc


def validate_epoch(state):
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

    val_acc = 100.0 * val_correct / val_total
    return val_loss, val_acc


def train_model(model, train_loader, val_loader, num_epochs=5, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("device", str(device))
    mlflow.log_param("train_samples", len(train_loader.dataset))
    mlflow.log_param("val_samples", len(val_loader.dataset))

    if hasattr(model, "reservoir_size"):
        mlflow.log_param("reservoir_size", model.reservoir_size)
    if hasattr(model, "rnn"):
        mlflow.log_param("hidden_size", model.rnn.hidden_size)

    logger.info(f"Training on {device}")
    logger.info(f"Model: {model.__class__.__name__}")

    state = TrainingState(model, device, criterion, optimizer, train_loader, val_loader)
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(state, epoch)
        val_loss, val_acc = validate_epoch(state)

        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {train_loss / len(train_loader):.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")

    mlflow.log_metric("final_train_accuracy", train_acc)
    mlflow.log_metric("final_val_accuracy", val_acc)
    mlflow.log_metric("training_time_seconds", total_time)

    mlflow.pytorch.log_model(model, "model")

    return {
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "training_time": total_time,
    }


def main():
    setup_logging("INFO")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(f"UCF101_RNN_vs_ESN_{timestamp}")

    # Dataset paths
    data_dir = "/mnt/echoes_data/ucf101"
    train_split = f"{data_dir}/splits_01/trainlist01.txt"

    # Create datasets
    train_dataset = UCF101Dataset(data_dir, train_split)

    # For validation, reuse some training data (quick test)
    val_dataset = UCF101Dataset(
        data_dir, train_split, class_to_idx=train_dataset.class_to_idx
    )
    # Take only subset for validation
    val_dataset.samples = val_dataset.samples[:100]

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Train models
    num_classes = len(train_dataset.class_to_idx)
    models = [
        SimpleRNN(input_size=112 * 112 * 3, hidden_size=64, num_classes=num_classes),
        SimpleESN(
            input_size=112 * 112 * 3, reservoir_size=500, num_classes=num_classes
        ),
    ]

    results = {}
    for model in models:
        with mlflow.start_run(run_name=f"{model.__class__.__name__}_experiment"):
            logger.info(f"\n{'=' * 50}")
            logger.info(f"Training {model.__class__.__name__}")
            logger.info(f"{'=' * 50}")

            result = train_model(model, train_loader, val_loader, num_epochs=3)
            results[model.__class__.__name__] = result

    # Print comparison
    logger.info(f"\n{'=' * 50}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 50}")

    for model_name, result in results.items():
        logger.info(f"{model_name}:")
        logger.info(f"  Train Acc: {result['final_train_acc']:.2f}%")
        logger.info(f"  Val Acc: {result['final_val_acc']:.2f}%")
        logger.info(f"  Time: {result['training_time']:.2f}s")


if __name__ == "__main__":
    main()
