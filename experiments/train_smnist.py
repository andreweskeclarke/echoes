#!/usr/bin/env python3
"""Train and compare ESN vs RNN on sMNIST and psMNIST.

seq_len=28, input_size=28 — pixels fed one row at a time.
Standard benchmark for evaluating long-range temporal dependencies.
"""

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
from data.smnist import SequentialMNIST, make_permutation
from models.simple_models import SimpleESN, SimpleRNN

logger = get_logger(__name__)

SEQ_LEN = 28
INPUT_SIZE = 28
NUM_CLASSES = 10


@dataclass
class TrainingState:
    model: nn.Module
    device: torch.device
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    val_loader: DataLoader


def train_epoch(state: TrainingState, epoch: int) -> tuple[float, float]:
    state.model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for batch_data, batch_target in tqdm(state.train_loader, desc=f"Epoch {epoch + 1}"):
        data, target = batch_data.to(state.device), batch_target.to(state.device)

        state.optimizer.zero_grad()
        output = state.model(data)
        loss = state.criterion(output, target)
        loss.backward()
        state.optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    return total_loss, 100.0 * correct / total


def validate_epoch(state: TrainingState) -> tuple[float, float]:
    state.model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_data, batch_target in state.val_loader:
            data, target = batch_data.to(state.device), batch_target.to(state.device)
            output = state.model(data)
            loss = state.criterion(output, target)

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return total_loss, 100.0 * correct / total


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 10,
    lr: float = 0.001,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

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

    logger.info(f"Training {model.__class__.__name__} on {device}")

    state = TrainingState(model, device, criterion, optimizer, train_loader, val_loader)

    _, initial_val_acc = validate_epoch(state)
    mlflow.log_metric("val_accuracy", initial_val_acc, step=0)
    logger.info(f"Initial val accuracy: {initial_val_acc:.2f}%")

    start_time = time.time()
    train_acc = val_acc = 0.0

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(state, epoch)
        _, val_acc = validate_epoch(state)

        mlflow.log_metric("train_loss", train_loss / len(train_loader), step=epoch + 1)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch + 1)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch + 1)

        logger.info(
            f"Epoch {epoch + 1}/{num_epochs}: "
            f"loss={train_loss / len(train_loader):.4f} "
            f"train={train_acc:.2f}% val={val_acc:.2f}%"
        )

    training_time = time.time() - start_time
    mlflow.log_metric("final_train_accuracy", train_acc)
    mlflow.log_metric("final_val_accuracy", val_acc)
    mlflow.log_metric("training_time_seconds", training_time)
    mlflow.pytorch.log_model(model, "model")

    return {
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "training_time": training_time,
    }


def run_benchmark(
    dataset_name: str,
    data_root: str,
    permutation: torch.Tensor | None,
    num_epochs: int,
    batch_size: int,
) -> dict:
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Benchmark: {dataset_name}")
    logger.info(f"{'=' * 60}")

    train_dataset = SequentialMNIST(data_root, train=True, permutation=permutation)
    test_dataset = SequentialMNIST(data_root, train=False, permutation=permutation)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    logger.info(f"Train: {len(train_dataset)}, Test: {len(test_dataset)}")

    model_configs = [
        SimpleRNN(
            input_size=INPUT_SIZE,
            hidden_size=128,
            num_classes=NUM_CLASSES,
            num_layers=1,
        ),
        SimpleESN(
            input_size=INPUT_SIZE,
            reservoir_size=1000,
            num_classes=NUM_CLASSES,
            spectral_radius=0.9,
        ),
    ]

    results = {}
    for model in model_configs:
        run_name = f"{dataset_name}_{model.__class__.__name__}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("dataset", dataset_name)
            mlflow.log_param("seq_len", SEQ_LEN)
            mlflow.log_param("input_size", INPUT_SIZE)
            mlflow.log_param("permuted", permutation is not None)

            logger.info(f"\nTraining {model.__class__.__name__}...")
            result = train_model(model, train_loader, val_loader, num_epochs=num_epochs)
            results[run_name] = result

    return results


def main():
    setup_logging("INFO")

    data_root = "/tmp/mnist_data"
    num_epochs = 10
    batch_size = 128

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mlflow.set_experiment(f"sMNIST_ESN_vs_RNN_{timestamp}")

    all_results = {}

    # sMNIST: natural row-by-row order
    smnist_results = run_benchmark(
        dataset_name="sMNIST",
        data_root=data_root,
        permutation=None,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    all_results.update(smnist_results)

    # psMNIST: fixed random pixel permutation
    permutation = make_permutation(seed=42)
    psmnist_results = run_benchmark(
        dataset_name="psMNIST",
        data_root=data_root,
        permutation=permutation,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )
    all_results.update(psmnist_results)

    logger.info(f"\n{'=' * 60}")
    logger.info("FINAL RESULTS")
    logger.info(f"{'=' * 60}")
    for run_name, result in all_results.items():
        logger.info(
            f"{run_name}: train={result['final_train_acc']:.2f}% "
            f"val={result['final_val_acc']:.2f}% "
            f"time={result['training_time']:.1f}s"
        )


if __name__ == "__main__":
    main()
