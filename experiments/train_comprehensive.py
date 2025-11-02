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
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from data.logging_config import get_logger, setup_logging
from experiments.dataset import UCF101Dataset
from models.simple_models import DeepESN, DeepRNN, SimpleESN, SimpleRNN

logger = get_logger(__name__)


@dataclass
class TrainingState:
    model: object
    device: torch.device
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    train_loader: DataLoader
    val_loader: DataLoader
    writer: SummaryWriter
    param_counts: dict = None

    def __post_init__(self):
        if self.param_counts is None:
            self.param_counts = count_parameters(self.model) if self.model else {}


def count_parameters(model):
    """Count trainable and total parameters in model"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


def log_model_config(model, lr, num_epochs, device, train_loader, val_loader):  # noqa: PLR0913
    param_counts = count_parameters(model)
    mlflow.log_param("model_type", model.__class__.__name__)
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("num_epochs", num_epochs)
    mlflow.log_param("device", str(device))
    mlflow.log_param("train_samples", len(train_loader.dataset))
    mlflow.log_param("val_samples", len(val_loader.dataset))
    mlflow.log_param("trainable_params", param_counts["trainable_params"])
    mlflow.log_param("total_params", param_counts["total_params"])

    if hasattr(model, "reservoir_size"):
        mlflow.log_param("reservoir_size", model.reservoir_size)
    if hasattr(model, "num_layers"):
        mlflow.log_param("num_layers", model.num_layers)
    if hasattr(model, "rnn"):
        mlflow.log_param("hidden_size", model.rnn.hidden_size)
    if hasattr(model, "hidden_size"):
        mlflow.log_param("hidden_size", model.hidden_size)

    return param_counts


def setup_tensorboard_logging(model, experiment_name, param_counts):
    run_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(f"tfruns/{experiment_name}/{run_name}")
    writer.add_text("Model/Architecture", str(model))
    writer.add_text(
        "Model/Parameters",
        f"Trainable: {param_counts['trainable_params']:,}, "
        f"Total: {param_counts['total_params']:,}",
    )
    return writer


def train_comprehensive_epoch(state, epoch):
    state.model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for batch_idx, (batch_data, batch_target) in enumerate(
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

        global_step = epoch * len(state.train_loader) + batch_idx
        if batch_idx % 10 == 0:
            state.writer.add_scalar("Loss/Train_Batch", loss.item(), global_step)

    train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
    return train_loss, train_acc


def validate_comprehensive_epoch(state):
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

    val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
    return val_loss, val_acc


def train_model(  # noqa: PLR0913
    model, train_loader, val_loader, num_epochs=10, lr=0.001, experiment_name="default"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    param_counts = log_model_config(
        model, lr, num_epochs, device, train_loader, val_loader
    )
    writer = setup_tensorboard_logging(model, experiment_name, param_counts)

    logger.info(f"Training on {device}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Trainable parameters: {param_counts['trainable_params']:,}")
    logger.info(f"Total parameters: {param_counts['total_params']:,}")

    state = TrainingState(
        model, device, criterion, optimizer, train_loader, val_loader, writer
    )
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_comprehensive_epoch(state, epoch)
        val_loss, val_acc = validate_comprehensive_epoch(state)

        avg_train_loss = (
            train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        )
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        logger.info(
            f"Epoch {epoch + 1}: Train Loss: {avg_train_loss:.4f}, "
            f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%"
        )

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")

    mlflow.log_metric("final_train_accuracy", train_acc)
    mlflow.log_metric("final_val_accuracy", val_acc)
    mlflow.log_metric("training_time_seconds", total_time)

    params_per_accuracy = param_counts["trainable_params"] / max(val_acc, 0.1)
    mlflow.log_metric("params_per_accuracy", params_per_accuracy)

    mlflow.pytorch.log_model(model, "model")
    writer.close()

    return {
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "training_time": total_time,
        "trainable_params": param_counts["trainable_params"],
        "total_params": param_counts["total_params"],
        "params_per_accuracy": params_per_accuracy,
    }


def main():
    setup_logging("INFO")

    # Set MLflow experiment
    experiment_name = "UCF101_Architecture_Comparison"
    mlflow.set_experiment(experiment_name)

    # Dataset paths
    data_dir = "/mnt/echoes_data/ucf101"
    train_split = f"{data_dir}/splits_01/trainlist01.txt"
    test_split = f"{data_dir}/splits_01/testlist01.txt"

    # Create datasets (all 101 classes)
    train_dataset = UCF101Dataset(data_dir, train_split)

    # Use test split for validation with SAME classes as training
    val_dataset = UCF101Dataset(
        data_dir, test_split, class_to_idx=train_dataset.class_to_idx
    )
    # Take subset for faster validation
    # val_dataset.samples = val_dataset.samples[:150]  # ~10 samples per class

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.class_to_idx)}")

    # Log class distribution
    logger.info("Classes used:")
    for class_name, idx in train_dataset.class_to_idx.items():
        train_count = sum(1 for _, label in train_dataset.samples if label == idx)
        val_count = sum(1 for _, label in val_dataset.samples if label == idx)
        logger.info(f"  {idx}: {class_name} - Train: {train_count}, Val: {val_count}")

    # Model configurations to test
    input_size = 112 * 112 * 3  # Flattened video frame
    num_classes = len(train_dataset.class_to_idx)

    model_configs = [
        # # Simple RNN configurations
        # {"type": "RNN", "hidden_size": 32, "num_layers": 1},
        # {"type": "RNN", "hidden_size": 64, "num_layers": 1},
        # {"type": "RNN", "hidden_size": 128, "num_layers": 1},
        # {"type": "RNN", "hidden_size": 256, "num_layers": 1},
        #
        # # Deep RNN configurations
        # {"type": "DeepRNN", "hidden_size": 64, "num_layers": 2},
        # {"type": "DeepRNN", "hidden_size": 64, "num_layers": 3},
        # {"type": "DeepRNN", "hidden_size": 128, "num_layers": 2},
        # {"type": "DeepRNN", "hidden_size": 128, "num_layers": 3},
        # Simple ESN configurations
        {"type": "ESN", "reservoir_size": 250},
        {"type": "ESN", "reservoir_size": 500},
        {"type": "ESN", "reservoir_size": 1000},
        {"type": "ESN", "reservoir_size": 2000},
        # Deep ESN configurations
        {"type": "DeepESN", "reservoir_size": 500, "num_layers": 2},
        {"type": "DeepESN", "reservoir_size": 500, "num_layers": 3},
        {"type": "DeepESN", "reservoir_size": 1000, "num_layers": 2},
        {"type": "DeepESN", "reservoir_size": 1000, "num_layers": 3},
    ]

    results = {}

    for config in model_configs:
        # Create model based on configuration
        if config["type"] == "RNN":
            model = SimpleRNN(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_classes=num_classes,
                num_layers=config["num_layers"],
            )
            model_name = f"RNN_h{config['hidden_size']}_L{config['num_layers']}"
        elif config["type"] == "DeepRNN":
            model = DeepRNN(
                input_size=input_size,
                hidden_size=config["hidden_size"],
                num_layers=config["num_layers"],
                num_classes=num_classes,
            )
            model_name = f"DeepRNN_h{config['hidden_size']}_L{config['num_layers']}"
        elif config["type"] == "ESN":
            model = SimpleESN(
                input_size=input_size,
                reservoir_size=config["reservoir_size"],
                num_classes=num_classes,
            )
            model_name = f"ESN_r{config['reservoir_size']}"
        else:  # DeepESN
            model = DeepESN(
                input_size=input_size,
                reservoir_size=config["reservoir_size"],
                num_layers=config["num_layers"],
                num_classes=num_classes,
            )
            model_name = f"DeepESN_r{config['reservoir_size']}_L{config['num_layers']}"

        with mlflow.start_run(run_name=f"{model_name}_experiment"):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'=' * 60}")

            result = train_model(
                model,
                train_loader,
                val_loader,
                num_epochs=10,
                lr=0.001,
                experiment_name=experiment_name,
            )
            results[model_name] = result

    # Print comprehensive comparison
    logger.info(f"\n{'=' * 80}")
    logger.info("COMPREHENSIVE RESULTS COMPARISON")
    logger.info(f"{'=' * 80}")

    # Sort by validation accuracy
    sorted_results = sorted(
        results.items(), key=lambda x: x[1]["final_val_acc"], reverse=True
    )

    logger.info(
        f"{'Model':<15} {'Val Acc':<8} {'Params':<12} {'Time':<8} {'Efficiency':<12}"
    )
    logger.info(f"{'-' * 70}")

    for model_name, result in sorted_results:
        efficiency = result["params_per_accuracy"]
        logger.info(
            f"{model_name:<15} {result['final_val_acc']:<8.2f} "
            f"{result['trainable_params']:<12,} {result['training_time']:<8.1f} "
            f"{efficiency:<12.0f}"
        )

    # Best model summary
    best_model = sorted_results[0]
    logger.info(
        f"\nBest Model: {best_model[0]} with "
        f"{best_model[1]['final_val_acc']:.2f}% validation accuracy"
    )
    logger.info(f"Training time: {best_model[1]['training_time']:.1f}s")
    logger.info(f"Parameters: {best_model[1]['trainable_params']:,}")


if __name__ == "__main__":
    main()
