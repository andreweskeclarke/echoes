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
from models.factory import create_model_from_config

logger = get_logger(__name__)


@dataclass
class TrainingState:
    model: object
    device: torch.device
    criterion: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler.LRScheduler
    train_loader: DataLoader
    val_loader: DataLoader
    writer: SummaryWriter
    learning_rate: float = 0.001
    num_epochs: int = 10
    param_counts: dict = None

    def __post_init__(self):
        if self.param_counts is None:
            self.param_counts = count_parameters(self.model) if self.model else {}

    def log_model_config(self):
        mlflow.log_param("model_type", self.model.__class__.__name__)
        mlflow.log_param("learning_rate", self.learning_rate)
        mlflow.log_param("num_epochs", self.num_epochs)
        mlflow.log_param("device", str(self.device))
        mlflow.log_param("train_samples", len(self.train_loader.dataset))
        mlflow.log_param("val_samples", len(self.val_loader.dataset))
        mlflow.log_param("trainable_params", self.param_counts["trainable_params"])
        mlflow.log_param("total_params", self.param_counts["total_params"])

        if hasattr(self.model, "reservoir_size"):
            mlflow.log_param("reservoir_size", self.model.reservoir_size)
        if hasattr(self.model, "num_layers"):
            mlflow.log_param("num_layers", self.model.num_layers)
        if hasattr(self.model, "rnn"):
            mlflow.log_param("hidden_size", self.model.rnn.hidden_size)
        if hasattr(self.model, "hidden_size"):
            mlflow.log_param("hidden_size", self.model.hidden_size)

    def log_train_epoch(self, epoch, train_loss, train_acc):
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Accuracy/Train", train_acc, epoch)

    def log_validation_epoch(self, epoch, val_loss, val_acc):
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)


def count_parameters(model):
    """Count trainable and total parameters in model"""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return {"trainable_params": trainable, "total_params": total}


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


def train_comprehensive_epoch(state: TrainingState, epoch: int):
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


def validate_comprehensive_epoch(state: TrainingState):
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


def train_model(model, train_loader, val_loader, **kwargs):
    num_epochs = kwargs.get("num_epochs", 10)
    lr = kwargs.get("lr", 0.001)
    experiment_name = kwargs.get("experiment_name", "default")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    param_counts = count_parameters(model)
    writer = setup_tensorboard_logging(model, experiment_name, param_counts)

    state = TrainingState(
        model=model,
        device=device,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        writer=writer,
        learning_rate=lr,
        num_epochs=num_epochs,
    )

    state.log_model_config()

    logger.info(f"Training on {device}")
    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Trainable parameters: {state.param_counts['trainable_params']:,}")
    logger.info(f"Total parameters: {state.param_counts['total_params']:,}")
    start_time = time.time()

    for epoch in range(num_epochs):
        train_loss, train_acc = train_comprehensive_epoch(state, epoch)
        val_loss, val_acc = validate_comprehensive_epoch(state)

        avg_train_loss = (
            train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        )
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

        state.log_train_epoch(epoch, avg_train_loss, train_acc)
        state.log_validation_epoch(epoch, avg_val_loss, val_acc)

        current_lr = state.scheduler.get_last_lr()[0]
        mlflow.log_metric("learning_rate", current_lr, step=epoch)
        state.writer.add_scalar("LearningRate", current_lr, epoch)

        logger.info(
            f"Epoch {epoch + 1}: Loss: {avg_train_loss:.4f}, "
            f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%, LR: {current_lr:.6f}"
        )

        state.scheduler.step()

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time:.2f}s")

    mlflow.log_metric("final_train_accuracy", train_acc)
    mlflow.log_metric("final_val_accuracy", val_acc)
    mlflow.log_metric("training_time_seconds", total_time)

    params_per_accuracy = state.param_counts["trainable_params"] / max(val_acc, 0.1)
    mlflow.log_metric("params_per_accuracy", params_per_accuracy)

    mlflow.pytorch.log_model(model, "model")
    state.writer.close()

    return {
        "final_train_acc": train_acc,
        "final_val_acc": val_acc,
        "training_time": total_time,
        "trainable_params": state.param_counts["trainable_params"],
        "total_params": state.param_counts["total_params"],
        "params_per_accuracy": params_per_accuracy,
    }


def main():
    setup_logging("INFO")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"UCF101_Architecture_Comparison_{timestamp}"
    mlflow.set_experiment(experiment_name)

    data_dir = "/mnt/echoes_data/ucf101"
    train_split = f"{data_dir}/splits_01/trainlist01.txt"
    test_split = f"{data_dir}/splits_01/testlist01.txt"

    train_dataset = UCF101Dataset(
        data_dir,
        train_split,
    )
    val_dataset = UCF101Dataset(
        data_dir,
        test_split,
        class_to_idx=train_dataset.class_to_idx,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=2,
    )

    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    logger.info(f"Number of classes: {len(train_dataset.class_to_idx)}")
    logger.info("Classes used:")
    for class_name, idx in train_dataset.class_to_idx.items():
        train_count = sum(1 for _, label in train_dataset.samples if label == idx)
        val_count = sum(1 for _, label in val_dataset.samples if label == idx)
        logger.info(f"  {idx}: {class_name} - Train: {train_count}, Val: {val_count}")

    # Model configurations to test
    input_size = 112 * 112 * 3  # Flattened video frame
    num_classes = len(train_dataset.class_to_idx)

    model_configs = [
        {"type": "RNN", "hidden_size": 256, "num_layers": 1, "lr": 0.01},
        {"type": "RNN", "hidden_size": 512, "num_layers": 1, "lr": 0.01},
        {"type": "DeepRNN", "hidden_size": 256, "num_layers": 2, "lr": 0.01},
        {"type": "DeepRNN", "hidden_size": 256, "num_layers": 3, "lr": 0.01},
        {"type": "DeepRNN", "hidden_size": 512, "num_layers": 2, "lr": 0.01},
        {"type": "ESN", "reservoir_size": 2000, "lr": 0.01},
        {"type": "ESN", "reservoir_size": 5000, "lr": 0.01},
        {"type": "DeepESN", "reservoir_size": 2000, "num_layers": 2, "lr": 0.01},
        {"type": "DeepESN", "reservoir_size": 2000, "num_layers": 3, "lr": 0.01},
        {"type": "DeepESN", "reservoir_size": 5000, "num_layers": 2, "lr": 0.01},
    ]

    results = {}

    for config in model_configs:
        model, model_name = create_model_from_config(config, input_size, num_classes)

        with mlflow.start_run(run_name=f"{model_name}_experiment"):
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Training {model_name}")
            logger.info(f"{'=' * 60}")

            result = train_model(
                model,
                train_loader,
                val_loader,
                num_epochs=25,
                lr=config.get("lr", 0.01),
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
