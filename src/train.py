"""
Training script for Cats vs Dogs CNN classifier.
Logs experiments to MLflow, saves model artifacts.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import mlflow

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import set_seed, get_device, setup_logging, NormalizedImageDataset
from src.models.cnn_model import CNNBaseline
from experiments.mlflow_setup import (
    configure_mlflow,
    log_confusion_matrix,
    log_loss_curves,
    log_classification_report,
)

logger = setup_logging()


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def compute_accuracy(outputs, labels):
    _, preds = torch.max(outputs, 1)
    return (preds == labels).float().mean().item()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, running_acc = 0.0, 0.0
    for inputs, labels in tqdm(loader, desc="Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        running_acc += compute_accuracy(outputs, labels) * inputs.size(0)

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss, running_acc = 0.0, 0.0
    all_preds, all_labels = [], []

    for inputs, labels in tqdm(loader, desc="Validation", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_acc += compute_accuracy(outputs, labels) * inputs.size(0)

        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_acc / len(loader.dataset)
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--experiment-name", type=str, default="cats-vs-dogs-classification")
    args = parser.parse_args()

    params = load_params()
    train_cfg = params["train"]

    # CLI overrides
    batch_size = args.batch_size or train_cfg["batch_size"]
    epochs = args.epochs or train_cfg["epochs"]
    lr = args.lr or train_cfg["learning_rate"]

    set_seed(train_cfg["seed"])
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load preprocessed uint8 tensors; NormalizedImageDataset normalizes
    # per-sample so only one batch is in float32 RAM at a time (avoids OOM
    # when the full float32 train set would be ~12 GB).
    train_data = torch.load("data/processed/train_tensors.pt")
    val_data = torch.load("data/processed/val_tensors.pt")

    train_dataset = NormalizedImageDataset(train_data["images"], train_data["labels"])
    val_dataset = NormalizedImageDataset(val_data["images"], val_data["labels"])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = CNNBaseline(
        num_classes=train_cfg["num_classes"],
        dropout=train_cfg["dropout"],
    ).to(device)
    logger.info(f"Model: {model.__class__.__name__}")
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable params: {n_trainable:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=train_cfg["weight_decay"],
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # MLflow setup
    configure_mlflow(experiment_name=args.experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")

        # Log params
        mlflow.log_params({
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": lr,
            "weight_decay": train_cfg["weight_decay"],
            "dropout": train_cfg["dropout"],
            "model": model.__class__.__name__,
            "optimizer": optimizer.__class__.__name__,
            "scheduler": scheduler.__class__.__name__,
            "seed": train_cfg["seed"],
        })

        # Training loop
        best_val_acc = 0.0
        patience_counter = 0
        early_stop = train_cfg["early_stopping_patience"]

        train_losses, val_losses = [], []
        train_accs, val_accs = [], []

        for epoch in range(1, epochs + 1):
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc, val_preds, val_labels = validate(
                model, val_loader, criterion, device
            )
            scheduler.step()

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_accs.append(train_acc)
            val_accs.append(val_acc)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "learning_rate": scheduler.get_last_lr()[0],
            }, step=epoch)

            logger.info(
                f"Epoch {epoch:2d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}"
            )

            # Early stopping & checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                torch.save(model.state_dict(), "models/model.pt")
                logger.info(f"  -> New best model saved (val_acc={val_acc:.4f})")

                # Log confusion matrix for best epoch
                log_confusion_matrix(
                    val_labels, val_preds,
                    class_names=["cat", "dog"],
                    step=epoch,
                )
                log_classification_report(
                    val_labels, val_preds,
                    class_names=["cat", "dog"],
                    step=epoch,
                )
            else:
                patience_counter += 1
                if patience_counter >= early_stop:
                    logger.info(f"Early stopping triggered after epoch {epoch}")
                    break

        # Log loss curves
        log_loss_curves(train_losses, val_losses, train_accs, val_accs)

        # Log model. MLflow 3.x defaults to 'pt2' (traced-graph) serialization,
        # which requires a TensorSpec signature; 'pickle' avoids tracing and is
        # the standard PyTorch flavor.
        example_input = torch.randn(1, 3, 224, 224)
        mlflow.pytorch.log_model(
            model,
            "model",
            serialization_format="pickle",
            input_example=example_input,
        )
        mlflow.log_artifact("models/model.pt", artifact_path="model")

        # Save training metrics
        metrics = {
            "best_val_accuracy": best_val_acc,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "total_epochs": len(train_losses),
        }
        Path("metrics").mkdir(exist_ok=True)
        with open("metrics/train_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"Training complete. Best val_acc: {best_val_acc:.4f}")
        mlflow.log_metrics({
            "best_val_accuracy": best_val_acc,
        })

    return run_id


if __name__ == "__main__":
    main()
