"""
MLflow setup for experiment tracking.
Logs parameters, metrics (accuracy, loss, precision, recall, F1, confusion matrix, loss curves).
"""

import os

# MLflow 3.x blocks the filesystem backend by default; keep it available
# so experiments can be logged to the local `mlruns/` directory.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow
import mlflow.pytorch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix, classification_report


def configure_mlflow(
    experiment_name: str = "cats-vs-dogs-classification",
    tracking_uri: str = "mlruns",
    artifact_location: str = "mlruns",
):
    """Set up the MLflow tracking server and experiment."""
    mlflow.set_tracking_uri(tracking_uri)
    exp = mlflow.get_experiment_by_name(experiment_name)
    if exp is None:
        mlflow.create_experiment(
            name=experiment_name,
            artifact_location=artifact_location,
        )
    mlflow.set_experiment(experiment_name)
    return mlflow


def log_confusion_matrix(y_true, y_pred, class_names, step, artifact_path="metrics"):
    """Log a confusion matrix plot to MLflow."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix (step {step})")

    path = Path(f"artifacts/{artifact_path}/confusion_matrix_step_{step}.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_loss_curves(train_losses, val_losses, train_accs, val_accs, artifact_path="metrics"):
    """Log loss and accuracy curves to MLflow."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, "b-o", label="Train Loss", markersize=4)
    ax1.plot(epochs, val_losses, "r-o", label="Val Loss", markersize=4)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, train_accs, "b-o", label="Train Acc", markersize=4)
    ax2.plot(epochs, val_accs, "r-o", label="Val Acc", markersize=4)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Training & Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    path = Path(f"artifacts/{artifact_path}/loss_accuracy_curves.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_classification_report(y_true, y_pred, class_names, step, artifact_path="metrics"):
    """Log classification report as a text artifact."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    path = Path(f"artifacts/{artifact_path}/classification_report_step_{step}.txt")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(report)
    mlflow.log_artifact(str(path), artifact_path=artifact_path)
