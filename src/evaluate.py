"""
Model evaluation script. Evaluates trained model on test set.
Logs metrics: accuracy, precision, recall, F1, and saves confusion matrix.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import get_device, setup_logging, NormalizedImageDataset
from src.models.cnn_model import CNNBaseline

logger = setup_logging()


@torch.no_grad()
def evaluate():
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load test data (uint8); NormalizedImageDataset normalizes per-sample
    test_data = torch.load("data/processed/test_tensors.pt")
    test_dataset = NormalizedImageDataset(test_data["images"], test_data["labels"])
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load model
    model = CNNBaseline(num_classes=2)
    state_dict = torch.load("models/model.pt", map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded from models/model.pt")

    # Evaluate
    all_preds, all_labels, all_probs = [], [], []
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)

        probs = torch.softmax(outputs, dim=1)
        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Metrics
    test_loss = running_loss / len(test_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="binary")
    recall = recall_score(all_labels, all_preds, average="binary")
    f1 = f1_score(all_labels, all_preds, average="binary")

    metrics = {
        "test_loss": round(test_loss, 6),
        "test_accuracy": round(accuracy, 6),
        "test_precision": round(precision, 6),
        "test_recall": round(recall, 6),
        "test_f1_score": round(f1, 6),
        "test_samples": len(all_labels),
    }

    Path("metrics").mkdir(exist_ok=True)
    with open("metrics/test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info("=" * 50)
    logger.info("Test Evaluation Results:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 50)

    # Confusion matrix plot
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["cat", "dog"], yticklabels=["cat", "dog"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Test Set Confusion Matrix")
    plt.tight_layout()
    plt.savefig("metrics/confusion_matrix.png", dpi=150)
    plt.close()
    logger.info("Confusion matrix saved to metrics/confusion_matrix.png")


if __name__ == "__main__":
    evaluate()
