"""
Experiment tracking wrapper.
Decorate training runs with MLflow logging.
"""

import mlflow


def log_model_params(model):
    """Log model architecture parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    mlflow.log_params({
        "model_total_params": total,
        "model_trainable_params": trainable,
        "model_architecture": model.__class__.__name__,
    })
