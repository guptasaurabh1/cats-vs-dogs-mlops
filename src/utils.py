"""
Shared utility functions for the MLOps pipeline.
"""

import random
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# Batch shape [1,3,1,1]: broadcasting a [N,3,H,W] batch against this gives
# [N,3,H,W] (matches the resize+ToTensor+Normalize used at inference).
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Per-channel shape [3,1,1] for single-image normalization in the dataset.
# A [3,H,W] image broadcast against [1,3,1,1] would prepend a dim -> [1,3,H,W];
# [3,1,1] broadcasts correctly to [3,H,W].
CHANNEL_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
CHANNEL_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


class NormalizedImageDataset(Dataset):
    """
    Dataset over preprocessed uint8 tensors.

    Images are stored as raw uint8 [0,255] to keep disk + RAM usage low.
    Each sample is normalized to float32 lazily inside __getitem__, so only a
    single batch is materialized in memory at a time (keeps peak RAM bounded).
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.dtype == torch.uint8:
            img = img.float() / 255.0
        mean = CHANNEL_MEAN.to(img.device)
        std = CHANNEL_STD.to(img.device)
        return (img - mean) / std, self.labels[idx]


def set_seed(seed: int = 42):
    """Set reproducibility seed across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def normalize_images(images: torch.Tensor) -> torch.Tensor:
    """
    Convert uint8 [0,255] tensors to normalized float32 for the model.

    Preprocessed tensors are stored as raw uint8 to save ~4x disk space.
    This undoes ToTensor (divide by 255) and applies ImageNet normalization,
    matching the transform used at inference time in predict.py.
    """
    if images.dtype == torch.uint8:
        images = images.float() / 255.0
    mean = IMAGENET_MEAN.to(images.device)
    std = IMAGENET_STD.to(images.device)
    return (images - mean) / std


def get_device() -> torch.device:
    """Get the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(log_dir: str = "logs"):
    """Configure logging to file and stdout."""
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("mlops")
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(Path(log_dir) / "pipeline.log")
    fh.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger
