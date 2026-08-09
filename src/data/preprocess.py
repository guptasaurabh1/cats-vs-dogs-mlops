"""
Data preprocessing pipeline for Cats vs Dogs classification.
Resizes images to 224x224, applies augmentation, and creates train/val/test splits.
"""

import sys
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.utils import set_seed

set_seed(42)

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
IMG_SIZE = 224


class CatDogDataset(Dataset):
    """Load cat/dog images from organized folder structure."""

    def __init__(self, root_dir: Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

        self.samples = []
        for cls in self.classes:
            cls_dir = self.root_dir / cls
            for f in sorted(cls_dir.iterdir()):
                if f.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    self.samples.append((str(f), self.class_to_idx[cls]))

        print(f"Loaded {len(self.samples)} images from {root_dir}")
        print(f"  Classes: {self.class_to_idx}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def get_train_transforms(img_size=IMG_SIZE):
    """Training transforms with augmentation for better generalization."""
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_val_transforms(img_size=IMG_SIZE):
    """Validation/test transforms (no augmentation), normalized."""
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_save_transforms(img_size=IMG_SIZE):
    """
    Transforms used when persisting preprocessed tensors.

    Keeps raw 0-255 integer values (uint8) so the .pt files are ~4x smaller
    than float32. Normalization happens at load time in train.py/evaluate.py.
    """
    return transforms.Compose([
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),  # -> float [0, 1]
        transforms.Lambda(lambda t: (t * 255).to(torch.uint8)),  # -> uint8 0-255
    ])


def preprocess_and_split(
    source_dir: Path = PROCESSED_DIR / "organized",
    output_dir: Path = PROCESSED_DIR,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    img_size: int = IMG_SIZE,
):
    """
    Preprocess images and create train/val/test splits.
    Saves preprocessed tensors for DVC versioning.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    full_dataset = CatDogDataset(source_dir, transform=get_save_transforms(img_size))
    n = len(full_dataset)
    rng = np.random.default_rng(42)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": Subset(full_dataset, train_idx),
        "val": Subset(full_dataset, val_idx),
        "test": Subset(full_dataset, test_idx),
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save split metadata
    split_info = {
        "train": len(train_idx),
        "val": len(val_idx),
        "test": len(test_idx),
        "img_size": img_size,
        "dtype": "uint8",
        "normalized": False,  # tensors are raw 0-255; normalize at load time
        "classes": full_dataset.classes,
        "class_to_idx": full_dataset.class_to_idx,
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # Save preprocessed tensors for each split
    for split_name, subset in splits.items():
        images, labels = [], []
        loader = DataLoader(subset, batch_size=64, shuffle=False, num_workers=0)
        for imgs, labs in tqdm(loader, desc=f"Saving {split_name}"):
            images.append(imgs)
            labels.append(labs)

        images_tensor = torch.cat(images, dim=0).to(torch.uint8)
        labels_tensor = torch.cat(labels, dim=0).to(torch.long)

        save_path = output_dir / f"{split_name}_tensors.pt"
        torch.save({"images": images_tensor, "labels": labels_tensor}, save_path)
        print(f"Saved {split_name}: {images_tensor.shape} {images_tensor.dtype} -> {save_path}")

    print(f"\nSplit sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    print(f"Preprocessing complete. Data saved to {output_dir.resolve()}")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Preprocess Cats vs Dogs dataset")
    parser.add_argument("--source", type=str, default=str(PROCESSED_DIR / "organized"))
    parser.add_argument("--output", type=str, default=str(PROCESSED_DIR))
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    args = parser.parse_args()

    preprocess_and_split(
        source_dir=Path(args.source),
        output_dir=Path(args.output),
        img_size=args.img_size,
    )


if __name__ == "__main__":
    main()
