"""
Unit tests for data preprocessing functions.
M3 requirement: tests for at least one data preprocessing function.
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.preprocess import get_train_transforms, get_val_transforms, IMG_SIZE


class TestPreprocessing:
    """Test suite for data preprocessing transforms."""

    def test_train_transforms_output_shape(self):
        """Verify training transform produces correct tensor shape."""
        transform = get_train_transforms()
        img = Image.new("RGB", (300, 400), color=(128, 128, 128))
        tensor = transform(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), (
            f"Expected (3, {IMG_SIZE}, {IMG_SIZE}), got {tensor.shape}"
        )

    def test_val_transforms_output_shape(self):
        """Verify validation transform produces correct tensor shape."""
        transform = get_val_transforms()
        img = Image.new("RGB", (200, 350), color=(64, 64, 64))
        tensor = transform(img)
        assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), (
            f"Expected (3, {IMG_SIZE}, {IMG_SIZE}), got {tensor.shape}"
        )

    def test_train_transforms_augmentation(self):
        """Training transforms should produce different results each call (random)."""
        transform = get_train_transforms()
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        results = set()
        for _ in range(10):
            t = transform(img)
            results.add(t.flatten().sum().item())
        # With augmentation, at least some calls should differ
        assert len(results) > 1, "Training transforms appear deterministic"

    def test_val_transforms_deterministic(self):
        """Validation transforms should be deterministic."""
        transform = get_val_transforms()
        img = Image.new("RGB", (300, 300), color=(100, 150, 200))
        t1 = transform(img)
        t2 = transform(img)
        assert (t1 == t2).all(), "Validation transforms are not deterministic"

    def test_normalization_range(self):
        """Verify normalized tensor values are roughly in expected range."""
        transform = get_val_transforms()
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
        tensor = transform(img)
        # After normalization with ImageNet stats, white should be positive
        assert tensor.min() > -3.0 and tensor.max() < 3.0, (
            f"Normalized values out of expected range: [{tensor.min():.2f}, {tensor.max():.2f}]"
        )

    def test_grayscale_conversion(self):
        """Grayscale images should be converted to RGB (3 channels)."""
        transform = get_val_transforms()
        img = Image.new("L", (IMG_SIZE, IMG_SIZE), color=128)
        tensor = transform(img)
        assert tensor.shape[0] == 3, (
            f"Grayscale conversion failed: expected 3 channels, got {tensor.shape[0]}"
        )

    def test_random_resized_crop_changes_size(self):
        """RandomResizedCrop in train transforms should handle varied input sizes."""
        transform = get_train_transforms()
        for size in [(200, 300), (400, 500), (224, 224)]:
            img = Image.new("RGB", size, color=(50, 100, 150))
            tensor = transform(img)
            assert tensor.shape == (3, IMG_SIZE, IMG_SIZE), (
                f"Failed for input size {size}: got {tensor.shape}"
            )

    def test_to_tensor_range(self):
        """ToTensor should scale pixel values to [0, 1]."""
        from torchvision import transforms as T
        to_tensor = T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor()])
        img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), color=(255, 255, 255))
        tensor = to_tensor(img)
        assert tensor.max() <= 1.0 and tensor.min() >= 0.0, (
            f"ToTensor range error: [{tensor.min():.4f}, {tensor.max():.4f}]"
        )
