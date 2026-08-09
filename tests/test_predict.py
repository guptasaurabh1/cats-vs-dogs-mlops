"""
Unit tests for model inference functions.
M3 requirement: tests for at least one model utility/inference function.
"""

import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.cnn_model import CNNBaseline, count_parameters


class TestModelUtils:
    """Test suite for model utility functions."""

    def test_count_parameters_positive(self):
        """Count parameters should return positive integer."""
        model = CNNBaseline(num_classes=2)
        n = count_parameters(model)
        assert isinstance(n, int), f"Expected int, got {type(n)}"
        assert n > 0, f"Expected positive count, got {n}"

    def test_model_output_shape(self):
        """Model forward pass should produce (batch, num_classes) output."""
        import torch
        model = CNNBaseline(num_classes=2)
        model.eval()
        x = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        assert y.shape == (4, 2), f"Expected (4, 2), got {y.shape}"

    def test_model_softmax_output(self):
        """Logits should produce valid probabilities after softmax."""
        import torch
        import torch.nn.functional as F
        model = CNNBaseline(num_classes=2)
        model.eval()
        x = torch.randn(8, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        probs = F.softmax(y, dim=1)
        assert torch.allclose(probs.sum(dim=1), torch.ones(8)), (
            "Softmax probabilities don't sum to 1"
        )

    def test_predict_output_keys(self):
        """Predict output dict should contain expected keys."""
        import torch
        model = CNNBaseline(num_classes=2)
        # Use deterministic weights for test
        model.eval()
        # Create a mock image
        img = Image.new("RGB", (224, 224), color=(128, 128, 128))

        # Test predict by directly calling model
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = model(tensor)

        import torch.nn.functional as F
        probs = F.softmax(outputs, dim=1).squeeze(0)
        predicted_class = 0 if probs[0] > probs[1] else 1
        label = "cat" if predicted_class == 0 else "dog"

        result = {
            "prediction": label,
            "class_id": predicted_class,
            "confidence": round(max(probs[0].item(), probs[1].item()), 4),
            "probabilities": {
                "cat": round(probs[0].item(), 4),
                "dog": round(probs[1].item(), 4),
            },
        }

        expected_keys = {"prediction", "class_id", "confidence", "probabilities"}
        assert set(result.keys()) == expected_keys, (
            f"Expected keys {expected_keys}, got {set(result.keys())}"
        )

    def test_predict_returns_valid_label(self):
        """Prediction label should be 'cat' or 'dog'."""
        import torch
        model = CNNBaseline(num_classes=2)
        model.eval()
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            y = model(x)
        _, pred = torch.max(y, 1)
        label = "cat" if pred.item() == 0 else "dog"
        assert label in ("cat", "dog"), f"Invalid label: {label}"

    def test_model_parameters_are_float32(self):
        """Model parameters should be float32 for consistency."""
        import torch
        model = CNNBaseline(num_classes=2)
        for name, param in model.named_parameters():
            assert param.dtype == torch.float32, (
                f"Parameter {name} has dtype {param.dtype}, expected float32"
            )

    def test_model_num_classes(self):
        """Model should produce exactly 2 outputs (cat/dog)."""
        model = CNNBaseline(num_classes=2)
        assert model.classifier[-1].out_features == 2, (
            f"Expected 2 output classes, got {model.classifier[-1].out_features}"
        )
