"""
Inference utility for the Cats vs Dogs classifier.
Used by the FastAPI service for prediction.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.models.cnn_model import CNNBaseline


class CatDogClassifier:
    """Wrapper for model inference with preprocessing."""

    def __init__(
        self,
        model_path: str = "models/model.pt",
        device: str = None,
    ):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = CNNBaseline(num_classes=2)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def preprocess(self, image: Image.Image) -> torch.Tensor:
        """Preprocess a PIL image for inference."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0)
        return tensor.to(self.device)

    @torch.no_grad()
    def predict(self, image: Image.Image) -> dict:
        """
        Run inference on a PIL image.
        Returns class probabilities and predicted label.
        """
        tensor = self.preprocess(image)
        outputs = self.model(tensor)
        probs = F.softmax(outputs, dim=1).squeeze(0)

        prob_cat = probs[0].item()
        prob_dog = probs[1].item()
        predicted_class = 0 if prob_cat > prob_dog else 1
        confidence = max(prob_cat, prob_dog)
        label = "cat" if predicted_class == 0 else "dog"

        return {
            "prediction": label,
            "class_id": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": {
                "cat": round(prob_cat, 4),
                "dog": round(prob_dog, 4),
            },
        }

    @torch.no_grad()
    def predict_batch(self, images: list) -> list:
        """Run inference on a list of PIL images."""
        results = []
        for img in images:
            results.append(self.predict(img))
        return results
