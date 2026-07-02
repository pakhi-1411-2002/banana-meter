from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.models.multitask_resnet import BananaMultiTaskResNet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = PROJECT_ROOT / "models" / "multitask_resnet18_best.pth"

STAGE_NAMES = {
    0: "Early",
    1: "Early-Mid",
    2: "Mid",
    3: "Late",
    4: "End-stage",
}

INFER_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def get_message(days_left: float, stage_name: str) -> str:
    if days_left >= 5:
        return "This banana appears to be in an early progression stage."
    if days_left >= 3:
        return "This banana looks mid-way through the progression timeline."
    if days_left >= 1.5:
        return "This banana appears to be entering a later progression stage."
    if days_left >= 0.5:
        return "This banana is near the end of the progression timeline."
    return "This banana appears to be at the end-stage of the progression timeline."


def load_model(model_path: str | Path = MODEL_PATH) -> BananaMultiTaskResNet:
    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    model = BananaMultiTaskResNet(
        num_classes=5,
        pretrained=False,
        freeze_backbone=False,
        dropout=0.3,
    ).to(DEVICE)

    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess_image(image_path: str | Path) -> torch.Tensor:
    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image = Image.open(image_path).convert("RGB")
    tensor = INFER_TRANSFORM(image).unsqueeze(0)  # [1, 3, 224, 224]
    return tensor.to(DEVICE)


def predict_image(
    image_path: str | Path,
    model_path: str | Path = MODEL_PATH,
) -> Dict[str, Any]:
    model = load_model(model_path)
    image_tensor = preprocess_image(image_path)

    with torch.no_grad():
        stage_logits, days_left_pred = model(image_tensor)

        stage_probs = F.softmax(stage_logits, dim=1)
        pred_stage_idx = int(torch.argmax(stage_probs, dim=1).item())
        confidence = float(stage_probs[0, pred_stage_idx].item())

        days_left = float(days_left_pred.item())
        days_left = max(0.0, min(7.0, days_left))  # clamp to sensible range

    stage_name = STAGE_NAMES[pred_stage_idx]
    message = get_message(days_left, stage_name)

    result = {
        "image_path": str(image_path),
        "predicted_stage_index": pred_stage_idx,
        "predicted_stage_name": stage_name,
        "predicted_days_left": round(days_left, 2),
        "confidence": round(confidence, 4),
        "message": message,
    }
    return result


if __name__ == "__main__":
    # Example manual test
    sample_path = "data/raw/test/honey banana/example.jpg"
    try:
        output = predict_image(sample_path)
        print(output)
    except Exception as e:
        print(f"Inference failed: {e}")