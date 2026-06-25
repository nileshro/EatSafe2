import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/best_model.pth"

device = torch.device("cpu")

# ── Load model checkpoint ─────────────────────────────────────────────────────
try:
    checkpoint  = torch.load(MODEL_PATH, map_location=device)
    classes     = checkpoint["classes"]
    num_classes = checkpoint["num_classes"]

    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _MODEL_LOADED = True

except Exception as e:
    print(f"Logo model load error: {e}")
    _MODEL_LOADED = False
    classes = []
    num_classes = 0
    model = None

# ── Transform ─────────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def predict_logo(image) -> dict:
    """
    Predict brand logo from a PIL Image.
    Returns {"brand": str, "confidence": float}.

    Confidence thresholds:
      >= 0.75 → confirmed brand name
      >= 0.45 → "Possibly <brand>" 
      <  0.45 → "Unknown"
    """
    # Model not loaded — skip verification gracefully
    if not _MODEL_LOADED or model is None:
        return {"brand": "Unknown", "confidence": 0.0}

    if not isinstance(image, Image.Image):
        image = Image.open(image)

    # Convert to RGB in case of RGBA/grayscale
    image = image.convert("RGB")

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs       = model(tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        confidence = confidence.item()

    predicted_class = classes[predicted.item()]

    if confidence >= 0.75:
        brand = predicted_class
    elif confidence >= 0.45:
        brand = "Possibly " + predicted_class
    else:
        brand = "Unknown"

    return {
        "brand": brand,
        "confidence": round(confidence, 4),
        "raw_class": predicted_class        # always available for debugging
    }