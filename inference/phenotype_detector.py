"""
Crop phenotype detector — runs trained ONNX model instead of LLM inference.

Replaces the original Llama-3.1-70B chat call with a real vision model
trained on PlantVillage tomato data (optionally Cosmos-augmented).

The model outputs 3 classes: Healthy / Stressed / Diseased.
The score_mapper expands this into the full 6-category phenotype format.
"""

import os
import json
import io
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from score_mapper import map_prediction_to_phenotype

# Model paths (relative to inference/ directory)
MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "models"))
ONNX_MODEL_PATH = MODEL_DIR / "phenotype_model.onnx"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
META_PATH = MODEL_DIR / "training_meta.json"

# Preprocessing constants (ImageNet normalization, matching training)
INPUT_SIZE = (224, 224)
NORMALIZE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
NORMALIZE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Lazy-loaded globals
_session = None
_class_names = None
_model_hash = None


def _load_model():
    """Lazy-load the ONNX model and metadata."""
    global _session, _class_names, _model_hash

    if _session is not None:
        return

    if not ONNX_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {ONNX_MODEL_PATH}. "
            "Run training first: python training/train.py"
        )

    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    _session = ort.InferenceSession(str(ONNX_MODEL_PATH), providers=providers)

    if CLASS_NAMES_PATH.exists():
        with open(CLASS_NAMES_PATH) as f:
            _class_names = json.load(f)
    else:
        _class_names = ["Diseased", "Healthy", "Stressed"]

    if META_PATH.exists():
        with open(META_PATH) as f:
            meta = json.load(f)
            _model_hash = meta.get("model_hash", "unknown")
    else:
        _model_hash = "unknown"

    print(f"[INFO] Phenotype model loaded: {ONNX_MODEL_PATH}")
    print(f"  Classes: {_class_names}")
    print(f"  Model hash: {_model_hash}")


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess a PIL image for model inference."""
    image = image.convert("RGB").resize(INPUT_SIZE, Image.BILINEAR)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = (arr - NORMALIZE_MEAN) / NORMALIZE_STD
    arr = arr.transpose(2, 0, 1)  # HWC → CHW
    return arr[np.newaxis, ...]  # Add batch dim


def load_image_from_source(image_source: str) -> Image.Image:
    """Load image from URL or local file path."""
    if image_source.startswith(("http://", "https://")):
        import httpx
        response = httpx.get(image_source, timeout=30, follow_redirects=True)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content))
    else:
        return Image.open(image_source)


def predict(image: Image.Image) -> dict:
    """
    Run inference on a single image.

    Returns:
        {
            "class": "Healthy" | "Stressed" | "Diseased",
            "confidence": float,
            "all_scores": {"Diseased": float, "Healthy": float, "Stressed": float},
            "model_hash": str
        }
    """
    _load_model()

    input_tensor = preprocess_image(image)
    logits = _session.run(None, {"image": input_tensor})[0]

    # Softmax
    exp_logits = np.exp(logits[0] - logits[0].max())
    probs = exp_logits / exp_logits.sum()

    idx = int(np.argmax(probs))

    return {
        "class": _class_names[idx],
        "confidence": float(probs[idx]),
        "all_scores": {name: float(p) for name, p in zip(_class_names, probs)},
        "model_hash": _model_hash,
    }


def analyze_crop_phenotype(image_url: str, metadata: dict) -> dict:
    """
    Main entry point — drop-in replacement for the original LLM-based function.

    Args:
        image_url: URL or local path to crop image
        metadata: Telemetry data (GPS, NDVI, soil moisture, etc.)

    Returns:
        Full phenotype evaluation dict matching the dashboard format.
    """
    image = load_image_from_source(image_url)
    prediction = predict(image)

    evaluation = map_prediction_to_phenotype(prediction)
    evaluation["target_id"] = metadata.get("target_id", "UNKNOWN")

    return evaluation


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python phenotype_detector.py <image_path_or_url>")
        sys.exit(1)

    result = analyze_crop_phenotype(sys.argv[1], {"target_id": "TEST-001"})
    print(json.dumps(result, indent=2))
