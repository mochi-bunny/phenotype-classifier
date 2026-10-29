"""
Crop disease diagnostics — runs trained ONNX model for pathology detection.

Uses the same trained model as phenotype_detector.py (shared backbone),
but maps output through the diagnostics score mapper for the pathology
dashboard view.
"""

import json
import sys
from PIL import Image

from phenotype_detector import load_image_from_source, predict
from score_mapper import map_prediction_to_diagnostics


def identify_crop_disease(image_url: str, metadata: dict) -> dict:
    """
    Main entry point — drop-in replacement for the original LLM-based function.

    Args:
        image_url: URL or local path to crop image
        metadata: Environmental telemetry data

    Returns:
        Disease diagnostics dict matching the dashboard pathology format.
    """
    image = load_image_from_source(image_url)
    prediction = predict(image)

    diagnostics = map_prediction_to_diagnostics(prediction)
    diagnostics["target_id"] = metadata.get("target_id", "UNKNOWN")

    return diagnostics


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python disease_diagnostics.py <image_path_or_url>")
        sys.exit(1)

    result = identify_crop_disease(sys.argv[1], {"target_id": "TEST-001"})
    print(json.dumps(result, indent=2))
