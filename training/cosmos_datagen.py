"""
Cosmos Transfer 2.5 — Synthetic crop image generation for phenotype detection training.

Generates lighting, weather, and condition variants from a seed dataset of labeled crop images.
Labels carry over from seed images since Cosmos transforms appearance, not content.

Run on Nebius GPU VM or Serverless Job:
    python cosmos_datagen.py --seed_dir data/seed_crops --output_dir data/synthetic --variants_per_image 20

References:
    - NVIDIA Cosmos Cookbook ag-scenarios recipe
    - Aigen case study: 1% real + 99% Cosmos synthetic
"""

import os
import json
import argparse
import itertools
from pathlib import Path
from PIL import Image
import torch
import numpy as np


# Cosmos Transfer 2.5 depth-conditioned model
COSMOS_MODEL_ID = "nvidia/cosmos-transfer2.5-12b-sample-depth2image"

# Condition axes for synthetic variant generation
LIGHTING_CONDITIONS = ["bright_midday_sun", "overcast_diffuse", "dawn_golden_hour", "dusk_low_angle", "artificial_greenhouse_led"]
WEATHER_CONDITIONS = ["clear_dry", "post_rain_wet_foliage", "light_fog", "humid_condensation"]
SOIL_CONDITIONS = ["dry_cracked", "moist_dark", "mulched"]

# Prompt templates per class
CLASS_PROMPTS = {
    "Healthy": "a healthy tomato plant with vibrant green leaves, strong upright stem, no disease spots or wilting",
    "Stressed": "a stressed tomato plant showing mild leaf curling, slight yellowing, early signs of nutrient deficiency",
    "Diseased": "a diseased tomato plant with visible leaf spots, brown necrotic lesions, fungal infection on foliage",
}


def load_cosmos_pipeline(model_id: str, device: str = "cuda"):
    """Load Cosmos Transfer 2.5 pipeline."""
    try:
        from cosmos_transfer2.pipelines import CosmosTransferPipeline

        pipeline = CosmosTransferPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
        )
        pipeline = pipeline.to(device)
        return pipeline
    except ImportError:
        print("[WARN] cosmos_transfer2 not installed. Attempting diffusers fallback.")
        from diffusers import StableDiffusionXLImg2ImgPipeline

        pipeline = StableDiffusionXLImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        pipeline = pipeline.to(device)
        return pipeline


def estimate_depth_map(image: Image.Image) -> Image.Image:
    """Generate a monocular depth map from an RGB image using DPT."""
    try:
        from transformers import DPTForDepthEstimation, DPTImageProcessor

        processor = DPTImageProcessor.from_pretrained("Intel/dpt-large")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-large").cuda()

        inputs = processor(images=image, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = model(**inputs)
            depth = outputs.predicted_depth

        depth = torch.nn.functional.interpolate(
            depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

        depth_np = depth.cpu().numpy()
        depth_normalized = ((depth_np - depth_np.min()) / (depth_np.max() - depth_np.min()) * 255).astype(np.uint8)
        return Image.fromarray(depth_normalized)
    except Exception as e:
        print(f"[WARN] Depth estimation failed: {e}. Using grayscale fallback.")
        return image.convert("L")


def build_prompt(class_name: str, lighting: str, weather: str, soil: str) -> str:
    """Build a Cosmos Transfer prompt for a specific condition combination."""
    base = CLASS_PROMPTS.get(class_name, CLASS_PROMPTS["Healthy"])
    return (
        f"Agricultural photograph of {base}, "
        f"under {lighting.replace('_', ' ')} lighting conditions, "
        f"{weather.replace('_', ' ')} weather, "
        f"{soil.replace('_', ' ')} soil visible. "
        f"High resolution, realistic field photography, greenhouse or polytunnel setting."
    )


def generate_variants(
    pipeline,
    seed_image: Image.Image,
    depth_map: Image.Image,
    class_name: str,
    num_variants: int,
    strength: float = 0.55,
) -> list[tuple[Image.Image, dict]]:
    """Generate synthetic variants of a seed image across condition axes."""
    condition_combos = list(itertools.product(LIGHTING_CONDITIONS, WEATHER_CONDITIONS, SOIL_CONDITIONS))
    np.random.shuffle(condition_combos)
    selected = condition_combos[:num_variants]

    results = []
    for lighting, weather, soil in selected:
        prompt = build_prompt(class_name, lighting, weather, soil)

        try:
            output = pipeline(
                prompt=prompt,
                image=seed_image,
                strength=strength,
                num_inference_steps=30,
                guidance_scale=7.5,
            )
            synthetic_image = output.images[0]

            metadata = {
                "class": class_name,
                "lighting": lighting,
                "weather": weather,
                "soil": soil,
                "strength": strength,
                "source": "cosmos_transfer_2.5",
                "prompt": prompt,
            }
            results.append((synthetic_image, metadata))
        except Exception as e:
            print(f"[ERROR] Generation failed for {lighting}/{weather}/{soil}: {e}")

    return results


def process_seed_dataset(
    pipeline,
    seed_dir: str,
    output_dir: str,
    variants_per_image: int = 20,
):
    """Process all seed images and generate synthetic variants."""
    seed_path = Path(seed_dir)
    output_path = Path(output_dir)

    manifest = []
    total_generated = 0

    for class_dir in sorted(seed_path.iterdir()):
        if not class_dir.is_dir():
            continue

        class_name = class_dir.name
        class_output = output_path / class_name
        class_output.mkdir(parents=True, exist_ok=True)

        print(f"\n[INFO] Processing class: {class_name}")

        image_files = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png"))
        print(f"  Found {len(image_files)} seed images")

        for img_idx, img_path in enumerate(image_files):
            print(f"  [{img_idx + 1}/{len(image_files)}] {img_path.name}")

            seed_image = Image.open(img_path).convert("RGB").resize((512, 512))
            depth_map = estimate_depth_map(seed_image)

            # Save original as part of training set
            original_out = class_output / f"seed_{img_path.stem}.jpg"
            seed_image.save(original_out, quality=95)
            manifest.append({
                "path": str(original_out.relative_to(output_path)),
                "class": class_name,
                "source": "original",
            })

            # Generate synthetic variants
            variants = generate_variants(
                pipeline, seed_image, depth_map, class_name, variants_per_image
            )

            for var_idx, (syn_image, meta) in enumerate(variants):
                filename = f"syn_{img_path.stem}_{var_idx:03d}.jpg"
                syn_path = class_output / filename
                syn_image.save(syn_path, quality=95)

                meta["path"] = str(syn_path.relative_to(output_path))
                manifest.append(meta)
                total_generated += 1

            print(f"    Generated {len(variants)} variants")

    # Write manifest
    manifest_path = output_path / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[DONE] Total synthetic images: {total_generated}")
    print(f"  Manifest: {manifest_path}")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Cosmos Transfer 2.5 synthetic crop data generation")
    parser.add_argument("--seed_dir", required=True, help="Directory with labeled seed images (class subfolders)")
    parser.add_argument("--output_dir", required=True, help="Output directory for synthetic dataset")
    parser.add_argument("--variants_per_image", type=int, default=20, help="Number of synthetic variants per seed image")
    parser.add_argument("--strength", type=float, default=0.55, help="Cosmos transfer strength (0=no change, 1=full regen)")
    parser.add_argument("--model_id", default=COSMOS_MODEL_ID, help="Cosmos model ID")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[INFO] Loading Cosmos Transfer 2.5 pipeline...")
    pipeline = load_cosmos_pipeline(args.model_id)

    print("[INFO] Generating synthetic dataset...")
    process_seed_dataset(
        pipeline,
        args.seed_dir,
        args.output_dir,
        args.variants_per_image,
    )


if __name__ == "__main__":
    main()
