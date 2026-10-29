"""
Crop phenotype evaluation via Vision-Language Model (VLM) on Nebius AI Studio.

Uses Qwen2.5-VL-72B-Instruct to score crop images across all 6 phenotype
judging categories directly — the VLM can distinguish leaf curl (abiotic stress)
from leaf spots (biotic stress), assess architecture, and evaluate yield traits
from visual cues.

Supports few-shot visual prompting with reference images for each class
to improve consistency and accuracy.

Usage:
    # Basic (zero-shot)
    python phenotype_vlm.py <image_url>

    # With reference images (few-shot)
    python phenotype_vlm.py <image_url> --ref_healthy ref/healthy.jpg --ref_stressed ref/stressed.jpg --ref_diseased ref/diseased.jpg

Environment:
    NEBIUS_API_KEY=your-key
"""

import os
import json
import sys
from typing import Optional
from openai import OpenAI

NEBIUS_BASE_URL = "https://api.studio.nebius.com/v1"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"

# Comprehensive phenotype evaluation prompt covering all 6 judging categories
PHENOTYPE_SYSTEM_PROMPT = """You are an expert agricultural phenotyping AI trained in crop science, plant pathology, and breeding selection. You evaluate crop images with the precision of a professional plant breeder and the rigor of a scientific judge.

Your evaluation must be grounded in what is visually observable in the image. Do not hallucinate traits you cannot see. If a trait cannot be assessed from the image (e.g., root system from a leaf photo), score it as null and state why."""

PHENOTYPE_EVAL_PROMPT = """Evaluate this crop image across all 6 official phenotype judging categories.

For each sub-trait, provide a score from 0.0 to 1.0 based on what you can visually assess:
- 0.0–0.3 = Poor / Severe deficiency or damage
- 0.3–0.6 = Below average / Moderate stress or weakness
- 0.6–0.8 = Good / Healthy with minor imperfections
- 0.8–1.0 = Excellent / Genetically desirable expression

**Category 1: Plant Architecture & Morphology**
- erect_leaf_angle_score: Are leaves angled upward (erect, efficient light capture) or drooping? Erect = high score.
- stalk_strength_and_diameter_score: Is the stem thick, rigid, and upright? Or thin, leaning, lodged?
- root_system_architecture_score: If roots are visible (seedling, exposed), are they well-branched and deep? If not visible, return null.
- branching_habit: Describe as one of: "Erect Canopy", "Semi-Erect", "Spreading", "Irregular / Compromised"

**Category 2: Yield & Reproductive Traits**
- harvest_index_estimate: What proportion of the plant's biomass appears to be harvestable fruit/grain vs vegetative growth? High fruit-to-leaf ratio = high score.
- kernel_fruit_density_count: Estimate the number of visible fruits, flowers, or grain heads. Return an integer.
- fertility_efficiency_score: Are flowers/fruits developing uniformly, or are there barren spots, dropped flowers, or aborted fruit?

**Category 3: Biotic Stress Resistance**
- necrotic_chlorotic_threshold_score: How much necrotic (brown/dead) or chlorotic (yellow) tissue is visible? 0.0 = extensive damage, 1.0 = none visible.
- pest_deterrence_score: Is there visible insect damage (holes, mines, webbing, frass)? 1.0 = no pest damage visible.
- hypersensitive_response_detected: Are there small, well-defined lesions suggesting the plant is actively fighting infection? true/false.

**Category 4: Abiotic Stress Resilience**
- stomatal_regulation_leaf_rolling_score: Are leaves rolled, curled, or wilted (suggesting water stress)? 1.0 = fully turgid, no rolling.
- stay_green_trait_active: Is the plant maintaining green color despite stress conditions? true/false.
- salinity_aluminum_tolerance_score: Are there marginal leaf burns or tip necrosis suggesting salt/mineral toxicity? 1.0 = no symptoms.

**Category 5: Quality, Phenology & Post-Harvest**
- synchronous_flowering_ripening_score: Are fruits/flowers at a uniform developmental stage? 1.0 = highly uniform ripening.
- photoperiod_insensitivity_score: Does the plant show appropriate flowering for its growth stage? Score based on apparent maturity match.
- pericarp_skin_toughness_score: Do fruits appear firm with intact skin, or soft/cracked/split? 1.0 = firm, intact.

**Category 6: Harvest & Processing Suitability**
- non_shattering_seed_pods_score: Are seed pods/fruits retained on the plant? 1.0 = fully retained, no shattering.
- detachment_force_score: Do fruits appear firmly attached to the plant? 1.0 = well-attached.

**Overall Assessment:**
- classification: "Genetically Desirable" (overall > 0.8) | "Standard" (0.5-0.8) | "Suboptimal" (< 0.5)
- overall_phenotypic_score: Weighted average of all assessed traits (0.0–1.0)
- collector_queue_priority: "High" | "Medium" | "Low"
- recommended_action: One sentence describing what a breeder should do with this plant.

Return your response as valid JSON matching this exact structure:
{
  "classification": string,
  "judging_categories": {
    "1_plant_architecture_and_morphology": {
      "erect_leaf_angle_score": float | null,
      "stalk_strength_and_diameter_score": float | null,
      "root_system_architecture_score": float | null,
      "branching_habit": string
    },
    "2_yield_and_reproductive_traits": {
      "harvest_index_estimate": float | null,
      "kernel_fruit_density_count": integer | null,
      "fertility_efficiency_score": float | null
    },
    "3_biotic_stress_resistance": {
      "necrotic_chlorotic_threshold_score": float,
      "pest_deterrence_score": float,
      "hypersensitive_response_detected": boolean
    },
    "4_abiotic_stress_resilience": {
      "stomatal_regulation_leaf_rolling_score": float,
      "stay_green_trait_active": boolean,
      "salinity_aluminum_tolerance_score": float | null
    },
    "5_quality_phenology_and_post_harvest": {
      "synchronous_flowering_ripening_score": float | null,
      "photoperiod_insensitivity_score": float | null,
      "pericarp_skin_toughness_score": float | null
    },
    "6_harvest_and_processing_suitability": {
      "non_shattering_seed_pods_score": float | null,
      "detachment_force_score": float | null
    }
  },
  "overall_phenotypic_score": float,
  "collector_queue_priority": string,
  "recommended_action": string,
  "visual_observations": string,
  "assessment_limitations": string
}"""

DISEASE_EVAL_PROMPT = """Evaluate this crop image for plant pathology and disease diagnostics.

Score each diagnostic vector from 0.0 (no symptoms) to 1.0 (severe):

**Category 1: Fungal Pathogens**
- rust_mildew_presence_score: Powdery/downy mildew, rust pustules (orange/brown spots on leaf undersides)
- leaf_spot_blight_score: Circular or irregular necrotic lesions with defined margins (Septoria, Alternaria, Bipolaris)
- root_rot_indicators: Crown discoloration, wilting despite adequate moisture, stem base darkening

**Category 2: Bacterial & Viral Infections**
- bacterial_blight_score: Water-soaked lesions, angular leaf spots, bacterial ooze or streaming
- viral_mosaic_distortion_score: Mosaic patterns, leaf curling/distortion, vein clearing, stunted growth
- systemic_necrosis_detected: Is there evidence of systemic spread (multiple plant parts affected)? true/false

**Category 3: Pest & Insect Damage**
- chewing_defoliation_score: Holes, ragged leaf edges, skeletonized leaves from caterpillars/beetles
- sucking_pest_chlorosis_score: Stippling, silvering, or yellowing from aphids/whiteflies/thrips/mites
- borer_entry_wound_detected: Stem holes, frass trails, sawdust-like deposits? true/false

**Category 4: Abiotic & Nutritional Stress**
- nitrogen_phosphorus_deficiency_score: Uniform yellowing (N), purple discoloration (P), interveinal chlorosis (Fe/Mn)
- sunscald_heat_stress_score: Bleached patches, leaf scorch on sun-exposed surfaces
- chemical_drift_burn_score: Irregular bleaching/necrosis patterns inconsistent with disease

**Overall:**
- pathology_status: "Healthy" | "Suspect" | "Infected"
- primary_diagnosis: Name the most likely condition
- confidence_score: 0.0–1.0
- treatment_urgency: "Critical" | "Monitor" | "None"
- recommended_action: One sentence treatment recommendation

Return valid JSON matching this structure:
{
  "pathology_status": string,
  "diagnostic_categories": {
    "1_fungal_pathogens": { "rust_mildew_presence_score": float, "leaf_spot_blight_score": float, "root_rot_indicators": float },
    "2_bacterial_and_viral_infections": { "bacterial_blight_score": float, "viral_mosaic_distortion_score": float, "systemic_necrosis_detected": boolean },
    "3_pest_and_insect_damage": { "chewing_defoliation_score": float, "sucking_pest_chlorosis_score": float, "borer_entry_wound_detected": boolean },
    "4_abiotic_and_nutritional_stress": { "nitrogen_phosphorus_deficiency_score": float, "sunscald_heat_stress_score": float, "chemical_drift_burn_score": float }
  },
  "primary_diagnosis": string,
  "confidence_score": float,
  "treatment_urgency": string,
  "recommended_action": string,
  "visual_observations": string
}"""

# Few-shot reference prompts
REFERENCE_HEALTHY_PROMPT = "REFERENCE IMAGE — HEALTHY: This is a verified healthy crop specimen. Note the vibrant green color, erect leaf architecture, strong stem, no lesions or discoloration. Use this as your baseline for a high-scoring plant."

REFERENCE_STRESSED_PROMPT = "REFERENCE IMAGE — STRESSED: This is a verified stressed crop specimen showing abiotic stress. Note the mild leaf curling, slight yellowing, possible nutrient deficiency. This represents a mid-range score."

REFERENCE_DISEASED_PROMPT = "REFERENCE IMAGE — DISEASED: This is a verified diseased crop specimen with pathogen damage. Note the necrotic lesions, chlorotic halos, possible fungal spots. This represents a low-scoring plant."


def create_client() -> OpenAI:
    api_key = os.environ.get("NEBIUS_API_KEY")
    if not api_key:
        raise EnvironmentError("NEBIUS_API_KEY environment variable not set")
    return OpenAI(base_url=NEBIUS_BASE_URL, api_key=api_key)


def build_messages(
    image_url: str,
    eval_prompt: str,
    metadata: Optional[dict] = None,
    ref_healthy_url: Optional[str] = None,
    ref_stressed_url: Optional[str] = None,
    ref_diseased_url: Optional[str] = None,
) -> list[dict]:
    """Build message list with optional few-shot reference images."""
    messages = [{"role": "system", "content": PHENOTYPE_SYSTEM_PROMPT}]

    # Few-shot reference images (if provided)
    if any([ref_healthy_url, ref_stressed_url, ref_diseased_url]):
        ref_content = [{"type": "text", "text": "I will show you reference images of verified crop specimens at known quality levels, then ask you to evaluate a new image.\n"}]

        if ref_healthy_url:
            ref_content.extend([
                {"type": "text", "text": REFERENCE_HEALTHY_PROMPT},
                {"type": "image_url", "image_url": {"url": ref_healthy_url}},
            ])
        if ref_stressed_url:
            ref_content.extend([
                {"type": "text", "text": REFERENCE_STRESSED_PROMPT},
                {"type": "image_url", "image_url": {"url": ref_stressed_url}},
            ])
        if ref_diseased_url:
            ref_content.extend([
                {"type": "text", "text": REFERENCE_DISEASED_PROMPT},
                {"type": "image_url", "image_url": {"url": ref_diseased_url}},
            ])

        messages.append({"role": "user", "content": ref_content})
        messages.append({"role": "assistant", "content": "I've reviewed the reference images and established baselines for Healthy, Stressed, and Diseased specimens. Please show me the image to evaluate."})

    # Query image
    query_content = []

    if metadata:
        query_content.append({
            "type": "text",
            "text": f"Environmental context for this image: {json.dumps(metadata)}\n",
        })

    query_content.append({"type": "text", "text": eval_prompt})
    query_content.append({"type": "image_url", "image_url": {"url": image_url}})

    messages.append({"role": "user", "content": query_content})

    return messages


def analyze_crop_phenotype_vlm(
    image_url: str,
    metadata: dict,
    ref_healthy_url: Optional[str] = None,
    ref_stressed_url: Optional[str] = None,
    ref_diseased_url: Optional[str] = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """
    Evaluate crop phenotype using a VLM on Nebius AI Studio.

    Args:
        image_url: URL or base64 data URI of the crop image
        metadata: Telemetry data (GPS, NDVI, soil moisture, etc.)
        ref_healthy_url: Optional reference image of a verified healthy specimen
        ref_stressed_url: Optional reference image of a verified stressed specimen
        ref_diseased_url: Optional reference image of a verified diseased specimen
        model: Nebius model ID

    Returns:
        Full phenotype evaluation dict matching the dashboard format.
    """
    client = create_client()

    messages = build_messages(
        image_url, PHENOTYPE_EVAL_PROMPT, metadata,
        ref_healthy_url, ref_stressed_url, ref_diseased_url,
    )

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    result["target_id"] = metadata.get("target_id", "UNKNOWN")
    result["inference_mode"] = "vlm"
    result["model"] = model
    result["few_shot"] = any([ref_healthy_url, ref_stressed_url, ref_diseased_url])

    return result


def identify_crop_disease_vlm(
    image_url: str,
    metadata: dict,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Evaluate crop disease diagnostics using a VLM."""
    client = create_client()

    messages = build_messages(image_url, DISEASE_EVAL_PROMPT, metadata)

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0.1,
        max_tokens=2000,
        response_format={"type": "json_object"},
    )

    result = json.loads(response.choices[0].message.content)
    result["target_id"] = metadata.get("target_id", "UNKNOWN")
    result["inference_mode"] = "vlm"
    result["model"] = model

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="VLM phenotype evaluation via Nebius")
    parser.add_argument("image_url", help="Image URL or path")
    parser.add_argument("--mode", default="phenotype", choices=["phenotype", "pathology"])
    parser.add_argument("--ref_healthy", default=None, help="Reference healthy image URL")
    parser.add_argument("--ref_stressed", default=None, help="Reference stressed image URL")
    parser.add_argument("--ref_diseased", default=None, help="Reference diseased image URL")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()

    metadata = {"target_id": "CLI-TEST-001"}

    if args.mode == "phenotype":
        result = analyze_crop_phenotype_vlm(
            args.image_url, metadata,
            ref_healthy_url=args.ref_healthy,
            ref_stressed_url=args.ref_stressed,
            ref_diseased_url=args.ref_diseased,
            model=args.model,
        )
    else:
        result = identify_crop_disease_vlm(args.image_url, metadata, model=args.model)

    print(json.dumps(result, indent=2))
