"""
Trait preference filtering — compares AI evaluation against user thresholds.

Unchanged from original: pure logic, model-agnostic.
"""

user_preferences = {
    "target_classification": "Genetically Desirable",
    "priority_level": "High",
    "target_thresholds": {
        "1_plant_architecture_and_morphology": {
            "erect_leaf_angle_score": 0.75,
            "stalk_strength_and_diameter_score": 0.80
        },
        "3_biotic_stress_resistance": {
            "pest_deterrence_score": 0.70
        }
    }
}

def filter_crop_by_preferences(evaluation_result: dict, preferences: dict) -> dict:
    """
    Compares an AI evaluation result against user-specified trait thresholds.
    """
    classification_match = evaluation_result.get("classification") == preferences.get("target_classification")
    priority_match = evaluation_result.get("collector_queue_priority") == preferences.get("priority_level")

    trait_evaluations = {}
    all_thresholds_met = True

    categories = evaluation_result.get("judging_categories", {})
    for category_name, required_traits in preferences.get("target_thresholds", {}).items():
        category_data = categories.get(category_name, {})
        trait_evaluations[category_name] = {}

        for trait, min_required in required_traits.items():
            actual_value = category_data.get(trait, 0.0)
            passed = actual_value >= min_required

            trait_evaluations[category_name][trait] = {
                "actual": actual_value,
                "required_minimum": min_required,
                "passed": passed
            }

            if not passed:
                all_thresholds_met = False

    selection_approved = classification_match and all_thresholds_met

    return {
        "target_id": evaluation_result.get("target_id"),
        "selection_approved": selection_approved,
        "classification_match": classification_match,
        "trait_breakdown": trait_evaluations,
        "recommended_action": evaluation_result.get("recommended_action")
    }
