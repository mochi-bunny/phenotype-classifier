"""
Phenotype Detection Skill — Learner's Permit bridge.

Wraps the trained phenotype model as a licensed skill that integrates
with the Learner's Permit trust gate (https://github.com/thiagopari/learners-permit).

Each inference is a trial: the model's prediction is compared against
ground truth, and pass/fail results feed into the Bayesian licensing gate.

Licence scope dimensions (per research doc §7.2):
    - cultivar (mandatory)       — e.g., "tomato_truss", "tomato_cherry"
    - light_window (mandatory)   — e.g., "noon_diffuse", "dawn_golden"
    - camera_rig (mandatory)     — e.g., "wrist_rgb_v1"
    - model_hash (mandatory)     — SHA-256 prefix of ONNX model
    - occlusion_class            — e.g., "none", "low", "medium", "high"
    - canopy_stage               — e.g., "early", "mid", "late"
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

# Add inference dir to path for model imports
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))

from phenotype_detector import predict, load_image_from_source, _load_model


@dataclass
class LicenceScope:
    """Defines the conditions under which a skill licence is valid."""
    skill_id: str = "phenotype_detect"
    cultivar: str = "tomato_truss"
    light_window: str = "noon_diffuse"
    camera_rig: str = "wrist_rgb_v1"
    model_hash: str = "unknown"
    occlusion_class: str = "none"
    canopy_stage: str = "mid"

    def key(self) -> str:
        """Unique licence key for the Bayesian gate."""
        return f"{self.skill_id}:{self.cultivar}:{self.light_window}:{self.camera_rig}:{self.model_hash}"


@dataclass
class TrialResult:
    """Result of a single licensing trial."""
    image_path: str
    ground_truth: str
    predicted_class: str
    confidence: float
    passed: bool
    scope: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class PhenotypeDetectionSkill:
    """
    Licensed phenotype detection skill for Learner's Permit.

    Usage:
        skill = PhenotypeDetectionSkill()
        result = skill.run_trial("test/img_001.jpg", "Healthy")
        # result.passed → True/False for the Bayesian gate
    """

    SKILL_ID = "phenotype_detect"

    def __init__(self, scope: Optional[LicenceScope] = None):
        _load_model()
        from phenotype_detector import _model_hash
        self.scope = scope or LicenceScope()
        self.scope.model_hash = _model_hash or "unknown"

    def predict(self, image_path: str) -> dict:
        """Run inference and return raw prediction."""
        image = load_image_from_source(image_path)
        return predict(image)

    def run_trial(self, image_path: str, ground_truth: str) -> TrialResult:
        """
        Single licensing trial.

        Compares model prediction against ground truth label.
        Returns a TrialResult with pass/fail for the Bayesian gate.
        """
        prediction = self.predict(image_path)
        passed = prediction["class"] == ground_truth

        return TrialResult(
            image_path=image_path,
            ground_truth=ground_truth,
            predicted_class=prediction["class"],
            confidence=prediction["confidence"],
            passed=passed,
            scope=asdict(self.scope),
        )

    def run_sweep(self, test_set: list[tuple[str, str]]) -> dict:
        """
        Run a full licensing sweep over a test set.

        Args:
            test_set: list of (image_path, ground_truth_label) tuples

        Returns:
            Sweep summary compatible with Learner's Permit gate input.
        """
        results = []
        for image_path, gt_label in test_set:
            result = self.run_trial(image_path, gt_label)
            results.append(result)

        successes = sum(1 for r in results if r.passed)
        total = len(results)
        success_rate = successes / total if total > 0 else 0.0

        return {
            "skill_id": self.SKILL_ID,
            "scope": asdict(self.scope),
            "total_trials": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": success_rate,
            "trials": [r.to_dict() for r in results],
            # Learner's Permit gate input format:
            "gate_input": {
                "skill": self.scope.key(),
                "results": [r.passed for r in results],
                "n_success": successes,
                "n_total": total,
            },
        }


def format_for_learners_permit(sweep_result: dict) -> dict:
    """
    Format sweep results for direct input to Learner's Permit trust gate.

    Maps to the format expected by permit/gate.py:
        - skill name (scoped key)
        - list of boolean trial results
        - threshold: 0.8 (supervised) or 0.99 (unattended)
    """
    return {
        "skill": sweep_result["gate_input"]["skill"],
        "trial_results": sweep_result["gate_input"]["results"],
        "threshold": 0.8,  # supervised tier
        "confidence_required": 0.95,  # P(rate >= threshold) >= this
    }


if __name__ == "__main__":
    print("Phenotype Detection Skill — Learner's Permit Bridge")
    print(f"  Skill ID: {PhenotypeDetectionSkill.SKILL_ID}")

    skill = PhenotypeDetectionSkill()
    print(f"  Scope: {json.dumps(asdict(skill.scope), indent=2)}")
    print(f"  Licence key: {skill.scope.key()}")
    print("\nReady for licensing trials. Use evaluation_sweep.py to run.")
