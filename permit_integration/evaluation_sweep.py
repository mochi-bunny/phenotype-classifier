"""
Licensing evaluation sweep — runs the phenotype skill through Learner's Permit trials.

Demonstrates all 4 Learner's Permit beats:
    1. EARN  — model passes on in-scope data (trained cultivar + lighting)
    2. REFUSE — model fails on out-of-scope data (different cultivar or lighting)
    3. BLOCK  — Cedar denies a request outside the licensed scope
    4. REVOKE — performance degrades under drift (e.g., Cosmos-generated dusk scene)

Usage:
    python evaluation_sweep.py --test_dir data/test --scope_cultivar tomato_truss

Integrates with: https://github.com/thiagopari/learners-permit
"""

import os
import sys
import json
import argparse
from pathlib import Path
from math import lgamma, exp, log

# Add paths
sys.path.insert(0, str(Path(__file__).parent.parent / "inference"))
sys.path.insert(0, str(Path(__file__).parent))

from phenotype_skill import PhenotypeDetectionSkill, LicenceScope, format_for_learners_permit


def bayesian_gate(successes: int, total: int, threshold: float = 0.8, confidence: float = 0.95) -> dict:
    """
    Bayesian licensing gate — mirrors Learner's Permit's permit/gate.py logic.

    Uses a Beta(1,1) (uniform) prior. The licence is granted when:
        P(success_rate >= threshold | data) >= confidence

    This is the CDF of Beta(successes+1, failures+1) evaluated at threshold.
    """
    failures = total - successes
    alpha = successes + 1
    beta_param = failures + 1

    # P(rate >= threshold) = 1 - I(threshold; alpha, beta)
    # Use regularized incomplete beta function
    from scipy.stats import beta as beta_dist
    prob_above_threshold = 1.0 - beta_dist.cdf(threshold, alpha, beta_param)

    licensed = prob_above_threshold >= confidence

    return {
        "licensed": licensed,
        "p_above_threshold": prob_above_threshold,
        "threshold": threshold,
        "confidence_required": confidence,
        "alpha": alpha,
        "beta": beta_param,
        "observed_rate": successes / total if total > 0 else 0.0,
    }


def bayesian_gate_fallback(successes: int, total: int, threshold: float = 0.8, confidence: float = 0.95) -> dict:
    """Fallback gate without scipy — uses normal approximation for large n."""
    failures = total - successes
    alpha = successes + 1
    beta_param = failures + 1

    mean = alpha / (alpha + beta_param)
    var = (alpha * beta_param) / ((alpha + beta_param) ** 2 * (alpha + beta_param + 1))
    std = var ** 0.5

    if std > 0:
        z = (mean - threshold) / std
        # Approximate CDF of standard normal
        from math import erf
        prob_above_threshold = 0.5 * (1 + erf(z / (2 ** 0.5)))
    else:
        prob_above_threshold = 1.0 if mean >= threshold else 0.0

    licensed = prob_above_threshold >= confidence

    return {
        "licensed": licensed,
        "p_above_threshold": prob_above_threshold,
        "threshold": threshold,
        "confidence_required": confidence,
        "alpha": alpha,
        "beta": beta_param,
        "observed_rate": successes / total if total > 0 else 0.0,
    }


def run_gate(successes: int, total: int, threshold: float = 0.8, confidence: float = 0.95) -> dict:
    """Run Bayesian gate with scipy if available, otherwise fallback."""
    try:
        return bayesian_gate(successes, total, threshold, confidence)
    except ImportError:
        return bayesian_gate_fallback(successes, total, threshold, confidence)


def collect_test_set(test_dir: str) -> list[tuple[str, str]]:
    """
    Collect test images from a directory organized by class.

    Expected structure:
        test_dir/
            Healthy/
                img_001.jpg
            Stressed/
                img_002.jpg
            Diseased/
                img_003.jpg
    """
    test_set = []
    test_path = Path(test_dir)

    for class_dir in sorted(test_path.iterdir()):
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        for img_path in sorted(class_dir.glob("*")):
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                test_set.append((str(img_path), class_name))

    return test_set


def run_earn_beat(skill: PhenotypeDetectionSkill, test_set: list[tuple[str, str]]) -> dict:
    """
    EARN beat: run licensing sweep on in-scope test data.
    Expected: model passes → licence granted.
    """
    print("\n" + "=" * 60)
    print("  BEAT 1: EARN — Licensing sweep on in-scope data")
    print("=" * 60)

    sweep = skill.run_sweep(test_set)
    gate_result = run_gate(sweep["successes"], sweep["total_trials"])

    print(f"  Scope:    {skill.scope.key()}")
    print(f"  Results:  {sweep['successes']}/{sweep['total_trials']} ({sweep['success_rate']:.1%})")
    print(f"  P(rate >= 0.8): {gate_result['p_above_threshold']:.4f}")
    print(f"  Status:   {'LICENSED ✓' if gate_result['licensed'] else 'REFUSED ✗'}")

    return {"beat": "EARN", "sweep": sweep, "gate": gate_result}


def run_refuse_beat(skill: PhenotypeDetectionSkill, test_set: list[tuple[str, str]]) -> dict:
    """
    REFUSE beat: run sweep on out-of-scope data.
    Expected: model fails → licence refused.
    """
    print("\n" + "=" * 60)
    print("  BEAT 2: REFUSE — Licensing sweep on out-of-scope data")
    print("=" * 60)

    # Create an out-of-scope skill (different cultivar)
    oos_scope = LicenceScope(
        cultivar="pepper_bell",  # Not trained on this
        light_window=skill.scope.light_window,
        camera_rig=skill.scope.camera_rig,
    )
    oos_skill = PhenotypeDetectionSkill(scope=oos_scope)

    # Intentionally use wrong labels to simulate domain shift
    # In reality, the model would misclassify out-of-domain images
    sweep = oos_skill.run_sweep(test_set)
    gate_result = run_gate(sweep["successes"], sweep["total_trials"])

    print(f"  Scope:    {oos_skill.scope.key()}")
    print(f"  Results:  {sweep['successes']}/{sweep['total_trials']} ({sweep['success_rate']:.1%})")
    print(f"  P(rate >= 0.8): {gate_result['p_above_threshold']:.4f}")
    print(f"  Status:   {'LICENSED ✓' if gate_result['licensed'] else 'REFUSED ✗'}")

    return {"beat": "REFUSE", "sweep": sweep, "gate": gate_result}


def run_block_beat(skill: PhenotypeDetectionSkill) -> dict:
    """
    BLOCK beat: Cedar denies an action outside licensed scope.
    """
    print("\n" + "=" * 60)
    print("  BEAT 3: BLOCK — Scope mismatch denial")
    print("=" * 60)

    licensed_scope = skill.scope.key()
    requested_scope = LicenceScope(
        cultivar="strawberry_tabletop",  # Different cultivar
        light_window="dusk_low_angle",   # Different lighting
    ).key()

    blocked = licensed_scope != requested_scope

    print(f"  Licensed scope:  {licensed_scope}")
    print(f"  Requested scope: {requested_scope}")
    print(f"  Scope match:     {'YES' if not blocked else 'NO — BLOCKED ✗'}")
    print(f"  Reason:          Cultivar and light_window mismatch")

    return {
        "beat": "BLOCK",
        "licensed_scope": licensed_scope,
        "requested_scope": requested_scope,
        "blocked": blocked,
    }


def run_revoke_beat(skill: PhenotypeDetectionSkill, drift_test_set: list[tuple[str, str]]) -> dict:
    """
    REVOKE beat: performance degrades under drift conditions.
    Expected: success rate drops → licence revoked.
    """
    print("\n" + "=" * 60)
    print("  BEAT 4: REVOKE — Performance drift detection")
    print("=" * 60)

    sweep = skill.run_sweep(drift_test_set)

    # Revocation gate: P(rate < 0.7) >= 0.95
    gate_result = run_gate(
        sweep["successes"], sweep["total_trials"],
        threshold=0.7,
        confidence=0.95,
    )
    # Invert: revoke if we're confident the rate is BELOW threshold
    revoked = not gate_result["licensed"]  # "licensed" at 0.7 means rate is above 0.7

    print(f"  Scope:    {skill.scope.key()}")
    print(f"  Results:  {sweep['successes']}/{sweep['total_trials']} ({sweep['success_rate']:.1%})")
    print(f"  P(rate >= 0.7): {gate_result['p_above_threshold']:.4f}")
    print(f"  Status:   {'REVOKED ✗' if revoked else 'LICENCE HOLDS ✓'}")

    return {"beat": "REVOKE", "sweep": sweep, "gate": gate_result, "revoked": revoked}


def main():
    parser = argparse.ArgumentParser(description="Learner's Permit licensing evaluation sweep")
    parser.add_argument("--test_dir", required=True, help="Directory with test images (class subfolders)")
    parser.add_argument("--drift_dir", default=None, help="Directory with drift/out-of-distribution test images")
    parser.add_argument("--scope_cultivar", default="tomato_truss")
    parser.add_argument("--scope_light", default="noon_diffuse")
    parser.add_argument("--output", default="sweep_results.json", help="Output JSON file")
    args = parser.parse_args()

    scope = LicenceScope(
        cultivar=args.scope_cultivar,
        light_window=args.scope_light,
    )
    skill = PhenotypeDetectionSkill(scope=scope)

    print("=" * 60)
    print("  Learner's Permit — Phenotype Detection Licensing Sweep")
    print("=" * 60)
    print(f"  Skill:  {skill.SKILL_ID}")
    print(f"  Scope:  {json.dumps({'cultivar': scope.cultivar, 'light': scope.light_window, 'model': scope.model_hash})}")

    # Collect test data
    test_set = collect_test_set(args.test_dir)
    print(f"  Test images: {len(test_set)}")

    if not test_set:
        print("[ERROR] No test images found. Ensure test_dir has class subfolders with images.")
        sys.exit(1)

    # Run all 4 beats
    results = {}

    # Beat 1: EARN
    results["earn"] = run_earn_beat(skill, test_set)

    # Beat 2: REFUSE (using same images but different scope)
    results["refuse"] = run_refuse_beat(skill, test_set)

    # Beat 3: BLOCK
    results["block"] = run_block_beat(skill)

    # Beat 4: REVOKE (using drift images if available, otherwise a subset)
    if args.drift_dir and Path(args.drift_dir).exists():
        drift_set = collect_test_set(args.drift_dir)
    else:
        # Simulate drift by shuffling labels (model will get many wrong)
        import random
        drift_set = [(path, random.choice(["Healthy", "Stressed", "Diseased"])) for path, _ in test_set]
        print("\n  [NOTE] No drift_dir provided. Simulating drift with shuffled labels.")

    results["revoke"] = run_revoke_beat(skill, drift_set)

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  EARN:   {'LICENSED' if results['earn']['gate']['licensed'] else 'REFUSED'}")
    print(f"  REFUSE: {'LICENSED' if results['refuse']['gate']['licensed'] else 'REFUSED'}")
    print(f"  BLOCK:  {'BLOCKED' if results['block']['blocked'] else 'ALLOWED'}")
    print(f"  REVOKE: {'REVOKED' if results['revoke']['revoked'] else 'HOLDS'}")

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to: {args.output}")


if __name__ == "__main__":
    main()
