#!/usr/bin/env python3
"""
CIB-Med-1 Stress Test Runner

Runs synthetic and semi-synthetic stress tests as described in Section 7:
- Semi-synthetic off-target injection
- Synthetic latent-factor model
- Coupling strength analysis

Usage:
    python scripts/run_stress_tests.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from cib_med.core import SemanticCoordinateSystem
from cib_med.core.evaluator import MockEvaluator
from cib_med.analysis.synthetic import (
    SyntheticLatentModel,
    SemiSyntheticPerturbation,
    StressTestSuite,
    SyntheticFactors,
)
from cib_med.utils import set_seed, get_device, setup_logger, save_results


def run_coupling_test(config: dict, output_dir: Path, logger):
    """Run synthetic coupling test (Section 7.2)."""
    logger.info("=" * 60)
    logger.info("Running Synthetic Coupling Test")
    logger.info("=" * 60)
    
    stress_config = config.get("stress_tests", {})
    rho_values = stress_config.get("coupling_rhos", [0.0, 0.3, 0.5, 0.7, 1.0])
    
    results = {}
    
    for rho in tqdm(rho_values, desc="Testing coupling strengths"):
        model = SyntheticLatentModel(coupling_rho=rho, seed=42)
        
        # Run multiple trajectories
        all_metrics = []
        for i in range(50):
            initial = SyntheticFactors(
                z_eff=np.random.uniform(0.2, 0.4),
                z_inf=np.random.uniform(0.1, 0.3),
                z_art=np.random.uniform(0.0, 0.1),
            )
            
            # Unconstrained trajectory
            traj = model.simulate_unconstrained_edit(initial, num_steps=15)
            metrics = model.compute_trajectory_metrics(traj)
            all_metrics.append(metrics)
        
        # Aggregate
        results[f"rho_{rho}"] = {
            "coupling_rho": rho,
            "mean_progression": float(np.mean([m["progression"] for m in all_metrics])),
            "std_progression": float(np.std([m["progression"] for m in all_metrics])),
            "mean_drift": float(np.mean([m["aggregate_drift"] for m in all_metrics])),
            "std_drift": float(np.std([m["aggregate_drift"] for m in all_metrics])),
            "mean_drift_infection": float(np.mean([m["drift_infection"] for m in all_metrics])),
            "mean_inversion_rate": float(np.mean([m["inversion_rate"] for m in all_metrics])),
        }
        
        logger.info(
            f"  ρ={rho:.1f}: progression={results[f'rho_{rho}']['mean_progression']:.3f}, "
            f"drift={results[f'rho_{rho}']['mean_drift']:.3f}"
        )
    
    return results


def run_constraint_effectiveness_test(config: dict, output_dir: Path, logger):
    """Test how constraint strength affects drift (Section 7.2)."""
    logger.info("=" * 60)
    logger.info("Running Constraint Effectiveness Test")
    logger.info("=" * 60)
    
    stress_config = config.get("stress_tests", {})
    lambda_values = stress_config.get("constraint_lambdas", [0.0, 0.5, 1.0, 2.0, 5.0])
    
    # Use moderate coupling
    model = SyntheticLatentModel(coupling_rho=0.5, seed=42)
    
    results = {}
    
    for lam in tqdm(lambda_values, desc="Testing constraint strengths"):
        all_metrics = []
        
        for i in range(50):
            initial = SyntheticFactors(
                z_eff=np.random.uniform(0.2, 0.4),
                z_inf=np.random.uniform(0.1, 0.3),
                z_art=np.random.uniform(0.0, 0.1),
            )
            
            # Constrained trajectory
            traj = model.simulate_constrained_edit(
                initial,
                num_steps=15,
                constraint_strength=lam,
            )
            metrics = model.compute_trajectory_metrics(traj)
            all_metrics.append(metrics)
        
        results[f"lambda_{lam}"] = {
            "lambda": lam,
            "mean_progression": float(np.mean([m["progression"] for m in all_metrics])),
            "std_progression": float(np.std([m["progression"] for m in all_metrics])),
            "mean_drift": float(np.mean([m["aggregate_drift"] for m in all_metrics])),
            "std_drift": float(np.std([m["aggregate_drift"] for m in all_metrics])),
        }
        
        logger.info(
            f"  λ={lam:.1f}: progression={results[f'lambda_{lam}']['mean_progression']:.3f}, "
            f"drift={results[f'lambda_{lam}']['mean_drift']:.3f}"
        )
    
    return results


def run_semi_synthetic_test(config: dict, output_dir: Path, logger, device):
    """Run semi-synthetic perturbation test (Section 7.1)."""
    logger.info("=" * 60)
    logger.info("Running Semi-Synthetic Perturbation Test")
    logger.info("=" * 60)
    
    stress_config = config.get("stress_tests", {})
    gammas = stress_config.get("perturbation_gammas", [0.0, 0.05, 0.1, 0.15, 0.2])
    
    perturbation = SemiSyntheticPerturbation(
        image_size=(224, 224),
        device=str(device),
    )
    
    # Create synthetic anchor
    anchor = torch.randn(1, 224, 224) * 0.1 + 0.5
    anchor = anchor.clamp(0, 1).to(device)
    
    results = {"locations": {}}
    
    for location in ["upper_lobe", "lower_lobe", "cardiac"]:
        logger.info(f"  Testing location: {location}")
        
        # Generate perturbation sweep
        perturbed = perturbation.generate_perturbation_sweep(
            anchor, gammas, location=location
        )
        
        location_results = []
        for gamma, img in perturbed:
            # Compute simple metrics (difference from anchor)
            diff = (img - anchor).abs().mean().item()
            location_results.append({
                "gamma": gamma,
                "mean_diff": diff,
            })
            logger.info(f"    γ={gamma:.2f}: mean_diff={diff:.4f}")
        
        results["locations"][location] = location_results
    
    return results


def run_stress_tests(config: dict, output_dir: Path):
    """Run all stress tests."""
    logger = setup_logger("stress_tests", log_file=output_dir / "stress_tests.log")
    
    set_seed(config.get("reproducibility", {}).get("seed", 42))
    device = get_device()
    logger.info(f"Using device: {device}")
    
    all_results = {}
    
    # Run coupling test
    logger.info("\n" + "=" * 70)
    logger.info("STRESS TESTS (Section 7)")
    logger.info("=" * 70 + "\n")
    
    all_results["coupling_test"] = run_coupling_test(config, output_dir, logger)
    
    # Run constraint effectiveness test
    all_results["constraint_test"] = run_constraint_effectiveness_test(
        config, output_dir, logger
    )
    
    # Run semi-synthetic test
    all_results["semi_synthetic_test"] = run_semi_synthetic_test(
        config, output_dir, logger, device
    )
    
    # Generate summary
    logger.info("\n" + "=" * 70)
    logger.info("STRESS TEST SUMMARY")
    logger.info("=" * 70)
    
    # Coupling test summary
    logger.info("\n1. Coupling Test (ρ = 0 → 1):")
    logger.info("   As coupling increases, unconstrained editing exploits shortcuts.")
    for key in sorted(all_results["coupling_test"].keys()):
        r = all_results["coupling_test"][key]
        logger.info(f"   ρ={r['coupling_rho']:.1f}: drift increases {r['mean_drift']:.3f}")
    
    # Constraint test summary
    logger.info("\n2. Constraint Effectiveness (λ = 0 → 5):")
    logger.info("   Stronger constraints reduce drift but may limit progression.")
    for key in sorted(all_results["constraint_test"].keys()):
        r = all_results["constraint_test"][key]
        logger.info(f"   λ={r['lambda']:.1f}: drift={r['mean_drift']:.3f}, prog={r['mean_progression']:.3f}")
    
    # Save results
    save_results(all_results, output_dir / "stress_test_results.json")
    logger.info(f"\nResults saved to {output_dir}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CIB-Med-1 Stress Tests")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "default.yaml",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.get("data", {}).get(
            "output_dir", "/Users/ttt/Downloads/cib_med_outputs"
        ))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"stress_tests_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_stress_tests(config, output_dir)


if __name__ == "__main__":
    main()
