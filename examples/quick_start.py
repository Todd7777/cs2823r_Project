#!/usr/bin/env python3
"""
CIB-Med-1 Quick Start Example

This example demonstrates the core functionality of the CIB-Med-1 benchmark:
1. Setting up the semantic coordinate system
2. Creating edit trajectories
3. Evaluating with CIB-Med-1 metrics
4. Comparing constrained vs unconstrained editing

Run this script to verify your installation and understand the API.
"""

import sys
from pathlib import Path

# Add parent to path for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

# Import CIB-Med-1 components
from cib_med.core import (
    SemanticCoordinateSystem,
    EditTrajectory,
    IsotonicCalibrator,
)
from cib_med.core.evaluator import MockEvaluator
from cib_med.core.anchor import Anchor
from cib_med.core.trajectory import TrajectoryStep
from cib_med.metrics import CIBMedBenchmark, MonotoneProgressionMetrics, OffTargetDriftMetrics
from cib_med.guidance import ConstrainedDiffusionGuidance, UnconstrainedGuidance
from cib_med.analysis.synthetic import SyntheticLatentModel, SyntheticFactors
from cib_med.utils import set_seed, get_device


def main():
    print("=" * 70)
    print("CIB-Med-1: Quick Start Example")
    print("=" * 70)
    
    # Set random seed for reproducibility
    set_seed(42)
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # =========================================================================
    # Step 1: Setup Evaluator and Coordinate System
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 1: Setting up Semantic Coordinate System")
    print("-" * 50)
    
    # Use MockEvaluator for demonstration (replace with real evaluator in production)
    evaluator = MockEvaluator(num_labels=14, seed=42)
    evaluator.to(device)
    
    # Create semantic coordinate system
    coord_system = SemanticCoordinateSystem(
        evaluator=evaluator,
        target_finding="Pleural Effusion",
        off_target_findings=[
            "Atelectasis", "Consolidation", "Pneumonia", "Edema",
            "Lung Opacity", "Cardiomegaly", "Enlarged Cardiomediastinum",
            "Pneumothorax", "Fibrosis", "Support Devices"
        ],
    )
    
    print(f"  Target finding: {coord_system.target_finding}")
    print(f"  Off-target findings: {len(coord_system.off_target_findings)}")
    
    # =========================================================================
    # Step 2: Create Synthetic Anchor Images
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 2: Creating Synthetic Anchor Images")
    print("-" * 50)
    
    num_anchors = 10
    anchors = []
    
    for i in range(num_anchors):
        # Create synthetic chest X-ray-like image
        image = torch.randn(1, 224, 224) * 0.15 + 0.5
        image = image.clamp(0, 1).to(device)
        
        # Compute semantic coordinates
        coords = coord_system.compute_coordinates(image)
        
        anchor = Anchor(
            image=image,
            coordinates=coords,
            source_id=f"demo_anchor_{i:03d}",
        )
        anchors.append(anchor)
    
    print(f"  Created {len(anchors)} anchor images")
    print(f"  Example anchor target score (p_eff): {anchors[0].coordinates.calibrated_target:.3f}")
    
    # =========================================================================
    # Step 3: Simulate Edit Trajectories
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 3: Simulating Edit Trajectories")
    print("-" * 50)
    
    def simulate_trajectory(anchor, editor_type="constrained", num_steps=15):
        """Simulate an edit trajectory using synthetic model."""
        steps = [TrajectoryStep(
            step=0,
            image=anchor.image,
            coordinates=anchor.coordinates,
        )]
        
        current_image = anchor.image.clone()
        
        for t in range(1, num_steps + 1):
            # Simulate editing step
            if editor_type == "unconstrained":
                # Unconstrained: increases target but drifts off-target
                noise = torch.randn_like(current_image) * 0.02
                # Add global darkening (mimics reward hacking via correlated features)
                current_image = current_image - 0.01 + noise
            else:
                # Constrained: increases target while preserving off-target
                noise = torch.randn_like(current_image) * 0.015
                # More targeted change in lower region only
                current_image = current_image + noise
                current_image[:, 150:, :] -= 0.008  # Subtle lower-region change
            
            current_image = current_image.clamp(0, 1)
            
            # Compute new coordinates
            coords = coord_system.compute_coordinates(current_image)
            
            steps.append(TrajectoryStep(
                step=t,
                image=current_image.clone(),
                coordinates=coords,
            ))
        
        return EditTrajectory(
            anchor=anchor,
            steps=steps,
            editor_name=editor_type,
        )
    
    # Generate trajectories for both methods
    unconstrained_trajectories = []
    constrained_trajectories = []
    
    print("  Generating unconstrained trajectories...")
    for anchor in tqdm(anchors, desc="  Unconstrained"):
        traj = simulate_trajectory(anchor, editor_type="unconstrained")
        unconstrained_trajectories.append(traj)
    
    print("  Generating constrained trajectories...")
    for anchor in tqdm(anchors, desc="  Constrained"):
        traj = simulate_trajectory(anchor, editor_type="constrained")
        constrained_trajectories.append(traj)
    
    print(f"  Generated {len(unconstrained_trajectories)} unconstrained trajectories")
    print(f"  Generated {len(constrained_trajectories)} constrained trajectories")
    
    # =========================================================================
    # Step 4: Evaluate with CIB-Med-1 Metrics
    # =========================================================================
    print("\n" + "-" * 50)
    print("Step 4: Evaluating with CIB-Med-1 Metrics")
    print("-" * 50)
    
    # Initialize benchmark
    benchmark = CIBMedBenchmark(coord_system)
    
    # Evaluate both methods
    print("\n  Evaluating unconstrained method...")
    unc_results = benchmark.evaluate(unconstrained_trajectories, editor_name="unconstrained")
    
    print("  Evaluating constrained method...")
    con_results = benchmark.evaluate(constrained_trajectories, editor_name="constrained")
    
    # =========================================================================
    # Step 5: Display Results
    # =========================================================================
    print("\n" + "=" * 70)
    print("RESULTS: CIB-Med-1 Benchmark Comparison")
    print("=" * 70)
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    TARGET PROGRESSION METRICS                    │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric               │ Unconstrained     │ Constrained         │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    unc_trend = unc_results.progression_aggregate['trend_correlation']['mean']
    con_trend = con_results.progression_aggregate['trend_correlation']['mean']
    print(f"│ Trend Correlation    │ {unc_trend:>8.3f}          │ {con_trend:>8.3f}            │")
    
    unc_inv = unc_results.progression_aggregate['inversion_rate']['mean']
    con_inv = con_results.progression_aggregate['inversion_rate']['mean']
    print(f"│ Inversion Rate       │ {unc_inv:>8.3f}          │ {con_inv:>8.3f}            │")
    
    unc_prog = unc_results.progression_aggregate['total_progression']['mean']
    con_prog = con_results.progression_aggregate['total_progression']['mean']
    print(f"│ Total Progression    │ {unc_prog:>8.3f}          │ {con_prog:>8.3f}            │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    print("\n┌─────────────────────────────────────────────────────────────────┐")
    print("│                    OFF-TARGET DRIFT METRICS                      │")
    print("├─────────────────────────────────────────────────────────────────┤")
    print(f"│ Metric               │ Unconstrained     │ Constrained         │")
    print("├─────────────────────────────────────────────────────────────────┤")
    
    unc_drift = unc_results.drift_aggregate['aggregate_drift_median']['mean']
    con_drift = con_results.drift_aggregate['aggregate_drift_median']['mean']
    print(f"│ Aggregate Drift      │ {unc_drift:>8.4f}          │ {con_drift:>8.4f}            │")
    
    unc_drift90 = unc_results.drift_aggregate['aggregate_drift_90th']['mean']
    con_drift90 = con_results.drift_aggregate['aggregate_drift_90th']['mean']
    print(f"│ 90th Percentile      │ {unc_drift90:>8.4f}          │ {con_drift90:>8.4f}            │")
    print("└─────────────────────────────────────────────────────────────────┘")
    
    # Summary
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    
    if unc_drift > con_drift * 1.5:
        print("\n[+] Constrained editing achieves LOWER OFF-TARGET DRIFT")
        print(f"  → Drift reduction: {((unc_drift - con_drift) / unc_drift * 100):.1f}%")
    
    if abs(unc_prog - con_prog) < 0.1:
        print("\n[+] Both methods achieve SIMILAR TARGET PROGRESSION")
        print("  → Target-only metrics would not distinguish them!")
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT: CIB-Med-1 reveals reward hacking that")
    print("target-only metrics miss. The unconstrained method")
    print("achieves progression by exploiting correlated features,")
    print("inducing off-target drift that violates semantic stability.")
    print("=" * 70)
    
    print("\nQuick start example complete.")
    print(f"\nNext steps:")
    print("  1. Replace MockEvaluator with a real radiology classifier")
    print("  2. Load real CXR anchors from MIMIC-CXR or CheXpert")
    print("  3. Run the full benchmark: python scripts/run_benchmark.py")
    print("  4. Explore ablation studies: python scripts/run_ablations.py")


if __name__ == "__main__":
    main()
