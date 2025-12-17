#!/usr/bin/env python3
"""
CIB-Med-1 Demonstration Script

This script demonstrates the complete CIB-Med-1 benchmark workflow:
1. Initialize semantic coordinate system
2. Generate edit trajectories with constrained vs unconstrained guidance
3. Evaluate using CIB-Med-1 metrics
4. Analyze results and generate visualizations

Can be converted to Jupyter notebook with: jupytext --to notebook demo_notebook.py

Usage:
    python examples/demo_notebook.py
"""

# %% [markdown]
# # CIB-Med-1: Controlled Incremental Biomarker Editing Benchmark
#
# This notebook demonstrates the complete CIB-Med-1 evaluation pipeline.

# %% Imports and Setup
import sys
from pathlib import Path

# Add parent for development
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cib_med.core import SemanticCoordinateSystem, EditTrajectory
from cib_med.core.evaluator import MockEvaluator
from cib_med.core.anchor import Anchor
from cib_med.core.trajectory import TrajectoryStep
from cib_med.core.semantic_coordinates import SemanticCoordinates
from cib_med.metrics import CIBMedBenchmark
from cib_med.analysis.synthetic import SyntheticLatentModel, SyntheticFactors
from cib_med.utils import set_seed, get_device

# Configuration
SEED = 42
NUM_ANCHORS = 20
NUM_TRAJECTORY_STEPS = 15
OUTPUT_DIR = Path("/Users/ttt/Downloads/cib_med_outputs/demo")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

set_seed(SEED)
device = get_device()
print(f"Using device: {device}")

# %% [markdown]
# ## 1. Initialize the Semantic Coordinate System
#
# The coordinate system maps images to clinically interpretable scores using a frozen radiology evaluator.

# %% Initialize Components
print("=" * 60)
print("1. Initializing Semantic Coordinate System")
print("=" * 60)

# Use MockEvaluator for demonstration
# In production, replace with a real trained classifier
evaluator = MockEvaluator(num_labels=14, seed=SEED)
evaluator.to(device)

# Define the coordinate system
OFF_TARGET_FINDINGS = [
    "Atelectasis", "Consolidation", "Pneumonia", "Edema",
    "Lung Opacity", "Cardiomegaly", "Enlarged Cardiomediastinum",
    "Pneumothorax", "Fibrosis", "Support Devices"
]

coord_system = SemanticCoordinateSystem(
    evaluator=evaluator,
    target_finding="Pleural Effusion",
    off_target_findings=OFF_TARGET_FINDINGS,
)

print(f"Target: {coord_system.target_finding}")
print(f"Off-target findings ({len(OFF_TARGET_FINDINGS)}):")
for i, f in enumerate(OFF_TARGET_FINDINGS[:5]):
    print(f"  - {f}")
print(f"  ... and {len(OFF_TARGET_FINDINGS) - 5} more")

# %% [markdown]
# ## 2. Create Synthetic Anchors
#
# Anchors are starting images for edit trajectories. They should have
# non-saturated target scores (p_eff ∈ [0.1, 0.9]).

# %% Create Anchors
print("\n" + "=" * 60)
print("2. Creating Anchor Images")
print("=" * 60)

anchors = []
for i in range(NUM_ANCHORS):
    # Create synthetic CXR-like image
    image = torch.randn(1, 224, 224) * 0.12 + 0.5
    image = image.clamp(0, 1).to(device)
    
    # Compute semantic coordinates
    coords = coord_system.compute_coordinates(image)
    
    anchor = Anchor(
        image=image,
        coordinates=coords,
        source_id=f"demo_anchor_{i:03d}",
    )
    anchors.append(anchor)

print(f"Created {len(anchors)} anchors")
print(f"Sample anchor target score: {anchors[0].coordinates.calibrated_target:.3f}")

# %% [markdown]
# ## 3. Simulate Edit Trajectories
#
# We simulate two types of editors:
# - **Unconstrained**: Increases target but allows off-target drift
# - **Constrained**: Increases target while preserving off-target stability

# %% Trajectory Simulation Functions
def simulate_unconstrained_trajectory(anchor, num_steps=15):
    """
    Simulate unconstrained editing that exploits correlations.
    This mimics reward hacking behavior.
    """
    steps = []
    current_target = anchor.coordinates.calibrated_target
    current_off_target = {f: anchor.coordinates.probabilities.get(f, 0.3) 
                         for f in OFF_TARGET_FINDINGS}
    
    for t in range(num_steps + 1):
        # Target increases
        target = current_target + 0.03 * t + np.random.randn() * 0.01
        target = np.clip(target, 0, 1)
        
        # Off-target ALSO increases (reward hacking via correlation)
        probs = {}
        for f in OFF_TARGET_FINDINGS:
            drift = 0.015 * t  # Systematic drift
            noise = np.random.randn() * 0.005
            probs[f] = np.clip(current_off_target[f] + drift + noise, 0, 1)
        
        coords = SemanticCoordinates(
            raw_logits={},
            probabilities=probs,
            calibrated_target=float(target),
        )
        
        steps.append(TrajectoryStep(
            step=t,
            image=anchor.image,  # Placeholder
            coordinates=coords,
        ))
    
    return EditTrajectory(anchor=anchor, steps=steps, editor_name="unconstrained")


def simulate_constrained_trajectory(anchor, num_steps=15):
    """
    Simulate constrained editing that preserves off-target stability.
    """
    steps = []
    current_target = anchor.coordinates.calibrated_target
    anchor_off_target = {f: anchor.coordinates.probabilities.get(f, 0.3) 
                        for f in OFF_TARGET_FINDINGS}
    
    for t in range(num_steps + 1):
        # Target increases (slightly slower due to constraint)
        target = current_target + 0.025 * t + np.random.randn() * 0.01
        target = np.clip(target, 0, 1)
        
        # Off-target stays STABLE (constrained)
        probs = {}
        for f in OFF_TARGET_FINDINGS:
            # Small random fluctuation but no systematic drift
            noise = np.random.randn() * 0.01
            probs[f] = np.clip(anchor_off_target[f] + noise, 0, 1)
        
        coords = SemanticCoordinates(
            raw_logits={},
            probabilities=probs,
            calibrated_target=float(target),
        )
        
        steps.append(TrajectoryStep(
            step=t,
            image=anchor.image,
            coordinates=coords,
        ))
    
    return EditTrajectory(anchor=anchor, steps=steps, editor_name="constrained")


# %% Generate Trajectories
print("\n" + "=" * 60)
print("3. Generating Edit Trajectories")
print("=" * 60)

unconstrained_trajectories = []
constrained_trajectories = []

print("Generating unconstrained trajectories...")
for anchor in tqdm(anchors, desc="Unconstrained"):
    traj = simulate_unconstrained_trajectory(anchor, NUM_TRAJECTORY_STEPS)
    unconstrained_trajectories.append(traj)

print("Generating constrained trajectories...")
for anchor in tqdm(anchors, desc="Constrained"):
    traj = simulate_constrained_trajectory(anchor, NUM_TRAJECTORY_STEPS)
    constrained_trajectories.append(traj)

print(f"\nGenerated {len(unconstrained_trajectories)} unconstrained trajectories")
print(f"Generated {len(constrained_trajectories)} constrained trajectories")

# %% [markdown]
# ## 4. Evaluate with CIB-Med-1 Benchmark
#
# The benchmark computes:
# - **Progression metrics**: Trend correlation, inversion rate
# - **Drift metrics**: Per-label and aggregate off-target drift

# %% Run Benchmark
print("\n" + "=" * 60)
print("4. Running CIB-Med-1 Benchmark Evaluation")
print("=" * 60)

benchmark = CIBMedBenchmark(coord_system)

print("\nEvaluating unconstrained method...")
unc_results = benchmark.evaluate(unconstrained_trajectories, editor_name="unconstrained")

print("Evaluating constrained method...")
con_results = benchmark.evaluate(constrained_trajectories, editor_name="constrained")

# %% [markdown]
# ## 5. Analyze Results
#
# Compare progression and drift between methods.

# %% Display Results
print("\n" + "=" * 70)
print("5. CIB-MED-1 BENCHMARK RESULTS")
print("=" * 70)

print("\n┌" + "─" * 68 + "┐")
print("│{:^68}│".format("TARGET PROGRESSION METRICS"))
print("├" + "─" * 68 + "┤")
print("│ {:25} │ {:18} │ {:18} │".format("Metric", "Unconstrained", "Constrained"))
print("├" + "─" * 68 + "┤")

# Trend correlation
unc_trend = unc_results.progression_aggregate['trend_correlation']['mean']
con_trend = con_results.progression_aggregate['trend_correlation']['mean']
print("│ {:25} │ {:^18.4f} │ {:^18.4f} │".format("Trend Correlation (ρ)", unc_trend, con_trend))

# Inversion rate
unc_inv = unc_results.progression_aggregate['inversion_rate']['mean']
con_inv = con_results.progression_aggregate['inversion_rate']['mean']
print("│ {:25} │ {:^18.4f} │ {:^18.4f} │".format("Inversion Rate", unc_inv, con_inv))

# Total progression
unc_prog = unc_results.progression_aggregate['total_progression']['mean']
con_prog = con_results.progression_aggregate['total_progression']['mean']
print("│ {:25} │ {:^18.4f} │ {:^18.4f} │".format("Total Progression", unc_prog, con_prog))

print("└" + "─" * 68 + "┘")

print("\n┌" + "─" * 68 + "┐")
print("│{:^68}│".format("OFF-TARGET DRIFT METRICS"))
print("├" + "─" * 68 + "┤")
print("│ {:25} │ {:18} │ {:18} │".format("Metric", "Unconstrained", "Constrained"))
print("├" + "─" * 68 + "┤")

# Aggregate drift
unc_drift = unc_results.drift_aggregate['aggregate_drift_median']['mean']
con_drift = con_results.drift_aggregate['aggregate_drift_median']['mean']
print("│ {:25} │ {:^18.4f} │ {:^18.4f} │".format("Aggregate Drift (D_off)", unc_drift, con_drift))

# 90th percentile
unc_drift90 = unc_results.drift_aggregate['aggregate_drift_90th']['mean']
con_drift90 = con_results.drift_aggregate['aggregate_drift_90th']['mean']
print("│ {:25} │ {:^18.4f} │ {:^18.4f} │".format("90th Percentile Drift", unc_drift90, con_drift90))

print("└" + "─" * 68 + "┘")

# %% Key Findings
print("\n" + "=" * 70)
print("KEY FINDINGS")
print("=" * 70)

drift_reduction = (unc_drift - con_drift) / unc_drift * 100 if unc_drift > 0 else 0
prog_diff = abs(unc_prog - con_prog) / max(unc_prog, con_prog) * 100

print(f"""
1. TARGET PROGRESSION:
   Both methods achieve similar target progression
   ({unc_prog:.3f} vs {con_prog:.3f}, difference: {prog_diff:.1f}%)
   
   → Under target-only metrics, both would appear equally effective!

2. OFF-TARGET DRIFT:
   Unconstrained: {unc_drift:.4f}
   Constrained:   {con_drift:.4f}
   
   → Constrained reduces drift by {drift_reduction:.1f}%
   → This reveals the REWARD HACKING in unconstrained editing

3. INTERPRETATION:
   The unconstrained method achieves target progression by
   exploiting correlated off-target features (reward hacking).
   
   CIB-Med-1 makes this failure mode VISIBLE through drift metrics.
""")

# %% [markdown]
# ## 6. Visualization
#
# Generate plots for analysis.

# %% Create Visualization
try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Comparison bar chart
    ax1 = axes[0]
    methods = ['Unconstrained', 'Constrained']
    
    x = np.arange(2)
    width = 0.35
    
    # Normalize for comparison
    prog_vals = [unc_prog, con_prog]
    drift_vals = [unc_drift, con_drift]
    
    bars1 = ax1.bar(x - width/2, prog_vals, width, label='Progression', color='#3498DB')
    bars2 = ax1.bar(x + width/2, drift_vals, width, label='Drift', color='#E74C3C')
    
    ax1.set_ylabel('Score')
    ax1.set_title('CIB-Med-1: Progression vs Drift Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Efficiency ratio
    ax2 = axes[1]
    efficiency = [prog_vals[i] / (drift_vals[i] + 0.01) for i in range(2)]
    colors = ['#E74C3C', '#27AE60']
    
    bars = ax2.bar(methods, efficiency, color=colors, edgecolor='black')
    ax2.set_ylabel('Efficiency (Progression / Drift)')
    ax2.set_title('Semantic Control Efficiency')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, eff in zip(bars, efficiency):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{eff:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    save_path = OUTPUT_DIR / 'demo_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nFigure saved to: {save_path}")
    plt.close()
    
except Exception as e:
    print(f"Visualization skipped: {e}")

# %% Summary
print("\n" + "=" * 70)
print("DEMO COMPLETE")
print("=" * 70)
print(f"""
CIB-Med-1 successfully demonstrated:

- Semantic coordinate system initialization
- Trajectory generation (unconstrained vs constrained)
- Benchmark evaluation with progression and drift metrics
- Detection of reward hacking through off-target drift

Output saved to: {OUTPUT_DIR}

Next steps:
- Replace MockEvaluator with a real radiology classifier
- Load real CXR images from MIMIC-CXR or CheXpert
- Run ablation studies: python scripts/run_ablations.py
- Generate paper figures: python scripts/generate_figures.py
""")


if __name__ == "__main__":
    print("\nDemo notebook executed successfully.")
