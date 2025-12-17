#!/usr/bin/env python3
"""
CIB-Med-1 Benchmark Runner

Main script for running the complete CIB-Med-1 benchmark evaluation.
Supports multiple editing methods and generates comprehensive reports.

Usage:
    python scripts/run_benchmark.py --config configs/default.yaml
    python scripts/run_benchmark.py --experiment main_comparison
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from tqdm import tqdm

from cib_med.core import (
    SemanticCoordinateSystem,
    RadiologyEvaluator,
    EditTrajectory,
    TrajectoryGenerator,
    AnchorSelector,
    IsotonicCalibrator,
)
from cib_med.metrics import CIBMedBenchmark, MonotoneProgressionMetrics, OffTargetDriftMetrics
from cib_med.guidance import ConstrainedDiffusionGuidance, UnconstrainedGuidance
from cib_med.models import DiffusionEditor, DDPMScheduler, UNet2DModel
from cib_med.data import CXRDataset, get_cxr_transforms
from cib_med.visualization import FigureGenerator
from cib_med.utils import set_seed, get_device, setup_logger, save_results


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def setup_evaluator(config: dict, device: torch.device):
    """Initialize the radiology evaluator."""
    evaluator_config = config.get("evaluator", {})
    
    model_type = evaluator_config.get("model_type", "densenet121")
    weights_path = evaluator_config.get("weights_path")
    
    if weights_path and Path(weights_path).exists():
        evaluator = RadiologyEvaluator.from_pretrained(
            model_type, weights_path=weights_path
        )
    else:
        # Use mock evaluator for demo/testing
        from cib_med.core.evaluator import MockEvaluator
        evaluator = MockEvaluator(num_labels=14)
    
    evaluator.to(device)
    return evaluator


def setup_coordinate_system(evaluator, config: dict):
    """Initialize the semantic coordinate system."""
    coord_config = config.get("coordinates", {})
    
    return SemanticCoordinateSystem(
        evaluator=evaluator,
        target_finding=coord_config.get("target_finding", "Pleural Effusion"),
        off_target_findings=coord_config.get("off_target_findings"),
    )


def load_anchors(config: dict, coord_system, device: torch.device):
    """Load or generate anchor images."""
    data_config = config.get("data", {})
    anchor_config = config.get("anchors", {})
    
    root_dir = Path(data_config.get("root_dir", "/Users/ttt/Downloads/cxr_data"))
    
    if root_dir.exists():
        # Load from dataset
        transform = get_cxr_transforms(augment=False, normalize=True)
        dataset = CXRDataset(
            root_dir=str(root_dir),
            split="test",
            transform=transform,
        )
        
        selector = AnchorSelector(
            coord_system=coord_system,
            target_range=tuple(anchor_config.get("target_range", [0.1, 0.9])),
        )
        
        anchor_set = selector.select_anchors(
            dataset,
            num_anchors=anchor_config.get("num_anchors", 100),
        )
        
        return anchor_set.anchors
    else:
        # Generate synthetic anchors for demo
        print(f"Warning: Data directory {root_dir} not found. Using synthetic anchors.")
        
        num_anchors = anchor_config.get("num_anchors", 10)
        anchors = []
        
        for i in range(num_anchors):
            # Create synthetic image
            image = torch.randn(1, 224, 224) * 0.1 + 0.5
            image = image.clamp(0, 1).to(device)
            
            # Compute coordinates
            coords = coord_system.compute_coordinates(image)
            
            from cib_med.core.anchor import Anchor
            anchors.append(Anchor(
                image=image,
                coordinates=coords,
                source_id=f"synthetic_{i:04d}",
            ))
        
        return anchors


def setup_editors(config: dict, evaluator, coord_system, device: torch.device):
    """Setup editing methods for comparison."""
    editors = {}
    
    # Create base diffusion model
    model = UNet2DModel(in_channels=1, out_channels=1, base_channels=64)
    model = model.to(device)
    scheduler = DDPMScheduler()
    
    # Unconstrained baseline
    unconstrained_guidance = UnconstrainedGuidance(
        evaluator=evaluator,
        target_finding="Pleural Effusion",
        guidance_scale=config.get("unconstrained_guidance", {}).get("guidance_scale", 7.5),
    )
    
    editors["unconstrained"] = DiffusionEditor(
        model=model,
        scheduler=scheduler,
        guidance_method=unconstrained_guidance,
        device=str(device),
    )
    
    # Constrained guidance
    constrained_config = config.get("constrained_guidance", {})
    constrained_guidance = ConstrainedDiffusionGuidance(
        evaluator=evaluator,
        target_finding="Pleural Effusion",
        off_target_findings=coord_system.off_target_findings,
        lambda_constraint=constrained_config.get("lambda_constraint", 1.0),
        alpha_variance=constrained_config.get("alpha_variance", 0.5),
        beta_anchor=constrained_config.get("beta_anchor", 0.3),
    )
    
    editors["constrained"] = DiffusionEditor(
        model=model,
        scheduler=scheduler,
        guidance_method=constrained_guidance,
        device=str(device),
    )
    
    return editors


def generate_trajectories(
    editor,
    anchors,
    coord_system,
    config: dict,
    editor_name: str,
):
    """Generate edit trajectories for all anchors."""
    traj_config = config.get("trajectory", {})
    num_steps = traj_config.get("num_steps", 20)
    
    trajectories = []
    
    for anchor in tqdm(anchors, desc=f"Generating trajectories ({editor_name})"):
        # Generate trajectory
        images = editor.generate_trajectory(
            anchor.image,
            num_steps=num_steps,
            step_noise_level=traj_config.get("noise_level", 0.1),
        )
        
        # Build EditTrajectory
        from cib_med.core.trajectory import TrajectoryStep
        
        steps = []
        for t, img in enumerate(images):
            coords = coord_system.compute_coordinates(img)
            steps.append(TrajectoryStep(
                step=t,
                image=img,
                coordinates=coords,
            ))
        
        trajectory = EditTrajectory(
            anchor=anchor,
            steps=steps,
            editor_name=editor_name,
        )
        trajectories.append(trajectory)
    
    return trajectories


def run_benchmark(config: dict, output_dir: Path):
    """Run the complete benchmark."""
    logger = setup_logger("cib_med", log_file=output_dir / "benchmark.log")
    
    # Setup
    set_seed(config.get("reproducibility", {}).get("seed", 42))
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize components
    logger.info("Initializing evaluator...")
    evaluator = setup_evaluator(config, device)
    
    logger.info("Setting up coordinate system...")
    coord_system = setup_coordinate_system(evaluator, config)
    
    logger.info("Loading anchors...")
    anchors = load_anchors(config, coord_system, device)
    logger.info(f"Loaded {len(anchors)} anchors")
    
    logger.info("Setting up editors...")
    editors = setup_editors(config, evaluator, coord_system, device)
    
    # Initialize benchmark
    benchmark = CIBMedBenchmark(coord_system)
    
    # Run evaluation for each editor
    all_results = {}
    
    for editor_name, editor in editors.items():
        logger.info(f"Evaluating: {editor_name}")
        
        # Generate trajectories
        trajectories = generate_trajectories(
            editor, anchors, coord_system, config, editor_name
        )
        
        # Evaluate
        results = benchmark.evaluate(trajectories, editor_name=editor_name)
        all_results[editor_name] = results
        
        # Log summary
        logger.info(f"  Trend Correlation: {results.progression_aggregate['trend_correlation']['mean']:.4f}")
        logger.info(f"  Inversion Rate: {results.progression_aggregate['inversion_rate']['mean']:.4f}")
        logger.info(f"  Aggregate Drift: {results.drift_aggregate['aggregate_drift_median']['mean']:.4f}")
    
    # Save results
    results_path = output_dir / "benchmark_results.json"
    save_results(
        {name: r.to_dict() for name, r in all_results.items()},
        results_path,
    )
    logger.info(f"Results saved to {results_path}")
    
    # Generate figures
    if config.get("visualization", {}).get("save_figures", True):
        logger.info("Generating figures...")
        fig_gen = FigureGenerator(output_dir / "figures")
        
        if len(all_results) >= 2:
            fig_gen.generate_figure_1(
                all_results.get("unconstrained"),
                all_results.get("constrained"),
            )
    
    logger.info("Benchmark complete!")
    return all_results


def main():
    parser = argparse.ArgumentParser(description="CIB-Med-1 Benchmark Runner")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Specific experiment to run from experiment_configs.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (overrides config)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line args
    if args.seed is not None:
        config.setdefault("reproducibility", {})["seed"] = args.seed
    
    # Setup output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.get("data", {}).get(
            "output_dir", "/Users/ttt/Downloads/cib_med_outputs"
        ))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(output_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    # Run benchmark
    run_benchmark(config, output_dir)


if __name__ == "__main__":
    main()
