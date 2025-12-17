#!/usr/bin/env python3
"""
CIB-Med-1 Ablation Study Runner

Runs systematic ablation studies as described in Section 5.4:
- Leave-one-out ablations
- Grouped ablations
- Weight ablations

Usage:
    python scripts/run_ablations.py --config configs/default.yaml
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from cib_med.core import SemanticCoordinateSystem, EditTrajectory
from cib_med.core.evaluator import MockEvaluator
from cib_med.core.anchor import Anchor
from cib_med.metrics import CIBMedBenchmark
from cib_med.analysis import AblationAnalyzer, generate_ablation_configs
from cib_med.guidance import ConstrainedDiffusionGuidance
from cib_med.models import DiffusionEditor, DDPMScheduler, UNet2DModel
from cib_med.utils import set_seed, get_device, setup_logger, save_results


def run_ablation_study(config: dict, output_dir: Path):
    """Run complete ablation study."""
    logger = setup_logger("ablation", log_file=output_dir / "ablation.log")
    
    # Setup
    set_seed(config.get("reproducibility", {}).get("seed", 42))
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Initialize evaluator (use mock for demo)
    evaluator = MockEvaluator(num_labels=14)
    evaluator.to(device)
    
    # Setup coordinate system
    coord_config = config.get("coordinates", {})
    coord_system = SemanticCoordinateSystem(
        evaluator=evaluator,
        target_finding=coord_config.get("target_finding", "Pleural Effusion"),
        off_target_findings=coord_config.get("off_target_findings"),
    )
    
    # Generate synthetic anchors
    anchor_config = config.get("anchors", {})
    num_anchors = min(anchor_config.get("num_anchors", 30), 30)  # Limit for ablations
    
    anchors = []
    for i in range(num_anchors):
        image = torch.randn(1, 224, 224) * 0.1 + 0.5
        image = image.clamp(0, 1).to(device)
        coords = coord_system.compute_coordinates(image)
        anchors.append(Anchor(
            image=image,
            coordinates=coords,
            source_id=f"ablation_anchor_{i:04d}",
        ))
    
    logger.info(f"Created {len(anchors)} anchors for ablation study")
    
    # Initialize benchmark and analyzer
    benchmark = CIBMedBenchmark(coord_system)
    
    # Generate ablation configurations
    ablation_configs = generate_ablation_configs(
        off_target_findings=coord_system.off_target_findings,
        include_grouped=True,
    )
    
    logger.info(f"Running {len(ablation_configs)} ablation configurations")
    
    # Setup base model
    model = UNet2DModel(in_channels=1, out_channels=1, base_channels=32)
    model = model.to(device)
    scheduler = DDPMScheduler(num_train_timesteps=100)
    
    # Run baseline first
    baseline_results = None
    all_ablation_results = {}
    
    for ablation_cfg in tqdm(ablation_configs, desc="Running ablations"):
        cfg_name = ablation_cfg["name"]
        ablated = ablation_cfg["ablated"]
        
        logger.info(f"Running: {cfg_name} (ablated: {ablated})")
        
        # Create guidance with ablated findings removed
        active_findings = [
            f for f in coord_system.off_target_findings
            if f not in ablated
        ]
        
        guidance = ConstrainedDiffusionGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=active_findings if active_findings else None,
            lambda_constraint=1.0,
        )
        
        editor = DiffusionEditor(
            model=model,
            scheduler=scheduler,
            guidance_method=guidance,
            device=str(device),
            noise_level=0.3,
        )
        
        # Generate trajectories
        trajectories = []
        for anchor in anchors[:10]:  # Subset for speed
            from cib_med.core.trajectory import TrajectoryStep
            
            images = editor.generate_trajectory(
                anchor.image,
                num_steps=10,
                step_noise_level=0.1,
            )
            
            steps = []
            for t, img in enumerate(images):
                coords = coord_system.compute_coordinates(img)
                steps.append(TrajectoryStep(step=t, image=img, coordinates=coords))
            
            trajectory = EditTrajectory(
                anchor=anchor,
                steps=steps,
                editor_name=cfg_name,
            )
            trajectories.append(trajectory)
        
        # Evaluate
        results = benchmark.evaluate(trajectories, editor_name=cfg_name)
        all_ablation_results[cfg_name] = results
        
        if cfg_name == "full":
            baseline_results = results
        
        logger.info(f"  Drift: {results.drift_aggregate['aggregate_drift_median']['mean']:.4f}")
    
    # Analyze ablation importance
    if baseline_results:
        analyzer = AblationAnalyzer(benchmark, baseline_results)
        
        for cfg_name, results in all_ablation_results.items():
            if cfg_name != "full":
                cfg = next(c for c in ablation_configs if c["name"] == cfg_name)
                # Add mock trajectory for analysis
                analyzer.ablation_results[cfg_name] = type('AblationResult', (), {
                    'config_name': cfg_name,
                    'ablated_components': cfg["ablated"],
                    'benchmark_results': results,
                    'delta_drift': (
                        results.drift_aggregate['aggregate_drift_median']['mean'] -
                        baseline_results.drift_aggregate['aggregate_drift_median']['mean']
                    ),
                    'importance_score': 0.0,
                    'to_dict': lambda self=cfg_name: {'name': self},
                })()
        
        # Generate report
        report = analyzer.generate_report()
        logger.info("\n" + report)
        
        # Save report
        with open(output_dir / "ablation_report.txt", "w") as f:
            f.write(report)
        
        # Compute importance ranking
        importance = analyzer.compute_leave_one_out_importance()
        logger.info("Leave-one-out importance ranking:")
        for finding, imp in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {finding}: {imp:+.4f}")
    
    # Save all results
    save_results(
        {name: r.to_dict() for name, r in all_ablation_results.items()},
        output_dir / "ablation_results.json",
    )
    
    logger.info(f"Ablation study complete. Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="CIB-Med-1 Ablation Study")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent.parent / "configs" / "default.yaml",
        help="Configuration file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory",
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    # Setup output
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = Path(config.get("data", {}).get(
            "output_dir", "/Users/ttt/Downloads/cib_med_outputs"
        ))
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir / f"ablation_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    run_ablation_study(config, output_dir)


if __name__ == "__main__":
    main()
