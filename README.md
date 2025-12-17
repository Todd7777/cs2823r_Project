# CIB-Med-1: A Benchmark for Reliable, Monotonic, and Clinically Faithful Editing of Chest Radiographs

[![NeurIPS 2025](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **CIB-Med-1** exposes reward hacking in directional medical image editing by measuring both target progression and off-target semantic drift.

## Overview

Diffusion-based generative models enable semantic image editing, but in medical imaging, **optimizing for target improvement alone permits reward hacking**: models can exploit correlations in the training distribution rather than manipulating the intended pathological factor.

CIB-Med-1 addresses this by providing:

- **Semantic Coordinate System**: Clinically interpretable coordinates from a frozen radiology evaluator
- **Trajectory-Level Metrics**: Evaluate entire edit trajectories, not just endpoints
- **Off-Target Drift Detection**: Quantify deviations along non-target clinical axes
- **Constrained Diffusion Guidance**: A method that optimizes target progression subject to bounded off-target change

### Key Findings

| Method | Target Progression ↑ | Off-Target Drift ↓ | Human Correlation (τ) |
|--------|---------------------|-------------------|----------------------|
| Unconstrained | 0.89 | 0.34 | 0.47 |
| Pix2Pix | 0.72 | 0.41 | 0.29 |
| **Constrained (Ours)** | **0.85** | **0.12** | **0.61** |

## Installation

```bash
# Clone the repository
git clone https://github.com/Todd7777/cs2823r_Project.git
cd cs2823r_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

## Quick Start

### 1. Basic Benchmark Evaluation

```python
from cib_med import CIBMedBenchmark, SemanticCoordinateSystem, RadiologyEvaluator
from cib_med.core import EditTrajectory

# Initialize components
evaluator = RadiologyEvaluator.from_pretrained("densenet121")
coord_system = SemanticCoordinateSystem(evaluator)
benchmark = CIBMedBenchmark(coord_system)

# Load or generate trajectories
trajectories = [...]  # Your edit trajectories

# Evaluate
results = benchmark.evaluate(trajectories, editor_name="my_editor")

# View results
print(f"Trend Correlation: {results.progression_aggregate['trend_correlation']['mean']:.3f}")
print(f"Off-Target Drift: {results.drift_aggregate['aggregate_drift_median']['mean']:.3f}")
```

### 2. Run Complete Benchmark

```bash
# Run with default configuration
python scripts/run_benchmark.py --config configs/default.yaml

# Run specific experiment
python scripts/run_benchmark.py --config configs/experiment_configs.yaml --experiment main_comparison
```

### 3. Generate Trajectories with Constrained Guidance

```python
from cib_med.guidance import ConstrainedDiffusionGuidance
from cib_med.models import DiffusionEditor, DDPMScheduler, UNet2DModel

# Setup model and guidance
model = UNet2DModel(in_channels=1, out_channels=1)
scheduler = DDPMScheduler()
guidance = ConstrainedDiffusionGuidance(
    evaluator=evaluator,
    target_finding="Pleural Effusion",
    off_target_findings=coord_system.off_target_findings,
    lambda_constraint=1.0,
)

# Create editor
editor = DiffusionEditor(model, scheduler, guidance_method=guidance)

# Generate trajectory
trajectory = editor.generate_trajectory(anchor_image, num_steps=20)
```

## Metrics

### Target Progression (D1)
- **Trend Correlation (ρ_trend)**: Spearman correlation between step index and target score
- **Inversion Rate**: Fraction of steps violating monotonicity

### Off-Target Stability (D2)
- **Per-Label Drift (D_k)**: Trajectory-averaged absolute drift per finding
- **Aggregate Drift (D_off)**: Median drift across all off-target findings
- **90th Percentile Drift**: Worst-case drift behavior

### Trajectory-Level Evaluation (D3)
All metrics operate on entire trajectories `{x^t}_{t=0}^T`, not single endpoints.

## Project Structure

```
CIB-Med-1/
├── cib_med/                    # Main package
│   ├── core/                   # Core components
│   │   ├── semantic_coordinates.py  # Clinical coordinate system
│   │   ├── evaluator.py            # Radiology evaluator interface
│   │   ├── trajectory.py           # Trajectory management
│   │   ├── calibration.py          # Isotonic/Platt calibration
│   │   └── anchor.py               # Anchor selection
│   ├── metrics/                # Evaluation metrics
│   │   ├── progression.py          # Monotone progression metrics
│   │   ├── drift.py                # Off-target drift metrics
│   │   └── benchmark.py            # Complete benchmark orchestration
│   ├── guidance/               # Diffusion guidance methods
│   │   ├── base.py                 # Base guidance classes
│   │   ├── constrained.py          # Constrained guidance (Section 4)
│   │   └── unconstrained.py        # Unconstrained baseline
│   ├── baselines/              # Baseline methods
│   │   ├── image_to_image.py       # Pix2Pix, CycleGAN
│   │   └── prompt_based.py         # Text-guided, InstructPix2Pix
│   ├── analysis/               # Analysis tools
│   │   ├── ablation.py             # Ablation analysis
│   │   ├── correlation.py          # Correlation analysis
│   │   ├── pareto.py               # Pareto efficiency
│   │   └── synthetic.py            # Stress tests (Section 7)
│   ├── visualization/          # Visualization
│   │   ├── plots.py                # Plotting functions
│   │   └── figures.py              # Paper figure generation
│   ├── models/                 # Model architectures
│   │   ├── diffusion.py            # Diffusion editor
│   │   └── unet.py                 # UNet architecture
│   ├── data/                   # Data utilities
│   │   ├── datasets.py             # Dataset classes
│   │   └── transforms.py           # Image transforms
│   └── utils/                  # Utilities
│       ├── io.py                   # I/O functions
│       ├── logging.py              # Logging setup
│       └── reproducibility.py      # Seed management
├── configs/                    # Configuration files
├── scripts/                    # Executable scripts
├── examples/                   # Example notebooks and scripts
├── tests/                      # Unit tests
└── docs/                       # Documentation
```

## Reproducing Paper Results

### Main Experiments (Section 5)

```bash
# Run main comparison
python scripts/run_benchmark.py --experiment main_comparison

# Generate Pareto frontier (Figure 3)
python scripts/run_pareto_sweep.py

# Run ablation study (Section 5.4)
python scripts/run_ablations.py
```

### Correlation Analysis (Section 6)

```bash
# Analyze drift vs. dataset association
python scripts/analyze_correlations.py
```

### Synthetic Stress Tests (Section 7)

```bash
# Run synthetic validation
python scripts/run_stress_tests.py
```

### Generate Paper Figures

```bash
# Generate all figures
python scripts/generate_figures.py --output_dir figures/
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
data:
  root_dir: "/Users/ttt/Downloads/cxr_data"
  output_dir: "/Users/ttt/Downloads/cib_med_outputs"

constrained_guidance:
  lambda_constraint: 1.0
  alpha_variance: 0.5
  beta_anchor: 0.3

trajectory:
  num_steps: 20
  guidance_scale: 7.5
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=cib_med --cov-report=html

# Run specific test module
pytest tests/test_metrics.py -v
```

## Citation

If you use CIB-Med-1 in your research, please cite:

```bibtex
@inproceedings{cibmed2025,
  title={CIB-Med-1: A Benchmark for Reliable, Monotonic, and Clinically Faithful Editing of Chest Radiographs},
  author={Todd},
  booktitle={Advances in Neural Information Processing Systems},
  year={2025}
}
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Ethical Considerations

**Important**: Edited medical images should be treated as *model probes* rather than synthetic patients. CIB-Med-1 is designed to diagnose what generative models can and cannot control, not to produce clinically valid synthetic data.

## Acknowledgments

- The CheXpert and MIMIC-CXR teams for providing benchmark datasets
- The diffusers library for pretrained diffusion models
- Radiology trainees who participated in human validation studies
