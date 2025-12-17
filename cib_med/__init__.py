"""
CIB-Med-1: Controlled Incremental Biomarker Editing for Medical Imaging
========================================================================

A comprehensive benchmark for reliable, monotonic, and clinically faithful
editing of chest radiographs, as described in the NeurIPS 2025 paper.

This package provides:
    - **Semantic Coordinate Systems**: Map images to clinically interpretable scores
    - **Trajectory-Level Metrics**: Evaluate entire edit trajectories (D1, D2, D3)
    - **Constrained Diffusion Guidance**: Optimize target subject to off-target bounds
    - **Baseline Methods**: Unconstrained, Pix2Pix, CycleGAN, InstructPix2Pix
    - **Ablation Framework**: Leave-one-out and grouped ablation studies
    - **Synthetic Stress Tests**: Validate metrics with known ground truth

Example:
    >>> from cib_med import CIBMedBenchmark, SemanticCoordinateSystem
    >>> coord_system = SemanticCoordinateSystem(evaluator, target_finding="Pleural Effusion")
    >>> benchmark = CIBMedBenchmark(coord_system)
    >>> results = benchmark.evaluate(trajectories, editor_name="my_editor")

For more information, see: https://github.com/Todd7777/cs2823r_Project

Copyright (c) 2025 Todd. MIT License.
"""

from __future__ import annotations

__version__ = "1.0.0"
__author__ = "Todd"
__license__ = "MIT"
__copyright__ = "Copyright (c) 2025 Todd"

# Core components
from cib_med.core.semantic_coordinates import SemanticCoordinateSystem
from cib_med.core.trajectory import EditTrajectory, TrajectoryGenerator
from cib_med.core.evaluator import RadiologyEvaluator

# Metrics
from cib_med.metrics.progression import MonotoneProgressionMetrics
from cib_med.metrics.drift import OffTargetDriftMetrics
from cib_med.metrics.benchmark import CIBMedBenchmark

__all__ = [
    # Core
    "SemanticCoordinateSystem",
    "EditTrajectory",
    "TrajectoryGenerator",
    "RadiologyEvaluator",
    # Metrics
    "MonotoneProgressionMetrics",
    "OffTargetDriftMetrics",
    "CIBMedBenchmark",
]
