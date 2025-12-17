"""
Basic sanity tests for CIB-Med-1.

These tests verify core functionality without requiring external dependencies.
Designed to pass in CI/CD environments.
"""

import sys
from pathlib import Path

# Ensure package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPackageStructure:
    """Verify package structure is correct."""

    def test_package_exists(self):
        """Test that cib_med package exists."""
        import cib_med
        assert hasattr(cib_med, "__version__")

    def test_version_format(self):
        """Test version follows semantic versioning."""
        import cib_med
        version = cib_med.__version__
        parts = version.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor"

    def test_core_module_exists(self):
        """Test core module is importable."""
        from cib_med import core
        assert core is not None

    def test_metrics_module_exists(self):
        """Test metrics module is importable."""
        from cib_med import metrics
        assert metrics is not None


class TestCoreComponents:
    """Test core component imports."""

    def test_semantic_coordinates_import(self):
        """Test SemanticCoordinateSystem is importable."""
        from cib_med.core.semantic_coordinates import SemanticCoordinateSystem
        assert SemanticCoordinateSystem is not None

    def test_trajectory_import(self):
        """Test trajectory classes are importable."""
        from cib_med.core.trajectory import EditTrajectory, TrajectoryStep
        assert EditTrajectory is not None
        assert TrajectoryStep is not None

    def test_calibration_import(self):
        """Test calibration classes are importable."""
        from cib_med.core.calibration import IsotonicCalibrator
        assert IsotonicCalibrator is not None


class TestMetricsComponents:
    """Test metrics component imports."""

    def test_progression_metrics_import(self):
        """Test progression metrics are importable."""
        from cib_med.metrics.progression import MonotoneProgressionMetrics
        assert MonotoneProgressionMetrics is not None

    def test_drift_metrics_import(self):
        """Test drift metrics are importable."""
        from cib_med.metrics.drift import OffTargetDriftMetrics
        assert OffTargetDriftMetrics is not None

    def test_benchmark_import(self):
        """Test benchmark class is importable."""
        from cib_med.metrics.benchmark import CIBMedBenchmark
        assert CIBMedBenchmark is not None


class TestGuidanceComponents:
    """Test guidance component imports."""

    def test_constrained_guidance_import(self):
        """Test constrained guidance is importable."""
        from cib_med.guidance.constrained import ConstrainedDiffusionGuidance
        assert ConstrainedDiffusionGuidance is not None

    def test_unconstrained_guidance_import(self):
        """Test unconstrained guidance is importable."""
        from cib_med.guidance.unconstrained import UnconstrainedGuidance
        assert UnconstrainedGuidance is not None


class TestAnalysisComponents:
    """Test analysis component imports."""

    def test_ablation_import(self):
        """Test ablation analyzer is importable."""
        from cib_med.analysis.ablation import AblationAnalyzer
        assert AblationAnalyzer is not None

    def test_pareto_import(self):
        """Test Pareto analyzer is importable."""
        from cib_med.analysis.pareto import ParetoAnalyzer
        assert ParetoAnalyzer is not None

    def test_synthetic_import(self):
        """Test synthetic stress test is importable."""
        from cib_med.analysis.synthetic import SyntheticLatentModel
        assert SyntheticLatentModel is not None


class TestUtilities:
    """Test utility functions."""

    def test_io_import(self):
        """Test I/O utilities are importable."""
        from cib_med.utils.io import save_results, load_results
        assert save_results is not None
        assert load_results is not None

    def test_reproducibility_import(self):
        """Test reproducibility utilities are importable."""
        from cib_med.utils.reproducibility import set_seed, get_device
        assert set_seed is not None
        assert get_device is not None


class TestDataComponents:
    """Test data component imports."""

    def test_datasets_import(self):
        """Test dataset classes are importable."""
        from cib_med.data.datasets import CXRDataset
        assert CXRDataset is not None

    def test_transforms_import(self):
        """Test transforms are importable."""
        from cib_med.data.transforms import get_cxr_transforms
        assert get_cxr_transforms is not None


class TestModelComponents:
    """Test model component imports."""

    def test_diffusion_import(self):
        """Test diffusion editor is importable."""
        from cib_med.models.diffusion import DiffusionEditor
        assert DiffusionEditor is not None

    def test_unet_import(self):
        """Test UNet model is importable."""
        from cib_med.models.unet import UNet2DModel
        assert UNet2DModel is not None


class TestVisualizationComponents:
    """Test visualization component imports."""

    def test_plots_import(self):
        """Test plotting functions are importable."""
        from cib_med.visualization.plots import plot_trajectory
        assert plot_trajectory is not None

    def test_figures_import(self):
        """Test figure generator is importable."""
        from cib_med.visualization.figures import FigureGenerator
        assert FigureGenerator is not None
