"""
Unit tests for CIB-Med-1 metrics.

Tests the core metric implementations:
- MonotoneProgressionMetrics
- OffTargetDriftMetrics
- CIBMedBenchmark
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cib_med.core.evaluator import MockEvaluator
from cib_med.core.semantic_coordinates import SemanticCoordinateSystem, SemanticCoordinates
from cib_med.core.trajectory import TrajectoryStep, EditTrajectory
from cib_med.core.anchor import Anchor
from cib_med.metrics.progression import MonotoneProgressionMetrics, ProgressionMetrics
from cib_med.metrics.drift import OffTargetDriftMetrics, DriftMetrics
from cib_med.metrics.benchmark import CIBMedBenchmark


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def evaluator(device):
    eval = MockEvaluator(num_labels=14, seed=42)
    eval.to(device)
    return eval


@pytest.fixture
def coord_system(evaluator):
    return SemanticCoordinateSystem(
        evaluator=evaluator,
        target_finding="Pleural Effusion",
        off_target_findings=["Atelectasis", "Consolidation", "Cardiomegaly"],
    )


@pytest.fixture
def sample_trajectory(coord_system, device):
    """Create a sample trajectory for testing."""
    # Create anchor
    anchor_image = torch.randn(1, 224, 224) * 0.1 + 0.5
    anchor_image = anchor_image.clamp(0, 1).to(device)
    anchor_coords = coord_system.compute_coordinates(anchor_image)
    
    anchor = Anchor(
        image=anchor_image,
        coordinates=anchor_coords,
        source_id="test_anchor",
    )
    
    # Create steps with increasing target scores
    steps = []
    for t in range(6):
        image = anchor_image + 0.01 * t * torch.randn_like(anchor_image)
        image = image.clamp(0, 1)
        coords = coord_system.compute_coordinates(image)
        
        # Manually set increasing target for testing
        coords = SemanticCoordinates(
            raw_logits=coords.raw_logits,
            probabilities=coords.probabilities,
            calibrated_target=0.3 + 0.1 * t,  # Increasing
        )
        
        steps.append(TrajectoryStep(step=t, image=image, coordinates=coords))
    
    return EditTrajectory(
        anchor=anchor,
        steps=steps,
        editor_name="test_editor",
    )


class TestMonotoneProgressionMetrics:
    """Tests for MonotoneProgressionMetrics."""
    
    def test_perfect_monotone(self):
        """Test metrics for perfectly monotone trajectory."""
        target_scores = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        metrics = MonotoneProgressionMetrics()
        result = metrics.compute(target_scores)
        
        assert isinstance(result, ProgressionMetrics)
        assert result.trend_correlation == pytest.approx(1.0, abs=0.01)
        assert result.inversion_rate == 0.0
        assert result.total_progression == pytest.approx(0.4, abs=0.01)
    
    def test_inversion_detection(self):
        """Test that inversions are detected."""
        target_scores = [0.1, 0.3, 0.2, 0.4, 0.5]  # One inversion at index 2
        
        metrics = MonotoneProgressionMetrics()
        result = metrics.compute(target_scores)
        
        assert result.inversion_rate == pytest.approx(0.25, abs=0.01)  # 1/4 inversions
        assert result.trend_correlation < 1.0
    
    def test_flat_trajectory(self):
        """Test metrics for flat trajectory."""
        target_scores = [0.5, 0.5, 0.5, 0.5]
        
        metrics = MonotoneProgressionMetrics()
        result = metrics.compute(target_scores)
        
        assert result.total_progression == 0.0
        assert result.inversion_rate == 0.0
    
    def test_trajectory_integration(self, sample_trajectory):
        """Test metrics on EditTrajectory object."""
        metrics = MonotoneProgressionMetrics()
        
        target_scores = sample_trajectory.get_target_progression()
        result = metrics.compute(target_scores)
        
        assert isinstance(result, ProgressionMetrics)
        assert 0 <= result.inversion_rate <= 1


class TestOffTargetDriftMetrics:
    """Tests for OffTargetDriftMetrics."""
    
    def test_zero_drift(self):
        """Test metrics when there's no drift."""
        anchor_values = {"Atelectasis": 0.3, "Consolidation": 0.2}
        trajectory_values = [
            {"Atelectasis": 0.3, "Consolidation": 0.2},
            {"Atelectasis": 0.3, "Consolidation": 0.2},
            {"Atelectasis": 0.3, "Consolidation": 0.2},
        ]
        
        metrics = OffTargetDriftMetrics(off_target_findings=["Atelectasis", "Consolidation"])
        result = metrics.compute(anchor_values, trajectory_values)
        
        assert isinstance(result, DriftMetrics)
        assert result.aggregate_drift_median == 0.0
        assert result.aggregate_drift_90th == 0.0
    
    def test_drift_detection(self):
        """Test that drift is properly detected."""
        anchor_values = {"Atelectasis": 0.3, "Consolidation": 0.2}
        trajectory_values = [
            {"Atelectasis": 0.4, "Consolidation": 0.3},  # +0.1, +0.1
            {"Atelectasis": 0.5, "Consolidation": 0.4},  # +0.2, +0.2
        ]
        
        metrics = OffTargetDriftMetrics(off_target_findings=["Atelectasis", "Consolidation"])
        result = metrics.compute(anchor_values, trajectory_values)
        
        assert result.aggregate_drift_median > 0
        assert result.per_label_drift["Atelectasis"] > 0
        assert result.per_label_drift["Consolidation"] > 0
    
    def test_per_label_ordering(self):
        """Test that per-label drift is correctly ordered."""
        anchor_values = {"A": 0.5, "B": 0.5}
        trajectory_values = [
            {"A": 0.6, "B": 0.5},  # A drifts more
            {"A": 0.7, "B": 0.5},
        ]
        
        metrics = OffTargetDriftMetrics(off_target_findings=["A", "B"])
        result = metrics.compute(anchor_values, trajectory_values)
        
        assert result.per_label_drift["A"] > result.per_label_drift["B"]


class TestCIBMedBenchmark:
    """Tests for the complete benchmark."""
    
    def test_benchmark_evaluation(self, coord_system, sample_trajectory):
        """Test complete benchmark evaluation."""
        benchmark = CIBMedBenchmark(coord_system)
        
        trajectories = [sample_trajectory]
        results = benchmark.evaluate(trajectories, editor_name="test")
        
        assert results.editor_name == "test"
        assert "trend_correlation" in results.progression_aggregate
        assert "aggregate_drift_median" in results.drift_aggregate
        assert len(results.trajectory_results) == 1
    
    def test_multi_trajectory(self, coord_system, device):
        """Test benchmark with multiple trajectories."""
        # Create multiple trajectories
        trajectories = []
        for i in range(5):
            anchor_image = torch.randn(1, 224, 224) * 0.1 + 0.5
            anchor_image = anchor_image.clamp(0, 1).to(device)
            anchor_coords = coord_system.compute_coordinates(anchor_image)
            
            anchor = Anchor(
                image=anchor_image,
                coordinates=anchor_coords,
                source_id=f"test_anchor_{i}",
            )
            
            steps = []
            for t in range(4):
                coords = SemanticCoordinates(
                    raw_logits={},
                    probabilities={"Atelectasis": 0.3, "Consolidation": 0.2, "Cardiomegaly": 0.1},
                    calibrated_target=0.2 + 0.1 * t,
                )
                steps.append(TrajectoryStep(step=t, image=anchor_image, coordinates=coords))
            
            trajectories.append(EditTrajectory(
                anchor=anchor,
                steps=steps,
                editor_name="test",
            ))
        
        benchmark = CIBMedBenchmark(coord_system)
        results = benchmark.evaluate(trajectories, editor_name="test")
        
        assert len(results.trajectory_results) == 5
        assert "mean" in results.progression_aggregate["trend_correlation"]
        assert "std" in results.progression_aggregate["trend_correlation"]
    
    def test_results_to_dict(self, coord_system, sample_trajectory):
        """Test results serialization."""
        benchmark = CIBMedBenchmark(coord_system)
        results = benchmark.evaluate([sample_trajectory], editor_name="test")
        
        result_dict = results.to_dict()
        
        assert isinstance(result_dict, dict)
        assert "editor_name" in result_dict
        assert "progression_aggregate" in result_dict
        assert "drift_aggregate" in result_dict


class TestMetricEdgeCases:
    """Test edge cases and error handling."""
    
    def test_single_step_trajectory(self):
        """Test handling of single-step trajectory."""
        target_scores = [0.5]
        
        metrics = MonotoneProgressionMetrics()
        result = metrics.compute(target_scores)
        
        assert result.total_progression == 0.0
        assert result.inversion_rate == 0.0
    
    def test_empty_off_target(self):
        """Test with no off-target findings."""
        anchor_values = {}
        trajectory_values = [{}]
        
        metrics = OffTargetDriftMetrics(off_target_findings=[])
        result = metrics.compute(anchor_values, trajectory_values)
        
        assert result.aggregate_drift_median == 0.0
    
    def test_negative_progression(self):
        """Test trajectory that goes backwards."""
        target_scores = [0.5, 0.4, 0.3, 0.2]
        
        metrics = MonotoneProgressionMetrics()
        result = metrics.compute(target_scores)
        
        assert result.total_progression < 0
        assert result.inversion_rate == 1.0  # All inversions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
