"""
Unit tests for CIB-Med-1 core components.

Tests:
- SemanticCoordinateSystem
- RadiologyEvaluator
- EditTrajectory
- Calibration
- Anchor selection
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cib_med.core.semantic_coordinates import (
    SemanticCoordinateSystem,
    SemanticCoordinates,
    ClinicalFinding,
    FindingCategory,
    STANDARD_FINDINGS,
)
from cib_med.core.evaluator import MockEvaluator, RadiologyEvaluator
from cib_med.core.trajectory import TrajectoryStep, EditTrajectory, TrajectoryGenerator
from cib_med.core.calibration import IsotonicCalibrator, PlattScalingCalibrator
from cib_med.core.anchor import Anchor, AnchorSelector


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def mock_evaluator(device):
    eval = MockEvaluator(num_labels=14, seed=42)
    eval.to(device)
    return eval


@pytest.fixture
def sample_image(device):
    image = torch.randn(1, 224, 224) * 0.1 + 0.5
    return image.clamp(0, 1).to(device)


class TestSemanticCoordinateSystem:
    """Tests for SemanticCoordinateSystem."""
    
    def test_initialization(self, mock_evaluator):
        """Test coordinate system initialization."""
        coord_sys = SemanticCoordinateSystem(
            evaluator=mock_evaluator,
            target_finding="Pleural Effusion",
        )
        
        assert coord_sys.target_finding == "Pleural Effusion"
        assert len(coord_sys.off_target_findings) > 0
    
    def test_compute_coordinates(self, mock_evaluator, sample_image):
        """Test coordinate computation."""
        coord_sys = SemanticCoordinateSystem(
            evaluator=mock_evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis", "Consolidation"],
        )
        
        coords = coord_sys.compute_coordinates(sample_image)
        
        assert isinstance(coords, SemanticCoordinates)
        assert 0 <= coords.calibrated_target <= 1
        assert "Atelectasis" in coords.probabilities
        assert "Consolidation" in coords.probabilities
    
    def test_batch_coordinates(self, mock_evaluator, device):
        """Test batch coordinate computation."""
        coord_sys = SemanticCoordinateSystem(
            evaluator=mock_evaluator,
            target_finding="Pleural Effusion",
        )
        
        # Batch of images
        batch = torch.randn(4, 1, 224, 224) * 0.1 + 0.5
        batch = batch.clamp(0, 1).to(device)
        
        coords_list = coord_sys.compute_coordinates_batch(batch)
        
        assert len(coords_list) == 4
        for coords in coords_list:
            assert isinstance(coords, SemanticCoordinates)
    
    def test_custom_off_target(self, mock_evaluator, sample_image):
        """Test custom off-target finding specification."""
        custom_findings = ["Atelectasis", "Cardiomegaly"]
        
        coord_sys = SemanticCoordinateSystem(
            evaluator=mock_evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=custom_findings,
        )
        
        assert coord_sys.off_target_findings == custom_findings
        
        coords = coord_sys.compute_coordinates(sample_image)
        for finding in custom_findings:
            assert finding in coords.probabilities


class TestMockEvaluator:
    """Tests for MockEvaluator."""
    
    def test_forward(self, device):
        """Test forward pass."""
        evaluator = MockEvaluator(num_labels=14, seed=42)
        evaluator.to(device)
        
        image = torch.randn(1, 1, 224, 224).to(device)
        logits = evaluator(image)
        
        assert logits.shape == (1, 14)
    
    def test_batch_forward(self, device):
        """Test batch forward pass."""
        evaluator = MockEvaluator(num_labels=14, seed=42)
        evaluator.to(device)
        
        batch = torch.randn(8, 1, 224, 224).to(device)
        logits = evaluator(batch)
        
        assert logits.shape == (8, 14)
    
    def test_reproducibility(self, device):
        """Test that same seed gives same results."""
        evaluator1 = MockEvaluator(num_labels=14, seed=42)
        evaluator2 = MockEvaluator(num_labels=14, seed=42)
        
        evaluator1.to(device)
        evaluator2.to(device)
        
        image = torch.randn(1, 1, 224, 224).to(device)
        
        logits1 = evaluator1(image)
        logits2 = evaluator2(image)
        
        assert torch.allclose(logits1, logits2)
    
    def test_get_logit(self, mock_evaluator, sample_image):
        """Test getting specific logit."""
        logit = mock_evaluator.get_logit(sample_image, "Pleural Effusion")
        
        assert isinstance(logit, torch.Tensor)
        assert logit.dim() == 0 or logit.shape[0] == 1


class TestTrajectory:
    """Tests for trajectory classes."""
    
    def test_trajectory_step(self, sample_image):
        """Test TrajectoryStep creation."""
        coords = SemanticCoordinates(
            raw_logits={"Pleural Effusion": 0.5},
            probabilities={"Pleural Effusion": 0.6},
            calibrated_target=0.6,
        )
        
        step = TrajectoryStep(
            step=0,
            image=sample_image,
            coordinates=coords,
        )
        
        assert step.step == 0
        assert step.image is sample_image
        assert step.coordinates.calibrated_target == 0.6
    
    def test_edit_trajectory(self, sample_image):
        """Test EditTrajectory creation."""
        anchor_coords = SemanticCoordinates(
            raw_logits={},
            probabilities={"Atelectasis": 0.3},
            calibrated_target=0.3,
        )
        
        anchor = Anchor(
            image=sample_image,
            coordinates=anchor_coords,
            source_id="test",
        )
        
        steps = []
        for t in range(5):
            coords = SemanticCoordinates(
                raw_logits={},
                probabilities={"Atelectasis": 0.3 + t * 0.05},
                calibrated_target=0.3 + t * 0.1,
            )
            steps.append(TrajectoryStep(step=t, image=sample_image, coordinates=coords))
        
        trajectory = EditTrajectory(
            anchor=anchor,
            steps=steps,
            editor_name="test_editor",
        )
        
        assert len(trajectory) == 5
        assert trajectory.editor_name == "test_editor"
    
    def test_get_target_progression(self, sample_image):
        """Test getting target progression values."""
        anchor_coords = SemanticCoordinates(
            raw_logits={},
            probabilities={},
            calibrated_target=0.2,
        )
        
        anchor = Anchor(
            image=sample_image,
            coordinates=anchor_coords,
            source_id="test",
        )
        
        steps = []
        expected_targets = [0.2, 0.4, 0.6, 0.8]
        for t, target in enumerate(expected_targets):
            coords = SemanticCoordinates(
                raw_logits={},
                probabilities={},
                calibrated_target=target,
            )
            steps.append(TrajectoryStep(step=t, image=sample_image, coordinates=coords))
        
        trajectory = EditTrajectory(anchor=anchor, steps=steps, editor_name="test")
        progression = trajectory.get_target_progression()
        
        assert len(progression) == 4
        assert progression == pytest.approx(expected_targets, abs=0.01)


class TestCalibration:
    """Tests for calibration classes."""
    
    def test_isotonic_calibrator(self):
        """Test isotonic calibration."""
        calibrator = IsotonicCalibrator()
        
        # Generate training data
        logits = np.linspace(-3, 3, 100)
        labels = (logits > 0).astype(float)
        # Add noise
        labels = np.clip(labels + np.random.randn(100) * 0.1, 0, 1)
        
        calibrator.fit(logits, labels)
        
        # Test calibration
        test_logits = np.array([-2.0, 0.0, 2.0])
        calibrated = calibrator.transform(test_logits)
        
        assert len(calibrated) == 3
        assert calibrated[0] < calibrated[1] < calibrated[2]  # Monotonic
        assert all(0 <= c <= 1 for c in calibrated)  # Valid probabilities
    
    def test_platt_calibrator(self):
        """Test Platt scaling calibration."""
        calibrator = PlattScalingCalibrator()
        
        logits = np.linspace(-3, 3, 100)
        labels = (logits > 0).astype(float)
        
        calibrator.fit(logits, labels)
        
        test_logits = np.array([-2.0, 0.0, 2.0])
        calibrated = calibrator.transform(test_logits)
        
        assert len(calibrated) == 3
        assert all(0 <= c <= 1 for c in calibrated)
    
    def test_calibrator_with_tensor(self):
        """Test calibrator with torch tensors."""
        calibrator = IsotonicCalibrator()
        
        logits = np.linspace(-3, 3, 100)
        labels = (logits > 0).astype(float)
        calibrator.fit(logits, labels)
        
        # Test with tensor
        test_tensor = torch.tensor([-2.0, 0.0, 2.0])
        calibrated = calibrator.transform(test_tensor.numpy())
        
        assert isinstance(calibrated, np.ndarray)


class TestAnchor:
    """Tests for anchor classes."""
    
    def test_anchor_creation(self, sample_image):
        """Test Anchor creation."""
        coords = SemanticCoordinates(
            raw_logits={},
            probabilities={"Atelectasis": 0.3},
            calibrated_target=0.5,
        )
        
        anchor = Anchor(
            image=sample_image,
            coordinates=coords,
            source_id="test_anchor",
            metadata={"split": "test"},
        )
        
        assert anchor.source_id == "test_anchor"
        assert anchor.coordinates.calibrated_target == 0.5
        assert anchor.metadata["split"] == "test"
    
    def test_anchor_validation(self, sample_image):
        """Test anchor validation against criteria."""
        coords = SemanticCoordinates(
            raw_logits={},
            probabilities={},
            calibrated_target=0.5,  # Within valid range
        )
        
        anchor = Anchor(
            image=sample_image,
            coordinates=coords,
            source_id="test",
        )
        
        # Should be valid for range [0.1, 0.9]
        assert 0.1 <= anchor.coordinates.calibrated_target <= 0.9


class TestStandardFindings:
    """Tests for standard finding definitions."""
    
    def test_standard_findings_exist(self):
        """Test that standard findings are defined."""
        assert len(STANDARD_FINDINGS) > 0
        assert "Pleural Effusion" in STANDARD_FINDINGS
    
    def test_finding_structure(self):
        """Test finding structure."""
        finding = STANDARD_FINDINGS["Pleural Effusion"]
        
        assert isinstance(finding, ClinicalFinding)
        assert finding.name == "Pleural Effusion"
        assert finding.is_target == True
        assert isinstance(finding.category, FindingCategory)
    
    def test_categories(self):
        """Test finding categories."""
        categories = set()
        for finding in STANDARD_FINDINGS.values():
            categories.add(finding.category)
        
        assert FindingCategory.PLEURAL in categories
        assert FindingCategory.PARENCHYMAL in categories


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
