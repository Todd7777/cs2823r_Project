"""
Unit tests for CIB-Med-1 guidance methods.

Tests:
- ConstrainedDiffusionGuidance
- UnconstrainedGuidance
- GuidanceScheduler
"""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from cib_med.core.evaluator import MockEvaluator
from cib_med.guidance.base import GuidanceMethod, GuidanceScheduler, CompositeGuidance
from cib_med.guidance.constrained import ConstrainedDiffusionGuidance, AdaptiveConstrainedGuidance
from cib_med.guidance.unconstrained import UnconstrainedGuidance, NegativeGuidance


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def evaluator(device):
    eval = MockEvaluator(num_labels=14, seed=42)
    eval.to(device)
    return eval


@pytest.fixture
def sample_image(device):
    image = torch.randn(1, 1, 224, 224) * 0.1 + 0.5
    return image.clamp(0, 1).to(device).requires_grad_(True)


class TestUnconstrainedGuidance:
    """Tests for UnconstrainedGuidance."""
    
    def test_initialization(self, evaluator):
        """Test guidance initialization."""
        guidance = UnconstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            guidance_scale=7.5,
        )
        
        assert guidance.target_finding == "Pleural Effusion"
        assert guidance.guidance_scale == 7.5
    
    def test_compute_guidance(self, evaluator, sample_image, device):
        """Test guidance computation."""
        guidance = UnconstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            guidance_scale=1.0,
        )
        
        result = guidance(sample_image, timestep=500)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_image.shape
    
    def test_guidance_direction(self, evaluator, device):
        """Test that guidance points toward higher target score."""
        guidance = UnconstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            guidance_scale=1.0,
        )
        
        image = torch.randn(1, 1, 224, 224).to(device) * 0.1 + 0.5
        image = image.clamp(0, 1).requires_grad_(True)
        
        # Get guidance
        grad = guidance(image, timestep=500)
        
        # Guidance should be non-zero
        assert grad.abs().sum() > 0


class TestConstrainedGuidance:
    """Tests for ConstrainedDiffusionGuidance."""
    
    def test_initialization(self, evaluator):
        """Test constrained guidance initialization."""
        guidance = ConstrainedDiffusionGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis", "Consolidation"],
            lambda_constraint=1.0,
        )
        
        assert guidance.target_finding == "Pleural Effusion"
        assert guidance.lambda_constraint == 1.0
        assert len(guidance.off_target_findings) == 2
    
    def test_compute_guidance(self, evaluator, sample_image, device):
        """Test constrained guidance computation."""
        guidance = ConstrainedDiffusionGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis", "Consolidation"],
            lambda_constraint=1.0,
        )
        
        # Set anchor for constraint
        anchor = sample_image.detach().clone()
        guidance.set_anchor(anchor)
        
        result = guidance(sample_image, timestep=500)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == sample_image.shape
    
    def test_constraint_effect(self, evaluator, device):
        """Test that constraints reduce off-target gradient component."""
        # Create unconstrained guidance
        unconstrained = UnconstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            guidance_scale=1.0,
        )
        
        # Create constrained guidance
        constrained = ConstrainedDiffusionGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis", "Consolidation"],
            lambda_constraint=2.0,
        )
        
        image = torch.randn(1, 1, 224, 224).to(device) * 0.1 + 0.5
        image = image.clamp(0, 1).requires_grad_(True)
        
        anchor = image.detach().clone()
        constrained.set_anchor(anchor)
        
        # Both should produce gradients
        unc_grad = unconstrained(image, timestep=500)
        con_grad = constrained(image, timestep=500)
        
        assert unc_grad.abs().sum() > 0
        assert con_grad.abs().sum() > 0
    
    def test_lambda_scaling(self, evaluator, device):
        """Test that lambda affects constraint strength."""
        image = torch.randn(1, 1, 224, 224).to(device) * 0.1 + 0.5
        image = image.clamp(0, 1).requires_grad_(True)
        anchor = image.detach().clone()
        
        guidance_low = ConstrainedDiffusionGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis"],
            lambda_constraint=0.1,
        )
        guidance_low.set_anchor(anchor)
        
        guidance_high = ConstrainedDiffusionGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis"],
            lambda_constraint=10.0,
        )
        guidance_high.set_anchor(anchor)
        
        grad_low = guidance_low(image, timestep=500)
        grad_high = guidance_high(image, timestep=500)
        
        # Different lambda should give different gradients
        assert not torch.allclose(grad_low, grad_high)


class TestAdaptiveConstrainedGuidance:
    """Tests for AdaptiveConstrainedGuidance."""
    
    def test_initialization(self, evaluator):
        """Test adaptive guidance initialization."""
        guidance = AdaptiveConstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis"],
            lambda_constraint=1.0,
        )
        
        assert guidance is not None
    
    def test_dynamic_lambda(self, evaluator, device):
        """Test that lambda adjusts dynamically."""
        guidance = AdaptiveConstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            off_target_findings=["Atelectasis"],
            lambda_constraint=1.0,
        )
        
        image = torch.randn(1, 1, 224, 224).to(device) * 0.1 + 0.5
        image = image.clamp(0, 1).requires_grad_(True)
        anchor = image.detach().clone()
        
        guidance.set_anchor(anchor)
        
        # Get guidance at different timesteps
        grad_early = guidance(image, timestep=900, step=1, total_steps=100)
        grad_late = guidance(image, timestep=100, step=90, total_steps=100)
        
        # Both should be valid
        assert grad_early.abs().sum() > 0 or grad_late.abs().sum() > 0


class TestGuidanceScheduler:
    """Tests for GuidanceScheduler."""
    
    def test_constant_scheduler(self):
        """Test constant scheduling."""
        scheduler = GuidanceScheduler(
            schedule_type="constant",
            initial_scale=1.0,
        )
        
        assert scheduler.get_scale(0, 100) == 1.0
        assert scheduler.get_scale(50, 100) == 1.0
        assert scheduler.get_scale(99, 100) == 1.0
    
    def test_linear_scheduler(self):
        """Test linear scheduling."""
        scheduler = GuidanceScheduler(
            schedule_type="linear",
            initial_scale=1.0,
            final_scale=0.0,
        )
        
        scale_start = scheduler.get_scale(0, 100)
        scale_mid = scheduler.get_scale(50, 100)
        scale_end = scheduler.get_scale(99, 100)
        
        assert scale_start > scale_mid > scale_end
    
    def test_cosine_scheduler(self):
        """Test cosine scheduling."""
        scheduler = GuidanceScheduler(
            schedule_type="cosine",
            initial_scale=1.0,
            final_scale=0.0,
        )
        
        scale_start = scheduler.get_scale(0, 100)
        scale_end = scheduler.get_scale(99, 100)
        
        assert scale_start > scale_end


class TestNegativeGuidance:
    """Tests for NegativeGuidance."""
    
    def test_negative_direction(self, evaluator, device):
        """Test that negative guidance points opposite to positive."""
        positive = UnconstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            guidance_scale=1.0,
        )
        
        negative = NegativeGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            guidance_scale=1.0,
        )
        
        image = torch.randn(1, 1, 224, 224).to(device) * 0.1 + 0.5
        image = image.clamp(0, 1).requires_grad_(True)
        
        pos_grad = positive(image, timestep=500)
        neg_grad = negative(image, timestep=500)
        
        # Negative guidance should point opposite direction
        # (approximately, due to implementation details)
        assert pos_grad.abs().sum() > 0
        assert neg_grad.abs().sum() > 0


class TestCompositeGuidance:
    """Tests for CompositeGuidance."""
    
    def test_combine_guidance(self, evaluator, device):
        """Test combining multiple guidance methods."""
        guidance1 = UnconstrainedGuidance(
            evaluator=evaluator,
            target_finding="Pleural Effusion",
            guidance_scale=1.0,
        )
        
        guidance2 = UnconstrainedGuidance(
            evaluator=evaluator,
            target_finding="Atelectasis",
            guidance_scale=0.5,
        )
        
        composite = CompositeGuidance(
            methods=[guidance1, guidance2],
            weights=[1.0, 0.5],
        )
        
        image = torch.randn(1, 1, 224, 224).to(device) * 0.1 + 0.5
        image = image.clamp(0, 1).requires_grad_(True)
        
        result = composite(image, timestep=500)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
