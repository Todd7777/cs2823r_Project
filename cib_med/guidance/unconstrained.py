"""
Unconstrained Guidance Methods.

This module implements unconstrained (target-only) guidance methods
that serve as baselines for comparison with constrained approaches.
"""

from typing import Dict, Any, Optional
import torch

from cib_med.core.evaluator import RadiologyEvaluator
from cib_med.guidance.base import ClassifierGuidance, GuidanceScheduler


class UnconstrainedGuidance(ClassifierGuidance):
    """
    Unconstrained classifier-guided diffusion (Clinical Baseline).
    
    Standard classifier guidance using only the target logit,
    as described in Eq. (4) of the paper:
    
    ∇_x G_target(x) = ∇_x ℓ_eff(x)
    
    This baseline isolates the effect of optimizing the target
    coordinate without off-target constraints.
    
    Args:
        evaluator: RadiologyEvaluator for computing gradients
        target_label: Target finding to increase
        guidance_scale: Multiplier for guidance gradient
        scheduler: Optional GuidanceScheduler for time-varying scale
    """
    
    name = "unconstrained"
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        target_label: str = "Pleural Effusion",
        guidance_scale: float = 100.0,
        scheduler: Optional[GuidanceScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            evaluator=evaluator,
            target_label=target_label,
            guidance_scale=guidance_scale,
            device=device,
        )
        self.scheduler = scheduler or GuidanceScheduler(base_scale=1.0)
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        total_steps: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute unconstrained target-only guidance.
        
        Args:
            x_t: Current image tensor
            t: Current timestep
            total_steps: Total number of steps
            
        Returns:
            Guidance gradient = scale * ∇_x ℓ_eff(x)
        """
        x_t = x_t.to(self.device)
        
        # Get scheduled scale
        scale = self.scheduler.get_scale(t, total_steps) * self.guidance_scale
        
        # Compute target gradient
        target_grad = self.compute_target_gradient(x_t)
        
        return scale * target_grad
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "unconstrained",
            "target_label": self.target_label,
            "guidance_scale": self.guidance_scale,
        }


class NegativeGuidance(ClassifierGuidance):
    """
    Negative guidance for decreasing target score.
    
    Used for generating trajectories in the opposite direction
    (e.g., decreasing effusion severity).
    """
    
    name = "negative"
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        target_label: str = "Pleural Effusion",
        guidance_scale: float = 100.0,
        scheduler: Optional[GuidanceScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            evaluator=evaluator,
            target_label=target_label,
            guidance_scale=guidance_scale,
            device=device,
        )
        self.scheduler = scheduler or GuidanceScheduler(base_scale=1.0)
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        total_steps: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """Compute negative (decreasing) guidance."""
        x_t = x_t.to(self.device)
        
        scale = self.scheduler.get_scale(t, total_steps) * self.guidance_scale
        target_grad = self.compute_target_gradient(x_t)
        
        # Negate to decrease target
        return -scale * target_grad
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "negative",
            "target_label": self.target_label,
            "guidance_scale": self.guidance_scale,
        }


class MultiTargetGuidance(ClassifierGuidance):
    """
    Guidance toward multiple targets simultaneously.
    
    Allows specifying multiple target labels with individual weights.
    """
    
    name = "multi_target"
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        target_labels: Dict[str, float],  # label -> weight (positive=increase, negative=decrease)
        guidance_scale: float = 100.0,
        scheduler: Optional[GuidanceScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        # Use first target as primary
        primary_target = list(target_labels.keys())[0]
        super().__init__(
            evaluator=evaluator,
            target_label=primary_target,
            guidance_scale=guidance_scale,
            device=device,
        )
        
        self.target_labels = target_labels
        self.scheduler = scheduler or GuidanceScheduler(base_scale=1.0)
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        total_steps: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """Compute multi-target guidance."""
        x_t = x_t.to(self.device)
        
        scale = self.scheduler.get_scale(t, total_steps) * self.guidance_scale
        
        total_grad = torch.zeros_like(x_t)
        
        for label, weight in self.target_labels.items():
            grad = self.compute_label_gradient(x_t, label)
            total_grad = total_grad + weight * grad
        
        return scale * total_grad
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "multi_target",
            "target_labels": self.target_labels,
            "guidance_scale": self.guidance_scale,
        }


class ClippedGuidance(ClassifierGuidance):
    """
    Guidance with gradient clipping for stability.
    
    Clips gradients to prevent extreme updates that can
    cause image degradation.
    """
    
    name = "clipped"
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        target_label: str = "Pleural Effusion",
        guidance_scale: float = 100.0,
        max_grad_norm: float = 1.0,
        scheduler: Optional[GuidanceScheduler] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__(
            evaluator=evaluator,
            target_label=target_label,
            guidance_scale=guidance_scale,
            device=device,
        )
        self.max_grad_norm = max_grad_norm
        self.scheduler = scheduler or GuidanceScheduler(base_scale=1.0)
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        total_steps: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """Compute clipped guidance."""
        x_t = x_t.to(self.device)
        
        scale = self.scheduler.get_scale(t, total_steps) * self.guidance_scale
        target_grad = self.compute_target_gradient(x_t)
        
        # Clip gradient norm
        grad_norm = torch.norm(target_grad)
        if grad_norm > self.max_grad_norm:
            target_grad = target_grad * self.max_grad_norm / grad_norm
        
        return scale * target_grad
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "clipped",
            "target_label": self.target_label,
            "guidance_scale": self.guidance_scale,
            "max_grad_norm": self.max_grad_norm,
        }
