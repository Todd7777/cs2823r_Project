"""
Base classes for diffusion guidance methods.

This module provides the abstract interface for guidance methods
used in directional medical image editing.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable, Tuple
import torch
import torch.nn as nn

from cib_med.core.evaluator import RadiologyEvaluator


class GuidanceMethod(ABC):
    """
    Abstract base class for diffusion guidance methods.
    
    Guidance methods modify the diffusion sampling process to steer
    generated images toward desired semantic properties.
    """
    
    name: str = "base"
    
    @abstractmethod
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute guidance gradient for the current diffusion step.
        
        Args:
            x_t: Current noisy image at timestep t
            t: Current diffusion timestep
            **kwargs: Additional arguments (e.g., anchor, noise_pred)
            
        Returns:
            Guidance gradient tensor with same shape as x_t
        """
        pass
    
    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary for this guidance method."""
        pass
    
    def __call__(
        self,
        x_t: torch.Tensor,
        t: int,
        **kwargs
    ) -> torch.Tensor:
        """Apply guidance to current sample."""
        return self.compute_guidance(x_t, t, **kwargs)


class ClassifierGuidance(GuidanceMethod):
    """
    Base class for classifier-based guidance methods.
    
    Uses gradients from a frozen classifier to guide diffusion sampling.
    
    Args:
        evaluator: RadiologyEvaluator for computing classifier gradients
        target_label: Name of target label to optimize
        guidance_scale: Multiplier for guidance gradient
        device: Compute device
    """
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        target_label: str = "Pleural Effusion",
        guidance_scale: float = 1.0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.evaluator = evaluator
        self.target_label = target_label
        self.guidance_scale = guidance_scale
        self.device = device
        
        self.target_idx = evaluator.get_label_index(target_label)
    
    def compute_target_gradient(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of target logit with respect to input.
        
        ∇_x ℓ_eff(x)
        
        Args:
            x: Input image tensor (requires_grad will be set)
            
        Returns:
            Gradient tensor
        """
        x = x.clone().detach().requires_grad_(True)
        
        logits = self.evaluator(x)
        target_logit = logits[:, self.target_idx].sum()
        
        target_logit.backward()
        
        return x.grad
    
    def compute_label_gradient(
        self,
        x: torch.Tensor,
        label_name: str,
    ) -> torch.Tensor:
        """
        Compute gradient of a specific label's logit.
        
        Args:
            x: Input image tensor
            label_name: Name of the label
            
        Returns:
            Gradient tensor
        """
        x = x.clone().detach().requires_grad_(True)
        
        logits = self.evaluator(x)
        label_idx = self.evaluator.get_label_index(label_name)
        label_logit = logits[:, label_idx].sum()
        
        label_logit.backward()
        
        return x.grad
    
    def get_current_scores(
        self,
        x: torch.Tensor,
    ) -> Dict[str, float]:
        """Get current classifier scores for all labels."""
        with torch.no_grad():
            logits = self.evaluator(x)
            probs = torch.sigmoid(logits)
        
        label_names = self.evaluator.get_label_names()
        return {
            name: float(probs[0, i])
            for i, name in enumerate(label_names)
        }


class GuidanceScheduler:
    """
    Schedules guidance strength over diffusion timesteps.
    
    Different scheduling strategies can improve guidance effectiveness
    and image quality.
    """
    
    def __init__(
        self,
        base_scale: float = 1.0,
        schedule_type: str = "constant",
        warmup_steps: int = 0,
        decay_rate: float = 0.0,
    ):
        self.base_scale = base_scale
        self.schedule_type = schedule_type
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
    
    def get_scale(self, t: int, total_steps: int) -> float:
        """
        Get guidance scale for timestep t.
        
        Args:
            t: Current timestep (0 to total_steps-1)
            total_steps: Total number of diffusion steps
            
        Returns:
            Guidance scale multiplier
        """
        progress = t / max(total_steps - 1, 1)
        
        if self.schedule_type == "constant":
            scale = self.base_scale
            
        elif self.schedule_type == "linear_decay":
            scale = self.base_scale * (1.0 - progress * self.decay_rate)
            
        elif self.schedule_type == "cosine":
            import math
            scale = self.base_scale * (1.0 + math.cos(math.pi * progress)) / 2.0
            
        elif self.schedule_type == "warmup":
            if t < self.warmup_steps:
                warmup_progress = t / max(self.warmup_steps, 1)
                scale = self.base_scale * warmup_progress
            else:
                scale = self.base_scale
                
        elif self.schedule_type == "warmup_decay":
            if t < self.warmup_steps:
                warmup_progress = t / max(self.warmup_steps, 1)
                scale = self.base_scale * warmup_progress
            else:
                post_warmup_progress = (t - self.warmup_steps) / max(total_steps - self.warmup_steps - 1, 1)
                scale = self.base_scale * (1.0 - post_warmup_progress * self.decay_rate)
        else:
            scale = self.base_scale
        
        return max(0.0, scale)


class CompositeGuidance(GuidanceMethod):
    """
    Combines multiple guidance methods.
    
    Useful for combining target guidance with off-target constraints
    or multiple objectives.
    """
    
    name = "composite"
    
    def __init__(
        self,
        guidances: list,
        weights: Optional[list] = None,
    ):
        self.guidances = guidances
        
        if weights is None:
            weights = [1.0] * len(guidances)
        self.weights = weights
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        **kwargs
    ) -> torch.Tensor:
        """Compute weighted sum of guidance gradients."""
        total_guidance = torch.zeros_like(x_t)
        
        for guidance, weight in zip(self.guidances, self.weights):
            grad = guidance.compute_guidance(x_t, t, **kwargs)
            total_guidance = total_guidance + weight * grad
        
        return total_guidance
    
    def get_config(self) -> Dict[str, Any]:
        return {
            "type": "composite",
            "num_components": len(self.guidances),
            "weights": self.weights,
            "component_configs": [g.get_config() for g in self.guidances],
        }
