"""
Constrained Diffusion Guidance.

This module implements the constrained diffusion guidance method described
in Section 4, which optimizes target progression subject to bounded off-target drift.
"""

from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from cib_med.core.evaluator import RadiologyEvaluator
from cib_med.core.semantic_coordinates import SemanticCoordinates
from cib_med.guidance.base import ClassifierGuidance, GuidanceScheduler


class ConstrainedDiffusionGuidance(ClassifierGuidance):
    """
    Constrained diffusion guidance for directional medical image editing.
    
    Implements the method from Section 4, combining:
    - Target guidance: ∇_x ℓ_eff(x) to increase effusion
    - Signed off-target barrier: penalizes deviation from anchor values
    - Anchor-adaptive weights: upweights unstable/present findings
    
    The guidance field (Eq. 5) is:
    
    ∇_x G(x) = ∇_x ℓ_eff(x) - λ Σ_k w_k(x^0) sign(v_k(x) - v_k(x^0)) ∇_x ℓ_k(x)
    
    Args:
        evaluator: RadiologyEvaluator for computing gradients
        target_label: Target finding to increase
        off_target_labels: List of off-target findings to constrain
        constraint_strength: λ parameter controlling constraint strength
        anchor_coords: Semantic coordinates of anchor image
        weight_alpha: α parameter for variance-based weighting
        weight_beta: β parameter for presence-based weighting
        guidance_scale: Overall guidance scale
        scheduler: Optional GuidanceScheduler
    """
    
    name = "constrained"
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        target_label: str = "Pleural Effusion",
        off_target_labels: Optional[List[str]] = None,
        constraint_strength: float = 1.0,
        anchor_coords: Optional[SemanticCoordinates] = None,
        weight_alpha: float = 1.0,
        weight_beta: float = 0.5,
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
        
        # Default off-target labels from paper Eq. (3)
        if off_target_labels is None:
            off_target_labels = [
                "Atelectasis", "Consolidation", "Pneumonia", "Edema", "Lung Opacity",
                "Cardiomegaly", "Enlarged Cardiomediastinum",
                "Pneumothorax", "Fracture", "Support Devices"
            ]
        
        # Filter to only labels available in evaluator
        available_labels = set(evaluator.get_label_names())
        self.off_target_labels = [l for l in off_target_labels if l in available_labels]
        
        self.constraint_strength = constraint_strength
        self.weight_alpha = weight_alpha
        self.weight_beta = weight_beta
        self.scheduler = scheduler or GuidanceScheduler(base_scale=1.0)
        
        # Anchor-related state
        self.anchor_coords = anchor_coords
        self.anchor_weights: Optional[Dict[str, float]] = None
        
        if anchor_coords is not None:
            self._compute_anchor_weights()
    
    def set_anchor(
        self,
        anchor_coords: SemanticCoordinates,
        local_variance: Optional[Dict[str, float]] = None,
    ):
        """
        Set anchor coordinates for constraint computation.
        
        Args:
            anchor_coords: Semantic coordinates of anchor image
            local_variance: Optional pre-computed variance estimates
        """
        self.anchor_coords = anchor_coords
        self._compute_anchor_weights(local_variance)
    
    def _compute_anchor_weights(
        self,
        local_variance: Optional[Dict[str, float]] = None,
    ):
        """
        Compute anchor-adaptive weights (Eq. 6).
        
        w_k(x^0) = 1 + α * Var(v_k | p_eff ∈ N(p_eff(x^0))) + β * v_k(x^0)
        """
        if self.anchor_coords is None:
            self.anchor_weights = {k: 1.0 for k in self.off_target_labels}
            return
        
        self.anchor_weights = {}
        
        for label in self.off_target_labels:
            # Base weight
            w = 1.0
            
            # Variance term (from local neighborhood)
            if local_variance is not None and label in local_variance:
                w += self.weight_alpha * local_variance[label]
            
            # Anchor presence term
            anchor_value = self.anchor_coords.probabilities.get(label, 0.0)
            w += self.weight_beta * anchor_value
            
            self.anchor_weights[label] = w
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        total_steps: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute constrained guidance gradient.
        
        Implements Eq. (5):
        ∇_x G(x) = ∇_x ℓ_eff(x) - λ Σ_k w_k sign(v_k - v_k^0) ∇_x ℓ_k(x)
        
        Args:
            x_t: Current image tensor
            t: Current timestep
            total_steps: Total number of edit steps
            **kwargs: Additional arguments
            
        Returns:
            Guidance gradient tensor
        """
        x_t = x_t.to(self.device)
        
        # Get scheduled scale
        scale = self.scheduler.get_scale(t, total_steps) * self.guidance_scale
        
        # Compute target gradient: ∇_x ℓ_eff(x)
        target_grad = self.compute_target_gradient(x_t)
        
        # Initialize total guidance with target term
        guidance = scale * target_grad
        
        # Add off-target constraint terms
        if self.anchor_coords is not None and self.constraint_strength > 0:
            constraint_grad = self._compute_constraint_gradient(x_t)
            guidance = guidance - self.constraint_strength * scale * constraint_grad
        
        return guidance
    
    def _compute_constraint_gradient(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute signed barrier constraint gradient.
        
        Σ_k w_k(x^0) sign(v_k(x) - v_k(x^0)) ∇_x ℓ_k(x)
        """
        x = x.clone().detach().requires_grad_(True)
        
        # Get current scores
        logits = self.evaluator(x)
        probs = torch.sigmoid(logits)
        
        # Accumulate weighted signed gradients
        total_grad = torch.zeros_like(x)
        
        for label in self.off_target_labels:
            try:
                label_idx = self.evaluator.get_label_index(label)
            except ValueError:
                continue
            
            # Get anchor and current values
            anchor_value = self.anchor_coords.probabilities.get(label, 0.5)
            current_value = float(probs[0, label_idx])
            
            # Compute sign of deviation
            sign = np.sign(current_value - anchor_value)
            
            if abs(sign) < 0.01:  # Near anchor, no constraint
                continue
            
            # Get weight
            weight = self.anchor_weights.get(label, 1.0)
            
            # Compute gradient for this label
            x_fresh = x.clone().detach().requires_grad_(True)
            logits_fresh = self.evaluator(x_fresh)
            label_logit = logits_fresh[:, label_idx].sum()
            label_logit.backward()
            
            # Add weighted signed gradient
            total_grad = total_grad + weight * sign * x_fresh.grad
        
        return total_grad
    
    def get_config(self) -> Dict[str, Any]:
        """Return configuration dictionary."""
        return {
            "type": "constrained",
            "target_label": self.target_label,
            "off_target_labels": self.off_target_labels,
            "constraint_strength": self.constraint_strength,
            "weight_alpha": self.weight_alpha,
            "weight_beta": self.weight_beta,
            "guidance_scale": self.guidance_scale,
        }


class AdaptiveConstrainedGuidance(ConstrainedDiffusionGuidance):
    """
    Adaptive constrained guidance with dynamic constraint adjustment.
    
    Automatically adjusts constraint strength based on observed drift
    during the trajectory.
    """
    
    name = "adaptive_constrained"
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        target_label: str = "Pleural Effusion",
        off_target_labels: Optional[List[str]] = None,
        base_constraint_strength: float = 1.0,
        max_constraint_strength: float = 5.0,
        drift_threshold: float = 0.05,
        **kwargs
    ):
        super().__init__(
            evaluator=evaluator,
            target_label=target_label,
            off_target_labels=off_target_labels,
            constraint_strength=base_constraint_strength,
            **kwargs
        )
        
        self.base_constraint_strength = base_constraint_strength
        self.max_constraint_strength = max_constraint_strength
        self.drift_threshold = drift_threshold
        
        # Track drift history
        self.drift_history: List[float] = []
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        total_steps: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """Compute guidance with adaptive constraint strength."""
        # Update constraint strength based on drift history
        self._update_constraint_strength(x_t)
        
        return super().compute_guidance(x_t, t, total_steps, **kwargs)
    
    def _update_constraint_strength(self, x: torch.Tensor):
        """Update constraint strength based on observed drift."""
        if self.anchor_coords is None:
            return
        
        # Compute current drift
        current_scores = self.get_current_scores(x)
        
        drifts = []
        for label in self.off_target_labels:
            anchor_value = self.anchor_coords.probabilities.get(label, 0.0)
            current_value = current_scores.get(label, 0.0)
            drifts.append(abs(current_value - anchor_value))
        
        current_drift = np.median(drifts)
        self.drift_history.append(current_drift)
        
        # Adjust constraint strength
        if current_drift > self.drift_threshold:
            # Increase constraint strength
            increase_factor = 1.0 + (current_drift - self.drift_threshold) / self.drift_threshold
            self.constraint_strength = min(
                self.base_constraint_strength * increase_factor,
                self.max_constraint_strength
            )
        else:
            # Gradually return to base strength
            self.constraint_strength = max(
                self.base_constraint_strength,
                self.constraint_strength * 0.95
            )
    
    def reset(self):
        """Reset adaptive state for new trajectory."""
        self.drift_history = []
        self.constraint_strength = self.base_constraint_strength


class ProjectedGuidance(ConstrainedDiffusionGuidance):
    """
    Projected gradient guidance.
    
    Projects the target gradient onto the nullspace of off-target
    gradients, ensuring orthogonality to constraint directions.
    """
    
    name = "projected"
    
    def __init__(
        self,
        evaluator: RadiologyEvaluator,
        projection_strength: float = 1.0,
        **kwargs
    ):
        super().__init__(evaluator=evaluator, **kwargs)
        self.projection_strength = projection_strength
    
    def compute_guidance(
        self,
        x_t: torch.Tensor,
        t: int,
        total_steps: int = 20,
        **kwargs
    ) -> torch.Tensor:
        """Compute projected guidance gradient."""
        x_t = x_t.to(self.device)
        
        scale = self.scheduler.get_scale(t, total_steps) * self.guidance_scale
        
        # Compute target gradient
        target_grad = self.compute_target_gradient(x_t)
        target_grad_flat = target_grad.view(-1)
        
        # Compute off-target gradients and project
        if self.off_target_labels:
            off_target_grads = []
            for label in self.off_target_labels:
                try:
                    grad = self.compute_label_gradient(x_t, label)
                    off_target_grads.append(grad.view(-1))
                except:
                    continue
            
            if off_target_grads:
                # Stack gradients into matrix
                A = torch.stack(off_target_grads, dim=0)  # [K, D]
                
                # Project target onto nullspace of A
                # g_proj = g - A^T (A A^T)^{-1} A g
                AAt = torch.mm(A, A.t())  # [K, K]
                AAt_inv = torch.inverse(AAt + 1e-6 * torch.eye(AAt.shape[0], device=self.device))
                Ag = torch.mv(A, target_grad_flat)  # [K]
                correction = torch.mv(A.t(), torch.mv(AAt_inv, Ag))  # [D]
                
                projected_grad = target_grad_flat - self.projection_strength * correction
                target_grad = projected_grad.view(target_grad.shape)
        
        return scale * target_grad


def create_guidance(
    guidance_type: str,
    evaluator: RadiologyEvaluator,
    **kwargs
) -> ClassifierGuidance:
    """
    Factory function to create guidance methods.
    
    Args:
        guidance_type: One of "constrained", "adaptive", "projected", "unconstrained"
        evaluator: RadiologyEvaluator instance
        **kwargs: Additional arguments for the guidance method
        
    Returns:
        Guidance method instance
    """
    if guidance_type == "constrained":
        return ConstrainedDiffusionGuidance(evaluator=evaluator, **kwargs)
    elif guidance_type == "adaptive":
        return AdaptiveConstrainedGuidance(evaluator=evaluator, **kwargs)
    elif guidance_type == "projected":
        return ProjectedGuidance(evaluator=evaluator, **kwargs)
    elif guidance_type == "unconstrained":
        from cib_med.guidance.unconstrained import UnconstrainedGuidance
        return UnconstrainedGuidance(evaluator=evaluator, **kwargs)
    else:
        raise ValueError(f"Unknown guidance type: {guidance_type}")
