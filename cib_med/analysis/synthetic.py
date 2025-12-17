"""
Synthetic and Semi-Synthetic Stress Tests.

This module implements the stress tests described in Section 7,
for validating benchmark metrics under controlled conditions.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SyntheticFactors:
    """Latent factors for synthetic model."""
    z_eff: float  # Target effusion factor
    z_inf: float  # Off-target infection/opacity factor
    z_art: float  # Artifact/texture factor
    
    def to_array(self) -> np.ndarray:
        return np.array([self.z_eff, self.z_inf, self.z_art])


class SyntheticLatentModel:
    """
    Fully synthetic latent-factor model for controlled experiments.
    
    From Section 7.2: "We define three latent variables: z_eff (target),
    z_inf (off-target infection/opacity), z_art (artifact)."
    
    The evaluator score depends on both target and off-target:
    ℓ_eff(x) = z_eff + ρ * z_inf
    
    where ρ controls spurious coupling.
    
    Args:
        coupling_rho: Coupling strength between target and off-target
        feature_dim: Dimension of synthetic "image" features
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        coupling_rho: float = 0.0,
        feature_dim: int = 256,
        seed: int = 42,
    ):
        self.coupling_rho = coupling_rho
        self.feature_dim = feature_dim
        self.rng = np.random.RandomState(seed)
        
        # Fixed mixing matrix Φ
        self.mixing_matrix = self.rng.randn(feature_dim, 3)
        self.mixing_matrix = self.mixing_matrix / np.linalg.norm(
            self.mixing_matrix, axis=0, keepdims=True
        )
    
    def generate_features(self, factors: SyntheticFactors) -> np.ndarray:
        """
        Generate synthetic features from latent factors.
        
        x = Φ(z_eff, z_inf, z_art)
        """
        z = factors.to_array()
        return self.mixing_matrix @ z
    
    def compute_target_score(self, factors: SyntheticFactors) -> float:
        """
        Compute target (effusion) score with coupling.
        
        ℓ_eff = z_eff + ρ * z_inf
        """
        return factors.z_eff + self.coupling_rho * factors.z_inf
    
    def compute_off_target_scores(self, factors: SyntheticFactors) -> Dict[str, float]:
        """Compute off-target scores (only depend on their own factors)."""
        return {
            "infection": factors.z_inf,
            "artifact": factors.z_art,
        }
    
    def simulate_unconstrained_edit(
        self,
        initial_factors: SyntheticFactors,
        step_size: float = 0.1,
        num_steps: int = 10,
    ) -> List[SyntheticFactors]:
        """
        Simulate unconstrained editing that exploits coupling.
        
        Under unconstrained editing, the optimizer can increase z_inf
        as a shortcut to increase ℓ_eff when ρ > 0.
        """
        trajectory = [initial_factors]
        current = initial_factors
        
        for _ in range(num_steps):
            # Gradient of ℓ_eff w.r.t factors: [1, ρ, 0]
            # Unconstrained editing moves along this gradient
            new_z_eff = current.z_eff + step_size
            new_z_inf = current.z_inf + step_size * self.coupling_rho
            new_z_art = current.z_art + step_size * 0.01 * self.rng.randn()
            
            current = SyntheticFactors(new_z_eff, new_z_inf, new_z_art)
            trajectory.append(current)
        
        return trajectory
    
    def simulate_constrained_edit(
        self,
        initial_factors: SyntheticFactors,
        step_size: float = 0.1,
        num_steps: int = 10,
        constraint_strength: float = 1.0,
    ) -> List[SyntheticFactors]:
        """
        Simulate constrained editing that maintains off-target stability.
        
        Constrained editing suppresses changes to z_inf and z_art.
        """
        trajectory = [initial_factors]
        current = initial_factors
        anchor_inf = initial_factors.z_inf
        anchor_art = initial_factors.z_art
        
        for _ in range(num_steps):
            # Target gradient: increase z_eff
            new_z_eff = current.z_eff + step_size
            
            # Off-target constrained to stay near anchor
            drift_inf = current.z_inf - anchor_inf
            drift_art = current.z_art - anchor_art
            
            new_z_inf = current.z_inf - constraint_strength * drift_inf * 0.5
            new_z_art = current.z_art - constraint_strength * drift_art * 0.5
            
            # Small noise
            new_z_inf += 0.01 * self.rng.randn()
            new_z_art += 0.01 * self.rng.randn()
            
            current = SyntheticFactors(new_z_eff, new_z_inf, new_z_art)
            trajectory.append(current)
        
        return trajectory
    
    def compute_trajectory_metrics(
        self,
        trajectory: List[SyntheticFactors],
    ) -> Dict[str, float]:
        """Compute metrics for a synthetic trajectory."""
        target_scores = [self.compute_target_score(f) for f in trajectory]
        inf_values = [f.z_inf for f in trajectory]
        art_values = [f.z_art for f in trajectory]
        
        # Target progression
        progression = target_scores[-1] - target_scores[0]
        
        # Off-target drift
        anchor_inf, anchor_art = inf_values[0], art_values[0]
        drift_inf = np.mean([abs(v - anchor_inf) for v in inf_values[1:]])
        drift_art = np.mean([abs(v - anchor_art) for v in art_values[1:]])
        
        # Monotonicity
        inversions = sum(1 for i in range(len(target_scores)-1) 
                        if target_scores[i+1] < target_scores[i])
        inversion_rate = inversions / (len(target_scores) - 1)
        
        return {
            "progression": progression,
            "drift_infection": drift_inf,
            "drift_artifact": drift_art,
            "aggregate_drift": (drift_inf + drift_art) / 2,
            "inversion_rate": inversion_rate,
        }


class SemiSyntheticPerturbation:
    """
    Semi-synthetic perturbations for stress testing on real images.
    
    From Section 7.1: Starting from anchor x^0, construct:
    x̃^0(γ) = x^0 + γ * m
    
    where m is a localized opacity mask.
    """
    
    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.image_size = image_size
        self.device = device
    
    def create_opacity_mask(
        self,
        location: str = "upper_lobe",
        intensity: float = 1.0,
        blur_sigma: float = 10.0,
    ) -> torch.Tensor:
        """
        Create a localized opacity mask.
        
        Args:
            location: "upper_lobe", "lower_lobe", "cardiac", "diffuse"
            intensity: Mask intensity
            blur_sigma: Gaussian blur for smooth edges
            
        Returns:
            Mask tensor [1, H, W]
        """
        H, W = self.image_size
        mask = torch.zeros(1, H, W, device=self.device)
        
        if location == "upper_lobe":
            # Upper region (inconsistent with pleural effusion)
            mask[:, :H//3, W//4:3*W//4] = intensity
        elif location == "lower_lobe":
            # Lower region (consistent with effusion)
            mask[:, 2*H//3:, W//4:3*W//4] = intensity
        elif location == "cardiac":
            # Central cardiac region
            cy, cx = H // 2, W // 2
            Y, X = torch.meshgrid(
                torch.arange(H, device=self.device),
                torch.arange(W, device=self.device),
                indexing='ij'
            )
            dist = ((Y - cy) ** 2 + (X - cx) ** 2).float().sqrt()
            mask[:] = intensity * torch.exp(-dist / (H // 4))
        elif location == "diffuse":
            # Diffuse throughout
            mask[:] = intensity * 0.5
        
        # Apply Gaussian blur for smooth edges
        if blur_sigma > 0:
            mask = self._gaussian_blur(mask, blur_sigma)
        
        return mask
    
    def _gaussian_blur(
        self,
        tensor: torch.Tensor,
        sigma: float,
    ) -> torch.Tensor:
        """Apply Gaussian blur to tensor."""
        kernel_size = int(6 * sigma + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        
        # Create Gaussian kernel
        x = torch.arange(kernel_size, device=self.device) - kernel_size // 2
        gauss = torch.exp(-x ** 2 / (2 * sigma ** 2))
        kernel = gauss / gauss.sum()
        
        # Apply separable convolution
        tensor = tensor.unsqueeze(0)  # [1, 1, H, W]
        
        # Horizontal
        tensor = F.conv2d(
            tensor,
            kernel.view(1, 1, 1, -1),
            padding=(0, kernel_size // 2),
        )
        # Vertical
        tensor = F.conv2d(
            tensor,
            kernel.view(1, 1, -1, 1),
            padding=(kernel_size // 2, 0),
        )
        
        return tensor.squeeze(0)
    
    def apply_perturbation(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Apply perturbation to image.
        
        x̃ = x + γ * m (darkening) or x̃ = x - γ * m (brightening)
        """
        image = image.to(self.device)
        perturbed = image - gamma * mask  # Darkening (adding opacity)
        return perturbed.clamp(0, 1)
    
    def generate_perturbation_sweep(
        self,
        image: torch.Tensor,
        gammas: List[float],
        location: str = "upper_lobe",
    ) -> List[Tuple[float, torch.Tensor]]:
        """
        Generate a sweep of perturbations at different intensities.
        
        Args:
            image: Original image
            gammas: List of perturbation strengths
            location: Perturbation location
            
        Returns:
            List of (gamma, perturbed_image) tuples
        """
        mask = self.create_opacity_mask(location=location)
        
        results = []
        for gamma in gammas:
            perturbed = self.apply_perturbation(image, mask, gamma)
            results.append((gamma, perturbed))
        
        return results


class StressTestSuite:
    """
    Complete stress test suite for CIB-Med-1 validation.
    
    Combines synthetic and semi-synthetic tests to validate
    that metrics behave as intended.
    """
    
    def __init__(
        self,
        coordinate_system=None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.coordinate_system = coordinate_system
        self.device = device
        
        self.synthetic_model = SyntheticLatentModel()
        self.perturbation = SemiSyntheticPerturbation(device=device)
    
    def run_coupling_test(
        self,
        rho_values: List[float] = [0.0, 0.3, 0.5, 0.7, 1.0],
        num_trajectories: int = 10,
    ) -> Dict[str, Dict[str, float]]:
        """
        Test how drift scales with coupling strength.
        
        From Section 7.2: "When ρ = 0, unconstrained editing increases
        z_eff without changing z_inf... As ρ increases, unconstrained
        editing increasingly raises z_inf as a shortcut."
        """
        results = {}
        
        for rho in rho_values:
            model = SyntheticLatentModel(coupling_rho=rho)
            
            all_metrics = []
            for _ in range(num_trajectories):
                # Random initial factors
                initial = SyntheticFactors(
                    z_eff=np.random.uniform(0.2, 0.4),
                    z_inf=np.random.uniform(0.1, 0.3),
                    z_art=np.random.uniform(0.0, 0.1),
                )
                
                # Unconstrained trajectory
                traj = model.simulate_unconstrained_edit(initial)
                metrics = model.compute_trajectory_metrics(traj)
                all_metrics.append(metrics)
            
            # Aggregate
            results[f"rho_{rho}"] = {
                "mean_progression": np.mean([m["progression"] for m in all_metrics]),
                "mean_drift": np.mean([m["aggregate_drift"] for m in all_metrics]),
                "mean_drift_infection": np.mean([m["drift_infection"] for m in all_metrics]),
            }
        
        return results
    
    def run_constraint_effectiveness_test(
        self,
        constraint_strengths: List[float] = [0.0, 0.5, 1.0, 2.0, 5.0],
        coupling_rho: float = 0.5,
    ) -> Dict[str, Dict[str, float]]:
        """
        Test how constraint strength affects drift.
        
        From Section 7.2: "Constrained guidance suppresses this behavior,
        maintaining stability in z_inf until the constraint binds."
        """
        model = SyntheticLatentModel(coupling_rho=coupling_rho)
        results = {}
        
        for strength in constraint_strengths:
            all_metrics = []
            for _ in range(10):
                initial = SyntheticFactors(
                    z_eff=np.random.uniform(0.2, 0.4),
                    z_inf=np.random.uniform(0.1, 0.3),
                    z_art=np.random.uniform(0.0, 0.1),
                )
                
                traj = model.simulate_constrained_edit(
                    initial, constraint_strength=strength
                )
                metrics = model.compute_trajectory_metrics(traj)
                all_metrics.append(metrics)
            
            results[f"lambda_{strength}"] = {
                "mean_progression": np.mean([m["progression"] for m in all_metrics]),
                "mean_drift": np.mean([m["aggregate_drift"] for m in all_metrics]),
            }
        
        return results
    
    def run_perturbation_detection_test(
        self,
        anchor_image: torch.Tensor,
        gammas: List[float] = [0.0, 0.05, 0.1, 0.15, 0.2],
    ) -> Dict[str, List[float]]:
        """
        Test if drift metrics detect semi-synthetic perturbations.
        
        From Section 7.1: "As γ increases, we observe monotone increases
        in the corresponding off-target coordinates."
        """
        if self.coordinate_system is None:
            return {"error": "coordinate_system required for perturbation test"}
        
        results = {"gammas": gammas, "drift_values": [], "target_values": []}
        
        perturbed_images = self.perturbation.generate_perturbation_sweep(
            anchor_image, gammas, location="upper_lobe"
        )
        
        anchor_coords = self.coordinate_system.compute_coordinates(anchor_image)
        
        for gamma, perturbed in perturbed_images:
            perturbed_coords = self.coordinate_system.compute_coordinates(perturbed)
            
            # Compute drift from anchor
            drifts = []
            for finding in self.coordinate_system.off_target_findings:
                anchor_val = anchor_coords.probabilities.get(finding, 0.0)
                perturbed_val = perturbed_coords.probabilities.get(finding, 0.0)
                drifts.append(abs(perturbed_val - anchor_val))
            
            results["drift_values"].append(np.median(drifts))
            results["target_values"].append(perturbed_coords.calibrated_target)
        
        return results
    
    def generate_report(self) -> str:
        """Generate stress test report."""
        coupling_results = self.run_coupling_test()
        constraint_results = self.run_constraint_effectiveness_test()
        
        lines = [
            "=" * 60,
            "CIB-Med-1 Stress Test Report",
            "=" * 60,
            "",
            "1. Coupling Test (Synthetic)",
            "-" * 40,
        ]
        
        for config, metrics in coupling_results.items():
            lines.append(f"  {config}:")
            lines.append(f"    Progression: {metrics['mean_progression']:.4f}")
            lines.append(f"    Drift: {metrics['mean_drift']:.4f}")
        
        lines.extend([
            "",
            "2. Constraint Effectiveness Test",
            "-" * 40,
        ])
        
        for config, metrics in constraint_results.items():
            lines.append(f"  {config}:")
            lines.append(f"    Progression: {metrics['mean_progression']:.4f}")
            lines.append(f"    Drift: {metrics['mean_drift']:.4f}")
        
        return "\n".join(lines)
