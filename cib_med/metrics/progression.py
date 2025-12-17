"""
Monotone Target Progression Metrics.

This module implements the target progression metrics described in Section 3.2,
measuring how consistently trajectories move along the target clinical axis.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Tuple
import numpy as np
from scipy.stats import spearmanr, kendalltau

from cib_med.core.trajectory import EditTrajectory


@dataclass
class ProgressionMetrics:
    """
    Container for progression metrics on a single trajectory.
    
    Attributes:
        trend_correlation: Spearman correlation ρ_trend between step index and target score
        inversion_rate: Fraction of steps violating monotonicity
        kendall_tau: Kendall's tau correlation (alternative rank metric)
        total_progression: Total change in target score from anchor to final
        mean_step_progression: Average progression per step
        max_regression: Maximum single-step decrease in target score
        progression_consistency: Ratio of positive steps to total steps
        final_target_score: Target score at final step
        anchor_target_score: Target score at anchor
    """
    trend_correlation: float
    inversion_rate: float
    kendall_tau: float
    total_progression: float
    mean_step_progression: float
    max_regression: float
    progression_consistency: float
    final_target_score: float
    anchor_target_score: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "trend_correlation": self.trend_correlation,
            "inversion_rate": self.inversion_rate,
            "kendall_tau": self.kendall_tau,
            "total_progression": self.total_progression,
            "mean_step_progression": self.mean_step_progression,
            "max_regression": self.max_regression,
            "progression_consistency": self.progression_consistency,
            "final_target_score": self.final_target_score,
            "anchor_target_score": self.anchor_target_score,
        }


class MonotoneProgressionMetrics:
    """
    Computes monotone target progression metrics for edit trajectories.
    
    Implements the metrics from Section 3.2:
    - Trend correlation ρ_trend (Spearman)
    - Inversion rate
    
    Plus additional progression analysis metrics.
    """
    
    def __init__(self):
        pass
    
    def compute(self, trajectory: EditTrajectory) -> ProgressionMetrics:
        """
        Compute progression metrics for a single trajectory.
        
        Args:
            trajectory: EditTrajectory to analyze
            
        Returns:
            ProgressionMetrics containing all computed metrics
        """
        # Extract target progression
        target_values = trajectory.get_target_progression()
        T = len(target_values) - 1  # Number of edit steps
        
        if T == 0:
            return self._empty_metrics(target_values[0])
        
        # Step indices
        step_indices = np.arange(len(target_values))
        
        # Trend correlation (Eq. from Section 3.2)
        # ρ_trend := corr_Spearman(t, p_eff(x^t))
        trend_corr, _ = spearmanr(step_indices, target_values)
        trend_corr = float(trend_corr) if not np.isnan(trend_corr) else 0.0
        
        # Kendall's tau (alternative rank correlation)
        kendall, _ = kendalltau(step_indices, target_values)
        kendall = float(kendall) if not np.isnan(kendall) else 0.0
        
        # Inversion rate (Eq. from Section 3.2)
        # InvRate := (1/T) Σ I[p_eff(x^{t+1}) < p_eff(x^t)]
        step_diffs = np.diff(target_values)
        inversions = np.sum(step_diffs < 0)
        inversion_rate = float(inversions / T)
        
        # Total progression
        total_progression = float(target_values[-1] - target_values[0])
        
        # Mean step progression
        mean_step_progression = float(np.mean(step_diffs))
        
        # Maximum regression (worst single-step decrease)
        max_regression = float(min(0.0, np.min(step_diffs)))
        
        # Progression consistency (ratio of forward steps)
        positive_steps = np.sum(step_diffs > 0)
        progression_consistency = float(positive_steps / T)
        
        return ProgressionMetrics(
            trend_correlation=trend_corr,
            inversion_rate=inversion_rate,
            kendall_tau=kendall,
            total_progression=total_progression,
            mean_step_progression=mean_step_progression,
            max_regression=max_regression,
            progression_consistency=progression_consistency,
            final_target_score=float(target_values[-1]),
            anchor_target_score=float(target_values[0]),
        )
    
    def compute_batch(
        self,
        trajectories: List[EditTrajectory],
    ) -> List[ProgressionMetrics]:
        """Compute progression metrics for multiple trajectories."""
        return [self.compute(t) for t in trajectories]
    
    def aggregate(
        self,
        metrics_list: List[ProgressionMetrics],
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate progression metrics across multiple trajectories.
        
        Returns statistics (mean, std, median, min, max) for each metric.
        """
        if not metrics_list:
            return {}
        
        # Collect values for each metric
        metric_names = [
            "trend_correlation", "inversion_rate", "kendall_tau",
            "total_progression", "mean_step_progression", "max_regression",
            "progression_consistency", "final_target_score", "anchor_target_score"
        ]
        
        aggregated = {}
        for name in metric_names:
            values = np.array([getattr(m, name) for m in metrics_list])
            aggregated[name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.percentile(values, 25)),
                "q75": float(np.percentile(values, 75)),
            }
        
        return aggregated
    
    def _empty_metrics(self, anchor_score: float) -> ProgressionMetrics:
        """Return metrics for degenerate trajectory (T=0)."""
        return ProgressionMetrics(
            trend_correlation=0.0,
            inversion_rate=0.0,
            kendall_tau=0.0,
            total_progression=0.0,
            mean_step_progression=0.0,
            max_regression=0.0,
            progression_consistency=0.0,
            final_target_score=anchor_score,
            anchor_target_score=anchor_score,
        )


def compute_human_correlation(
    model_ordering: List[int],
    human_ordering: List[int],
) -> Dict[str, float]:
    """
    Compute correlation between model-predicted and human-perceived ordering.
    
    Used for validation against human expert judgment (Section 5.5).
    
    Args:
        model_ordering: Ranking indices from model trajectory
        human_ordering: Ranking indices from human raters
        
    Returns:
        Dictionary with Kendall tau and Spearman correlation
    """
    kendall, p_kendall = kendalltau(model_ordering, human_ordering)
    spearman, p_spearman = spearmanr(model_ordering, human_ordering)
    
    return {
        "kendall_tau": float(kendall) if not np.isnan(kendall) else 0.0,
        "kendall_p_value": float(p_kendall) if not np.isnan(p_kendall) else 1.0,
        "spearman_rho": float(spearman) if not np.isnan(spearman) else 0.0,
        "spearman_p_value": float(p_spearman) if not np.isnan(p_spearman) else 1.0,
    }


def compute_trajectory_smoothness(target_values: np.ndarray) -> Dict[str, float]:
    """
    Compute smoothness metrics for target progression.
    
    Measures how smooth/jerky the progression is.
    
    Args:
        target_values: Array of target scores over trajectory
        
    Returns:
        Dictionary with smoothness metrics
    """
    if len(target_values) < 3:
        return {"smoothness": 1.0, "jerk": 0.0, "acceleration_variance": 0.0}
    
    # First derivative (velocity)
    velocity = np.diff(target_values)
    
    # Second derivative (acceleration)
    acceleration = np.diff(velocity)
    
    # Jerk (third derivative) - measure of non-smoothness
    if len(acceleration) > 1:
        jerk = np.diff(acceleration)
        jerk_magnitude = float(np.mean(np.abs(jerk)))
    else:
        jerk_magnitude = 0.0
    
    # Acceleration variance
    acc_variance = float(np.var(acceleration)) if len(acceleration) > 0 else 0.0
    
    # Smoothness score (inverse of jerk, normalized)
    smoothness = 1.0 / (1.0 + jerk_magnitude * 10)
    
    return {
        "smoothness": smoothness,
        "jerk": jerk_magnitude,
        "acceleration_variance": acc_variance,
        "velocity_mean": float(np.mean(velocity)),
        "velocity_std": float(np.std(velocity)),
    }
