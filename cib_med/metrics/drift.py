"""
Off-Target Semantic Drift Metrics.

This module implements the off-target drift metrics described in Section 3.3,
measuring deviations along non-target clinical axes.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set
import numpy as np
from scipy.stats import spearmanr

from cib_med.core.trajectory import EditTrajectory
from cib_med.core.semantic_coordinates import FindingCategory


@dataclass
class DriftMetrics:
    """
    Container for off-target drift metrics on a single trajectory.
    
    Attributes:
        per_label_drift: D_k for each off-target finding (Eq. 4)
        aggregate_drift_median: D_off - median of per-label drifts
        aggregate_drift_90: D_off^(90) - 90th percentile of drifts
        aggregate_drift_max: Maximum per-label drift
        aggregate_drift_mean: Mean of per-label drifts
        drift_by_category: Aggregated drift by finding category
        max_drift_label: Label with highest drift
        drift_trajectory: Per-step aggregate drift values
    """
    per_label_drift: Dict[str, float]
    aggregate_drift_median: float
    aggregate_drift_90: float
    aggregate_drift_max: float
    aggregate_drift_mean: float
    drift_by_category: Dict[str, float]
    max_drift_label: str
    drift_trajectory: List[float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "per_label_drift": self.per_label_drift,
            "aggregate_drift_median": self.aggregate_drift_median,
            "aggregate_drift_90": self.aggregate_drift_90,
            "aggregate_drift_max": self.aggregate_drift_max,
            "aggregate_drift_mean": self.aggregate_drift_mean,
            "drift_by_category": self.drift_by_category,
            "max_drift_label": self.max_drift_label,
            "drift_trajectory": self.drift_trajectory,
        }


class OffTargetDriftMetrics:
    """
    Computes off-target semantic drift metrics for edit trajectories.
    
    Implements metrics from Section 3.3:
    - Per-label drift D_k (Eq. 4)
    - Aggregate drift D_off (median)
    - Worst-case drift D_off^(90) (90th percentile)
    
    Args:
        off_target_findings: List of off-target finding names
        finding_categories: Mapping from finding names to categories
    """
    
    def __init__(
        self,
        off_target_findings: List[str],
        finding_categories: Optional[Dict[str, FindingCategory]] = None,
    ):
        self.off_target_findings = off_target_findings
        
        # Default category mapping
        if finding_categories is None:
            from cib_med.core.semantic_coordinates import STANDARD_FINDINGS
            self.finding_categories = {
                name: STANDARD_FINDINGS[name].category
                for name in off_target_findings
                if name in STANDARD_FINDINGS
            }
        else:
            self.finding_categories = finding_categories
    
    def compute(self, trajectory: EditTrajectory) -> DriftMetrics:
        """
        Compute drift metrics for a single trajectory.
        
        Args:
            trajectory: EditTrajectory to analyze
            
        Returns:
            DriftMetrics containing all computed metrics
        """
        T = trajectory.num_steps
        if T == 0:
            return self._empty_metrics()
        
        # Get anchor values
        anchor_coords = trajectory.anchor.coordinates
        
        # Compute per-label drift D_k (Eq. 4)
        # D_k := (1/T) Σ_{t=1}^{T} |v_k(x^t) - v_k(x^0)|
        per_label_drift = {}
        per_label_trajectory = {f: [] for f in self.off_target_findings}
        
        for finding in self.off_target_findings:
            anchor_value = anchor_coords.probabilities.get(finding, 0.0)
            
            drift_sum = 0.0
            for step in trajectory.steps[1:]:  # Skip anchor
                step_value = step.coordinates.probabilities.get(finding, 0.0)
                drift_sum += abs(step_value - anchor_value)
                per_label_trajectory[finding].append(abs(step_value - anchor_value))
            
            per_label_drift[finding] = drift_sum / T
        
        # Aggregate metrics
        drift_values = list(per_label_drift.values())
        
        # D_off := median_k D_k
        aggregate_drift_median = float(np.median(drift_values))
        
        # D_off^(90) := quantile_0.9({D_k})
        aggregate_drift_90 = float(np.percentile(drift_values, 90))
        
        # Additional aggregates
        aggregate_drift_max = float(np.max(drift_values))
        aggregate_drift_mean = float(np.mean(drift_values))
        
        # Max drift label
        max_drift_label = max(per_label_drift, key=per_label_drift.get)
        
        # Drift by category
        drift_by_category = self._aggregate_by_category(per_label_drift)
        
        # Per-step aggregate drift trajectory
        drift_trajectory = []
        for t in range(T):
            step_drifts = [per_label_trajectory[f][t] for f in self.off_target_findings]
            drift_trajectory.append(float(np.median(step_drifts)))
        
        return DriftMetrics(
            per_label_drift=per_label_drift,
            aggregate_drift_median=aggregate_drift_median,
            aggregate_drift_90=aggregate_drift_90,
            aggregate_drift_max=aggregate_drift_max,
            aggregate_drift_mean=aggregate_drift_mean,
            drift_by_category=drift_by_category,
            max_drift_label=max_drift_label,
            drift_trajectory=drift_trajectory,
        )
    
    def compute_batch(
        self,
        trajectories: List[EditTrajectory],
    ) -> List[DriftMetrics]:
        """Compute drift metrics for multiple trajectories."""
        return [self.compute(t) for t in trajectories]
    
    def aggregate(
        self,
        metrics_list: List[DriftMetrics],
    ) -> Dict[str, Dict[str, float]]:
        """
        Aggregate drift metrics across multiple trajectories.
        
        Returns statistics for each metric.
        """
        if not metrics_list:
            return {}
        
        aggregated = {}
        
        # Aggregate scalar metrics
        scalar_metrics = [
            "aggregate_drift_median", "aggregate_drift_90",
            "aggregate_drift_max", "aggregate_drift_mean"
        ]
        
        for name in scalar_metrics:
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
        
        # Aggregate per-label drift
        aggregated["per_label_drift"] = {}
        for finding in self.off_target_findings:
            values = np.array([m.per_label_drift.get(finding, 0.0) for m in metrics_list])
            aggregated["per_label_drift"][finding] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "median": float(np.median(values)),
            }
        
        # Aggregate by category
        categories = set(self.finding_categories.values())
        aggregated["drift_by_category"] = {}
        for cat in categories:
            cat_name = cat.value if hasattr(cat, 'value') else str(cat)
            values = np.array([
                m.drift_by_category.get(cat_name, 0.0) for m in metrics_list
            ])
            aggregated["drift_by_category"][cat_name] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }
        
        return aggregated
    
    def _aggregate_by_category(
        self,
        per_label_drift: Dict[str, float],
    ) -> Dict[str, float]:
        """Aggregate drift by finding category."""
        category_drifts = {}
        
        for cat in set(self.finding_categories.values()):
            cat_name = cat.value if hasattr(cat, 'value') else str(cat)
            findings_in_cat = [
                f for f, c in self.finding_categories.items()
                if c == cat and f in per_label_drift
            ]
            
            if findings_in_cat:
                cat_values = [per_label_drift[f] for f in findings_in_cat]
                category_drifts[cat_name] = float(np.mean(cat_values))
            else:
                category_drifts[cat_name] = 0.0
        
        return category_drifts
    
    def _empty_metrics(self) -> DriftMetrics:
        """Return metrics for degenerate trajectory."""
        return DriftMetrics(
            per_label_drift={f: 0.0 for f in self.off_target_findings},
            aggregate_drift_median=0.0,
            aggregate_drift_90=0.0,
            aggregate_drift_max=0.0,
            aggregate_drift_mean=0.0,
            drift_by_category={},
            max_drift_label=self.off_target_findings[0] if self.off_target_findings else "",
            drift_trajectory=[],
        )


class DriftCorrelationAnalyzer:
    """
    Analyzes relationship between drift and dataset associations.
    
    Implements the analysis from Section 6.2, relating per-label drift D_k
    to empirical target-off_target associations Assoc(k).
    """
    
    def __init__(
        self,
        associations: Dict[str, float],
        off_target_findings: List[str],
    ):
        """
        Args:
            associations: Assoc(k) values from anchor distribution
            off_target_findings: List of off-target finding names
        """
        self.associations = associations
        self.off_target_findings = off_target_findings
    
    def analyze(
        self,
        drift_metrics: DriftMetrics,
    ) -> Dict[str, float]:
        """
        Analyze relationship between drift and associations.
        
        Args:
            drift_metrics: DriftMetrics from a trajectory
            
        Returns:
            Dictionary with correlation analysis results
        """
        # Get paired values
        drift_values = []
        assoc_values = []
        
        for finding in self.off_target_findings:
            if finding in drift_metrics.per_label_drift and finding in self.associations:
                drift_values.append(drift_metrics.per_label_drift[finding])
                assoc_values.append(abs(self.associations[finding]))  # Use absolute association
        
        if len(drift_values) < 3:
            return {"drift_assoc_correlation": 0.0, "n_findings": len(drift_values)}
        
        # Compute correlation between drift and association magnitude
        corr, p_value = spearmanr(drift_values, assoc_values)
        
        return {
            "drift_assoc_correlation": float(corr) if not np.isnan(corr) else 0.0,
            "drift_assoc_p_value": float(p_value) if not np.isnan(p_value) else 1.0,
            "n_findings": len(drift_values),
            "mean_drift": float(np.mean(drift_values)),
            "mean_assoc": float(np.mean(assoc_values)),
        }
    
    def analyze_batch(
        self,
        drift_metrics_list: List[DriftMetrics],
    ) -> Dict[str, float]:
        """
        Analyze drift-association relationship across multiple trajectories.
        
        Returns aggregated analysis results.
        """
        results = [self.analyze(m) for m in drift_metrics_list]
        
        correlations = [r["drift_assoc_correlation"] for r in results]
        
        return {
            "mean_drift_assoc_correlation": float(np.mean(correlations)),
            "std_drift_assoc_correlation": float(np.std(correlations)),
            "median_drift_assoc_correlation": float(np.median(correlations)),
            "n_trajectories": len(results),
        }


def compute_signed_drift(
    trajectory: EditTrajectory,
    off_target_findings: List[str],
) -> Dict[str, float]:
    """
    Compute signed (directional) drift for each finding.
    
    Unlike |D_k|, this preserves the direction of change,
    useful for understanding systematic vs. random drift.
    
    Args:
        trajectory: EditTrajectory to analyze
        off_target_findings: List of off-target finding names
        
    Returns:
        Dictionary mapping findings to signed drift values
    """
    if trajectory.num_steps == 0:
        return {f: 0.0 for f in off_target_findings}
    
    anchor_coords = trajectory.anchor.coordinates
    final_coords = trajectory.final.coordinates
    
    signed_drift = {}
    for finding in off_target_findings:
        anchor_value = anchor_coords.probabilities.get(finding, 0.0)
        final_value = final_coords.probabilities.get(finding, 0.0)
        signed_drift[finding] = final_value - anchor_value
    
    return signed_drift


def compute_drift_rate(
    trajectory: EditTrajectory,
    off_target_findings: List[str],
) -> Dict[str, List[float]]:
    """
    Compute per-step drift rate for each finding.
    
    Useful for identifying when drift occurs in the trajectory.
    
    Args:
        trajectory: EditTrajectory to analyze
        off_target_findings: List of off-target finding names
        
    Returns:
        Dictionary mapping findings to lists of per-step drift rates
    """
    drift_rates = {f: [] for f in off_target_findings}
    
    for step_t, step_tp1 in trajectory.iter_pairs():
        for finding in off_target_findings:
            val_t = step_t.coordinates.probabilities.get(finding, 0.0)
            val_tp1 = step_tp1.coordinates.probabilities.get(finding, 0.0)
            drift_rates[finding].append(val_tp1 - val_t)
    
    return drift_rates
