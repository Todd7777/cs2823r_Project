"""
Correlation and Association Analysis.

This module implements the correlation analysis described in Section 6,
examining the relationship between drift magnitude and dataset associations.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.stats import linregress

from cib_med.core.trajectory import EditTrajectory
from cib_med.core.semantic_coordinates import SemanticCoordinates
from cib_med.metrics.drift import DriftMetrics


@dataclass
class CorrelationResult:
    """Results from correlation analysis."""
    correlation: float
    p_value: float
    method: str
    n_samples: int
    slope: Optional[float] = None
    intercept: Optional[float] = None
    r_squared: Optional[float] = None


class CorrelationAnalyzer:
    """
    Analyzes correlations in CIB-Med-1 benchmark results.
    
    Implements analyses from Section 6:
    - Target-off_target associations in anchor distribution
    - Drift-association relationships
    - Drift structure across methods
    """
    
    def __init__(self, off_target_findings: List[str]):
        self.off_target_findings = off_target_findings
    
    def compute_anchor_associations(
        self,
        anchor_coordinates: List[SemanticCoordinates],
        method: str = "spearman",
    ) -> Dict[str, CorrelationResult]:
        """
        Compute associations between target and off-target coordinates
        in the anchor distribution.
        
        Implements Assoc(k) from Section 6.1:
        Assoc(k) = corr_Spearman(p_eff(x^0), v_k(x^0))
        
        Args:
            anchor_coordinates: List of anchor SemanticCoordinates
            method: Correlation method ("spearman", "pearson", "kendall")
            
        Returns:
            Dictionary mapping finding names to CorrelationResult
        """
        n = len(anchor_coordinates)
        if n < 3:
            return {f: CorrelationResult(0.0, 1.0, method, n) 
                    for f in self.off_target_findings}
        
        # Extract target values
        target_values = np.array([c.calibrated_target for c in anchor_coordinates])
        
        results = {}
        for finding in self.off_target_findings:
            off_target_values = np.array([
                c.probabilities.get(finding, 0.0) for c in anchor_coordinates
            ])
            
            # Compute correlation
            if method == "spearman":
                corr, p_val = spearmanr(target_values, off_target_values)
            elif method == "pearson":
                corr, p_val = pearsonr(target_values, off_target_values)
            elif method == "kendall":
                corr, p_val = kendalltau(target_values, off_target_values)
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Linear regression for slope
            slope, intercept, r_val, _, _ = linregress(target_values, off_target_values)
            
            results[finding] = CorrelationResult(
                correlation=float(corr) if not np.isnan(corr) else 0.0,
                p_value=float(p_val) if not np.isnan(p_val) else 1.0,
                method=method,
                n_samples=n,
                slope=float(slope),
                intercept=float(intercept),
                r_squared=float(r_val ** 2),
            )
        
        return results
    
    def analyze_drift_vs_association(
        self,
        drift_metrics_list: List[DriftMetrics],
        associations: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Analyze relationship between drift magnitude and associations.
        
        From Section 6.2: "labels with larger empirical association to the
        target tend to exhibit larger drift under unconstrained editing."
        
        Args:
            drift_metrics_list: List of DriftMetrics from trajectories
            associations: Assoc(k) values from anchor distribution
            
        Returns:
            Analysis results including correlation
        """
        # Aggregate drift per label across trajectories
        aggregated_drift = {}
        for finding in self.off_target_findings:
            drifts = [m.per_label_drift.get(finding, 0.0) for m in drift_metrics_list]
            aggregated_drift[finding] = np.mean(drifts)
        
        # Prepare paired values
        drift_values = []
        assoc_values = []
        findings = []
        
        for finding in self.off_target_findings:
            if finding in associations and finding in aggregated_drift:
                drift_values.append(aggregated_drift[finding])
                assoc_values.append(abs(associations[finding]))
                findings.append(finding)
        
        if len(drift_values) < 3:
            return {
                "correlation": 0.0,
                "p_value": 1.0,
                "n_findings": len(drift_values),
            }
        
        drift_arr = np.array(drift_values)
        assoc_arr = np.array(assoc_values)
        
        # Compute correlation
        corr, p_val = spearmanr(drift_arr, assoc_arr)
        
        # Linear fit
        slope, intercept, r_val, _, _ = linregress(assoc_arr, drift_arr)
        
        return {
            "correlation": float(corr) if not np.isnan(corr) else 0.0,
            "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
            "n_findings": len(drift_values),
            "slope": float(slope),
            "intercept": float(intercept),
            "r_squared": float(r_val ** 2),
            "per_finding": {
                f: {"drift": d, "association": a}
                for f, d, a in zip(findings, drift_values, assoc_values)
            },
        }
    
    def compare_drift_structure(
        self,
        drift_metrics_by_method: Dict[str, List[DriftMetrics]],
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare drift structure across different editing methods.
        
        Analyzes whether different methods have similar drift patterns.
        
        Args:
            drift_metrics_by_method: Mapping from method name to DriftMetrics list
            
        Returns:
            Pairwise correlation of drift patterns between methods
        """
        # Compute drift profiles per method
        profiles = {}
        for method, metrics_list in drift_metrics_by_method.items():
            profile = {}
            for finding in self.off_target_findings:
                drifts = [m.per_label_drift.get(finding, 0.0) for m in metrics_list]
                profile[finding] = np.mean(drifts)
            profiles[method] = profile
        
        # Compute pairwise correlations
        methods = list(profiles.keys())
        comparisons = {}
        
        for i, method1 in enumerate(methods):
            for method2 in methods[i+1:]:
                values1 = [profiles[method1][f] for f in self.off_target_findings]
                values2 = [profiles[method2][f] for f in self.off_target_findings]
                
                corr, p_val = spearmanr(values1, values2)
                
                key = f"{method1}_vs_{method2}"
                comparisons[key] = {
                    "correlation": float(corr) if not np.isnan(corr) else 0.0,
                    "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
                }
        
        return comparisons
    
    def compute_inter_finding_correlations(
        self,
        anchor_coordinates: List[SemanticCoordinates],
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute correlations between pairs of off-target findings.
        
        Useful for understanding co-occurrence patterns.
        
        Args:
            anchor_coordinates: List of anchor SemanticCoordinates
            
        Returns:
            Dictionary mapping (finding1, finding2) to correlation
        """
        n = len(anchor_coordinates)
        if n < 3:
            return {}
        
        correlations = {}
        
        for i, finding1 in enumerate(self.off_target_findings):
            for finding2 in self.off_target_findings[i+1:]:
                values1 = np.array([
                    c.probabilities.get(finding1, 0.0) for c in anchor_coordinates
                ])
                values2 = np.array([
                    c.probabilities.get(finding2, 0.0) for c in anchor_coordinates
                ])
                
                corr, _ = spearmanr(values1, values2)
                correlations[(finding1, finding2)] = float(corr) if not np.isnan(corr) else 0.0
        
        return correlations
    
    def identify_entangled_findings(
        self,
        associations: Dict[str, float],
        threshold: float = 0.3,
    ) -> Tuple[List[str], List[str]]:
        """
        Identify highly entangled vs. independent findings.
        
        Args:
            associations: Assoc(k) values
            threshold: Correlation threshold for entanglement
            
        Returns:
            Tuple of (entangled_findings, independent_findings)
        """
        entangled = []
        independent = []
        
        for finding, assoc in associations.items():
            if abs(assoc) >= threshold:
                entangled.append(finding)
            else:
                independent.append(finding)
        
        return entangled, independent


def analyze_mnar_indicators(
    anchor_coordinates: List[SemanticCoordinates],
    metadata: Optional[List[Dict]] = None,
) -> Dict[str, float]:
    """
    Analyze indicators of missing-not-at-random (MNAR) structure.
    
    From Section 2.4: Clinical imaging datasets are observational and
    not missing at random.
    
    Args:
        anchor_coordinates: List of anchor coordinates
        metadata: Optional metadata (e.g., patient demographics)
        
    Returns:
        MNAR analysis results
    """
    results = {}
    
    # Check for target score distribution skewness
    target_scores = np.array([c.calibrated_target for c in anchor_coordinates])
    
    from scipy.stats import skew, kurtosis
    
    results["target_skewness"] = float(skew(target_scores))
    results["target_kurtosis"] = float(kurtosis(target_scores))
    
    # Check for finding co-occurrence patterns
    # High co-occurrence might indicate selection effects
    finding_means = {}
    for coord in anchor_coordinates:
        for finding, prob in coord.probabilities.items():
            if finding not in finding_means:
                finding_means[finding] = []
            finding_means[finding].append(prob)
    
    for finding, probs in finding_means.items():
        results[f"{finding}_prevalence"] = float(np.mean(probs))
        results[f"{finding}_variance"] = float(np.var(probs))
    
    return results
