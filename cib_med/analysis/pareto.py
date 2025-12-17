"""
Pareto Efficiency Analysis.

This module implements the progression-drift tradeoff analysis
described in Section 5.3, identifying Pareto-optimal operating points.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.spatial import ConvexHull

from cib_med.metrics.benchmark import BenchmarkResults


@dataclass
class ParetoPoint:
    """A point in the progression-drift space."""
    method_name: str
    progression: float
    drift: float
    is_pareto_optimal: bool = False
    efficiency: float = 0.0  # progression / drift ratio


class ParetoAnalyzer:
    """
    Analyzes Pareto efficiency of editing methods.
    
    The Pareto frontier represents the optimal tradeoff between
    maximizing progression and minimizing drift.
    """
    
    def __init__(self):
        self.points: List[ParetoPoint] = []
        self.pareto_frontier: List[ParetoPoint] = []
    
    def add_method_results(
        self,
        results: BenchmarkResults,
        method_name: Optional[str] = None,
    ):
        """Add benchmark results for a method."""
        if method_name is None:
            method_name = results.editor_name
        
        progression = results.progression_aggregate["total_progression"]["mean"]
        drift = results.drift_aggregate["aggregate_drift_median"]["mean"]
        
        efficiency = progression / (drift + 1e-6) if drift > 0 else float('inf')
        
        point = ParetoPoint(
            method_name=method_name,
            progression=progression,
            drift=drift,
            efficiency=efficiency,
        )
        
        self.points.append(point)
    
    def add_point(
        self,
        method_name: str,
        progression: float,
        drift: float,
    ):
        """Add a single point manually."""
        efficiency = progression / (drift + 1e-6) if drift > 0 else float('inf')
        
        point = ParetoPoint(
            method_name=method_name,
            progression=progression,
            drift=drift,
            efficiency=efficiency,
        )
        
        self.points.append(point)
    
    def compute_pareto_frontier(self) -> List[ParetoPoint]:
        """
        Compute the Pareto frontier.
        
        A point is Pareto-optimal if no other point has both
        higher progression AND lower drift.
        
        Returns:
            List of Pareto-optimal points
        """
        if not self.points:
            return []
        
        # Reset optimality flags
        for p in self.points:
            p.is_pareto_optimal = True
        
        # Check each point against all others
        for i, p1 in enumerate(self.points):
            for j, p2 in enumerate(self.points):
                if i == j:
                    continue
                
                # p2 dominates p1 if p2 has >= progression and <= drift,
                # with at least one strict inequality
                if (p2.progression >= p1.progression and 
                    p2.drift <= p1.drift and
                    (p2.progression > p1.progression or p2.drift < p1.drift)):
                    p1.is_pareto_optimal = False
                    break
        
        self.pareto_frontier = [p for p in self.points if p.is_pareto_optimal]
        
        # Sort by progression
        self.pareto_frontier.sort(key=lambda p: p.progression)
        
        return self.pareto_frontier
    
    def compute_area_under_frontier(self) -> float:
        """
        Compute area under the Pareto frontier.
        
        Larger area indicates better overall performance.
        
        Returns:
            Area under frontier (normalized)
        """
        if len(self.pareto_frontier) < 2:
            return 0.0
        
        # Sort by drift (x-axis)
        sorted_points = sorted(self.pareto_frontier, key=lambda p: p.drift)
        
        # Compute area using trapezoidal rule
        area = 0.0
        for i in range(len(sorted_points) - 1):
            p1, p2 = sorted_points[i], sorted_points[i + 1]
            width = p2.drift - p1.drift
            height = (p1.progression + p2.progression) / 2
            area += width * height
        
        return area
    
    def compute_hypervolume(
        self,
        reference_point: Tuple[float, float] = (0.0, 1.0),
    ) -> float:
        """
        Compute hypervolume indicator.
        
        The hypervolume is the volume of the space dominated by
        the Pareto frontier, bounded by a reference point.
        
        Args:
            reference_point: (min_progression, max_drift) reference
            
        Returns:
            Hypervolume value
        """
        if not self.pareto_frontier:
            return 0.0
        
        ref_prog, ref_drift = reference_point
        
        # Filter points that dominate reference
        valid_points = [
            p for p in self.pareto_frontier
            if p.progression > ref_prog and p.drift < ref_drift
        ]
        
        if not valid_points:
            return 0.0
        
        # Sort by drift ascending
        sorted_points = sorted(valid_points, key=lambda p: p.drift)
        
        # Compute hypervolume
        hypervolume = 0.0
        prev_drift = ref_drift
        
        for p in sorted_points:
            width = prev_drift - p.drift
            height = p.progression - ref_prog
            hypervolume += width * height
            prev_drift = p.drift
        
        return hypervolume
    
    def find_knee_point(self) -> Optional[ParetoPoint]:
        """
        Find the "knee" point on the Pareto frontier.
        
        The knee is where the tradeoff between progression and drift
        changes most rapidly, indicating diminishing returns.
        
        Returns:
            Knee point or None
        """
        if len(self.pareto_frontier) < 3:
            return self.pareto_frontier[0] if self.pareto_frontier else None
        
        # Sort by progression
        sorted_points = sorted(self.pareto_frontier, key=lambda p: p.progression)
        
        # Compute second derivative (curvature)
        max_curvature = 0.0
        knee_point = sorted_points[0]
        
        for i in range(1, len(sorted_points) - 1):
            p0, p1, p2 = sorted_points[i-1], sorted_points[i], sorted_points[i+1]
            
            # First derivatives
            d1_prog = p1.progression - p0.progression
            d1_drift = p1.drift - p0.drift
            d2_prog = p2.progression - p1.progression
            d2_drift = p2.drift - p1.drift
            
            # Approximate curvature
            if abs(d1_drift) > 1e-6 and abs(d2_drift) > 1e-6:
                slope1 = d1_prog / d1_drift
                slope2 = d2_prog / d2_drift
                curvature = abs(slope2 - slope1)
                
                if curvature > max_curvature:
                    max_curvature = curvature
                    knee_point = p1
        
        return knee_point
    
    def compute_efficiency_ranking(self) -> List[Tuple[str, float]]:
        """
        Rank methods by efficiency (progression/drift ratio).
        
        Returns:
            List of (method_name, efficiency) sorted by efficiency
        """
        rankings = [(p.method_name, p.efficiency) for p in self.points]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def compute_dominance_matrix(self) -> Dict[str, Dict[str, bool]]:
        """
        Compute pairwise dominance relationships.
        
        Returns:
            Matrix where [A][B] = True if A dominates B
        """
        matrix = {}
        
        for p1 in self.points:
            matrix[p1.method_name] = {}
            for p2 in self.points:
                if p1.method_name == p2.method_name:
                    matrix[p1.method_name][p2.method_name] = False
                else:
                    dominates = (
                        p1.progression >= p2.progression and
                        p1.drift <= p2.drift and
                        (p1.progression > p2.progression or p1.drift < p2.drift)
                    )
                    matrix[p1.method_name][p2.method_name] = dominates
        
        return matrix
    
    def get_summary(self) -> Dict:
        """Generate summary of Pareto analysis."""
        self.compute_pareto_frontier()
        
        return {
            "num_methods": len(self.points),
            "num_pareto_optimal": len(self.pareto_frontier),
            "pareto_optimal_methods": [p.method_name for p in self.pareto_frontier],
            "knee_point": self.find_knee_point().method_name if self.find_knee_point() else None,
            "hypervolume": self.compute_hypervolume(),
            "efficiency_ranking": self.compute_efficiency_ranking(),
            "points": [
                {
                    "method": p.method_name,
                    "progression": p.progression,
                    "drift": p.drift,
                    "efficiency": p.efficiency,
                    "is_pareto_optimal": p.is_pareto_optimal,
                }
                for p in self.points
            ],
        }


def compute_constraint_sweep(
    results_by_lambda: Dict[float, BenchmarkResults],
) -> Dict[str, List[float]]:
    """
    Analyze results across different constraint strengths (λ).
    
    From Section 5.3: "Varying the constraint strength λ reveals
    a smooth tradeoff between target progression and off-target stability."
    
    Args:
        results_by_lambda: Mapping from λ values to benchmark results
        
    Returns:
        Dictionary with progression and drift arrays indexed by λ
    """
    lambdas = sorted(results_by_lambda.keys())
    
    progressions = []
    drifts = []
    trend_correlations = []
    
    for lam in lambdas:
        r = results_by_lambda[lam]
        progressions.append(r.progression_aggregate["total_progression"]["mean"])
        drifts.append(r.drift_aggregate["aggregate_drift_median"]["mean"])
        trend_correlations.append(r.progression_aggregate["trend_correlation"]["mean"])
    
    return {
        "lambdas": lambdas,
        "progressions": progressions,
        "drifts": drifts,
        "trend_correlations": trend_correlations,
    }
