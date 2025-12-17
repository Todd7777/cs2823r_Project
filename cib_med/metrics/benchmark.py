"""
CIB-Med-1 Benchmark Implementation.

This module provides the complete benchmark evaluation pipeline,
combining progression and drift metrics with analysis tools.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import json
import numpy as np
from datetime import datetime

from cib_med.core.trajectory import EditTrajectory, TrajectoryDataset
from cib_med.core.semantic_coordinates import SemanticCoordinateSystem
from cib_med.metrics.progression import MonotoneProgressionMetrics, ProgressionMetrics
from cib_med.metrics.drift import OffTargetDriftMetrics, DriftMetrics, DriftCorrelationAnalyzer


@dataclass
class TrajectoryResult:
    """Results for a single trajectory."""
    trajectory_id: str
    anchor_id: str
    editor_name: str
    progression: ProgressionMetrics
    drift: DriftMetrics
    
    def to_dict(self) -> Dict:
        return {
            "trajectory_id": self.trajectory_id,
            "anchor_id": self.anchor_id,
            "editor_name": self.editor_name,
            "progression": self.progression.to_dict(),
            "drift": self.drift.to_dict(),
        }


@dataclass
class BenchmarkResults:
    """
    Complete benchmark results for an editing method.
    
    Attributes:
        editor_name: Name of the editing method
        num_trajectories: Number of trajectories evaluated
        trajectory_results: Individual trajectory results
        progression_aggregate: Aggregated progression metrics
        drift_aggregate: Aggregated drift metrics
        drift_correlation_analysis: Analysis of drift vs. association
        pareto_efficiency: Progression-drift tradeoff metrics
        timestamp: Evaluation timestamp
        config: Benchmark configuration used
    """
    editor_name: str
    num_trajectories: int
    trajectory_results: List[TrajectoryResult]
    progression_aggregate: Dict[str, Dict[str, float]]
    drift_aggregate: Dict[str, Dict[str, float]]
    drift_correlation_analysis: Optional[Dict[str, float]] = None
    pareto_efficiency: Optional[Dict[str, float]] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    config: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "editor_name": self.editor_name,
            "num_trajectories": self.num_trajectories,
            "trajectory_results": [r.to_dict() for r in self.trajectory_results],
            "progression_aggregate": self.progression_aggregate,
            "drift_aggregate": self.drift_aggregate,
            "drift_correlation_analysis": self.drift_correlation_analysis,
            "pareto_efficiency": self.pareto_efficiency,
            "timestamp": self.timestamp,
            "config": self.config,
        }
    
    def save(self, path: Path):
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, path: Path) -> "BenchmarkResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        # Reconstruct trajectory results
        trajectory_results = []
        for tr_data in data.get("trajectory_results", []):
            trajectory_results.append(TrajectoryResult(
                trajectory_id=tr_data["trajectory_id"],
                anchor_id=tr_data["anchor_id"],
                editor_name=tr_data["editor_name"],
                progression=ProgressionMetrics(**tr_data["progression"]),
                drift=DriftMetrics(**tr_data["drift"]),
            ))
        
        return cls(
            editor_name=data["editor_name"],
            num_trajectories=data["num_trajectories"],
            trajectory_results=trajectory_results,
            progression_aggregate=data["progression_aggregate"],
            drift_aggregate=data["drift_aggregate"],
            drift_correlation_analysis=data.get("drift_correlation_analysis"),
            pareto_efficiency=data.get("pareto_efficiency"),
            timestamp=data.get("timestamp", ""),
            config=data.get("config", {}),
        )
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"CIB-Med-1 Benchmark Results: {self.editor_name}",
            f"=" * 50,
            f"Trajectories evaluated: {self.num_trajectories}",
            f"",
            f"Target Progression:",
            f"  Trend correlation (ρ): {self.progression_aggregate['trend_correlation']['mean']:.3f} ± {self.progression_aggregate['trend_correlation']['std']:.3f}",
            f"  Inversion rate: {self.progression_aggregate['inversion_rate']['mean']:.3f} ± {self.progression_aggregate['inversion_rate']['std']:.3f}",
            f"  Total progression: {self.progression_aggregate['total_progression']['mean']:.3f} ± {self.progression_aggregate['total_progression']['std']:.3f}",
            f"",
            f"Off-Target Drift:",
            f"  D_off (median): {self.drift_aggregate['aggregate_drift_median']['mean']:.4f} ± {self.drift_aggregate['aggregate_drift_median']['std']:.4f}",
            f"  D_off^90: {self.drift_aggregate['aggregate_drift_90']['mean']:.4f} ± {self.drift_aggregate['aggregate_drift_90']['std']:.4f}",
            f"  D_off (max): {self.drift_aggregate['aggregate_drift_max']['mean']:.4f} ± {self.drift_aggregate['aggregate_drift_max']['std']:.4f}",
        ]
        
        if self.pareto_efficiency:
            lines.extend([
                f"",
                f"Pareto Efficiency:",
                f"  Efficiency score: {self.pareto_efficiency.get('efficiency_score', 0):.3f}",
            ])
        
        return "\n".join(lines)


class CIBMedBenchmark:
    """
    Main benchmark class for CIB-Med-1 evaluation.
    
    Orchestrates trajectory-level evaluation using progression and drift metrics.
    
    Args:
        coordinate_system: SemanticCoordinateSystem for computing coordinates
        off_target_findings: List of off-target finding names
        associations: Optional pre-computed target-off_target associations
    """
    
    def __init__(
        self,
        coordinate_system: SemanticCoordinateSystem,
        off_target_findings: Optional[List[str]] = None,
        associations: Optional[Dict[str, float]] = None,
    ):
        self.coordinate_system = coordinate_system
        
        if off_target_findings is None:
            off_target_findings = coordinate_system.off_target_findings
        self.off_target_findings = off_target_findings
        
        self.associations = associations
        
        # Initialize metric computers
        self.progression_metrics = MonotoneProgressionMetrics()
        self.drift_metrics = OffTargetDriftMetrics(off_target_findings)
        
        if associations is not None:
            self.drift_analyzer = DriftCorrelationAnalyzer(
                associations, off_target_findings
            )
        else:
            self.drift_analyzer = None
    
    def evaluate_trajectory(
        self,
        trajectory: EditTrajectory,
    ) -> TrajectoryResult:
        """
        Evaluate a single trajectory.
        
        Args:
            trajectory: EditTrajectory to evaluate
            
        Returns:
            TrajectoryResult with progression and drift metrics
        """
        progression = self.progression_metrics.compute(trajectory)
        drift = self.drift_metrics.compute(trajectory)
        
        return TrajectoryResult(
            trajectory_id=f"{trajectory.anchor_id}_{trajectory.editor_name}",
            anchor_id=trajectory.anchor_id,
            editor_name=trajectory.editor_name,
            progression=progression,
            drift=drift,
        )
    
    def evaluate(
        self,
        trajectories: List[EditTrajectory],
        editor_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BenchmarkResults:
        """
        Evaluate multiple trajectories and compute aggregate metrics.
        
        Args:
            trajectories: List of trajectories to evaluate
            editor_name: Name of the editing method
            config: Configuration used for generation
            
        Returns:
            BenchmarkResults with all metrics and analysis
        """
        if not trajectories:
            raise ValueError("No trajectories to evaluate")
        
        if editor_name is None:
            editor_name = trajectories[0].editor_name
        
        # Evaluate individual trajectories
        trajectory_results = [self.evaluate_trajectory(t) for t in trajectories]
        
        # Aggregate progression metrics
        progression_list = [r.progression for r in trajectory_results]
        progression_aggregate = self.progression_metrics.aggregate(progression_list)
        
        # Aggregate drift metrics
        drift_list = [r.drift for r in trajectory_results]
        drift_aggregate = self.drift_metrics.aggregate(drift_list)
        
        # Drift-association correlation analysis
        drift_correlation_analysis = None
        if self.drift_analyzer is not None:
            drift_correlation_analysis = self.drift_analyzer.analyze_batch(drift_list)
        
        # Compute Pareto efficiency metrics
        pareto_efficiency = self._compute_pareto_efficiency(
            progression_list, drift_list
        )
        
        return BenchmarkResults(
            editor_name=editor_name,
            num_trajectories=len(trajectories),
            trajectory_results=trajectory_results,
            progression_aggregate=progression_aggregate,
            drift_aggregate=drift_aggregate,
            drift_correlation_analysis=drift_correlation_analysis,
            pareto_efficiency=pareto_efficiency,
            config=config or {},
        )
    
    def evaluate_dataset(
        self,
        dataset: TrajectoryDataset,
        editor_name: Optional[str] = None,
    ) -> BenchmarkResults:
        """Evaluate a trajectory dataset."""
        return self.evaluate(list(dataset), editor_name)
    
    def compare_methods(
        self,
        results_list: List[BenchmarkResults],
    ) -> Dict[str, Any]:
        """
        Compare multiple editing methods.
        
        Args:
            results_list: List of BenchmarkResults for different methods
            
        Returns:
            Comparison dictionary with rankings and analysis
        """
        comparison = {
            "methods": [r.editor_name for r in results_list],
            "rankings": {},
            "metrics": {},
        }
        
        # Extract key metrics for comparison
        metrics_to_compare = [
            ("trend_correlation", "progression_aggregate", True),  # Higher is better
            ("inversion_rate", "progression_aggregate", False),  # Lower is better
            ("total_progression", "progression_aggregate", True),
            ("aggregate_drift_median", "drift_aggregate", False),  # Lower is better
            ("aggregate_drift_90", "drift_aggregate", False),
        ]
        
        for metric_name, aggregate_name, higher_better in metrics_to_compare:
            values = []
            for r in results_list:
                agg = getattr(r, aggregate_name, {})
                if metric_name in agg:
                    values.append(agg[metric_name]["mean"])
                else:
                    values.append(float('nan'))
            
            comparison["metrics"][metric_name] = {
                r.editor_name: v for r, v in zip(results_list, values)
            }
            
            # Compute ranking
            valid_indices = [i for i, v in enumerate(values) if not np.isnan(v)]
            if valid_indices:
                sorted_indices = sorted(
                    valid_indices,
                    key=lambda i: values[i],
                    reverse=higher_better
                )
                ranking = {results_list[i].editor_name: rank + 1 
                          for rank, i in enumerate(sorted_indices)}
                comparison["rankings"][metric_name] = ranking
        
        # Compute overall ranking (average of individual rankings)
        overall_ranks = {}
        for r in results_list:
            ranks = [
                comparison["rankings"][m].get(r.editor_name, len(results_list))
                for m in comparison["rankings"]
            ]
            overall_ranks[r.editor_name] = np.mean(ranks)
        
        comparison["overall_ranking"] = dict(sorted(
            overall_ranks.items(), key=lambda x: x[1]
        ))
        
        return comparison
    
    def _compute_pareto_efficiency(
        self,
        progression_list: List[ProgressionMetrics],
        drift_list: List[DriftMetrics],
    ) -> Dict[str, float]:
        """
        Compute Pareto efficiency metrics for progression-drift tradeoff.
        
        The Pareto frontier represents the optimal tradeoff between
        maximizing progression and minimizing drift.
        """
        if not progression_list or not drift_list:
            return {}
        
        # Extract progression and drift values
        progressions = np.array([p.total_progression for p in progression_list])
        drifts = np.array([d.aggregate_drift_median for d in drift_list])
        
        # Compute efficiency score (progression per unit drift)
        # Higher is better - more progression for less drift
        with np.errstate(divide='ignore', invalid='ignore'):
            efficiency = progressions / (drifts + 1e-6)
            efficiency = np.where(np.isfinite(efficiency), efficiency, 0.0)
        
        # Find Pareto-optimal points
        pareto_mask = self._find_pareto_front(progressions, -drifts)
        
        return {
            "efficiency_score": float(np.mean(efficiency)),
            "efficiency_std": float(np.std(efficiency)),
            "pareto_fraction": float(np.mean(pareto_mask)),
            "mean_progression": float(np.mean(progressions)),
            "mean_drift": float(np.mean(drifts)),
            "progression_per_drift": float(np.mean(progressions) / (np.mean(drifts) + 1e-6)),
        }
    
    def _find_pareto_front(
        self,
        objective1: np.ndarray,
        objective2: np.ndarray,
    ) -> np.ndarray:
        """Find Pareto-optimal points (both objectives to maximize)."""
        n = len(objective1)
        pareto_mask = np.ones(n, dtype=bool)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if j dominates i
                    if (objective1[j] >= objective1[i] and 
                        objective2[j] >= objective2[i] and
                        (objective1[j] > objective1[i] or objective2[j] > objective2[i])):
                        pareto_mask[i] = False
                        break
        
        return pareto_mask


class AblationBenchmark:
    """
    Ablation study framework for CIB-Med-1.
    
    Implements the ablation protocol from Section 5.4:
    - Leave-one-out ablations
    - Grouped ablations
    - Weight ablations
    """
    
    def __init__(
        self,
        base_benchmark: CIBMedBenchmark,
        coordinate_system: SemanticCoordinateSystem,
    ):
        self.base_benchmark = base_benchmark
        self.coordinate_system = coordinate_system
    
    def leave_one_out_ablation(
        self,
        trajectories_by_config: Dict[str, List[EditTrajectory]],
    ) -> Dict[str, BenchmarkResults]:
        """
        Perform leave-one-out ablation analysis.
        
        Args:
            trajectories_by_config: Mapping from ablation config name to trajectories
                                   e.g., {"full": [...], "no_Edema": [...], ...}
                                   
        Returns:
            Dictionary mapping config names to benchmark results
        """
        results = {}
        for config_name, trajectories in trajectories_by_config.items():
            results[config_name] = self.base_benchmark.evaluate(
                trajectories,
                editor_name=f"ablation_{config_name}",
            )
        return results
    
    def compute_ablation_importance(
        self,
        ablation_results: Dict[str, BenchmarkResults],
        baseline_key: str = "full",
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute importance scores for ablated components.
        
        Δ_k := D_off^(-k) - D_off (from Section 3.5)
        
        Args:
            ablation_results: Results from leave_one_out_ablation
            baseline_key: Key for the full (non-ablated) configuration
            
        Returns:
            Dictionary mapping ablated components to importance metrics
        """
        if baseline_key not in ablation_results:
            raise ValueError(f"Baseline key '{baseline_key}' not found in results")
        
        baseline = ablation_results[baseline_key]
        baseline_drift = baseline.drift_aggregate["aggregate_drift_median"]["mean"]
        baseline_progression = baseline.progression_aggregate["total_progression"]["mean"]
        
        importance = {}
        for config_name, results in ablation_results.items():
            if config_name == baseline_key:
                continue
            
            ablated_drift = results.drift_aggregate["aggregate_drift_median"]["mean"]
            ablated_progression = results.progression_aggregate["total_progression"]["mean"]
            
            # Δ_k for drift (positive means removing k increases drift)
            delta_drift = ablated_drift - baseline_drift
            
            # Delta for progression (negative means removing k decreases progression)
            delta_progression = ablated_progression - baseline_progression
            
            importance[config_name] = {
                "delta_drift": delta_drift,
                "delta_progression": delta_progression,
                "drift_impact": delta_drift / (baseline_drift + 1e-6),
                "progression_impact": delta_progression / (abs(baseline_progression) + 1e-6),
            }
        
        return importance
    
    def grouped_ablation(
        self,
        trajectories_by_group: Dict[str, List[EditTrajectory]],
    ) -> Tuple[Dict[str, BenchmarkResults], Dict[str, Dict[str, float]]]:
        """
        Perform grouped ablation analysis.
        
        Groups findings by category (parenchymal, cardiomediastinal, etc.)
        and evaluates the effect of removing each group.
        
        Args:
            trajectories_by_group: Mapping from group name to trajectories
            
        Returns:
            Tuple of (results_dict, importance_dict)
        """
        results = self.leave_one_out_ablation(trajectories_by_group)
        importance = self.compute_ablation_importance(results)
        return results, importance
