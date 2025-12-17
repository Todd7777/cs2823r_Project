"""
Ablation Study Analysis.

This module implements the ablation analysis framework described in
Section 3.5 and 5.4, for identifying which off-target constraints
are most critical for preventing drift.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
import json
import numpy as np

from cib_med.core.trajectory import EditTrajectory
from cib_med.core.semantic_coordinates import FindingCategory, STANDARD_FINDINGS
from cib_med.metrics.benchmark import CIBMedBenchmark, BenchmarkResults


@dataclass
class AblationResult:
    """
    Results from a single ablation configuration.
    
    Attributes:
        config_name: Name of ablation configuration
        ablated_components: Components removed in this ablation
        benchmark_results: Full benchmark results
        delta_drift: Change in drift compared to baseline
        delta_progression: Change in progression compared to baseline
        importance_score: Computed importance of ablated components
    """
    config_name: str
    ablated_components: List[str]
    benchmark_results: BenchmarkResults
    delta_drift: float = 0.0
    delta_progression: float = 0.0
    importance_score: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "config_name": self.config_name,
            "ablated_components": self.ablated_components,
            "delta_drift": self.delta_drift,
            "delta_progression": self.delta_progression,
            "importance_score": self.importance_score,
        }


class AblationAnalyzer:
    """
    Analyzer for ablation study results.
    
    Implements analysis protocols from Section 3.5:
    - Leave-one-out importance: Δ_k = D_off^(-k) - D_off
    - Grouped ablation analysis
    - Weight ablation analysis
    
    Args:
        benchmark: CIBMedBenchmark instance for evaluation
        baseline_results: Results from full (non-ablated) configuration
    """
    
    def __init__(
        self,
        benchmark: CIBMedBenchmark,
        baseline_results: Optional[BenchmarkResults] = None,
    ):
        self.benchmark = benchmark
        self.baseline_results = baseline_results
        self.ablation_results: Dict[str, AblationResult] = {}
        
        # Define finding groups for grouped ablations
        self.finding_groups = self._build_finding_groups()
    
    def _build_finding_groups(self) -> Dict[str, List[str]]:
        """Build groups of findings by clinical category."""
        groups = {
            "parenchymal": [],
            "cardiomediastinal": [],
            "pleural": [],
            "chronic": [],
            "artifact": [],
        }
        
        for name, finding in STANDARD_FINDINGS.items():
            if finding.is_target:
                continue
            cat_name = finding.category.value
            if cat_name in groups:
                groups[cat_name].append(name)
        
        return groups
    
    def set_baseline(self, results: BenchmarkResults):
        """Set baseline results for comparison."""
        self.baseline_results = results
    
    def add_ablation_result(
        self,
        config_name: str,
        trajectories: List[EditTrajectory],
        ablated_components: List[str],
    ) -> AblationResult:
        """
        Add and analyze results from an ablation configuration.
        
        Args:
            config_name: Name for this ablation configuration
            trajectories: Trajectories generated with ablation
            ablated_components: List of ablated component names
            
        Returns:
            AblationResult with computed metrics
        """
        # Run benchmark
        benchmark_results = self.benchmark.evaluate(
            trajectories,
            editor_name=f"ablation_{config_name}",
        )
        
        # Compute deltas if baseline exists
        delta_drift = 0.0
        delta_progression = 0.0
        importance_score = 0.0
        
        if self.baseline_results is not None:
            baseline_drift = self.baseline_results.drift_aggregate["aggregate_drift_median"]["mean"]
            ablated_drift = benchmark_results.drift_aggregate["aggregate_drift_median"]["mean"]
            delta_drift = ablated_drift - baseline_drift
            
            baseline_prog = self.baseline_results.progression_aggregate["total_progression"]["mean"]
            ablated_prog = benchmark_results.progression_aggregate["total_progression"]["mean"]
            delta_progression = ablated_prog - baseline_prog
            
            # Importance score: higher delta_drift = more important constraint
            importance_score = delta_drift / (abs(baseline_drift) + 1e-6)
        
        result = AblationResult(
            config_name=config_name,
            ablated_components=ablated_components,
            benchmark_results=benchmark_results,
            delta_drift=delta_drift,
            delta_progression=delta_progression,
            importance_score=importance_score,
        )
        
        self.ablation_results[config_name] = result
        return result
    
    def compute_leave_one_out_importance(self) -> Dict[str, float]:
        """
        Compute leave-one-out importance for each ablated component.
        
        Δ_k := D_off^(-k) - D_off
        
        Returns:
            Dictionary mapping component names to importance values
        """
        importance = {}
        
        for config_name, result in self.ablation_results.items():
            if len(result.ablated_components) == 1:
                component = result.ablated_components[0]
                importance[component] = result.delta_drift
        
        return importance
    
    def compute_grouped_importance(self) -> Dict[str, float]:
        """
        Compute importance for each finding group.
        
        Returns:
            Dictionary mapping group names to importance values
        """
        group_importance = {}
        
        for group_name in self.finding_groups.keys():
            config_name = f"no_{group_name}"
            if config_name in self.ablation_results:
                group_importance[group_name] = self.ablation_results[config_name].delta_drift
        
        return group_importance
    
    def rank_by_importance(
        self,
        importance_type: str = "leave_one_out",
    ) -> List[Tuple[str, float]]:
        """
        Rank components by importance.
        
        Args:
            importance_type: "leave_one_out" or "grouped"
            
        Returns:
            List of (component, importance) tuples sorted by importance
        """
        if importance_type == "leave_one_out":
            importance = self.compute_leave_one_out_importance()
        else:
            importance = self.compute_grouped_importance()
        
        return sorted(importance.items(), key=lambda x: x[1], reverse=True)
    
    def analyze_importance_vs_association(
        self,
        associations: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Analyze relationship between importance and empirical associations.
        
        Args:
            associations: Assoc(k) values from anchor distribution
            
        Returns:
            Correlation analysis results
        """
        from scipy.stats import spearmanr
        
        importance = self.compute_leave_one_out_importance()
        
        # Get paired values
        importance_vals = []
        assoc_vals = []
        
        for component in importance:
            if component in associations:
                importance_vals.append(importance[component])
                assoc_vals.append(abs(associations[component]))
        
        if len(importance_vals) < 3:
            return {"correlation": 0.0, "p_value": 1.0, "n_components": len(importance_vals)}
        
        corr, p_val = spearmanr(importance_vals, assoc_vals)
        
        return {
            "correlation": float(corr) if not np.isnan(corr) else 0.0,
            "p_value": float(p_val) if not np.isnan(p_val) else 1.0,
            "n_components": len(importance_vals),
        }
    
    def generate_report(self) -> str:
        """Generate human-readable ablation analysis report."""
        lines = [
            "=" * 60,
            "CIB-Med-1 Ablation Analysis Report",
            "=" * 60,
            "",
        ]
        
        if self.baseline_results:
            lines.extend([
                "Baseline Configuration:",
                f"  Drift (D_off): {self.baseline_results.drift_aggregate['aggregate_drift_median']['mean']:.4f}",
                f"  Progression: {self.baseline_results.progression_aggregate['total_progression']['mean']:.4f}",
                "",
            ])
        
        # Leave-one-out results
        loo_importance = self.compute_leave_one_out_importance()
        if loo_importance:
            lines.extend([
                "Leave-One-Out Importance (Δ_k):",
                "-" * 40,
            ])
            for component, importance in self.rank_by_importance("leave_one_out"):
                lines.append(f"  {component}: {importance:+.4f}")
            lines.append("")
        
        # Grouped results
        group_importance = self.compute_grouped_importance()
        if group_importance:
            lines.extend([
                "Grouped Ablation Importance:",
                "-" * 40,
            ])
            for group, importance in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {group}: {importance:+.4f}")
            lines.append("")
        
        return "\n".join(lines)
    
    def save(self, path: Path):
        """Save ablation analysis results."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save summary
        summary = {
            "leave_one_out_importance": self.compute_leave_one_out_importance(),
            "grouped_importance": self.compute_grouped_importance(),
            "ablation_results": {
                name: result.to_dict()
                for name, result in self.ablation_results.items()
            },
        }
        
        with open(path / "ablation_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        
        # Save report
        with open(path / "ablation_report.txt", "w") as f:
            f.write(self.generate_report())
    
    @classmethod
    def load(cls, path: Path, benchmark: CIBMedBenchmark) -> "AblationAnalyzer":
        """Load ablation analysis from saved results."""
        path = Path(path)
        
        with open(path / "ablation_summary.json") as f:
            summary = json.load(f)
        
        analyzer = cls(benchmark)
        
        # Reconstruct ablation results
        for name, data in summary.get("ablation_results", {}).items():
            # Note: Full reconstruction would need trajectory data
            pass
        
        return analyzer


def generate_ablation_configs(
    off_target_findings: List[str],
    include_grouped: bool = True,
) -> List[Dict[str, Any]]:
    """
    Generate ablation configuration specifications.
    
    Args:
        off_target_findings: List of off-target finding names
        include_grouped: Whether to include grouped ablations
        
    Returns:
        List of configuration dictionaries
    """
    configs = []
    
    # Baseline (no ablation)
    configs.append({
        "name": "full",
        "ablated": [],
        "description": "Full constraint set (baseline)",
    })
    
    # Leave-one-out ablations
    for finding in off_target_findings:
        configs.append({
            "name": f"no_{finding.replace(' ', '_')}",
            "ablated": [finding],
            "description": f"Ablate {finding} constraint",
        })
    
    # Grouped ablations
    if include_grouped:
        groups = {
            "parenchymal": ["Atelectasis", "Consolidation", "Pneumonia", "Edema", "Lung Opacity"],
            "cardiomediastinal": ["Cardiomegaly", "Enlarged Cardiomediastinum"],
            "pleural": ["Pneumothorax", "Pleural Thickening", "Pleural Other"],
            "chronic": ["Fibrosis", "Emphysema"],
            "artifact": ["Support Devices", "Fracture"],
        }
        
        for group_name, group_findings in groups.items():
            ablated = [f for f in group_findings if f in off_target_findings]
            if ablated:
                configs.append({
                    "name": f"no_{group_name}",
                    "ablated": ablated,
                    "description": f"Ablate {group_name} constraint group",
                })
    
    return configs
