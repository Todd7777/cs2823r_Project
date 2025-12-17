"""Visualization tools for CIB-Med-1 benchmark results."""

from cib_med.visualization.plots import (
    plot_trajectory,
    plot_drift_heatmap,
    plot_pareto_frontier,
    plot_ablation_importance,
    plot_drift_vs_association,
)
from cib_med.visualization.figures import FigureGenerator

__all__ = [
    "plot_trajectory",
    "plot_drift_heatmap",
    "plot_pareto_frontier",
    "plot_ablation_importance",
    "plot_drift_vs_association",
    "FigureGenerator",
]
