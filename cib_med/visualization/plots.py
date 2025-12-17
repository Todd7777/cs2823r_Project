"""
Plotting functions for CIB-Med-1 visualization.

Provides publication-quality figures for benchmark results.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.lines import Line2D
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from cib_med.core.trajectory import EditTrajectory
from cib_med.metrics.benchmark import BenchmarkResults
from cib_med.metrics.drift import DriftMetrics


def _check_matplotlib():
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib and seaborn required for visualization")


def plot_trajectory(
    trajectory: EditTrajectory,
    off_target_findings: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Plot target progression and off-target drift for a trajectory.
    
    Args:
        trajectory: EditTrajectory to visualize
        off_target_findings: Specific findings to plot (default: top 5 by drift)
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    
    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Get data
    target_values = trajectory.get_target_progression()
    steps = np.arange(len(target_values))
    
    # Plot target progression
    ax1 = axes[0]
    ax1.plot(steps, target_values, 'b-o', linewidth=2, markersize=8, label='Effusion Score')
    ax1.set_ylabel('Target Score (p_eff)', fontsize=12)
    ax1.set_title(f'Trajectory: {trajectory.editor_name}', fontsize=14)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Highlight inversions
    for i in range(len(target_values) - 1):
        if target_values[i + 1] < target_values[i]:
            ax1.axvspan(i, i + 1, alpha=0.2, color='red')
    
    # Plot off-target findings
    ax2 = axes[1]
    
    if off_target_findings is None:
        # Get all available off-target findings
        sample_coords = trajectory.steps[0].coordinates
        off_target_findings = [
            k for k in sample_coords.probabilities.keys()
            if k != "Pleural Effusion"
        ][:5]  # Top 5
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(off_target_findings)))
    
    anchor_values = {
        f: trajectory.anchor.coordinates.probabilities.get(f, 0.0)
        for f in off_target_findings
    }
    
    for finding, color in zip(off_target_findings, colors):
        values = trajectory.get_off_target_progression(finding)
        # Plot relative to anchor
        relative_values = values - anchor_values[finding]
        ax2.plot(steps, relative_values, '-o', color=color, linewidth=1.5,
                markersize=5, label=finding, alpha=0.8)
    
    ax2.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax2.set_xlabel('Edit Step', fontsize=12)
    ax2.set_ylabel('Drift from Anchor', fontsize=12)
    ax2.legend(loc='upper right', fontsize=9, ncol=2)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drift_heatmap(
    drift_metrics_list: List[DriftMetrics],
    off_target_findings: List[str],
    method_names: Optional[List[str]] = None,
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Plot heatmap of per-label drift across methods/trajectories.
    
    Args:
        drift_metrics_list: List of DriftMetrics
        off_target_findings: Finding names for y-axis
        method_names: Optional method names for x-axis
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    
    # Build drift matrix
    n_methods = len(drift_metrics_list)
    n_findings = len(off_target_findings)
    
    drift_matrix = np.zeros((n_findings, n_methods))
    
    for j, dm in enumerate(drift_metrics_list):
        for i, finding in enumerate(off_target_findings):
            drift_matrix[i, j] = dm.per_label_drift.get(finding, 0.0)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    im = ax.imshow(drift_matrix, aspect='auto', cmap='YlOrRd')
    
    # Labels
    ax.set_yticks(np.arange(n_findings))
    ax.set_yticklabels(off_target_findings, fontsize=10)
    
    if method_names:
        ax.set_xticks(np.arange(n_methods))
        ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=10)
    else:
        ax.set_xticks(np.arange(n_methods))
        ax.set_xticklabels([f'Traj {i+1}' for i in range(n_methods)], fontsize=9)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Drift (D_k)', fontsize=12)
    
    ax.set_title('Off-Target Drift Heatmap', fontsize=14)
    ax.set_xlabel('Method/Trajectory', fontsize=12)
    ax.set_ylabel('Off-Target Finding', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pareto_frontier(
    results_list: List[BenchmarkResults],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
    highlight_pareto: bool = True,
) -> "plt.Figure":
    """
    Plot progression-drift Pareto frontier.
    
    Args:
        results_list: List of BenchmarkResults for different methods
        figsize: Figure size
        save_path: Optional path to save
        highlight_pareto: Whether to highlight Pareto-optimal points
        
    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract data
    progressions = []
    drifts = []
    names = []
    
    for r in results_list:
        progressions.append(r.progression_aggregate['total_progression']['mean'])
        drifts.append(r.drift_aggregate['aggregate_drift_median']['mean'])
        names.append(r.editor_name)
    
    progressions = np.array(progressions)
    drifts = np.array(drifts)
    
    # Find Pareto frontier
    pareto_mask = np.ones(len(progressions), dtype=bool)
    for i in range(len(progressions)):
        for j in range(len(progressions)):
            if i != j:
                if (progressions[j] >= progressions[i] and 
                    drifts[j] <= drifts[i] and
                    (progressions[j] > progressions[i] or drifts[j] < drifts[i])):
                    pareto_mask[i] = False
                    break
    
    # Color scheme
    colors = plt.cm.tab10(np.linspace(0, 1, len(names)))
    
    # Plot all points
    for i, (prog, drift, name, color) in enumerate(zip(progressions, drifts, names, colors)):
        marker = 's' if pareto_mask[i] and highlight_pareto else 'o'
        size = 150 if pareto_mask[i] and highlight_pareto else 100
        ax.scatter(drift, prog, c=[color], s=size, marker=marker, 
                  label=name, edgecolors='black', linewidths=1)
    
    # Plot Pareto frontier line
    if highlight_pareto and np.sum(pareto_mask) > 1:
        pareto_points = list(zip(drifts[pareto_mask], progressions[pareto_mask]))
        pareto_points.sort()
        pareto_drifts, pareto_progs = zip(*pareto_points)
        ax.plot(pareto_drifts, pareto_progs, 'k--', alpha=0.5, linewidth=2)
    
    ax.set_xlabel('Off-Target Drift (D_off)', fontsize=12)
    ax.set_ylabel('Target Progression', fontsize=12)
    ax.set_title('Progression-Drift Pareto Frontier', fontsize=14)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add arrow indicating optimal direction
    ax.annotate('', xy=(0.1, 0.9), xytext=(0.3, 0.7),
               xycoords='axes fraction',
               arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(0.2, 0.8, 'Optimal', transform=ax.transAxes, fontsize=10, color='green')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_ablation_importance(
    importance_scores: Dict[str, float],
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Plot ablation importance scores (Δ_k).
    
    Args:
        importance_scores: Mapping from component name to importance
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    
    # Sort by importance
    sorted_items = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
    names, values = zip(*sorted_items)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by sign
    colors = ['red' if v > 0 else 'blue' for v in values]
    
    bars = ax.barh(range(len(names)), values, color=colors, alpha=0.7, edgecolor='black')
    
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color='black', linewidth=1)
    
    ax.set_xlabel('Δ_k (Drift Increase when Ablated)', fontsize=12)
    ax.set_title('Ablation Importance Analysis', fontsize=14)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add legend
    red_patch = mpatches.Patch(color='red', alpha=0.7, label='Increases Drift')
    blue_patch = mpatches.Patch(color='blue', alpha=0.7, label='Decreases Drift')
    ax.legend(handles=[red_patch, blue_patch], loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_drift_vs_association(
    drift_values: Dict[str, float],
    association_values: Dict[str, float],
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Plot drift magnitude vs. empirical association.
    
    From Section 6.2: "labels with larger empirical association to the
    target tend to exhibit larger drift under unconstrained editing."
    
    Args:
        drift_values: D_k values per finding
        association_values: Assoc(k) values per finding
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Get paired values
    findings = list(set(drift_values.keys()) & set(association_values.keys()))
    
    drifts = [drift_values[f] for f in findings]
    assocs = [abs(association_values[f]) for f in findings]
    
    # Scatter plot
    ax.scatter(assocs, drifts, s=100, alpha=0.7, edgecolors='black')
    
    # Add labels
    for f, d, a in zip(findings, drifts, assocs):
        ax.annotate(f, (a, d), textcoords='offset points', xytext=(5, 5),
                   fontsize=8, alpha=0.8)
    
    # Linear fit
    if len(drifts) > 2:
        from scipy.stats import linregress
        slope, intercept, r_value, _, _ = linregress(assocs, drifts)
        x_line = np.linspace(min(assocs), max(assocs), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r--', linewidth=2, 
               label=f'Linear fit (R²={r_value**2:.3f})')
    
    ax.set_xlabel('|Assoc(k)| (Empirical Association)', fontsize=12)
    ax.set_ylabel('D_k (Drift)', fontsize=12)
    ax.set_title('Drift vs. Dataset Association', fontsize=14)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_trajectory_grid(
    trajectories: List[EditTrajectory],
    nrows: int = 2,
    ncols: int = 4,
    step_indices: Optional[List[int]] = None,
    figsize: Tuple[int, int] = (16, 8),
    save_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Plot grid of trajectory images.
    
    Args:
        trajectories: List of trajectories (uses first one)
        nrows: Number of rows
        ncols: Number of columns
        step_indices: Which steps to show (default: evenly spaced)
        figsize: Figure size
        save_path: Optional path to save
        
    Returns:
        Matplotlib figure
    """
    _check_matplotlib()
    
    trajectory = trajectories[0]
    
    if step_indices is None:
        total_steps = len(trajectory.steps)
        step_indices = np.linspace(0, total_steps - 1, nrows * ncols, dtype=int)
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()
    
    for ax, step_idx in zip(axes, step_indices):
        step = trajectory.steps[step_idx]
        
        # Get image
        if step.image is not None:
            img = step.image.cpu().numpy()
            if img.ndim == 3:
                img = img.squeeze(0) if img.shape[0] == 1 else img.transpose(1, 2, 0)
            ax.imshow(img, cmap='gray')
        
        target_score = step.coordinates.calibrated_target
        ax.set_title(f'Step {step_idx}\np_eff={target_score:.3f}', fontsize=10)
        ax.axis('off')
    
    plt.suptitle(f'Edit Trajectory: {trajectory.editor_name}', fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
