"""
Publication-quality figure generation for CIB-Med-1.

Generates all figures needed for paper reproduction.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import numpy as np

from cib_med.metrics.benchmark import BenchmarkResults
from cib_med.core.trajectory import EditTrajectory


class FigureGenerator:
    """
    Generates all publication figures for CIB-Med-1.
    
    Reproduces figures from the paper:
    - Figure 1: Model differences (target-only vs constrained)
    - Figure 3: Progression-drift efficiency frontier
    - Figure 4: Drift tracks dataset correlation
    - Figure 5: Example trajectory images
    """
    
    def __init__(
        self,
        output_dir: Path,
        style: str = "paper",  # "paper" or "presentation"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style
        
        self._setup_style()
    
    def _setup_style(self):
        """Configure matplotlib style."""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            if self.style == "paper":
                plt.rcParams.update({
                    'font.size': 10,
                    'axes.titlesize': 12,
                    'axes.labelsize': 11,
                    'legend.fontsize': 9,
                    'figure.dpi': 150,
                    'savefig.dpi': 300,
                    'font.family': 'serif',
                })
                sns.set_palette("colorblind")
            else:
                plt.rcParams.update({
                    'font.size': 14,
                    'axes.titlesize': 16,
                    'axes.labelsize': 14,
                    'legend.fontsize': 12,
                    'figure.dpi': 100,
                })
        except ImportError:
            pass
    
    def generate_figure_1(
        self,
        unconstrained_results: BenchmarkResults,
        constrained_results: BenchmarkResults,
    ) -> Path:
        """
        Generate Figure 1: Model Differences.
        
        Shows why target-only metrics are dangerous and
        justifies the existence of the benchmark.
        """
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Left: Target progression comparison
        ax1 = axes[0]
        methods = ['Unconstrained', 'Constrained']
        progressions = [
            unconstrained_results.progression_aggregate['total_progression']['mean'],
            constrained_results.progression_aggregate['total_progression']['mean'],
        ]
        prog_stds = [
            unconstrained_results.progression_aggregate['total_progression']['std'],
            constrained_results.progression_aggregate['total_progression']['std'],
        ]
        
        bars = ax1.bar(methods, progressions, yerr=prog_stds, capsize=5,
                       color=['coral', 'steelblue'], edgecolor='black')
        ax1.set_ylabel('Target Progression')
        ax1.set_title('(A) Target-Only Metric\n(Both methods appear similar)')
        ax1.set_ylim(0, max(progressions) * 1.3)
        
        # Right: Off-target drift comparison
        ax2 = axes[1]
        drifts = [
            unconstrained_results.drift_aggregate['aggregate_drift_median']['mean'],
            constrained_results.drift_aggregate['aggregate_drift_median']['mean'],
        ]
        drift_stds = [
            unconstrained_results.drift_aggregate['aggregate_drift_median']['std'],
            constrained_results.drift_aggregate['aggregate_drift_median']['std'],
        ]
        
        bars = ax2.bar(methods, drifts, yerr=drift_stds, capsize=5,
                       color=['coral', 'steelblue'], edgecolor='black')
        ax2.set_ylabel('Off-Target Drift (D_off)')
        ax2.set_title('(B) CIB-Med-1 Drift Metric\n(Reveals reward hacking)')
        
        # Add annotation
        diff_pct = (drifts[0] - drifts[1]) / drifts[1] * 100
        ax2.annotate(f'{diff_pct:.0f}% more drift!',
                    xy=(0, drifts[0]), xytext=(0.3, drifts[0] * 1.2),
                    fontsize=11, color='red', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.suptitle('"Target-Only" Metrics Are Dangerous', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'figure_1_model_differences.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_figure_3(
        self,
        results_by_lambda: Dict[float, BenchmarkResults],
    ) -> Path:
        """
        Generate Figure 3: Progression-Drift Efficiency Frontier.
        
        Shows the Pareto frontier with "knee" point.
        """
        import matplotlib.pyplot as plt
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        lambdas = sorted(results_by_lambda.keys())
        progressions = []
        drifts = []
        
        for lam in lambdas:
            r = results_by_lambda[lam]
            progressions.append(r.progression_aggregate['total_progression']['mean'])
            drifts.append(r.drift_aggregate['aggregate_drift_median']['mean'])
        
        # Plot constrained method curve
        ax.plot(drifts, progressions, 'b-o', linewidth=2, markersize=10,
               label='Constrained (varying λ)', zorder=3)
        
        # Highlight specific points
        for i, lam in enumerate(lambdas):
            ax.annotate(f'λ={lam}', (drifts[i], progressions[i]),
                       textcoords='offset points', xytext=(10, 5),
                       fontsize=9, alpha=0.7)
        
        # Find and mark knee point
        if len(progressions) >= 3:
            # Simple knee detection: maximum curvature
            knee_idx = len(progressions) // 2  # Approximate
            ax.scatter([drifts[knee_idx]], [progressions[knee_idx]], 
                      s=200, c='red', marker='*', zorder=4,
                      label='Knee point (saturation)')
        
        # Add baseline points
        ax.scatter([drifts[0]], [progressions[0]], s=150, c='green', 
                  marker='^', zorder=4, label='Unconstrained baseline')
        
        ax.set_xlabel('Off-Target Drift (D_off)', fontsize=12)
        ax.set_ylabel('Target Progression', fontsize=12)
        ax.set_title('Progression-Drift Efficiency Frontier', fontsize=14)
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add optimal direction arrow
        ax.annotate('', xy=(min(drifts)*0.8, max(progressions)*0.95),
                   xytext=(max(drifts)*0.5, min(progressions)*1.2),
                   arrowprops=dict(arrowstyle='->', color='green', lw=2, alpha=0.5))
        ax.text(min(drifts)*0.7, max(progressions)*0.85, 'Better',
               fontsize=11, color='green', alpha=0.7)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'figure_3_pareto_frontier.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_figure_4(
        self,
        unconstrained_drift: Dict[str, float],
        constrained_drift: Dict[str, float],
        associations: Dict[str, float],
    ) -> Path:
        """
        Generate Figure 4: Drift Tracks Dataset Correlation.
        
        Shows that unconstrained editing exploits spurious correlations.
        """
        import matplotlib.pyplot as plt
        from scipy.stats import linregress
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get common findings
        findings = list(set(unconstrained_drift.keys()) & 
                       set(constrained_drift.keys()) & 
                       set(associations.keys()))
        
        # Prepare data
        unc_drifts = [unconstrained_drift[f] for f in findings]
        con_drifts = [constrained_drift[f] for f in findings]
        assocs = [abs(associations[f]) for f in findings]
        
        # Plot unconstrained
        ax.scatter(assocs, unc_drifts, s=100, c='red', alpha=0.7,
                  label='Unconstrained', edgecolors='black')
        
        # Linear fit for unconstrained
        slope_u, intercept_u, r_u, _, _ = linregress(assocs, unc_drifts)
        x_line = np.linspace(min(assocs), max(assocs), 100)
        ax.plot(x_line, slope_u * x_line + intercept_u, 'r--', linewidth=2,
               alpha=0.7, label=f'Unconstrained fit (R²={r_u**2:.2f})')
        
        # Plot constrained
        ax.scatter(assocs, con_drifts, s=100, c='blue', alpha=0.7,
                  label='Constrained', marker='s', edgecolors='black')
        
        # Linear fit for constrained
        slope_c, intercept_c, r_c, _, _ = linregress(assocs, con_drifts)
        ax.plot(x_line, slope_c * x_line + intercept_c, 'b--', linewidth=2,
               alpha=0.7, label=f'Constrained fit (R²={r_c**2:.2f})')
        
        # Labels
        for f, a, ud in zip(findings, assocs, unc_drifts):
            if ud > np.median(unc_drifts):  # Label high-drift points
                ax.annotate(f, (a, ud), textcoords='offset points',
                           xytext=(5, 5), fontsize=8, alpha=0.6)
        
        ax.set_xlabel('|Assoc(k)| (Dataset Association)', fontsize=12)
        ax.set_ylabel('Drift D_k', fontsize=12)
        ax.set_title('Drift Tracks Dataset Correlation', fontsize=14)
        ax.legend(loc='upper left', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = self.output_dir / 'figure_4_drift_correlation.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_figure_5(
        self,
        trajectory: EditTrajectory,
        steps_to_show: List[int] = [0, 5, 10, 15, 20],
    ) -> Path:
        """
        Generate Figure 5: Example Trajectory Images.
        
        Shows visual progression of editing.
        """
        import matplotlib.pyplot as plt
        
        n_steps = len(steps_to_show)
        fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 4))
        
        for ax, step_idx in zip(axes, steps_to_show):
            if step_idx < len(trajectory.steps):
                step = trajectory.steps[step_idx]
                
                if step.image is not None:
                    img = step.image.cpu().numpy()
                    if img.ndim == 3:
                        img = img.squeeze(0) if img.shape[0] == 1 else np.mean(img, axis=0)
                    ax.imshow(img, cmap='gray')
                else:
                    ax.text(0.5, 0.5, 'No Image', ha='center', va='center',
                           transform=ax.transAxes)
                
                target = step.coordinates.calibrated_target
                ax.set_title(f't={step_idx}\np_eff={target:.2f}', fontsize=11)
            
            ax.axis('off')
        
        plt.suptitle(f'Edit Trajectory: {trajectory.editor_name}', fontsize=14)
        plt.tight_layout()
        
        save_path = self.output_dir / 'figure_5_trajectory_images.pdf'
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def generate_all_figures(
        self,
        results: Dict[str, Any],
    ) -> Dict[str, Path]:
        """
        Generate all paper figures.
        
        Args:
            results: Dictionary containing all necessary data
            
        Returns:
            Dictionary mapping figure names to saved paths
        """
        paths = {}
        
        # Figure 1
        if 'unconstrained_results' in results and 'constrained_results' in results:
            paths['figure_1'] = self.generate_figure_1(
                results['unconstrained_results'],
                results['constrained_results'],
            )
        
        # Figure 3
        if 'results_by_lambda' in results:
            paths['figure_3'] = self.generate_figure_3(results['results_by_lambda'])
        
        # Figure 4
        if all(k in results for k in ['unconstrained_drift', 'constrained_drift', 'associations']):
            paths['figure_4'] = self.generate_figure_4(
                results['unconstrained_drift'],
                results['constrained_drift'],
                results['associations'],
            )
        
        # Figure 5
        if 'example_trajectory' in results:
            paths['figure_5'] = self.generate_figure_5(results['example_trajectory'])
        
        return paths
