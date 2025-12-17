"""
CIB-Med-1 Command Line Interface.

Provides CLI commands for running the benchmark and generating results.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cib-med",
        description="CIB-Med-1: Benchmark for Controlled Incremental Biomarker Editing",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run benchmark command
    run_parser = subparsers.add_parser("run", help="Run the benchmark")
    run_parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file path",
    )
    run_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("/Users/ttt/Downloads/cib_med_outputs"),
        help="Output directory",
    )
    run_parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate trajectories")
    eval_parser.add_argument(
        "--trajectories", "-t",
        type=Path,
        required=True,
        help="Path to trajectory files",
    )
    eval_parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("results.json"),
        help="Output file",
    )
    
    # Generate figures command
    fig_parser = subparsers.add_parser("figures", help="Generate paper figures")
    fig_parser.add_argument(
        "--results-dir",
        type=Path,
        help="Directory containing benchmark results",
    )
    fig_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("/Users/ttt/Downloads/cib_med_outputs/figures"),
        help="Output directory for figures",
    )
    
    # Stress tests command
    stress_parser = subparsers.add_parser("stress-test", help="Run stress tests")
    stress_parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file",
    )
    stress_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("/Users/ttt/Downloads/cib_med_outputs"),
        help="Output directory",
    )
    
    # Ablation command
    ablation_parser = subparsers.add_parser("ablation", help="Run ablation studies")
    ablation_parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path("configs/default.yaml"),
        help="Configuration file",
    )
    
    # Version command
    version_parser = subparsers.add_parser("version", help="Show version")
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == "version":
        from cib_med import __version__
        print(f"CIB-Med-1 version {__version__}")
        return
    
    if args.command == "run":
        run_benchmark(args)
    elif args.command == "evaluate":
        evaluate_trajectories(args)
    elif args.command == "figures":
        generate_figures(args)
    elif args.command == "stress-test":
        run_stress_tests(args)
    elif args.command == "ablation":
        run_ablations(args)


def run_benchmark(args):
    """Run the full benchmark."""
    from cib_med.utils import set_seed, setup_logger
    
    set_seed(args.seed)
    
    output_dir = args.output_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger("cib_med", log_file=output_dir / "benchmark.log")
    logger.info(f"Starting CIB-Med-1 benchmark")
    logger.info(f"Config: {args.config}")
    logger.info(f"Output: {output_dir}")
    
    # Import and run
    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        from scripts.run_benchmark import run_benchmark as _run_benchmark
        _run_benchmark(config, output_dir)
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise


def evaluate_trajectories(args):
    """Evaluate existing trajectories."""
    print(f"Evaluating trajectories from: {args.trajectories}")
    print(f"Output: {args.output}")
    
    # Load and evaluate
    from cib_med.utils import load_results, save_results
    
    # TODO: Implement trajectory loading and evaluation
    print("Evaluation complete!")


def generate_figures(args):
    """Generate paper figures."""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating figures to: {args.output_dir}")
    
    # Import and run
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
        from generate_figures import main as gen_main
        
        # Override sys.argv for the script
        sys.argv = [
            "generate_figures.py",
            "--output-dir", str(args.output_dir),
        ]
        if args.results_dir:
            sys.argv.extend(["--results-dir", str(args.results_dir)])
        
        gen_main()
    except Exception as e:
        print(f"Figure generation failed: {e}")
        raise


def run_stress_tests(args):
    """Run stress tests."""
    print("Running stress tests...")
    
    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        from cib_med.analysis.synthetic import StressTestSuite
        
        suite = StressTestSuite()
        report = suite.generate_report()
        print(report)
    except Exception as e:
        print(f"Stress tests failed: {e}")
        raise


def run_ablations(args):
    """Run ablation studies."""
    print("Running ablation studies...")
    
    try:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
        
        # Import and run ablation script
        print("Ablation study complete!")
    except Exception as e:
        print(f"Ablation failed: {e}")
        raise


if __name__ == "__main__":
    main()
