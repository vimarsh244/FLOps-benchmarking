#!/usr/bin/env python3
"""
Run all auto-generated experiment configurations sequentially.

This script runs all experiments from the experiment subfolders (cifar10, cifar100, tinyimagenet)
with proper WandB logging and the [auto-simulation] tag.

Uses conda environment 'flops' for running experiments.

Usage:
    python scripts/run_all_experiments.py
    python scripts/run_all_experiments.py --dataset cifar10
    python scripts/run_all_experiments.py --dataset cifar10 --scenario baseline
    python scripts/run_all_experiments.py --strategy fedavg
    python scripts/run_all_experiments.py --dry-run
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# project root
PROJECT_ROOT = Path(__file__).parent.parent
CONF_DIR = PROJECT_ROOT / "conf" / "experiment"

# conda environment name
CONDA_ENV = "flops"

# experiment order (for organizing runs)
DATASETS = ["cifar10", "cifar100", "tinyimagenet"]
SCENARIOS = ["baseline", "node_drop"]
STRATEGIES = ["fedavg", "fedprox", "scaffold", "mifa", "fedadam", "fedyogi", "clusteredfl"]
MODELS = ["simplecnn", "simplecnn_large", "resnet18"]
DISTRIBUTIONS = ["iid", "niid_medium", "niid_high"]


def get_experiment_configs(
    dataset: Optional[str] = None,
    scenario: Optional[str] = None,
    strategy: Optional[str] = None,
    model: Optional[str] = None,
    distribution: Optional[str] = None,
) -> list[Path]:
    """Get all experiment config files matching the given filters.
    
    Args:
        dataset: filter by dataset (cifar10, cifar100, tinyimagenet)
        scenario: filter by scenario (baseline, node_drop)
        strategy: filter by strategy (fedavg, fedprox, etc.)
        model: filter by model (simplecnn, resnet18, etc.)
        distribution: filter by distribution (iid, niid_medium, niid_high)
    
    Returns:
        list of paths to experiment config files
    """
    configs = []
    
    # filter datasets
    datasets = [dataset] if dataset else DATASETS
    
    # filter scenarios
    scenarios = [scenario] if scenario else SCENARIOS
    
    for ds in datasets:
        for sc in scenarios:
            config_dir = CONF_DIR / ds / sc
            if not config_dir.exists():
                continue
            
            for config_file in sorted(config_dir.glob("*.yaml")):
                # parse filename to extract components
                name = config_file.stem  # e.g., "fedavg_simplecnn_iid"
                parts = name.split("_")
                
                # extract strategy (first part)
                cfg_strategy = parts[0]
                
                # extract model and distribution
                # handle simplecnn_large case
                if len(parts) >= 4 and parts[1] == "simplecnn" and parts[2] == "large":
                    cfg_model = "simplecnn_large"
                    cfg_distribution = "_".join(parts[3:])
                else:
                    cfg_model = parts[1]
                    cfg_distribution = "_".join(parts[2:])
                
                # apply filters
                if strategy and cfg_strategy != strategy:
                    continue
                if model and cfg_model != model:
                    continue
                if distribution and cfg_distribution != distribution:
                    continue
                
                configs.append(config_file)
    
    return configs


def get_config_relative_path(config_path: Path) -> str:
    """Get the relative path from experiment directory for Hydra override.
    
    Args:
        config_path: absolute path to config file
    
    Returns:
        relative path for Hydra override (e.g., "cifar10/baseline/fedavg_simplecnn_iid")
    """
    # get path relative to conf/experiment
    rel_path = config_path.relative_to(CONF_DIR)
    # remove .yaml extension
    return str(rel_path.with_suffix(""))


def run_experiment(config_path: Path, dry_run: bool = False) -> tuple[bool, float]:
    """Run a single experiment.
    
    Args:
        config_path: path to the experiment config file
        dry_run: if True, just print what would be run
    
    Returns:
        tuple of (success, duration_seconds)
    """
    rel_path = get_config_relative_path(config_path)
    
    # build command with conda activation
    python_cmd = f"python -m src.main +experiment={rel_path}"
    
    # wrap in bash with conda activate
    bash_cmd = f"source $(conda info --base)/etc/profile.d/conda.sh && conda activate {CONDA_ENV} && cd {PROJECT_ROOT} && {python_cmd}"
    
    print(f"\n{'='*80}")
    print(f"Running: {rel_path}")
    print(f"Environment: conda activate {CONDA_ENV}")
    print(f"Command: {python_cmd}")
    print(f"{'='*80}")
    
    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True, 0.0
    
    start_time = time.time()
    
    try:
        # run the experiment via bash with conda
        result = subprocess.run(
            bash_cmd,
            shell=True,
            executable="/bin/bash",
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,  # let output stream to console
        )
        duration = time.time() - start_time
        print(f"\n✓ Completed in {duration:.1f}s")
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n✗ Failed after {duration:.1f}s (exit code {e.returncode})")
        return False, duration
    except KeyboardInterrupt:
        duration = time.time() - start_time
        print(f"\n⚠ Interrupted after {duration:.1f}s")
        raise


def run_all_experiments(
    dataset: Optional[str] = None,
    scenario: Optional[str] = None,
    strategy: Optional[str] = None,
    model: Optional[str] = None,
    distribution: Optional[str] = None,
    dry_run: bool = False,
    skip_failed: bool = True,
) -> None:
    """Run all experiments matching the given filters.
    
    Args:
        dataset: filter by dataset
        scenario: filter by scenario
        strategy: filter by strategy
        model: filter by model
        distribution: filter by distribution
        dry_run: if True, just print what would be run
        skip_failed: if True, continue to next experiment on failure
    """
    configs = get_experiment_configs(
        dataset=dataset,
        scenario=scenario,
        strategy=strategy,
        model=model,
        distribution=distribution,
    )
    
    if not configs:
        print("No experiment configs found matching the given filters.")
        return
    
    print(f"\n{'#'*80}")
    print(f"# FLOps Auto-Simulation Experiment Runner")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total experiments: {len(configs)}")
    if dataset:
        print(f"# Dataset filter: {dataset}")
    if scenario:
        print(f"# Scenario filter: {scenario}")
    if strategy:
        print(f"# Strategy filter: {strategy}")
    if model:
        print(f"# Model filter: {model}")
    if distribution:
        print(f"# Distribution filter: {distribution}")
    print(f"{'#'*80}")
    
    successful = 0
    failed = 0
    skipped = 0
    total_duration = 0.0
    failed_experiments = []
    
    try:
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}]", end="")
            
            success, duration = run_experiment(config, dry_run=dry_run)
            total_duration += duration
            
            if success:
                successful += 1
            else:
                failed += 1
                failed_experiments.append(get_config_relative_path(config))
                if not skip_failed:
                    print("\nStopping due to failure (use --skip-failed to continue)")
                    break
                    
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    
    # print summary
    print(f"\n{'#'*80}")
    print(f"# Experiment Run Summary")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total duration: {total_duration/60:.1f} minutes")
    print(f"# Successful: {successful}")
    print(f"# Failed: {failed}")
    print(f"# Remaining: {len(configs) - successful - failed}")
    print(f"{'#'*80}")
    
    if failed_experiments:
        print("\nFailed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")


def main():
    parser = argparse.ArgumentParser(
        description="Run all auto-generated FL simulation experiments.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # run all experiments (uses conda env 'flops' automatically)
    python scripts/run_all_experiments.py
    
    # run only cifar10 experiments
    python scripts/run_all_experiments.py --dataset cifar10
    
    # run only baseline scenarios
    python scripts/run_all_experiments.py --scenario baseline
    
    # run only fedavg experiments
    python scripts/run_all_experiments.py --strategy fedavg
    
    # run only resnet18 experiments on cifar10
    python scripts/run_all_experiments.py --dataset cifar10 --model resnet18
    
    # dry run to see what would be executed
    python scripts/run_all_experiments.py --dry-run
    
    # list all available experiments
    python scripts/run_all_experiments.py --list
    
Note: Each experiment runs in the 'flops' conda environment.
        """
    )
    
    parser.add_argument(
        "--dataset",
        choices=DATASETS,
        help="Filter by dataset"
    )
    parser.add_argument(
        "--scenario",
        choices=SCENARIOS,
        help="Filter by scenario (baseline or node_drop)"
    )
    parser.add_argument(
        "--strategy",
        choices=STRATEGIES,
        help="Filter by aggregation strategy"
    )
    parser.add_argument(
        "--model",
        choices=MODELS,
        help="Filter by model architecture"
    )
    parser.add_argument(
        "--distribution",
        choices=DISTRIBUTIONS,
        help="Filter by data distribution"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing"
    )
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        default=True,
        help="Continue to next experiment on failure (default: True)"
    )
    parser.add_argument(
        "--stop-on-failure",
        action="store_true",
        help="Stop on first failure"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all experiments matching filters and exit"
    )
    
    args = parser.parse_args()
    
    # handle stop-on-failure flag
    skip_failed = not args.stop_on_failure
    
    # list mode
    if args.list:
        configs = get_experiment_configs(
            dataset=args.dataset,
            scenario=args.scenario,
            strategy=args.strategy,
            model=args.model,
            distribution=args.distribution,
        )
        print(f"Found {len(configs)} experiments:\n")
        for config in configs:
            print(f"  {get_config_relative_path(config)}")
        return
    
    # run experiments
    run_all_experiments(
        dataset=args.dataset,
        scenario=args.scenario,
        strategy=args.strategy,
        model=args.model,
        distribution=args.distribution,
        dry_run=args.dry_run,
        skip_failed=skip_failed,
    )


if __name__ == "__main__":
    main()

