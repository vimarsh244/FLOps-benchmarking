#!/usr/bin/env python3
"""
Run experiments on distributed hardware testbed.

This script runs FL experiments on physical devices (Raspberry Pis, Jetson devices)
using the hardware testbed configured in deployment/ansible/inventory.yml.

The script:
1. Syncs code to all devices using Ansible
2. Runs experiments sequentially with specified configurations
3. Collects results after each experiment

Usage:
    # run all predefined hardware experiments
    python scripts/run_hardware_experiments.py
    
    # run specific experiments
    python scripts/run_hardware_experiments.py --strategy diws --partitioner iid
    
    # run diws and diws_fhe with both iid and niid
    python scripts/run_hardware_experiments.py --strategy diws diws_fhe --partitioner iid dirichlet_high
    
    # dry run to see what would be executed
    python scripts/run_hardware_experiments.py --dry-run
    
    # skip code sync (useful if already synced)
    python scripts/run_hardware_experiments.py --no-sync
"""

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

# project root
PROJECT_ROOT = Path(__file__).parent.parent
INVENTORY_PATH = PROJECT_ROOT / "deployment" / "ansible" / "inventory.yml"
SYNC_CODE_PLAYBOOK = PROJECT_ROOT / "deployment" / "ansible" / "sync_code.yml"
RUN_EXPERIMENT_PLAYBOOK = PROJECT_ROOT / "deployment" / "ansible" / "run_experiment.yml"

# available options
STRATEGIES = ["diws", "diws_fhe", "fedavg", "fedprox", "scaffold", "mifa", "fedadam", "fedyogi", "clusteredfl", "fdms"]
PARTITIONERS = {
    "iid": "iid",
    "niid_high": "dirichlet_high",  # alpha=0.1
    "niid_medium": "dirichlet_medium",  # alpha=0.5
    "niid_low": "dirichlet",  # alpha=1.0
}
SCENARIOS = ["baseline", "node_drop", "node_drop_standard"]
DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
MODELS = ["simplecnn", "simplecnn_large", "resnet18", "resnet18_gn"]


class ExperimentConfig:
    """Configuration for a single hardware experiment."""
    
    def __init__(
        self,
        strategy: str,
        partitioner: str,
        dataset: str = "cifar10",
        model: str = "resnet18",
        scenario: str = "baseline",
        num_rounds: int = 50,
        num_clients: int = 20,
        **kwargs
    ):
        self.strategy = strategy
        self.partitioner = partitioner
        self.dataset = dataset
        self.model = model
        self.scenario = scenario
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.extra_args = kwargs
        
    @property
    def name(self) -> str:
        """Generate experiment name."""
        return f"{self.strategy}_{self.dataset}_{self.model}_{self.scenario}_{self.partitioner}"
    
    def to_hydra_overrides(self) -> str:
        """Convert to Hydra command line overrides."""
        overrides = [
            f"strategy={self.strategy}",
            f"partitioner={self.partitioner}",
            f"dataset={self.dataset}",
            f"model={self.model}",
            f"scenario={self.scenario}",
            f"server.num_rounds={self.num_rounds}",
            f"client.num_clients={self.num_clients}",
            "hardware=distributed",
        ]
        
        # add any extra arguments
        for key, value in self.extra_args.items():
            overrides.append(f"{key}={value}")
        
        return " ".join(overrides)


def sync_code_to_devices(inventory_path: Path, dry_run: bool = False) -> bool:
    """Sync code to all devices using Ansible.
    
    Args:
        inventory_path: Path to Ansible inventory file
        dry_run: If True, just print what would be run
        
    Returns:
        True if sync successful, False otherwise
    """
    print(f"\n{'='*80}")
    print("Syncing code to all devices...")
    print(f"{'='*80}")
    
    cmd = [
        "ansible-playbook",
        str(SYNC_CODE_PLAYBOOK),
        "-i",
        str(inventory_path),
    ]
    
    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,  # let output stream to console
        )
        print("✓ Code sync completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Code sync failed (exit code {e.returncode})")
        return False


def run_hardware_experiment(
    config: ExperimentConfig,
    dry_run: bool = False,
) -> Tuple[bool, float]:
    """Run a single hardware experiment.
    
    Args:
        config: Experiment configuration
        dry_run: If True, just print what would be run
        
    Returns:
        tuple of (success, duration_seconds)
    """
    print(f"\n{'='*80}")
    print(f"Running experiment: {config.name}")
    print(f"{'='*80}")
    print(f"Strategy: {config.strategy}")
    print(f"Partitioner: {config.partitioner}")
    print(f"Dataset: {config.dataset}")
    print(f"Model: {config.model}")
    print(f"Scenario: {config.scenario}")
    print(f"Rounds: {config.num_rounds}")
    print(f"Clients: {config.num_clients}")
    print(f"{'='*80}")
    
    # build command
    hydra_overrides = config.to_hydra_overrides()
    ansible_cmd = [
        "ansible-playbook",
        str(RUN_EXPERIMENT_PLAYBOOK),
        "-i",
        str(INVENTORY_PATH),
        "-e",
        json.dumps({"hydra_overrides": hydra_overrides}),
    ]

    print(f"\nCommand: {' '.join(ansible_cmd)}")
    
    if dry_run:
        print("[DRY RUN] Would execute the above command")
        return True, 0.0
    
    start_time = time.time()
    
    try:
        # run the experiment via ansible
        result = subprocess.run(
            ansible_cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,  # let output stream to console
        )
        duration = time.time() - start_time
        print(f"\n✓ Experiment completed in {duration:.1f}s ({duration/60:.1f} minutes)")
        return True, duration
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"\n✗ Experiment failed after {duration:.1f}s (exit code {e.returncode})")
        return False, duration
    except KeyboardInterrupt:
        duration = time.time() - start_time
        print(f"\n⚠ Experiment interrupted after {duration:.1f}s")
        raise


def generate_experiment_configs(
    strategies: Optional[List[str]] = None,
    partitioners: Optional[List[str]] = None,
    datasets: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    scenarios: Optional[List[str]] = None,
    **kwargs
) -> List[ExperimentConfig]:
    """Generate experiment configurations based on filters.
    
    Args:
        strategies: List of strategies to run (default: diws, diws_fhe)
        partitioners: List of partitioners (default: iid, dirichlet_high)
        datasets: List of datasets (default: cifar10)
        models: List of models (default: resnet18)
        scenarios: List of scenarios (default: baseline)
        **kwargs: Additional arguments to pass to experiments
        
    Returns:
        List of experiment configurations
    """
    # defaults for hardware experiments
    if strategies is None:
        strategies = ["diws", "diws_fhe"]
    if partitioners is None:
        partitioners = ["iid", "dirichlet_high"]
    if datasets is None:
        datasets = ["cifar10"]
    if models is None:
        models = ["resnet18"]
    if scenarios is None:
        scenarios = ["baseline"]
    
    configs = []
    for strategy in strategies:
        for partitioner in partitioners:
            for dataset in datasets:
                for model in models:
                    for scenario in scenarios:
                        config = ExperimentConfig(
                            strategy=strategy,
                            partitioner=partitioner,
                            dataset=dataset,
                            model=model,
                            scenario=scenario,
                            **kwargs
                        )
                        configs.append(config)
    
    return configs


def run_all_experiments(
    configs: List[ExperimentConfig],
    sync_code: bool = True,
    dry_run: bool = False,
    skip_failed: bool = True,
) -> None:
    """Run all hardware experiments sequentially.
    
    Args:
        configs: List of experiment configurations to run
        sync_code: Whether to sync code before running experiments
        dry_run: If True, just print what would be run
        skip_failed: If True, continue to next experiment on failure
    """
    print(f"\n{'#'*80}")
    print(f"# Hardware Experiment Runner")
    print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total experiments: {len(configs)}")
    print(f"# Inventory: {INVENTORY_PATH}")
    print(f"{'#'*80}")
    
    # sync code to devices
    if sync_code:
        if not sync_code_to_devices(INVENTORY_PATH, dry_run=dry_run):
            print("\n✗ Code sync failed. Aborting.")
            return
        # wait a bit for sync to settle
        if not dry_run:
            print("\nWaiting 5 seconds for sync to settle...")
            time.sleep(5)
    else:
        print("\n⚠ Skipping code sync (--no-sync specified)")
    
    successful = 0
    failed = 0
    total_duration = 0.0
    failed_experiments = []
    
    try:
        for i, config in enumerate(configs, 1):
            print(f"\n[{i}/{len(configs)}] {config.name}")
            
            success, duration = run_hardware_experiment(config, dry_run=dry_run)
            total_duration += duration
            
            if success:
                successful += 1
            else:
                failed += 1
                failed_experiments.append(config.name)
                if not skip_failed:
                    print("\n✗ Stopping due to failure (use --skip-failed to continue)")
                    break
            
            # wait between experiments to let things settle
            if not dry_run and i < len(configs):
                print("\nWaiting 10 seconds before next experiment...")
                time.sleep(10)
                
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user.")
    
    # print summary
    print(f"\n{'#'*80}")
    print(f"# Experiment Run Summary")
    print(f"# Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"# Total duration: {total_duration/60:.1f} minutes ({total_duration/3600:.1f} hours)")
    print(f"# Successful: {successful}")
    print(f"# Failed: {failed}")
    print(f"# Remaining: {len(configs) - successful - failed}")
    print(f"{'#'*80}")
    
    if failed_experiments:
        print("\n✗ Failed experiments:")
        for exp in failed_experiments:
            print(f"  - {exp}")


def main():
    parser = argparse.ArgumentParser(
        description="Run FL experiments on distributed hardware testbed.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # run default experiments (DIWS + DIWS-FHE with IID + NIID)
    python scripts/run_hardware_experiments.py
    
    # run only DIWS with IID
    python scripts/run_hardware_experiments.py --strategy diws --partitioner iid
    
    # run DIWS-FHE with both IID and NIID (high skew)
    python scripts/run_hardware_experiments.py --strategy diws_fhe --partitioner iid dirichlet_high
    
    # run with different dataset and model
    python scripts/run_hardware_experiments.py --dataset cifar100 --model resnet18_gn
    
    # dry run to see what would be executed
    python scripts/run_hardware_experiments.py --dry-run
    
    # skip code sync (useful if already synced recently)
    python scripts/run_hardware_experiments.py --no-sync
    
    # run with more rounds
    python scripts/run_hardware_experiments.py --num-rounds 100
    
    # list all experiments that would run
    python scripts/run_hardware_experiments.py --list
        """,
    )
    
    parser.add_argument(
        "--strategy",
        nargs="+",
        choices=STRATEGIES,
        help="Strategies to run (default: diws diws_fhe)"
    )
    parser.add_argument(
        "--partitioner",
        nargs="+",
        choices=list(PARTITIONERS.keys()),
        help="Data partitioners (default: iid niid_high)"
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        choices=DATASETS,
        help="Datasets to use (default: cifar10)"
    )
    parser.add_argument(
        "--model",
        nargs="+",
        choices=MODELS,
        help="Models to use (default: resnet18)"
    )
    parser.add_argument(
        "--scenario",
        nargs="+",
        choices=SCENARIOS,
        help="Scenarios to run (default: baseline)"
    )
    parser.add_argument(
        "--num-rounds",
        type=int,
        default=50,
        help="Number of training rounds (default: 50)"
    )
    parser.add_argument(
        "--num-clients",
        type=int,
        default=20,
        help="Number of clients (default: 20)"
    )
    parser.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip code sync to devices"
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
        help="List all experiments that would run and exit"
    )
    
    args = parser.parse_args()
    
    # convert partitioner names
    partitioners = None
    if args.partitioner:
        partitioners = [PARTITIONERS[p] for p in args.partitioner]
    
    # handle stop-on-failure flag
    skip_failed = not args.stop_on_failure
    
    # generate experiment configurations
    configs = generate_experiment_configs(
        strategies=args.strategy,
        partitioners=partitioners,
        datasets=args.dataset,
        models=args.model,
        scenarios=args.scenario,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
    )
    
    if not configs:
        print("No experiments configured.")
        return
    
    # list mode
    if args.list:
        print(f"Found {len(configs)} experiments:\n")
        for config in configs:
            print(f"  {config.name}")
            print(f"    Command: python -m src.main {config.to_hydra_overrides()}")
            print()
        return
    
    # run experiments
    run_all_experiments(
        configs=configs,
        sync_code=not args.no_sync,
        dry_run=args.dry_run,
        skip_failed=skip_failed,
    )


if __name__ == "__main__":
    main()
