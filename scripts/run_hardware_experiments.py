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
    python scripts/run_hardware_experiments.py --compare-diws --compare-iid-niid
    
    # run using a specific config file
    python scripts/run_hardware_experiments.py --config conf/hardware_experiments/diws_iid.yaml
    
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
SYNC_FHE_CONTEXT_PLAYBOOK = PROJECT_ROOT / "deployment" / "ansible" / "sync_fhe_context.yml"
RUN_EXPERIMENT_PLAYBOOK = PROJECT_ROOT / "deployment" / "ansible" / "run_experiment.yml"

# available options
STRATEGIES = [
    "diws",
    "diws_fhe",
    "fedavg",
    "fedprox",
    "scaffold",
    "mifa",
    "fedadam",
    "fedyogi",
    "clusteredfl",
    "fdms",
]
PARTITIONERS = {
    "iid": "iid",
    "niid_high": "dirichlet_high",  # alpha=0.1
    "niid_medium": "dirichlet_medium",  # alpha=0.5
    "niid_low": "dirichlet",  # alpha=1.0
}
NIID_LEVELS = {
    "high": "niid_high",
    "medium": "niid_medium",
    "low": "niid_low",
}
SCENARIOS = ["baseline", "node_drop", "node_drop_standard", "node_drop_standard_20clients"]
DATASETS = ["cifar10", "cifar100", "tiny_imagenet"]
MODELS = ["simplecnn", "simplecnn_large", "resnet18", "resnet18_gn"]


class ExperimentConfig:
    """Configuration for a single hardware experiment."""

    def __init__(
        self,
        strategy: Optional[str],
        partitioner: Optional[str],
        dataset: Optional[str] = "cifar10",
        model: Optional[str] = "resnet18",
        scenario: Optional[str] = "baseline",
        num_rounds: int = 50,
        num_clients: int = 20,
        config_path: Optional[str] = None,
        config_name: Optional[str] = None,
        logging_backend: Optional[str] = "wandb",
        wandb_project: Optional[str] = None,
        wandb_run_name: Optional[str] = None,
        wandb_mode: Optional[str] = None,
        wandb_tags: Optional[List[str]] = None,
        **kwargs,
    ):
        self.strategy = strategy
        self.partitioner = partitioner
        self.dataset = dataset
        self.model = model
        self.scenario = scenario
        self.num_rounds = num_rounds
        self.num_clients = num_clients
        self.config_path = config_path
        self.config_name = config_name
        self.logging_backend = logging_backend
        self.wandb_project = wandb_project
        self.wandb_run_name = wandb_run_name
        self.wandb_mode = wandb_mode
        self.wandb_tags = wandb_tags or []
        self.extra_args = kwargs

    @property
    def name(self) -> str:
        """Generate experiment name."""
        parts = []
        if self.config_name:
            parts.append(self.config_name)
        if self.strategy:
            parts.append(self.strategy)
        if self.dataset:
            parts.append(self.dataset)
        if self.model:
            parts.append(self.model)
        if self.scenario:
            parts.append(self.scenario)
        if self.partitioner:
            parts.append(self.partitioner)
        return "_".join(parts) if parts else "experiment"

    def to_hydra_overrides(self) -> str:
        """Convert to Hydra command line overrides."""
        overrides: List[str] = []
        
        # If using a config file, only use config-path/config-name
        if self.config_path or self.config_name:
            if self.config_path:
                overrides.append(f"--config-path {self.config_path}")
            if self.config_name:
                overrides.append(f"--config-name {self.config_name}")
            # Don't add any other overrides - config file has everything
            return " ".join(overrides)
        
        # Otherwise, build overrides from individual parameters
        if self.strategy:
            overrides.append(f"strategy={self.strategy}")
        if self.partitioner:
            overrides.append(f"partitioner={self.partitioner}")
        if self.dataset:
            overrides.append(f"dataset={self.dataset}")
        if self.model:
            overrides.append(f"model={self.model}")
        if self.scenario:
            overrides.append(f"scenario={self.scenario}")
        overrides.extend(
            [
                f"server.num_rounds={self.num_rounds}",
                f"client.num_clients={self.num_clients}",
                "hardware=distributed",
            ]
        )

        if self.logging_backend and self.logging_backend != "none":
            overrides.append(f"logging={self.logging_backend}")
            if self.logging_backend == "wandb":
                if self.wandb_project:
                    overrides.append(
                        f"logging.wandb.project={_format_override_value(self.wandb_project)}"
                    )
                if self.wandb_run_name:
                    overrides.append(
                        f"logging.wandb.run_name={_format_override_value(self.wandb_run_name)}"
                    )
                if self.wandb_mode:
                    overrides.append(f"logging.wandb.mode={self.wandb_mode}")
                if self.wandb_tags:
                    tags = ",".join(_format_override_value(tag) for tag in self.wandb_tags)
                    overrides.append(f"logging.wandb.tags=[{tags}]")

        # add any extra arguments
        for key, value in self.extra_args.items():
            overrides.append(f"{key}={value}")

        return " ".join(overrides)

    def to_ansible_command(self) -> str:
        hydra_overrides = self.to_hydra_overrides()
        ansible_cmd = [
            "ansible-playbook",
            str(RUN_EXPERIMENT_PLAYBOOK),
            "-i",
            str(INVENTORY_PATH),
            "-e",
            "flops_sync_code=false",
            "-e",
            json.dumps({"hydra_overrides": hydra_overrides}),
        ]
        return " ".join(ansible_cmd)


def _format_override_value(value: str) -> str:
    """Format override value to be shell-safe for hydra."""
    if any(ch.isspace() for ch in value) or any(ch in value for ch in ["'", '"']):
        if "'" in value and '"' not in value:
            return f'"{value}"'
        if '"' in value and "'" not in value:
            return f"'{value}'"
        escaped = value.replace('"', '\\"')
        return f'"{escaped}"'
    return value


def _normalize_overrides(values: Optional[List[str]]) -> List[Optional[str]]:
    return values if values else [None]


def _resolve_config_files(config_files: List[str]) -> List[Tuple[str, str]]:
    resolved = []
    for config_file in config_files:
        path = Path(config_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        try:
            conf_root = PROJECT_ROOT / "conf"
            relative_from_conf = path.relative_to(conf_root)
        except ValueError:
            print(
                f"⚠ Config file {path} is outside conf/. "
                "Hydra may not resolve other config groups correctly."
            )
            relative_from_conf = path

        # Hydra: keep config-path at conf/ so all groups are discoverable
        # and set config-name to the path relative to conf/ (without suffix)
        config_name = str(relative_from_conf.with_suffix(""))

        # run_distributed.py runs from src/, so config-path should be ../conf
        config_path = Path("..") / "conf"
        resolved.append((str(config_path), config_name))
    return resolved


def sync_fhe_context_to_devices(inventory_path: Path, dry_run: bool = False) -> bool:
    """Sync FHE context files to all client devices using Ansible.

    This is required before running diws_fhe experiments in distributed mode.
    The server generates the FHE context and this function distributes the
    client_context.pkl to all clients.

    Args:
        inventory_path: Path to Ansible inventory file
        dry_run: If True, just print what would be run

    Returns:
        True if sync successful, False otherwise
    """
    print(f"\n{'='*80}")
    print("Syncing FHE context to all client devices...")
    print(f"{'='*80}")

    cmd = [
        "ansible-playbook",
        str(SYNC_FHE_CONTEXT_PLAYBOOK),
        "-i",
        str(inventory_path),
    ]

    if dry_run:
        print(f"[DRY RUN] Would execute: {' '.join(cmd)}")
        return True

    try:
        subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            capture_output=False,  # let output stream to console
        )
        print("-> FHE context sync completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"-> FHE context sync failed (exit code {e.returncode})")
        return False


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
        "flops_sync_code=false",
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
    config_files: Optional[List[str]] = None,
    logging_backend: Optional[str] = "wandb",
    wandb_project: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_mode: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
    **kwargs,
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
    use_defaults = not config_files
    if strategies is None and use_defaults:
        strategies = ["diws", "diws_fhe"]
    if partitioners is None and use_defaults:
        partitioners = ["iid", "dirichlet_high"]
    if datasets is None and use_defaults:
        datasets = ["cifar10"]
    if models is None and use_defaults:
        models = ["resnet18"]
    if scenarios is None and use_defaults:
        scenarios = ["baseline"]

    configs = []
    resolved_configs = _resolve_config_files(config_files) if config_files else [(None, None)]
    for config_path, config_name in resolved_configs:
        for strategy in _normalize_overrides(strategies):
            for partitioner in _normalize_overrides(partitioners):
                for dataset in _normalize_overrides(datasets):
                    for model in _normalize_overrides(models):
                        for scenario in _normalize_overrides(scenarios):
                            config = ExperimentConfig(
                                strategy=strategy,
                                partitioner=partitioner,
                                dataset=dataset,
                                model=model,
                                scenario=scenario,
                                config_path=config_path,
                                config_name=config_name,
                                logging_backend=logging_backend,
                                wandb_project=wandb_project,
                                wandb_run_name=wandb_run_name,
                                wandb_mode=wandb_mode,
                                wandb_tags=wandb_tags,
                                **kwargs,
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

    # check if any experiment uses diws_fhe and sync FHE context if needed
    uses_fhe = any(c.strategy == "diws_fhe" for c in configs)
    if uses_fhe:
        print("\n ->-> Detected diws_fhe strategy - syncing FHE context to clients...")
        if not sync_fhe_context_to_devices(INVENTORY_PATH, dry_run=dry_run):
            print("\n-> FHE context sync failed. Aborting.")
            return
        if not dry_run:
            print("\nWaiting 3 seconds for FHE context sync to settle...")
            time.sleep(3)

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
    python scripts/run_hardware_experiments.py --strategy diws_fhe --partitioner iid niid_high
    
    # run with different dataset and model
    python scripts/run_hardware_experiments.py --dataset cifar100 --model resnet18_gn
    
    # compare diws vs diws_fhe on iid vs niid (medium) for baseline and node_drop_standard_20clients
    python scripts/run_hardware_experiments.py --compare-diws --compare-iid-niid --compare-scenarios

    # run all node-drop scenarios
    python scripts/run_hardware_experiments.py --node-drop
    
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
        help="Strategies to run (default: diws diws_fhe)",
    )
    parser.add_argument(
        "--partitioner",
        nargs="+",
        choices=list(PARTITIONERS.keys()),
        help="Data partitioners (default: iid niid_high)",
    )
    parser.add_argument(
        "--compare-diws",
        action="store_true",
        help="Run diws and diws_fhe (overrides --strategy)",
    )
    parser.add_argument(
        "--compare-iid-niid",
        action="store_true",
        help="Run iid and niid partitioners (overrides --partitioner)",
    )
    parser.add_argument(
        "--niid-levels",
        nargs="+",
        choices=["high", "medium", "low", "all"],
        help="NIID levels for --compare-iid-niid (default: medium)",
    )
    parser.add_argument(
        "--dataset", nargs="+", choices=DATASETS, help="Datasets to use (default: cifar10)"
    )
    parser.add_argument(
        "--model", nargs="+", choices=MODELS, help="Models to use (default: resnet18)"
    )
    parser.add_argument(
        "--scenario", nargs="+", choices=SCENARIOS, help="Scenarios to run (default: baseline)"
    )
    parser.add_argument(
        "--compare-scenarios",
        action="store_true",
        help="Run baseline and node_drop_standard_20clients (overrides --scenario)",
    )
    parser.add_argument(
        "--node-drop",
        action="store_true",
        help="Run all node-drop scenarios (overrides --scenario/--compare-scenarios)",
    )
    parser.add_argument(
        "--include-node-drop",
        action="store_true",
        help="Include node_drop with --compare-scenarios",
    )
    parser.add_argument(
        "--config",
        "--config-file",
        nargs="+",
        dest="config_files",
        help="Hydra config file(s) to run (e.g., conf/hardware_experiments/diws_iid.yaml)",
    )
    parser.add_argument(
        "--logging",
        choices=["wandb", "offline", "none"],
        default="wandb",
        help="Logging backend override (default: wandb)",
    )
    parser.add_argument("--wandb-project", help="W&B project name override")
    parser.add_argument("--wandb-run-name", help="W&B run name override")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline"],
        help="W&B mode override",
    )
    parser.add_argument("--wandb-tags", nargs="+", help="W&B tags override")
    parser.add_argument(
        "--num-rounds", type=int, default=50, help="Number of training rounds (default: 50)"
    )
    parser.add_argument(
        "--num-clients", type=int, default=20, help="Number of clients (default: 20)"
    )
    parser.add_argument("--no-sync", action="store_true", help="Skip code sync to devices")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--skip-failed",
        action="store_true",
        default=True,
        help="Continue to next experiment on failure (default: True)",
    )
    parser.add_argument("--stop-on-failure", action="store_true", help="Stop on first failure")
    parser.add_argument(
        "--list", action="store_true", help="List all experiments that would run and exit"
    )

    args = parser.parse_args()

    # resolve strategies
    strategies = args.strategy
    if args.compare_diws:
        if args.strategy:
            print("⚠ --compare-diws specified; ignoring --strategy")
        strategies = ["diws", "diws_fhe"]

    # resolve partitioners
    partitioners = None
    if args.compare_iid_niid:
        if args.partitioner:
            print("⚠ --compare-iid-niid specified; ignoring --partitioner")
        levels = args.niid_levels or ["medium"]
        if "all" in levels:
            levels = ["high", "medium", "low"]
        niid_keys = [NIID_LEVELS[level] for level in levels]
        partitioners = ["iid"] + niid_keys
    elif args.partitioner:
        partitioners = args.partitioner

    if partitioners:
        partitioners = [PARTITIONERS[p] for p in partitioners]

    # handle stop-on-failure flag
    skip_failed = not args.stop_on_failure

    # resolve scenarios
    scenarios = args.scenario
    if args.node_drop:
        if args.scenario or args.compare_scenarios:
            print("⚠ --node-drop specified; ignoring --scenario/--compare-scenarios")
        scenarios = [s for s in SCENARIOS if s.startswith("node_drop")]
    elif args.compare_scenarios:
        if args.scenario:
            print("⚠ --compare-scenarios specified; ignoring --scenario")
        scenarios = ["baseline", "node_drop_standard_20clients"]
        if args.include_node_drop and "node_drop" not in scenarios:
            scenarios.append("node_drop")

    # generate experiment configurations
    configs = generate_experiment_configs(
        strategies=strategies,
        partitioners=partitioners,
        datasets=args.dataset,
        models=args.model,
        scenarios=scenarios,
        config_files=args.config_files,
        num_rounds=args.num_rounds,
        num_clients=args.num_clients,
        logging_backend=args.logging,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        wandb_mode=args.wandb_mode,
        wandb_tags=args.wandb_tags,
    )

    if not configs:
        print("No experiments configured.")
        return

    # list mode
    if args.list:
        print(f"Found {len(configs)} experiments:\n")
        for config in configs:
            print(f"  {config.name}")
            print(f"    Hydra overrides: {config.to_hydra_overrides()}")
            print(f"    Ansible command: {config.to_ansible_command()}")
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
