#!/usr/bin/env python3
"""generates experiment configuration files for all combinations of parameters."""

import os
from pathlib import Path

# base path for experiment configs
CONF_DIR = Path(__file__).parent.parent / "conf" / "experiment"

# experiment parameters
DATASETS = {
    "cifar10": {
        "num_clients": 10,
        "node_drop_scenario": "node_drop_rolling_10clients",
        "models": ["simplecnn", "resnet18"],
        "batch_size": 32,
        "local_epochs": 2,
        "learning_rate": 0.01,
    },
    "cifar100": {
        "num_clients": 20,
        "node_drop_scenario": "node_drop_rolling_20clients",
        "models": ["simplecnn", "resnet18"],
        "batch_size": 32,
        "local_epochs": 2,
        "learning_rate": 0.01,
    },
    "tiny_imagenet": {
        "num_clients": 25,
        "node_drop_scenario": "node_drop_rolling_25clients",
        "models": ["simplecnn_large", "resnet18"],
        "batch_size": 64,
        "local_epochs": 3,
        "learning_rate": 0.001,
    },
}

STRATEGIES = ["fedavg", "fedprox", "scaffold", "mifa", "fedadam", "fedyogi", "clusteredfl"]

DISTRIBUTIONS = {
    "iid": {"partitioner": "iid", "alpha": None},
    "niid_medium": {"partitioner": "dirichlet_medium", "alpha": 0.5},
    "niid_high": {"partitioner": "dirichlet_high", "alpha": 0.1},
}

SCENARIOS = {
    "baseline": {"scenario_override": "baseline", "enabled": False},
    "node_drop": {"scenario_override": None, "enabled": True},  # will be set per dataset
}

# strategy-specific config sections
STRATEGY_CONFIGS = {
    "fedavg": "",
    "fedprox": """
strategy:
  proximal_mu: 0.1
""",
    "scaffold": """
strategy:
  server_lr: 1.0
  client_lr: 0.01
""",
    "mifa": """
strategy:
  base_server_lr: 1.0
  client_lr: 0.01
""",
    "fedadam": """
strategy:
  server_lr: 0.01
  beta_1: 0.9
  beta_2: 0.99
""",
    "fedyogi": """
strategy:
  server_lr: 0.01
  beta_1: 0.9
  beta_2: 0.99
""",
    "clusteredfl": """
strategy:
  cosine_similarity_threshold: 0.7
  split_warmup_rounds: 5
""",
}


def generate_config(
    dataset: str,
    strategy: str,
    model: str,
    distribution: str,
    scenario: str,
    dataset_config: dict,
    dist_config: dict,
) -> str:
    """generate a single experiment config yaml content."""
    
    num_clients = dataset_config["num_clients"]
    batch_size = dataset_config["batch_size"]
    local_epochs = dataset_config["local_epochs"]
    learning_rate = dataset_config["learning_rate"]
    
    # determine partitioner
    partitioner = dist_config["partitioner"]
    
    # determine scenario
    if scenario == "node_drop":
        scenario_override = dataset_config["node_drop_scenario"]
    else:
        scenario_override = "baseline"
    
    # wandb run name
    dist_suffix = distribution
    scenario_suffix = scenario
    run_name = f"{dataset}_{strategy}_{model}_{dist_suffix}_{scenario_suffix}"
    
    # build config
    config = f"""# @package _global_
# {dataset.upper()} experiment: {strategy} + {model} + {distribution} + {scenario}
# auto-generated configuration for simulation benchmarking

defaults:
  - override /dataset: {dataset}
  - override /model: {model}
  - override /strategy: {strategy}
  - override /partitioner: {partitioner}
  - override /scenario: {scenario_override}
  - override /logging: wandb
  - override /hardware: simulation

# experiment identification
experiment:
  name: {run_name}
  seed: 42

# server configuration
server:
  num_rounds: 50
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  min_fit_clients: {max(1, num_clients - 4)}
  min_evaluate_clients: {max(1, num_clients - 4)}
  min_available_clients: {num_clients}
  accept_failures: true

# client configuration
client:
  num_clients: {num_clients}
  local_epochs: {local_epochs}
  batch_size: {batch_size}
  learning_rate: {learning_rate}
  optimizer: adam
"""

    # add strategy-specific config
    if STRATEGY_CONFIGS.get(strategy):
        config += STRATEGY_CONFIGS[strategy]
    
    # add partitioner config for dirichlet
    if dist_config["alpha"] is not None:
        config += f"""
partitioner:
  alpha: {dist_config["alpha"]}
"""

    # add training and hardware config
    config += f"""
# training configuration
training:
  device: auto
  use_amp: false

# evaluation
evaluation:
  test_fraction: 0.2
  eval_every_n_rounds: 1

# hardware config - 2 GPUs, 16 CPU cores
hardware:
  ray:
    num_cpus: 16
    num_gpus: 2
    client_resources:
      num_cpus: {max(1, 16 // num_clients)}
      num_gpus: {round(2.0 / num_clients, 2)}

# wandb logging configuration
logging:
  wandb:
    run_name: {run_name}
    tags:
      - auto-simulation
      - {dataset}
      - {strategy}
      - {distribution}
      - {scenario}
"""

    return config


def generate_all_configs():
    """generate all experiment configuration files."""
    
    total_configs = 0
    
    for dataset, dataset_config in DATASETS.items():
        # map dataset name for folder (tiny_imagenet -> tinyimagenet)
        dataset_folder = dataset.replace("_", "")
        
        for scenario in SCENARIOS.keys():
            scenario_dir = CONF_DIR / dataset_folder / scenario
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            for strategy in STRATEGIES:
                for model in dataset_config["models"]:
                    for dist_name, dist_config in DISTRIBUTIONS.items():
                        filename = f"{strategy}_{model}_{dist_name}.yaml"
                        filepath = scenario_dir / filename
                        
                        config_content = generate_config(
                            dataset=dataset,
                            strategy=strategy,
                            model=model,
                            distribution=dist_name,
                            scenario=scenario,
                            dataset_config=dataset_config,
                            dist_config=dist_config,
                        )
                        
                        with open(filepath, "w") as f:
                            f.write(config_content)
                        
                        total_configs += 1
                        print(f"Generated: {filepath.relative_to(CONF_DIR.parent.parent)}")
    
    print(f"\nTotal configs generated: {total_configs}")


if __name__ == "__main__":
    generate_all_configs()

