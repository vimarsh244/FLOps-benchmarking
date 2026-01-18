# FLOps Benchmarking

A FL benchmarking framework built with Flower and PyTorch. Supports multiple FL strategies, datasets, models, and scenarios and using hydra for configs.

## Features

- **Multiple FL Strategies**: FedAvg, FedProx, MIFA, ClusteredFL, SCAFFOLD, FedAdam, FedYogi
- **Multiple Datasets**: CIFAR-10, CIFAR-100, Tiny-ImageNet
- **Multiple Models**: SimpleCNN, ResNet18/34/50, Vision Transformer (ViT)
- **Data Partitioning**: IID and Non-IID (Dirichlet) partitioning
- **Scenarios**: Baseline, Node Drop (configurable disconnect/rejoin), Timeout handling
- **Execution Modes**: Simulation (Flower/Ray) and Distributed (physical devices)
- **Logging**: Weights & Biases integration + offline JSON/CSV logging
- **Monitoring**: CPU, GPU, memory, network, and power consumption metrics
- **Visualization**: Automatic and manual plotting scripts
- **Hardware Deployment**: Ansible playbooks for Raspberry Pi and Jetson devices

## Quick Start

### Installation

```bash
# clone the repository
cd FLOps-benchmarking

# create virtual environment
# recommended to use conda
conda create -n flops python=3.10
conda activate flops

# can use venv
# python -m venv .venv
# source .venv/bin/activate  # Linux/Mac

# install dependencies
pip install -r requirements.txt
```

### Authenticate W&B with `.env`

1. Copy `env.example` to `.env` and paste your secrets from the [W&B settings page](https://wandb.ai/authorize):
   ```bash
   cp env.example .env
   ```
2. Fill in `WANDB_API_KEY`, `WANDB_ENTITY`, `WANDB_PROJECT`, and optionally tweak `WANDB_MODE`. The W&B backend treats the `WANDB_API_KEY` environment variable as a non-interactive login token (`wandb login` automatically consumes it as documented in [wandb.login](https://docs.wandb.ai/ref/python/login)).
3. Run experiments as usual. `python-dotenv` is loaded at startup, so `src.main` and `src.run_distributed` always expose the values to W&B.
4. If you need to refresh the local CLI cache, you can still run `wandb login --relogin "$WANDB_API_KEY"` and it will reuse the same environment variable instead of prompting.

Keep your real `.env` out of version control (already covered by `.gitignore`).

### Run a Simulation

```bash
# run with default configuration (FedAvg, CIFAR-10, ResNet18)
python -m src.main

# run with different strategy
python -m src.main strategy=fedprox

# run with different dataset and model
python -m src.main dataset=cifar100 model=simplecnn

# run with non-IID data
python -m src.main partitioner=dirichlet partitioner.alpha=0.5

# run with node drop scenario
python -m src.main scenario=node_drop
```

## Configuration

All configuration is managed through Hydra. The main configuration groups are:

| Group | Options | Description |
|-------|---------|-------------|
| `strategy` | fedavg, fedprox, mifa, clusteredfl, scaffold, fedadam, fedyogi | FL aggregation strategy |
| `dataset` | cifar10, cifar100, tiny_imagenet | Training dataset |
| `model` | simplecnn, resnet18, resnet34, resnet50, vit | Model architecture |
| `scenario` | baseline, node_drop, timeout | Experiment scenario |
| `partitioner` | iid, dirichlet | Data partitioning method |
| `hardware` | simulation, distributed | Execution mode |
| `logging` | wandb, offline | Logging backend |

### Example Configurations

```bash
# fedprox with high proximal term
python -m src.main strategy=fedprox strategy.proximal_mu=0.5

# non-iid with alpha=0.1 (very heterogeneous)
python -m src.main partitioner=dirichlet partitioner.alpha=0.1

# more clients and rounds
python -m src.main client.num_clients=20 server.num_rounds=100

# node drop scenario with custom timing
python -m src.main scenario=node_drop \
  'scenario.drop_events=[{client_ids:[2,3],disconnect_round:10,rejoin_round:40}]'

# enable W&B logging
python -m src.main logging=wandb logging.wandb.project=my-project
```

### Configuration Files

Configuration files are in `conf/`:

```
conf/
├── config.yaml          # main config
├── strategy/            # strategy configs
├── dataset/             # dataset configs
├── model/               # model configs
├── scenario/            # scenario configs
├── partitioner/         # partitioning configs
├── hardware/            # simulation/distributed configs
└── logging/             # logging configs
```

## Strategies

### FedAvg
Standard Federated Averaging. Clients train locally and server averages their updates.

```bash
python -m src.main strategy=fedavg
```

### FedProx
FedAvg with a proximal term to handle heterogeneity.

```bash
python -m src.main strategy=fedprox strategy.proximal_mu=0.1
```

### MIFA
Handles device unavailability by maintaining per-client update tables.

```bash
python -m src.main strategy=mifa strategy.base_server_lr=0.1
```

### ClusteredFL
Dynamically clusters clients based on update similarity.

```bash
python -m src.main strategy=clusteredfl strategy.split_warmup_rounds=5
```

### SCAFFOLD
Uses control variates to reduce client drift.

```bash
python -m src.main strategy=scaffold strategy.server_lr=1.0
```

### DIWS
Substitutes dropped client updates based on label distribution.

```bash
python -m src.main strategy=diws
```

### FedOpt (FedAdam, FedYogi)
Adaptive server-side optimization.

```bash
python -m src.main strategy=fedadam strategy.server_lr=0.01
python -m src.main strategy=fedyogi strategy.server_lr=0.01
```

## Scenarios

### Baseline
All clients participate in every round.

```bash
python -m src.main scenario=baseline
```

### Node Drop
Clients disconnect and rejoin at specified rounds.

```bash
# use default drop events
python -m src.main scenario=node_drop

# custom drop events
python -m src.main scenario=node_drop \
  'scenario.drop_events=[{client_ids:[2],disconnect_round:5,rejoin_round:30}]'
```

### Timeout
Simulates straggler clients with configurable delays.

```bash
python -m src.main scenario=timeout scenario.timeout_seconds=30 \
  scenario.simulate_stragglers.straggler_probability=0.2
```

## Hardware Deployment

For running on physical devices (Raspberry Pi, Jetson, etc.):

### 1. Setup Inventory

Edit `deployment/ansible/inventory.yml` with your device information:

```yaml
raspberry_pi:
  hosts:
    rpi5-1:
      ansible_host: 192.168.1.101
      ansible_user: pi
      partition_id: 0
```

### 2. Setup Devices

```bash
# setup python environment on all devices
ansible-playbook deployment/ansible/setup_environment.yml -i deployment/ansible/inventory.yml
```

### 3. Sync Code

```bash
# sync code to all devices
ansible-playbook deployment/ansible/sync_code.yml -i deployment/ansible/inventory.yml
```

### 4. Run Experiment

```bash
# run distributed experiment
ansible-playbook deployment/ansible/run_experiment.yml -i deployment/ansible/inventory.yml
```

### 5. Collect Results

```bash
# collect results from all devices
ansible-playbook deployment/ansible/collect_results.yml -i deployment/ansible/inventory.yml \
  -e experiment_name=my_experiment
```

## Visualization

### Automatic Plotting

After an experiment, plots are automatically generated in the `plots/` subdirectory.

```bash
# manually generate plots for an experiment
python scripts/plot.py outputs/fedavg_cifar10_baseline/2024-01-01_12-00-00/
```

### Compare Experiments

```bash
# compare multiple experiments
python scripts/compare_experiments.py \
  outputs/fedavg_cifar10_baseline/2024-01-01_12-00-00/ \
  outputs/fedprox_cifar10_baseline/2024-01-01_12-00-00/ \
  -o comparison_report/
```

## Monitoring

System metrics are automatically collected during experiments:

- **CPU**: Usage percentage, per-core usage
- **Memory**: Usage percentage, used/available GB
- **Network**: Bytes sent/received
- **GPU**: Utilization, memory, temperature (NVIDIA/Jetson)
- **Power**: Total power consumption (device-dependent)

Enable in config:

```yaml
logging:
  log_config:
    system_metrics: true
```

## Project Structure

```
FLOps-benchmarking/
├── conf/                   # hydra configuration files
├── src/
│   ├── strategies/         # FL strategy implementations
│   ├── models/             # model architectures
│   ├── datasets/           # dataset loading/partitioning
│   ├── scenarios/          # scenario handlers
│   ├── clients/            # flower client
│   ├── server/             # flower server
│   ├── monitoring/         # system metrics collection
│   └── utils/              # logging, helpers
├── scripts/                # plotting and comparison scripts
├── deployment/
│   ├── ansible/            # ansible playbooks
│   └── ssh/                # ssh utilities
├── outputs/                # experiment outputs
├── pyproject.toml
├── requirements.txt
└── README.md
```

