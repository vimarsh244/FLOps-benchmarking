"""Distributed execution runner for physical hardware deployment."""

import subprocess
import time
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from src.utils.helpers import load_env_file
from src.utils.logging import get_logger, setup_logging
from src.utils.wandb_logger import init_wandb_logger, finish_wandb

load_env_file()


@dataclass
class DeviceInfo:
    """Information about a remote device."""

    name: str
    host: str
    user: str
    ssh_key: str
    python_env: str
    device_type: str


def run_distributed_experiment(cfg: DictConfig) -> None:
    """Run federated learning experiment on distributed hardware.

    Args:
        cfg: Hydra configuration
    """
    logger = get_logger()
    logger.info("Starting distributed experiment")

    # parse device configurations
    devices = []
    for dev_cfg in cfg.hardware.get("devices", []):
        devices.append(
            DeviceInfo(
                name=dev_cfg.name,
                host=dev_cfg.host,
                user=dev_cfg.user,
                ssh_key=dev_cfg.ssh_key,
                python_env=dev_cfg.python_env,
                device_type=dev_cfg.device_type,
            )
        )

    if not devices:
        raise ValueError("No devices configured for distributed execution")

    logger.info(f"Configured {len(devices)} devices:")
    for dev in devices:
        logger.info(f"  - {dev.name} ({dev.device_type}): {dev.user}@{dev.host}")

    # deployment settings
    deploy_cfg = cfg.hardware.get("deployment", {})
    use_ansible = deploy_cfg.get("use_ansible", True)
    sync_code = deploy_cfg.get("sync_code", True)
    remote_path = deploy_cfg.get("remote_path", "~/flops-benchmarking")

    if sync_code:
        if use_ansible:
            logger.info("Syncing code to remote devices with Ansible...")
            sync_code_with_ansible(cfg, remote_path)
        else:
            logger.info("Syncing code to remote devices...")
            sync_code_to_devices(devices, remote_path)

    if use_ansible:
        run_with_ansible(cfg, devices, remote_path)
    else:
        run_with_ssh(cfg, devices, remote_path)


def sync_code_to_devices(devices: List[DeviceInfo], remote_path: str) -> None:
    """Sync code to all remote devices using rsync.

    Args:
        devices: List of device configurations
        remote_path: Path on remote devices
    """
    logger = get_logger()
    local_path = Path(__file__).parent.parent

    def sync_to_device(dev: DeviceInfo) -> bool:
        try:
            cmd = [
                "rsync",
                "-avz",
                "--delete",
                "-e",
                f"ssh -i {dev.ssh_key} -o StrictHostKeyChecking=no",
                f"{local_path}/",
                f"{dev.user}@{dev.host}:{remote_path}/",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode != 0:
                logger.error(f"Failed to sync to {dev.name}: {result.stderr}")
                return False
            logger.info(f"Synced code to {dev.name}")
            return True
        except Exception as e:
            logger.error(f"Error syncing to {dev.name}: {e}")
            return False

    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = {executor.submit(sync_to_device, dev): dev for dev in devices}
        for future in as_completed(futures):
            dev = futures[future]
            try:
                success = future.result()
                if not success:
                    logger.warning(f"Failed to sync to {dev.name}")
            except Exception as e:
                logger.error(f"Exception syncing to {dev.name}: {e}")


def sync_code_with_ansible(cfg: DictConfig, remote_path: str) -> None:
    """Sync code using the Ansible playbook.

    Args:
        cfg: Hydra configuration
        remote_path: Path on remote devices
    """
    logger = get_logger()
    project_root = Path(__file__).parent.parent
    inventory_path = cfg.hardware.deployment.get("ansible_inventory")
    if not inventory_path:
        logger.warning("Ansible inventory not configured, skipping Ansible sync")
        return

    inventory_file = Path(inventory_path)
    if not inventory_file.is_absolute():
        inventory_file = project_root / inventory_file

    playbook_path = project_root / "deployment" / "ansible" / "sync_code.yml"
    if not inventory_file.exists() or not playbook_path.exists():
        logger.warning("Ansible inventory or playbook missing, skipping Ansible sync")
        return

    cmd = [
        "ansible-playbook",
        "-i",
        str(inventory_file),
        str(playbook_path),
        "-e",
        f"flops_repo_path={remote_path}",
    ]
    result = subprocess.run(cmd, cwd=project_root)
    if result.returncode != 0:
        logger.error("Ansible sync failed")


def build_hydra_overrides(cfg: DictConfig) -> str:
    """Build hydra override string from configuration.

    Args:
        cfg: Hydra configuration

    Returns:
        String of hydra overrides for Ansible playbook
    """
    overrides = []

    # strategy
    if cfg.get("strategy") and cfg.strategy.get("name"):
        overrides.append(f"strategy={cfg.strategy.name}")

    # partitioner
    if cfg.get("partitioner") and cfg.partitioner.get("name"):
        overrides.append(f"partitioner={cfg.partitioner.name}")

    # dataset
    if cfg.get("dataset") and cfg.dataset.get("name"):
        overrides.append(f"dataset={cfg.dataset.name}")

    # model
    if cfg.get("model") and cfg.model.get("name"):
        overrides.append(f"model={cfg.model.name}")

    # scenario
    if cfg.get("scenario") and cfg.scenario.get("name"):
        overrides.append(f"scenario={cfg.scenario.name}")

    # server settings
    if cfg.get("server"):
        if cfg.server.get("num_rounds"):
            overrides.append(f"server.num_rounds={cfg.server.num_rounds}")

    # client settings
    if cfg.get("client"):
        if cfg.client.get("num_clients"):
            overrides.append(f"client.num_clients={cfg.client.num_clients}")

    # always include hardware=distributed
    overrides.append("hardware=distributed")

    return " ".join(overrides)


def run_with_ansible(
    cfg: DictConfig,
    devices: List[DeviceInfo],
    remote_path: str,
) -> None:
    """Run experiment using Ansible for orchestration.

    Args:
        cfg: Hydra configuration
        devices: List of device configurations
        remote_path: Path on remote devices
    """
    import json

    logger = get_logger()
    logger.info("Running with Ansible orchestration")

    # check for ansible inventory
    inventory_path = cfg.hardware.deployment.get("ansible_inventory")
    project_root = Path(__file__).parent.parent
    inventory_file = None
    if inventory_path:
        inventory_file = Path(inventory_path)
        if not inventory_file.is_absolute():
            inventory_file = project_root / inventory_file

    if inventory_file and inventory_file.exists():
        # run ansible playbook
        playbook_path = project_root / "deployment" / "ansible" / "run_experiment.yml"
        if playbook_path.exists():
            # build hydra overrides from config
            hydra_overrides = build_hydra_overrides(cfg)
            logger.info(f"Hydra overrides: {hydra_overrides}")

            cmd = [
                "ansible-playbook",
                "-i",
                str(inventory_file),
                str(playbook_path),
                "-e",
                f"flops_repo_path={remote_path}",
                "-e",
                f"server_address={cfg.hardware.server.host}:{cfg.hardware.server.port}",
                "-e",
                json.dumps({"hydra_overrides": hydra_overrides}),
            ]
            logger.info(f"Running Ansible: {' '.join(cmd)}")
            subprocess.run(cmd, cwd=project_root)
        else:
            logger.warning("Ansible playbook not found, falling back to SSH")
            run_with_ssh(cfg, devices, remote_path)
    else:
        logger.warning("Ansible inventory not found, falling back to SSH")
        run_with_ssh(cfg, devices, remote_path)


def run_with_ssh(
    cfg: DictConfig,
    devices: List[DeviceInfo],
    remote_path: str,
) -> None:
    """Run experiment using direct SSH connections.

    Args:
        cfg: Hydra configuration
        devices: List of device configurations
        remote_path: Path on remote devices
    """
    logger = get_logger()

    logger.info("Running with direct SSH connections")

    server_cfg = cfg.hardware.get("server", {})
    server_host = server_cfg.get("host", "0.0.0.0")
    server_port = server_cfg.get("port", 8080)

    # start server locally or on first device
    logger.info(f"Starting server on {server_host}:{server_port}")

    # start clients on each device
    def start_client(dev: DeviceInfo, partition_id: int) -> Optional[subprocess.Popen]:
        try:
            cmd = [
                "ssh",
                "-i",
                dev.ssh_key,
                "-o",
                "StrictHostKeyChecking=no",
                f"{dev.user}@{dev.host}",
                f"cd {remote_path} && {dev.python_env} -m src.client_runner "
                f"--server-address {server_host}:{server_port} "
                f"--partition-id {partition_id}",
            ]
            logger.info(f"Starting client on {dev.name} (partition {partition_id})")
            return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            logger.error(f"Failed to start client on {dev.name}: {e}")
            return None

    # start all clients
    processes = []
    for i, dev in enumerate(devices):
        proc = start_client(dev, i)
        if proc:
            processes.append((dev, proc))

    # wait for completion
    logger.info("Waiting for clients to complete...")
    for dev, proc in processes:
        try:
            stdout, stderr = proc.communicate(timeout=3600)  # 1 hour timeout
            if proc.returncode != 0:
                logger.warning(f"Client on {dev.name} exited with code {proc.returncode}")
                logger.warning(f"stderr: {stderr.decode()}")
            else:
                logger.info(f"Client on {dev.name} completed successfully")
        except subprocess.TimeoutExpired:
            logger.warning(f"Client on {dev.name} timed out, terminating")
            proc.terminate()

    logger.info("Distributed experiment completed")


def _init_wandb_for_server(cfg: DictConfig):
    """Initialize wandb for the distributed server.

    In distributed mode, wandb runs only on the server and collects
    all aggregated metrics from clients. The strategies already call
    log_round_metrics() during aggregation, so we just need to
    initialize wandb here.

    Args:
        cfg: Hydra configuration

    Returns:
        wandb run instance or None if initialization fails
    """
    logger = get_logger()

    try:
        import wandb
        from omegaconf import OmegaConf

        wandb_cfg = cfg.logging.get("wandb", {})

        # convert omegaconf to dict for wandb
        config_dict = OmegaConf.to_container(cfg, resolve=True)

        project = wandb_cfg.get("project", "flops-benchmarking")
        run_name = wandb_cfg.get("run_name") or f"{cfg.experiment.name}-distributed"
        mode = wandb_cfg.get("mode", "online")

        logger.info(f"Initializing W&B on server: project={project}, name={run_name}, mode={mode}")

        run = wandb.init(
            project=project,
            entity=wandb_cfg.get("entity"),
            name=run_name,
            tags=list(wandb_cfg.get("tags", [])) + ["distributed"],
            notes=wandb_cfg.get("notes"),
            mode=mode,
            config=config_dict,
        )

        if run:
            logger.info(f"W&B run initialized: {run.name} (id: {run.id})")
            logger.info(f"W&B run URL: {run.get_url()}")
            # initialize global wandb logger for strategies to use
            init_wandb_logger(run)
        else:
            logger.warning("W&B run returned None")

        return run

    except ImportError:
        logger.warning("wandb not installed, skipping W&B logging. Install with: pip install wandb")
        return None
    except Exception as e:
        logger.warning(f"Failed to initialize W&B: {e}")
        import traceback

        logger.debug(traceback.format_exc())
        return None


def run_server(cfg: DictConfig) -> None:
    """Run the Flower server for distributed mode.

    Args:
        cfg: Hydra configuration
    """
    import flwr as fl
    from src.server.server_app import create_strategy
    from src.models.registry import get_model_from_config
    from src.server.server_app import get_initial_parameters

    logger = get_logger()
    wandb_run = None

    if cfg.logging.get("backend") == "wandb":
        wandb_run = _init_wandb_for_server(cfg)

    # create model for initial parameters
    model = get_model_from_config(cfg.model, cfg.dataset)
    initial_parameters = get_initial_parameters(model)

    # create strategy
    strategy = create_strategy(cfg, initial_parameters=initial_parameters)

    # server config
    # use config values, potentially overridden by CLI args
    server_cfg = cfg.hardware.get("server", {})
    host = server_cfg.get("host", "0.0.0.0")
    port = server_cfg.get("port", 8080)

    num_rounds = cfg.server.get("num_rounds")

    logger.info(f"Starting Flower server on {host}:{port} for {num_rounds} rounds")

    try:
        # start server
        fl.server.start_server(
            server_address=f"{host}:{port}",
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
        )
    except Exception as e:
        logger.error(f"Error starting server: {e}")
        raise e
    finally:
        # ensure wandb is properly finished even if server crashes
        if wandb_run is not None:
            logger.info("Finishing wandb run...")
            finish_wandb()
            logger.info("Wandb run finished")


def run_client(
    server_address: str,
    partition_id: int,
    cfg: DictConfig,
) -> None:
    """Run a Flower client for distributed mode.

    Args:
        server_address: Address of the Flower server
        partition_id: Client partition ID
        cfg: Hydra configuration
    """
    import flwr as fl
    from src.clients.registry import get_client_class, get_client_type_for_strategy
    from src.models.registry import get_model_from_config
    from src.datasets.loader import load_data
    from src.scenarios.registry import get_scenario

    logger = get_logger()
    logger.info(f"Starting client {partition_id}, connecting to {server_address}")

    # determine client type based on strategy
    client_type = cfg.client.get("client_type", None)
    if client_type is None:
        strategy_name = cfg.strategy.get("name", "fedavg")
        client_type = get_client_type_for_strategy(strategy_name)

    ClientClass = get_client_class(client_type)
    logger.info(f"Using client type: {client_type} ({ClientClass.__name__})")

    # create model
    model = get_model_from_config(cfg.model, cfg.dataset)

    # load data
    trainloader, valloader = load_data(
        partition_id=partition_id,
        num_partitions=cfg.client.num_clients,
        dataset_cfg=cfg.dataset,
        partitioner_cfg=cfg.partitioner,
        batch_size=cfg.client.batch_size,
        test_fraction=cfg.evaluation.test_fraction,
        dataloader_cfg=cfg.training,
    )

    # create scenario handler
    scenario = get_scenario(cfg.scenario)

    # create client using the selected class
    client = ClientClass(
        model=model,
        trainloader=trainloader,
        valloader=valloader,
        partition_id=partition_id,
        config=cfg,
        scenario_handler=scenario,
    )

    # start client
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),
    )


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for distributed mode.
    
    This can be called with:
        python -m src.run_distributed hardware.run_mode=server
        python -m src.run_distributed hardware.run_mode=client \
            hardware.server_address=HOST:PORT hardware.partition_id=ID
    
    Args:
        cfg: Hydra configuration
    """
    import sys

    logger = setup_logging(level="INFO")

    mode = cfg.hardware.get("run_mode")
    if not mode:
        logger.error(
            "Missing hardware.run_mode. Set hardware.run_mode=server or "
            "hardware.run_mode=client in your overrides."
        )
        sys.exit(1)

    mode = str(mode).lower()

    if mode == "server":
        logger.info("Starting in server mode")
        # optionally parse host/port from command line
        # for now, use config
        run_server(cfg)

    elif mode == "client":
        logger.info("Starting in client mode")
        server_address = cfg.hardware.get("server_address")
        if not server_address:
            server_cfg = cfg.hardware.get("server", {})
            host = server_cfg.get("host", "localhost")
            port = server_cfg.get("port", 8080)
            server_address = f"{host}:{port}"
            logger.info(f"Using server address from config: {server_address}")

        partition_id = cfg.hardware.get("partition_id")
        if partition_id is None:
            logger.error("hardware.partition_id is required for client mode")
            sys.exit(1)

        run_client(server_address, partition_id, cfg)

    else:
        logger.error(f"Unknown mode: {mode}. Use 'server' or 'client'")
        sys.exit(1)


if __name__ == "__main__":
    main()
