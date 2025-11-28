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

from src.utils.logging import get_logger, setup_logging
from src.utils.wandb_logger import init_wandb_logger, finish_wandb


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
        devices.append(DeviceInfo(
            name=dev_cfg.name,
            host=dev_cfg.host,
            user=dev_cfg.user,
            ssh_key=dev_cfg.ssh_key,
            python_env=dev_cfg.python_env,
            device_type=dev_cfg.device_type,
        ))
    
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
                "rsync", "-avz", "--delete",
                "-e", f"ssh -i {dev.ssh_key} -o StrictHostKeyChecking=no",
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
    logger = get_logger()
    logger.info("Running with Ansible orchestration")
    
    # check for ansible inventory
    inventory_path = cfg.hardware.deployment.get("ansible_inventory")
    if inventory_path and Path(inventory_path).exists():
        # run ansible playbook
        playbook_path = Path(__file__).parent.parent / "deployment" / "ansible" / "run_experiment.yml"
        if playbook_path.exists():
            cmd = [
                "ansible-playbook",
                "-i", inventory_path,
                str(playbook_path),
                "-e", f"remote_path={remote_path}",
                "-e", f"server_address={cfg.hardware.server.host}:{cfg.hardware.server.port}",
            ]
            subprocess.run(cmd)
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
                "-i", dev.ssh_key,
                "-o", "StrictHostKeyChecking=no",
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
    
    # initialize wandb on server if enabled
    # this collects all client metrics since strategies call log_round_metrics()
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
    from src.clients.base_client import FlowerClient
    from src.models.registry import get_model_from_config
    from src.datasets.loader import load_data
    from src.scenarios.registry import get_scenario
    
    logger = get_logger()
    logger.info(f"Starting client {partition_id}, connecting to {server_address}")
    
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
    )
    
    # create scenario handler
    scenario = get_scenario(cfg.scenario)
    
    # create client
    client = FlowerClient(
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

if __name__ == "__main__":
    # Initialize logging
    setup_logging()
    
    # Argument parser
    parser = argparse.ArgumentParser(description="Flower Distributed Runner")
    subparsers = parser.add_subparsers(dest="mode", required=True, help="Mode to run: server or client")

    # Server subcommand
    server_parser = subparsers.add_parser("server", help="Run the Flower Server")
    server_parser.add_argument("--host", type=str, default="0.0.0.0", help="Server bind host")
    server_parser.add_argument("--port", type=int, default=8080, help="Server bind port")
    # Client subcommand
    client_parser = subparsers.add_parser("client", help="Run a Flower Client")
    client_parser.add_argument("--server-address", type=str, required=True, help="Server address (IP:PORT)")
    client_parser.add_argument("--partition-id", type=int, required=True, help="Data partition ID")

    args = parser.parse_args()

    # Load default Hydra configuration
    # This assumes your 'conf' directory is at the root of the repo (parent of src)
    with hydra.initialize(version_base=None, config_path="../conf"):
        cfg = hydra.compose(config_name="config")

    if args.mode == "server":
        # Override config with CLI arguments
        # FIX: Unlock the config dict so we can add 'server' keys if they are missing
        with open_dict(cfg):
            if "hardware" not in cfg:
                cfg.hardware = {}
            if "server" not in cfg.hardware:
                cfg.hardware.server = {}
            
            cfg.hardware.server.host = args.host
            cfg.hardware.server.port = args.port
            
            if "server" not in cfg:
                cfg.server = {}
            
            # handle wandb configuration from CLI
            if "logging" not in cfg:
                cfg.logging = {}
                    
        run_server(cfg)

    elif args.mode == "client":
        run_client(args.server_address, args.partition_id, cfg)
