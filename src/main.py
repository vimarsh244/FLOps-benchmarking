"""Main entry point for FLOps benchmarking experiments."""

import sys
from pathlib import Path
from datetime import datetime

import hydra
from omegaconf import DictConfig, OmegaConf

from src.utils.helpers import load_env_file, set_seed, save_config
from src.utils.logging import setup_logging, get_logger, ExperimentLogger

# load .env before hydra relocates the working directory so wandb sees WANDB_API_KEY
load_env_file()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for experiments.
    
    Args:
        cfg: Hydra configuration
    """
    # setup logging
    logger = setup_logging(level="INFO")
    logger.info("Starting FLOps Benchmarking Experiment")
    logger.info(f"Configuration:\n{OmegaConf.to_yaml(cfg)}")
    
    # set random seed
    set_seed(cfg.experiment.seed)
    
    # determine execution mode
    mode = cfg.hardware.get("mode", "simulation")
    
    if mode == "simulation":
        run_simulation(cfg)
    elif mode == "distributed":
        run_distributed(cfg)
    else:
        raise ValueError(f"Unknown hardware mode: {mode}")


def run_simulation(cfg: DictConfig) -> None:
    """Run federated learning simulation.
    
    Args:
        cfg: Hydra configuration
    """
    from flwr.simulation import run_simulation as flwr_run_simulation
    from flwr.client import ClientApp
    from flwr.server import ServerApp
    
    from src.clients.base_client import create_client_fn
    from src.server.server_app import create_server_fn
    
    logger = get_logger()
    logger.info("Running in simulation mode")
    
    # create output directory
    output_dir = Path(cfg.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # save config
    save_config(cfg, output_dir)
    
    # create experiment logger
    with ExperimentLogger(cfg, output_dir) as exp_logger:
        # create client app
        client_fn = create_client_fn(cfg)
        client_app = ClientApp(client_fn=client_fn)
        
        # create server app
        server_fn = create_server_fn(cfg)
        server_app = ServerApp(server_fn=server_fn)
        
        # configure simulation
        num_clients = cfg.client.num_clients
        
        # ray backend configuration
        ray_cfg = cfg.hardware.get("ray", {})
        backend_config = {
            "init_args": {"num_cpus": ray_cfg.get("num_cpus", 4)},
            "client_resources": {
                "num_cpus": ray_cfg.get("client_resources", {}).get("num_cpus", 1),
                "num_gpus": ray_cfg.get("client_resources", {}).get("num_gpus", 0.0),
            },
        }
        
        logger.info(f"Starting simulation with {num_clients} clients")
        logger.info(f"Strategy: {cfg.strategy.name}")
        logger.info(f"Dataset: {cfg.dataset.name}")
        logger.info(f"Model: {cfg.model.name}")
        logger.info(f"Scenario: {cfg.scenario.name}")
        logger.info(f"Partitioner: {cfg.partitioner.name}")
        
        # run simulation
        history = flwr_run_simulation(
            server_app=server_app,
            client_app=client_app,
            num_supernodes=num_clients,
            backend_name="ray",
            backend_config=backend_config,
            verbose_logging=cfg.hardware.simulation.get("verbose", False),
        )
        
        # debug: log history status
        logger.info(f"History object received: {history is not None}")
        if history is not None:
            logger.info(f"History object type: {type(history)}")
        else:
            logger.warning("History object is None - this may be expected in some Flower versions")
        
        # process and log results
        if history is not None:
            # debug: print what attributes the history has
            logger.info(f"History object type: {type(history)}")
            logger.info(f"History attributes: {dir(history)}")
            
            # collect all metrics per round into a combined dict
            round_metrics = {}
            
            # collect distributed losses (from evaluate)
            if hasattr(history, "losses_distributed") and history.losses_distributed:
                logger.info(f"losses_distributed: {history.losses_distributed}")
                for round_num, loss in history.losses_distributed:
                    if round_num not in round_metrics:
                        round_metrics[round_num] = {}
                    round_metrics[round_num]["loss"] = loss
            
            # collect centralized losses
            if hasattr(history, "losses_centralized") and history.losses_centralized:
                logger.info(f"losses_centralized: {history.losses_centralized}")
                for round_num, loss in history.losses_centralized:
                    if round_num not in round_metrics:
                        round_metrics[round_num] = {}
                    round_metrics[round_num]["centralized_loss"] = loss
            
            # collect distributed evaluate metrics (accuracy etc.)
            if hasattr(history, "metrics_distributed") and history.metrics_distributed:
                logger.info(f"metrics_distributed: {history.metrics_distributed}")
                metrics_dist = history.metrics_distributed
                # handle nested structure like {"evaluate": {"accuracy": [...]}}
                if isinstance(metrics_dist, dict):
                    for key, values in metrics_dist.items():
                        if isinstance(values, list):
                            # direct format: {"accuracy": [(round, val), ...]}
                            for round_num, value in values:
                                if round_num not in round_metrics:
                                    round_metrics[round_num] = {}
                                round_metrics[round_num][key] = value
                        elif isinstance(values, dict):
                            # nested format: {"evaluate": {"accuracy": [...]}}
                            for subkey, subvalues in values.items():
                                if isinstance(subvalues, list):
                                    for round_num, value in subvalues:
                                        if round_num not in round_metrics:
                                            round_metrics[round_num] = {}
                                        round_metrics[round_num][f"{key}_{subkey}"] = value
            
            # collect centralized metrics
            if hasattr(history, "metrics_centralized") and history.metrics_centralized:
                logger.info(f"metrics_centralized: {history.metrics_centralized}")
                for key, values in history.metrics_centralized.items():
                    if isinstance(values, list):
                        for round_num, value in values:
                            if round_num not in round_metrics:
                                round_metrics[round_num] = {}
                            round_metrics[round_num][f"centralized_{key}"] = value
            
            # collect fit metrics (train_loss etc.)
            if hasattr(history, "metrics_distributed_fit") and history.metrics_distributed_fit:
                logger.info(f"metrics_distributed_fit: {history.metrics_distributed_fit}")
                metrics_fit = history.metrics_distributed_fit
                if isinstance(metrics_fit, dict):
                    for key, values in metrics_fit.items():
                        if isinstance(values, list):
                            for round_num, value in values:
                                if round_num not in round_metrics:
                                    round_metrics[round_num] = {}
                                round_metrics[round_num][f"train_{key}"] = value
                        elif isinstance(values, dict):
                            for subkey, subvalues in values.items():
                                if isinstance(subvalues, list):
                                    for round_num, value in subvalues:
                                        if round_num not in round_metrics:
                                            round_metrics[round_num] = {}
                                        round_metrics[round_num][f"train_{key}_{subkey}"] = value
            
            # log all collected metrics per round
            logger.info(f"Collected metrics for {len(round_metrics)} rounds")
            for round_num in sorted(round_metrics.keys()):
                metrics = round_metrics[round_num]
                logger.info(f"Round {round_num} metrics: {metrics}")
                exp_logger.log_round(round_num, metrics)
            
            # print summary
            if round_metrics:
                last_round = max(round_metrics.keys())
                last_metrics = round_metrics[last_round]
                logger.info(f"Final round {last_round} metrics: {last_metrics}")
            else:
                logger.warning("No metrics were collected from the history object!")
        else:
            logger.warning("History object is None - metrics will not be saved!")
        
        logger.info("Simulation completed!")
        logger.info(f"Results saved to: {output_dir}")


def run_distributed(cfg: DictConfig) -> None:
    """Run federated learning on distributed hardware.
    
    Args:
        cfg: Hydra configuration
    """
    logger = get_logger()
    logger.info("Running in distributed mode")
    
    # import distributed runner
    from src.run_distributed import run_distributed_experiment
    
    run_distributed_experiment(cfg)


if __name__ == "__main__":
    main()

