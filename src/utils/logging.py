"""Logging utilities for FLOps benchmarking."""

import logging
import sys
from pathlib import Path
from typing import Optional, Union
from datetime import datetime

from omegaconf import DictConfig


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
        format_string: Optional custom format string
    
    Returns:
        Configured logger
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    
    if format_string is None:
        format_string = "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
    
    # create formatter
    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")
    
    # get root logger
    logger = logging.getLogger("flops")
    logger.setLevel(level)
    
    # clear existing handlers
    logger.handlers.clear()
    
    # console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # file handler if specified
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "flops") -> logging.Logger:
    """Get a logger instance.
    
    Args:
        name: Logger name
    
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class ExperimentLogger:
    """High-level experiment logger with support for multiple backends.
    
    Supports:
    - Console logging
    - File logging (JSON, CSV)
    - Weights & Biases (if enabled)
    """

    def __init__(
        self,
        config: DictConfig,
        output_dir: Optional[Union[str, Path]] = None,
    ):
        """Initialize experiment logger.
        
        Args:
            config: Hydra configuration
            output_dir: Output directory for logs
        """
        self.config = config
        base_output_dir = Path(output_dir) if output_dir else Path("outputs")
        base_output_dir.mkdir(parents=True, exist_ok=True)
        
        # check if logs should be saved to a subdirectory
        offline_cfg = config.logging.get("offline", {})
        if offline_cfg.get("enabled", False):
            save_dir = offline_cfg.get("save_dir")
            if save_dir:
                # use the save_dir from config (already resolved by Hydra)
                self.output_dir = Path(save_dir)
            else:
                self.output_dir = base_output_dir / "logs"
        else:
            self.output_dir = base_output_dir
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._metrics_history = []
        self._system_metrics_history = []
        self._client_metrics = {}
        
        # setup console logger
        log_level = config.logging.get("console", {}).get("level", "INFO")
        self.logger = setup_logging(
            level=log_level,
            log_file=self.output_dir / "experiment.log",
        )
        
        # setup wandb if enabled
        self.wandb_run = None
        if config.logging.get("backend") == "wandb":
            self._init_wandb(config)

    def _init_wandb(self, config: DictConfig) -> None:
        """Initialize Weights & Biases."""
        try:
            import wandb
            from omegaconf import OmegaConf
            from src.utils.wandb_logger import init_wandb_logger
            
            wandb_cfg = config.logging.get("wandb", {})
            
            # convert omegaconf to dict for wandb
            config_dict = OmegaConf.to_container(config, resolve=True)
            
            project = wandb_cfg.get("project", "flops-benchmarking")
            run_name = wandb_cfg.get("run_name") or config.experiment.name
            mode = wandb_cfg.get("mode", "online")
            
            self.logger.info(f"Initializing W&B: project={project}, name={run_name}, mode={mode}")
            
            self.wandb_run = wandb.init(
                project=project,
                entity=wandb_cfg.get("entity"),
                name=run_name,
                tags=list(wandb_cfg.get("tags", [])),
                notes=wandb_cfg.get("notes"),
                mode=mode,
                config=config_dict,
            )
            
            if self.wandb_run:
                self.logger.info(f"W&B run initialized: {self.wandb_run.name} (id: {self.wandb_run.id})")
                self.logger.info(f"W&B run URL: {self.wandb_run.get_url()}")
                # initialize global wandb logger for strategies
                init_wandb_logger(self.wandb_run)
            else:
                self.logger.warning("W&B run returned None")
            
        except ImportError:
            self.logger.warning("wandb not installed, skipping W&B logging. Install with: pip install wandb")
        except Exception as e:
            self.logger.warning(f"Failed to initialize W&B: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())

    def log_round(
        self,
        round_num: int,
        metrics: dict,
        prefix: str = "",
    ) -> None:
        """Log metrics for a training round.
        
        Args:
            round_num: Current round number
            metrics: Dictionary of metrics
            prefix: Optional prefix for metric names
        """
        # add to history
        entry = {"round": round_num, **metrics}
        self._metrics_history.append(entry)
        
        # log to console
        metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" 
                               for k, v in metrics.items())
        self.logger.info(f"Round {round_num}: {metrics_str}")
        
        # log to wandb with step for proper ordering
        if self.wandb_run is not None:
            import wandb
            log_dict = {f"{prefix}{k}" if prefix else k: v for k, v in metrics.items()}
            # use step parameter to ensure proper round ordering
            wandb.log(log_dict, step=round_num)
            self.logger.debug(f"Logged to W&B: {log_dict}")

    def log_client(
        self,
        client_id: int,
        round_num: int,
        metrics: dict,
    ) -> None:
        """Log metrics for a specific client.
        
        Args:
            client_id: Client ID
            round_num: Current round number
            metrics: Dictionary of metrics
        """
        if client_id not in self._client_metrics:
            self._client_metrics[client_id] = []
        
        entry = {"round": round_num, **metrics}
        self._client_metrics[client_id].append(entry)

    def log_system(
        self,
        metrics: dict,
    ) -> None:
        """Log system metrics.
        
        Args:
            metrics: Dictionary of system metrics
        """
        entry = {"timestamp": datetime.now().isoformat(), **metrics}
        self._system_metrics_history.append(entry)
        
        if self.wandb_run is not None:
            import wandb
            wandb.log({f"system/{k}": v for k, v in metrics.items()})

    def save(self) -> None:
        """Save all logged data to files."""
        import json
        import csv
        
        # save metrics history
        if self._metrics_history:
            metrics_path = self.output_dir / "metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self._metrics_history, f, indent=2)
            
            # also save as CSV
            csv_path = self.output_dir / "metrics.csv"
            keys = list(self._metrics_history[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self._metrics_history)
        
        # save system metrics
        if self._system_metrics_history:
            system_path = self.output_dir / "system_metrics.json"
            with open(system_path, "w") as f:
                json.dump(self._system_metrics_history, f, indent=2)
        
        # save client metrics
        if self._client_metrics:
            clients_path = self.output_dir / "client_metrics.json"
            with open(clients_path, "w") as f:
                json.dump(self._client_metrics, f, indent=2)
        
        self.logger.info(f"Saved logs to {self.output_dir}")

    def finish(self) -> None:
        """Finish logging and cleanup."""
        self.save()
        
        if self.wandb_run is not None:
            import wandb
            wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.finish()
        return False

