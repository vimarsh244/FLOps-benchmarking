"""Wandb logging utility for federated learning strategies.

This module provides a global wandb logger that strategies can use to log metrics
during the aggregation phase, since the new Flower simulation API with ServerApp/ClientApp
doesn't return the History object to the caller.
"""

from typing import Dict, Any, Optional
import logging

# global wandb run reference
_wandb_run = None
_logger = logging.getLogger("flops")


def init_wandb_logger(run) -> None:
    """Initialize the global wandb logger with a run instance.
    
    Args:
        run: wandb run instance
    """
    global _wandb_run
    _wandb_run = run
    if run is not None:
        _logger.info(f"Wandb logger initialized with run: {run.name}")


def get_wandb_run():
    """Get the global wandb run instance.
    
    Returns:
        wandb run instance or None if not initialized
    """
    return _wandb_run


def log_metrics(metrics: Dict[str, Any], step: Optional[int] = None, prefix: str = "") -> None:
    """Log metrics to wandb.
    
    Args:
        metrics: dictionary of metric name to value
        step: optional step/round number
        prefix: optional prefix for metric names
    """
    global _wandb_run
    
    if _wandb_run is None:
        return
    
    try:
        import wandb
        
        # add prefix to metric names if provided
        if prefix:
            log_dict = {f"{prefix}/{k}": v for k, v in metrics.items()}
        else:
            log_dict = dict(metrics)
        
        # log to wandb
        if step is not None:
            wandb.log(log_dict, step=step)
        else:
            wandb.log(log_dict)
        
        _logger.debug(f"Logged to wandb (step={step}): {log_dict}")
        
    except Exception as e:
        _logger.warning(f"Failed to log to wandb: {e}")


def log_round_metrics(
    round_num: int,
    fit_metrics: Optional[Dict[str, Any]] = None,
    evaluate_metrics: Optional[Dict[str, Any]] = None,
    loss: Optional[float] = None,
) -> None:
    """Log metrics for a training round.
    
    Args:
        round_num: current round number
        fit_metrics: aggregated fit metrics from clients
        evaluate_metrics: aggregated evaluation metrics from clients
        loss: aggregated loss value
    """
    global _wandb_run
    
    # build log dict for both console and wandb
    log_dict = {"round": round_num}
    
    # add fit metrics with train_ prefix
    if fit_metrics:
        for k, v in fit_metrics.items():
            if isinstance(v, (int, float)):
                log_dict[f"train/{k}"] = v
    
    # add eval metrics with eval_ prefix
    if evaluate_metrics:
        for k, v in evaluate_metrics.items():
            if isinstance(v, (int, float)):
                log_dict[f"eval/{k}"] = v
    
    # add loss
    if loss is not None:
        log_dict["eval/loss"] = loss
    
    # always log to console so user can see progress
    metrics_str = ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in log_dict.items())
    _logger.info(f"Round {round_num} metrics: {metrics_str}")
    
    # log to wandb if available
    if _wandb_run is not None:
        try:
            import wandb
            wandb.log(log_dict, step=round_num)
            _logger.debug(f"Round {round_num} metrics logged to wandb")
        except Exception as e:
            _logger.warning(f"Failed to log round metrics to wandb: {e}")


def finish_wandb() -> None:
    """Finish the wandb run."""
    global _wandb_run
    
    if _wandb_run is not None:
        try:
            import wandb
            wandb.finish()
            _wandb_run = None
        except Exception as e:
            _logger.warning(f"Failed to finish wandb run: {e}")

