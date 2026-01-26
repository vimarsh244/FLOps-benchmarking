"""Utility functions and helpers."""

import logging
import os
import json
import random
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import yaml


def load_env_file(filename: str = ".env") -> bool:
    """Load environment variables from a .env file if available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    project_root = Path(__file__).resolve().parents[2]
    env_path = project_root / filename

    if not env_path.exists():
        return False

    loaded = load_dotenv(dotenv_path=env_path, override=False)
    if loaded:
        logging.getLogger(__name__).debug(f"loaded environment variables from {env_path}")
    return loaded


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # for reproducibility (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device(device_cfg: str = "auto") -> torch.device:
    """Get the device to use for training.

    Args:
        device_cfg: Device configuration (auto, cpu, cuda, mps)

    Returns:
        torch.device object
    """
    if device_cfg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_cfg)


def save_results(
    results: Dict[str, Any],
    output_dir: Union[str, Path],
    filename: str = "results.json",
) -> Path:
    """Save results to a JSON file.

    Args:
        results: Dictionary of results to save
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    # convert numpy types to python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        return obj

    with open(output_path, "w") as f:
        json.dump(convert(results), f, indent=2)

    return output_path


def load_results(path: Union[str, Path]) -> Dict[str, Any]:
    """Load results from a JSON file.

    Args:
        path: Path to results file

    Returns:
        Dictionary of results
    """
    with open(path, "r") as f:
        return json.load(f)


def save_config(
    config: Any,
    output_dir: Union[str, Path],
    filename: str = "config.yaml",
) -> Path:
    """Save configuration to a YAML file.

    Args:
        config: Configuration object (DictConfig or dict)
        output_dir: Output directory
        filename: Output filename

    Returns:
        Path to saved file
    """
    from omegaconf import OmegaConf

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / filename

    if hasattr(config, "__dict__"):
        config_dict = OmegaConf.to_container(config, resolve=True)
    else:
        config_dict = dict(config)

    with open(output_path, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False)

    return output_path


def get_num_model_parameters(model: torch.nn.Module) -> int:
    """Get the number of trainable parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_bytes(size: int) -> str:
    """Format bytes as human-readable string.

    Args:
        size: Size in bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.2f} {unit}"
        size /= 1024
    return f"{size:.2f} PB"


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string (e.g., "1h 23m 45s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = seconds // 60
    seconds = seconds % 60

    if minutes < 60:
        return f"{int(minutes)}m {int(seconds)}s"

    hours = minutes // 60
    minutes = minutes % 60

    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
