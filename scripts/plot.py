"""Plotting utilities for FLOps benchmarking experiments."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import argparse

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# set style for better looking plots
plt.style.use("seaborn-v0_8-whitegrid")


def load_metrics(metrics_path: Union[str, Path]) -> List[Dict]:
    """Load metrics from JSON file.

    Args:
        metrics_path: Path to metrics.json file

    Returns:
        List of metric dictionaries
    """
    with open(metrics_path) as f:
        return json.load(f)


def plot_training_progress(
    metrics: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Training Progress",
    show: bool = True,
) -> plt.Figure:
    """Plot training loss and accuracy over rounds.

    Args:
        metrics: List of metric dictionaries with 'round', 'loss', 'accuracy' keys
        output_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    rounds = [m.get("round", i) for i, m in enumerate(metrics)]
    losses = [m.get("loss") for m in metrics]
    accuracies = [m.get("accuracy") for m in metrics]

    # filter out None values
    loss_data = [(r, l) for r, l in zip(rounds, losses) if l is not None]
    acc_data = [(r, a) for r, a in zip(rounds, accuracies) if a is not None]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color_loss = "#e74c3c"
    color_acc = "#3498db"

    ax1.set_xlabel("Round", fontsize=12)
    ax1.set_ylabel("Loss", color=color_loss, fontsize=12)
    ax1.set_title(title, fontsize=14, fontweight="bold")

    if loss_data:
        loss_rounds, loss_values = zip(*loss_data)
        ax1.plot(
            loss_rounds,
            loss_values,
            "o-",
            color=color_loss,
            label="Loss",
            linewidth=2,
            markersize=4,
        )
    ax1.tick_params(axis="y", labelcolor=color_loss)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Accuracy", color=color_acc, fontsize=12)

    if acc_data:
        acc_rounds, acc_values = zip(*acc_data)
        ax2.plot(
            acc_rounds,
            acc_values,
            "s-",
            color=color_acc,
            label="Accuracy",
            linewidth=2,
            markersize=4,
        )
    ax2.tick_params(axis="y", labelcolor=color_acc)

    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_client_participation(
    metrics: List[Dict],
    num_clients: int,
    drop_events: Optional[List[Dict]] = None,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Client Participation",
    show: bool = True,
) -> plt.Figure:
    """Plot client participation over rounds with drop events highlighted.

    Args:
        metrics: List of metric dictionaries
        num_clients: Total number of clients
        drop_events: Optional list of drop event configs
        output_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    rounds = [m.get("round", i) for i, m in enumerate(metrics)]
    max_round = max(rounds) if rounds else 50

    fig, ax = plt.subplots(figsize=(12, 6))

    # create participation matrix
    participation = np.ones((num_clients, max_round + 1))

    # mark drops if provided
    if drop_events:
        for event in drop_events:
            client_ids = event.get("client_ids", [])
            start = event.get("disconnect_round", 0)
            end = event.get("rejoin_round", max_round)
            for cid in client_ids:
                if cid < num_clients:
                    participation[cid, start:end] = 0

    # plot heatmap
    cmap = plt.cm.RdYlGn
    im = ax.imshow(participation, aspect="auto", cmap=cmap, interpolation="nearest", vmin=0, vmax=1)

    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Client ID", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_yticks(range(num_clients))

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Participating", fontsize=10)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Disconnected", "Connected"])

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_system_metrics(
    metrics: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "System Metrics",
    show: bool = True,
) -> plt.Figure:
    """Plot system metrics (CPU, memory, network) over time.

    Args:
        metrics: List of system metric dictionaries
        output_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    timestamps = list(range(len(metrics)))

    cpu = [m.get("cpu_percent", 0) for m in metrics]
    memory = [m.get("memory_percent", 0) for m in metrics]
    net_sent = [m.get("net_sent_mb", 0) for m in metrics]
    net_recv = [m.get("net_recv_mb", 0) for m in metrics]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # cpu usage
    axes[0, 0].plot(timestamps, cpu, color="#e74c3c", linewidth=2)
    axes[0, 0].fill_between(timestamps, cpu, alpha=0.3, color="#e74c3c")
    axes[0, 0].set_xlabel("Sample")
    axes[0, 0].set_ylabel("CPU Usage (%)")
    axes[0, 0].set_title("CPU Usage")
    axes[0, 0].set_ylim(0, 100)
    axes[0, 0].grid(True, alpha=0.3)

    # memory usage
    axes[0, 1].plot(timestamps, memory, color="#3498db", linewidth=2)
    axes[0, 1].fill_between(timestamps, memory, alpha=0.3, color="#3498db")
    axes[0, 1].set_xlabel("Sample")
    axes[0, 1].set_ylabel("Memory Usage (%)")
    axes[0, 1].set_title("Memory Usage")
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3)

    # network sent
    axes[1, 0].plot(timestamps, net_sent, color="#2ecc71", linewidth=2)
    axes[1, 0].fill_between(timestamps, net_sent, alpha=0.3, color="#2ecc71")
    axes[1, 0].set_xlabel("Sample")
    axes[1, 0].set_ylabel("Network Sent (MB)")
    axes[1, 0].set_title("Network Traffic - Sent")
    axes[1, 0].grid(True, alpha=0.3)

    # network received
    axes[1, 1].plot(timestamps, net_recv, color="#9b59b6", linewidth=2)
    axes[1, 1].fill_between(timestamps, net_recv, alpha=0.3, color="#9b59b6")
    axes[1, 1].set_xlabel("Sample")
    axes[1, 1].set_ylabel("Network Received (MB)")
    axes[1, 1].set_title("Network Traffic - Received")
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def plot_power_consumption(
    metrics: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Power Consumption",
    show: bool = True,
) -> plt.Figure:
    """Plot power consumption over time.

    Args:
        metrics: List of power metric dictionaries
        output_path: Optional path to save the figure
        title: Plot title
        show: Whether to display the plot

    Returns:
        matplotlib Figure object
    """
    timestamps = list(range(len(metrics)))

    total_power = [m.get("power_total_w", 0) or 0 for m in metrics]
    gpu_power = [m.get("power_gpu_w", 0) or 0 for m in metrics]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(timestamps, total_power, color="#e74c3c", linewidth=2, label="Total Power")
    ax.fill_between(timestamps, total_power, alpha=0.3, color="#e74c3c")

    if any(gpu_power):
        ax.plot(timestamps, gpu_power, color="#3498db", linewidth=2, label="GPU Power")
        ax.fill_between(timestamps, gpu_power, alpha=0.3, color="#3498db")

    ax.set_xlabel("Sample", fontsize=12)
    ax.set_ylabel("Power (W)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # add energy consumption annotation
    if total_power:
        interval = 5.0  # assume 5 second intervals
        total_energy = sum(total_power) * interval / 3600  # Wh
        ax.annotate(
            f"Total Energy: {total_energy:.2f} Wh",
            xy=(0.98, 0.98),
            xycoords="axes fraction",
            ha="right",
            va="top",
            fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def auto_plot_experiment(
    experiment_dir: Union[str, Path],
    show: bool = False,
) -> List[Path]:
    """Automatically generate all plots for an experiment.

    Args:
        experiment_dir: Path to experiment output directory
        show: Whether to display plots

    Returns:
        List of generated plot paths
    """
    exp_dir = Path(experiment_dir)
    plots_dir = exp_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    generated = []

    # plot training metrics - check both logs subdirectory and direct location
    metrics_path = exp_dir / "logs" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = exp_dir / "metrics.json"

    if metrics_path.exists():
        metrics = load_metrics(metrics_path)

        # training progress
        fig_path = plots_dir / "training_progress.png"
        plot_training_progress(metrics, fig_path, show=show)
        generated.append(fig_path)
        plt.close()

    # plot system metrics - check both locations
    system_path = exp_dir / "logs" / "system_metrics.json"
    if not system_path.exists():
        system_path = exp_dir / "system_metrics.json"

    if system_path.exists():
        with open(system_path) as f:
            system_metrics = json.load(f)

        fig_path = plots_dir / "system_metrics.png"
        plot_system_metrics(system_metrics, fig_path, show=show)
        generated.append(fig_path)
        plt.close()

    # plot client metrics - check both locations
    client_path = exp_dir / "logs" / "client_metrics.json"
    if not client_path.exists():
        client_path = exp_dir / "client_metrics.json"

    if client_path.exists():
        # additional client-specific plots could go here
        pass

    print(f"Generated {len(generated)} plots in {plots_dir}")
    return generated


def main():
    """Main entry point for plotting script."""
    parser = argparse.ArgumentParser(description="Plot FLOps experiment results")
    parser.add_argument("experiment_dir", type=str, help="Path to experiment output directory")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    auto_plot_experiment(args.experiment_dir, show=args.show)


if __name__ == "__main__":
    main()
