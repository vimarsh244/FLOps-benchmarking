"""Compare multiple experiments and generate comparison plots."""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np


# set style
plt.style.use('seaborn-v0_8-whitegrid')

# color palette for multiple experiments
COLORS = list(mcolors.TABLEAU_COLORS.values())


def load_experiment(exp_dir: Union[str, Path]) -> Dict:
    """Load experiment data from directory.
    
    Args:
        exp_dir: Path to experiment directory
    
    Returns:
        Dictionary with experiment data
    """
    exp_dir = Path(exp_dir)
    
    data = {
        "name": exp_dir.name,
        "path": str(exp_dir),
        "metrics": None,
        "config": None,
        "system_metrics": None,
    }
    
    # load metrics
    metrics_path = exp_dir / "logs" / "metrics.json"
    if not metrics_path.exists():
        metrics_path = exp_dir / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            data["metrics"] = json.load(f)
    
    # load config
    config_path = exp_dir / "config.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            data["config"] = yaml.safe_load(f)
    
    # load system metrics
    system_path = exp_dir / "logs" / "system_metrics.json"
    if system_path.exists():
        with open(system_path) as f:
            data["system_metrics"] = json.load(f)
    
    return data


def compare_accuracy(
    experiments: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Accuracy Comparison",
    show: bool = True,
) -> plt.Figure:
    """Compare accuracy across experiments.
    
    Args:
        experiments: List of experiment data dictionaries
        output_path: Optional path to save figure
        title: Plot title
        show: Whether to display plot
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, exp in enumerate(experiments):
        if exp["metrics"] is None:
            continue
        
        rounds = [m.get("round", j) for j, m in enumerate(exp["metrics"])]
        accuracies = [m.get("accuracy") for m in exp["metrics"]]
        
        # filter None values
        valid_data = [(r, a) for r, a in zip(rounds, accuracies) if a is not None]
        if not valid_data:
            continue
        
        r, a = zip(*valid_data)
        
        color = COLORS[i % len(COLORS)]
        label = exp.get("name", f"Experiment {i+1}")
        
        ax.plot(r, a, "o-", color=color, label=label, 
                linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def compare_loss(
    experiments: List[Dict],
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Loss Comparison",
    show: bool = True,
) -> plt.Figure:
    """Compare loss across experiments.
    
    Args:
        experiments: List of experiment data dictionaries
        output_path: Optional path to save figure
        title: Plot title
        show: Whether to display plot
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, exp in enumerate(experiments):
        if exp["metrics"] is None:
            continue
        
        rounds = [m.get("round", j) for j, m in enumerate(exp["metrics"])]
        losses = [m.get("loss") for m in exp["metrics"]]
        
        valid_data = [(r, l) for r, l in zip(rounds, losses) if l is not None]
        if not valid_data:
            continue
        
        r, l = zip(*valid_data)
        
        color = COLORS[i % len(COLORS)]
        label = exp.get("name", f"Experiment {i+1}")
        
        ax.plot(r, l, "o-", color=color, label=label,
                linewidth=2, markersize=4, alpha=0.8)
    
    ax.set_xlabel("Round", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def compare_convergence(
    experiments: List[Dict],
    target_accuracy: float = 0.8,
    output_path: Optional[Union[str, Path]] = None,
    title: str = "Convergence Comparison",
    show: bool = True,
) -> plt.Figure:
    """Compare convergence speed across experiments.
    
    Args:
        experiments: List of experiment data dictionaries
        target_accuracy: Target accuracy for convergence
        output_path: Optional path to save figure
        title: Plot title
        show: Whether to display plot
    
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names = []
    rounds_to_target = []
    final_accuracies = []
    
    for exp in experiments:
        if exp["metrics"] is None:
            continue
        
        name = exp.get("name", "Unknown")
        names.append(name)
        
        accuracies = [m.get("accuracy", 0) for m in exp["metrics"]]
        final_accuracies.append(accuracies[-1] if accuracies else 0)
        
        # find round to reach target
        target_round = None
        for i, acc in enumerate(accuracies):
            if acc and acc >= target_accuracy:
                target_round = i + 1
                break
        rounds_to_target.append(target_round)
    
    x = np.arange(len(names))
    width = 0.35
    
    # plot rounds to target
    bars1 = ax.bar(x - width/2, 
                   [r if r else 0 for r in rounds_to_target],
                   width, label=f"Rounds to {target_accuracy*100:.0f}% acc",
                   color=COLORS[0], alpha=0.8)
    
    # mark experiments that didn't reach target
    for i, (bar, r) in enumerate(zip(bars1, rounds_to_target)):
        if r is None:
            bar.set_color("gray")
            bar.set_alpha(0.3)
            ax.annotate("Not reached", xy=(bar.get_x() + bar.get_width()/2, 5),
                       ha="center", va="bottom", fontsize=8, rotation=90)
    
    ax2 = ax.twinx()
    bars2 = ax2.bar(x + width/2, final_accuracies, width, 
                    label="Final Accuracy", color=COLORS[1], alpha=0.8)
    
    ax.set_xlabel("Experiment", fontsize=12)
    ax.set_ylabel("Rounds to Target", fontsize=12)
    ax2.set_ylabel("Final Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right")
    
    # combine legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    
    if show:
        plt.show()
    
    return fig


def generate_comparison_report(
    experiment_dirs: List[Union[str, Path]],
    output_dir: Union[str, Path],
    show: bool = False,
) -> Path:
    """Generate comprehensive comparison report.
    
    Args:
        experiment_dirs: List of experiment directories to compare
        output_dir: Output directory for report
        show: Whether to display plots
    
    Returns:
        Path to output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # load all experiments
    experiments = []
    for exp_dir in experiment_dirs:
        exp = load_experiment(exp_dir)
        if exp["metrics"]:
            experiments.append(exp)
            print(f"Loaded: {exp['name']}")
    
    if not experiments:
        print("No valid experiments found!")
        return output_dir
    
    print(f"\nComparing {len(experiments)} experiments...")
    
    # generate comparison plots
    compare_accuracy(experiments, output_dir / "accuracy_comparison.png", show=show)
    plt.close()
    
    compare_loss(experiments, output_dir / "loss_comparison.png", show=show)
    plt.close()
    
    compare_convergence(experiments, output_path=output_dir / "convergence_comparison.png", show=show)
    plt.close()
    
    # generate summary table
    summary = []
    for exp in experiments:
        if exp["metrics"]:
            accuracies = [m.get("accuracy", 0) for m in exp["metrics"] if m.get("accuracy")]
            losses = [m.get("loss", 0) for m in exp["metrics"] if m.get("loss")]
            
            summary.append({
                "name": exp["name"],
                "rounds": len(exp["metrics"]),
                "final_accuracy": accuracies[-1] if accuracies else 0,
                "max_accuracy": max(accuracies) if accuracies else 0,
                "final_loss": losses[-1] if losses else 0,
                "min_loss": min(losses) if losses else 0,
            })
    
    # save summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nGenerated comparison report in: {output_dir}")
    print(f"  - accuracy_comparison.png")
    print(f"  - loss_comparison.png")
    print(f"  - convergence_comparison.png")
    print(f"  - summary.json")
    
    return output_dir


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare FLOps experiments")
    parser.add_argument("experiments", nargs="+", type=str,
                       help="Paths to experiment directories")
    parser.add_argument("-o", "--output", type=str, default="comparison_report",
                       help="Output directory for comparison report")
    parser.add_argument("--show", action="store_true",
                       help="Display plots interactively")
    
    args = parser.parse_args()
    
    generate_comparison_report(args.experiments, args.output, show=args.show)


if __name__ == "__main__":
    main()

