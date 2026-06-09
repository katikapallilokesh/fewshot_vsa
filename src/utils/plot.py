"""
plot.py
-------
Visualisation utilities for training histories and final comparison plots.
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


# ---------------------------------------------------------------------------
# Training curve (per run)
# ---------------------------------------------------------------------------

def plot_training_history(
    history: dict[str, list[float]],
    title_prefix: str = "",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Plots train/val accuracy and loss side-by-side for a single training run.

    Args:
        history      : dict with keys train_loss, val_loss, train_acc, val_acc
        title_prefix : e.g. "Baseline 10-shot"
        save_path    : if given, save figure to this path
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history["train_acc"]) + 1)

    ax1.plot(epochs, history["train_acc"], label="Train")
    ax1.plot(epochs, history["val_acc"],   label="Validation")
    ax1.set_title(f"{title_prefix} Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_loss"], label="Train")
    ax2.plot(epochs, history["val_loss"],   label="Validation")
    ax2.set_title(f"{title_prefix} Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


# ---------------------------------------------------------------------------
# Final comparison plots (across k-shot values)
# ---------------------------------------------------------------------------

def _smooth(x: np.ndarray, y: np.ndarray, n: int = 200):
    """Fit a spline for smoother line plots."""
    spl = make_interp_spline(x, y)
    xs  = np.linspace(x.min(), x.max(), n)
    return xs, spl(xs)


def plot_kshot_metric(
    k_shots: list[int],
    baseline_vals: list[float],
    single_ss_vals: list[float],
    iterative_ss_vals: list[float],
    metric_name: str = "Accuracy",
    split: str = "Test",
    save_path: Optional[str | Path] = None,
) -> None:
    """
    Comparison line plot across k-shot values for the three conditions.

    Args:
        k_shots            : list of k values, e.g. [1,5,10,...,100]
        baseline_vals      : metric value per k (baseline condition)
        single_ss_vals     : metric value per k (single semi-supervised)
        iterative_ss_vals  : metric value per k (iterative semi-supervised)
        metric_name        : y-axis label, e.g. "Accuracy", "F1"
        split              : "Train" or "Test" (used in title)
        save_path          : optional path to save figure
    """
    x = np.array(k_shots, dtype=float)
    colors = {"Baseline": "blue", "Single SS": "red", "Iterative SS": "green"}
    series = {
        "Baseline":     np.array(baseline_vals,      dtype=float),
        "Single SS":    np.array(single_ss_vals,      dtype=float),
        "Iterative SS": np.array(iterative_ss_vals,   dtype=float),
    }

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, y in series.items():
        color = colors[label]
        ax.scatter(x, y, marker="*", color=color, zorder=3)
        if len(x) > 3:   # spline needs at least 4 points
            xs, ys = _smooth(x, y)
            ax.plot(xs, ys, color=color, label=label)
        else:
            ax.plot(x, y, color=color, label=label)

    ax.set_title(f"{split} {metric_name}")
    ax.set_xlabel("k-shot")
    ax.set_ylabel(metric_name)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_pseudo_samples(
    k_shots: list[int],
    single_ss_samples: list[int],
    iterative_ss_samples: list[int],
    save_path: Optional[str | Path] = None,
) -> None:
    """Plots the number of pseudo-labelled samples selected per k-shot value."""
    x = np.array(k_shots, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))

    for label, vals, color in [
        ("Single SS",    single_ss_samples,    "blue"),
        ("Iterative SS", iterative_ss_samples, "red"),
    ]:
        y = np.array(vals, dtype=float)
        ax.scatter(x, y, marker="*", color=color, zorder=3)
        if len(x) > 3:
            xs, ys = _smooth(x, y)
            ax.plot(xs, ys, color=color, label=label)
        else:
            ax.plot(x, y, color=color, label=label)

    ax.set_title("Pseudo-labelled Samples Selected")
    ax.set_xlabel("k-shot")
    ax.set_ylabel("Samples Selected")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_all_results(
    results: dict,
    k_shots: list[int],
    plot_dir: str | Path = "plots",
) -> None:
    """
    Convenience wrapper — generates all comparison plots from the aggregated
    results dict produced by run_fewshot_experiments().

    `results` structure (produced by run_experiment.py):
        {
          "test_accuracy":  {"baseline": [...], "single": [...], "iterative": [...]},
          "test_f1":        { ... },
          "train_accuracy": { ... },
          "train_loss":     { ... },
          "test_loss":      { ... },
          "pseudo_samples": {"single": [...], "iterative": [...]},
        }
    """
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    metric_map = {
        "test_accuracy":  ("Test",  "Accuracy"),
        "test_f1":        ("Test",  "F1 Score"),
        "test_loss":      ("Test",  "Loss"),
        "train_accuracy": ("Train", "Accuracy"),
        "train_loss":     ("Train", "Loss"),
    }

    for key, (split, metric_name) in metric_map.items():
        if key not in results:
            continue
        d = results[key]
        plot_kshot_metric(
            k_shots=k_shots,
            baseline_vals=d["baseline"],
            single_ss_vals=d["single"],
            iterative_ss_vals=d["iterative"],
            metric_name=metric_name,
            split=split,
            save_path=plot_dir / f"{key}.png",
        )

    if "pseudo_samples" in results:
        ps = results["pseudo_samples"]
        plot_pseudo_samples(
            k_shots=k_shots,
            single_ss_samples=ps["single"],
            iterative_ss_samples=ps["iterative"],
            save_path=plot_dir / "pseudo_samples.png",
        )

    print(f"[plots] saved to {plot_dir}/")
