"""Visualization utilities for MO-SPPS experimental results.

Section 29-30. Generates plots for Pareto fronts, metric curves,
component usage, and comparative analysis.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    pass


def plot_pareto_front(
    archive_solutions: list[set[int]],
    archive_objectives: list[np.ndarray],
    title: str = "Pareto Front",
    xlabel: str = "f1",
    ylabel: str = "f2",
    ax=None,
    **kwargs,
):
    """Plot the 2D Pareto front from archive.

    Args:
        archive_solutions: List of solution sets.
        archive_objectives: List of objective vectors.
        title, xlabel, ylabel: Plot labels.
        ax: Optional matplotlib axis.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))

    objs = np.array(archive_objectives)
    ax.scatter(objs[:, 0], objs[:, 1], c="steelblue", s=30, alpha=0.7)

    # Highlight non-dominated front
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

    return ax


def plot_metric_curves(
    history: dict,
    metrics: list[str] | None = None,
    figsize: tuple = (14, 10),
):
    """Plot metric curves from tracker history.

    Args:
        history: MetricsTracker history dict.
        metrics: List of metric keys to plot. Defaults to all.
        figsize: Figure size.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    if metrics is None:
        metrics = [
            "archive_size",
            "hypervolume",
            "avg_jaccard_distance",
            "component_entropy_norm",
            "reuse_concentration",
        ]

    n_plots = len(metrics)
    n_cols = min(3, n_plots)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows * n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)

    x_key = "fe_count" if "fe_count" in history else "iteration"

    for idx, metric in enumerate(metrics):
        if metric not in history:
            continue
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]

        x = history[x_key]
        y = history[metric]
        ax.plot(x, y, linewidth=1.5)
        ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("FE count" if x_key == "fe_count" else "Iteration")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(metrics), n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig


def plot_component_usage_heatmap(
    occupancy_history: list[dict[int, int]],
    num_components: int,
    figsize: tuple = (12, 6),
):
    """Plot component usage heatmap over iterations.

    Args:
        occupancy_history: List of occupancy dicts per recorded iteration.
        num_components: Total number of components.
        figsize: Figure size.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    n_ticks = len(occupancy_history)
    data = np.zeros((num_components, n_ticks))

    for t, occupancy in enumerate(occupancy_history):
        for j in range(num_components):
            data[j, t] = occupancy.get(j, 0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(data, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Component ID")
    ax.set_title("Component Usage Heatmap")
    plt.colorbar(im, ax=ax, label="Occupancy")

    return fig


def compare_pareto_fronts(
    results: dict[str, tuple[list[np.ndarray], str]],
    title: str = "Pareto Front Comparison",
    xlabel: str = "f1",
    ylabel: str = "f2",
    figsize: tuple = (10, 8),
):
    """Compare Pareto fronts from multiple algorithms.

    Args:
        results: dict mapping label -> (objectives_list, color/style)
        title, xlabel, ylabel: Plot labels.
        figsize: Figure size.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available")
        return

    fig, ax = plt.subplots(figsize=figsize)

    for label, (objectives, style) in results.items():
        objs = np.array(objectives)
        ax.scatter(
            objs[:, 0], objs[:, 1], label=label, s=30, alpha=0.6
        )

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig
