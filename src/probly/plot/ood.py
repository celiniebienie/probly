"""Plotting utilities for OOD evaluation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import numpy as np
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure


def plot_histogram(
    id_scores: np.ndarray,
    ood_scores: np.ndarray,
    ax: Axes | None = None,
    bins: int = 50,
    title: str = "Score Distribution",
) -> Figure:
    """Plot ID vs OOD score histogram."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.get_figure()

    ax.hist(
        id_scores,
        bins=bins,
        alpha=0.6,
        density=True,
        label="In-Distribution",
    )
    ax.hist(
        ood_scores,
        bins=bins,
        alpha=0.6,
        density=True,
        label="Out-of-Distribution",
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper center")
    ax.grid(True, linestyle="--", alpha=0.5)

    return fig