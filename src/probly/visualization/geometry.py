"""Geometry-related visualization utilities for credal sets."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


class CredalVisualizer:
    """Class to collect all the geometric plots."""

    def __init__(self) -> None:  # noqa: D107
        pass

    def probs_to_coords_2d(self, probs: np.ndarray) -> tuple:
        """Convert 2D probabilities to 2D coordinates.

        Args:
        probs: probability vector for 2 classes.

        returns: cartesian coordinates.
        """
        p1, p2 = probs
        x = p2
        y = 0
        return x, y

    def interval_plot(
        self,
        probs: np.ndarray,
        labels: list[str] | None = None,
        ax: plt.Axes = None,
    ) -> plt.Axes:
        """Plot the interval plot.

        Args:
        probs: probability vector for 2 classes.
        labels: labels for the interval plot.
        ax: matplotlib axes.Axes.

        returns: plot.
        """
        coords = np.array([self.probs_to_coords_2d(p) for p in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 1))

        y_marg = np.array([0.1, -0.1])

        plt.plot([0, 1], [0, 0], color="black", linewidth=4, zorder=0)

        coord_max = np.max(coords[:, 0])
        coord_min = np.min(coords[:, 0])
        ax.fill_betweenx(y_marg, coord_max, coord_min, color="purple", alpha=0.5, zorder=2)

        ax.scatter(coords[:, 0], coords[:, 1], color="green", zorder=1)

        ax.axis("off")
        ax.set_ylim((-0.2, 0.2))

        y_anchor = -0.07
        x_beg = 0
        x_mid = 0.5
        x_end = 1

        n_classes = probs.shape[-1]

        if labels is None:
            labels = [f"C{i + 1}" for i in range(n_classes)]

        if len(labels) != n_classes:
            msg = f"Number of labels ({len(labels)}) must match number of classes ({n_classes})."
            raise ValueError(msg)

        ax.text(x_beg, y_anchor, "0 ", ha="center", va="top")
        ax.text(x_mid, y_anchor, "0.5", ha="center", va="top")
        ax.text(x_end, y_anchor, "1 ", ha="center", va="top")
        ax.text(x_beg, y_anchor - 0.07, f"{labels[0]}", ha="center", va="top")
        ax.text(x_end, y_anchor - 0.07, f"{labels[1]}", ha="center", va="top")

        return ax


points_2d = np.array(
    [
        [0.2, 0.8],
        [0.5, 0.5],
        [0.1, 0.9],
    ],
)

viz = CredalVisualizer()
ax = viz.interval_plot(points_2d)  # 1 row, 2 columns
plt.show()
