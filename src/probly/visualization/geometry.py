from __future__ import annotations  # noqa: D100

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull


class CredalVisualizer:
    """Class to collect all the geometric plots."""

    def __init__(self) -> None:  # noqa: D107
        pass

    def probs_to_coords_3D(self, probs: np.ndarray) -> tuple:
        """Convert ternary probabilities to 2D coordinates."""
        p1, p2, p3 = probs  # noqa: RUF059
        x = p2 + 0.5 * p3
        y = (np.sqrt(3) / 2) * p3
        return x, y

    def probs_to_coords_2D(self, probs: np.ndarray) -> tuple:
        """Convert 2D probabilities to 2D coordinates."""
        p1, p2 = probs
        x = p2
        y = 0
        return x, y

    def ternary_plot(
        self,
        probs: np.ndarray,
        ax: mpl.axes.Axes = None,
        **scatter_kwargs: mpl.Kwargs,
    ) -> mpl.axes.Axes:
        """Plot ternary scatter points."""
        msg = "Input must have 3 dimensions."
        if probs.shape[1] != 3:
            raise ValueError(msg)

        coords = np.array([self.probs_to_coords_3D(p) for p in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # noqa: RUF059

        # Draw triangle
        verts = np.array([
            [0, 0],
            [1, 0],
            [0.5, np.sqrt(3) / 2],
        ])
        triangle = plt.Polygon(verts, closed=True, fill=False)
        ax.add_patch(triangle)

        # Scatter points
        ax.scatter(coords[:, 0], coords[:, 1], **scatter_kwargs)

        return ax


    def interval_plot(
        self,
        probs:np.ndarray,
        ax: mpl.axes.Axes = None,
        **scatter_kwargs: mpl.Kwargs
    ) -> mpl.axes.Axes:

        coords = np.array([self.probs_to_coords_2D(p) for p in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 1))

        y_marg = np.array([0.1, -0.1])

        plt.plot([0,1], [0,0], color = 'black', linewidth = 4, zorder = 0)

        coord_max = np.max(coords[:,0])
        coord_min = np.min(coords[:,0])
        ax.fill_betweenx(y_marg, coord_max, coord_min, color = 'purple', alpha = 0.5, zorder = 2)

        ax.scatter(coords[:, 0], coords[:, 1], color = 'green', zorder = 1)


        ax.axis('off')
        ax.set_ylim((-0.2, 0.2))

        y_anchor = -0.07
        x_beg = 0
        x_mid = 0.5
        x_end = 1

        ax.text(x_beg, y_anchor, '0 ', ha='center', va='top')
        ax.text(x_mid, y_anchor, '0.5', ha='center', va='top')
        ax.text(x_end, y_anchor, '1 ', ha='center', va='top')
        ax.text(x_beg, y_anchor -0.07, 'Class A', ha='center', va='top')
        ax.text(x_end, y_anchor -0.07, 'Class B', ha='center', va='top')

        return ax


    def plot_convex_hull(
        self,
        probs: np.ndarray,
        ax: mpl.axes.Axes = None,
        facecolor: str = "lightgreen",
        alpha: float = 0.4,
        edgecolor: str = "green",
        linewidth: float = 2.0,
    ) -> mpl.axes.Axes:
        """Draw the convex hull around ternary points.
        Handles special cases:
        - 1 point (degenerate)
        - 2 points (line segment)
        - ≥3 points (polygon).
        """  # noqa: D205
        coords = np.array([self.probs_to_coords_3D(p) for p in probs])

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))  # noqa: RUF059

        # Handle degenerate cases
        unique = np.unique(coords, axis=0)

        if len(unique) == 1:
            # Single point — no hull possible
            ax.scatter(unique[:, 0], unique[:, 1], color="green", s=80)
            return ax

        if len(unique) == 2:
            # Two distinct points — hull is a line segment
            ax.plot(unique[:, 0], unique[:, 1], color=edgecolor, linewidth=linewidth)
            return ax

        # Try to compute convex hull
        try:
            hull = ConvexHull(coords)

            hull_pts = coords[hull.vertices]

            poly = plt.Polygon(
                hull_pts,
                closed=True,
                facecolor=facecolor,
                edgecolor=edgecolor,
                alpha=alpha,
                linewidth=linewidth,
            )
            ax.add_patch(poly)

        except Exception:  # noqa: BLE001
            # Remaining degeneracy: 3+ collinear points
            # Get endpoints via projection on PCA / longest distance
            dists = np.linalg.norm(coords[:, None] - coords[None, :], axis=-1)
            i, j = np.unravel_index(np.argmax(dists), dists.shape)
            ax.plot(
                [coords[i, 0], coords[j, 0]],
                [coords[i, 1], coords[j, 1]],
                color=edgecolor,
                linewidth=linewidth,
            )

        return ax


points_2D = np.array(
    [
        [0.2, 0.8],
        [0.5, 0.5],
        [0.1, 0.9]
    ]
)


points = np.array(
    [
        [0.7, 0.2, 0.1],
        [0.4, 0.3, 0.3],
        [0.1, 0.8, 0.1],
        [0.8, 0.1, 0.1],
        [0.3, 0.1, 0.6],
        [0.33, 0.33, 0.34],
    ],
)

viz = CredalVisualizer()
fig, axes = plt.subplots(2, 1, figsize=(6, 12))  # 1 row, 2 columns

viz.ternary_plot(points, ax=axes[0], color="blue", s=50)
viz.plot_convex_hull(points, ax=axes[0])


viz.interval_plot(points_2D, ax=axes[1])

plt.show()
