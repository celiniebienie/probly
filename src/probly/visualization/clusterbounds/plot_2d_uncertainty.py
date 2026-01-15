"""Plots the probabilities of uncertainty clusters."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC


def check_shape(input_data: np.ndarray) -> np.ndarray:
    """Sanity check.

    Args:
    input_data: 3D numpy array with shape (n_clusters, 2, n_samples).

    Returns:
    input_data or Error Message.
    """
    msg_type = "Input must be a NumPy array."
    msg_empty = "Input must not be empty."
    msg_ndim = "Input must be a 3D array."
    msg_middle = "Middle dimension must be 2 (representing two classes)."
    msg_values = "All probabilities must be between 0 and 1."

    if not isinstance(input_data, np.ndarray):
        raise TypeError(msg_type)
    if input_data.size == 0:
        raise ValueError(msg_empty)
    if input_data.ndim != 3:
        raise ValueError(msg_ndim)

    if input_data.shape[1] != 2:
        raise ValueError(msg_middle)
    if (input_data < 0).any() or (input_data > 1).any():
        raise ValueError(msg_values)

    return input_data


def _reshape_data_for_x(input_data: np.ndarray) -> np.ndarray:
    """Reshapes clusters into two dimensions.

    Args:
        input_data: 3D numpy array with shape (n_clusters, 2, n_samples).
    """
    X = input_data.transpose(0, 2, 1)  # noqa: N806
    X = X.reshape(-1, 2)  # noqa: N806
    return X


def _reshape_data_for_y(input_data: np.ndarray) -> np.ndarray:
    n_classes = input_data.shape[0]
    n_samples = input_data.shape[2]
    y = []
    for i in range(n_classes):
        y.extend([i] * n_samples)
    y = np.array(y)
    return y


def plot_2d_uncertainty(
    clusters: np.ndarray,
    ax: plt.Axes = None,
    cmap_name: str = "coolwarm",
) -> plt.Axes:
    """Plot 2D uncertainty for clusters.

    Args:
        clusters: 3D numpy array with shape (n_clusters, 2, n_samples).
        ax: Optional matplotlib Axes to plot on.
        cmap_name: Name of matplotlib colormap to use.

    Returns:
        Matplotlib Axes with the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    validated_data = check_shape(clusters)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.set_xlabel("Class A")
    ax.set_ylabel("Class B")

    ax.set_title("2D Uncertainty")

    n_clusters = validated_data.shape[0]

    cmap = plt.get_cmap(cmap_name)
    colors = cmap(np.linspace(0.4, 1, n_clusters))

    for i, (x, y) in enumerate(validated_data):
        ax.scatter(x, y, s=10, alpha=0.8, color=colors[i], label=f"Cluster {i}")

    X = _reshape_data_for_x(validated_data)  # noqa: N806
    y = _reshape_data_for_y(validated_data)

    clf = SVC(kernel="rbf", C=0.5, gamma="scale")
    clf.fit(X, y)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 400),
        np.linspace(y_min, y_max, 400),
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    Z = clf.predict(grid).reshape(xx.shape)  # noqa: N806

    ax.contourf(xx, yy, Z, alpha=0.4, levels=np.arange(n_clusters + 1) - 0.5, cmap=cmap)

    ax.legend(loc="upper right")
    plt.show()

    return ax
