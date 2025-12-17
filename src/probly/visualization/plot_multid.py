"""Geometry-related visualization utilities for credal sets."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from matplotlib.patches import Circle, Patch, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D


def radar_factory(num_vars: int, frame: str = "polygon") -> np.ndarray:  # noqa: C901
    """Create radar chart angles and register a custom radar projection."""
    theta = np.linspace(0.0, 2.0 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path: Path) -> Path:
            """Interpolate paths so gridlines become straight in radar coordinates."""
            if path._interpolation_steps > 1:  # noqa: SLF001
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = "radar"
        PolarTransform = RadarTransform

        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location("N")

        def fill(
            self,
            *args: object,
            closed: bool = True,
            **kwargs: object,
        ) -> list[Patch]:
            patches = super().fill(*args, closed=closed, **kwargs)
            return cast("list[Patch]", patches)

        def plot(
            self,
            *args: object,
            **kwargs: object,
        ) -> list[Line2D]:
            lines = cast("list[Line2D]", super().plot(*args, **kwargs))
            for line in lines:
                self._close_line(line)
            return lines

        def _close_line(self, line: Line2D) -> None:
            x, y = line.get_data()
            if x[0] != x[-1]:
                line.set_data(np.append(x, x[0]), np.append(y, y[0]))

        def set_varlabels(self, labels: list[str]) -> None:
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self) -> Patch:
            if frame == "polygon":
                return RegularPolygon((0.5, 0.5), num_vars, radius=0.5)
            return Circle((0.5, 0.5), 0.5)

        def _gen_axes_spines(self) -> dict[str, Spine]:
            if frame == "polygon":
                spine = Spine(
                    axes=self,
                    spine_type="circle",
                    path=Path.unit_regular_polygon(num_vars),
                )
                spine.set_transform(
                    Affine2D().scale(0.5).translate(0.5, 0.5) + self.transAxes,
                )
                return {"polar": spine}
            spines = super()._gen_axes_spines()
            return cast("dict[str, Spine]", spines)

    register_projection(RadarAxes)

    return np.append(theta, theta[0])


class CredalVisualizer:
    """Collection of geometric plots for credal predictions."""

    def spider_plot(
        self,
        lower: np.ndarray | None,
        upper: np.ndarray | None,
        mle: np.ndarray | None,
        labels: list[str],
        title: str = "Credal Prediction",
        rmax: float = 1.0,
        ax: Axes | None = None,
    ) -> Axes:
        num_classes = len(labels)

        if mle is None and (lower is None or upper is None):
            msg = "Either 'mle' or both 'lower' and 'upper' must be provided."
            raise ValueError(msg)

        if mle is not None and len(mle) != num_classes:
            msg = "mle must have the same length as labels."
            raise ValueError(msg)

        if lower is not None and upper is not None and (
            len(lower) != num_classes or len(upper) != num_classes
        ):
            msg = "lower and upper must have the same length as labels."
            raise ValueError(msg)

        theta = radar_factory(num_classes)

        if ax is None:
            _, ax = plt.subplots(
                figsize=(6, 6),
                subplot_kw={"projection": "radar"},
            )

        ax.set_rgrids(np.linspace(0.0, rmax, 6)[1:])
        ax.set_ylim(0.0, rmax)
        ax.set_varlabels(labels)

        if lower is not None and upper is not None:
            lower_c = np.append(lower, lower[0])
            upper_c = np.append(upper, upper[0])

            ax.fill_between(
                theta,
                lower_c,
                upper_c,
                alpha=0.30,
                label="Credal band (lower–upper)",
            )

            ax.plot(theta, lower_c, linestyle="--", linewidth=1.5, label="Lower bound")
            ax.plot(theta, upper_c, linestyle="-", linewidth=1.5, label="Upper bound")

        if mle is not None:
            idx = int(np.argmax(mle))
            ax.scatter(
                [theta[idx]],
                [float(mle[idx])],
                s=80,
                color="red",
                label="MLE (argmax)",
                zorder=5,
            )

        ax.set_title(title, pad=20)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
        return ax