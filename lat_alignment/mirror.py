"""
Functions to describe the mirror surface.
"""

import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property
from typing import Optional

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from megham.transform import (
    apply_transform,
    decompose_affine,
    decompose_rotation,
    get_affine,
    get_rigid,
)
from numpy.typing import NDArray
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

logger = logging.getLogger("lat_alignment")

# fmt: off
a = {'primary' : 
        np.array([
            [0., 0., -57.74022, 1.5373825, 1.154294, -0.441762, 0.0906601,],
            [0., 0., 0., 0., 0., 0., 0.],
            [-72.17349, 1.8691899, 2.8859421, -1.026471, 0.2610568, 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [1.8083973, -0.603195, 0.2177414, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0.],
            [0.0394559, 0., 0., 0., 0., 0., 0.,]
        ]),
     'secondary' : 
        np.array([
            [0., 0., 103.90461, 6.6513025, 2.8405781, -0.7819705, -0.0400483,  0.0896645],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [115.44758, 7.3024355, 5.7640389, -1.578144, -0.0354326, 0.2781226, 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [2.9130983, -0.8104051, -0.0185283, 0.2626023, 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0.],
            [-0.0250794, 0.0709672, 0., 0., 0., 0., 0., 0.,],
            [0., 0., 0., 0., 0., 0., 0., 0.]
        ])
     }
# fmt: on


def mirror_surface(
    x: NDArray[np.float32], y: NDArray[np.float32], a: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Analytic form of the mirror surface.

    Parameters
    ----------
    x : NDArray[np.float32]
        X positions to calculate at in mm.
    y : NDArray[np.float32]
        Y positions to calculate at in mm.
        Should have the same shape as `x`.
    a : NDArray[np.float32]
        Coeffecients of the mirror function.
        Use `a_primary` for the primary mirror.
        Use `a_secondary` for the secondary mirror.

    Returns
    -------
    z : NDArray[np.float32]
        Z position of the mirror at each input coordinate.
        Has the same shape as `x`.
    """
    z = np.zeros_like(x)
    Rn = 3000.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            z += a[i, j] * (x / Rn) ** i * (y / Rn) ** j
    return z


def mirror_norm(
    x: NDArray[np.float32], y: NDArray[np.float32], a: NDArray[np.float32]
) -> NDArray[np.float32]:
    """
    Analytic form of the vector normal to the mirror surface.

    Parameters
    ----------
    x : NDArray[np.float32]
        X positions to calculate at in mm.
    y : NDArray[np.float32]
        Y positions to calculate at in mm.
        Should have the same shape as `x`.
    a : NDArray[np.float32]
        Coeffecients of the mirror function.
        Use `a_primary` for the primary mirror.
        Use `a_secondary` for the secondary mirror.

    Returns
    -------
    normals : NDArray[np.float32]
        Unit vector normal to the mirror surface at each input coordinate.
        Has shape `shape(x) + (3,)`.
    """
    Rn = 3000.0

    x_n = np.zeros_like(x)
    y_n = np.zeros_like(y)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i != 0:
                x_n += a[i, j] * (x ** (i - 1)) / (Rn**i) * (y / Rn) ** j
            if j != 0:
                y_n += a[i, j] * (x / Rn) ** i * (y ** (j - 1)) / (Rn**j)

    z_n = -1 * np.ones_like(x_n)
    normals = np.array((x_n, y_n, z_n)).T
    normals /= np.linalg.norm(normals, axis=-1)[:, np.newaxis]
    return normals


@dataclass
class Panel:
    """
    Dataclass for storing a mirror panel.

    Attributes
    ----------
    mirror : str
        Which mirror this panel is for.
        Should be 'primary' or 'secondary'.
    row : int
        The row of the panel.
    col : int
        The column of the panel.
    corners : NDArray[np.float32]
        Array of panel corners.
        Should have shape `(4, 3)`.
    measurements : NDArray[np.float32]
        The measurement data for this panel.
        Should be in the mirror's internal coords.
        Should have shape `(npoint, 3)`.
    nom_adj : NDArray[np.float32]
        The nominal position of the adjusters in the mirror internal coordinates.
        Should have shape `(5, 3)`.
    compensate : float, default: 0
        The amount (in mm) to compensate the model surface by.
        This is to account for things like the Faro SMR.
    """

    mirror: str
    row: int
    col: int
    corners: NDArray[np.float32]
    measurements: NDArray[np.float32]
    nom_adj: NDArray[np.float32]
    compensate: float = field(default=0.0)
    adjuster_radius: float = field(default=50.0)

    def __post_init__(self):
        self.measurements = np.atleast_2d(self.measurements)

    def __setattr__(self, name, value):
        if (
            name == "nom_adj"
            or name == "mirror"
            or name == "measurements"
            or name == "compensate"
        ):
            self.__dict__.pop("can_surface", None)
            self.__dict__.pop("model", None)
            self.__dict__.pop("residuals", None)
            self.__dict__.pop("transformed_residuals", None)
            self.__dict__.pop("res_norm", None)
            self.__dict__.pop("rms", None)
            self.__dict__.pop("meas_surface", None)
            self.__dict__.pop("meas_adj", None)
            self.__dict__.pop("meas_adj_resid", None)
            self.__dict__.pop("model_transformed", None)
            self.__dict__.pop("_transform", None)
        elif name == "adjuster_radius":
            self.__dict__.pop("meas_adj_resid", None)
        return super().__setattr__(name, value)

    @cached_property
    def model(self):
        """
        The modeled mirror surface at the locations of the measurementss.
        """
        model = self.measurements.copy()
        model[:, 2] = mirror_surface(model[:, 0], model[:, 1], a[self.mirror])
        if self.compensate != 0.0:
            compensation = self.compensate * mirror_norm(
                model[:, 0], model[:0], a[self.mirror]
            )
            model += compensation
        return model

    @cached_property
    def _transform(self):
        return get_rigid(self.model, self.measurements, center_dst=True, method="mean")

    @property
    def rot(self):
        """
        Rotation that aligns the model to the measurements.
        """
        return self._transform[0]

    @property
    def shift(self):
        """
        Shift that aligns the model to the measurements.
        """
        return self._transform[1]

    @cached_property
    def can_surface(self):
        """
        Get the cannonical points to define the panel surface.
        These are the adjuster positions projected only the mirror surface.
        Note that this is in the nominal coordinates not the measured ones.
        """
        can_z = mirror_surface(self.nom_adj[:, 0], self.nom_adj[:, 1], a[self.mirror])
        points = self.nom_adj.copy()
        points[:, 2] = can_z
        return points

    @cached_property
    def meas_surface(self):
        """
        The cannonical surface transformed to be in the measured coordinates.
        """
        return apply_transform(self.can_surface, self.rot, self.shift)

    @cached_property
    def meas_adj(self):
        """
        The adjuster points transformed to be in the measured coordinates.
        """
        return apply_transform(self.nom_adj, self.rot, self.shift)

    @cached_property
    def meas_adj_resid(self):
        """
        A correction that can be applied to `meas_adj` where we compute
        the average residual of measured points from the transformed model
        that are within `adjuster_radius` of the adjuster point in `xy`.
        """
        resid = np.zeros(len(self.meas_adj))
        for i, adj in enumerate(self.meas_adj):
            dists = np.linalg.norm(self.measurements[:, :2] - adj[:2], axis=-1)
            msk = dists <= self.adjuster_radius
            if np.sum(msk) == 0:
                continue
            resid[i] = np.mean(self.transformed_residuals[msk, 2])

        return resid

    @cached_property
    def model_transformed(self):
        """
        The model transformed to be in the measured coordinates.
        """
        return apply_transform(self.model, self.rot, self.shift)

    @cached_property
    def residuals(self):
        """
        Get residuals between model and measurements.
        """
        return self.measurements - self.model

    @cached_property
    def transformed_residuals(self):
        """
        Get residuals between transformed model and measurements.
        """
        return self.measurements - self.model_transformed

    @cached_property
    def res_norm(self):
        """
        Get norm of residuals between transformed model and measurements.
        """
        return np.linalg.norm(self.residuals, axis=-1)

    @cached_property
    def rms(self):
        """
        Get rms between model and measurements.
        """
        return np.sqrt(np.mean(self.residuals[:, 2].ravel() ** 2))


def gen_panels(
    mirror: str,
    measurements: dict[str, NDArray[np.float32]],
    corners: dict[tuple[int, int], NDArray[np.float32]],
    adjusters: dict[tuple[int, int], NDArray[np.float32]],
    compensate: float = 0.0,
    adjuster_radius: float = 50.0,
) -> list[Panel]:
    """
    Use a set of measurements to generate panel objects.

    Parameters
    ----------
    mirror : str
        The mirror these panels belong to.
        Should be 'primary' or 'secondary'.
    measurements : dict[str, NDArray[np.float32]]
        The photogrammetry data.
        Dict is data indexed by the target names.
    corners : dict[tuple[int, int], ndarray[np.float32]]
        The corners. This is indexed by a (row, col) tuple.
        Each entry is `(4, 3)` array where each row is a corner.
    adjusters : dict[tuple[int, int], NDArray[np.float32]]
        Nominal adjuster locations.
        This is indexed by a (row, col) tuple.
        Each entry is `(5, 3)` array where each row is an adjuster.
    compensate : float, default: 0.0
        Amount (in mm) to compensate the model surface by.
        This is to account for things like the faro SMR.
    adjuster_radius : float, default: 50.0
        The radius in XY of points that an adjuster should use to
        compute a secondary correction on its position.
        Should be in mm.

    Returns
    -------
    panels : list[Panels]
        A list of panels with the transforme initialized to the identity.
    """
    points = defaultdict(list)
    # dumb brute force
    corr = np.arange(4, dtype=int)
    for _, point in measurements.items():
        for rc, crns in corners.items():
            x = crns[:, 0] > point[0]
            y = crns[:, 1] > point[1]
            val = x.astype(int) + 2 * y.astype(int)
            if np.array_equal(np.sort(val), corr):
                points[rc] += [point]
                break

    # Now init the objects
    panels = []
    for (row, col), meas in points.items():
        meas = np.vstack(meas, dtype=np.float32)
        panel = Panel(
            mirror,
            row,
            col,
            corners[(row, col)],
            meas,
            adjusters[(row, col)],
            compensate,
            adjuster_radius,
        )
        panels += [panel]
    return panels


def remove_cm(
    meas,
    mirror,
    compensate: float = 0,
    thresh: float = 10,
    cut_thresh: float = 50,
    niters: int = 10,
) -> tuple[
    dict[str, NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32]]
]:
    """
    Fit for the common mode transformation from the model to the measurements of all panels and them remove it.

    Parameters
    ----------
    meas : dict[str, NDArray[np.float32]]
        The photogrammetry data.
        Dict is data indexed by the target names.
    mirror : str
        The mirror this data belong to.
        Should be 'primary' or 'secondary'.
    compensate : float, default: 0
        Compensation to apply to model.
        This is to account for the radius of a Faro SMR.
    thresh : float, default: 10
        How many times higher than the median residual a point needs to have to be
        considered an outlier.
    niters : int, default: 10
        How many iterations of common mode fitting to do.

    Returns
    -------
    kept_points: dict[str, NDArray[np.float32]]
        The points that were successfully fit.
    common_mode : tuple[NDArray[np.float32], NDArray[np.float32]]
        The common mode that was removed.
        The first element is an affine matrix and
        the second is the shift.
    """
    logger.info("Removing common mode for %s", mirror)

    def _cm(x, panel):
        panel.measurements[:] -= x[1:4]
        rot = Rotation.from_euler("xyz", x[4:])
        panel.measurements = rot.apply(panel.measurements)
        panel.measurements *= x[0]

    def _opt(x, panel):
        p2 = deepcopy(panel)
        _cm(x, p2)
        return p2.rms

    # make a fake panel for the full mirror
    corners = np.array(
        ([-3300, -3300, 0], [-3300, 3300, 0], [3300, 3300, 0], [3300, -3300, 0])
    )  # ack hardcoded
    labels = np.array(list(meas.keys()))
    data = np.array(list(meas.values()))
    corr = np.arange(4, dtype=int)
    x = np.vstack([corners[:, 0] > dat[0] for dat in data])
    y = np.vstack([corners[:, 1] > dat[1] for dat in data])
    val = x.astype(int) + 2 * y.astype(int)
    val = np.sort(val, axis=-1)
    msk = (val == corr).all(-1)
    data = data[msk]
    labels = labels[msk]
    panel = Panel(
        mirror,
        -1,
        -1,
        np.zeros((4, 3), "float32"),
        data,
        np.zeros((5, 3), "float32"),
        compensate,
    )
    data = data.copy()
    data_clean = data.copy()
    logger.info("\tRemoved %d points not on mirror surface", np.sum(~msk))

    x0 = np.hstack([np.ones(1), np.zeros(6)])
    bounds = [(-0.95, 1.05)] + [(-100, 100)] * 3 + [(0, 2 * np.pi)] * 3

    for i in range(niters):
        if len(panel.measurements) < 3:
            raise ValueError
        logger.debug("\titer %d for common mode fit", i)
        cut = panel.res_norm > thresh * np.median(panel.res_norm)
        if np.sum(cut) > 0:
            panel.measurements = panel.measurements[~cut]
            data = data[~cut]

        logger.debug("\t\tRemoving a naive common mode shift of %s", str(panel.shift))
        panel.measurements -= panel.shift
        panel.measurements @= panel.rot.T

        res = minimize(_opt, x0, (panel,), bounds=bounds)
        logger.debug(
            "\t\tRemoving a fit common mode with scale %f, shift %s, and rotation %s",
            res.x[0],
            str(res.x[1:4]),
            str(res.x[4:]),
        )
        _cm(res.x, panel)

        logger.debug(
            "\t\tRemoving a secondary common mode shift of %s and rotation of %s",
            str(panel.shift),
            str(np.rad2deg(decompose_rotation(panel.rot))),
        )
        panel.measurements -= panel.shift
        panel.measurements @= panel.rot.T

    aff, sft = get_affine(
        data, panel.measurements, method="mean", weights=np.ones(len(data))
    )
    scale, shear, rot = decompose_affine(aff)
    rot = decompose_rotation(rot)
    logger.info(
        "\tFull common mode is:\n\t\t\tshift = %s mm\n\t\t\tscale = %s\n\t\t\tshear = %s\n\t\t\trot = %s deg",
        str(sft),
        str(scale),
        str(shear),
        str(np.rad2deg(rot)),
    )

    panel.measurements = apply_transform(data_clean, aff, sft)
    cut = panel.res_norm > cut_thresh * np.median(panel.res_norm)
    if np.sum(cut) > 0:
        logger.info("\tRemoving %d bad points from mirror", np.sum(cut))
        panel.measurements = panel.measurements[~cut]
    logger.info("\tMirror has %d good points", len(panel.measurements))

    return {l: d for l, d in zip(labels, panel.measurements)}, (aff, sft)


def plot_panels(
    panels: list[Panel], title_str: str, vmax: Optional[float] = None
) -> Figure:
    """
    Make a plot containing panel residuals and histogram.
    TODO: Correlation?

    Parameters
    ----------
    panels : list[Panel]
        The panels to plot.
    title_str : str
        The title string, rms will me appended.
    vmax : Optional[float], default: None
        The max of the colorbar. vmin will be -1 times this.
        Set to None to compute automatically.
        Should be in um.

    Returns
    -------
    figure : Figure
        The figure with panels plotted on it.
    """
    res_all = np.vstack([panel.residuals for panel in panels]) * 1000
    model_all = np.vstack([panel.model for panel in panels])
    if vmax is None:
        vmax = np.max(np.abs(res_all[:, 2]))
    if vmax is None:
        raise ValueError("vmax still None?")
    gs = gridspec.GridSpec(2, 2, width_ratios=[20, 1], height_ratios=[2, 1])
    fig = plt.figure()
    ax0 = plt.subplot(gs[0])
    cax = plt.subplot(gs[1])
    ax1 = plt.subplot(gs[2:])
    cb = None
    for panel in panels:
        ax0.tricontourf(
            panel.model[:, 0],
            panel.model[:, 1],
            panel.residuals[:, 2] * 1000,
            vmin=-1 * vmax,
            vmax=vmax,
            cmap="coolwarm",
            alpha=0.6,
        )
        cb = ax0.scatter(
            panel.model[:, 0],
            panel.model[:, 1],
            s=40,
            c=panel.residuals[:, 2] * 1000,
            vmin=-1 * vmax,
            vmax=vmax,
            cmap="coolwarm",
            marker="o",
            alpha=0.9,
            linewidth=2,
            edgecolor="black",
        )
        ax0.scatter(
            panel.meas_adj[:, 0],
            panel.meas_adj[:, 1],
            marker="x",
            linewidth=1,
            color="black",
        )
    ax0.tricontourf(
        model_all[:, 0],
        model_all[:, 1],
        res_all[:, 2],
        vmin=-1 * vmax,
        vmax=vmax,
        cmap="coolwarm",
        alpha=0.2,
    )
    ax0.set_xlabel("x (mm)")
    ax0.set_ylabel("y (mm)")
    ax0.set_xlim(-3300, 3300)  # ack hardcoded!
    ax0.set_ylim(-3300, 3300)
    if cb is not None:
        fig.colorbar(cb, cax)
    ax0.set_aspect("equal")
    for panel in panels:
        ax0.add_patch(
            Polygon(panel.corners[[0, 1, 3, 2], :2], fill=False, color="black")
        )

    ax1.hist(res_all[:, 2], bins=len(panels))
    ax1.set_xlabel("z residual (um)")

    points = np.array([len(panel.measurements) for panel in panels])
    rms = np.array([panel.rms for panel in panels])
    tot_rms = 1000 * np.sum(rms * points) / np.sum(points)
    fig.suptitle(f"{title_str}, RMS={tot_rms:.2f} um")

    plt.show()

    return fig
