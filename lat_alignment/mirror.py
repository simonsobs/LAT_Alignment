"""
Functions to describe the mirror surface.
"""
from dataclasses import dataclass
from functools import cached_property
import numpy as np
from numpy.typing import NDArray
from numpy import ndarray
from megham.transform import apply_transform, get_rigid
from collections import defaultdict
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.figure import Figure

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


def mirror_surface(x: ndarray, y: ndarray, a: ndarray) -> ndarray:
    """
    Analytic form for the mirror

    @param x: x positions to calculate at
    @param y: y positions to calculate at
    @param a: Coeffecients of the mirror function
              Use a_primary for the primary mirror
              Use a_secondary for the secondary mirror

    @return z: z position of the mirror at each xy
    """
    z = np.zeros_like(x)
    Rn = 3000.0
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            z += a[i, j] * (x / Rn) ** i * (y / Rn) ** j
    return z


def mirror_norm(x: ndarray, y: ndarray, a: ndarray) -> ndarray:
    """
    Analytic form of mirror normal vector

    @param x: x positions to calculate at
    @param y: y positions to calculate at
    @param a: Coeffecients of the mirror function
              Use a_primary for the primary mirror
              Use a_secondary for the secondary mirror

    @return normals: Unit vector normal to mirror at each xy
    """
    Rn = 3000.0

    x_n = np.zeros_like(x)
    y_n = np.zeros_like(y)
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i != 0:
                x_n += a[i, j] * (x ** (i - 1)) / (Rn ** i) * (y / Rn) ** j
            if j != 0:
                y_n += a[i, j] * (x / Rn) ** i * (y ** (j - 1)) / (Rn ** j)

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
    measurments : NDArray[np.float32]
        The measurment data for this panel.
        Should be in the mirror's internal coords.
        Should have shape `(npoint, 3)`.
    model : NDArray[np.float32]
        The modeled panelfor this panel.
        This is in the nominal coordinates not the measured ones.
        Should have same shape as `measurments`.
    nom_adj : NDArray[np.float32]
        The nominal position of the adjusters in the mirror internal coordinates.
        Should have shape `(5, 3)`.
    rot : NDArray[np.float32]
        Rotation matrix that takes the mirror model to the measurments.
        Should have shape `(3, 3)`.
    shift : NDArray[np.float32]
        Shift that takes the mirror model to the measurments.
        Should have shape `(3,)`.
    """
    mirror : str
    row : int
    col : int
    corners : NDArray[np.float32]
    measurments : NDArray[np.float32]
    model: NDArray[np.float32]
    nom_adj : NDArray[np.float32]
    rot : NDArray[np.float32]
    shift : NDArray[np.float32]

    def __set_attr__(self, name, value):
        if name == "rot" or name == "shift":
            self.__dict__.pop("meas_surface", None)
            self.__dict__.pop("meas_adj", None)
            self.__dict__.pop("model_transformed", None)
            self.__dict__.pop("residuals", None)
            self.__dict__.pop("rms", None)
        elif name == "nom_adj" or name == "mirror":
            self.__dict__.pop("can_surface", None)
            self.__dict__.pop("meas_surface", None)
            self.__dict__.pop("meas_adj", None)
        elif name == "model":
            self.__dict__.pop("residuals", None)
            self.__dict__.pop("rms", None)
        return super().__setattr__(name, value)

    @cached_property
    def can_surface(self):
        """
        Get the cannonical points to define the panel surface.
        These are the adjuster positions projected only the mirror surface.
        Note that this is in the nominal coordinates not the measured ones.
        """
        can_z = mirror_surface(self.nom_adj[:, 0], self.nom_adj[:, 1], a[self.mirror])
        points = can_z.copy()
        points[:, 2] = can_z
        return points

    @cached_property
    def meas_surface(self):
        """
        Transform the cannonical points to be in the measured coordinates.
        """
        return apply_transform(self.rot, self.shift, self.can_surface)
    
    @cached_property
    def meas_adj(self):
        """
        Transform the adjuster points to be in the measured coordinates.
        """
        return apply_transform(self.rot, self.shift, self.nom_adj)
    
    @cached_property
    def model_transformed(self):
        """
        Transform the model points to be in the measured coordinates.
        """
        return apply_transform(self.rot, self.shift, self.model)

    @cached_property
    def residuals(self):
        """
        Get residuals between transformed model and measurments.
        """
        return self.measurments - self.model_transformed

    @cached_property
    def res_norm(self):
        """
        Get norm of residuals between transformed model and measurments.
        """
        return np.linalg.norm(self.residuals) 

    @cached_property
    def rms(self):
        """
        Get rms between transformed model and measurments.
        """
        return np.sqrt(np.mean(self.res_norm ** 2))
    

def gen_panels(mirror: str, measurments: dict[str, NDArray[np.float32]], corners: dict[tuple[int, int], NDArray[np.float32]], adjusters: dict[tuple[int, int], NDArray[np.float32]]) -> list[Panel]:
    """
    Use a set of measurments to generate panel objects.

    Parameters
    ----------
    mirror : str
        The mirror these panels belong to.
        Should be 'primary' or 'secondary'.
    measurments : dict[str, NDArray[np.float32]]
        The photogrammetry data.
        Dict is data indexed by the target names.
    corners : dict[tuple[int, int], ndarray[np.float32]]
        The corners. This is indexed by a (row, col) tuple.
        Each entry is `(4, 3)` array where each row is a corner.
    adjusters : dict[tuple[int, int], NDArray[np.float32]]
        Nominal adjuster locations.
        This is indexed by a (row, col) tuple.
        Each entry is `(5, 3)` array where each row is an adjuster.

    Returns
    -------
    panels : list[Panels]
        A list of panels with the transforme initialized to the identity.
    """
    rot = np.eye(3, dtype=np.float32)
    shift = np.zeros(3, dtype=np.float32)

    points = defaultdict(list)
    # dumb brute force
    corr = np.arange(4, dtype=int)
    for _, point in measurments.items():
        for rc, crns in corners.items():
            x = crns[:, 0] > point[:, 0]
            y = crns[:, 1] > point[:, 1]
            val = x.astype(int) + 2*y.astype(int)
            if np.array_equal(np.sort(val), corr):
                point[rc] += [point]
                break

    # Now init the objects
    panels = []
    for (row, col), meas in points:
        meas = np.vstack(meas, dtype=np.float32)
        panel = Panel(mirror, row, col, corners[(row, col)], meas, np.zeros_like(meas, dtype=np.float32), adjusters[(row, col)], rot, shift)
        panels += [panel]
    return panels

def panel_model(pars: NDArray[np.float32], panel: Panel, compensate: float) -> NDArray[np.float32]:
    """
    Model function to use when fitting panel transform.

    Parameters
    ----------
    pars : NDArray[np.float32]
        The fit pars.
        These are just a small shift to apply to the model inputs.
        Should have shape `(2,)`.
    panel : Panel
        The panel to be fit.
    compensate : float
        Compensation to apply to model.
        This is to account for the radius of a Faro SMR.

    Returns
    -------
    model : NDArray[np.float32]
        The modeled panel surface.
        Has the same shape as `panel.measurments`.
    """
    model = panel.measurments.copy()
    model[:, :2] += pars
    model[:, 2] = mirror_surface(model[:, 0], model[:, 1], a[panel.mirror])
    if compensate != 0.0:
        compensation = compensate * mirror_norm(model[:, 0], model[: 0], a[panel.mirror])
        model += compensation
    return model

def panel_objective(pars: NDArray[np.float32], panel: Panel, compensate: float) -> float:
    """
    Objective function to use when fitting panel transform.

    Parameters
    ----------
    pars : NDArray
        The fit pars.
        These are just a small shift to apply to the model inputs.
        Should have shape `(2,)`.
    panel : Panel
        The panel to be fit.
    compensate : float
        Compensation to apply to model.
        This is to account for the radius of a Faro SMR.

    Returns
    -------
    chisq : float
        The chi squared.
    """
    model = panel_model(pars, panel, compensate)
    rot, sft = get_rigid(model, panel.measurments)
    return np.sum((apply_transform(model, rot, sft) - panel.measurments)**2)

def fit_panels(panels: list[Panel], max_off: float = 5, compensate: float = 0, thresh: float = 1.5):
    """
    Fit for the transformation from the model to the measurments.

    Parameters
    ----------
    panels : list[Panel]
        Panels to fit, will be modified in place.
    max_off : float, default: 5
        The maximum offset allowed in the model coordinates.
        This is in mm.
    compensate : float, default: 0
        Compensation to apply to model.
        This is to account for the radius of a Faro SMR.
    thresh : float, default: 1.5
        How many times higher than the median residual a point needs to have to be
        considered an outlier.
    """
    x0 = np.zeros(2)
    bounds = [(-max_off, max_off)]*2
    for panel in panels:
        res = minimize(panel_objective, x0, (panel, compensate), bounds=bounds)
        panel.model = panel_model(res.x, panel, compensate)
        panel.rot, panel.shift = get_rigid(panel.model, panel.measurments)
        cut = panel.residuals < thresh*np.median(panel.residuals)
        if np.sum(cut) > 0: 
            print(f"Removing {np.sum(cut)} points from panel {panel.row}, {panel.col}")
            panel.measurments = panel.measurments[cut]
            res = minimize(panel_objective, x0, (panel, compensate), bounds=bounds)
            panel.model = panel_model(res.x, panel, compensate)
            panel.rot, panel.shift = get_rigid(panel.model, panel.measurments)


def plot_panels(panels: list[Panel], title_str: str) -> Figure:
    """
    Make a plot containing panel residuals and histogram.
    TODO: Correlation?

    Parameters
    ----------
    panels : list[Panel]
        The panels to plot.

    title_str : str
        The title string, rms will me appended.

    Returns
    -------
    figure : Figure
        The figure with panels plotted on it.
    """
    res_all = np.hstack([panel.residuals for panel in panels])*100
    min_res, max_res = np.min(res_all[:, 2]), np.max(res_all[:, 2])
    codes = [Path.MOVETO] + [Path.LINETO]*3 + [Path.CLOSEPOLY]
    fig, (ax0, ax1) = plt.subplots(1, 2, gridspec_kw={"width_ratios":[2,1]})
    cb = None
    for panel in panels:
        vertices = [(corner[0], corner[1]) for corner in panel.corners]
        path = Path(vertices, codes)
        pathpatch = PathPatch(path, facecolor='none', edgecolor='green')
        ax0.add_patch(pathpatch)
        cb = ax0.tricontourf(panel.model_transformed[:, 0], panel.model_transformed[:, 1], panel.residuals[:, 2], vmin=min_res, vmax=max_res)
    if cb is not None:
        fig.colorbar(cb, ax0)
    ax0.set_xlabel("x (mm)")
    ax0.set_ylabel("y (mm)")

    ax1.hist(res_all[:, 2], bins=len(panels))
    ax1.set_xlabel("z residual (um)")

    points = np.array([len(panel.measurments) for panel in panels])
    rms = np.array([len(panel.rms) for panel in panels])
    tot_rms = (np.sum(rms*points)/np.sum(points))*100
    fig.suptitle(f"{title_str}, RMS={tot_rms:.2f} um")

    return fig
