import numpy as np
from numpy.typing import NDArray
import yaml
from collections import defaultdict
from megham.utils import make_edm
import matplotlib.pyplot as plt

from .transforms import align_photo, coord_transform 

def load_photo(path: str, align: bool =True, err_thresh: float =2, plot: bool=True, **kwargs) -> dict[str, NDArray[np.float32]]:
    """
    Load photogrammetry data.
    Assuming first column is target names and next three are (x, y , z).

    Parameters
    ----------
    path : str
        The path to the photogrammetry data.
    align : bool, default: True
        If True align using the invar points.
    err_thresh : float, default: 2
        How many times the median photogrammetry error
        a target need to have to be cut.
    plot: bool, default: True
        If True display a scatter plot of targets.
    **kwargs
        Arguments to pass to `align_photo`.

    Returns
    -------
    data : dict[str, NDArray[np.float32]]
        The photogrammetry data.
        Dict is indexed by the target names.
    """
    labels = np.genfromtxt(path, dtype=str, delimiter=",", usecols=(0,))
    coords = np.genfromtxt(path, dtype=np.float32, delimiter=",", usecols=(1,2,3))
    errs = np.genfromtxt(path, dtype=np.float32, delimiter=",", usecols=(4,5,6))
    msk = (np.char.find(labels, "TARGET") >= 0) + (np.char.find(labels, "CODE") >= 0)

    labels, coords, errs = labels[msk], coords[msk], errs[msk]
    err = np.linalg.norm(errs, axis=-1)

    if align:
        labels, coords, msk = align_photo(labels, coords, **kwargs)
        err = err[msk]
    trg_msk = (np.char.find(labels, "TARGET") >= 0)
    labels = labels[trg_msk]
    coords = coords[trg_msk]
    err = err[trg_msk]

    err_msk = err < err_thresh*np.median(err)
    labels, coords, err = labels[err_msk], coords[err_msk], err[err_msk]

    # Lets find and remove doubles
    # Dumb brute force
    edm = make_edm(coords)
    np.fill_diagonal(edm, np.nan)
    to_kill = []
    for i in range(len(edm)):
        if i in to_kill:
            continue
        imin = np.nanargmin(edm[i])
        if edm[i][imin] > 10:
            continue
        if err[i] < err[imin]:
            to_kill += [imin]
        else:
            to_kill += [i]
    msk = ~np.isin(np.arange(len(coords), dtype=int), to_kill)
    labels, coords = labels[msk], coords[msk]

    if plot:
        plt.scatter(coords[:, 0], coords[:, 1], c=coords[:, 2], marker="x")
        plt.colorbar()
        plt.show()

    data = {label:coord for label, coord in zip(labels, coords)}
    return data

def load_corners(path: str) -> dict[tuple[int, int], NDArray[np.float32]]:
    """
    Get panel corners from file.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    corners : dict[tuple[int, int], ndarray[np.float32]]
        The corners. This is indexed by a (row, col) tuple.
        Each entry is `(4, 3)` array where each row is a corner.
    """
    with open(path) as file:
        corners_raw = yaml.safe_load(file)

    corners = {(panel[7], panel[9]):np.vstack([np.array(coord.split(), np.float32) for coord in coords]) for panel, coords in corners_raw.items()}
    return corners

def load_adjusters(path: str, mirror: str) -> dict[tuple[int, int], NDArray[np.float32]]:
    """
    Get nominal adjuster locations from file. 

    Parameters
    ----------
    path : str
        Path to the data file.
    mirror : str, default: 'primary'
        The mirror that these points belong to.
        Should be either: 'primary' or 'secondary'.

    Returns
    -------
    adjusters : dict[tuple[int, int], NDArray[np.float32]]
        Nominal adjuster locations.
        This is indexed by a (row, col) tuple.
        Each entry is `(5, 3)` array where each row is an adjuster.
    """
    if mirror not in ["primary", "secondary"]:
        raise ValueError(f"Invalid mirror: {mirror}")
    def _transform(coords):
        coords = np.atleast_2d(coords)
        coords -= np.array([80, 0, 0]) # apply CAD shift
        return coord_transform(coords, "va_global", f"opt_{mirror}")

    # TODO: cleaner transform call
    adjusters = defaultdict(list)
    c_points = np.genfromtxt(path, dtype=str)
    for point in c_points:
        row = point[0][6]
        col = point[0][7]
        adjusters[(row, col)] += [_transform(np.array(point[2:], dtype=np.float32))[0]]
    adjusters = {rc : np.vstack(pts) for rc, pts in adjusters.items()}

    return adjusters
