import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from megham.utils import make_edm
from numpy.typing import NDArray

from .transforms import coord_transform
from .photogrammetry import Dataset

logger = logging.getLogger("lat_alignment")

def load_photo( path: str, err_thresh: float = 2, doubles_dist: float = 10, plot: bool = True) -> Dataset:
    """
    Load photogrammetry data.
    Assuming first column is target names and next three are (x, y , z).

    Parameters
    ----------
    path : str
        The path to the photogrammetry data.
    err_thresh : float, default: 2
        How many times the median photogrammetry error
        a target need to have to be cut.
    plot: bool, default: True
        If True display a scatter plot of targets.

    Returns
    -------
    data : Dataset 
        The photogrammetry data.
    """
    logger.info("Loading measurement data")
    labels = np.genfromtxt(path, dtype=str, delimiter=",", usecols=(0,))
    coords = np.genfromtxt(path, dtype=np.float32, delimiter=",", usecols=(1, 2, 3))
    errs = np.genfromtxt(path, dtype=np.float32, delimiter=",", usecols=(4, 5, 6))
    msk = (np.char.find(labels, "TARGET") >= 0) + (np.char.find(labels, "CODE") >= 0)

    labels, coords, errs = labels[msk], coords[msk], errs[msk]
    err = np.linalg.norm(errs, axis=-1)
    trg_msk = (np.char.find(labels, "TARGET") >= 0)
    code_msk = (np.char.find(labels, "CODE") >= 0)

    err_msk = (err < err_thresh * np.median(err[trg_msk])) + code_msk
    labels, coords, err = labels[err_msk], coords[err_msk], err[err_msk]
    logger.info("\t%d good points loaded", len(coords))
    logger.info("\t%d high error points not loaded", np.sum(~err_msk))

    # Lets find and remove doubles
    # Dumb brute force
    trg_msk = (np.char.find(labels, "TARGET") >= 0)
    edm = make_edm(coords[trg_msk, :2])
    np.fill_diagonal(edm, np.nan)
    to_kill = []
    for i in range(len(edm)):
        if labels[trg_msk][i] in to_kill:
            continue
        imin = np.nanargmin(edm[i])
        if edm[i][imin] > doubles_dist:
            continue
        if err[trg_msk][i] < err[trg_msk][imin]:
            to_kill += [labels[trg_msk][imin]]
        else:
            to_kill += [labels[trg_msk][i]]
    msk = ~np.isin(labels, to_kill)
    logger.info("\tFound and removed %d doubles", len(to_kill))
    labels, coords = labels[msk], coords[msk]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], marker="x")
        plt.show()

    data = {label: coord for label, coord in zip(labels, coords)}
    return Dataset(data) 


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

    corners = {
        (panel[7], panel[9]): np.vstack(
            [np.array(coord.split(), np.float32) for coord in coords]
        )
        for panel, coords in corners_raw.items()
    }
    return corners


def load_adjusters(
    path: str, mirror: str
) -> dict[tuple[int, int], NDArray[np.float32]]:
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
        coords -= np.array([120, 0, 0])  # cancel out shift
        return coord_transform(coords, "va_global", f"opt_{mirror}")

    # TODO: cleaner transform call
    adjusters = defaultdict(list)
    c_points = np.genfromtxt(path, dtype=str)
    for point in c_points:
        row = point[0][6]
        col = point[0][7]
        adjusters[(row, col)] += [_transform(np.array(point[2:], dtype=np.float32))[0]]
    adjusters = {rc: np.vstack(pts) for rc, pts in adjusters.items()}

    return adjusters
