import logging
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import yaml
from megham.utils import make_edm
from numpy.typing import NDArray

from .transforms import align_photo, coord_transform

logger = logging.getLogger("lat_alignment")


def load_photo(
    path: str,
    align: bool = True,
    reference: dict = {},
    err_thresh: float = 2,
    plot: bool = True,
    **kwargs,
) -> tuple[
    dict[str, NDArray[np.float32]], tuple[NDArray[np.float32], NDArray[np.float32]]
]:
    """
    Load photogrammetry data.
    Assuming first column is target names and next three are (x, y , z).

    Parameters
    ----------
    path : str
        The path to the photogrammetry data.
    align : bool, default: True
        If True align using the invar points.
    reference : dict, default: {}
        Reference dictionary for alignment.
        See `transforms.align_photo` for details.
        This is only used is `align` is `True`.
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
    alignment : tuple[NDArray[np.float32], NDArray[np.float32]]
        The transformation that aligned the points.
        The first element is a rotation matrix and
        the second is the shift.
    """
    logger.info("Loading measurement data")
    labels = np.genfromtxt(path, dtype=str, delimiter=",", usecols=(0,))
    coords = np.genfromtxt(path, dtype=np.float32, delimiter=",", usecols=(1, 2, 3))
    errs = np.genfromtxt(path, dtype=np.float32, delimiter=",", usecols=(4, 5, 6))
    msk = (np.char.find(labels, "TARGET") >= 0) + (np.char.find(labels, "CODE") >= 0)

    labels, coords, errs = labels[msk], coords[msk], errs[msk]
    err = np.linalg.norm(errs, axis=-1)

    if align:
        labels, coords, msk, alignment = align_photo(
            labels, coords, reference, plot=plot, **kwargs
        )
        err = err[msk]
    else:
        alignment = (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
    trg_msk = np.char.find(labels, "TARGET") >= 0
    labels = labels[trg_msk]
    coords = coords[trg_msk]
    err = err[trg_msk]

    err_msk = err < err_thresh * np.median(err)
    labels, coords, err = labels[err_msk], coords[err_msk], err[err_msk]
    logger.info("\t%d points loaded", len(coords))

    # Lets find and remove doubles
    # Dumb brute force
    edm = make_edm(coords[:, :2])
    np.fill_diagonal(edm, np.nan)
    to_kill = []
    for i in range(len(edm)):
        if i in to_kill:
            continue
        imin = np.nanargmin(edm[i])
        if edm[i][imin] > 20:
            continue
        if err[i] < err[imin]:
            to_kill += [imin]
        else:
            to_kill += [i]
    msk = ~np.isin(np.arange(len(coords), dtype=int), to_kill)
    logger.info("\tFound and removed %d doubles", len(to_kill))
    labels, coords = labels[msk], coords[msk]

    if plot:
        plt.scatter(coords[:, 0], coords[:, 1], c=coords[:, 2], marker="x")
        plt.colorbar()
        plt.show()

    data = {label: coord for label, coord in zip(labels, coords)}
    return data, alignment


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
