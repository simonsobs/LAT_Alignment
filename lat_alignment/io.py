import numpy as np
from numpy.typing import NDArray
import yaml

from .transforms import align_photo, cad_to_secondary, cad_to_primary

def load_photo(path: str, align=True, **kwargs) -> dict[str, NDArray[np.float32]]:
    """
    Load photogrammetry data.
    Assuming first column is target names and next three are (x, y , z).

    Parameters
    ----------
    path : str
        The path to the photogrammetry data.
    align : bool, default: True
        If True align using the invar points.
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

    if align:
        coords = align_photo(labels, coords, **kwargs)

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

    corners = {(panel[7], panel[9]):np.vstack([np.array(coord.split[" "], np.float32) for coord in coords]) for panel, coords in corners_raw.items()}
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
    if mirror == "primary":
        transform = cad_to_primary 
    else:
        transform = cad_to_secondary 

    adjusters = defaultdict(list)
    c_points = np.genfromtxt(path, dtype=str)
    for point in c_points:
        row = point[0][6]
        col = point[0][7]
        adjusters[(row, col)] += [np.array(point[2:], dtype=float32)]
    adjusters = {rc : np.vstack(pts) for rc, pts in adjusters.items()}

    return adjusters
