import logging
from typing import Optional
import os
from collections import defaultdict
from functools import partial
from importlib.resources import files

import matplotlib.pyplot as plt
import numpy as np
import yaml
from megham.utils import make_edm
from numpy.typing import NDArray

from .dataset import Dataset, DatasetPhotogrammetry, DatasetReference
from .transforms import coord_transform

logger = logging.getLogger("lat_alignment")


def _load_tracker_yaml(path: str):
    with open(path) as file:
        dat = yaml.safe_load(file)
    if "reference" in dat:
        ref_path = dat["reference"]
    else:
        ref_path = str(files("lat_alignment.data").joinpath("reference.yaml"))
    with open(ref_path) as file:
        reference = yaml.safe_load(file)
    ref_transform = partial(
        coord_transform, cfrom=reference["coords"], cto="opt_global"
    )
    dat_transform = partial(
        coord_transform, cfrom=dat.get("coords", "opt_global"), cto="opt_global"
    )

    # Add the data
    data = {}
    for elem in reference.keys():
        if elem == "coords":
            continue
        r = reference[elem]
        null = np.zeros((len(r), 3)) + np.nan
        d = dat.get(elem, null)
        e = dat.get(f"{elem}_err", null)
        if len(d) != len(r):
            raise ValueError(f"{elem} has {len(d)} points instead of {len(r)}!")
        if len(e) != len(r):
            raise ValueError(f"{elem} error has {len(e)} elements instead of {len(r)}!")
        for i, point in enumerate(r.keys()):
            data[f"{point}_ref"] = ref_transform(r[point][0])
            data[point] = dat_transform(d[i])
            data[f"{point}_err"] = dat_transform(e[i])
    return DatasetReference(data)


def _load_tracker_txt(
    path: str, group_dist: float = 0.02, group_thresh: float = 0.02, err: float = 0.005, calc_sys_err: bool=False,  cam_transform_path: Optional[str]=None,dist_err: float=8e-7, ang_err: float=5e-6
):
    data = np.genfromtxt(
        path, usecols=(3, 4, 5), skip_header=1, dtype=str, delimiter="\t"
    )
    data = np.char.replace(data, ",", "").astype(float)

    errs = err * np.ones_like(data)
    if calc_sys_err:
        data_faro = data.copy()
        if cam_transform_path is not None:
            coord_align = np.genfromtxt(cam_transform_path)
            data_faro = (np.linalg.inv(coord_align)@np.vstack([data_faro.T, np.zeros(len(data_faro))]))[:3].T
        r = np.linalg.norm(data_faro, axis=1)
        theta = np.arccos(data_faro[:, 2]/r)
        phi = np.arctan2(data_faro[:, 1], data_faro[:, 0])
        errs_sphere = np.column_stack([r*dist_err, np.arcsin(np.sin(theta)*ang_err), np.arcsin(np.sin(phi)*ang_err)]) 

        # Taking the linear appriximation, we may expect a small bias 
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        errs[:, 0] += np.sqrt((errs_sphere[:, 0]*st*cp)**2 + (errs_sphere[:, 1]*r*ct*cp)**2 + (errs_sphere[:, 2]*r*st*sp)**2)#/r
        errs[:, 1] += np.sqrt((errs_sphere[:, 0]*st*sp)**2 + (errs_sphere[:, 1]*r*ct*sp)**2 + (errs_sphere[:, 2]*r*st*cp)**2)#/r
        errs[:, 2] += np.sqrt((errs_sphere[:, 0]*ct)**2 + (errs_sphere[:, 1]*r*st)**2)/r

        # Brute force...
        # data_sphere = np.column_stack([r, theta, phi])
        # rng = np.random.default_rng()
        # sphere_dist = data_sphere + rng.normal(size=(1000,)+errs_sphere.shape)*errs_sphere
        # sphere_dist = np.column_stack([sphere_dist[:, :, 0]*np.sin(sphere_dist[:, :, 1])*np.cos(sphere_dist[:, :, 1]), sphere_dist[:, :, 0]*np.sin(sphere_dist[:, :, 1])*np.sin(sphere_dist[:, :, 1]), sphere_dist[:, :, 0]*np.cos(sphere_dist[:, :, 1])])
        # errs += np.std(sphere_dist, axis=0)

    data = np.hstack([data, errs])

    to_kill = []
    if group_dist > 0 and group_thresh > 0:
        done = []
        edm = make_edm(data[:, :2])
        np.fill_diagonal(edm, np.nan)
        for i in range(len(edm)):
            if i in to_kill or i in done:
                continue
            group_idx = np.hstack(([i], np.where(edm[i] <= group_dist)[0]))
            done += group_idx.tolist()
            if len(group_idx) == 1:
                continue
            zs = data[group_idx, 2]
            bad_zs = np.abs(zs - np.median(zs)) > group_thresh
            to_kill += group_idx[bad_zs].tolist()
        logger.info("\tFound and removed %d bad group points", len(to_kill))
    data = {f"TARGET_{i}": np.array([dat[:3], dat[3:]]) for i, dat in enumerate(data) if i not in to_kill}

    return Dataset(data)


def _load_tracker_csv(path: str):
    _ = path
    raise NotImplementedError(
        "Loading tracker data from a csv file not yet implemented"
    )


def load_tracker(path: str, group_dist=0.02, group_thresh=0.02, err=0.005, calc_sys_err: bool=False,  cam_transform_path: Optional[str]=None,dist_err: float=8e-7, ang_err: float=5e-6) -> Dataset:
    """
    Load laser tracker data.

    Parameters
    ----------
    path : str
        The path to the laser tracker data.
        The type of data will be infered from the extension.
    group_dist : float, default: 0.02
        Distance between points in xy needed to group them for cuts.
        Only used for `.txt` files.
        Set to 0 to disable.
    group_thresh : float, default: 0.02
        Difference in z between point and the median z for a group to cut at.
        Only used for `.txt` files.
        Set to 0 to disable.
    err : float, default: .005
        The base error to assume for the tracker data.
        Only used for `.txt` files.
    calc_sys_err : bool, default: False
        It `True` calculate the systematic error based on provided tracker specs.
        Only used for `.txt` files.
    cam_transform_path : Optional[str], default: None
        Alignment matrix exported from CAM2.
        Used when calculating systematic error.
        If not provided we assume the data is the the FARO's internal coordinates.
        Only used for `.txt` files.
    dist_err : float, default 8e-7
        The systematic error as a function of distance in mm/mm.
        Only used for `.txt` files.
    ang_err : float, default 8e-7
        The systematic error as a function of angle in mm/mm.
        Only used for `.txt` files.

    Returns
    -------
    data : Dataset
        The tracker data.
        For txt or csv files this will be the base `Dataset` class.
        For yaml files this will be a `DatasetReference`.
    """
    ext = os.path.splitext(path)[1]
    if ext == ".yaml":
        return _load_tracker_yaml(path)
    elif ext == ".txt":
        return _load_tracker_txt(path, group_dist, group_thresh, err, calc_sys_err, cam_transform_path, dist_err, ang_err)
    elif ext == ".csv":
        return _load_tracker_csv(path)
    raise ValueError(f"Invalid tracker data with extension {ext}")


def load_photo(
    path: str, err_thresh: float = 2, doubles_dist: float = 10, plot: bool = True
) -> Dataset:
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
    coords = np.genfromtxt(path, dtype=np.float64, delimiter=",", usecols=(1, 2, 3))
    errs = np.genfromtxt(path, dtype=np.float64, delimiter=",", usecols=(4, 5, 6))
    msk = (np.char.find(labels, "TARGET") >= 0) + (np.char.find(labels, "CODE") >= 0)

    labels, coords, errs = labels[msk], coords[msk], errs[msk]
    err = np.linalg.norm(errs, axis=-1)
    trg_msk = np.char.find(labels, "TARGET") >= 0
    code_msk = np.char.find(labels, "CODE") >= 0

    err_msk = (err < err_thresh * np.median(err[trg_msk])) + code_msk
    labels, coords, err, errs = (
        labels[err_msk],
        coords[err_msk],
        err[err_msk],
        errs[err_msk],
    )
    logger.info("\t%d good points loaded", len(coords))
    logger.info("\t%d high error points not loaded", np.sum(~err_msk))

    # Lets find and remove doubles
    # Dumb brute force
    trg_msk = np.char.find(labels, "TARGET") >= 0
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
    labels, coords, err, errs = labels[msk], coords[msk], err[msk], errs[msk]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        p = ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            marker="x",
            c=err,
            vmax=np.percentile(err, 90),
        )
        fig.colorbar(p)
        plt.show()

    data = {label: np.array([coord, err]) for label, coord, err in zip(labels, coords, errs)}
    return DatasetPhotogrammetry(data)


def load_data(path: str, source: str = "photo", **kwargs) -> Dataset:
    """
    Load a dataset from path.

    Parameters
    ----------
    path : str
        The path to the data to load.
    source : str, default: 'photo'
        The data source. Current valid options are:

        * photo
        * tracker
    **kwargs
        Arguments to pass the relevent loader function.
        See `load_photo` and `load_tracker` for details.
    """
    if source == "photo":
        return load_photo(path, **kwargs)
    elif source == "tracker":
        return load_tracker(path, **kwargs)
    raise ValueError("Invalid data source")


def load_corners(path: str) -> dict[tuple[int, int], NDArray[np.float64]]:
    """
    Get panel corners from file.

    Parameters
    ----------
    path : str
        Path to the data file.

    Returns
    -------
    corners : dict[tuple[int, int], ndarray[np.float64]]
        The corners. This is indexed by a (row, col) tuple.
        Each entry is `(4, 3)` array where each row is a corner.
    """
    with open(path) as file:
        corners_raw = yaml.safe_load(file)

    corners = {
        (panel[7], panel[9]): np.vstack(
            [np.array(coord.split(), np.float64) for coord in coords]
        )
        for panel, coords in corners_raw.items()
    }
    return corners


def load_adjusters(
    path: str, mirror: str
) -> dict[tuple[int, int], NDArray[np.float64]]:
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
    adjusters : dict[tuple[int, int], NDArray[np.float64]]
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
        adjusters[(row, col)] += [_transform(np.array(point[2:], dtype=np.float64))[0]]
    adjusters = {rc: np.vstack(pts) for rc, pts in adjusters.items()}

    return adjusters
