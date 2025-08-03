"""
Script for analyzing trajectory of a point on optical elements.
"""

import argparse
import logging
import os
from functools import partial
from importlib.resources import files

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import yaml
from megham.transform import (
    apply_transform,
    decompose_affine,
    decompose_rotation,
    get_affine,
    get_rigid,
)
from numpy.typing import NDArray
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from skspatial.objects import Sphere

from .error import get_hwfe, get_pointing_error
from .io import load_tracker
from .refpoint import RefCollection, RefTOD
from .traj_plots import (
    plot_all_ax,
    plot_all_dir,
    plot_by_ax,
    plot_by_ax_point,
    plot_hist,
)
from .transforms import coord_transform

mpl.rcParams["lines.markersize"] *= 1.5

# This is dumb! Should get from the nominal file somehow!
LABELS = {
    "primary": [
        "primary_lower_right",
        "primary_lower_left",
        "primary_upper_right",
        "primary_upper_left",
    ],
    "secondary": [
        "secondary_lower_right",
        "secondary_lower_left",
        "secondary_upper_right",
        "secondary_upper_left",
    ],
    "receiver": ["receiver_1", "receiver_2", "receiver_3", "receiver_4"],
}

local_coords = {
    "primary": "opt_primary",
    "secondary": "opt_secondary",
    "receiver": "opt_global",
}


def _plot_point_and_hwfe(data, ref, get_transform, plt_root, logger, skip_missing):
    logger.info("Calculating pointing error and HWFE")
    tods = {
        elem.name: elem.data
        for elem in data.elems
        if elem.data.size > 0 and elem.data.shape[1] > 3
    }
    if len(tods) == 0:
        logger.error("\tNo TODs found! Can't calculate!")
        return
    for elem in LABELS.keys():
        if elem in tods:
            if tods[elem].shape[1] < 4:
                logger.error(
                    "Only %d points found! Filling with reference...",
                    tods[elem].shape[1],
                )
                tods[elem] = np.zeros((data.npoints,) + ref[elem].shape) + ref[elem]
            continue
        logger.warning("No %s TOD found, filling with reference...", elem)
        tods[elem] = np.zeros((data.npoints,) + ref[elem].shape) + ref[elem]

    hwfes = np.zeros(data.npoints) + np.nan
    pes = np.zeros(data.npoints) + np.nan
    missing = []
    for i in range(data.npoints):
        _data = {}
        tot = 0
        for elem in tods.keys():
            meas = tods[elem][i]
            msk = np.isfinite(meas).all(axis=1)
            tot += np.sum(msk) / len(meas)
            _data[elem] = meas
            _data[f"{elem}_err"] = np.zeros_like(meas)
            _data[f"{elem}_ref"] = ref[elem]
            _data[f"{elem}_msk"] = msk
        if tot < len(tods):
            if skip_missing:
                continue
            missing += [i]
        try:
            hwfes[i] = get_hwfe(_data, get_transform, False)
            pes[i] = get_pointing_error(_data, get_transform, False)
        except ValueError:
            logger.error(
                "\t\tFailed to get transform for a data point! Filling with nans"
            )
            continue

    # Plot TOD
    plt_root_err = os.path.join(plt_root, "error")
    os.makedirs(plt_root_err, exist_ok=True)
    plot_all_dir(
        data.meas_number,
        hwfes,
        data.direction,
        missing,
        "Measurement (#)",
        "HWFE (um-rms)",
        f"HWFE over time",
        plt_root_err,
    )
    plot_all_dir(
        data.meas_number,
        pes,
        data.direction,
        missing,
        "Measurement (#)",
        'Pointing Error (")',
        f"Pointing Error over time",
        plt_root_err,
    )

    # Plot distribution
    plot_hist(hwfes, data.direction, "HWFE (um-rms)", "HWFE Distribution", plt_root_err)
    plot_hist(
        pes,
        data.direction,
        'Pointing Error (")',
        "Pointing Error Distribution",
        plt_root_err,
    )

    # Now by angle
    plot_all_dir(
        data.angle,
        hwfes,
        data.direction,
        missing,
        "Angle (deg)",
        "HWFE (um-rms)",
        f"HWFE by Angle",
        plt_root_err,
    )
    plot_all_dir(
        data.angle,
        pes,
        data.direction,
        missing,
        "Angle (deg)",
        'Pointing Error (")',
        f"Pointing Error by Angle",
        plt_root_err,
    )


def _plot_transform(
    data, ref, get_transform, plt_root, logger, skip_missing, local=False
):
    logger.info("Plotting transformation information")
    for elem in data.keys():
        logger.info("\tGetting transforms for %s", elem)
        if data[elem].data.size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        if data[elem].ntods < 4:
            logger.error("\t\tOnly %d points found! Skipping...", data[elem].ntods)
            continue
        src = data[elem].data
        dst = ref[elem]
        if local:
            dst = coord_transform(dst, "opt_global", local_coords[elem])
        sfts = np.zeros((len(src), 3)) + np.nan
        rots = np.zeros((len(src), 3)) + np.nan
        scales = np.zeros((len(src), 3)) + np.nan
        resids = np.zeros((len(src), len(dst), 3)) + np.nan
        missing = []
        for i, _src in enumerate(src):
            if not np.all(np.isfinite(_src)):
                if skip_missing:
                    continue
                missing += [i]
            if local:
                _src = coord_transform(_src, "opt_global", local_coords[elem])
            try:
                aff, sft = get_transform(_src, dst)
            except ValueError:
                logger.error(
                    "\t\tFailed to get transform for a data point! Filling with nans"
                )
                continue
            scale, _, rot = decompose_affine(aff)
            rot = np.rad2deg(decompose_rotation(rot))
            sfts[i] = sft
            rots[i] = rot
            scales[i] = scale
            trf = apply_transform(_src, aff, sft)
            resids[i] = dst - trf

        # Lets plot shift and rotation
        # First with time
        plt_root_elem = os.path.join(plt_root, elem)
        os.makedirs(plt_root_elem, exist_ok=True)
        plot_all_ax(
            data[elem].meas_number,
            sfts,
            missing,
            "Measurement (#)",
            "Shift (mm)",
            f"{elem} Shifts over time",
            plt_root_elem,
        )
        plot_all_ax(
            data[elem].meas_number,
            rots,
            missing,
            "Measurement (#)",
            "Rotation (deg)",
            f"{elem} Rotation over time",
            plt_root_elem,
        )
        plot_all_ax(
            data[elem].meas_number,
            scales,
            missing,
            "Measurement (#)",
            "Scale Factor",
            f"{elem} Scale over time",
            plt_root_elem,
        )

        # Now by angle
        direction = data[elem].direction
        plot_by_ax(
            data[elem].angle,
            sfts,
            direction,
            missing,
            "angle_tod",
            "Angle (deg)",
            "shift (mm)",
            f"{elem} Shifts by Angle",
            os.path.join(plt_root, elem),
        )
        plot_by_ax(
            data[elem].angle,
            rots,
            direction,
            missing,
            "angle_tod",
            "Angle (deg)",
            "rotation (deg)",
            f"{elem} Rotation by Angle",
            os.path.join(plt_root, elem),
        )
        plot_by_ax(
            data[elem].angle,
            scales,
            direction,
            missing,
            "angle_tod",
            "Angle (deg)",
            "scale ",
            f"{elem} Scale by Angle",
            os.path.join(plt_root, elem),
        )

        # Plot resids
        for xax, xlab in [
            ("angle", "Angle (deg)"),
            ("meas_number", "Measurement (#)"),
        ]:
            x = getattr(data[elem], xax)
            plot_by_ax_point(
                data[elem].tod_names,
                x,
                resids,
                direction,
                missing,
                xax,
                xlab,
                f"{elem} Residuals",
                os.path.join(plt_root, elem),
            )


def _plot_path(data, plt_root, logger):
    for elem in data.keys():
        logger.info("Plotting %s trajectory", elem)
        os.makedirs(os.path.join(plt_root, elem), exist_ok=True)
        if data[elem].data.size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        # Plot raw trajectories
        for xax, xlab in [
            ("angle", "Angle (deg)"),
            ("meas_number", "Measurement (#)"),
        ]:
            x = getattr(data[elem], xax)
            dat = data[elem].data
            direction = data[elem].direction
            plot_by_ax_point(
                data[elem].tod_names,
                x,
                dat,
                direction,
                [],
                xax,
                xlab,
                f"{elem} Trajectory",
                os.path.join(plt_root, elem),
            )


def _plot_traj_error(data, plt_root, logger):
    for elem in data.keys():
        logger.info("\t\tPlotting %s trajectory error", elem)
        if data[elem].data.size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        _, axs = plt.subplots(
            2,
            data[elem].ntods,
            sharex=False,
            sharey=False,
            figsize=(24, 20),
            layout="constrained",
        )
        axs = np.reshape(np.array(axs), (int(2), data[elem].ntods))
        for i, point in enumerate(data[elem].tod_names):
            dat = data[elem].data[:, i, :]
            angle = data[elem].angle
            direction = data[elem].direction
            diffs = [[], [], []]
            rmss = [[], [], []]
            ang_u = np.unique(angle)
            for ang in np.unique(angle):
                msk = angle == ang
                _dat = dat[msk]
                dire = direction[msk]
                rmss[0] += [np.linalg.norm(np.nanstd(_dat[dire == 0], axis=0))]
                rmss[1] += [np.linalg.norm(np.nanstd(_dat[dire < 0], axis=0))]
                rmss[2] += [np.linalg.norm(np.nanstd(_dat[dire > 0], axis=0))]
                diffs[0] += [pdist(_dat[dire == 0])]
                diffs[1] += [pdist(_dat[dire < 0])]
                diffs[2] += [pdist(_dat[dire > 0])]

            # Plot distribution
            for j, (label, color) in enumerate(
                [("Stationary", "black"), ("Decreasing", "blue"), ("Increasing", "red")]
            ):
                d = np.hstack(diffs[j]).ravel()
                msk = np.isfinite(d)
                if len(d[msk]) == 0:
                    continue
                axs[0, i].hist(d[msk], bins="auto", color=color, alpha=0.5, label=label)
            axs[0, i].legend()
            axs[0, i].set_xlabel("Distance Between Repeated Points (mm)")
            axs[0, i].set_ylabel("Count")
            axs[0, i].set_title(point)
            axs[0, i].autoscale()

            # Plot rms
            axs[1, i].scatter(
                ang_u, rmss[0], color="black", marker="o", alpha=0.5, label="Stationary"
            )
            axs[1, i].scatter(
                ang_u, rmss[1], color="blue", marker="x", alpha=0.5, label="Decreasing"
            )
            axs[1, i].scatter(
                ang_u, rmss[2], color="red", marker="+", alpha=0.5, label="Increasing"
            )
            axs[1, i].set_xlabel("Angle (deg)")
            axs[1, i].set_ylabel("RMS (mm)")
        plt.suptitle(f"{elem} Trajectory Error")
        plt.savefig(
            os.path.join(plt_root, elem, f"{elem}_error.png"),
            bbox_inches="tight",
        )
        plt.close()


def _quantize_angle(theta, dtheta, start):
    if dtheta != 0:
        theta_corr = theta - theta[0]
        theta_corr = dtheta * np.round(theta_corr / dtheta, 0) + theta[0]
    else:
        theta_corr = np.ones_like(theta) * start

    return theta_corr


def _get_sphere_and_angle(data, start, logger):
    # Get the best fit radius and center
    sphere = Sphere.best_fit(data)
    logger.debug(
        "\t\tFit a radius of %s and a center at %s",
        str(sphere.radius),
        str(sphere.point),
    )

    # Recover the angle
    d_data = data - sphere.point
    # Assuming optical global coordinates)
    theta = np.rad2deg(np.arctan2(d_data[:, 0], d_data[:, 2]))

    # Correct based on start position
    theta -= theta[0] - start

    return theta, sphere


def get_angle(
    data: NDArray[np.float64],
    mode: str,
    start: float,
    sep: float,
    logger: logging.Logger,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Reconstruct the angle of an opitcal element from data of a point.

    Parameters
    ----------
    data : NDArray[np.float64]
        A `(npoint, ndim)` array describing the motion of a point.
    mode : str
        The mode this data was taken in.
        Should be `continious` or `step`.
    start : float
        The angle of the element at the first data point.
        Should be in degrees.
    sep : float
        The seperation between measurements.
        If we are in `continious` mode this should be in mm.
        If we are in `step` mode this should be in deg.
    logger : logging.Logger
        The logger object to use.

    Returns
    -------
    angle : NDArray[np.float64]
        The reconstructed angle in degrees.
        Will be a `(npoint,)` array.
    center : NDArray[np.float64]
        The center of rotation.
    """
    logger.info(
        "\tReconstructing angle in %s mode using a start of %f deg and a seperation of %f",
        mode,
        start,
        sep,
    )
    # Recover the angle
    theta, sphere = _get_sphere_and_angle(data, start, logger)

    # Convert delta to an angle delta
    if mode == "continious":
        logger.warning(
            "\t\tReconstructing angle from continious data, this is approximate and should not be used for pointing corrections! Trajectory Errors will also only be approximate!"
        )
        dtheta = np.rad2deg(sep / sphere.radius) / 32.0
    elif mode == "step":
        dtheta = sep
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Quantize
    theta_corr = _quantize_angle(theta, dtheta, start)

    return theta_corr, np.array(sphere.point, np.float64)


def correct_rot(
    src: NDArray[np.float64],
    angle: NDArray[np.float64],
    cent: NDArray[np.float64],
    off: float = 0,
) -> NDArray[np.float64]:
    """
    Remove the rotation of an element from a point.
    For example undo corotation from a point on the LATR.

    Parameters
    ----------
    src : NDArray[np.float64]
        The data to rotate.
        Should be `(npoint, ndim)`.
    angle : NDArray[np.float64]
        The angle in degrees of the element at each data point.
        Should by `(npoint,)`.
    cent : NDArray[np.float64]
        The center of rotation of the point.
        Should be `(ndim,)`.

    Returns
    -------
    src : NDArray[np.float64]
        The data with the rotation removed.
        The input is also modified in place.
    """
    for i, (pt, ang) in enumerate(zip(src, angle)):
        rot = Rotation.from_euler("Y", -1 * (ang - off), degrees=True)
        src[i] = rot.apply(pt - cent) + rot.apply(cent)
    return src


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to data config file")
    parser.add_argument(
        "--affine",
        "-a",
        action="store_true",
        help="Pass to compute affine instead of rigid rotation",
    )
    parser.add_argument(
        "--log_level", "-l", default="INFO", help="the log level to use"
    )
    args = parser.parse_args()
    logging.basicConfig()
    logger = logging.getLogger("lat_alignment")
    logger.setLevel(args.log_level.upper())
    plt_root = os.path.splitext(args.config)[0]

    # Pick the fitter
    get_transform = get_rigid
    if args.affine:
        get_transform = partial(get_affine, force_svd=True)

    # Load data and do basic processing
    with open(args.config) as file:
        cfg = yaml.safe_load(file)
    if "reference" in cfg:
        ref_path = cfg["reference"]
    else:
        ref_path = str(files("lat_alignment.data").joinpath("reference.yaml"))
    with open(ref_path) as file:
        reference = yaml.safe_load(file)
    ref = {}
    ref["primary"] = np.array([p for p, _ in reference["primary"]])
    ref["secondary"] = np.array([p for p, _ in reference["secondary"]])
    ref["receiver"] = np.array([p for p, _ in reference["receiver"]])
    # TODO: This is ugly and bad
    for i, p in enumerate(["lower_right", "lower_left", "upper_right", "upper_left"]):
        ref[f"primary_{p}"] = ref["primary"][i]
        ref[f"secondary_{p}"] = ref["secondary"][i]
        ref[f"receiver_{i+1}"] = ref["receiver"][i]
    data = {"primary": [], "secondary": [], "receiver": []}
    for elem in data.keys():
        if elem not in cfg:
            logger.info("%s not in config file", elem)
            continue
        for point in cfg[elem].keys():
            point = point.lower()
            logger.info("Loading %s", point)
            if point in data[elem]:
                raise ValueError(f"{elem} already in data!")
            if point not in ref:
                raise ValueError(f"No reference for {point} found!")
            dat = load_tracker(cfg[elem][point]["path"])
            if not isinstance(dat, np.ndarray):
                raise ValueError("Loaded data is not array!")
            mode = cfg[elem][point]["mode"]
            start = cfg[elem][point]["start"]
            sep = cfg[elem][point]["sep"]
            angle, cent = get_angle(dat, mode, start, sep, logger)
            off = 0
            if elem in ["primary", "secondary"]:
                off = 90
                angle = angle % 360
            if cfg.get("correct_rot", False):
                dat = correct_rot(dat, angle, cent, off)
            data[elem] += [RefTOD(point, dat, angle)]

    # Construct the dataclass
    data = RefCollection.construct(data, logger, pad=cfg.get("pad", False))
    data.elems = [elem.reorder(LABELS[elem.name], False) for elem in data.elems]

    # Check motion of each element
    _plot_path(data, plt_root, logger)
    _plot_traj_error(data, plt_root, logger)
    _plot_transform(
        data,
        ref,
        get_transform,
        plt_root,
        logger,
        cfg.get("skip_missing", False),
        cfg.get("local", False),
    )
    _plot_point_and_hwfe(
        data, ref, get_transform, plt_root, logger, cfg.get("skip_missing", False)
    )

    # Save if we want
    if not cfg.get("save", False):
        sys.exit(0)
    logger.info("Saving reconstructed TODs")
    if cfg.get("local", False):
        logger.info("\tSaving in local coordinates")
        outdir = os.path.join(plt_root, "tods")
        os.makedirs(outdir, exist_ok=True)
        for elem in data.elems:
            for tod in elem.tods:
                dat = tod.data
                if cfg.get("local", False):
                    dat = coord_transform(dat, "opt_global", local_coords[elem.name])
                to_save = np.column_stack([tod.angle, np.sign(tod.direction), tod.data])
                np.savetxt(
                    os.path.join(outdir, f"{tod.name}.csv"),
                    to_save,
                    header="angle direction x y z",
                )
