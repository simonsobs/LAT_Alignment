"""
Script for analyzing trajectory of a point on optical elements.
"""

import argparse
import functools
import logging
import operator
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
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from skspatial.objects import Sphere

from .error import get_hwfe, get_pointing_error
from .io import load_tracker
from .traj_plots import plot_by_ax_point, plot_by_ax, plot_all_ax, plot_all_dir, plot_hist

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

def _plot_point_and_hwfe(data, ref, get_transform, plt_root, logger, skip_missing):
    logger.info("Calculating pointing error and HWFE")
    tods = {
        elem: data[elem]["tod"]
        for elem in data.keys()
        if "tod" in data[elem] and data[elem]["tod"].size > 0
    }
    if len(tods) == 0:
        logger.error("\tNo TODs found! Can't calculate!")
        return
    npts = len(tods[list(tods.keys())[0]])
    angle = np.vstack([data[elem]["angle_tod"] for elem in tods.keys()])
    direction = np.vstack([data[elem]["direction_tod"] for elem in tods.keys()])
    if not (np.isclose(angle, angle[0]) | np.isnan(angle)).all():
        logger.error("\t\tAngles don't match across all elements! Skipping...")
        return
    angle = angle[0]
    direction = direction[0]
    for elem in LABELS.keys():
        if elem in tods:
            if tods[elem].shape[1] < 4:
                logger.error(
                    "Only %d points found! Filling with reference...",
                    tods[elem].shape[1],
                )
                tods[elem] = np.zeros((npts,) + ref[elem].shape) + ref[elem]
            continue
        logger.warning("No %s TOD found, filling with reference...", elem)
        tods[elem] = np.zeros((npts,) + ref[elem].shape) + ref[elem]

    hwfes = np.zeros(npts) + np.nan
    pes = np.zeros(npts) + np.nan
    missing = []
    for i in range(npts):
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
    x = np.arange(npts)
    plot_all_dir(x, hwfes, direction, missing, "Measurement (#)", "HWFE (um-rms)", f"HWFE over time", plt_root_err)
    plot_all_dir(x, pes, direction, missing, "Measurement (#)", 'Pointing Error (")', f"Pointing Error over time", plt_root_err)

    # Plot distribution
    plot_hist(hwfes, direction, "HWFE (um-rms)", "HWFE Distribution", plt_root_err)
    plot_hist(pes, direction, 'Pointing Error (")', "Pointing Error Distribution", plt_root_err)

    # Now by angle
    plot_all_dir(angle, hwfes, direction, missing, "Angle (deg)", "HWFE (um-rms)", f"HWFE by Angle", plt_root_err)
    plot_all_dir(angle, pes, direction, missing, "Angle (deg)", 'Pointing Error (")', f"Pointing Error by Angle", plt_root_err)

def _plot_transform(data, ref, get_transform, plt_root, logger, skip_missing):
    logger.info("Plotting transformation information")
    for elem in data.keys():
        logger.info("\tGetting transforms for %s", elem)
        if "tod" not in data[elem] or data[elem]["tod"].size == 0:
            logger.error("\t\t%s TOD not found! Skipping...", elem)
            continue
        if len(data[elem]["points"]) < 4:
            logger.error(
                "\t\tOnly %d points found! Skipping...", len(data[elem]["points"])
            )
            continue
        src = data[elem]["tod"]
        dst = ref[elem]
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
        x = np.arange(len(sfts))
        plot_all_ax(x, sfts, missing, "Measurement (#)", "Shift (mm)", f"{elem} Shifts over time", plt_root_elem)
        plot_all_ax(x, rots, missing, "Measurement (#)", "Rotation (deg)", f"{elem} Rotation over time", plt_root_elem)
        plot_all_ax(x, scales, missing, "Measurement (#)", "Scale Factor", f"{elem} Scale over time", plt_root_elem)

        # Now by angle
        direction = data[elem]["direction_tod"]
        x = data[elem]["angle_tod"]
        plot_by_ax(x, sfts, direction, missing, "angle_tod", "Angle (deg)", "shift (mm)", f"{elem} Shifts by Angle", os.path.join(plt_root, elem))
        plot_by_ax(x, rots, direction, missing, "angle_tod", "Angle (deg)", "rotation (deg)", f"{elem} Rotation by Angle", os.path.join(plt_root, elem))
        plot_by_ax(x, scales, direction, missing, "angle_tod", "Angle (deg)", "scale ", f"{elem} Scale by Angle", os.path.join(plt_root, elem))

        # Plot resids
        for xax, xlab in [ ("angle_tod", "Angle (deg)"), ("meas_number", "Measurement (#)"), ]:
            x = data[elem][xax]
            plot_by_ax_point(data[elem]["points"], x, resids, direction, missing, xax, xlab, f"{elem} Residuals", os.path.join(plt_root, elem))

def _plot_path(data, plt_root, logger):
    for elem in data.keys():
        logger.info("Plotting %s trajectory", elem)
        os.makedirs(os.path.join(plt_root, elem), exist_ok=True)
        if "tod" not in data[elem] or data[elem]["tod"].size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        # Plot raw trajectories
        for xax, xlab in [
            ("angle_tod", "Angle (deg)"),
            ("meas_number", "Measurement (#)"),
        ]:
            x = data[elem][xax]
            dat = data[elem]["tod"]
            direction = data[elem]["direction_tod"]
            plot_by_ax_point(data[elem]["points"], x, dat, direction, [], xax, xlab, f"{elem} Trajectory", os.path.join(plt_root, elem))

def _plot_traj_error(data, plt_root, logger):
    for elem in data.keys():
        logger.info("\t\tPlotting %s trajectory error", elem)
        if "tod" not in data[elem] or data[elem]["tod"].size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        _, axs = plt.subplots(
            2,
            len(data[elem]["points"]),
            sharex=False,
            sharey=False,
            figsize=(24, 20),
            layout="constrained",
        )
        axs = np.reshape(np.array(axs), (int(2), len(data[elem]["points"])))
        for i, point in enumerate(data[elem]["points"]):
            dat = data[elem]["tod"][:, i, :]
            angle = data[elem]["angle_tod"]
            direction = data[elem]["direction_tod"]
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
                if len(d) == 0:
                    continue
                axs[0, i].hist(d, bins="auto", color=color, alpha=0.5, label=label)
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


def _pad_missing(arr1, arr2):
    master = arr1
    to_pad = arr2
    if len(arr2) > len(arr1):
        master = arr2
        to_pad = arr1
    pad_msk = np.zeros_like(master, bool)
    if len(master) == len(to_pad):
        return to_pad, pad_msk
    if not np.isclose(to_pad[0], master[0]):
        pstart = np.where(np.isclose(master, to_pad[0]))[0][0]
        to_pad = np.hstack((master[:pstart], to_pad))
        pad_msk[:pstart] = True
    dmaster = np.diff(master)
    dpad = np.diff(to_pad)
    while not np.allclose(dmaster[: len(dpad)], dpad):
        didx = np.where(~np.isclose(dmaster[: len(dpad)], dpad))[0][0]
        to_insert = master[didx + 1 : didx + 2]
        to_pad = np.hstack((to_pad[: didx + 1], to_insert, to_pad[didx + 1 :]))
        pad_msk[didx + 1 : didx + 2] = True
        dpad = np.diff(to_pad)
        if len(to_pad) == len(master):
            break
    if not np.isclose(to_pad[-1], master[-1]):
        pend = np.where(np.isclose(master, to_pad[-1]))[0][-1] + 1
        to_pad = np.hstack((to_pad, master[pend:]))
        pad_msk[pend:] = True
    return to_pad, pad_msk


def _add_tod(data, logger, pad=False):
    # TODO: dataclass just for TOD?
    npoints = np.hstack(
        [
            [len(data[elem][point]["data"]) for point in data[elem].keys()]
            for elem in data.keys()
        ]
    )
    if not np.all(npoints == npoints[0]):
        if not pad:
            raise ValueError("Not all points have the same number of measurements!")
        logger.warning("\tPadding data with nans")
        master_angle = [
            [
                data[elem][point]["angle"]
                for point in LABELS[elem]
                if point in data[elem]
            ]
            for elem in LABELS.keys()
            if elem in data
        ]
        master_angle = functools.reduce(operator.iconcat, master_angle, [])
        nangs = np.array([len(ang) for ang in master_angle])
        master_angle = master_angle[np.argmax(nangs)]
        for elem in data.keys():
            for point in data[elem].keys():
                ang_pad, pad_msk = _pad_missing(
                    master_angle, data[elem][point]["angle"]
                )
                data[elem][point]["angle"] = ang_pad
                dat_pad = (
                    np.zeros((len(ang_pad),) + data[elem][point]["data"].shape[1:])
                    + np.nan
                )
                dat_pad[~pad_msk] = data[elem][point]["data"]
                data[elem][point]["data"] = dat_pad
                direction = np.diff(ang_pad)
                direction = np.hstack((direction, [direction[-1]]))
                data[elem][point]["direction"] = direction

    for elem in data.keys():
        logger.info("\tConstructing TOD for %s", elem)
        angle = np.atleast_2d(
            np.array(
                [
                    data[elem][point]["angle"]
                    for point in LABELS[elem]
                    if point in data[elem]
                ]
            )
        )
        direction = np.atleast_2d(
            np.array(
                [
                    data[elem][point]["direction"]
                    for point in LABELS[elem]
                    if point in data[elem]
                ]
            )
        )
        if not (np.isclose(angle, angle[0]) | np.isnan(angle)).all():
            logger.error("\t\tAngles don't match across all points! Skipping...")
            continue
        angle = angle[0]
        direction = direction[0]
        points = [point for point in LABELS[elem] if point in data[elem]]
        src = np.swapaxes(
            np.atleast_3d(np.array([data[elem][point]["data"] for point in points])),
            0,
            1,
        )
        if src.size == 0:
            logger.info("\t\tNo data found! Not making TOD")
        data[elem]["tod"] = src
        data[elem]["angle_tod"] = angle
        data[elem]["direction_tod"] = direction
        data[elem]["points"] = points
        data[elem]["meas_number"] = np.arange(len(src))
    return data


def _quantize_angle(theta, dtheta, start):
    if dtheta != 0:
        theta_corr = theta - theta[0]
        theta_corr = dtheta * np.round(theta_corr / dtheta, 0) + theta[0]
    else:
        theta_corr = np.ones_like(theta) * start

    # Figure out left vs right vs static
    direction = np.diff(theta_corr)
    # Lets just make the last point keep the same direction
    direction = np.hstack((direction, [direction[-1]]))

    return theta_corr, direction


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


def get_angle(data, mode, start, sep, logger):
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
    theta_corr, direction = _quantize_angle(theta, dtheta, start)

    return theta_corr, direction, sphere.point


def correct_rot(src, angle, cent, off=0):
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
    transform_str = "rigid"
    if args.affine:
        get_transform = partial(get_affine, force_svd=True)
        transform_str = "affine"

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
    data = {"primary": {}, "secondary": {}, "receiver": {}}
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
            data[elem][point] = {}
            data[elem][point]["data"] = dat
            data[elem][point]["mode"] = cfg[elem][point]["mode"]
            data[elem][point]["start"] = cfg[elem][point]["start"]
            data[elem][point]["sep"] = cfg[elem][point]["sep"]
            dat = data[elem][point]
            angle, direction, cent = get_angle(
                dat["data"], dat["mode"], dat["start"], dat["sep"], logger
            )
            off = 0
            if elem in ["primary", "secondary"]:
                off = 90
                angle = angle % 360
            if cfg.get("correct_rot", False):
                corr = correct_rot(dat["data"], angle, cent, off)
                data[elem][point]["data"] = corr
            data[elem][point]["angle"] = angle
            data[elem][point]["direction"] = direction

    # Check motion of each element
    data = _add_tod(data, logger, cfg.get("pad", False))
    _plot_path(data, plt_root, logger)
    _plot_traj_error(data, plt_root, logger)
    _plot_transform(
        data, ref, get_transform, plt_root, logger, cfg.get("skip_missing", False)
    )
    _plot_point_and_hwfe(
        data, ref, get_transform, plt_root, logger, cfg.get("skip_missing", False)
    )
