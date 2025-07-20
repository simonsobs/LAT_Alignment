"""
Script for analyzing trajectory of a point on optical elements.
"""
import argparse
import logging
import os
import sys
from copy import deepcopy
from numpy.lib import angle
import yaml
from functools import partial
from importlib.resources import files
from scipy.interpolate import make_smoothing_spline
from scipy.spatial.distance import pdist

import matplotlib.pyplot as plt
import numpy as np
from megham.transform import (
    apply_transform,
    decompose_affine,
    decompose_rotation,
    get_affine,
    get_rigid,
)
from skspatial.objects import Sphere
from tqdm import tqdm

from .io import load_tracker
from .transforms import coord_transform
from .error import get_hwfe, get_pointing_error

# This is dumb! Should get from the nominal file somehow!
LABELS = {"primary": ["primary_lower_right", "primary_lower_left", "primary_upper_right", "primary_upper_left"],
          "secondary": ["secondary_lower_right", "secondary_lower_left", "secondary_upper_right", "secondary_upper_left"],
          "receiver": ["receiver_1", "receiver_2", "receiver_3", "receiver_4"]}

def _plot_point_and_hwfe(data, ref, get_transform, plt_root, logger):
    logger.info("Calculating pointing error and HWFE")
    tods = {elem: data[elem]["tod"] for elem in data.keys() if "tod" in data[elem] and data[elem]["tod"].size > 0}
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
                logger.error("Only %d points found! Filling with reference...", tods[elem].shape[1])
                tods[elem] = np.zeros((npts,) + ref[elem].shape) + ref[elem]
            continue
        logger.warning("No %s TOD found, filling with reference...", elem)
        tods[elem] = np.zeros((npts,) + ref[elem].shape) + ref[elem]

    hwfes = np.zeros(npts) + np.nan
    pes = np.zeros(npts) + np.nan
    for i in range(npts):
        _data = {}
        tot = 0
        for elem in tods.keys():
            meas = tods[elem][i]
            msk = np.isfinite(meas).all(axis=1)
            tot += np.sum(msk)/len(meas)
            _data[elem] = meas 
            _data[f"{elem}_err"] = np.zeros_like(meas)
            _data[f"{elem}_ref"] = ref[elem]
            _data[f"{elem}_msk"] = msk
        if tot < len(tods):
            # TODO: implement work around when we only have 3 points
            logger.error("Too few points in measurement! Skipping...")
            continue
        hwfes[i] = get_hwfe(_data, get_transform, False)
        pes[i] = get_pointing_error(_data, get_transform, False)

    # Plot TOD
    os.makedirs(os.path.join(plt_root, "error"), exist_ok=True)
    t = np.arange(npts)
    plt.scatter(t, hwfes, alpha=.5)
    plt.xlabel("Measurement #")
    plt.ylabel("HWFE (um-rms)")
    plt.title(f"HWFE over time")
    plt.savefig(os.path.join(plt_root, "error", "hwfe_tod.png"), bbox_inches = "tight")
    plt.close()

    plt.scatter(t, pes, alpha=.5)
    plt.xlabel("Measurement #")
    plt.ylabel('Pointing Error (")')
    plt.title(f"Pointing Error over time")
    plt.savefig(os.path.join(plt_root, "error", "pe_tod.png"), bbox_inches = "tight")
    plt.close()

    # Plot distribution
    if len(direction == 0) > 0:
        plt.hist(hwfes[direction == 0], bins='auto', color="black", alpha=.5, label="Stationary")
    if len(direction < 0) > 0:
        plt.hist(hwfes[direction < 0], bins='auto', color="blue", alpha=.5, label="Decreasing")
    if len(direction > 0) > 0:
        plt.hist(hwfes[direction > 0], bins='auto', color="red", alpha=.5, label="Increasing")
    plt.legend()
    plt.xlabel("HWFE (um-rms)")
    plt.ylabel("Counts")
    plt.title("HWFE Distribution")
    plt.savefig(os.path.join(plt_root, "error", "hwfe_dist.png"), bbox_inches = "tight")
    plt.close()

    plt.hist(pes[direction == 0], bins='auto', color="black", alpha=.5, label="Stationary")
    plt.hist(pes[direction < 0], bins='auto', color="blue", alpha=.5, label="Decreasing")
    plt.hist(pes[direction > 0], bins='auto', color="red", alpha=.5, label="Increasing")
    plt.legend()
    plt.xlabel('Pointing Error (")')
    plt.ylabel("Counts")
    plt.title("Pointing Error Distribution")
    plt.savefig(os.path.join(plt_root, "error", "pe_dist.png"), bbox_inches = "tight")
    plt.close()

    # Now by angle
    plt.scatter(angle[direction == 0], hwfes[direction == 0], color="black", alpha=.5, label="Stationary")
    plt.scatter(angle[direction < 0], hwfes[direction < 0],  color="blue", alpha=.5, label="Decreasing")
    plt.scatter(angle[direction > 0], hwfes[direction > 0],  color="red", alpha=.5, label="Increasing")
    plt.legend()
    plt.xlabel("Angle (deg)")
    plt.suptitle(f"HWFE by Angle")
    plt.savefig(os.path.join(plt_root, "error", "hwfe_ang.png"), bbox_inches = "tight")
    plt.close()

    plt.scatter(angle[direction == 0], pes[direction == 0], color="black", alpha=.5, label="Stationary")
    plt.scatter(angle[direction < 0], pes[direction < 0],  color="blue", alpha=.5, label="Decreasing")
    plt.scatter(angle[direction > 0], pes[direction > 0],  color="red", alpha=.5, label="Increasing")
    plt.legend()
    plt.xlabel("Angle (deg)")
    plt.suptitle(f"Pointing Error by Angle")
    plt.savefig(os.path.join(plt_root, "error", "pe_ang.png"), bbox_inches = "tight")
    plt.close()

def _plot_transform(data, ref, get_transform, plt_root, logger):
    logger.info("Plotting transformation information")
    for elem in data.keys():
        logger.info("\tGetting transforms for %s", elem)
        if "tod" not in data[elem] or data[elem]["tod"].size == 0:
            logger.error("\t\t%s TOD not found! Skipping...", elem)
            continue
        if len(data[elem]["points"]) < 4:
            logger.error("\t\tOnly %d points found! Skipping...", len(data[elem]["points"]))
            continue
        src = data[elem]["tod"]
        dst = ref[elem]
        sfts = np.zeros((len(src), 3)) + np.nan
        rots = np.zeros((len(src), 3)) + np.nan
        for i, _src in enumerate(src):
            try:
                aff, sft = get_transform(_src, dst)
            except ValueError:
                logger.error("\t\tFailed to get transform for a data point! Filling with nans")
                continue
            _, _, rot = decompose_affine(aff)
            rot = np.rad2deg(decompose_rotation(rot))
            sfts[i] = sft
            rots[i] = rot 

        # Lets plot shift and rotation
        # First with time
        os.makedirs(os.path.join(plt_root, elem), exist_ok=True)
        t = np.arange(len(sfts))
        plt.scatter(t, sfts[:, 0], alpha=.5, label="x")
        plt.scatter(t, sfts[:, 1], alpha=.5, label="y")
        plt.scatter(t, sfts[:, 2], alpha=.5, label="z")
        plt.legend()
        plt.xlabel("Measurement #")
        plt.ylabel("Shift (mm)")
        plt.title(f"{elem} Shifts over time")
        plt.savefig(os.path.join(plt_root, elem, "shift_tod.png"), bbox_inches = "tight")
        plt.close()

        plt.scatter(t, rots[:, 0], alpha=.5, label="x")
        plt.scatter(t, rots[:, 1], alpha=.5, label="y")
        plt.scatter(t, rots[:, 2], alpha=.5, label="z")
        plt.legend()
        plt.xlabel("Measurement #")
        plt.ylabel("Rotation (deg)")
        plt.title(f"{elem} Rotation over time")
        plt.savefig(os.path.join(plt_root, elem, "rot_tod.png"), bbox_inches = "tight")
        plt.close()

        # Now by angle
        angle = data[elem]["angle_tod"]
        direction = data[elem]["direction_tod"]
        _, axs = plt.subplots(3, 1, sharex=True)
        for i, dim in enumerate(["x", "y", "z"]):
            axs[i].scatter(angle[direction == 0], sfts[direction == 0, i], color="black", marker="o", alpha=.25, label="Stationary")
            axs[i].scatter(angle[direction < 0], sfts[direction < 0, i], color="blue", marker="x", alpha=.25, label="Decreasing")
            axs[i].scatter(angle[direction > 0], sfts[direction > 0, i], color="red", marker="+", alpha=.25, label="Increasing")
            axs[i].set_ylabel(f"{dim} shift (mm)")
        axs[0].legend()
        axs[-1].set_xlabel("Angle (deg)")
        plt.suptitle(f"{elem} Shifts by Angle")
        plt.savefig(os.path.join(plt_root, elem, "shift_ang.png"), bbox_inches = "tight")
        plt.close()

        _, axs = plt.subplots(3, 1, sharex=True)
        for i, dim in enumerate(["x", "y", "z"]):
            axs[i].scatter(angle[direction == 0], rots[direction == 0, i], color="black", marker="o", alpha=.25, label="Stationary")
            axs[i].scatter(angle[direction < 0], rots[direction < 0, i], color="blue", marker="x", alpha=.25, label="Decreasing")
            axs[i].scatter(angle[direction > 0], rots[direction > 0, i], color="red", marker="+", alpha=.25, label="Increasing")
            axs[i].set_ylabel(f"{dim} rotation (deg)")
        axs[0].legend()
        axs[-1].set_xlabel("Angle (deg)")
        plt.suptitle(f"{elem} Rotation by Angle")
        plt.savefig(os.path.join(plt_root, elem, "rot_ang.png"), bbox_inches = "tight")
        plt.close()

def _add_tod(data, logger):
    # TODO: dataclass just for TOD?
    npoints = np.hstack([[len(data[elem][point]["data"]) for point in data[elem].keys()] for elem in data.keys()])
    if not np.all(npoints == npoints[0]):
        raise ValueError("Not all points have the same number of measurements!")
    for elem in data.keys():
        logger.info("\tConstructing TOD for %s", elem)
        angle = np.atleast_2d(np.array([data[elem][point]["angle"] for point in LABELS[elem] if point in data[elem]]))
        direction = np.atleast_2d(np.array([data[elem][point]["direction"] for point in LABELS[elem] if point in data[elem]]))
        if not (np.isclose(angle, angle[0]) | np.isnan(angle)).all():
            logger.error("\t\tAngles don't match across all points! Skipping...")
            continue
        angle = angle[0]
        direction = direction[0]
        points = [point for point in LABELS[elem] if point in data[elem]]
        src = np.swapaxes(np.atleast_3d(np.array([data[elem][point]["data"] for point in points])), 0, 1)
        if src.size == 0:
            logger.info("\t\tNo data found! Not making TOD")
        data[elem]["tod"] = src
        data[elem]["angle_tod"] = angle
        data[elem]["direction_tod"] = direction
        data[elem]["points"] = points
        data[elem]["meas_number"] = np.arange(len(src))
    return data

def _plot_path(data, plt_root, logger):
    os.makedirs(os.path.join(plt_root, "trajectory"), exist_ok=True)
    for elem in data.keys():
        logger.info("Plotting %s trajectory", elem)
        if "tod" not in data[elem] or data[elem]["tod"].size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        # Plot raw trajectories
        for xax, xlab in [("angle_tod", "Angle (deg)"), ("meas_number", "Measurement (#)")]:
            _, axs = plt.subplots(3, len(data[elem]["points"]), sharex=True, sharey=False, figsize=(24, 20), layout="constrained")
            axs = np.reshape(np.array(axs), (int(3), len(data[elem]["points"])))
            for i, point in enumerate(data[elem]["points"]):
                dat = data[elem]["tod"]
                x = data[elem][xax]
                direction = data[elem]["direction_tod"]
                for j, dim in enumerate(["x", "y", "z"]):
                    axs[j, i].scatter(x[direction == 0], dat[direction == 0, i, j], color="black", marker="o", alpha=.25, label="Stationary")
                    axs[j, i].scatter(x[direction < 0], dat[direction < 0, i, j], color="blue", marker="x", alpha=.25, label="Decreasing")
                    axs[j, i].scatter(x[direction > 0], dat[direction > 0, i, j], color="red", marker="+", alpha=.25, label="Increasing")
                    axs[0, i].set_title(point)
                    axs[-1, i].set_xlabel(xlab)
                    axs[j, 0].set_ylabel(f"{dim} (mm)")
            axs[-1, 0].legend()
            plt.suptitle(f"{elem} Trajectory")
            plt.savefig(os.path.join(plt_root, "trajectory", f"{elem}_trajectory_{xax}.png"), bbox_inches = "tight")
            plt.close()

    
def _plot_traj_error(data, plt_root, logger):
    for elem in data.keys():
        logger.info("\t\tPlotting %s trajectory error", elem)
        if "tod" not in data[elem] or data[elem]["tod"].size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        _, axs = plt.subplots(2, len(data[elem]["points"]), sharex=False, sharey=False, figsize=(24, 20), layout="constrained")
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
                rmss[0] += [np.linalg.norm(np.nanstd(_dat[dire == 0], axis = 0))]
                rmss[1] += [np.linalg.norm(np.nanstd(_dat[dire < 0], axis = 0))]
                rmss[2] += [np.linalg.norm(np.nanstd(_dat[dire > 0], axis = 0))]
                diffs[0] += [pdist(_dat[dire == 0])]
                diffs[1] += [pdist(_dat[dire < 0])]
                diffs[2] += [pdist(_dat[dire > 0])]

            # Plot distribution
            for j, (label, color) in enumerate([("Stationary", "black"), ("Decreasing", "blue"), ("Increasing", "red")]):
                d = np.hstack(diffs[j]).ravel()
                if len(d) == 0:
                    continue
                axs[0, i].hist(d, bins='auto', color=color, alpha=.5, label=label)
            axs[0, i].legend()
            axs[0, i].set_xlabel("Distance Between Repeated Points (mm)")
            axs[0, i].set_ylabel("Count")
            axs[0, i].set_title(point)
            axs[0, i].autoscale()

            # Plot rms
            axs[1, i].scatter(ang_u, rmss[0], color="black", marker="o", alpha=.25, label="Stationary")
            axs[1, i].scatter(ang_u, rmss[1], color="blue", marker="x", alpha=.25, label="Decreasing")
            axs[1, i].scatter(ang_u, rmss[2], color="red", marker="+", alpha=.25, label="Increasing")
            axs[1, i].set_xlabel("Angle (deg)")
            axs[1, i].set_ylabel("RMS (mm)")
        plt.suptitle(f"{elem} Trajectory Error")
        plt.savefig(os.path.join(plt_root, "trajectory", f"{elem}_error.png"), bbox_inches = "tight")
        plt.close()


def _get_angle_cont(data, start, sep, logger):
    logger.warning("\t\tReconstructing angle from continious data, this is approximate and should not be used for pointing corrections! Trajectory Errors will also only be approximate!")

    # Get the best fit radius and center
    sphere = Sphere.best_fit(data)
    logger.debug("\t\tFit a radius of %s and a center at %s", str(sphere.radius), str(sphere.point))

    # Recover the angle
    d_data = data - sphere.point
    # Assuming optical global coordinates)
    theta = np.rad2deg(np.arctan2(d_data[:, 0], d_data[:, 2]))

    # Correct based on start position
    theta -= theta[0] - start

    # Convert delta to an angle delta
    dtheta = np.rad2deg(sep/sphere.radius)/32.
    
    # Quantize
    if dtheta != 0:
        theta_corr = theta - theta[0]
        theta_corr = dtheta * (theta_corr//dtheta) + theta[0]
    else:
        theta_corr = np.ones_like(theta)*start

    # Figure out left vs right vs static
    direction = np.diff(theta_corr)
    # Lets just make the last point keep the same direction
    direction = np.hstack((direction, [direction[-1]]))
    
    return theta_corr, direction

def _get_angle_step(data, start, sep, logger):
    raise NotImplementedError("Step wise angle reconstruction not implemented")
    
def get_angle(data, mode, start, sep, logger):
    logger.info("\tReconstructing angle in %s mode using a start of %f deg and a seperation of %f", mode, start, sep)
    if mode == "continious":
        return _get_angle_cont(data, start, sep, logger)
    elif mode == "step":
        return _get_angle_step(data, start, sep, logger)
    raise ValueError(f"Invalid mode: {mode}")

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
            angle, direction = get_angle(dat["data"], dat["mode"], dat["start"], dat["sep"], logger)
            data[elem][point]["angle"] = angle
            data[elem][point]["direction"] = direction 


    # Check motion of each element
    data = _add_tod(data, logger)
    _plot_path(data, plt_root, logger)
    _plot_traj_error(data, plt_root, logger)
    _plot_transform(data, ref, get_transform, plt_root, logger)
    _plot_point_and_hwfe(data, ref, get_transform, plt_root, logger)
