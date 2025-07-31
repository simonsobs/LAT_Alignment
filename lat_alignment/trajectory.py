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
from megham.transform import decompose_affine, decompose_rotation, get_affine, get_rigid, apply_transform
from scipy.spatial.distance import pdist
from scipy.spatial.transform import Rotation
from skspatial.objects import Sphere

from .error import get_hwfe, get_pointing_error
from .io import load_tracker

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
    os.makedirs(os.path.join(plt_root, "error"), exist_ok=True)
    t = np.arange(npts)
    plt.scatter(t, hwfes, alpha=0.5)
    plt.scatter(t[missing], hwfes[missing], color="gray", alpha=1, marker="1")
    plt.xlabel("Measurement #")
    plt.ylabel("HWFE (um-rms)")
    plt.title(f"HWFE over time")
    plt.savefig(os.path.join(plt_root, "error", "hwfe_tod.png"), bbox_inches="tight")
    plt.close()

    plt.scatter(t, pes, alpha=0.5)
    plt.scatter(t[missing], pes[missing], color="gray", alpha=1, marker="1")
    plt.xlabel("Measurement #")
    plt.ylabel('Pointing Error (")')
    plt.title(f"Pointing Error over time")
    plt.savefig(os.path.join(plt_root, "error", "pe_tod.png"), bbox_inches="tight")
    plt.close()

    # Plot distribution
    if len(direction == 0) > 0:
        plt.hist(
            hwfes[direction == 0],
            bins="auto",
            color="black",
            alpha=0.5,
            label="Stationary",
        )
    if len(direction < 0) > 0:
        plt.hist(
            hwfes[direction < 0],
            bins="auto",
            color="blue",
            alpha=0.5,
            label="Decreasing",
        )
    if len(direction > 0) > 0:
        plt.hist(
            hwfes[direction > 0],
            bins="auto",
            color="red",
            alpha=0.5,
            label="Increasing",
        )
    plt.legend()
    plt.xlabel("HWFE (um-rms)")
    plt.ylabel("Counts")
    plt.title("HWFE Distribution")
    plt.savefig(os.path.join(plt_root, "error", "hwfe_dist.png"), bbox_inches="tight")
    plt.close()

    if len(direction == 0) > 0:
        plt.hist(
            pes[direction == 0],
            bins="auto",
            color="black",
            alpha=0.5,
            label="Stationary",
        )
    if len(direction < 0) > 0:
        plt.hist(
            pes[direction < 0], bins="auto", color="blue", alpha=0.5, label="Decreasing"
        )
    if len(direction > 0) > 0:
        plt.hist(
            pes[direction > 0], bins="auto", color="red", alpha=0.5, label="Increasing"
        )
    plt.legend()
    plt.xlabel('Pointing Error (")')
    plt.ylabel("Counts")
    plt.title("Pointing Error Distribution")
    plt.savefig(os.path.join(plt_root, "error", "pe_dist.png"), bbox_inches="tight")
    plt.close()

    # Now by angle
    plt.scatter(
        angle[direction == 0],
        hwfes[direction == 0],
        color="black",
        alpha=0.5,
        label="Stationary",
    )
    plt.scatter(
        angle[direction < 0],
        hwfes[direction < 0],
        color="blue",
        alpha=0.5,
        label="Decreasing",
    )
    plt.scatter(
        angle[direction > 0],
        hwfes[direction > 0],
        color="red",
        alpha=0.5,
        label="Increasing",
    )
    plt.scatter(angle[missing], hwfes[missing], color="gray", alpha=1, marker="1")
    plt.legend()
    plt.xlabel("Angle (deg)")
    plt.suptitle(f"HWFE by Angle")
    plt.savefig(os.path.join(plt_root, "error", "hwfe_ang.png"), bbox_inches="tight")
    plt.close()

    plt.scatter(
        angle[direction == 0],
        pes[direction == 0],
        color="black",
        alpha=0.5,
        label="Stationary",
    )
    plt.scatter(
        angle[direction < 0],
        pes[direction < 0],
        color="blue",
        alpha=0.5,
        label="Decreasing",
    )
    plt.scatter(
        angle[direction > 0],
        pes[direction > 0],
        color="red",
        alpha=0.5,
        label="Increasing",
    )
    plt.scatter(angle[missing], pes[missing], color="gray", alpha=1, marker="1")
    plt.legend()
    plt.xlabel("Angle (deg)")
    plt.suptitle(f"Pointing Error by Angle")
    plt.savefig(os.path.join(plt_root, "error", "pe_ang.png"), bbox_inches="tight")
    plt.close()


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
        os.makedirs(os.path.join(plt_root, elem), exist_ok=True)
        t = np.arange(len(sfts))
        plt.scatter(t, sfts[:, 0], alpha=0.5, label="x")
        plt.scatter(t, sfts[:, 1], alpha=0.5, label="y")
        plt.scatter(t, sfts[:, 2], alpha=0.5, label="z")
        plt.scatter(t[missing], sfts[missing, 0], color="gray", marker="1")
        plt.scatter(t[missing], sfts[missing, 1], color="gray", marker="1")
        plt.scatter(t[missing], sfts[missing, 2], color="gray", marker="1")
        plt.legend()
        plt.xlabel("Measurement #")
        plt.ylabel("Shift (mm)")
        plt.title(f"{elem} Shifts over time")
        plt.savefig(os.path.join(plt_root, elem, "shift_tod.png"), bbox_inches="tight")
        plt.close()

        plt.scatter(t, rots[:, 0], alpha=0.5, label="x")
        plt.scatter(t, rots[:, 1], alpha=0.5, label="y")
        plt.scatter(t, rots[:, 2], alpha=0.5, label="z")
        plt.scatter(t[missing], rots[missing, 0], color="gray", marker="1")
        plt.scatter(t[missing], rots[missing, 1], color="gray", marker="1")
        plt.scatter(t[missing], rots[missing, 2], color="gray", marker="1")
        plt.legend()
        plt.xlabel("Measurement #")
        plt.ylabel("Rotation (deg)")
        plt.title(f"{elem} Rotation over time")
        plt.savefig(os.path.join(plt_root, elem, "rot_tod.png"), bbox_inches="tight")
        plt.close()

        plt.scatter(t, scales[:, 0], alpha=0.5, label="x")
        plt.scatter(t, scales[:, 1], alpha=0.5, label="y")
        plt.scatter(t, scales[:, 2], alpha=0.5, label="z")
        plt.scatter(t[missing], scales[missing, 0], color="gray", marker="1")
        plt.scatter(t[missing], scales[missing, 1], color="gray", marker="1")
        plt.scatter(t[missing], scales[missing, 2], color="gray", marker="1")
        plt.legend()
        plt.xlabel("Measurement #")
        plt.ylabel("Scale Factor")
        plt.title(f"{elem} Scale over time")
        plt.savefig(os.path.join(plt_root, elem, "scale_tod.png"), bbox_inches="tight")
        plt.close()

        # Now by angle
        angle = data[elem]["angle_tod"]
        direction = data[elem]["direction_tod"]
        _, axs = plt.subplots(3, 1, sharex=True)
        for i, dim in enumerate(["x", "y", "z"]):
            axs[i].scatter(
                angle[direction == 0],
                sfts[direction == 0, i],
                color="black",
                marker="o",
                alpha=0.25,
                label="Stationary",
            )
            axs[i].scatter(
                angle[direction < 0],
                sfts[direction < 0, i],
                color="blue",
                marker="x",
                alpha=0.25,
                label="Decreasing",
            )
            axs[i].scatter(
                angle[direction > 0],
                sfts[direction > 0, i],
                color="red",
                marker="+",
                alpha=0.25,
                label="Increasing",
            )
            axs[i].scatter(angle[missing], sfts[missing, i], color="gray", marker="1")
            axs[i].set_ylabel(f"{dim} shift (mm)")
        axs[0].legend()
        axs[-1].set_xlabel("Angle (deg)")
        plt.suptitle(f"{elem} Shifts by Angle")
        plt.savefig(os.path.join(plt_root, elem, "shift_ang.png"), bbox_inches="tight")
        plt.close()

        _, axs = plt.subplots(3, 1, sharex=True)
        for i, dim in enumerate(["x", "y", "z"]):
            axs[i].scatter(
                angle[direction == 0],
                rots[direction == 0, i],
                color="black",
                marker="o",
                alpha=0.25,
                label="Stationary",
            )
            axs[i].scatter(
                angle[direction < 0],
                rots[direction < 0, i],
                color="blue",
                marker="x",
                alpha=0.25,
                label="Decreasing",
            )
            axs[i].scatter(
                angle[direction > 0],
                rots[direction > 0, i],
                color="red",
                marker="+",
                alpha=0.25,
                label="Increasing",
            )
            axs[i].scatter(angle[missing], rots[missing, i], color="gray", marker="1")
            axs[i].set_ylabel(f"{dim} rotation (deg)")
        axs[0].legend()
        axs[-1].set_xlabel("Angle (deg)")
        plt.suptitle(f"{elem} Rotation by Angle")
        plt.savefig(os.path.join(plt_root, elem, "rot_ang.png"), bbox_inches="tight")
        plt.close()

        _, axs = plt.subplots(3, 1, sharex=True)
        for i, dim in enumerate(["x", "y", "z"]):
            axs[i].scatter(
                angle[direction == 0],
                scales[direction == 0, i],
                color="black",
                marker="o",
                alpha=0.25,
                label="Stationary",
            )
            axs[i].scatter(
                angle[direction < 0],
                scales[direction < 0, i],
                color="blue",
                marker="x",
                alpha=0.25,
                label="Decreasing",
            )
            axs[i].scatter(
                angle[direction > 0],
                scales[direction > 0, i],
                color="red",
                marker="+",
                alpha=0.25,
                label="Increasing",
            )
            axs[i].scatter(angle[missing], scales[missing, i], color="gray", marker="1")
            axs[i].set_ylabel(f"{dim} scale")
        axs[0].legend()
        axs[-1].set_xlabel("Angle (deg)")
        plt.suptitle(f"{elem} Scale by Angle")
        plt.savefig(os.path.join(plt_root, elem, "scale_ang.png"), bbox_inches="tight")
        plt.close()

        # Plot resids
        for xax, xlab in [
            ("angle_tod", "Angle (deg)"),
            ("meas_number", "Measurement (#)"),
        ]:
            _, axs = plt.subplots(
                3,
                len(data[elem]["points"]),
                sharex=True,
                sharey=False,
                figsize=(24, 20),
                layout="constrained",
            )
            axs = np.reshape(np.array(axs), (int(3), len(data[elem]["points"])))
            for i, point in enumerate(data[elem]["points"]):
                x = data[elem][xax]
                direction = data[elem]["direction_tod"]
                for j, dim in enumerate(["x", "y", "z"]):
                    axs[j, i].scatter(
                        x[direction == 0],
                        resids[direction == 0, i, j],
                        color="black",
                        marker="o",
                        alpha=0.5,
                        label="Stationary",
                    )
                    axs[j, i].scatter(
                        x[direction < 0],
                        resids[direction < 0, i, j],
                        color="blue",
                        marker="x",
                        alpha=0.5,
                        label="Decreasing",
                    )
                    axs[j, i].scatter(
                        x[direction > 0],
                        resids[direction > 0, i, j],
                        color="red",
                        marker="+",
                        alpha=0.5,
                        label="Increasing",
                    )
                    axs[0, i].set_title(point)
                    axs[-1, i].set_xlabel(xlab)
                    axs[j, 0].set_ylabel(f"{dim} (mm)")
            axs[-1, 0].legend()
            plt.suptitle(f"{elem} Residuals")
            plt.savefig(
                os.path.join(plt_root, elem, f"resids_{xax}.png"),
                bbox_inches="tight",
            )
            plt.close()


def _plot_path(data, plt_root, logger):
    os.makedirs(os.path.join(plt_root, "trajectory"), exist_ok=True)
    for elem in data.keys():
        logger.info("Plotting %s trajectory", elem)
        if "tod" not in data[elem] or data[elem]["tod"].size == 0:
            logger.warning("\tNo TOD found! Skipping...")
            continue
        # Plot raw trajectories
        for xax, xlab in [
            ("angle_tod", "Angle (deg)"),
            ("meas_number", "Measurement (#)"),
        ]:
            _, axs = plt.subplots(
                3,
                len(data[elem]["points"]),
                sharex=True,
                sharey=False,
                figsize=(24, 20),
                layout="constrained",
            )
            axs = np.reshape(np.array(axs), (int(3), len(data[elem]["points"])))
            for i, point in enumerate(data[elem]["points"]):
                dat = data[elem]["tod"]
                x = data[elem][xax]
                direction = data[elem]["direction_tod"]
                for j, dim in enumerate(["x", "y", "z"]):
                    axs[j, i].scatter(
                        x[direction == 0],
                        dat[direction == 0, i, j],
                        color="black",
                        marker="o",
                        alpha=0.5,
                        label="Stationary",
                    )
                    axs[j, i].scatter(
                        x[direction < 0],
                        dat[direction < 0, i, j],
                        color="blue",
                        marker="x",
                        alpha=0.5,
                        label="Decreasing",
                    )
                    axs[j, i].scatter(
                        x[direction > 0],
                        dat[direction > 0, i, j],
                        color="red",
                        marker="+",
                        alpha=0.5,
                        label="Increasing",
                    )
                    axs[0, i].set_title(point)
                    axs[-1, i].set_xlabel(xlab)
                    axs[j, 0].set_ylabel(f"{dim} (mm)")
            axs[-1, 0].legend()
            plt.suptitle(f"{elem} Trajectory")
            plt.savefig(
                os.path.join(plt_root, "trajectory", f"{elem}_trajectory_{xax}.png"),
                bbox_inches="tight",
            )
            plt.close()


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
            os.path.join(plt_root, "trajectory", f"{elem}_error.png"),
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
    _plot_transform(data, ref, get_transform, plt_root, logger, cfg.get("skip_missing", False))
    _plot_point_and_hwfe(data, ref, get_transform, plt_root, logger, cfg.get("skip_missing", False))
