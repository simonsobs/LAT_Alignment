"""
Script for calculating HWFE and pointing error.
Also tells you how to move whatever elements are included.
"""

import argparse
import logging
import os
import sys
from copy import deepcopy
from functools import partial

import matplotlib.pyplot as plt
import numpy as np
from megham.transform import (
    apply_transform,
    decompose_affine,
    decompose_rotation,
    get_affine,
    get_rigid,
)
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

from .io import load_tracker
from .transforms import coord_transform, affine_basis_transform

elements = ["primary", "secondary", "receiver"]
hwfe_factors = {
    "secondary": [0.00034, 0.00076, 0.0029, 0.044, 0.042, 0.014],
    "receiver": [0.0, 0.0, 0.0026, 0.0, 0.0, 0.0],
}
mm_to_um = 1000
rad_to_arcsec = 3600 * 180 / np.pi


def get_hwfe(data, get_transform, add_err=False) -> float:
    # Put everything in M1 coordinates
    data_m1 = deepcopy(data)
    for element in elements:
        dat = data_m1[element]
        if add_err:
            dat += np.nan_to_num(data_m1[f"{element}_err"])
        data_m1[element] = coord_transform(dat, "opt_global", "opt_primary")
        data_m1[f"{element}_ref"] = coord_transform(
            data_m1[f"{element}_ref"], "opt_global", "opt_primary"
        )

    # Transform for M1 perfect
    aff_m1, sft_m1 = get_transform(
        data_m1["primary"][data_m1["primary_msk"]],
        data_m1["primary_ref"][data_m1["primary_msk"]],
        method="mean",
    )

    hwfe = 0
    for element in hwfe_factors.keys():
        src = data_m1[element][data_m1[f"{element}_msk"]]
        dst = data_m1[f"{element}_ref"][data_m1[f"{element}_msk"]]

        # Apply the transform to align M1
        src = apply_transform(src, aff_m1, sft_m1)

        # Get the new transform
        aff, sft = get_transform(src, dst, method="mean")
        _, _, rot = decompose_affine(aff)
        rot = decompose_rotation(rot)

        # compute HWFE
        vals = np.hstack([sft * mm_to_um, rot * rad_to_arcsec]).ravel()
        hwfe += float(np.sum((np.array(hwfe_factors[element]) * vals) ** 2))
    return np.sqrt(hwfe)

def get_pointing_error(data, get_transform, add_err=False, thresh=.1):
    thresh = np.deg2rad(thresh/3600)
    rots = np.zeros((2, 3))
    # Get rotations
    for i, (element, factor) in enumerate([("primary", 1), ("secondary", 2)]):
        src = np.array(data[element])
        if add_err:
            src += np.nan_to_num(data[f"{element}_err"])
        # Put things in the local coords
        src = coord_transform(src, "opt_global", f"opt_{element}")[data[f"{element}_msk"]]
        dst = coord_transform(np.array(data[f"{element}_ref"]), "opt_global", f"opt_{element}")[data[f"{element}_msk"]]
        # Get rotation
        aff, _= get_transform(src, dst)
        *_, rot = decompose_affine(aff)
        rot = decompose_rotation(rot)
        rot[rot < thresh] = 0 # Help prevent float errors
        rot[-1] = 0 # clocking doesn't matter
        # Put into global coords
        aff = R.from_euler('xyz', rot, False).as_matrix()
        aff, _ = affine_basis_transform(aff, np.zeros(3, np.float32), f"opt_{element}", "opt_global")
        *_, rot = decompose_affine(aff)
        rot = decompose_rotation(rot)
        rot[rot < thresh] = 0
        rots[i] = rot*factor
    tot_rot = np.linalg.norm(np.sum(rots, 0))
    return 3600*np.rad2deg(tot_rot)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="path to data file")
    parser.add_argument(
        "--affine",
        "-a",
        action="store_true",
        help="Pass to compute affine instead of rigid rotation",
    )
    parser.add_argument(
        "--n_draws",
        "-n",
        default=10000,
        type=int,
        help="Number of draws from the error distribution to do when estimating HWFE error",
    )
    parser.add_argument(
        "--log_level", "-l", default="INFO", help="the log level to use"
    )
    args = parser.parse_args()
    logging.basicConfig()
    logger = logging.getLogger("lat_alignment")
    logger.setLevel(args.log_level.upper())

    # Pick the fitter
    get_transform = get_rigid
    transform_str = "rigid"
    if args.affine:
        get_transform = partial(get_affine, force_svd=True)
        transform_str = "affine"
    # Load data
    logger.info("Loading data from %s", args.path)
    ext = os.path.splitext(args.path)[1]
    if ext != ".yaml":
        raise ValueError("Data for HWFE script must be a yaml file")
    data = load_tracker(args.path)

    # Get the transform for each element assuming no error
    have_err = False
    for element in elements:
        logger.info("Getting transform for %s", element)
        src = np.array(data[element])
        dst = np.array(data[f"{element}_ref"])
        if np.all(np.isnan(src)):
            logger.info("\tElement is all nan!, assuming it is perfect")
            src = dst.copy()
            data[element] = src
        have = np.all(np.isfinite(src), axis=1)
        have_err += np.any(np.isfinite(data[f"{element}_err"]))
        if np.sum(have) < 3:
            raise ValueError(f"Only {np.sum(have)} points found!")
        data[f"{element}_msk"] = have
        aff, sft = get_transform(src[have], dst[have], method="mean")
        scale, shear, rot = decompose_affine(aff)
        rot = decompose_rotation(rot)
        logger.info("\tShift is %s mm", str(sft))
        logger.info("\tRotation is %s deg", str(np.rad2deg(rot)))
        logger.info("\tScale is %s", scale)
        logger.info("\tShear is %s", shear)

    # Get HWFE
    hwfe = get_hwfe(data, get_transform)
    logger.info("HWFE is %f", hwfe)
    
    # Get pointing offset
    po = get_pointing_error(data, get_transform)
    logger.info("Pointing offset is %f", po)

    # Error Propagation
    if not have_err:
        logger.info("No errors found")
        sys.exit()

    logger.info("Propagating errors")
    hwfe_werr = np.zeros(args.n_draws)
    po_werr = np.zeros(args.n_draws)
    rng = np.random.default_rng(12345)

    for i in tqdm(range(args.n_draws)):
        _data = deepcopy(data)
        _data["primary_err"] *= rng.normal(size=(4, 3))
        _data["secondary_err"] *= rng.normal(size=(4, 3))
        _data["receiver_err"] *= rng.normal(size=(4, 3))
        hwfe_werr[i] = get_hwfe(_data, get_transform, True)
        po_werr[i] = get_pointing_error(_data, get_transform, True)
    logger.info("\tStandard deviation of HWFE error dist is %f", np.std(hwfe_werr))
    logger.info("\tStandard deviation of pointing error dist is %f", np.std(po_werr))
    plt.hist(
        hwfe_werr,
        density=True,
        bins="auto",
        label=f"Mean: {np.mean(hwfe_werr):.2f}\nSTD: {np.std(hwfe_werr):.2f}",
        alpha=0.7,
    )
    plt.axvline(hwfe, label=f"Without Error: {hwfe:.2f}", color="black")
    plt.legend()
    plt.xlabel("HWFE (um-rms)")
    plt.ylabel("Density")
    plt.title(f"HWFE With Uncorrellated Error ({transform_str})")
    plt.savefig(os.path.splitext(args.path)[0] + f"_hwfe_error_{transform_str}.png")
    plt.close()
    plt.hist(
        po_werr,
        density=True,
        bins="auto",
        label=f"Mean: {np.mean(po_werr):.2f}\nSTD: {np.std(po_werr):.2f}",
        alpha=0.7,
    )
    plt.axvline(po, label=f"Without Error: {po:.2f}", color="black")
    plt.legend()
    plt.xlabel('Pointing Error (")')
    plt.ylabel("Density")
    plt.title(f"Pointing Error With Uncorrellated Error ({transform_str})")
    plt.savefig(os.path.splitext(args.path)[0] + f"_pointing_error_{transform_str}.png")
