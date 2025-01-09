"""
Main driver script for running the alignment.
You typically want to use the `lat_alignment` entrypoint rather than
calling this directly.
"""

import argparse
import logging
import os
from functools import partial
from importlib.resources import files

import megham.transform as mt
import numpy as np
import yaml
from numpy.typing import NDArray
from pqdm.processes import pqdm

from . import adjustments as adj
from . import bearing as br
from . import io
from . import mirror as mir
from . import photogrammetry as pg
from . import transforms as tf


def log_alignment(alignment, logger):
    aff, shift = alignment
    scale, shear, rot = mt.decompose_affine(aff)
    logger.debug("\tFinal shift is %s mm", str(shift))
    logger.debug(
        "\tFinal rotation is %s deg", str(np.rad2deg(mt.decompose_rotation(rot)))
    )
    logger.debug("\tFinal scale is %s", str(scale))
    logger.debug("\tFinal shear is %s", str(shear))


def adjust_panel(panel: mir.Panel, mnum: int, cfg: dict) -> NDArray[np.float32]:
    """
    Helper function to get the adjustments for a single panel.

    Parameters
    ----------
    panel : mir.Panel
        The mirror panel to adjust.
    mnum : int
        The mirror number.
        1 for the primary and 2 for the secondary.
    cfg : dict
        The configuration dictionairy.

    Returns
    -------
    adjustments : NDArray[np.float32]
        The adjustments to make for the panel.
        This is a 17 element array with the following structure:
        `[mnum, panel_row, panel_col, dx, dy, d_adj1, ..., d_adj5, dx_err, dy_err, d_adj1_err, ..., d_adj5_err]`.
    """
    adjustments = np.zeros(17, np.float32)
    adjustments[0] = mnum
    adjustments[1] = panel.row
    adjustments[2] = panel.col
    meas_adj = panel.meas_adj.copy()
    meas_adj[:, 2] += panel.meas_adj_resid
    meas_surface = panel.meas_surface.copy()
    meas_surface[:, 2] += panel.meas_adj_resid
    dx, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(
        panel.can_surface, meas_surface, meas_adj, **cfg.get("adjust", {})
    )
    adjustments[3:] = np.array(
        [dx, dy] + list(d_adj) + [dx_err, dy_err] + list(d_adj_err)
    )

    return adjustments


def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    parser.add_argument(
        "--log_level", "-l", default="INFO", help="the log level to use"
    )
    args = parser.parse_args()
    logging.basicConfig()
    logger = logging.getLogger("lat_alignment")
    logger.setLevel(args.log_level.upper())
    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    mode = cfg.get("mode", "panel")
    cfgdir = os.path.dirname(os.path.abspath(args.config))
    meas_file = os.path.abspath(os.path.join(cfgdir, cfg["measurement"]))
    title_str = cfg["title"]
    logger.info("Begining alignment %s in %s mode", title_str, mode)
    logger.debug("Using measurement file: %s", meas_file)

    dat_dir = os.path.abspath(os.path.join(cfgdir, cfg.get("data_dir", "/")))
    if "data_dir" in cfg:
        logger.info("Using data files from %s", dat_dir)
        ref_path = os.path.join(dat_dir, "reference.yaml")
    else:
        logger.info("Using packaged data files")
        ref_path = str(files("lat_alignment.data").joinpath("reference.yaml"))
    with open(ref_path) as file:
        reference = yaml.safe_load(file)
    dataset = io.load_photo(meas_file, **cfg.get("load", {}))

    if mode == "panel":
        mirror = cfg["mirror"]
        if mirror == "primary":
            mnum = 1
        elif mirror == "secondary":
            mnum = 2
        else:
            raise ValueError(f"Invalid mirror: {mirror}")
        logger.info("Aligning panels for the %s mirror", mirror)

        if "data_dir" in cfg:
            corner_path = os.path.join(dat_dir, f"{mirror}_corners.yaml")
            adj_path = os.path.join(dat_dir, f"{mirror}_adj.csv")
        else:
            corner_path = str(
                files("lat_alignment.data").joinpath(f"{mirror}_corners.yaml")
            )
            adj_path = str(files("lat_alignment.data").joinpath(f"{mirror}_adj.csv"))

        # load files
        corners = io.load_corners(corner_path)
        adjusters = io.load_adjusters(adj_path, mirror)

        # init, fit, and plot panels
        dataset, _ = pg.align_photo(
            dataset, reference, True, mirror, **cfg.get("align_photo", {})
        )
        dataset, _ = mir.remove_cm(
            dataset, mirror, cfg.get("compensate", 0), **cfg.get("common_mode", {})
        )
        panels = mir.gen_panels(
            mirror,
            dataset,
            corners,
            adjusters,
            cfg.get("compensate", 0),
            cfg.get("adjuster_radius", 100),
        )
        logger.info("Found measurements for %d panels", len(panels))
        fig = mir.plot_panels(panels, title_str, vmax=cfg.get("vmax", None))
        fig.savefig(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.png"))

        # calc and save adjustments
        logger.info("Caluculating adjustments")
        _adjust = partial(adjust_panel, mnum=mnum, cfg=cfg)
        adjustments = np.vstack(pqdm(panels, _adjust, n_jobs=8))
        order = np.lexsort((adjustments[2], adjustments[1], adjustments[0]))
        adjustments = adjustments[order]
        np.savetxt(
            os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.csv"),
            adjustments,
            fmt=["%d", "%d", "%d"] + ["%.5f"] * 14,
        )
    elif mode == "optical":
        align_to = cfg["align_to"]
        if align_to not in ["primary", "secondary", "receiver", "bearing"]:
            raise ValueError(f"Invalid element specified for 'align_to': {align_to}")
        logger.info("Aligning all optical elements to the %s", align_to)
        dataset, _ = pg.align_photo(
            dataset, reference, False, "all", False, **cfg.get("align_photo", {})
        )

        # Load data and compute the transformation to align with the model
        # We want to put all the transformations into opt_global
        elements = {}  # {element_name : full_alignment}
        identity = (np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32))
        try:
            meas, alignment = pg.align_photo(
                dataset.copy(), reference, True, "primary", **cfg.get("align_photo", {})
            )
            meas, common_mode = mir.remove_cm(
                meas, "primary", cfg.get("compensate", 0), **cfg.get("common_mode", {})
            )
            full_alignment = mt.compose_transform(*alignment, *common_mode)
            full_alignment = tf.affine_basis_transform(
                full_alignment[0], full_alignment[1], "opt_primary", "opt_global", False
            )
            log_alignment(full_alignment, logger)
        except Exception as e:
            logger.warning(
                "Failed to load primary due to error: \n\t%s\n if the primary was not in your data you can ignore this.",
                str(e),
            )
            meas = {}
            full_alignment = identity
        if len(meas) >= 4:
            elements["primary"] = full_alignment
        try:
            meas, alignment = pg.align_photo(
                dataset.copy(), reference, True, "primary", **cfg.get("align_photo", {})
            )
            meas, common_mode = mir.remove_cm(
                meas,
                "secondary",
                cfg.get("compensate", 0),
                **cfg.get("common_mode", {}),
            )
            full_alignment = mt.compose_transform(*alignment, *common_mode)
            full_alignment = tf.affine_basis_transform(
                full_alignment[0],
                full_alignment[1],
                "opt_secondary",
                "opt_global",
                False,
            )
            log_alignment(full_alignment, logger)
        except Exception as e:
            logger.warning(
                "Failed to load secondary due to error: \n\t%s\n if the secondary was not in your data you can ignore this.",
                str(e),
            )
            meas = {}
            full_alignment = identity
        if len(meas) >= 4:
            elements["secondary"] = full_alignment
        try:
            meas, alignment = pg.align_photo(
                dataset.copy(),
                reference,
                False,
                "bearing",
                **cfg.get("align_photo", {}),
            )
            meas, cyl_fit = br.cylinder_fit(meas)
            full_alignment = mt.compose_transform(*alignment, *cyl_fit)
            log_alignment(full_alignment, logger)
        except Exception as e:
            print(
                f"Failed to load bearing due to error: \n\t{e}\n if the bearing was not in your data you can ignore this."
            )
            meas = {}
            full_alignment = identity
        if len(meas) >= 4:
            elements["bearing"] = full_alignment
        try:
            meas, full_alignment = pg.align_photo(
                dataset.copy(),
                reference,
                False,
                "receiver",
                **cfg.get("align_photo", {}),
            )
            log_alignment(full_alignment, logger)
        except Exception as e:
            print(
                f"Failed to load receiver due to error: \n\t{e}\n if the receiver was not in your data you can ignore this."
            )
            meas = {}
            full_alignment = identity
        if len(meas) >= 4:
            elements["receiver"] = full_alignment
        if len(elements) < 2:
            raise ValueError(
                f"Only {len(elements)} optical elements found in measurment. Can't align!"
            )
        if align_to not in elements:
            raise ValueError(
                f"Specified 'align_to' element ({align_to}) not found in measurment. Can't align!"
            )
        logger.info(
            "Found %d optical elements in measurement: %s",
            len(elements),
            str(list(elements.keys())),
        )

        # Now combine with the align_to alignment
        logger.info("Composing transforms to align with %s fixed", align_to)
        transforms = {}
        align_to_inv = mt.invert_transform(*elements[align_to])
        for element, full_transform in elements.items():
            aff, sft = mt.compose_transform(*full_transform, *align_to_inv)
            scale, shear, rot = mt.decompose_affine(aff)
            rot = np.rad2deg(mt.decompose_rotation(rot))
            transform = {
                "shift": sft.tolist(),
                "rot": rot.tolist(),
                "scale": scale.tolist(),
                "shear": shear.tolist(),
            }
            transforms[element] = transform

        # Save
        with open(
            os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.yaml"), "w"
        ) as file:
            yaml.dump(transforms, file)

    else:
        raise ValueError(f"Invalid mode: {mode}")
    logger.info("Outputs can be found in %s", cfgdir)
