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

import matplotlib.pyplot as plt
import megham.transform as mt
import numpy as np
import yaml
from numpy.typing import NDArray
from pqdm.processes import pqdm

from . import adjustments as adj
# from . import bearing as br
from . import data_alignment as da
from . import dataset as ds
from . import io
from . import mirror as mir
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


def adjust_panel(
    panel: mir.Panel, mnum: int, fit: bool, cfg: dict
) -> NDArray[np.float64]:
    """
    Helper function to get the adjustments for a single panel.

    Parameters
    ----------
    panel : mir.Panel
        The mirror panel to adjust.
    mnum : int
        The mirror number.
        1 for the primary and 2 for the secondary.
    fit: bool
        If True fit for the adjustments by modeling them as rotations of the panel.
        If False just use the raw residuals.
    cfg : dict
        The configuration dictionairy.

    Returns
    -------
    adjustments : NDArray[np.float64]
        The adjustments to make for the panel.
        This is a 17 element array with the following structure:
        `[mnum, panel_row, panel_col, dx, dy, d_adj1, ..., d_adj5, dx_err, dy_err, d_adj1_err, ..., d_adj5_err]`.
    """
    adjustments = np.zeros(17, np.float64)
    adjustments[0] = mnum
    adjustments[1] = panel.row
    adjustments[2] = panel.col
    if fit:
        meas_adj = panel.meas_adj.copy()
        meas_adj[:, 2] -= panel.meas_adj_resid
        meas_surface = panel.meas_surface.copy()
        meas_surface[:, 2] -= panel.meas_adj_resid
        dx, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(
            panel.can_surface, meas_surface, meas_adj, **cfg.get("adjust", {})
        )
    else:
        dx = 0
        dx_err = 0
        dy = 0
        dy_err = 0
        d_adj = -1 * panel.adj_resid
        d_adj_err = np.zeros_like(d_adj)
    # The primary has x and z opposite to what is intuitive
    if mnum == 1:
        dx *= -1
        d_adj *= -1
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
    meas_files = cfg["measurement"]
    if isinstance(meas_files, str):
        meas_files = [meas_files]
    meas_files = [os.path.abspath(os.path.join(cfgdir, meas)) for meas in meas_files]
    tracker_yamls = cfg.get("tracker_yaml", "")
    if isinstance(tracker_yamls, str):
        tracker_yamls = [tracker_yamls]
    title_str = cfg["title"]
    logger.info("Begining alignment %s in %s mode", title_str, mode)
    logger.debug("Using measurement files: %s", meas_files)

    dat_dir = os.path.abspath(os.path.join(cfgdir, cfg.get("data_dir", "/")))
    if "data_dir" in cfg:
        logger.info("Using data files from %s", dat_dir)
        ref_path = os.path.join(dat_dir, "reference.yaml")
    else:
        logger.info("Using packaged data files")
        ref_path = str(files("lat_alignment.data").joinpath("reference.yaml"))
    with open(ref_path) as file:
        reference = yaml.safe_load(file)
    datasets = [
        io.load_data(meas_file, **cfg.get("load", {"source": "photo"}))
        for meas_file in meas_files
    ]
    if "data_dir" in cfg:
        corner_path_m1 = os.path.join(dat_dir, f"primary_corners.yaml")
        adj_path_m1 = os.path.join(dat_dir, f"primary_adj.csv")
        corner_path_m2 = os.path.join(dat_dir, f"secondary_corners.yaml")
        adj_path_m2 = os.path.join(dat_dir, f"secondary_adj.csv")
    else:
        corner_path_m1 = str(
            files("lat_alignment.data").joinpath(f"primary_corners.yaml")
        )
        adj_path_m1 = str(files("lat_alignment.data").joinpath(f"primary_adj.csv"))
        corner_path_m2 = str(
            files("lat_alignment.data").joinpath(f"secondary_corners.yaml")
        )
        adj_path_m2 = str(files("lat_alignment.data").joinpath(f"secondary_adj.csv"))

    # load files
    corners = {
        "primary": io.load_corners(corner_path_m1),
        "secondary": io.load_corners(corner_path_m2),
    }
    adjusters = {
        "primary": io.load_adjusters(adj_path_m1, "primary"),
        "secondary": io.load_adjusters(adj_path_m2, "secondary"),
    }

    if mode == "panel":
        mirror = cfg["mirror"]
        if mirror == "primary":
            mnum = 1
        elif mirror == "secondary":
            mnum = 2
        else:
            raise ValueError(f"Invalid mirror: {mirror}")
        logger.info("Aligning panels for the %s mirror", mirror)

        # init, fit, and plot panels
        data_dict = {}
        for i, dataset in enumerate(datasets):
            try:
                if isinstance(dataset, ds.DatasetPhotogrammetry):
                    dataset, _ = da.align_photo(
                        dataset, reference, True, mirror, **cfg.get("align_photo", {})
                    )
                else:
                    dataset, _ = da.align_tracker(
                        dataset,
                        tracker_yamls[i],
                        mirror,
                        **cfg.get("align_tracker", {}),
                    )
            except Exception as e:
                logger.error(
                    "Failed to align to reference points, with error %s", str(e)
                )
                bootstrap_from = cfg.get("bootstrap_from", "all")
                logger.info("Bootstrapping from %s", bootstrap_from)
                if isinstance(dataset, ds.DatasetPhotogrammetry):
                    dataset, (aff, sft) = da.align_photo(
                        dataset,
                        reference,
                        True,
                        bootstrap_from,
                        **cfg.get("align_photo", {}),
                    )
                else:
                    dataset, (aff, sft) = da.align_tracker(
                        dataset,
                        tracker_yamls[i],
                        bootstrap_from,
                        **cfg.get("align_tracker", {}),
                    )
                cfrom = "opt_global"
                if bootstrap_from == "primary":
                    cfrom = "opt_primary"
                elif bootstrap_from == "secondary":
                    cfrom = "opt_secondary"
                points = tf.coord_transform(
                        dataset.points, cfrom, f"opt_{mirror}"
                    )
                aff, sft = tf.affine_basis_transform(aff, sft, cfrom, f"opt_{mirror}")
                errs = tf.err_transform(dataset.errs, aff)
                dataset.data_dict = {l: np.array([p, e]) for l, p, e in zip(dataset.labels, points, errs)}
            ddict = {f"{l}_{i}": np.array([p, e]) for l, p, e in zip(dataset.labels, dataset.points, dataset.errs)}
            data_dict = data_dict | ddict
        dataset = datasets[0].__class__(data_dict)
        append = ""
        if "sample_every" in cfg:
            i, j = cfg["sample_every"]
            ddict = {l: np.array([p, e]) for l, p, e in zip(dataset.labels[i::j], dataset.points[i::j], dataset.errs[i::j])}
            dataset.data_dict = ddict
            append = f"_{i}_{j}"

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(
            dataset.targets[:, 0],
            dataset.targets[:, 1],
            dataset.targets[:, 2],
            marker="x",
        )
        plt.show()
        dataset, _ = mir.remove_cm(
            dataset, mirror, cfg.get("compensate", 0), **cfg.get("common_mode", {})
        )
        panels = mir.gen_panels(
            mirror,
            dataset,
            corners[mirror],
            adjusters[mirror],
            cfg.get("compensate", 0),
            cfg.get("adjuster_radius", 100),
        )
        if cfg.get("only_adj", True):
            for panel in panels:
                panel.measurements = panel.measurements[panel.adj_msk]
                panel.meas_err = panel.meas_err[panel.adj_msk]
            measurements = np.vstack([panel.measurements for panel in panels])
            errs = np.vstack([panel.meas_err for panel in panels])
            data = {"TARGET" + str(i): np.array([meas, err]) for i, (meas, err) in enumerate(zip(measurements, errs))}
            dataset = datasets[0].__class__(data)
            dataset, _ = mir.remove_cm(
                dataset, mirror, cfg.get("compensate", 0), **cfg.get("common_mode", {})
            )
            panels = mir.gen_panels(
                mirror,
                dataset,
                corners[mirror],
                adjusters[mirror],
                cfg.get("compensate", 0),
                cfg.get("adjuster_radius", 100),
            )
        logger.info("Found measurements for %d panels", len(panels))
        fig = mir.plot_panels(
            panels, False, title_str, vmax=cfg.get("vmax", None), use_iqr=cfg.get("iqr", False)
        )
        fig.savefig(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}{append}.png"))
        fig = mir.plot_panels(
            panels, True, title_str, vmax=cfg.get("vmax", None), use_iqr=cfg.get("iqr", False)
        )
        fig.savefig(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}_err{append}.png"))
        res_all = np.vstack([panel.residuals for panel in panels])
        model_all = np.vstack([panel.model for panel in panels])
        res_err_all = np.vstack([panel.residuals_err for panel in panels])
        mir_out = np.hstack([model_all, res_all, res_err_all])
        np.savetxt(
            os.path.join(cfgdir, f"{title_str.replace(' ', '_')}_surface{append}.txt"),
            mir_out,
            header="x y z x_res y_res z_res x_res_err y_res_err z_res_err",
        )

        # calc and save adjustments
        logger.info("Caluculating adjustments")
        _adjust = partial(
            adjust_panel, mnum=mnum, fit=cfg.get("fit_adjustments", True), cfg=cfg
        )
        adjustments = np.vstack(pqdm(panels, _adjust, n_jobs=8))
        order = np.lexsort((adjustments[:, 2], adjustments[:, 1], adjustments[:, 0]))
        adjustments = adjustments[order]
        np.savetxt(
            os.path.join(cfgdir, f"{title_str.replace(' ', '_')}{append}.csv"),
            adjustments,
            fmt=["%d", "%d", "%d"] + ["%.5f"] * 14,
        )
    elif mode == "optical":
        if len(datasets) > 1 or len(tracker_yamls) > 1:
            raise ValueError("Cannot have multiple files in optical mode")
        dataset = datasets[0]
        tracker_yaml = tracker_yamls[0]
        align_to = cfg["align_to"]
        if align_to not in ["primary", "secondary", "receiver", "bearing"]:
            raise ValueError(f"Invalid element specified for 'align_to': {align_to}")
        logger.info("Aligning all optical elements to the %s", align_to)
        if isinstance(dataset, ds.DatasetPhotogrammetry):
            dataset, _ = da.align_photo(
                dataset, reference, True, "all", **cfg.get("align_photo", {})
            )
        else:
            dataset, _ = da.align_tracker(
                dataset, tracker_yaml, "all", **cfg.get("align_tracker", {})
            )

        # Load data and compute the transformation to align with the model
        # We want to put all the transformations into opt_global
        elements = {}  # {element_name : full_alignment}
        identity = (np.eye(3, dtype=np.float64), np.zeros(3, dtype=np.float64))
        try:
            if isinstance(dataset, ds.DatasetPhotogrammetry):
                meas, alignment = da.align_photo(
                    dataset, reference, True, "primary", **cfg.get("align_photo", {})
                )
            else:
                meas, alignment = da.align_tracker(
                    dataset,
                    cfg["tracker_yaml"],
                    "primary",
                    **cfg.get("align_tracker", {}),
                )
            meas, common_mode = mir.remove_cm(
                meas, "primary", cfg.get("compensate", 0), **cfg.get("common_mode", {})
            )
            full_alignment = mt.compose_transform(*common_mode, *alignment)
            if cfg.get("only_adj", True):
                panels = mir.gen_panels(
                    "primary",
                    meas,
                    corners["primary"],
                    adjusters["primary"],
                    cfg.get("compensate", 0),
                    cfg.get("adjuster_radius", 100),
                )
                for panel in panels:
                    panel.measurements = panel.measurements[panel.adj_msk]
                    panel.meas_err = panel.meas_err[panel.adj_msk]
                measurements = np.vstack([panel.measurements for panel in panels])
                errs = np.vstack([panel.meas_err for panel in panels])
                data = {"TARGET" + str(i): np.array([meas, err]) for i, (meas, err) in enumerate(zip(measurements, errs))}
                meas = meas.__class__(data)
                meas, common_mode_2 = mir.remove_cm(
                    meas,
                    "primary",
                    cfg.get("compensate", 0),
                    **cfg.get("common_mode", {}),
                )
                full_alignment = mt.compose_transform(*common_mode_2, *full_alignment)
            full_alignment = tf.affine_basis_transform(
                full_alignment[0], full_alignment[1], "opt_primary", "opt_global"
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
            if isinstance(dataset, ds.DatasetPhotogrammetry):
                meas, alignment = da.align_photo(
                    dataset, reference, True, "secondary", **cfg.get("align_photo", {})
                )
            else:
                meas, alignment = da.align_tracker(
                    dataset,
                    cfg["tracker_yaml"],
                    "secondary",
                    **cfg.get("align_tracker", {}),
                )
            meas, common_mode = mir.remove_cm(
                meas,
                "secondary",
                cfg.get("compensate", 0),
                **cfg.get("common_mode", {}),
            )
            full_alignment = mt.compose_transform(*common_mode, *alignment)
            if cfg.get("only_adj", True):
                panels = mir.gen_panels(
                    "secondary",
                    meas,
                    corners["secondary"],
                    adjusters["secondary"],
                    cfg.get("compensate", 0),
                    cfg.get("adjuster_radius", 100),
                )
                for panel in panels:
                    panel.measurements = panel.measurements[panel.adj_msk]
                    panel.meas_err = panel.meas_err[panel.adj_msk]
                measurements = np.vstack([panel.measurements for panel in panels])
                errs = np.vstack([panel.meas_err for panel in panels])
                data = {"TARGET" + str(i): np.array([meas, err]) for i, (meas, err) in enumerate(zip(measurements, errs))}
                meas = meas.__class__(data)
                meas, common_mode_2 = mir.remove_cm(
                    meas,
                    "secondary",
                    cfg.get("compensate", 0),
                    **cfg.get("common_mode", {}),
                )
                full_alignment = mt.compose_transform(*common_mode_2, *full_alignment)
            full_alignment = tf.affine_basis_transform(
                full_alignment[0], full_alignment[1], "opt_secondary", "opt_global"
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
            if isinstance(dataset, ds.DatasetPhotogrammetry):
                meas, alignment = da.align_photo(
                    dataset, reference, True, "bearing", **cfg.get("align_photo", {})
                )
                meas, cyl_fit = br.cylinder_fit(meas)
                full_alignment = mt.compose_transform(*alignment, *cyl_fit)
            else:
                full_alignment, alignment = da.align_tracker(
                    dataset,
                    cfg["tracker_yaml"],
                    "bearing",
                    **cfg.get("align_tracker", {}),
                )
                logger.warning("Can't do cylinder fit on bearing with tracker data!")
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
            if isinstance(dataset, ds.DatasetPhotogrammetry):
                meas, alignment = da.align_photo(
                    dataset, reference, True, "receiver", **cfg.get("align_photo", {})
                )
            else:
                meas, alignment = da.align_tracker(
                    dataset,
                    cfg["tracker_yaml"],
                    "receiver",
                    **cfg.get("align_tracker", {}),
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
            aff, sft = mt.compose_transform(*align_to_inv, *full_transform)
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
