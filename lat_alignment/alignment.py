"""
Main driver script for running the alignment.
You typically want to use the `lat_alignment` entrypoint rather than
calling this directly.
"""

import argparse
import os
from functools import partial
from importlib.resources import files

import numpy as np
import yaml
from numpy.typing import NDArray
from pqdm.processes import pqdm

from . import adjustments as adj
from . import io
from . import mirror as mir


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
    dy, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(
        panel.can_surface, meas_surface, meas_adj, **cfg.get("adjust", {})
    )
    adjustments[3:] = np.array(
        [dy, dy] + list(d_adj) + [dx_err, dy_err] + list(d_adj_err)
    )

    return adjustments


def main():
    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="path to config file")
    args = parser.parse_args()
    with open(args.config) as file:
        cfg = yaml.safe_load(file)

    mirror = cfg["mirror"]
    if mirror == "primary":
        mnum = 1
    elif mirror == "secondary":
        mnum = 2
    else:
        raise ValueError(f"Invalid mirror: {mirror}")

    cfgdir = os.path.dirname(os.path.abspath(args.config))
    meas_file = os.path.abspath(os.path.join(cfgdir, cfg["measurement"]))
    if "data_dir" in cfg:
        dat_dir = os.path.abspath(os.path.join(cfgdir, cfg["data_dir"]))
        corner_path = os.path.join(dat_dir, f"{mirror}_corners.yaml")
        adj_path = os.path.join(dat_dir, f"{mirror}_adj.csv")
    else:
        corner_path = str(
            files("lat_alignment.data").joinpath(f"{mirror}_corners.yaml")
        )
        adj_path = str(files("lat_alignment.data").joinpath(f"{mirror}_adj.csv"))

    # load files
    meas = io.load_photo(meas_file, True, mirror=mirror, **cfg.get("load", {}))
    corners = io.load_corners(corner_path)
    adjusters = io.load_adjusters(adj_path, mirror)

    # init, fit, and plot panels
    meas = mir.remove_cm(
        meas, mirror, cfg.get("compensate", 0), **cfg.get("common_mode", {})
    )
    panels = mir.gen_panels(
        mirror,
        meas,
        corners,
        adjusters,
        cfg.get("compensate", 0),
        cfg.get("adjuster_radius", 100),
    )
    title_str = cfg["title"]
    fig = mir.plot_panels(panels, title_str, vmax=cfg.get("vmax", None))
    fig.savefig(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.png"))

    # calc and save adjustments
    _adjust = partial(adjust_panel, mnum=mnum, cfg=cfg)
    adjustments = np.vstack(pqdm(panels, _adjust, n_jobs=8))
    order = np.lexsort((adjustments[2], adjustments[1], adjustments[0]))
    adjustments = adjustments[order]
    np.savetxt(
        os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.csv"),
        adjustments,
        fmt=["%d", "%d", "%d"] + ["%.5f"] * 14,
    )
