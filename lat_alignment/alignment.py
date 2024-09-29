import os
import yaml
import numpy as np
import argparse
from . import mirror as mir, adjustments as adj, io
import matplotlib.pyplot as plt
from pqdm.processes import pqdm
from functools import partial

def adjust_panel(panel, mnum, cfg):
    adjustments = np.zeros(17)
    adjustments[0] = mnum
    adjustments[1] = panel.row
    adjustments[2] = panel.col
    meas_adj = panel.meas_adj.copy()
    meas_adj[:, 2] += panel.meas_adj_resid
    meas_surface = panel.meas_surface.copy()
    meas_surface[:, 2] += panel.meas_adj_resid
    dy, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(panel.can_surface, meas_surface, meas_adj, **cfg.get("adjust", {}))
    adjustments[3:] = np.array([dy, dy] + list(d_adj) + [dx_err, dy_err] + list(d_adj_err))
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
    dat_dir = os.path.abspath(os.path.join(cfgdir, cfg.get("dat_dir", "../../data")))

    # load files
    meas = io.load_photo(meas_file, True, mirror=mirror, **cfg.get("load", {}))
    corners = io.load_corners(os.path.join(dat_dir, f"{mirror}_corners.yaml"))
    adjusters = io.load_adjusters(os.path.join(dat_dir, f"{mirror}_adj.csv"), mirror)

    # init, fit, and plot panels
    meas = mir.remove_cm(meas, mirror, cfg.get("compensate", 0), **cfg.get("common_mode", {}))
    panels = mir.gen_panels(mirror, meas, corners, adjusters, cfg.get("compensate", 0), cfg.get("adjuster_radius", 200))
    title_str = cfg["title"]
    fig = mir.plot_panels(panels, title_str, vmax = cfg.get("vmax", None))
    fig.savefig(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.png"))

    # calc and save adjustments
    _adjust = partial(adjust_panel, mnum=mnum, cfg=cfg)
    adjustments = np.vstack(pqdm(panels, _adjust, n_jobs=8))
    order = np.lexsort((adjustments[2], adjustments[1], adjustments[0]))
    adjustments = adjustments[order]
    np.savetxt(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.csv"), adjustments, fmt=["%d", "%d", "%d"]+["%.5f"]*14)
