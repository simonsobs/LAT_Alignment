import os
import yaml
import numpy as np
import argparse
from . import mirror as mir, adjustments as adj, io

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

    cfgdir = os.path.basename(os.path.abspath(args.config))
    meas_file = os.path.abspath(os.path.join(cfgdir, cfg["measurment"]))
    dat_dir = os.path.abspath(os.path.join(cfgdir, cfg.get["dat_dir", "../data"]))

    # load files
    meas = io.load_photo(meas_file, True, mirror=mirror, **cfg.get("align", {}))
    corners = io.load_corners(os.path.join(dat_dir, f"{mirror}_corners.yaml"))
    adjusters = io.load_adjusters(os.path.join(dat_dir, f"{mirror}_adj.txt"))

    # init, fit, and plot panels
    panels = mir.gen_panels(mirror, meas, corners, adjusters)
    mir.fit_panels(panels, **cfg.get("fit", {}))
    title_str = cfg["title"]
    fig = mir.plot_panels(panels, title_str)
    fig.savefig(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.png"))

    # calc and save adjustments
    adjustments = np.zeros((len(panels), 9), dtype=np.float32)
    for i, panel in enumerate(panels):
        adjustments[i][0] = mnum
        adjustments[i][1] = panel.row
        adjustments[i][2] = panel.col
        adjustments[i][3:] = adj.calc_adjustments(panel.can_surface, panel.meas_surface, panel.meas_adj, **cfg.get("adjust", {}))
    order = np.lexsort((adjustments[2], adjustments[1], adjustments[0]))
    adjustments = adjustments[order]
    np.savetxt(os.path.join(cfgdir, f"{title_str.replace(' ', '_')}.csv"), adjustments)
