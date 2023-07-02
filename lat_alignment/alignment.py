"""
Perform alignment of LAT mirrors

Author: Saianeesh Keshav Haridas
"""
import os
import sys
import argparse as ap
import logging
import matplotlib.pyplot as plt
import numpy as np
import yaml
from . import adjustments as adj
from . import coordinate_transforms as ct
from . import mirror_fit as mf


logger = logging.getLogger()
logger.setLevel(logging.INFO)
consoleHandler = logging.StreamHandler()
logger.addHandler(consoleHandler)


def _plot_panel(mirror_path, panel_name, points, residuals):
    b_path, m_path = os.path.split(os.path.normpath(mirror_path))
    plot_path = os.path.join(b_path, "plots", m_path)
    plt.tricontourf(points[:, 0], points[:, 1], points[:, 2])
    plt.title("Surface of " + panel_name)
    plt.savefig(os.path.join(plot_path, panel_name + "_surface.png"))
    plt.close()

    plt.hist(residuals[:, 2])
    plt.xlabel("Residual (mm)")
    plt.title("Residual distribution of " + panel_name)
    plt.savefig(os.path.join(plot_path, panel_name + "_hist.png"))
    plt.close()

    ps, ps_dists = mf.res_power_spect(residuals)
    plt.plot(ps_dists, ps)
    plt.xlabel("Scale (mm)")
    plt.title("Power spectrum of " + panel_name)
    plt.savefig(os.path.join(plot_path, panel_name + "_ps.png"))
    plt.close()


def get_panel_points(
    panels,
    mirror_path,
    can_adj,
    coord_trans,
    origin_shift,
    compensation,
    mirror_fit_func,
    plots=False,
):
    """
    Get critical points for a panel.

    @param panels: The filenames for each panel in the mirror directory
    @param mirror_path: Path to the mirror directory
    @param out_file: The output file to write to
    @param can_adj: Cannonical positions of adjustors
    @param coord_trans: The coordinate transform to apply to measured points
    @param origin_shift: The origin_shift to pass to coord_trans
    @param compensation: Compensation to apply to measurement
    @param mirror_fit_func: The function used to fit the mirror
    @param cm_sub: Set to True for common mode subtracted adjustments

    @returns panel_points: A dict with structure {panel_name: (can_points, points, adjustors)}
    """
    panel_points = {}
    for p in panels:
        panel_path = os.path.join(mirror_path, p)
        if not os.path.isfile(panel_path):
            logger.warning(panel_path + " does not seem to be a panel")
            continue
        panel_name = os.path.splitext(p)[0]
        logger.info("Fitting panel " + panel_name)

        # Lookup cannonical alignment points and adjustor locations
        if panel_name not in can_adj.keys():
            logger.warning(
                "Panel %s not found in cannonical adjustor position spreadsheet",
                panel_name,
            )
            logger.warning("Moving on to next panel")
            continue
        if int(panel_name[5]) == 1:
            mirror_a = mf.a_primary
            mirror_trans = ct.cad_to_primary
        else:
            mirror_a = mf.a_secondary
            mirror_trans = ct.cad_to_secondary
        adjustors = mirror_trans(can_adj[panel_name], 0)
        can_z = mf.mirror(adjustors[:, 0], adjustors[:, 1], mirror_a)
        can_points = np.hstack((adjustors[:, :2], can_z[:, np.newaxis]))

        # Load pointcloud from data
        points = np.genfromtxt(
            panel_path, skip_header=1, usecols=(3, 4, 5), dtype=str, delimiter="\t"
        )
        points = np.array(
            list(map(lambda p: p.replace(",", ""), points.flatten())), dtype=float
        ).reshape(points.shape)

        # Transform points to mirror coordinates
        points = coord_trans(points, origin_shift)

        # Fit to mirror surface
        popt, rms = mf.mirror_fit(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            compensation,
            mirror_fit_func,
            bounds=[
                (-50, 50),
                (-50, 50),
                (-50, 50),
                (-np.pi / 18.0, np.pi / 18.0),
                (-np.pi / 18.0, np.pi / 18.0),
                (-np.pi / 18.0, np.pi / 18.0),
            ],
        )
        logger.info("RMS of surface is: %.3f", rms)

        # Calculate residuals
        residuals = mf.calc_residuals(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            compensation,
            mirror_fit_func,
            *popt,
        )

        # Look for outlier points
        res_med = np.median(residuals[:, 2])
        res_std = np.std(residuals[:, 2])
        outlim_l = res_med - 3 * res_std
        outlim_r = res_med + 3 * res_std
        outliers = np.where(
            (residuals[:, 2] < outlim_l) | (residuals[:, 2] > outlim_r)
        )[0]
        for outl in outliers:
            logger.warning("Potential outlier at point %d", outl)

        # Fit for tension
        tension = 0
        popt_t, rms_t = mf.tension_fit(
            residuals,
            bounds=[
                (-50, 50),
                (-50, 50),
                (-50, 50),
                (0, np.inf),
                (0, np.inf),
            ],
        )

        # If the fit tension improves rms, use it
        if rms_t < rms:
            tension = popt_t[2]

        # Generate plots
        if plots:
            _plot_panel(mirror_path, panel_name, points, residuals)

        # Transform cannonical alignment points and adjustors to measurement basis
        points = mf.transform_point(can_points, *popt)
        adjustors = mf.transform_point(adjustors, *popt)

        # Apply tension to center of panel
        points[-1, -1] += tension
        adjustors[-1, -1] += tension

        panel_points[panel_name] = (can_points, points, adjustors)

    return panel_points


def mirror_cm_sub(panel_points):
    """
    Remove common mode from panel points.

    @param panel_points: Dict from get_panel_points

    @returns panel_points: Points with common mode removed.
    """
    diff = []
    for points in panel_points.values():
        diff.append(points[0] - points[1])
    diff = np.vstack(diff)
    cm = np.median(diff, axis=0)
    logger.info("Removing a common mode of %s mm.", str(cm))
    panel_points = {
        name: (points[0] - cm, points[1], points[2] - cm)
        for name, points in panel_points.items()
    }

    return panel_points


def get_adjustments(panel_points):
    """
    Calculate adjustments for all panels in a mirror.

    @param panel_points: Dict from get_panel_points

    @return adjustments: Dict of adjustments and errors
    """
    adjustments = {}
    for name, panel in panel_points.items():
        # Calculate adjustments
        dx, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(*panel)
        # TODO: Convert these into turns of the adjustor rods

        adjustments[name] = np.hstack(((dx, dy), d_adj, (dx_err, dy_err), d_adj_err))

    return adjustments


def optimize_adjustors(adjustments, adjustors, low, high):
    """
    Optimize adjustments to keep things in bounds.

    @param adjustments: Dict of adjustments and errors.
    @param adjustors: Dict of current adjusttor positions.
    @param low: Low end of adjustment range.
    @param high: High end of adjustor range.

    @returns adjustments: Updated adjustments.
    """
    positions = np.zeros(len(adjustments) * 5)
    names = list(adjustments.keys())
    for i, panel in enumerate(names):
        positions[i * 5 : (i + 1) * 5] = np.add(
            adjustments[panel][2:7], adjustors.get(panel, np.zeros(7))[2:7]
        )

    def _out_of_range(offset):
        new = positions - offset
        under = low - new[new < low]
        over = new[new > high] - high

        return np.sum(under) + np.sum(over)

    oor = _out_of_range(0)
    if oor == 0:
        logger.info("No adjustors out of range")
        return adjustments
    # An ugly brute force solution
    # But the scope is small and the solution range is flat
    coarse, step = np.linspace(low - high, high - low, retstep=True)
    oor = np.array([_out_of_range(off) for off in coarse])
    min_i = np.argmin(oor)
    min_oor = oor[min_i]
    min_off = coarse[min_i]
    fine = np.linspace(min_off - (np.sign(min_off) * step), min_off)
    oor = np.array([_out_of_range(off) for off in fine])
    min_i = np.where(oor <= min_oor)[0][0]
    offset = fine[min_i]

    logger.info("Applying an offset of %f to all z adjustors", offset)
    positions -= offset

    under = np.where(positions < low)[0]
    for i in under:
        logger.warning(
            "Adjustor %d of panel %s under range by %f",
            i % 5 + 1,
            names[i // 5],
            low - positions[i],
        )
    over = np.where(positions > high)[0]
    for i in over:
        logger.warning(
            "Adjustor %d of panel %s over range by %f",
            i % 5 + 1,
            names[i // 5],
            positions[i] - high,
        )

    for i, panel in enumerate(names):
        adjustments[panel][2:7] = positions[i * 5 : (i + 1) * 5]

    return adjustments


def log_adjustments(adjustments):
    """
    Log perscibed adjustments.

    @param adjustments: Dict of adjustments and errors.
    """
    for name, adjustment in adjustments.items():
        logger.info("Aligning panel %s", name)
        dx, dy, *d_adj = adjustment[:7]
        dx_err, dy_err, *d_adj_err = adjustment[7:]
        if dx < 0:
            x_dir = "left"
        else:
            x_dir = "right"
        logger.info("\tMove panel %.3f ± %.3f mm to the %s", dx, dx_err, x_dir)
        if dy < 0:
            y_dir = "down"
        else:
            y_dir = "up"
        logger.info("\tMove panel %.3f ± %.3f mm to the %s", dy, dy_err, y_dir)

        for i in range(len(d_adj)):
            d = d_adj[i]
            d_err = d_adj_err[i]
            if d < 0:
                d_dir = "in"
            else:
                d_dir = "out"
            logger.info("\tMove adjustor %d %.3f ± %.3f mm %s", i + 1, d, d_err, d_dir)


def update_adjustors(adjustments, adjustors):
    """
    Update adjustor postions.

    @param adjustments: Dict of adjustments and errors.
    @param adjustors: Dict of current adjusttor positions.

    @returns adjustors: Updated adjustor positions.
    """
    for panel in adjustments.keys():
        adjustors[panel] = np.add(
                adjustments[panel][:7], adjustors.get(panel, np.zeros(7))
        ).tolist()

    return adjustors


def main():
    # Parse command line arguments and load config
    parser = ap.ArgumentParser(
        description="Compute alignment for LAT mirrors, see README for more details"
    )
    parser.add_argument("config", help="Path to configuration file, should be a yaml")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        cfg = yaml.safe_load(file)
    measurement_dir = cfg.get(
        "measurement_dir", os.path.dirname(os.path.abspath(args.config))
    )
    can_dir = cfg.get(
        "cannonical_points",
        os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)),
                os.path.pardir,
                "can_points",
            )
        ),
    )
    coordinates = cfg.get("coordinates", "cad")
    origin_shift = np.array(cfg.get("shift", np.zeros(3, dtype=float)), float)
    compensation = cfg.get("compensation", 0.0)
    cm_sub = cfg.get("cm_sub", False)
    plots = cfg.get("plots", False)
    adj_low = cfg.get("adj_low", -1.0)
    adj_high = cfg.get("adj_low", 1.0)
    adj_path = cfg.get("adj_path", None)
    adj_out = cfg.get("adj_out", os.path.join(measurement_dir, "adjustors.yaml"))

    # Check that measurement directory exists
    if not os.path.exists(measurement_dir):
        logger.error(
            "Supplied measurement directory does not exist. Please double check the path"
        )
        sys.exit()

    # Check that cannonical points directory exists
    if not os.path.exists(can_dir):
        logger.error(
            "Supplied cannonical points directory does not exist. Please double check the path"
        )
        sys.exit()

    # Make sure that shift is correct shape
    if len(origin_shift) != 3:
        logger.error(
            "Coordinate origin shift invalid shape. \
            Please supply values for x, y, and z in mm."
        )
        sys.exit()

    # Check if coordinate system is valid
    valid_coords = ["cad", "global", "primary", "secondary"]
    if coordinates not in valid_coords:
        logger.error(
            "Coordinate system '%s' not valid\n Please use one of the following instead: cad, global, primary, secondary",
            coordinates,
        )
        sys.exit()

    # Initialize output file
    log_file = cfg.get("log", os.path.join(measurement_dir, "output.txt"))
    if log_file is not None:
        fileHandler = logging.FileHandler(log_file, "w+")
        logger.addHandler(fileHandler)
    logger.info("Starting alignment procedure for measurement at: %s", measurement_dir)
    logger.info("Using coordinate system: %s", coordinates)
    logger.info("Using origin shift: %s", str(origin_shift))
    logger.info("Applying compensation: %f mm", compensation)
    logger.info("Common mode subtraction set to: %s", str(cm_sub))

    # Initialize cannonical adjustor positions
    can_adj = {}

    # Load adjustor position
    if adj_path is None:
        adjustors = {}
    else:
        if not os.path.isfile(adj_path):
            logger.error("Provided adjustor postion file doesn't seem to exist")
            sys.exit()
        with open(adj_path, "r") as file:
            adjustors = yaml.safe_load(file)

    # Align primary mirror
    primary_path = os.path.join(measurement_dir, "M1")
    if os.path.exists(primary_path):
        logger.info("Aligning primary mirror")

        # Make plot directory
        if plots:
            os.makedirs(os.path.join(measurement_dir, "plots", "M1"), exist_ok=True)

        # Load cannonical adjustor points
        m1_can = os.path.join(can_dir, "M1.txt")
        if not os.path.exists(m1_can):
            logger.error("Cannonical points for M1 not found")
            sys.exit()
        c_points = np.genfromtxt(m1_can, dtype=str)
        for i in range(int(c_points.shape[0] / 5)):
            pan_points = c_points[5 * i : 5 * (i + 1)]
            can_adj[pan_points[0, 0]] = np.array(pan_points[:, 2:], dtype=float)

        # Get all panel files
        panels = os.listdir(primary_path)
        if len(panels) == 0:
            logger.warning("No panels found for primary mirror")

        # Figure out which coordinate transform to use
        if coordinates == "cad":
            coord_trans = ct.cad_to_primary
        elif coordinates == "global":
            coord_trans = ct.global_to_primary
        elif coordinates == "primary":
            coord_trans = ct.shift_coords
        else:
            coord_trans = ct.secondary_to_primary

        # Align panels
        primary = get_panel_points(
            panels,
            primary_path,
            can_adj,
            coord_trans,
            origin_shift,
            compensation,
            mf.primary_fit_func,
            plots,
        )

        if cm_sub:
            mirror_cm_sub(primary)

        adjustments = get_adjustments(primary)
        adjustments = optimize_adjustors(adjustments, adjustors, adj_low, adj_high)
        log_adjustments(adjustments)
        adjustors = update_adjustors(adjustments, adjustors)

    # Align secondary mirror
    secondary_path = os.path.join(measurement_dir, "M2")
    if os.path.exists(secondary_path):
        logger.info("Aligning secondary mirror")

        # Make plot directory
        if plots:
            os.makedirs(os.path.join(measurement_dir, "plots", "M2"), exist_ok=True)

        # Load cannonical adjustor points
        m2_can = os.path.join(can_dir, "M2.txt")
        if not os.path.exists(m2_can):
            logger.error("Cannonical points for M2 not found")
            sys.exit()
        c_points = np.genfromtxt(m2_can, dtype=str)
        for i in range(int(c_points.shape[0] / 5)):
            pan_points = c_points[5 * i : 5 * (i + 1)]
            can_adj[pan_points[0, 0]] = np.array(pan_points[:, 2:], dtype=float)

        # Get all panel files
        panels = os.listdir(secondary_path)
        if len(panels) == 0:
            logger.warning("No panels found for secondary mirror")

        # Figure out which coordinate transform to use
        if coordinates == "cad":
            coord_trans = ct.cad_to_secondary
        elif coordinates == "global":
            coord_trans = ct.global_to_secondary
        elif coordinates == "secondary":
            coord_trans = ct.shift_coords
        else:
            coord_trans = ct.primary_to_secondary

        # Align panels
        secondary = get_panel_points(
            panels,
            secondary_path,
            can_adj,
            coord_trans,
            origin_shift,
            compensation,
            mf.secondary_fit_func,
            plots,
        )
        if cm_sub:
            mirror_cm_sub(secondary)

        adjustments = get_adjustments(secondary)
        adjustments = optimize_adjustors(adjustments, adjustors, adj_low, adj_high)
        log_adjustments(adjustments)
        adjustors = update_adjustors(adjustments, adjustors)

    # Save adjustor postions
    with open(adj_out, "w") as file:
        yaml.dump(adjustors, file, default_flow_style=None)


if __name__ == "__main__":
    main()
