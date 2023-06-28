"""
Perform alignment of LAT mirrors

Author: Saianeesh Keshav Haridas
"""
import os
import sys
import argparse as ap
import numpy as np
import matplotlib.pyplot as plt
import yaml
import adjustments as adj
import coordinate_transforms as ct
import mirror_fit as mf


def output(file, string):
    """
    Print and save to file at same time

    @param file: File pointer to write to
    @param string: String to print and save
    """
    file.write(string + "\n")
    print(string)


def get_panel_points(
    panels,
    mirror_path,
    out_file,
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
            output(out_file, panel_path + " does not seem to be a panel")
            continue
        panel_name = os.path.splitext(p)[0]
        output(out_file, "Fitting panel " + panel_name)

        # Lookup cannonical alignment points and adjustor locations
        if panel_name not in can_adj.keys():
            output(
                out_file,
                "Panel "
                + panel_name
                + " not found in cannonical adjustor position spreadsheet",
            )
            output(out_file, "Moving on to next panel")
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
        output(out_file, "RMS of surface is: " + str(round(rms, 3)))

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
            output(out_file, "WARNING: Potential outlier at point " + str(outl))

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

        # Transform cannonical alignment points and adjustors to measurement basis
        points = mf.transform_point(can_points, *popt)
        adjustors = mf.transform_point(adjustors, *popt)

        # Apply tension to center of panel
        points[-1, -1] += tension
        adjustors[-1, -1] += tension

        panel_points["panel_name"] = (can_points, points, adjustors)

    return panel_points


def mirror_cm_sub(panel_points, out_file):
    """
    Remove common mode from panel points.

    @param panel_points: Dict from get_panel_points
    @param out_file: File to output to

    @returns panel_points: Points with common mode removed.
    """
    diff = []
    for points in panel_points.values():
        points.append(points[0] - points[1])
    diff = np.vstack(diff)
    cm = np.median(diff, axis=0)
    output(out_file, f"Removing a common mode of {cm}.")
    panel_points = {
        name: (points[0] - cm, points[1], points[2] - cm)
        for name, points in panel_points.items()
    }

    return panel_points


def get_adjustments(panel_points, out_file):
    """
    Calculate adjustments for all panels in a mirror.

    @param panel_points: Dict from get_panel_points
    @param out_file: File to output to
    """
    for name, panel in panel_points.items():
        output(out_file, f"Aligning panel {name}")
        # Calculate adjustments
        dx, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(*panel)
        # TODO: Convert these into turns of the adjustor rods
        if dx < 0:
            x_dir = "left"
        else:
            x_dir = "right"
        output(
            out_file,
            "\tMove panel "
            + str(abs(round(dx, 3)))
            + " ± "
            + str(abs(round(dx_err, 3)))
            + " mm to the "
            + x_dir,
        )
        if dy < 0:
            y_dir = "down"
        else:
            y_dir = "up"
        output(
            out_file,
            "\tMove panel "
            + str(abs(round(dy, 3)))
            + " ± "
            + str(abs(round(dy_err, 3)))
            + " mm "
            + y_dir,
        )

        for i in range(len(d_adj)):
            d = d_adj[i]
            d_err = d_adj_err[i]
            if d < 0:
                d_dir = "in"
            else:
                d_dir = "out"
            output(
                out_file,
                "\tMove adjustor "
                + str(i + 1)
                + " "
                + str(abs(round(d, 3)))
                + " ± "
                + str(abs(round(d_err, 3)))
                + " mm "
                + d_dir,
            )


# Parse command line arguments and load config
parser = ap.ArgumentParser(
    description="Compute alignment for LAT mirrors, see README for more details"
)
parser.add_argument("config", help="Path to configuration file, should be a yaml")
args = parser.parse_args()

with open(args.config, "r") as file:
    cfg = yaml.safe_load(file)
measurement_dir = cfg["measurement_dir"]
coordinates = cfg.get("coordinates", "cad")
origin_shift = np.array(cfg.get("shift", np.zeros(3, dtype=float)), float)
compensation = cfg.get("compensation", 0.0)
cm_sub = cfg.get("cm_sub", False)
plots = cfg.get("plots", False)

# Check that measurement directory exists
if not os.path.exists(measurement_dir):
    print("Supplied measurement directory does not exist. Please double check the path")
    sys.exit()

# Make sure that shift is correct shape
if len(origin_shift) != 3:
    print(
        "Coordinate origin shift invalid shape. \
        Please supply values for x, y, and z in mm seperated by spaces"
    )
    sys.exit()

# Check if coordinate system is valid
valid_coords = ["cad", "global", "primary", "secondary"]
if coordinates not in valid_coords:
    print(
        "Coordinate system '",
        coordinates,
        "' not valid\n Please use one of the following instead: cad, global, primary, secondary",
        sep="",
    )
    sys.exit()

# Initialize output file
out_file = open(os.path.join(measurement_dir, "output.txt"), "w+")
output(out_file, "Starting alignment procedure for measurement at: " + measurement_dir)
output(out_file, "Using coordinate system: " + coordinates)
output(out_file, "Using origin shift: " + str(origin_shift))
output(out_file, "Applying compensation: " + str(compensation) + " mm")
output(out_file, "Common mode subtraction set to: " + str(cm_sub))

# Initialize cannonical adjustor positions
can_adj = {}

# Align primary mirror
primary_path = os.path.join(measurement_dir, "M1")
if os.path.exists(primary_path):
    output(out_file, "Aligning primary mirror")

    # Make plot directory
    if plots:
        os.makedirs(os.path.join(measurement_dir, "plots", "M1"), exist_ok=True)

    # Load cannonical adjustor points
    m1_can = "./can_points/M1.txt"
    if not os.path.exists(m1_can):
        output(out_file, "Cannonical points for M1 not found")
        sys.exit()
    c_points = np.genfromtxt(m1_can, dtype=str)
    for i in range(int(c_points.shape[0] / 5)):
        pan_points = c_points[5 * i : 5 * (i + 1)]
        can_adj[pan_points[0, 0]] = np.array(pan_points[:, 2:], dtype=float)

    # Get all panel files
    panels = os.listdir(primary_path)
    if len(panels) == 0:
        output(out_file, "No panels found for primary mirror")

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
        out_file,
        can_adj,
        coord_trans,
        origin_shift,
        compensation,
        mf.primary_fit_func,
        plots,
    )

    if cm_sub:
        mirror_cm_sub(primary, out_file)

    get_adjustments(primary, out_file)

# Align secondary mirror
secondary_path = os.path.join(measurement_dir, "M2")
if os.path.exists(secondary_path):
    output(out_file, "Aligning secondary mirror")

    # Make plot directory
    if plots:
        os.makedirs(os.path.join(measurement_dir, "plots", "M2"), exist_ok=True)

    # Load cannonical adjustor points
    m2_can = "./can_points/M2.txt"
    if not os.path.exists(m2_can):
        output(out_file, "Cannonical points for M2 not found")
        sys.exit()
    c_points = np.genfromtxt(m2_can, dtype=str)
    for i in range(int(c_points.shape[0] / 5)):
        pan_points = c_points[5 * i : 5 * (i + 1)]
        can_adj[pan_points[0, 0]] = np.array(pan_points[:, 2:], dtype=float)

    # Get all panel files
    panels = os.listdir(secondary_path)
    if len(panels) == 0:
        output(out_file, "No panels found for secondary mirror")

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
        out_file,
        can_adj,
        coord_trans,
        origin_shift,
        compensation,
        mf.secondary_fit_func,
        plots,
    )
    if cm_sub:
        mirror_cm_sub(secondary, out_file)

    get_adjustments(secondary, out_file)
