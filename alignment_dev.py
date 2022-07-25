"""
Perform alignment of LAT mirrors

Author: Saianeesh Keshav Haridas
"""
import os
import sys
import argparse as ap
import numpy as np
import adjustments as adj
import coordinate_transforms as ct
import mirror_fit as mf
import matplotlib.pyplot as plt


def output(file, string):
    """
    Print and save to file at same time

    @param file: File pointer to write to
    @param string: String to print and save
    """
    file.write(string + "\n")
    print(string)


def align_panels(
    panels,
    mirror_path,
    out_file,
    can_adj,
    coord_trans,
    origin_shift,
    compensation,
    mirror_fit_func,
    cm_sub=False,
    plots=False,
):
    """
    Align panels of mirror

    @param panels: The filenames for each panel in the mirror directory
    @param mirror_path: Path to the mirror directory
    @param out_file: The output file to write to
    @param can_adj: Cannonical positions of adjustors
    @param coord_trans: The coordinate transform to apply to measured points
    @param origin_shift: The origin_shift to pass to coord_trans
    @param compensation: Compensation to apply to measurement
    @param mirror_fit_func: The function used to fit the mirror
    @param cm_sub: Set to True for common mode subtracted adjustments
    """
    for p in panels:
        panel_path = os.path.join(mirror_path, p)
        if not os.path.isfile(panel_path):
            output(out_file, panel_path + " does not seem to be a panel")
            continue
        panel_name = os.path.splitext(p)[0]
        output(out_file, "Aligning panel " + panel_name)

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

        #points = points[::3]
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
            *popt
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
            ps_array = np.array(ps)
            ps_dists_array = np.array(ps_dists)
            datum = np.column_stack([ps_array, ps_dists_array])
            datafile_path = measurement_dir +'\ps.txt'
            np.savetxt(datafile_path , datum, fmt='%f')
            print(plot_path)


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

        # Calculate adjustments
        dx, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(
            can_points, points, adjustors, cm_sub
        )
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


# Parse command line arguments
parser = ap.ArgumentParser(
    description="Compute alignment for LAT mirrors, see README for more details"
)
parser.add_argument("measurement_dir", help="Root directory for measurement to use")
parser.add_argument(
    "-c",
    "--coordinates",
    help="Measurement coordinate system, overrides setting in config file.\
    Valid options are: cad, global, primary, secondary.",
)
parser.add_argument(
    "-s",
    "--shift",
    nargs=3,
    help="Origin shift to apply in mm, overrides setting in config file",
    type=float,
)
parser.add_argument(
    "-f",
    "--compensation",
    help="FARO compensation in mm to apply",
    type=float,
)
parser.add_argument(
    "-cm",
    "--commonmode",
    help="Pass to subtract common mode from adjustments",
    action="store_true",
)
parser.add_argument(
    "-p",
    "--plots",
    help="Generate plots of panel surfaces and power spectra",
    action="store_true",
)
args = parser.parse_args()

measurement_dir = args.measurement_dir
coordinates = args.coordinates
origin_shift = args.shift
compensation = args.compensation
cm_sub = args.commonmode
plots = args.plots

# Check that measurement directory exists
if not os.path.exists(measurement_dir):
    print("Supplied measurement directory does not exist. Please double check the path")
    sys.exit()

# Load in config file if needed
if (coordinates is None) or (origin_shift is None) or (compensation is None):
    confpath = os.path.join(measurement_dir, "config.txt")
    if not os.path.exists(confpath):
        print(
            "Config file doesn't exist and equivalent command line arguments were not given"
        )
        sys.exit()
    config = dict(np.genfromtxt(confpath, dtype=str, delimiter="\t"))
    if coordinates is None and "coords" in config.keys():
        coordinates = config["coords"]
    if origin_shift is None and "shift" in config.keys():
        origin_shift = np.array(config["shift"].split(), dtype=float)
    if compensation is None and "compensation" in config.keys():
        compensation = float(config["compensation"])
    if cm_sub is False and "cm_sub" in config.keys():
        compensation = bool(config["cm_sub"])
    if plots is False and "plots" in config.keys():
        compensation = bool(config["plots"])


# Set some defaults
if coordinates is None:
    coordinates = "cad"
if origin_shift is None:
    origin_shift = np.zeros(3, dtype=float)
if compensation is None:
    compensation = 0.0

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
    align_panels(
        panels,
        primary_path,
        out_file,
        can_adj,
        coord_trans,
        origin_shift,
        compensation,
        mf.primary_fit_func,
        cm_sub,
        plots,
    )

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
    align_panels(
        panels,
        secondary_path,
        out_file,
        can_adj,
        coord_trans,
        origin_shift,
        compensation,
        mf.secondary_fit_func,
        cm_sub,
        plots,
    )
