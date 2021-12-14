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
        # Will need to change genfromtxt args based on what FARO software outputs
        # NOTE: This is for copy pasted files, for files exported with report
        #       use usefoles=(5, 6, 7) instead
        # TODO: Figure out a way to be agnostic to this
        points = np.genfromtxt(panel_path, skip_header=1, usecols=(4, 5, 6), dtype=str)
        points = np.array(
            list(map(lambda p: p.replace(",", ""), points.flatten())), dtype=float
        ).reshape(points.shape)

        # Transform points to mirror coordinates and compensate
        points = coord_trans(points, origin_shift)
        if compensation != 0.0:
            points = ct.compensate(points, compensation)

        # Fit to mirror surface
        popt, rms = mf.mirror_fit(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            mirror_fit_func,
        )
        output(out_file, "RMS of surface is: " + str(rms))

        # Transform cannonical alignment points and adjustors to measurement basis
        points = mf.transform_point(can_points, *popt)
        adjustors = mf.transform_point(adjustors, *popt)

        # Calculate adjustments
        dx, dy, d_adj, dx_err, dy_err, d_adj_err = adj.calc_adjustments(
            can_points, points, adjustors
        )
        # TODO: Convert these into turns of the adjustor rods
        if dx < 0:
            x_dir = "left"
        else:
            x_dir = "right"
        output(
            out_file,
            "\tMove panel "
            + str(abs(dx))
            + " ± "
            + str(abs(dx_err))
            + " mm to the "
            + x_dir,
        )
        if dy < 0:
            y_dir = "down"
        else:
            y_dir = "up"
        output(
            out_file,
            "\tMove panel " + str(abs(dy)) + " ± " + str(abs(dy_err)) + " mm " + y_dir,
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
                + str(i)
                + " "
                + str(abs(d))
                + " ± "
                + str(abs(d_err))
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
args = parser.parse_args()

measurement_dir = args.measurement_dir
coordinates = args.coordinates
origin_shift = args.shift
compensation = args.compensation

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
    if coordinates is None:
        coordinates = config["coords"]
    if origin_shift is None:
        origin_shift = config["shift"].split()
    if compensation is None:
        compensation = config["compensation"]

# Set some defaults
if origin_shift is None:
    origin_shift = np.zeros(3, dtype=float)
if compensation is None:
    compensation = 0.0

# Cast config options
origin_shift = np.array(origin_shift, dtype=float)
compensation = float(compensation)

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

# Initialize cannonical adjustor positions
can_adj = {}

# Align primary mirror
primary_path = os.path.join(measurement_dir, "M1")
if os.path.exists(primary_path):
    output(out_file, "Aligning primary mirror")

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
    )

# Align secondary mirror
secondary_path = os.path.join(measurement_dir, "M2")
if os.path.exists(secondary_path):
    output(out_file, "Aligning secondary mirror")

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
    )
