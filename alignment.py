import os
import sys
import argparse as ap
import numpy as np
import adjustments as adj
import mirror_fit as mf

"""
File structure:
    Each measurement session will have its own directory with some useful naming scheme
        Something like YYYYMMDD_num is probably good
    Each mirror will have its own subdir
    Within each mirror directory have a file for each panel whose name is the panel name
    File will contain pointcloud of measurements
    In root of directory we need some sort of config file that tells you what coordinate system the measurements were taken in as well as any adjustments that need to be applied to the model (ie: an origin shift to account for something that is in the wrong place but can't be moved)
        (these could also just be command line arguments)
    Also need to have some sort of lookup table that contains the positions of the alignmnt points and adjustors for each panel (in the mirror coordinates)
    
Workflow:
    Read in config file/parse command line arguments
    Load measurements on a per panel basis
    Transform panel from measurement coordinates to mirror coordinates
    Fit using mf
    Transform adjustor and alignment point locations with fit params
    Fit for adjustments with adj
    Print out adjustments and save to a file in root of measurement dir
"""

v_m1 = (0, 0, 3600)  # mm
v_m2 = (0, -4800, 0)  # mm
a_m1 = -np.arctan(0.5)
a_m2 = np.arctan(1.0 / 3.0) - np.pi / 2


def global_to_mirror(coords, shift, v_m, a_m):
    """
    Transform from global coordinates to mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)
    @param v_m: Standard origin shift for the mirror
    @param a_m: Angle for coordinate rotation (reffered to as alpha in Vertex docs)

    @return m_coords: The points in the mirror coords
    """
    shifted_coords = coords - shift - v_m
    rot_mat = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(a_m), np.sin(a_m)],
            [0.0, -np.sin(a_m), np.cos(a_m)],
        ]
    )
    m_coords = np.zeros(shifted_coords.shape)
    for i, point in enumerate(shifted_coords):
        m_coords[i] = rot_mat @ point
    return m_coords


def global_to_primary(coords, shift):
    """
    Transform from global coordinates to primary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in the primary mirror coords
    """
    return global_to_mirror(coords, shift, v_m1, a_m1)


def global_to_secondary(coords, shift):
    """
    Transform from global coordinates to secondary mirror coordinates

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in the secondary mirror coords
    """
    return global_to_mirror(coords, shift, v_m2, a_m2)


def shift_coords(coords, shift):
    """
    Apply origin shift to coordinate

    @param coords: Array of points to transform
    @param shift: Origin shift in mm (in addition to the standard one for the mirror)

    @return m_coords: The points in the secondary mirror coords
    """
    return coords - shift


def output(file, string):
    """
    Print and save to file at same time

    @param file: File pointer to write to
    @param string: String to print and save
    """
    file.write(string + "\n")
    print(string)


def align_panels(panels, mirror_path, out_file, coord_trans, origin_shift, mirror_fit_func):
    """
    Align panels of mirror

    @param panels: The filenames for each panel in the mirror directory
    @param mirror_path: Path to the mirror directory
    @param out_file: The output file to write to
    @param coord_trans: The coordinate transform to apply to measured points
    @param origin_shift: The origin_shift to pass to coord_trans
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
        # TODO: Make lookup tables for this
        # Temporary for now
        can_points = np.zeros((5, 3))
        adjustors = np.zeros((5, 3))

        # Load pointcloud from data
        # Will need to change genfromtxt args based on what FARO software outputs
        points = np.genfromtxt(panel_path, dtype=float)

        # Transform points to mirror coordinates
        points = coord_trans(points, origin_shift)

        # Fit to mirror surface
        popt, rms = mf.mirror_fit(
            points[:, 0], points[:, 1], points[:, 2], mirror_fit_func
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
            "Move panel " + str(dx) + " ± " + str(dx_err) + " mm to the " + x_dir,
        )
        if dy < 0:
            y_dir = "down"
        else:
            y_dir = "up"
        output(out_file, "Move panel " + str(dy) + " ± " + str(dy_err) + " mm " + y_dir)

        for i in range(len(d_adj)):
            d = d_adj[i]
            d_err = d_adj_err[i]
            if d < 0:
                d_dir = "in"
            else:
                d_dir = "out"
            output(
                out_file,
                "Move adjustor "
                + str(i)
                + " "
                + str(d)
                + " ± "
                + str(d_err)
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
    Valid options are: global, primary, secondary.",
)
parser.add_argument(
    "-s",
    "--shift",
    nargs=3,
    help="Origin shift to apply in mm, overrides setting in config file",
    type=float,
)
args = parser.parse_args()

measurement_dir = args.measurement_dir
coordinates = args.coordinates
origin_shift = args.shift

# Check that measurement directory exists
if not os.path.exists(measurement_dir):
    print("Supplied measurement directory does not exist. Please double check the path")
    sys.exit()

# Load in config file if needed
if (coordinates is None) or (origin_shift is None):
    confpath = os.path.join(measurement_dir, "config.txt")
    if not os.path.exists(confpath):
        print(
            "Config file doesn't exist and equivalent command line arguments were not given"
        )
        sys.exit()
    config = dict(np.genfromtxt(confpath, dtype=str))
    if coordinates is None:
        coordinates = config["coords"]
    if origin_shift is None:
        origin_shift = config["shift"].split()

# Cast shift as float array
origin_shift = np.array(origin_shift, dtype=float)

# Make sure that shift is correct shape
if len(origin_shift) != 3:
    print(
        "Coordinate origin shift invalid shape. \
        Please supply values for x, y, and z in mm seperated by spaces"
    )
    sys.exit()

# Check if coordinate system is valid
valid_coords = ["global", "primary", "secondary"]
if coordinates not in valid_coords:
    print(
        "Coordinate system '",
        coordinates,
        "' not valid\n Please use one of the following instead: global, primary, secondary",
        sep="",
    )
    sys.exit()

# Initialize output file
out_file = open(os.path.join(measurement_dir, "output.txt"))
output(out_file, "Starting alignment procedure for measurement at: " + measurement_dir)
output(out_file, "Using coordinate system: " + coordinates)
output(out_file, "Using origin shift: " + str(origin_shift))

# Align primary mirror
primary_path = os.path.join(measurement_dir, "M1")
if os.path.exists(primary_path):
    output(out_file, "Aligning primary mirror")

    # Get all panel files
    panels = os.listdir(primary_path)
    if len(panels) == 0:
        output(out_file, "No panels found for primary mirror")

    # Figure out which coordinate transform to use
    if coordinates is "global":
        coord_trans = global_to_primary
    elif coordinates is "primary":
        coord_trans = shift_coords
    else:
        coord_trans = primary_to_secondary

    # Align panels
    align_panels(panels, primary_path, out_file, coord_trans, origin_shift, mf.primary_fit_func)