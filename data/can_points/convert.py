import argparse as ap
import os

import numpy as np

# Parse command line arguments
parser = ap.ArgumentParser(
    description="Convert CSV of vertex adjustor spreadsheet to text file readable by alignment script"
)
parser.add_argument("path", help="Path to file to convert")
args = parser.parse_args()
path = args.path

pts = np.genfromtxt(path, dtype=str, skip_header=2, delimiter=",")

blank = np.zeros(pts.shape[1], dtype=str)
mask = ~(pts == blank).all(1)
pts = pts[mask]

col_mask = np.ones(pts.shape[1], dtype=bool)
col_mask[[0, 6, 7]] = False
pts = pts[:, col_mask]

copy = np.where(pts[:, 4] == "-->")[0]
pts[copy, 2:5] = pts[copy, 7:10]

copy = np.where(pts[:, 7] == "<--")[0]
pts[copy, 7:10] = pts[copy, 2:5]

new_pts = np.zeros((2 * pts.shape[0], int(pts.shape[1] / 2)), dtype="<U12")
new_pts[: pts.shape[0]] = pts[:, : new_pts.shape[1]]
new_pts[pts.shape[0] :] = pts[:, new_pts.shape[1] :]


blank = np.zeros(new_pts.shape[1], dtype=str)
mask = ~(new_pts == blank).all(1)
pts = new_pts[mask]

new_path = os.path.splitext(path)[0] + ".txt"
np.savetxt(new_path, pts, fmt="%s")
