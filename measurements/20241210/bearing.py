import matplotlib.pyplot as plt
import numpy as np
from megham.transform import apply_transform, decompose_affine, get_affine, get_rigid
from megham.utils import make_edm

from lat_alignment.transforms import coord_transform

path = "./20241210_Bearing_Bundle3.csv"
labels = np.genfromtxt(path, dtype=str, delimiter=",", usecols=(0,))
coords = np.genfromtxt(path, dtype=np.float32, delimiter=",", usecols=(1, 2, 3))
nest_labels = ["TARGET49", "TARGET8", "TARGET65", "TARGET26"]
nest_n = np.array([9.0, 14.0, 24.0, 29.0]) - 1
nest_x = 2040.0 * np.sin(np.deg2rad(10 * nest_n))
nest_y = 3990.4 * np.ones(len(nest_labels)) + 100
nest_z = -2040.0 * np.cos(np.deg2rad(10 * nest_n))
nest_model = np.column_stack([nest_x, nest_y, nest_z])
nest_meas = [coords[labels == l] for l in nest_labels]
nest_meas = np.vstack(nest_meas)
triu_idx = np.triu_indices(len(nest_meas), 1)
scale_fac = np.nanmedian(make_edm(nest_model)[triu_idx] / make_edm(nest_meas)[triu_idx])
nest_meas *= scale_fac
aff, sft = get_rigid(nest_meas, nest_model, method="mean")
nest_meas_t = apply_transform(nest_meas, aff, sft)
plt.scatter(nest_model[:, 0], nest_model[:, 2], c=nest_model[:, 1])
plt.scatter(nest_meas_t[:, 0], nest_meas_t[:, 2], c=nest_meas_t[:, 1])
plt.show()
print(nest_model - nest_meas_t)
coords_t = apply_transform(coords * scale_fac, aff, sft)

zero = coords_t[labels == "TARGET12"]
print(list(zero))
zero = coords_t[labels == "CODE90"]
print(list(zero))

# coords_t = coord_transform(coords_t, "opt_global", "va_global")
ref_labels = ["TARGET72", "TARGET69", "TARGET66", "TARGET70"]
ref = [coords_t[labels == l] for l in ref_labels]
ref = np.vstack(ref)

codes = ["CODE91", "CODE92", "CODE93", "CODE94"]
for r, c in zip(ref, codes):
    print([list(r), [c]])
