import megham.transform as mt
import numpy as np

from lat_alignment import transforms as lt

m1_nom = np.array(
    [
        [2400.94, -2698.56, 4819.33],
        [-2397.25, -2698.83, 4821.03],
        [2397.31, 2656.61, 2142.25],
        [-2399.23, 2658.4, 2141.8],
    ]
)
m2_nom = np.array(
    [
        [-1998.53, -3762.8, -2550.87],
        [1993.61, -3763.22, -2551.27],
        [1998.7, -5497.53, 2652.13],
        [-1995.38, -5496.5, 2651.13],
    ]
)
m2_nom_loc = lt.coord_transform(m2_nom, "opt_global", "opt_secondary")
m1_meas = np.array(
    [
        [2400.4679, -2700.7464, 4818.4219],
        [-2397.7851, -2700.0503, 4821.8107],
        [2397.6569, 2655.6100, 2142.3663],
        [-2398.9775, 2657.7954, 2142.5339],
    ]
)
aff_m1, sft_m1 = mt.get_rigid(m1_meas, m1_nom, method="mean")
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
print(sft_m1)
print(rot_m1)
m2 = mt.apply_transform(m2_nom, aff_m1, sft_m1)
m2_loc = lt.coord_transform(m2, "opt_global", "opt_secondary")
aff, sft = mt.get_rigid(m2_loc, m2_nom_loc, method="mean")
rot = np.rad2deg(mt.decompose_rotation(aff))
print(sft)
print(rot)
