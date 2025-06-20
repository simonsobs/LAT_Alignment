import megham.transform as mt
import numpy as np

m1_nom = np.array(
    [
        [2400.94, -2698.56, 4819.33],
        [-2397.25, -2698.83, 4821.03],
        [2397.31, 2656.61, 2142.25],
        [-2399.23, 2658.4, 2141.8],
    ]
)
m1_1 = np.array(
    [
        [2400.25555, -2704.48242, 4822.016575],
        [-2397.77349, -2703.48859, 4825.26354],
        [2397.059225, 2652.766942, 2148.898725],
        [-2399.1326, 2655.754072, 2150.259505],
    ]
)
aff_m1, sft_m1 = mt.get_rigid(m1_1, m1_nom, method="mean")
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
sft_m1
rot_m1
m1_2 = np.array(
    [
        [2400.5737, -2702.0379, 4815.3764],
        [-2397.5774, -2701.6915, 4817.5906],
        [2397.0047, 2656.7671, 2145.4533],
        [-2399.04199, 2659.0962, 2145.52339],
    ]
)
aff_m1, sft_m1 = mt.get_rigid(m1_2, m1_nom, method="mean")
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
sft_m1
rot_m1
8000 * np.sin(np.deg2rad(-0.006))
8000 * np.cos(np.deg2rad(-0.006))
m1_4 = np.array(
    [
        [2399.9074, -2712.452, 4810.4387],
        [-2398.3389, -2711.1196, 4813.6509],
        [2397.3564, 2650.0409, 2148.0532],
        [-2399.27048, 2653.5114, 2149.0171],
    ]
)
aff_m1, sft_m1 = mt.get_rigid(m1_4, m1_nom, method="mean")
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
sft_m1
rot_m1
m1_nom - m1_4
m1_nom
m1_5 = np.array(
    [
        [2400.6571, -2700.2512, 4818.2599],
        [-2397.6446, -2699.7200, 4820.5443],
        [2397.5671, 2655.5117, 2141.8443],
        [-2399.3142, 2658.1266, 2141.88195],
    ]
)
aff_m1, sft_m1 = mt.get_rigid(m1_4, m1_nom, method="mean")
aff_m1, sft_m1 = mt.get_rigid(m1_5, m1_nom, method="mean")
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
sft_m1
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
rot_m1
m2_nom = np.array(
    [
        [-1998.53, -3762.8, -2550.87],
        [1993.61, -3763.22, -2551.27],
        [1998.7, -5497.53, 2652.13],
        [-1995.38, -5496.5, 2651.13],
    ]
)
m2 = mt.apply_transform(m2_nom, aff_m1, sft_m1)
from lat_alignment import transforms as lt

m2_loc = lt.coord_transform(m2, "opt_global", "opt_secondary")
m2_nom_loc = lt.coord_transform(m2_nom, "opt_global", "opt_secondary")
aff, sft = mt.get_rigid(m2_loc, m2_nom_loc, method="mean")
aff
rot = np.rad2deg(mt.decompose_rotation(aff))
rot
sft
m1_sft
sft_m1
0.84 * np.sin(np.deg2rad(26.57))
0.84 * np.cos(np.deg2rad(26.57))
m1_sft
m1_5
m1_6 = np.array(
    [
        [2400.6631, -2699.1928, 4817.9389],
        [-2397.6276, -2699.31463, 4820.6279],
        [2396.781, 2656.6625, 2142.0119],
        [-2399.8378, 2658.6696, 2142.4941],
    ]
)
aff_m1, sft_m1 = mt.get_rigid(m1_6, m1_nom, method="mean")
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
sft_m1
rot_m1
m2 = mt.apply_transform(m2_nom, aff_m1, sft_m1)
m2_loc = lt.coord_transform(m2, "opt_global", "opt_secondary")
aff, sft = mt.get_rigid(m2_loc, m2_nom_loc, method="mean")
rot = np.rad2deg(mt.decompose_rotation(aff))
sft
rot
