# coding: utf-8
import numpy as np
scott = np.array([[2398.88,-2698.11,4831.90], [-2399.53,-2697.06,4832.30], [2397.23,2655.97,2152.34], [-2399.41, 2659.02, 2150.48]])
us = np.array([[2417.36,-6592.02,4821.94], [-2381.32,-6593.76,4835.90], [2405.07,-1237.00,2144.49], [-2391.57,-1236.61,2158.51]])
import megham.transform as mt
aff, sft = get_affine(scott, us)
aff, sft = mt.get_affine(scott, us)
sft
aff, sft = mt.get_rigid(scott, us)
sft
mt.decompose_rotation(aff)
get_ipython().run_line_magic('pinfo', 'mt.decompose_rotation')
rot = np.rad2deg(mt.decompose_rotation(aff))
rot
sft
aff2d, sft2d = mt.get_rigid(scott[:, [True, False, True]], us[:, [True, False, True]])
rot2d = np.rad2deg(mt.decompose_rotation(aff2d))
sft2d
rot2d
aff, sft = mt.get_rigid(scott, us, method='mean')
rot = np.rad2deg(mt.decompose_rotation(aff))
sft
rot
aff2d, sft2d = mt.get_rigid(scott[:, [True, False, True]], us[:, [True, False, True]], method='mean')
rot2d = np.rad2deg(mt.decompose_rotation(aff2d))
sft2d
rot2d
