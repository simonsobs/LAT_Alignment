# coding: utf-8
import megham.transform as mt
import numpy as np
m2_nom = np.array([[-1998.53, -3762.8, -2550.87],[1993.61, -3763.22, -2551.27],[1998.7, -5497.53, 2652.13],[-1995.38, -5496.5, 2651.13]])
m2 = np.array([[-2012.894, -3661.579, -2538.597],[1979.745, -3662.224, -2559.774],[2011.505,-5401.453,2642.541],[-1982.579,-5400.197,2662.124]])
aff_m2, sft_m2 = mt.get_rigid(m2, m2_nom, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
from lat_alignment import transforms as lt
get_ipython().run_line_magic('pinfo', 'lt.affine_basis_transform')
lt.affine_basis_transform(aff_m2, sft_m2, "opt_global", "va_global", False)
aff_va, sft_va = lt.affine_basis_transform(aff_m2, sft_m2, "opt_global", "va_global", False)
rot_va = np.rad2deg(mt.decompose_rotation(aff_va))
rot_va
lt.affine_basis_transform(aff_m2, sft_m2, "opt_global", "va_global", True)
sft_va
sft_m2
rot_m2
rot_va
sft_m2
rot_m2
aff_m2@aff_m2.T
sft_m2@(aff_m2.T)
rot_m2
aff_m2
m2
m2 - m2_nom
m2_nom
m2@aff_m2
m2@aff_m2 - m2
m2
m2_2 = np.array([[-2015.9823, -3666.8686, -2542.5121],[1976.6936, -3665.8099, -2563.8122],[2008.0144,-5400.0011,2640.1319],[-1985.90265,-5397.5004,2659.6216]])
aff_m2, sft_m2 = mt.get_rigid(m2_2, m2_nom, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
sft_m2[0] = 0
m2_2@aff_m2 + sft
m2_2@aff_m2 + sft_m2
m2
m2_2
m2_2@aff_m2 + sft_m2
m2_2@aff_m2
m2_3 = np.array([[-2015.3309, -3663.6039, -2544.57699],[1977.3001, -3665.3086, -2565.9097],[2008.70694,-5400.8096,2637.5768],[-1985.2571,-5398.4858,2657.1557]])
aff_m2, sft_m2 = mt.get_rigid(m2_3, m2_nom, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
y_cm = np.array([0, -97.96905643, 0])
aff_m2, sft_m2 = mt.get_rigid(m2_3, m2_nom, method='mean')
aff_m2, sft_m2 = mt.get_rigid(m2_3 + y_cm, m2_nom, method='mean')
sft_m2
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
rot_m2
m2_4 = np.array([[-2013.80477, -3664.35761, -2544.391234],[1978.23959, -3665.3005, -2566.1368],[2010.7395,-5400.9365,26371216],[-1983.261,-5399.2626,2656.9356]])
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom, method='mean')
rot_m2
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
m2_4 = np.array([[-2013.80477, -3664.35761, -2544.391234],[1978.23959, -3665.3005, -2566.1368],[2010.7395,-5400.9365,2637.1216],[-1983.261,-5399.2626,2656.9356]])
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
m2_4 - m2_3
(m2_4 + y_cm)@rot
(m2_4 + y_cm)@rot_m2
(m2_4 + y_cm)@aff_m2
m2_4 + y_cm
from scipy.spatial.transform import Rotation as R
r2 = R.from_euler("Y", [-1*rot_m2[1]], degrees=True)
m2_nom_rot = r2.apply(m2_nom)
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom_rot, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
rot_m2
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
r2 = R.from_euler("Y", [rot_m2[1]], degrees=True)
m2_nom_rot = r2.apply(m2_nom)
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
rot_m2
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom_rot, method='mean')
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
aff_m2, sft_m2 = mt.get_rigid(m2_4 + y_cm, m2_nom_rot, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
(m2_4 + y_cm)@aff_m2 + sft_m2
((m2_4 + y_cm)@aff_m2 + sft_m2) - m2_nom_rot
m2_5 = np.array([[-2011.3187, -3664.7254, -2545.0128],[1981.3521, -3663.4957, -2566.35556],[2013.4447,-5400.50511,2636.6325],[-1980.8064,-5500.9025,2655.47103]])
aff_m2, sft_m2 = mt.get_rigid(m2_5 + y_cm, m2_nom_rot, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2[0] = 0
sft_m2
aff_m2, sft_m2 = mt.get_rigid(m2_5 + y_cm, m2_nom_rot, method='mean')
sft_m2
m2_5 = np.array([[-2011.3187, -3664.7254, -2545.0128],[1981.3521, -3663.4957, -2566.35556],[2013.4447,-5400.50511,2636.6325],[-1980.8064,-5500.9025,2655.47103]])
m2_5 - m2_4
m2_5 = np.array([[-2011.3187, -3664.7254, -2545.0128],[1981.3521, -3663.4957, -2566.35556],[2013.4447,-5400.50511,2636.6325],[-1980.8064,-5400.9025,2655.47103]])
aff_m2, sft_m2 = mt.get_rigid(m2_5 + y_cm, m2_nom_rot, method='mean')
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
rot_m2
