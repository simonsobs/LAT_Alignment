# coding: utf-8
import megham.transform as mt
import numpy as np

m1 = np.array([2416.614119,-6595.836727,4813.948184],[-2382.073056,-6595.759455,4830.627112],[2405.124325,-1238.157415,2141.693447],[-2392.537523,-1236.687137,2156.648307]])
m1 = np.array([[2416.614119,-6595.836727,4813.948184],[-2382.073056,-6595.759455,4830.627112],[2405.124325,-1238.157415,2141.693447],[-2392.537523,-1236.687137,2156.648307]])
m2 = np.array([[-2005.078291,-7651.933572,-2543.164877],[1987.258081,-7652.457738,-2555.584845],[2008.112123,-9391.239271,2646.833902],[-1986.072995,-9390.194963,2657.575995]])
m1_nom = np.array([[2400.94, -2698.56, 4819.33],[-2397.25, -2698.83, 4821.03],[2397.31, 2656.61, 2142.25],[-2399.23, 2658.4, 2141.8]])
m2_nom = np.array([[-1998.53, -3762.8, -2550.87],[1993.61, -3763.22, -2551.27],[1998.7, -5497.53, 2652.13],[-1995.38, -5496.5, 2651.13]])
aff, sft = mt.get_rigid(m2_nom, m2, method='mean)
aff, sft = mt.get_rigid(m2_nom, m2, method='mean')
m1_exp = mt.apply_transform(m1_nom, aff, aft)
m1_exp = mt.apply_transform(m1_nom, aff, sft)
m1_exp - m1
aff2, sft2 = mt.get_rigid(m1_exp, m1, method='mean')
rot2 = np.rad2deg(mt.decompose_rotation(aff2))
rot2
sft2
m1_exp - m1
m1
rot2
m1_pred = mt.apply_transform(m1_exp, aff2, sft2)
m1_pred-m1
rot2
.0078/57 * 6000
rot
rot = np.rad2deg(mt.decompose_rotation(aff))
rot
sft
aff_m1, sft_m1 = mt.get_rigid(m1, m1_nom, method='mean')
m2_exp = mt.apply_transform(m2, aff_m1, sft_m1)
aff_m2, sft_m2 = mt.get_rigid(m2_exp, m2_nom, method='mean')
np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m2
sft_m1
sft
m1 - m1_nom
(m1 - m1_nom)[:, [True, False, True]]
np.linalg.norm((m1 - m1_nom)[:, [True, False, True]])
np.linalg.norm((m2 - m2_nom)[:, [True, False, True]])
m2 - m2_nom
np.linalg.norm((m2 - m2_nom)[:, [True, False, True]], axis=-1)
sft
sft_m1
sft_m1 + sft
m2 - m2_nom
m1 - m1_nom
(m1 - m1_nom) - (m2 - m2_nom)
aff_m2, sft_m2 = mt.get_rigid(m2, m2_nom, method='mean')
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
sft_m1
sft_m2
sft_m1 - sft_m2
rot_m1 - rot_m2
rot_m1
rot_m2
y_cm_sft = np.array([0, (sft_m1[1] + sft_m2[1])/2, 0])
y_cm_rot = np.array([0, (rot_m1[1] + rot_m2[1])/2, 0])
sft_m1 - y_cm_sft
rot_m1 - y_cm_rot
sft_m2 - y_cm_sft
rot_m2 - y_cm_rot
m1_nom
m1_nom
m2_nom
m1
m2
get_ipython().run_line_magic('pinfo', 'mt.get_rigid')
aff, sft = mt.get_affine(m1, m1_nom, method='mean')
sft
aff, sft = mt.get_affine(m1, m1_nom, method='mean', weights=np.ones(len(m1)))
sft
aff_m1, sft_m1 = mt.get_rigid(m1, m1_nom, method='mean')
sft_m1
aff_m1, sft_m1 = mt.get_rigid(m1, m1_nom, method='median')
sft_m1
y_cm_sft
aff_m1, sft_m1 = mt.get_rigid(m1 - y_cm_sft, m1_nom, method='median')
sft_m1
aff_m1, sft_m1 = mt.get_rigid(m1 - y_cm_sft, m1_nom, method='mean')
sft_m1
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm_sft, m1_nom, method='median')
sft_m1
y_cm_sft
y_cm = np.array([0, 4410, 0])
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm, m1_nom, method='median')
sft_m1
aff_m1, sft_m1 = mt.get_rigid(m1 - y_cm, m1_nom, method='median')
sft_m1
y_cm = np.array([0, 4360, 0])
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm, m1_nom, method='median')
sft_m1
y_cm = np.array([0, 4410-545, 0])
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm, m1_nom, method='median')
sft_m1
y_cm = np.array([0, 4410-545-25.4, 0])
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm, m1_nom, method='median')
sft_m1
y_cm = np.array([0, 4410-545+25.4, 0])
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm, m1_nom, method='median')
sft_m1
sft_m2
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm, m1_nom, method='mean')
sft_m1
aff_m1, sft_m1 = mt.get_affine(m1 + y_cm, m1_nom, method='mean')
sft_m1
aff_m1, sft_m1 = mt.get_rigid(m1 + y_cm, m1_nom, method='mean')
aff_m2, sft_m2 = mt.get_rigid(m2 + y_cm, m2_nom, method='mean')
sft_m2
m1 + y_cm
m1
(m1 + y_cm) - m1_nom
(m2 + y_cm) - m2_nom
m2
m1_nom - mt.apply_transform(m1 + y_cm, aff_m1, sft_m1)
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
rot_m1
sft_m1
rot_m2 = np.rad2deg(mt.decompose_rotation(aff_m2))
rot_m2
m2_nom - mt.apply_transform(m2 + y_cm, aff_m1, sft_m1)
m2_nom - mt.apply_transform(m2 + y_cm, aff_m1, sft_m2)
m2_nom - mt.apply_transform(m2 + y_cm, aff_m2, sft_m2)
y_rot = (rot_m1[1] - rot_m2[1])
from scipy.spatial.transform import Rotation as R
get_ipython().run_line_magic('pinfo', 'R.from_euler')
r = R.from_euler("Y", [y_rot])
r
r.apply(m1)
r.apply(m1) - m1_nom
r = R.from_euler("Y", [-y_rot])
r.apply(m1) - m1_nom
r = R.from_euler("Y", [y_rot], degrees=True)
r.apply(m1) - m1_nom
m1 - m1_nom
r = R.from_euler("Y", [-y_rot], degrees=True)
r.apply(m1) - m1_nom
y_rot
y_rot = (rot_m1[1] + rot_m2[1])/2
r = R.from_euler("Y", [y_rot], degrees=True)
r.apply(m1) - m1_nom
r = R.from_euler("Y", [-y_rot], degrees=True)
r.apply(m1) - m1_nom
m1 - m1_nom
aff_m1, sft_m1 = mt.get_rigid(r.apply(m1) + y_cm, m1_nom, method='mean')
sft_m1
rot_m1
rot_m1 = np.rad2deg(mt.decompose_rotation(aff_m1))
rot_m1
sft_m1
m2
m2_nom
y_cm
4410 - 545
sft_m2
rot_m2
m2 - y_cm
m2 + y_cm
sft_m2
y_cm
sft_m1
rot_m1
rot_m2
m2_nom
