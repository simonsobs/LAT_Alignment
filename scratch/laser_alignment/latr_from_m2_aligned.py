
import megham.transform as mt
import numpy as np

latr_nom = np.array([[-1030.0000046587363, 6996.4, -179.99997401227625], [-359.1154521541986, 6996.4, 982.0061569386937], [359.11545215419835, 6996.4, 982.0061569386938], [1030.0000046587363, 6996.4, -179.99997401227645]])
latr_pt_5 = latr_nom[2]
latr_pt_5[2] *= -1
latr_nom = np.vstack([latr_nom[[True, False, True, True]], latr_pt_5])
latr_nom
latr_1 = np.array([[-1031.0669, 6996.39, -191.1107], [341.44298, 6987.692, 990.4713], [1028.9343, 6985.7814, -161.7898], [-348.7398, 6993.8709, -983.7343])
latr_1 = np.array([[-1031.0669, 6996.39, -191.1107], [341.44298, 6987.692, 990.4713], [1028.9343, 6985.7814, -161.7898], [-348.7398, 6993.8709, -983.7343]])
aff_l, sft_; = mt.get_rigid(latr_1, latr_nom, method='mean')
aff_l, sft_l = mt.get_rigid(latr_1, latr_nom, method='mean')
sft_l
latr_nom
latr_q
latr_1
latr_nom[-1][0] *= -1
latr_nom
latr_nom - latr_1
latr_nom[1][2] *= -1
latr_nom - latr_1
aff_l, sft_l = mt.get_rigid(latr_1, latr_nom, method='mean')
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
sft_l
rot_l
latr_2 = np.array([[-1033.955, 6996.382, -176.78], [355.286, 6987.606, 985.385], [1026.4807, 6985.822, -176.471], [-362.497, 6993.902, -978.791]])
aff_l, sft_l = mt.get_rigid(latr_2, latr_nom, method='mean')
sft_l
latr_nom
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
rot_l
latr_nom - latr_2
latr_3 = np.array([[-1033.4623, 6997.069, -176.876], [355.611, 6995.1213, 985.1908], [1026.819, 6995.516, -176.654], [-362.4245, 6997.8483, -979.0334]])
aff_l, sft_l = mt.get_rigid(latr_2, latr_nom, method='mean')
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
rot_l
aff_l, sft_l = mt.get_rigid(latr_3, latr_nom, method='mean')
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
rot_l
latr_nom - latr_3
latr_4 = np.array([[-1033.1582, 6995.97, -176.4853], [356.0597, 6994.637, 985.5317], [1027.1206, 6995.4984, -176.393], [-362.1488, 6996.3764, -978.699]])
aff_l, sft_l = mt.get_rigid(latr_4, latr_nom, method='mean')
rot_l
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
rot_l
latr_nom - latr_4
sft_l
latr_5 = np.array([[-1030.557, 6996.200, -178.308], [359.08, 6996.6566, 983.343], [1029.946, 6997.0637, -178.75], [-359.5163, 6996.426, -980.6245]])
latr_6 = np.array([[-1030.557, 6996.200, -178.308], [359.08, 6996.6566, 983.343], [1029.946, 6997.0637, -178.75], [-359.5163, 6996.426, -980.6245]])
aff_l, sft_l = mt.get_rigid(latr_6, latr_nom, method='mean')
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
sft_l
sft_l
rot_l
latr_nom - latr_6
