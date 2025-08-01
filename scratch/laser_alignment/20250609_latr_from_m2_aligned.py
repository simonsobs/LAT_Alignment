import megham.transform as mt
import numpy as np

latr_nom = np.array(
    [
        [-1030.0000046587363, 6996.4, -179.99997401227625],
        [-359.1154521541986, 6996.4, 982.0061569386937],
        [359.11545215419835, 6996.4, 982.0061569386938],
        [1030.0000046587363, 6996.4, -179.99997401227645],
    ]
)

latr_1 = np.array(
    [
        [-1026.3203, 6971.6790, -188.7562],
        [-367.8071, 6995.9203, 979.9324],
        [350.9053, 6959.7925, 987.3629],
        [1033.788, 6954.1965, -167.5630],
    ]
)
print(latr_1 - latr_nom)
aff_l, sft_l = mt.get_rigid(latr_1, latr_nom, method="mean")
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
print(sft_l)
print(rot_l)

print("\n")
latr_2 = np.array(
    [
        [-1027.1245, 7002.2564, -183.1318],
        [-362.0947, 6996.41, 981.9131],
        [356.6602, 6990.1879, 985.305],
        [1033.1217, 6984.5959, -173.4455],
    ]
)
print(latr_2 - latr_nom)
aff_l, sft_l = mt.get_rigid(latr_2, latr_nom, method="mean")
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
print(sft_l)
print(rot_l)

print("\n")
latr_3 = np.array(
    [
        [-1028.0913, 7002.2792, -177.7705],
        [-356.9708, 6996.3847, 983.778],
        [361.7935, 6990.1613, 983.4305],
        [1032.2109, 6984.596, -178.8448],
    ]
)
print(latr_3 - latr_nom)
aff_l, sft_l = mt.get_rigid(latr_3, latr_nom, method="mean")
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
print(sft_l)
print(rot_l)

print("\n")
latr_5 = np.array(
    [
        [-1029.6509, 6996.0031, -178.8171],
        [-358.5162, 6996.4772, 982.7903],
        [360.2958, 6996.7401, 982.4520],
        [1030.7805, 6996.89, -179.8284],
    ]
)
print(latr_5 - latr_nom)
aff_l, sft_l = mt.get_rigid(latr_5, latr_nom, method="mean")
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
print(sft_l)
print(rot_l)

print("\n")
latr_6 = np.array(
    [
        [-1030.3001, 6996.3941, -179.2926],
        [-359.6773, 6996.5214, 982.177],
        [359.1087, 6996.4176, 982.6029],
        [1030.1029, 6996.2352, -179.3772],
    ]
)
print(latr_6 - latr_nom)
aff_l, sft_l = mt.get_rigid(latr_6, latr_nom, method="mean")
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
print(sft_l)
print(rot_l)

print("\n")
latr_7 = np.array(
    [
        [-1030.2578, 6996.401, -179.2929],
        [-359.6616, 6996.5164, 982.6493],
        [359.1369, 6996.4168, 982.6657],
        [1030.1505, 6996.2558, -179.2974],
    ]
)
print(latr_7 - latr_nom)
aff_l, sft_l = mt.get_rigid(latr_7, latr_nom, method="mean")
rot_l = np.rad2deg(mt.decompose_rotation(aff_l))
print(sft_l)
print(rot_l)
