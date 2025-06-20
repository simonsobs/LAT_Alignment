import numpy as np
import glob
import os
import matplotlib.pyplot as plt

def load_data(fname: str):
    dtype = [("feature", 'U100'), ("id", int), ("x", np.float32), ("y", np.float32), ("z", np.float32), ("theta", float), ("deviation", np.float32)]
    data = np.genfromtxt(fname, dtype=dtype, usecols=(1,2,3,4,5,6,9), delimiter=',', skip_header=1)
    data = data[np.argsort(data["id"])]
    return data

def get_angle(data, phase_shift):
    angle = (np.rad2deg(np.arctan2(data['z'], data['x'])) + 360)%360
    # angle -= phase_shift
    angle -= 90
    angle %= 360
    data['theta'] = angle
    return data

files = glob.glob("*.csv")
data = {os.path.splitext(os.path.basename(file))[0] : load_data(file) for file in files}

static = {name:dat for name, dat in data.items() if "static_points" in name.lower()}
bearing = {name:dat for name, dat in data.items() if "bearing" in name.lower()}
m1 = {name:dat for name, dat in data.items() if "m1" in name.lower()}
m2 = {name:dat for name, dat in data.items() if "m2" in name.lower()}

# static_90 = static["Static_Points_90"]
# phase_shifts = {point['feature'].split("_Point")[0]:(np.rad2deg(np.arctan2(point['z'], point['x'])) + 360)%360 for point in static_90}
bearing = {name:get_angle(dat, 0.) for name, dat in bearing.items()}
m1 = {name:get_angle(dat, 0) for name, dat in m1.items()}
m2 = {name:get_angle(dat, 0) for name, dat in m2.items()}
# m1 = {name:get_angle(dat, phase_shifts["_".join(name.split('_')[:3])]) for name, dat in m1.items()}
# m2 = {name:get_angle(dat, phase_shifts["_".join(name.split('_')[:3])]) for name, dat in m2.items()}

# name = "M2_Upper_Right_2"
# dat = m2[name]
# plt.scatter(dat['theta'], dat['deviation'], label=f'{name}')
# plt.show()

for name, dat in bearing.items():
    direction = np.sign(np.gradient(dat["theta"]))
    plt.scatter(dat['theta'][direction == 1], dat['deviation'][direction == 1], label=f'{name}_forwards')
    plt.scatter(dat['theta'][direction == -1], dat['deviation'][direction == -1],label=f'{name}_backwards')
plt.legend()
plt.show()

for name, dat in m1.items():
    plt.scatter(dat['theta'], dat['deviation'], label=f'{name}')
plt.legend()
plt.show()

for name, dat in m1.items():
    direction = np.sign(np.gradient(dat["theta"]))
    plt.scatter(dat['theta'][direction == 1], dat['deviation'][direction == 1], label=f'{name}_forwards')
    plt.scatter(dat['theta'][direction == -1], dat['deviation'][direction == -1],label=f'{name}_backwards')
plt.legend()
plt.show()

for name, dat in m2.items():
    plt.scatter(dat['theta'], dat['deviation'], label=f'{name}')
plt.legend()
plt.show()

for name, dat in m2.items():
    direction = np.sign(np.gradient(dat["theta"]))
    plt.scatter(dat['theta'][direction == 1], dat['deviation'][direction == 1], label=f'{name}_forwards')
    plt.scatter(dat['theta'][direction == -1], dat['deviation'][direction == -1],label=f'{name}_backwards')
plt.legend()
plt.show()
