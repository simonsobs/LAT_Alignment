import numpy as np

from lat_alignment.transforms import coord_transform

labels = ["CODE121", "CODE122", "CODE123", "CODE124"]
y = 6996.4 * np.ones(len(labels))

theta_off = 9.91275
r = 2091.21974 / 2
thetas = np.deg2rad(
    np.array([90 - theta_off, 150 + theta_off, 210 - theta_off, 270 + theta_off])
)
x = -1 * r * np.sin(thetas)
z = -1 * r * np.cos(thetas)

refs = np.column_stack((x, y, z))
# refs = coord_transform(refs, "opt_global", "va_global")

for ref, label in zip(refs, labels):
    print([list(ref), [label]])
