"""
Visualize and generate test measurement patterns
Author: Nathnael Kahassai

"""

import matplotlib.pyplot as plt
import numpy as np
import sys

[program, panel_path, factor] = sys.argv

x = np.genfromtxt(panel_path, skip_header=1, usecols=(3), dtype=str, delimiter="\t") 
y = np.genfromtxt(panel_path, skip_header=1, usecols=(4), dtype=str, delimiter="\t") 

newx = []
newy = []
for a,b in zip(x,y):
    newx.append(float(a.replace(',','')))
    newy.append(float(b.replace(',','')))

newx = newx[::int(factor)]
newy = newy[::int(factor)]

print(len(newx), len(newy))

fig, ax = plt.subplots()
ax.plot(newx, newy, 'o', color='black')
ax.set_title('FARO MEASUREMENT POINTS')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_aspect('equal')

plt.show()