import matplotlib.pyplot as plt
import numpy as np

#panel_path = 'measurements/20220630_01/M1/01-011411.txt'
#panel_path = 'measurements/20220707_01/M2/01-012551.txt'
panel_path = 'measurements/20220707_02/M2/01-012551.txt'

x = np.genfromtxt(panel_path, skip_header=1, usecols=(3), dtype=str, delimiter="\t") 
y = np.genfromtxt(panel_path, skip_header=1, usecols=(4), dtype=str, delimiter="\t") 

newx = []
newy = []
for a,b in zip(x,y):
    newx.append(float(a.replace(',','')))
    newy.append(float(b.replace(',','')))
    
fig, ax = plt.subplots()
ax.plot(newx, newy, 'o', color='black')
ax.set_title('Line Graph using NUMPY')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_aspect('equal')

plt.show()