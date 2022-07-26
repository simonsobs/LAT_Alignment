"""
Visualize and generate test measurement patterns
Author: Nathnael Kahassai

"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import adjustments as adj
import coordinate_transforms as ct
import mirror_fit as mf
import matplotlib.pyplot as plt

#program, panel path, factor to reduce points by
[program, panel_path, factor] = sys.argv

panel_name = panel_path[-13:-4] # XX-XXXXXXXX.txt

########################
#IMPORT MIRROR ADJUSTERS

#Pick Between M1, M2
if int(panel_path.find('M2')) == -1:
    adj_path = 'can_points/M1.txt'
else:
    adj_path = 'can_points/M2.txt'

#Import txt
adj_coord = np.char.split(np.genfromtxt(adj_path, usecols=(0), dtype=str, delimiter="\n"))

adj_table = []
for i in adj_coord:
    adj_table.append(i)

#Find Specific Panel
adj_location = []
for sublist in adj_table:
    if sublist[0] == panel_name:
        adj_location.append(sublist)

#Adjuster Locations
adjx = [p[2] for p in adj_location]
adjy = [p[3] for p in adj_location]
adjz = [p[4] for p in adj_location]

#########################
#Extract FARO Test Points
x = np.genfromtxt(panel_path, skip_header=1, usecols=(3), dtype=str, delimiter="\t") 
y = np.genfromtxt(panel_path, skip_header=1, usecols=(4), dtype=str, delimiter="\t") 
z = np.genfromtxt(panel_path, skip_header=1, usecols=(5), dtype=str, delimiter="\t") 

#Transformation
threed = np.array((x,y,z))
#tran = ct.cad_to_secondary(threed, [[0], 0])
print(threed)

newx = []
newy = []
newz = []
for a,b,c in zip(x,y,z):
    newx.append(float(a.replace(',','')))
    newy.append(float(b.replace(',','')))
    newz.append(float(c.replace(',','')))

#Reduce by factor
newx = newx[::int(factor)]
newy = newy[::int(factor)]
newz = newz[::int(factor)]


#Plotting
fig, ax = plt.subplots()
ax.plot(newx, newy, 'o', color='black')
ax.plot(adjx, adjy, '*', color='red')
ax.set_title('FARO MEASUREMENT POINTS')
ax.set_xlabel('X')
ax.set_ylabel('Y')
#ax.set_aspect('equal')

blank = [0] * len(newx)

#Export txt
datum = np.column_stack([blank,blank,newx,newy,newz])
datafile_path = panel_path + panel_name + '.txt'
np.savetxt(datafile_path , datum, delimiter='\t', fmt='%f')


plt.show()