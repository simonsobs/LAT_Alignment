"""
Visualize and generate test measurement patterns
Author: Nathnael Kahassai

"""

import sys
import os
import math
import shutil
import matplotlib.pyplot as plt
import numpy as np
import coordinate_transforms as ct
import matplotlib.pyplot as plt
from datetime import date

#program, panel path, factor to reduce points by
[program, panel_path, factor, locality] = sys.argv

locality = int(locality)

# XX-XXXXXXXX.txt
panel_name = panel_path[-13:-4] 
print(panel_path)

########################
#IMPORT MIRROR ADJUSTERS

#Pick Between M1, M2
if int(panel_name[5]) == 1:
    adj_path = 'can_points/M1.txt'
    cord_trans = ct.cad_to_primary
    flip_cad = ct.primary_to_cad
    mira = 'M1'
else:
    adj_path = 'can_points/M2.txt'
    cord_trans = ct.cad_to_secondary
    flip_cad = ct.secondary_to_cad
    mira = 'M2'


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

for i in range(len(adjx)):
    adjx[i] = float(adjx[i]) - float(163.24548606)

print(adjy)
print(adjx)


#########################
#Extract FARO Test Points
x = np.genfromtxt(panel_path, skip_header=1, usecols=(3), dtype=str, delimiter="\t") 
y = np.genfromtxt(panel_path, skip_header=1, usecols=(4), dtype=str, delimiter="\t") 
z = np.genfromtxt(panel_path, skip_header=1, usecols=(5), dtype=str, delimiter="\t") 

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

###############
#Transformation

#Transform Points
point_array = np.array((newx,newy,newz), dtype=float)
tran_points = cord_trans(point_array.T, 0).T

#Transform Adjusters
adj_array = np.array((adjx,adjy,adjz), dtype=float)
tran_adj = cord_trans(adj_array.T, 0).T

print(tran_adj)

###########
#Sectioning

proximity = 150 #millimeters in proximity

#Adjuster Locality

def adj_local(x, y, z, adj, lim):
    adj_x = adj[0][0:5]
    #res = sorted(zip(adj_x, adj_y), key=lambda k: [k[1], k[0]])

    out_x = []
    out_y = []
    out_z = []

    j = 0
    while j <= (len(adj_x)-1):
        i = 0
        p = 0 #max points near adjusters = p
        while i <= (len(x)-1):
            a = math.dist((x[i],y[i]),(adj[0][j],adj[1][j]))
            if a <= lim and p < 6:
                out_x.append(x[i])
                out_y.append(y[i])
                out_z.append(z[i])
                p += 1
            i += 1
        j += 1

    return out_x, out_y, out_z

#Inverse Adjuster Locality

def adj_far(x, y, z, adj, lim):
    anti = adj_local(x, y, z, adj, lim)

    out_x = [p for p in x if p not in anti[0]]
    out_y = [p for p in y if p not in anti[1]]
    out_z = [p for p in z if p not in anti[2]]

    return out_x, out_y, out_z


#########
#Plotting

fig, ax = plt.subplots()

#Pointcloud Settings
# 0 for Normal, 1 for Cerca Adjusters, 2 for Lejos Adjusters

if locality == 0:
    ax.plot(tran_points[0], tran_points[1], 'o', color='black')
elif locality == 1:
    proxima_points = adj_local(tran_points[0],tran_points[1],tran_points[2],tran_adj,proximity)
    print(len(proxima_points[0]),len(proxima_points[0]),len(tran_points[0]))
    ax.plot(proxima_points[0], proxima_points[1], 'o', color='blue')

elif locality == 2:
    proxima_points = adj_far(tran_points[0],tran_points[1],tran_points[2],tran_adj,proximity)
    print(len(proxima_points[0]),len(proxima_points[0]),len(tran_points[0]))
    ax.plot(proxima_points[0], proxima_points[1], 'o', color='green')

ax.plot(tran_adj[0], tran_adj[1], '*', color='red')
ax.set_title('FARO Measurement Points for Panel ' + panel_name)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
plt.show()

#######
#Saving 
'''
#Compile txt
if locality == 1 or locality == 2:
    blank = [0] * len(proxima_points[0])

    #Retransform to compatible coordinates
    adj_retran = np.array((proxima_points[0],proxima_points[1],proxima_points[2]), dtype=float)
    final_retran = flip_cad(adj_retran.T, 0).T
    datum = np.column_stack([blank,blank,blank,final_retran[0],final_retran[1],final_retran[2]])

else:
    blank = [0] * len(newx)
    datum = np.column_stack([blank,blank,blank,newx,newy,newz])

#Folder Creation
today = date.today()
d1 = today.strftime("%Y%m%d")
folder_name = d1 + '_0' + factor + '_' + str(locality) + '_150s_'
os.makedirs(os.path.join('measurements', folder_name, mira))

#Save txt
savefile_path = 'measurements/' + folder_name + "/" + mira
savefile_name = savefile_path + '/' + panel_path
np.savetxt(savefile_name, datum, delimiter='\t', fmt='%f')

#Copy config file
config_path = 'measurements/' + folder_name + '/config.txt'
shutil.copyfile('config.txt', config_path)

#Save plot figure
plot_path = 'measurements/' + folder_name + '/spatial.png'
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.show()


#Create description file
descrip_path = 'measurements/' + folder_name + '/description.txt'
descrip = ['Test pattern reduced by a factor of' + factor]
np.savetxt(descrip_path, descrip, delimiter='\t')
#add future prompt for description

'''