"""
Visualize and generate test measurement patterns
Author: Nathnael Kahassai

"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import adjustments as adj
import coordinate_transforms as ct
import mirror_fit as mf
import matplotlib.pyplot as plt
from datetime import date

#program, panel path, factor to reduce points by
[program, panel_path, factor] = sys.argv

# XX-XXXXXXXX.txt
panel_name = panel_path[-13:-4] 
print(panel_name)
print(panel_path)

########################
#IMPORT MIRROR ADJUSTERS

#Pick Between M1, M2
if int(panel_name[5]) == 1:
    adj_path = 'can_points/M1.txt'
    cord_trans = ct.cad_to_primary
    mira = 'M1'
else:
    adj_path = 'can_points/M2.txt'
    cord_trans = ct.cad_to_secondary
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

print(adj_location)

#Adjuster Locations
adjx = [p[2] for p in adj_location]
adjy = [p[3] for p in adj_location]
adjz = [p[4] for p in adj_location]


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

#########
#Plotting
fig, ax = plt.subplots()
ax.plot(tran_points[0], tran_points[1], 'o', color='black')
ax.plot(tran_adj[0], tran_adj[1], '*', color='red')
ax.set_title('FARO Measurement Points for Panel ' + panel_name)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_aspect('equal')
plt.show()

###########
#Sectioning

def split_sort(ex, wy, divd):
    lenx = np.max(wy)-np.min(wy)
    leny = np.max(ex)-np.min(ex)

    x_bin = lenx/divd
    y_bin = leny/divd

    res = sorted(zip(ex, wy), key=lambda k: [k[1], k[0]])

    '''
    lists = []
    for p in range(divd):
        lists.append([])
    '''
    return res, x_bin, y_bin

#print(split_sort(newx,newy,6))

#######
#Saving 

#Compile txt
blank = [0] * len(newx)
datum = np.column_stack([blank,blank,blank,newx,newy,newz])

#Folder Creation
today = date.today()
d1 = today.strftime("%Y%m%d")
folder_name = d1 + '_0' + factor
os.makedirs(os.path.join('measurements', folder_name, mira))

#Save txt
savefile_path = 'measurements/' + folder_name + "/" + mira
savefile_name = savefile_path + '/' + panel_path
np.savetxt(savefile_name, datum, delimiter='\t', fmt='%f')

#Create config file
config_path = 'measurements/' + folder_name + '/config.txt'
c1 = ['coords','shift','compensation']
c2 = ['cad', '0 0 0', '19.05']
ctotal = np.column_stack([c1,c2])
np.savetxt(config_path, ctotal, delimiter='\t')

#Create description file\
descrip_path = 'measurements/' + folder_name + '/description.txt'
descrip = ['Test pattern reduced by a factor of' + factor]
np.savetxt(descrip_path, descrip, delimiter='\t')
