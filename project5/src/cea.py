"""
Purpose: Project 5 - DOE with rocket engine properties and ML model to replicate CEA
Author: Syam Evani
"""

# Standard imports
import os

# Additional imports
import numpy as np
import itertools
import pandas as pd
from rocketcea.cea_obj import CEA_Obj
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Local imports
# None

#--------------------------------------------------------
# Factor design
#--------------------------------------------------------
# Three factors
pc = np.linspace(50, 500, 10)   # [psia] Chamber pressure
mr = np.linspace(0.5, 5, 10)    # [--]   Ratio of ox/fuel mass
e = np.linspace(10, 100,10)     # [--]   Ratio of exit area to throat area

# Generate all possible combinations
combinations = list(itertools.product(pc, mr, e))

#--------------------------------------------------------
# Generate data and visuals to show the data
#--------------------------------------------------------
results = []
C = CEA_Obj( oxName='O2', fuelName='CH4')
for i, design in enumerate(combinations):
    isp = C.get_Isp(Pc=design[0], MR=design[1], eps=design[2])   # [s] Specific impulse vacuum
    results.append((design[0], design[1], design[2], isp))

# Convert results to a pandas df and output to text file
isp_results = pd.DataFrame(results, columns=['pc', 'mr', 'e', 'isp'])
txt_file_path = os.path.join(os.getenv('USERPROFILE'), 'repos', 'mece-6397-doe', 'project5', 'output', 'isp_results.txt')
with open(txt_file_path, 'w') as file:
    file.write(isp_results.to_string(index=False))

# Make scatter plot of data
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(isp_results['pc'], isp_results['mr'], isp_results['e'], c=isp_results['isp'], cmap='viridis')
plt.colorbar(sc, label=r'$I_{SP}$ [s]')
ax.set_xlabel('Pc [psia]')
ax.set_ylabel('MR')
ax.set_zlabel(r'$\epsilon$')
ax.set_box_aspect(aspect=None, zoom=0.9)
plt.savefig(os.path.join(os.getenv('USERPROFILE'),'repos', 'mece-6397-doe', 'project5', 'output', 'isp_data.png'))
plt.close()

# Contour plot: Pc vs MR
xi = np.linspace(isp_results['pc'].min(), isp_results['pc'].max(), 100)
yi = np.linspace(isp_results['mr'].min(), isp_results['mr'].max(), 100)
zi = griddata((isp_results['pc'], isp_results['mr']), isp_results['isp'], (xi[None, :], yi[:, None]), method='cubic')
plt.contourf(xi, yi, zi, levels=14, cmap='viridis')
plt.colorbar(label=r'$I_{SP}$ [s]')
plt.xlabel('Pc')
plt.ylabel('MR')
plt.title(r'Pc vs MR for $I_{SP}$')
plt.savefig(os.path.join(os.getenv('USERPROFILE'),'repos', 'mece-6397-doe', 'project5', 'output', 'contour_pc_mr.png'))
plt.close()

# Contour plot: e vs MR
xi = np.linspace(isp_results['e'].min(), isp_results['e'].max(), 100)
yi = np.linspace(isp_results['mr'].min(), isp_results['mr'].max(), 100)
zi = griddata((isp_results['e'], isp_results['mr']), isp_results['isp'], (xi[None, :], yi[:, None]), method='cubic')
plt.contourf(xi, yi, zi, levels=14, cmap='viridis')
plt.colorbar(label=r'$I_{SP}$ [s]')
plt.xlabel(r'$\epsilon$')
plt.ylabel('MR')
plt.title(r'$\epsilon$ vs MR of $I_{SP}$')
plt.savefig(os.path.join(os.getenv('USERPROFILE'),'repos', 'mece-6397-doe', 'project5', 'output', 'contour_e_mr.png'))
plt.close()