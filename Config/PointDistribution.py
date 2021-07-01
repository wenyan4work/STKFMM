import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import argparse

# define rc params
params = {
    # 'backend': 'Agg',
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath} \usepackage[notextcomp]{stix} \usepackage[T1]{fontenc} \usepackage{bm}'}
plt.rcParams.update(params)

parser = argparse.ArgumentParser()
parser.add_argument("point_file")
args = parser.parse_args()
print("Parsing "+args.point_file)

pts3D = np.loadtxt(args.point_file, delimiter=',')[:, 0:3]

skip = 500
azm = -60

fig = plt.figure(figsize=(4.0, 3.0), dpi=150, constrained_layout=True)
ax1 = fig.add_subplot(111, projection='3d')
ax1.scatter(pts3D[::skip, 0], pts3D[::skip, 1], pts3D[::skip, 2], marker='x')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')
# ax1.set_xlim(0, 1)
# ax1.set_ylim(0, 1)
# ax1.set_zlim(0, 1)
ax1.grid(False)
ax1.view_init(10, azm)
# ax1.set_aspect('equal')

plt.savefig('PointDist.png', dpi=600)
