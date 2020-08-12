import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
import json

# define rc params
params = {
    'backend': 'Agg',
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 7,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': True,
    'text.latex.preamble': [r'\usepackage{amsmath}',
                            r'\usepackage[notextcomp]{stix}',
                            r'\usepackage[T1]{fontenc}',
                            r'\usepackage{bm}']}
plt.rcParams.update(params)

component = {
    'laplace_PGrad': [r'$\phi$', r'$\phi_{,x}$', r'$\phi_{,y}$', r'$\phi_{,z}$'],
    'laplace_PGradGrad': [r'$\phi$', r'$\phi_{,x}$', r'$\phi_{,y}$', r'$\phi_{,z}$', r'$\phi_{,xx}$', r'$\phi_{,xy}$', r'$\phi_{,xz}$', r'$\phi_{,yy}$', r'$\phi_{,yz}$', r'$\phi_{,zz}$'],
    'laplace_QPGradGrad': [r'$\phi$', r'$\phi_{,x}$', r'$\phi_{,y}$', r'$\phi_{,z}$', r'$\phi_{,xx}$', r'$\phi_{,xy}$', r'$\phi_{,xz}$', r'$\phi_{,yy}$', r'$\phi_{,yz}$', r'$\phi_{,zz}$'],
    'stokes_vel': [r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_regvel': [r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_regftvelomega': [r'$u_x$', r'$u_y$', r'$u_z$', r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'],
    'rpy_ulapu': [r'$u_x$', r'$u_y$', r'$u_z$', r'$\nabla^2 u_x$', r'$\nabla^2 u_y$', r'$\nabla^2 u_z$'],
    'stokes_PVel': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_PVelGrad': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$', r'$p_{,x}$', r'$p_{,y}$', r'$p_{,z}$', r'$u_{x,x}$', r'$u_{x,y}$', r'$u_{x,z}$', r'$u_{y,x}$', r'$u_{y,y}$', r'$u_{y,z}$', r'$u_{z,x}$', r'$u_{z,y}$', r'$u_{z,z}$'],
    'stokes_PVelLaplacian': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$', r'$\nabla^2 u_x$', r'$\nabla^2 u_y$', r'$\nabla^2 u_z$'],
    'stokes_Traction': [r'$\sigma_{xx}$', r'$\sigma_{xy}$', r'$\sigma_{xz}$', r'$\sigma_{yx}$', r'$\sigma_{yy}$', r'$\sigma_{yz}$', r'$\sigma_{zx}$', r'$\sigma_{zy}$', r'$\sigma_{zz}$']
}

kernelTitle = {
    'laplace_PGrad': r'Laplace $[q,\bm{d}]\to[\phi,\nabla \phi]$',
    'laplace_PGradGrad': r'Laplace $[q,\bm{d}]\to[\phi,\nabla\phi,\nabla\nabla\phi]$',
    'laplace_QPGradGrad': r'Laplace Quadrupole $\bm{Q}\to[\phi,\nabla\phi,\nabla\nabla\phi]$',
    'stokes_vel': r'Stokeslet $\bm{f}\to\bm{u}$',
    'stokes_regvel': r'Regularized Stokeslet force $[\bm{f},\epsilon]\to\bm{u}$',
    'stokes_regftvelomega': r'Regularized Stokeslet force and torque $[\bm{f},\bm{t},\epsilon]\to[\bm{u},\bm{\omega}]$',
    'rpy_ulapu': r'RPY tensor $[\bm{f},b]\to[\bm{u},\nabla^2\bm{u}]$',
    'stokes_PVel': r'StokesPVel $[\bm{f},q,\bm{D}]\to[p,\bm{u}]$',
    'stokes_PVelGrad': r'StokesPVel $[\bm{f},q,\bm{D}]\to[p,\bm{u},\nabla p,\nabla\bm{u}]$',
    'stokes_PVelLaplacian': r'StokesPVel $[\bm{f},q,\bm{D}] \to [p,\bm{u},\nabla^2\bm{u}]$',
    'stokes_Traction': r'StokesPVel traction $[\bm{f},q,\bm{D}] \to \bm{\sigma}$',
}


def parseError(error):
    '''
    convert error from a list of dict to numpy 2D array
    '''
    if not error:
        return None

    result = []
    for item in error:
        result.append([
            item["errorRMS"], item["errorL2"], item["errorMaxRel"],   # error before drift correction
            item["drift"], item["driftL2"], item["errorL2WithoutDrift"]  # error after drift correction
        ]
        )
    return np.transpose(np.array(result))


def plotRecord(ax, multOrder, treeTime, runTime, name, errorname, error, compname):
    dim = error.shape[2]
    # 0: errorRMS, 1: errorL2, 2: errorMaxRel, 3: drift, etc
    plotComponent = 1
    for i in range(dim):
        ax.semilogy(multOrder, error[:, plotComponent, i], '--o',
                    fillstyle='none', label=compname[i])

    ax.set_title(name)
    ax.legend(loc='upper left', ncol=4,
              title=r'$\epsilon_{L2}$ for each component')
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(errorname)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.set_ylim(np.amin(error[:, plotComponent, :])/100, np.amax(error[:, plotComponent, :])*100)
    axt = ax.twinx()
    axt.plot(multOrder, runTime, 'x-r', label=r'$t_{run}$')
    axt.plot(multOrder, treeTime, 'x-k', label=r'$t_{tree}$')
    axt.set_ylabel("time (second)")
    axt.set_ylim(0, max(runTime)*1.5)
    axt.legend(title='time')

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    axt.tick_params(axis="y", direction="in")

    return


def plotData(data):
    '''
    data should be records for the same kernel
    '''
    data.sort(key=lambda k: k['multOrder'])
    kernel = data[0]["kernel"]
    multOrder = []
    treeTime = []
    runTime = []
    error = dict()
    if data[0]["errorVerify"]:
        error["Verification Error"] = []
    if data[0]["errorConvergence"]:
        error["Convergence Error"] = []
    if data[0]["errorTranslate"]:
        error["Translation Error"] = []

    for record in data:
        multOrder.append(record["multOrder"])
        treeTime.append(record["treeTime"])
        runTime.append(record["runTime"])
        if record["errorVerify"]:
            error["Verification Error"].append(
                parseError(record["errorVerify"]))
        if record["errorConvergence"]:
            error["Convergence Error"].append(
                parseError(record["errorConvergence"]))
        if record["errorTranslate"]:
            error["Translation Error"].append(
                parseError(record["errorTranslate"]))

    nax = len(error.keys())
    fig = plt.figure(
        figsize=(4.0*nax, 3.0), dpi=150, constrained_layout=True)
    # fig.suptitle(kernelTitle[kernel])
    axs = fig.subplots(nrows=1, ncols=nax, squeeze=False)
    index = 0
    for k in error.keys():
        error[k] = np.array(error[k])
        ax = axs.flat[index]
        print(kernelTitle[kernel]+r' '+k)
        plotRecord(ax, multOrder, treeTime, runTime,
                   kernelTitle[kernel], k, error[k], component[kernel])
        index += 1
    plt.savefig('Test_'+kernel+'.png')


parser = argparse.ArgumentParser()
parser.add_argument("logfile")
args = parser.parse_args()
print("Parsing "+args.logfile)

f = open(args.logfile,)
logs = json.load(f)

kernelset = set()
for record in logs:
    kernelset.add(record["kernel"])
print("Found logs for kernels: ", kernelset)

for kernel in kernelset:
    data = []
    for record in logs:
        if record["kernel"] == kernel:
            data.append(record)
    plotData(data)
