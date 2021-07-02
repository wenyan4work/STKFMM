import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
import json
import re

import os

# define rc params
params = {
    # 'backend': 'Agg',
    'font.family': 'serif',
    'font.size': 9,
    'axes.labelsize': 9,
    'legend.fontsize': 6,
    'xtick.labelsize': 7,
    'ytick.labelsize': 7,
    'text.usetex': True,
    'text.latex.preamble': r'\usepackage{amsmath}' +
    r'\usepackage[notextcomp]{stix}' +
    r'\usepackage[T1]{fontenc}' +
    r'\usepackage{bm}'
}
plt.rcParams.update(params)

component = {
    'laplace_grad': [r'$\partial \phi/\partial x$', r'$\partial \phi/\partial y$', r'$\partial \phi/\partial z$'],
    'laplace_PGrad': [r'$\phi$', r'$\partial \phi/\partial x$', r'$\partial \phi/\partial y$', r'$\partial \phi/\partial z$'],
    'laplace_PGradGrad': [r'$\phi$', r'$\partial \phi/\partial x$', r'$\partial \phi/\partial y$', r'$\partial \phi/\partial z$',
                          r'$\partial^2\phi/\partial x^2$', r'$\partial^2\phi/\partial x\partial y$', r'$\partial^2\phi/\partial x \partial z$',
                          r'$\partial^2\phi/\partial y^2$', r'$\partial^2\phi/\partial y \partial z$', r'$\partial^2\phi/\partial z^2$'],
    'laplace_QPGradGrad': [r'$\phi$', r'$\partial \phi/\partial x$', r'$\partial \phi/\partial y$', r'$\partial \phi/\partial z$',
                           r'$\partial^2\phi/\partial x^2$', r'$\partial^2\phi/\partial x\partial y$', r'$\partial^2\phi/\partial x \partial z$',
                           r'$\partial^2\phi/\partial y^2$', r'$\partial^2\phi/\partial y \partial z$', r'$\partial^2\phi/\partial z^2$'],
    'stokes_vel': [r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_regvel': [r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_regftvelomega': [r'$u_x$', r'$u_y$', r'$u_z$', r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'],
    'rpy_ulapu': [r'$u_x$', r'$u_y$', r'$u_z$', r'$\nabla^2 u_x$', r'$\nabla^2 u_y$', r'$\nabla^2 u_z$'],
    'stokes_PVel': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_PVelGrad': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$', r'$\partial p/\partial x$', r'$\partial p/\partial y$', r'$\partial p/\partial z$',
                        r'$\partial u_x/\partial x$', r'$\partial u_x/\partial y$', r'$\partial u_x/\partial z$',
                        r'$\partial u_y/\partial x$', r'$\partial u_y/\partial y$', r'$\partial u_y/\partial z$',
                        r'$\partial u_z/\partial x$', r'$\partial u_z/\partial y$', r'$\partial u_z/\partial z$'],
    'stokes_PVelLaplacian': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$', r'$\nabla^2 u_x$', r'$\nabla^2 u_y$', r'$\nabla^2 u_z$'],
    'stokes_Traction': [r'$\sigma_{xx}$', r'$\sigma_{xy}$', r'$\sigma_{xz}$', r'$\sigma_{yx}$', r'$\sigma_{yy}$', r'$\sigma_{yz}$', r'$\sigma_{zx}$', r'$\sigma_{zy}$', r'$\sigma_{zz}$']
}

kernelTitle = {
    'laplace_grad': r'Laplace $[q,\bm{d}]\to[\nabla \phi]$',
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

errorLabel = [r'$\epsilon_{\rm RMS}$ for each component',
              r'$\epsilon_{L2}$ for each component',
              r'$\max \epsilon_{\rm rel}$ for each component']
errorName = ['RMS', 'L2Rel', 'MaxRel']


def parseError(error):
    '''
    convert error from a list of dict to numpy 2D array
    '''
    if not error:
        return None

    result = []
    for item in error:
        result.append([
            item["errorRMS"], item["errorL2"], item["errorMaxRel"], item["drift"]
        ]
        )
    return np.transpose(np.array(result))


def plotRecord(ax, multOrder, treeTime, runTime, name, errorname, error, compname):
    dim = error.shape[2]
    # 0: errorRMS, 1: errorL2, 2: errorMaxRel, 3: drift, etc
    for i in range(dim):
        ax.semilogy(multOrder, error[:, plotComponent, i], '--o',
                    fillstyle='none', label=compname[i])

    ax.set_title(name)
    ax.legend(loc='upper left', ncol=3,
              title=errorLabel[plotComponent])
    ax.set_xlabel(r"$m$")
    ax.set_ylabel(errorname)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(5))
    ax.set_ylim(np.amin(error[:, plotComponent, :]) /
                100, np.amax(error[:, plotComponent, :])*100)
    axt = ax.twinx()
    axt.plot(multOrder, runTime, 'x-r', label=r'$t_{run}$')
    axt.plot(multOrder, treeTime, 'x-k', label=r'$t_{tree}$')
    axt.set_ylabel("time (seconds)")
    axt.set_ylim(0, max(runTime)*1.5)
    axt.legend(title='time')

    ax.tick_params(axis="x", direction="in")
    ax.tick_params(axis="y", direction="in")
    axt.tick_params(axis="y", direction="in")

    return


def plotData(data, plotComponent):
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
        figsize=(4.0*nax, 3.0), dpi=600, constrained_layout=True)
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
    foldername = os.path.basename(os.getcwd())
    plt.savefig('Test_'+foldername+'_'+kernel +
                '_'+errorName[plotComponent]+'.png')


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
    for plotComponent in [0, 1, 2]:
        plotData(data, plotComponent)
