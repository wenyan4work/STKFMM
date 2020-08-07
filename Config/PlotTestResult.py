import numpy as np
import scipy as sp
import matplotlib as mpl
import matplotlib.pyplot as plt

import argparse
import json


component = {
    'laplace_PGrad': [r'$\phi$', r'$\phi_{,x}$', r'$\phi_{,y}$', r'$\phi_{,z}$'],
    'laplace_PGradGrad': [r'$\phi$', r'$\phi_{,x}$', r'$\phi_{,y}$', r'$\phi_{,z}$', r'$\phi_{,xx}$', r'$\phi_{,xy}$', r'$\phi_{,xz}$', r'$\phi_{,yy}$', r'$\phi_{,yz}$', r'$\phi_{,zz}$'],
    'laplace_QPGradGrad': [r'$\phi$', r'$\phi_{,x}$', r'$\phi_{,y}$', r'$\phi_{,z}$', r'$\phi_{,xx}$', r'$\phi_{,xy}$', r'$\phi_{,xz}$', r'$\phi_{,yy}$', r'$\phi_{,yz}$', r'$\phi_{,zz}$'],
    'stokes_vel': [r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_regvel': [r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_regftvelomega': [r'$u_x$', r'$u_y$', r'$u_z$', r'$\omega_x$', r'$\omega_y$', r'$\omega_z$'],
    'rpy_ulapu': [r'$u_x$', r'$u_y$', r'$u_z$', r'$\nabla^2 u_x$', r'$\nabla^2 u_y$', r'$\nabla^2 u_z$'],
    'stokes_PVel': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$'],
    'stokes_PVelGrad': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$', r'$p_{,x}$', r'$p_{,y}$', r'$p_{,z}$', r'$ux_{,x}$', r'$ux_{,y}$', r'$ux_{,z}$', r'$uy_{,x}$', r'$uy_{,y}$', r'$uy_{,z}$', r'$uz_{,x}$', r'$uz_{,y}$', r'$uz_{,z}$'],
    'stokes_PVelLaplacian': [r'$p$', r'$u_x$', r'$u_y$', r'$u_z$', r'$\nabla^2 u_x$', r'$\nabla^2 u_y$', r'$\nabla^2 u_z$'],
    'stokes_Traction': [r'$\sigma_{xx}$', r'$\sigma_{xy}$', r'$\sigma_{xz}$', r'$\sigma_{yx}$', r'$\sigma_{yy}$', r'$\sigma_{yz}$', r'$\sigma_{zx}$', r'$\sigma_{zy}$', r'$\sigma_{zz}$']
}


def parseError(error):
    '''
    convert error from a list of dict to numpy 2D array
    '''
    if not error:
        return None

    result = []
    for item in error:
        result.append([item["drift"], item["driftL2"],
                       item["errorRMS"], item["errorL2"], item["errorMaxRel"]])
    return np.transpose(np.array(result))


def plotRecord(ax, multOrder, treeTime, runTime, name, error, compname):
    dim = error.shape[2]
    for i in range(dim):
        ax.semilogy(multOrder, error[:, 1, i], 'x', label=compname[i])
    ax.set_prop_cycle(None)  # reset color cycle
    for i in range(dim):
        ax.semilogy(multOrder, error[:, 3, i], '--o')

    ax.set_title(name)
    ax.legend(loc='upper left', ncol=4,
              title=r'component error: x $\delta_{L2}$, o $\epsilon_{L2}$')
    ax.set_xlabel(r"$m$")
    ax.set_ylabel("Error")
    ax.set_ylim(1e-15, 1)
    axt = ax.twinx()
    axt.plot(multOrder, runTime, '-', label=r'$t_{run}$')
    axt.plot(multOrder, treeTime, '-', label=r'$t_{tree}$')
    axt.set_ylabel("time")
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
        figsize=(6.0*nax, 5.0), dpi=150, constrained_layout=True)
    axs = fig.subplots(nrows=1, ncols=nax, squeeze=False)
    index = 0
    for k in error.keys():
        error[k] = np.array(error[k])
        ax = axs.flat[index]
        plotRecord(ax, multOrder, treeTime, runTime,
                   kernel+' '+k, error[k], component[kernel])
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
