import subprocess


def runCmd(cmd):
    print(cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    process.wait()
    print(process.returncode)
    return


for kernel in ['Laplace', 'Stokeslet', 'StokesPVel']:
    for m in [6, 8, 10, 12, 14, 16]:
        for dim in [1, 2, 3]:
            cmd = './M2L'+kernel+' {:d} {:d} > log'.format(dim, m)+kernel+'_{:d}_{:d}'.format(
                dim, m)
            runCmd(cmd)
