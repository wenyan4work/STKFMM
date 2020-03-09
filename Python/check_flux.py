import numpy as np
from scipy.integrate import trapz

vecnames = ["ux.xhat", "uy.yhat", "uz.zhat", "lapux.xhat", "lapuy.yhat", "lapuz.zhat"]

for p in [6, 8, 10, 12, 14]:
  with open("trgPoints{}_128.txt".format(p), "r") as f:
      r_vectors = []
      ulapu = []
      for line in f:
          srcPair = line.split(';')
          coord = [float(i) for i in srcPair[0].split(' ')]
          val = [float(i) for i in srcPair[1].strip().split(' ')]
          r_vectors.append(coord)
          ulapu.append(val)

      ulapu = np.array(ulapu)
      r_vectors = np.array(r_vectors)

  # move periodic vectors back where they belong
  r_vectors[r_vectors==1]=0.0

  # Get number of points in each dimension
  nPts = int(round((r_vectors.shape[0])**(1./3.)))
  assert(nPts**3 == r_vectors.shape[0])
  x = r_vectors[0:nPts, 2]

  print("p: {}".format(p))
  # calc flux for u and lapu
  for dim in range(0, 6):
    val=ulapu[r_vectors[:,dim % 3]==0, dim].reshape((nPts, nPts))

    print("{}: {}".format(vecnames[dim], trapz(trapz(val,x), x)))
