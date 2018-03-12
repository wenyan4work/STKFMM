/*
 * ChebNodal.cpp
 *
 *  Created on: Oct 4, 2016
 *      Author: wyan
 */

#include "ChebNodal.h"

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>

// chebN equals to the number of points - 1
ChebNodal::ChebNodal(int chebN) : chebN(chebN) {
    points.resize(chebN + 1);
    weights.resize(chebN + 1);

    calcWeight();
}

void ChebNodal::calcWeight() {
    assert(chebN > 0);
    /* Python code:
     * Dkn=np.zeros((pCheb+1,pCheb+1))
     for k in range(pCheb+1):
     for n in range(pCheb+1):
     Dkn[k,n]=np.cos(k*n/pCheb*np.pi)*2.0/pCheb
     if(n==0 or n==pCheb):
     Dkn[k,n]=np.cos(k*n/pCheb*np.pi)*1.0/pCheb
     dvec=np.zeros(pCheb+1)
     for i in range(pCheb+1):
     if(i%2==1):
     dvec[i]=0
     else:
     dvec[i]=2/(1.0-i**2)
     dvec[0]=1
     weightCC=np.dot(Dkn.transpose(),dvec)
     *
     * */
    const double Pi = 3.1415926535897932384626433;

    double *Dkn = new double[(chebN + 1) * (chebN + 1)];
    for (int k = 0; k < chebN + 1; k++) {
        int n = 0;
        Dkn[k * (chebN + 1) + n] = cos(k * n * Pi / chebN) / chebN;
        for (n = 1; n < chebN; n++) {
            Dkn[k * (chebN + 1) + n] = cos(k * n * Pi / chebN) * 2 / chebN;
        }
        n = chebN;
        Dkn[k * (chebN + 1) + n] = cos(k * n * Pi / chebN) / chebN;
    }
    double *dvec = new double[chebN + 1];
    for (int i = 0; i < chebN + 1; i++) {
        dvec[i] = i % 2 == 1 ? 0 : 2 / (1.0 - static_cast<double>(i * i));
    }
    dvec[0] = 1;
    points.resize(chebN + 1);
    weights.resize(chebN + 1);
    for (int i = 0; i < chebN + 1; i++) {
        double temp = 0;
        for (int j = 0; j < chebN + 1; j++) {
            temp += Dkn[j * (chebN + 1) + i] * dvec[j]; // not optimal layout for speed.
        }
        weights[i] = temp;
        points[i] = -cos(i * Pi / chebN);
    }

    delete[] Dkn;
    delete[] dvec;
}
