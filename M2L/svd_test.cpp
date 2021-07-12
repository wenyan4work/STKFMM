#include "SVD_pvfmm.hpp"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

// Assume A=(m,n), m>n
// U = (m,n), S = (n,n), VT = (n,n)
void testSVD(const EMat &U, const EVec &Sdiag, const EMat &VT, const EMat &A, const EVec &x, const EVec &b) {
    EMat S(U.cols(), VT.rows());
    S = Sdiag.asDiagonal();

    // step 1, test if USVT==A
    EMat Arecon = U * (S * VT);
    EMat Aerror = Arecon - A;
    printf("Aerror max min %g, %g\n", Aerror.maxCoeff(), Aerror.minCoeff());

    // step 2, test backward error
    EVec Sdiaginv = Sdiag;
    for (int i = 0; i < Sdiaginv.size(); i++) {
        Sdiaginv[i] = Sdiaginv[i] < Sdiag[0] * eps ? 0 : 1.0 / Sdiaginv[i];
    }

    EMat V = VT.transpose();
    for (int i = 0; i < Sdiaginv.size(); i++) {
        V.col(i) *= Sdiaginv[i];
    }

    EVec x2 = V * (U.transpose() * b);
    EVec b2 = A * x2;
    EVec xerror = x2 - x;
    EVec berror = b2 - b;
    printf("xerror max min %g, %g\n", xerror.maxCoeff(), xerror.minCoeff());
    printf("berror max min %g, %g\n", berror.maxCoeff(), berror.minCoeff());
}

inline double pot(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < eps ? 0 : 1 / rnorm;
}

int main(int argc, char **argv) {
    Eigen::initParallel();

    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {-(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {-(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};

    auto pointMEquiv = surface(pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv, 0);
    auto pointMCheck = surface(pCheck, (double *)&(pCenterCheck[0]), scaleCheck, 0);

    // Aup for solving MEquiv
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointMCheck.size() / 3;
    EMat Aup(checkN, equivN);
    for (int k = 0; k < checkN; k++) {
        EVec3 Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const EVec3 Lpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            Aup(k, l) = pot(Cpoint, Lpoint);
        }
    }

    EVec x(Aup.cols());
    x.setRandom();
    EVec b = Aup * x;

    // jacobi svd
    using std::cout;
    using std::endl;

    {
        cout << "JacobiSVD" << endl;
        Eigen::JacobiSVD<EMat> svd(Aup, Eigen::ComputeThinU | Eigen::ComputeThinV);
        EMat U = svd.matrixU();
        EMat VT = svd.matrixV().transpose();
        EVec Svec = svd.singularValues();
        testSVD(U, Svec, VT, Aup, x, b);
    }
    // this triggers error #13212make -j: Reference to ebx in function requiring stack alignment
    // {
    //     cout << "BDCSVD" << endl;
    //     Eigen::BDCSVD<EMat> svd(Aup, Eigen::ComputeThinU | Eigen::ComputeThinV);
    //     EMat U = svd.matrixU();
    //     EMat VT = svd.matrixV().transpose();
    //     EVec Svec = svd.singularValues();
    //     testSVD(U, Svec, VT, Aup, x, b);
    // }
    {
        cout << "HouseholderQR" << endl;
        EVec x2 = Aup.colPivHouseholderQr().solve(b);
        EVec b2 = Aup * x2;
        EVec xerror = x2 - x;
        EVec berror = b2 - b;
        printf("xerror max min %g, %g\n", xerror.maxCoeff(), xerror.minCoeff());
        printf("berror max min %g, %g\n", berror.maxCoeff(), berror.minCoeff());
    }

    return 0;
}