#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

#include <Eigen/Dense>

using Evec = Eigen::VectorXd;
using Emat = Eigen::MatrixXd;
using EVec3 = Eigen::Vector3d;

// Assume A=(m,n), m>n
// U = (m,n), S = (n,n), VT = (n,n)
void testSVD(const Emat &U, const Evec &Sdiag, const Emat &VT, const Emat &A, const Evec &x, const Evec &b) {
    Emat S(U.cols(), VT.rows());
    S = Sdiag.asDiagonal();

    // step 1, test if USVT==A
    Emat Arecon = U * (S * VT);
    Emat Aerror = Arecon - A;
    printf("Aerror max min %g, %g\n", Aerror.maxCoeff(), Aerror.minCoeff());

    const double eps = std::numeric_limits<double>::epsilon();
    // step 2, test backward error
    Evec Sdiaginv = Sdiag;
    for (int i = 0; i < Sdiaginv.size(); i++) {
        Sdiaginv[i] = Sdiaginv[i] < Sdiag[0] * eps ? 0 : 1.0 / Sdiaginv[i];
    }

    Emat V = VT.transpose();
    for (int i = 0; i < Sdiaginv.size(); i++) {
        V.col(i) *= Sdiaginv[i];
    }

    Evec x2 = V * (U.transpose() * b);
    Evec b2 = A * x2;
    Evec xerror = x2 - x;
    Evec berror = b2 - b;
    printf("xerror max min %g, %g\n", xerror.maxCoeff(), xerror.minCoeff());
    printf("berror max min %g, %g\n", berror.maxCoeff(), berror.minCoeff());
}

/**
 * \brief Returns the coordinates of points on the surface of a cube.
 * \param[in] p Number of points on an edge of the cube is (n+1)
 * \param[in] c Coordinates to the centre of the cube (3D array).
 * \param[in] alpha Scaling factor for the size of the cube.
 * \param[in] depth Depth of the cube in the octree.
 * \return Vector with coordinates of points on the surface of the cube in the
 * format [x0 y0 z0 x1 y1 z1 .... ].
 */

template <class Real_t>
std::vector<Real_t> surface(int p, Real_t *c, Real_t alpha, int depth) {
    size_t n_ = (6 * (p - 1) * (p - 1) + 2); // Total number of points.

    std::vector<Real_t> coord(n_ * 3);
    coord[0] = coord[1] = coord[2] = -1.0;
    size_t cnt = 1;
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = -1.0;
            coord[cnt * 3 + 1] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = (2.0 * j - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * i - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = -1.0;
            coord[cnt * 3 + 2] = (2.0 * (j + 1) - p + 1) / (p - 1);
            cnt++;
        }
    for (int i = 0; i < p - 1; i++)
        for (int j = 0; j < p - 1; j++) {
            coord[cnt * 3] = (2.0 * (i + 1) - p + 1) / (p - 1);
            coord[cnt * 3 + 1] = (2.0 * j - p + 1) / (p - 1);
            coord[cnt * 3 + 2] = -1.0;
            cnt++;
        }
    for (size_t i = 0; i < (n_ / 2) * 3; i++)
        coord[cnt * 3 + i] = -coord[i];

    Real_t r = 0.5 * pow(0.5, depth);
    Real_t b = alpha * r;
    for (size_t i = 0; i < n_; i++) {
        coord[i * 3 + 0] = (coord[i * 3 + 0] + 1.0) * b + c[0];
        coord[i * 3 + 1] = (coord[i * 3 + 1] + 1.0) * b + c[1];
        coord[i * 3 + 2] = (coord[i * 3 + 2] + 1.0) * b + c[2];
    }
    return coord;
}

inline double pot(const EVec3 &target, const EVec3 &source) {
    EVec3 rst = target - source;
    double rnorm = rst.norm();
    return rnorm < 1e-12 ? 0 : 1 / rnorm;
}

int main(int argc, char **argv) {
    Eigen::initParallel();

    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = 2 * atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {-(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {-(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};

    auto pointMEquiv = surface(pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv, 0);
    // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(pCheck, (double *)&(pCenterCheck[0]), scaleCheck, 0);
    // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    // Aup for solving MEquiv
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointMCheck.size() / 3;
    Eigen::MatrixXd Aup(checkN, equivN);
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
            Aup(k, l) = pot(Cpoint, Lpoint);
        }
    }

    Evec x(Aup.cols());
    x.setRandom();
    Evec b = Aup * x;

    // jacobi svd
    using std::cout;
    using std::endl;

    {
        cout << "JacobiSVD" << endl;
        Eigen::JacobiSVD<Emat> svd(Aup, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Emat U = svd.matrixU();
        Emat VT = svd.matrixV().transpose();
        Evec Svec = svd.singularValues();
        testSVD(U, Svec, VT, Aup, x, b);
    }
    {
        cout << "BDCSVD" << endl;
        Eigen::BDCSVD<Emat> svd(Aup, Eigen::ComputeThinU | Eigen::ComputeThinV);
        Emat U = svd.matrixU();
        Emat VT = svd.matrixV().transpose();
        Evec Svec = svd.singularValues();
        testSVD(U, Svec, VT, Aup, x, b);
    }
    {
        cout << "HouseholderQR" << endl;
        Evec x2 = Aup.colPivHouseholderQr().solve(b);
        Evec b2 = Aup * x2;
        Evec xerror = x2 - x;
        Evec berror = b2 - b;
        printf("xerror max min %g, %g\n", xerror.maxCoeff(), xerror.minCoeff());
        printf("berror max min %g, %g\n", berror.maxCoeff(), berror.minCoeff());
    }

    return 0;
}