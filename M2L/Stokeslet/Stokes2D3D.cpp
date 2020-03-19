/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

#include <Eigen/Dense>

#include <iomanip>
#include <iostream>

#define DIRECTLAYER 2
#define PI314 (3.1415926535897932384626433)

namespace Stokes2D3D {

inline double ERFC(double x) { return std::erfc(x); }
inline double ERF(double x) { return std::erf(x); }

/*
 * def AEW(xi,rvec):
 r=np.sqrt(rvec.dot(rvec))
 A = 2*(xi*np.exp(-(xi**2)*(r**2))/(np.sqrt(np.pi)*r**2)+ss.erfc(xi*r)/(2*r**3))
 *(r*r*np.identity(3)+np.outer(rvec,rvec)) -
 4*xi/np.sqrt(np.pi)*np.exp(-(xi**2)*(r**2))*np.identity(3)
 return A
 *
 * */
inline Eigen::Matrix3d AEW(const double xi, const Eigen::Vector3d &rvec) {
    const double r = rvec.norm();
    Eigen::Matrix3d A =
        2 *
            (xi * exp(-(xi * xi) * (r * r)) / (sqrt(PI314) * r * r) +
             erfc(xi * r) / (2 * r * r * r)) *
            (r * r * Eigen::Matrix3d::Identity() + (rvec * rvec.transpose())) -
        4 * xi / sqrt(PI314) * exp(-(xi * xi) * (r * r)) *
            Eigen::Matrix3d::Identity();
    return A;
}

inline double lbda(double k, double xi, double z) {
    return exp(-k * k / (4 * xi * xi) - (xi * xi) * (z * z));
}

inline double thetaplus(double k, double xi, double z) {
    return exp(k * z) * ERFC(k / (2 * xi) + xi * z);
}

inline double thetaminus(double k, double xi, double z) {
    return exp(-k * z) * ERFC(k / (2 * xi) - xi * z);
}

inline double J00(double k, double xi, double z) {
    return sqrt(PI314) * lbda(k, xi, z) * xi;
}

inline double J10(double k, double xi, double z) {
    return PI314 * (thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (4 * k);
}

inline double J20(double k, double xi, double z) {
    return sqrt(PI314) * lbda(k, xi, z) / (4 * k * k * xi) +
           PI314 *
               ((thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (8 * k * k * k) +
                (thetaminus(k, xi, z) - thetaplus(k, xi, z)) * z / (8 * k * k) -
                (thetaplus(k, xi, z) + thetaminus(k, xi, z)) /
                    (16 * k * (xi * xi)));
}

inline double J12(double k, double xi, double z) {
    return PI314 * (-thetaplus(k, xi, z) - thetaminus(k, xi, z)) * k / 4 +
           sqrt(PI314) * lbda(k, xi, z) * xi;
}

inline double J22(double k, double xi, double z) {
    return PI314 * ((thetaplus(k, xi, z) + thetaminus(k, xi, z)) * k /
                        (16 * xi * xi) +
                    (thetaplus(k, xi, z) + thetaminus(k, xi, z)) / (8 * k) +
                    (thetaplus(k, xi, z) - thetaminus(k, xi, z)) * z / 8) -
           sqrt(PI314) * lbda(k, xi, z) / (4 * xi);
}

inline double K11(double k, double xi, double z) {
    return PI314 * ((thetaminus(k, xi, z) - thetaplus(k, xi, z))) / 4;
}

inline double K12(double k, double xi, double z) {
    return PI314 *
           ((thetaplus(k, xi, z) - thetaminus(k, xi, z)) / (16 * xi * xi) +
            (thetaminus(k, xi, z) + thetaplus(k, xi, z)) * z / (8 * k));
}

inline void QI(const Eigen::Vector3d &kvec, double xi, double z,
               Eigen::Matrix3d &QI) {
    // 3*3 tensor
    // kvec: np.array([k1,k2,0])
    double knorm = sqrt(kvec[0] * kvec[0] + kvec[1] * kvec[1]);
    QI = 2 * (J00(knorm, xi, z) / (4 * xi * xi) + J10(knorm, xi, z)) *
         Eigen::Matrix3d::Identity();
}

inline void Qkk(const Eigen::Vector3d &kvec, double xi, double z,
                Eigen::Matrix3d &Qreal, Eigen::Matrix3d &Qimg) {
    double k1 = kvec[0];
    double k2 = kvec[1];
    double knorm = sqrt(k1 * k1 + k2 * k2);
    auto j10 = J10(knorm, xi, z);
    auto j20 = J20(knorm, xi, z);
    auto j12 = J12(knorm, xi, z);
    auto j22 = J22(knorm, xi, z);

    auto k11 = K11(knorm, xi, z);
    auto k12 = K12(knorm, xi, z);
    Qreal.setZero();
    Qreal(0, 0) = k1 * k1;
    Qreal(1, 1) = k2 * k2;
    Qreal(0, 1) = k1 * k2;
    Qreal(1, 0) = k1 * k2;

    Qreal *= (j10 / (4 * (xi * xi)) + j20);
    Qreal(2, 2) = (j12 / (4 * xi * xi) + j22);
    Qreal *= -2;

    Qimg.setZero();
    Qimg(0, 2) = k1;
    Qimg(1, 2) = k2;
    Qimg(2, 0) = k1;
    Qimg(2, 1) = k2;
    // Qimg=np.array([[0,0,k1],[0,0,k2],[k1,k2,0]])*( k11/(4*xi**2) + k12 )
    Qimg *= (k11 / (4 * xi * xi) + k12);
    Qimg *= -2;
}

// inline Eigen::Matrix3d uFk0(double xi, double zmn) {
//	Eigen::Matrix3d wavek0;
//	wavek0 = -(4.0 / 1) * (PI314 * (zmn) * ERF(zmn * xi) + sqrt(PI314) / (2
//* xi) * exp(-zmn * zmn * xi * xi));
//	return wavek0;
//
//}

inline void GkernelEwald(const Eigen::Vector3d &rvecIn, Eigen::Matrix3d &Gsum) {
    const double xi = 2;
    Eigen::Vector3d rvec = rvecIn;
    rvec[0] = rvec[0] - floor(rvec[0]);
    rvec[1] = rvec[1] - floor(rvec[1]); // reset to a periodic cell

    const double r = rvec.norm();
    Eigen::Matrix3d real = Eigen::Matrix3d::Zero();
    const int N = 5;
    if (r < 1e-14) {
        auto Gself = -4 * xi / sqrt(PI314) *
                     Eigen::Matrix3d::Identity(); // the self term
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                if (i == 0 && j == 0) {
                    continue;
                }
                real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, 0));
            }
        }
        real += Gself;
    } else {
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, 0));
            }
        }
    }

    // k
    Eigen::Matrix3d wave = Eigen::Matrix3d::Zero();

    double zmn = rvec[2];
    Eigen::Vector3d rhomn = rvec;
    rhomn[2] = 0;
    Eigen::Matrix3d Qreal;
    Eigen::Matrix3d Qimg;
    Eigen::Matrix3d QImat;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            Eigen::Vector3d kvec(2 * PI314 * i, 2 * PI314 * j, 0);
            if (i == 0 and j == 0) {
                continue;
            }
            Qkk(kvec, xi, zmn, Qreal, Qimg);
            QI(kvec, xi, zmn, QImat);
            wave = wave + (QImat + Qreal) * cos(kvec.dot(rhomn)) -
                   (Qimg)*sin(kvec.dot(rhomn));
        }
    }
    wave *= 4;

    // k=0
    Eigen::Matrix3d waveK0;
    waveK0.setZero();
    /*
     *   I2fn=force
     I2fn[2]=0
     wavek0=-(4/1)*(np.pi*(zmn)*ss.erf(zmn*xi)+np.sqrt(np.pi)/(2*xi)*np.exp(-zmn**2*xi**2))*I2fn
     *
     * */
    waveK0 = -(4 / 1.0) *
             (PI314 * (zmn)*ERF(zmn * xi) +
              sqrt(PI314) / (2 * xi) * exp(-zmn * zmn * xi * xi)) *
             Eigen::Matrix3d::Identity();
    waveK0(2, 2) = 0;

    Gsum = real + wave + waveK0;
}

inline void Gkernel(const Eigen::Vector3d &target,
                    const Eigen::Vector3d &source, Eigen::Matrix3d &answer) {
    auto rst = target - source;
    double rnorm = rst.norm();
    if (rnorm < 1e-14) {
        answer = Eigen::Matrix3d::Zero();
        return;
    }
    auto part2 = rst * rst.transpose() / (rnorm * rnorm * rnorm);
    auto part1 = Eigen::Matrix3d::Identity() / rnorm;
    answer = part1 + part2;
}

// Out of Layer 1
inline void GkernelEwaldO1(const Eigen::Vector3d &rvec,
                           Eigen::Matrix3d &GsumO1) {
    Eigen::Matrix3d Gfree = Eigen::Matrix3d::Zero();
    GkernelEwald(rvec, GsumO1);
    const int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            Gkernel(rvec, Eigen::Vector3d(i, j, 0), Gfree);
            GsumO1 -= Gfree;
        }
    }
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

int main(int argc, char **argv) {
    Eigen::initParallel();
    Eigen::setNbThreads(1);
    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {
        -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {
        -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};

    const double scaleLEquiv = 1.05;
    const double scaleLCheck = 2.95;
    const double pCenterLEquiv[3] = {
        -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2};
    const double pCenterLCheck[3] = {
        -(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2};

    auto pointMEquiv = surface(
        pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(
        pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    auto pointLEquiv = surface(
        pEquiv, (double *)&(pCenterLCheck[0]), scaleLCheck,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointLCheck = surface(
        pCheck, (double *)&(pCenterLEquiv[0]), scaleLEquiv,
        0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    //	for (int i = 0; i < pointLEquiv.size() / 3; i++) {
    //		std::cout << pointLEquiv[3 * i] << " " << pointLEquiv[3 * i + 1]
    //<< " " << pointLEquiv[3 * i + 2] << " "
    //				<< std::endl;
    //	}
    //
    //	for (int i = 0; i < pointLCheck.size() / 3; i++) {
    //		std::cout << pointLCheck[3 * i] << " " << pointLCheck[3 * i + 1]
    //<< " " << pointLCheck[3 * i + 2] << " "
    //				<< std::endl;
    //	}

    // const int imageN = 100; // images to sum
    // calculate the operator M2L with least square
    const int equivN = pointMEquiv.size() / 3;
    const int checkN = pointLCheck.size() / 3;
    Eigen::MatrixXd M2L(3 * equivN, 3 * equivN);
    Eigen::MatrixXd A(3 * checkN, 3 * equivN);
#pragma omp parallel for
    for (int k = 0; k < checkN; k++) {
        Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                               pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l],
                                         pointLEquiv[3 * l + 1],
                                         pointLEquiv[3 * l + 2]);
            Gkernel(Cpoint, Lpoint, G);
            A.block<3, 3>(3 * k, 3 * l) = G;
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1],
                                     pointMEquiv[3 * i + 2]);
        //		std::cout<<"debug:"<<Mpoint<<std::endl;
        // assemble linear system
        Eigen::MatrixXd f(3 * checkN, 3);
        for (int k = 0; k < checkN; k++) {
            Eigen::Matrix3d temp = Eigen::Matrix3d::Zero();
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1],
                                   pointLCheck[3 * k + 2]);
            //			std::cout<<"debug:"<<k<<std::endl;
            // sum the images
            // use 3D Ewald subtract the first layer
            GkernelEwaldO1(Cpoint - Mpoint, temp);
            f.block<3, 3>(3 * k, 0) = temp;
        }

        M2L.block(0, 3 * i, 3 * equivN, 3) =
            (ApinvU.transpose() * (ApinvVT.transpose() * f));
    }

    // dump M2L
    for (int i = 0; i < 3 * equivN; i++) {
        for (int j = 0; j < 3 * equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific
                      << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    /*
     * pointForce=[(np.array([1.0,0,0]),np.array([0.1,0.55,0.2]))
     ,(np.array([-1.0,1.0,1.0]),np.array([0.5,0.1,0.3]))
     ,(np.array([0.0,0.0,-1.0]),np.array([0.8,0.5,0.7]))]
     * */
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forcePoint(3);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forceValue(3);
    forcePoint[0] = Eigen::Vector3d(0.1, 0.5, 0.5);
    forceValue[0] = Eigen::Vector3d(1, 0, 0);
    forcePoint[1] = Eigen::Vector3d(0.9, 0.5, 0.5);
    forceValue[1] = Eigen::Vector3d(-1, 0, 0);
    forcePoint[2] = Eigen::Vector3d(0.0, 0.0, 0.0);
    forceValue[2] = Eigen::Vector3d(0, 0, 0);

    // solve M
    A.resize(3 * checkN, 3 * equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(3 * checkN);
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d temp = Eigen::Vector3d::Zero();
        Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1],
                               pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < forcePoint.size(); p++) {
            Gkernel(Cpoint, forcePoint[p], G);
            temp = temp + G * (forceValue[p]);
        }
        f.block<3, 1>(3 * k, 0) = temp;
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1],
                                   pointMEquiv[3 * l + 2]);
            Gkernel(Cpoint, Mpoint, G);
            A.block<3, 3>(3 * k, 3 * l) = G;
        }
    }
    pinv(A, ApinvU, ApinvVT);
    Eigen::VectorXd Msource = (ApinvU.transpose() * (ApinvVT.transpose() * f));
    // impose net charge equal
    double fx = 0, fy = 0, fz = 0;
    for (int i = 0; i < equivN; i++) {
        fx += Msource[3 * i];
        fy += Msource[3 * i + 1];
        fz += Msource[3 * i + 2];
    }
    std::cout << "fx svd before correction: " << fx << std::endl;
    std::cout << "fy svd before correction: " << fy << std::endl;
    std::cout << "fz svd before correction: " << fz << std::endl;
    double fnetx = 0;
    double fnety = 0;
    double fnetz = 0;
    for (size_t p = 0; p < forcePoint.size(); p++) {
        fnetx += (forceValue[p][0]);
        fnety += (forceValue[p][1]);
        fnetz += (forceValue[p][2]);
    }

    std::cout << "Msource: " << Msource << std::endl;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forcePointExt(0);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>
        forceValueExt(0);
    for (size_t p = 0; p < forcePoint.size(); p++) {
        for (int i = -DIRECTLAYER; i < DIRECTLAYER + 1; i++) {
            for (int j = -DIRECTLAYER; j < DIRECTLAYER + 1; j++) {
                forcePointExt.push_back(Eigen::Vector3d(i, j, 0) +
                                        forcePoint[p]);
                forceValueExt.push_back(forceValue[p]);
            }
        }
    }

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    Eigen::Vector3d samplePoint(0.4, 0.5, 0.5);
    Eigen::Vector3d Usample(0, 0, 0);
    Eigen::Vector3d UsampleSP(0, 0, 0);
    Eigen::Matrix3d G;
    for (size_t p = 0; p < forcePointExt.size(); p++) {
        Gkernel(samplePoint, forcePointExt[p], G);
        Usample = Usample + G * (forceValueExt[p]);
    }
    std::cout << "Usample Direct:" << Usample << std::endl;
    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1],
                               pointLEquiv[3 * p + 2]);
        Eigen::Vector3d Fpoint(M2Lsource[3 * p], M2Lsource[3 * p + 1],
                               M2Lsource[3 * p + 2]);
        Gkernel(samplePoint, Lpoint, G);
        UsampleSP = UsampleSP + G * (Fpoint);
    }

    std::cout << "Usample M2L:" << UsampleSP << std::endl;
    std::cout << "Usample M2L total:" << UsampleSP + Usample << std::endl;

    Eigen::Vector3d UsampleDirect = 0 * Usample;
    for (size_t p = 0; p < forcePoint.size(); p++) {
        GkernelEwald(samplePoint - forcePoint[p], G);
        UsampleDirect += G * (forceValue[p]);
    }
    std::cout << "Usample Ewald:" << UsampleDirect << std::endl;

    std::cout << "error" << UsampleSP + Usample - UsampleDirect << std::endl;

    return 0;
}

} // namespace Stokes2D3D

#undef DIRECTLAYER
#undef PI314
