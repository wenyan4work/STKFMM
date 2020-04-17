/*
 * StokesM2L.cpp
 *
 *  Created on: Oct 12, 2016
 *      Author: wyan
 */

#include "SVD_pvfmm.hpp"

#include <Eigen/Dense>

#include <chrono>
#include <iomanip>
#include <iostream>

#define DIRECTLAYER 2
#define PI314 3.1415926535897932384626433

namespace Stokes3D3D {

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
    Eigen::Matrix3d A = 2 * (xi * exp(-(xi * xi) * (r * r)) / (sqrt(PI314) * r * r) + erfc(xi * r) / (2 * r * r * r)) *
                            (r * r * Eigen::Matrix3d::Identity() + (rvec * rvec.transpose())) -
                        4 * xi / sqrt(PI314) * exp(-(xi * xi) * (r * r)) * Eigen::Matrix3d::Identity();
    return A;
}

/*
 *
 def BEW(xi,kvec):
 k=np.sqrt(kvec.dot(kvec))
 B =
 8*np.pi*(1+k*k/(4*(xi**2)))*((k**2)*np.identity(3)-np.outer(kvec,kvec))/(k**4)
 return B*np.exp(-k**2/(4*xi**2))
 *
 * */
inline Eigen::Matrix3d BEW(const double xi, const Eigen::Vector3d &kvec) {
    const double k = kvec.norm();
    Eigen::Matrix3d B = 8 * PI314 * (1 + k * k / (4 * (xi * xi))) *
                        ((k * k) * Eigen::Matrix3d::Identity() - (kvec * kvec.transpose())) / (k * k * k * k);
    B *= exp(-k * k / (4 * xi * xi));
    return B;
}

/*
 * def stokes3DEwald(rvec,force):
 xi = 2
 r=np.sqrt(rvec.dot(rvec))
 real = 0
 N=4
 for i in range(-N,N+1):
 for j in range(-N,N+1):
 for k in range(-N,N+1):
 real = real + AEW(xi,rvec+1.0*np.array([i,j,k])).dot(force)
 wave = 0
 N=4
 for i in range(-N,N+1):
 for j in range(-N,N+1):
 for k in range(-N,N+1):
 kvec=2*np.pi*np.array([i,j,k]) # L = 1
 if(i==0 and j==0 and k==0):
 continue
 else:
 wave = wave + BEW(xi,kvec).dot(force)*np.exp(-complex(0,1)*kvec.dot(rvec))

 return (np.real(wave)+real)

 * */
inline void GkernelEwald(const Eigen::Vector3d &rvec, Eigen::Matrix3d &Gsum) {
    const double xi = 2;
    const double r = rvec.norm();
    Eigen::Matrix3d real = Eigen::Matrix3d::Zero();
    const int N = 5;
    if (r < 1e-14) {
        auto Gself = -4 * xi / sqrt(PI314) * Eigen::Matrix3d::Identity(); // the self term
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                for (int k = -N; k < N + 1; k++) {
                    if (i == 0 && j == 0 && k == 0) {
                        continue;
                    }
                    real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, k));
                }
            }
        }
        real += Gself;
    } else {
        for (int i = -N; i < N + 1; i++) {
            for (int j = -N; j < N + 1; j++) {
                for (int k = -N; k < N + 1; k++) {
                    real = real + AEW(xi, rvec + Eigen::Vector3d(i, j, k));
                }
            }
        }
    }
    Eigen::Matrix3d wave = Eigen::Matrix3d::Zero();

    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                Eigen::Vector3d kvec(2 * PI314 * i, 2 * PI314 * j, 2 * PI314 * k);
                if (i == 0 and j == 0 and k == 0) {
                    continue;
                } else {
                    wave = wave + BEW(xi, kvec) * cos(kvec.dot(rvec));
                }
            }
        }
    }
    Gsum = real + wave;
}

inline void Gkernel(const Eigen::Vector3d &target, const Eigen::Vector3d &source, Eigen::Matrix3d &answer) {
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

/*
 *
 def stokes3DM2L(rvec,force):
 uEwald=stokes3DEwald(rvec,force)
 uNB=0
 N=3
 for i in range(-N,N+1):
 for j in range(-N,N+1):
 for k in range(-N,N+1):
 uNB=uNB+Gkernel(rvec-np.array([i,j,k])).dot(force)
 return uEwald-uNB
 * */
// Out of Layer 1
inline void GkernelEwaldO1(const Eigen::Vector3d &rvec, Eigen::Matrix3d &GsumO1) {
    Eigen::Matrix3d Gfree = Eigen::Matrix3d::Zero();
    GkernelEwald(rvec, GsumO1);
    const int N = DIRECTLAYER;
    for (int i = -N; i < N + 1; i++) {
        for (int j = -N; j < N + 1; j++) {
            for (int k = -N; k < N + 1; k++) {
                Gkernel(rvec, Eigen::Vector3d(i, j, k), Gfree);
                GsumO1 -= Gfree;
            }
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
    std::chrono::high_resolution_clock::time_point t1 = std::chrono::high_resolution_clock::now();

    const int pEquiv = atoi(argv[1]); // (8-1)^2*6 + 2 points
    const int pCheck = atoi(argv[1]);
    const double scaleEquiv = 1.05;
    const double scaleCheck = 2.95;
    const double pCenterEquiv[3] = {-(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2, -(scaleEquiv - 1) / 2};
    const double pCenterCheck[3] = {-(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2, -(scaleCheck - 1) / 2};

    const double scaleLEquiv = 1.05;
    const double scaleLCheck = 2.95;
    const double pCenterLEquiv[3] = {-(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2, -(scaleLEquiv - 1) / 2};
    const double pCenterLCheck[3] = {-(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2, -(scaleLCheck - 1) / 2};

    auto pointMEquiv = surface(pEquiv, (double *)&(pCenterEquiv[0]), scaleEquiv,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointMCheck = surface(pCheck, (double *)&(pCenterCheck[0]), scaleCheck,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0

    auto pointLEquiv = surface(pEquiv, (double *)&(pCenterLCheck[0]), scaleLCheck,
                               0); // center at 0.5,0.5,0.5, periodic box 1,1,1, scale 1.05, depth = 0
    auto pointLCheck = surface(pCheck, (double *)&(pCenterLEquiv[0]), scaleLEquiv,
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
        Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
        for (int l = 0; l < equivN; l++) {
            const Eigen::Vector3d Lpoint(pointLEquiv[3 * l], pointLEquiv[3 * l + 1], pointLEquiv[3 * l + 2]);
            Gkernel(Cpoint, Lpoint, G);
            A.block<3, 3>(3 * k, 3 * l) = G;
        }
    }
    Eigen::MatrixXd ApinvU(A.cols(), A.rows());
    Eigen::MatrixXd ApinvVT(A.cols(), A.rows());
    pinv(A, ApinvU, ApinvVT);

#pragma omp parallel for
    for (int i = 0; i < equivN; i++) {
        const Eigen::Vector3d Mpoint(pointMEquiv[3 * i], pointMEquiv[3 * i + 1], pointMEquiv[3 * i + 2]);
        //		std::cout<<"debug:"<<Mpoint<<std::endl;
        // assemble linear system
        Eigen::MatrixXd f(3 * checkN, 3);
        for (int k = 0; k < checkN; k++) {
            Eigen::Matrix3d temp = Eigen::Matrix3d::Zero();
            Eigen::Vector3d Cpoint(pointLCheck[3 * k], pointLCheck[3 * k + 1], pointLCheck[3 * k + 2]);
            //			std::cout<<"debug:"<<k<<std::endl;
            // sum the images
            // use 3D Ewald subtract the first layer
            GkernelEwaldO1(Cpoint - Mpoint, temp);
            f.block<3, 3>(3 * k, 0) = temp;
        }

        M2L.block(0, 3 * i, 3 * equivN, 3) = (ApinvU.transpose() * (ApinvVT.transpose() * f));
    }
    std::chrono::high_resolution_clock::time_point t2 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count();

    // dump M2L
    for (int i = 0; i < 3 * equivN; i++) {
        for (int j = 0; j < 3 * equivN; j++) {
            std::cout << i << " " << j << " " << std::scientific << std::setprecision(18) << M2L(i, j) << std::endl;
        }
    }

    std::cout << "Precomputing time:" << duration / 1e6 << std::endl;

    /*
     * pointForce=[(np.array([1.0,0,0]),np.array([0.1,0.55,0.2]))
     ,(np.array([-1.0,1.0,1.0]),np.array([0.5,0.1,0.3]))
     ,(np.array([0.0,0.0,-1.0]),np.array([0.8,0.5,0.7]))]
     * */
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> forcePoint(3);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> forceValue(3);
    forcePoint[0] = Eigen::Vector3d(0.1, 0.55, 0.2);
    forceValue[0] = Eigen::Vector3d(1, 0, 0);
    forcePoint[1] = Eigen::Vector3d(0.5, 0.1, 0.3);
    forceValue[1] = Eigen::Vector3d(-1, 1, 1);
    forcePoint[2] = Eigen::Vector3d(0.8, 0.5, 0.7);
    forceValue[2] = Eigen::Vector3d(0, 0, -1);

    // solve M
    A.resize(3 * checkN, 3 * equivN);
    ApinvU.resize(A.cols(), A.rows());
    ApinvVT.resize(A.cols(), A.rows());
    Eigen::VectorXd f(3 * checkN);
    for (int k = 0; k < checkN; k++) {
        Eigen::Vector3d temp = Eigen::Vector3d::Zero();
        Eigen::Matrix3d G = Eigen::Matrix3d::Zero();
        Eigen::Vector3d Cpoint(pointMCheck[3 * k], pointMCheck[3 * k + 1], pointMCheck[3 * k + 2]);
        for (size_t p = 0; p < forcePoint.size(); p++) {
            Gkernel(Cpoint, forcePoint[p], G);
            temp = temp + G * (forceValue[p]);
        }
        f.block<3, 1>(3 * k, 0) = temp;
        for (int l = 0; l < equivN; l++) {
            Eigen::Vector3d Mpoint(pointMEquiv[3 * l], pointMEquiv[3 * l + 1], pointMEquiv[3 * l + 2]);
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
    /*
     * fx=(fx-fnet[0])/len(MPoints)
     fy=(fy-fnet[1])/len(MPoints)
     fz=(fz-fnet[2])/len(MPoints)
     * */
    fx = (fx - fnetx) / equivN;
    fy = (fy - fnety) / equivN;
    fz = (fz - fnetz) / equivN;
    for (int i = 0; i < equivN; i++) {
        Msource[3 * i] -= fx;
        Msource[3 * i + 1] -= fy;
        Msource[3 * i + 2] -= fz;
    }
    std::cout << "Msource: " << Msource << std::endl;

    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> forcePointExt(0);
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> forceValueExt(0);
    for (size_t p = 0; p < forcePoint.size(); p++) {
        for (int i = -DIRECTLAYER; i < DIRECTLAYER + 1; i++) {
            for (int j = -DIRECTLAYER; j < DIRECTLAYER + 1; j++) {
                for (int k = -DIRECTLAYER; k < DIRECTLAYER + 1; k++) {
                    forcePointExt.push_back(Eigen::Vector3d(i, j, k) + forcePoint[p]);
                    forceValueExt.push_back(forceValue[p]);
                }
            }
        }
    }

    Eigen::VectorXd M2Lsource = M2L * (Msource);

    Eigen::Vector3d samplePoint(0.5, 0.5, 0.5);
    Eigen::Vector3d Usample(0, 0, 0);
    Eigen::Vector3d UsampleSP(0, 0, 0);
    Eigen::Matrix3d G;
    for (size_t p = 0; p < forcePointExt.size(); p++) {
        Gkernel(samplePoint, forcePointExt[p], G);
        Usample = Usample + G * (forceValueExt[p]);
    }
    std::cout << "Usample Direct:" << Usample << std::endl;
    for (int p = 0; p < equivN; p++) {
        Eigen::Vector3d Lpoint(pointLEquiv[3 * p], pointLEquiv[3 * p + 1], pointLEquiv[3 * p + 2]);
        Eigen::Vector3d Fpoint(M2Lsource[3 * p], M2Lsource[3 * p + 1], M2Lsource[3 * p + 2]);
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

} // namespace Stokes3D3D

#undef PI314
#undef DIRECTLAYER
