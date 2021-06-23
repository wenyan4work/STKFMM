/*
 * SVD_pvfmm.hpp
 *
 *  Created on: Feb 20, 2017
 *      Author: wyan
 */

#ifndef SVD_PVFMM_HPP_
#define SVD_PVFMM_HPP_

#include <cassert>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include <Eigen/Dense>

using EVec2 = Eigen::Vector2d;
using EVec3 = Eigen::Vector3d;
using EVec4 = Eigen::Vector4d;
using EMat2 = Eigen::Matrix2d;
using EMat3 = Eigen::Matrix3d;
using EMat4 = Eigen::Matrix4d;

using EVec = Eigen::VectorXd;
using EMat = Eigen::MatrixXd;

void saveEMat(const EMat &mat, const std::string &fname) {
    FILE *fptr = fopen(fname.c_str(), "r");
    const int M = mat.rows();
    const int N = mat.cols();
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            fprintf(fptr, "%d %d %18.16e\n", i, j, mat(i, j));
        }
    }
    fclose(fptr);
}

template <class T>
inline void gemm(char TransA, char TransB, int M, int N, int K, T alpha, T *A, int lda, T *B, int ldb, T beta, T *C,
                 int ldc) {
    if ((TransA == 'N' || TransA == 'n') && (TransB == 'N' || TransB == 'n')) {
#pragma omp parallel for
        for (auto n = 0; n < N; n++) {     // Columns of C
            for (auto m = 0; m < M; m++) { // Rows of C
                T AxB = 0;
                for (auto k = 0; k < K; k++) {
                    AxB += A[m + lda * k] * B[k + ldb * n];
                }
                C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
            }
        }
    } else if (TransA == 'N' || TransA == 'n') {
#pragma omp parallel for
        for (auto n = 0; n < N; n++) {     // Columns of C
            for (auto m = 0; m < M; m++) { // Rows of C
                T AxB = 0;
                for (auto k = 0; k < K; k++) {
                    AxB += A[m + lda * k] * B[n + ldb * k];
                }
                C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
            }
        }
    } else if (TransB == 'N' || TransB == 'n') {
#pragma omp parallel for
        for (auto n = 0; n < N; n++) {     // Columns of C
            for (auto m = 0; m < M; m++) { // Rows of C
                T AxB = 0;
                for (auto k = 0; k < K; k++) {
                    AxB += A[k + lda * m] * B[k + ldb * n];
                }
                C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
            }
        }
    } else {
#pragma omp parallel for
        for (auto n = 0; n < N; n++) {     // Columns of C
            for (auto m = 0; m < M; m++) { // Rows of C
                T AxB = 0;
                for (auto k = 0; k < K; k++) {
                    AxB += A[k + lda * m] * B[n + ldb * k];
                }
                C[m + ldc * n] = alpha * AxB + (beta == 0 ? 0 : beta * C[m + ldc * n]);
            }
        }
    }
}

#define U(i, j) U_[(i)*dim[0] + (j)]
#define S(i, j) S_[(i)*dim[1] + (j)]
#define V(i, j) V_[(i)*dim[1] + (j)]

template <class T>
void GivensL(T *S_, const size_t dim[2], size_t m, T a, T b) {
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

#pragma omp parallel for
    for (size_t i = 0; i < dim[1]; i++) {
        T S0 = S(m + 0, i);
        T S1 = S(m + 1, i);
        S(m, i) += S0 * (c - 1);
        S(m, i) += S1 * (-s);

        S(m + 1, i) += S0 * (s);
        S(m + 1, i) += S1 * (c - 1);
    }
}

template <class T>
void GivensR(T *S_, const size_t dim[2], size_t m, T a, T b) {
    T r = sqrt(a * a + b * b);
    T c = a / r;
    T s = -b / r;

#pragma omp parallel for
    for (size_t i = 0; i < dim[0]; i++) {
        T S0 = S(i, m + 0);
        T S1 = S(i, m + 1);
        S(i, m) += S0 * (c - 1);
        S(i, m) += S1 * (-s);

        S(i, m + 1) += S0 * (s);
        S(i, m + 1) += S1 * (c - 1);
    }
}

template <class T>
void SVD(const size_t dim[2], T *U_, T *S_, T *V_, T eps = -1) {
    assert(dim[0] >= dim[1]);
#ifdef SVD_DEBUG
    Matrix<T> M0(dim[0], dim[1], S_);
#endif

    { // Bi-diagonalization
        size_t n = std::min(dim[0], dim[1]);
        std::vector<T> house_vec(std::max(dim[0], dim[1]));
        for (size_t i = 0; i < n; i++) {
            // Column Householder
            {
                T x1 = S(i, i);
                if (x1 < 0)
                    x1 = -x1;

                T x_inv_norm = 0;
                for (size_t j = i; j < dim[0]; j++) {
                    x_inv_norm += S(j, i) * S(j, i);
                }
                if (x_inv_norm > 0)
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                T alpha = sqrt(1 + x1 * x_inv_norm);
                T beta = x_inv_norm / alpha;

                house_vec[i] = -alpha;
                for (size_t j = i + 1; j < dim[0]; j++) {
                    house_vec[j] = -beta * S(j, i);
                }
                if (S(i, i) < 0)
                    for (size_t j = i + 1; j < dim[0]; j++) {
                        house_vec[j] = -house_vec[j];
                    }
            }
#pragma omp parallel for
            for (size_t k = i; k < dim[1]; k++) {
                T dot_prod = 0;
                for (size_t j = i; j < dim[0]; j++) {
                    dot_prod += S(j, k) * house_vec[j];
                }
                for (size_t j = i; j < dim[0]; j++) {
                    S(j, k) -= dot_prod * house_vec[j];
                }
            }
#pragma omp parallel for
            for (size_t k = 0; k < dim[0]; k++) {
                T dot_prod = 0;
                for (size_t j = i; j < dim[0]; j++) {
                    dot_prod += U(k, j) * house_vec[j];
                }
                for (size_t j = i; j < dim[0]; j++) {
                    U(k, j) -= dot_prod * house_vec[j];
                }
            }

            // Row Householder
            if (i >= n - 1)
                continue;
            {
                T x1 = S(i, i + 1);
                if (x1 < 0)
                    x1 = -x1;

                T x_inv_norm = 0;
                for (size_t j = i + 1; j < dim[1]; j++) {
                    x_inv_norm += S(i, j) * S(i, j);
                }
                if (x_inv_norm > 0)
                    x_inv_norm = 1 / sqrt(x_inv_norm);

                T alpha = sqrt(1 + x1 * x_inv_norm);
                T beta = x_inv_norm / alpha;

                house_vec[i + 1] = -alpha;
                for (size_t j = i + 2; j < dim[1]; j++) {
                    house_vec[j] = -beta * S(i, j);
                }
                if (S(i, i + 1) < 0)
                    for (size_t j = i + 2; j < dim[1]; j++) {
                        house_vec[j] = -house_vec[j];
                    }
            }
#pragma omp parallel for
            for (size_t k = i; k < dim[0]; k++) {
                T dot_prod = 0;
                for (size_t j = i + 1; j < dim[1]; j++) {
                    dot_prod += S(k, j) * house_vec[j];
                }
                for (size_t j = i + 1; j < dim[1]; j++) {
                    S(k, j) -= dot_prod * house_vec[j];
                }
            }
#pragma omp parallel for
            for (size_t k = 0; k < dim[1]; k++) {
                T dot_prod = 0;
                for (size_t j = i + 1; j < dim[1]; j++) {
                    dot_prod += V(j, k) * house_vec[j];
                }
                for (size_t j = i + 1; j < dim[1]; j++) {
                    V(j, k) -= dot_prod * house_vec[j];
                }
            }
        }
    }

    size_t k0 = 0;
    size_t iter = 0;
    if (eps < 0) {
        eps = 1.0;
        while (eps + (T)1.0 > 1.0)
            eps *= 0.5;
        eps *= 64.0;
    }
    while (k0 < dim[1] - 1) { // Diagonalization
        iter++;

        T S_max = 0.0;
        for (size_t i = 0; i < dim[1]; i++)
            S_max = (S_max > S(i, i) ? S_max : S(i, i));

        while (k0 < dim[1] - 1 && std::abs<T>(S(k0, k0 + 1)) <= eps * S_max)
            k0++;
        if (k0 == dim[1] - 1)
            continue;

        size_t n = k0 + 2;
        while (n < dim[1] && std::abs<T>(S(n - 1, n)) > eps * S_max)
            n++;

        T alpha = 0;
        T beta = 0;
        { // Compute mu
            T C[2][2];
            C[0][0] = S(n - 2, n - 2) * S(n - 2, n - 2);
            if (n - k0 > 2)
                C[0][0] += S(n - 3, n - 2) * S(n - 3, n - 2);
            C[0][1] = S(n - 2, n - 2) * S(n - 2, n - 1);
            C[1][0] = S(n - 2, n - 2) * S(n - 2, n - 1);
            C[1][1] = S(n - 1, n - 1) * S(n - 1, n - 1) + S(n - 2, n - 1) * S(n - 2, n - 1);

            T b = -(C[0][0] + C[1][1]) / 2;
            T c = C[0][0] * C[1][1] - C[0][1] * C[1][0];
            T d = 0;
            if (b * b - c > 0)
                d = sqrt(b * b - c);
            else {
                T b = (C[0][0] - C[1][1]) / 2;
                T c = -C[0][1] * C[1][0];
                if (b * b - c > 0)
                    d = sqrt(b * b - c);
            }

            T lambda1 = -b + d;
            T lambda2 = -b - d;

            T d1 = lambda1 - C[1][1];
            d1 = (d1 < 0 ? -d1 : d1);
            T d2 = lambda2 - C[1][1];
            d2 = (d2 < 0 ? -d2 : d2);
            T mu = (d1 < d2 ? lambda1 : lambda2);

            alpha = S(k0, k0) * S(k0, k0) - mu;
            beta = S(k0, k0) * S(k0, k0 + 1);
        }

        for (size_t k = k0; k < n - 1; k++) {
            size_t dimU[2] = {dim[0], dim[0]};
            size_t dimV[2] = {dim[1], dim[1]};
            GivensR(S_, dim, k, alpha, beta);
            GivensL(V_, dimV, k, alpha, beta);

            alpha = S(k, k);
            beta = S(k + 1, k);
            GivensL(S_, dim, k, alpha, beta);
            GivensR(U_, dimU, k, alpha, beta);

            alpha = S(k, k + 1);
            beta = S(k, k + 2);
        }

        { // Make S bi-diagonal again
            for (size_t i0 = k0; i0 < n - 1; i0++) {
                for (size_t i1 = 0; i1 < dim[1]; i1++) {
                    if (i0 > i1 || i0 + 1 < i1)
                        S(i0, i1) = 0;
                }
            }
            for (size_t i0 = 0; i0 < dim[0]; i0++) {
                for (size_t i1 = k0; i1 < n - 1; i1++) {
                    if (i0 > i1 || i0 + 1 < i1)
                        S(i0, i1) = 0;
                }
            }
            for (size_t i = 0; i < dim[1] - 1; i++) {
                if (std::abs<T>(S(i, i + 1)) <= eps * S_max) {
                    S(i, i + 1) = 0;
                }
            }
        }
        // std::cout<<iter<<' '<<k0<<' '<<n<<'\n';
    }

    { // Check Error
#ifdef SVD_DEBUG
        Matrix<T> U0(dim[0], dim[0], U_);
        Matrix<T> S0(dim[0], dim[1], S_);
        Matrix<T> V0(dim[1], dim[1], V_);
        Matrix<T> E = M0 - U0 * S0 * V0;
        T max_err = 0;
        T max_nondiag0 = 0;
        T max_nondiag1 = 0;
        for (size_t i = 0; i < E.Dim(0); i++)
            for (size_t j = 0; j < E.Dim(1); j++) {
                if (max_err < pvfmm::fabs<T>(E[i][j]))
                    max_err = pvfmm::fabs<T>(E[i][j]);
                if ((i > j + 0 || i + 0 < j) && max_nondiag0 < pvfmm::fabs<T>(S0[i][j]))
                    max_nondiag0 = pvfmm::fabs<T>(S0[i][j]);
                if ((i > j + 1 || i + 1 < j) && max_nondiag1 < pvfmm::fabs<T>(S0[i][j]))
                    max_nondiag1 = pvfmm::fabs<T>(S0[i][j]);
            }
        std::cout << max_err << '\n';
        std::cout << max_nondiag0 << '\n';
        std::cout << max_nondiag1 << '\n';
#endif
    }
}

#undef U
#undef S
#undef V
#undef SVD_DEBUG

template <class T>
inline void svd(char *JOBU, char *JOBVT, int *M, int *N, T *A, int *LDA, T *S, T *U, int *LDU, T *VT, int *LDVT,
                T *WORK, int *LWORK, int *INFO) {

    const size_t dim[2] = {static_cast<size_t>(std::max(*N, *M)), static_cast<size_t>(std::min(*N, *M))};

    std::vector<T> Udata(dim[0] * dim[0], 0);
    std::vector<T> Vdata(dim[1] * dim[1], 0);
    std::vector<T> Sdata(dim[0] * dim[1], 0);
    T *U_ = Udata.data();
    T *V_ = Vdata.data();
    T *S_ = Sdata.data();
    //	T* U_ = mem::aligned_new < T > (dim[0] * dim[0]);
    //	memset(U_, 0, dim[0] * dim[0] * sizeof(T));
    //	T* V_ = mem::aligned_new < T > (dim[1] * dim[1]);
    //	memset(V_, 0, dim[1] * dim[1] * sizeof(T));
    //	T* S_ = mem::aligned_new < T > (dim[0] * dim[1]);

    const size_t lda = *LDA;
    const size_t ldu = *LDU;
    const size_t ldv = *LDVT;

    if (dim[1] == static_cast<size_t>(*M)) {
        for (size_t i = 0; i < dim[0]; i++) {
            for (size_t j = 0; j < dim[1]; j++) {
                S_[i * dim[1] + j] = A[i * lda + j];
            }
        }
    } else {
        for (size_t i = 0; i < dim[0]; i++) {
            for (size_t j = 0; j < dim[1]; j++) {
                S_[i * dim[1] + j] = A[j * lda + i];
            }
        }
    }
    for (size_t i = 0; i < dim[0]; i++) {
        U_[i * dim[0] + i] = 1;
    }
    for (size_t i = 0; i < dim[1]; i++) {
        V_[i * dim[1] + i] = 1;
    }

    SVD<T>(dim, U_, S_, V_, (T)-1);

    for (size_t i = 0; i < dim[1]; i++) { // Set S
        S[i] = S_[i * dim[1] + i];
    }
    if (dim[1] == static_cast<size_t>(*M)) { // Set U
        for (size_t i = 0; i < dim[1]; i++)
            for (int j = 0; j < *M; j++) {
                U[j + ldu * i] = V_[j + i * dim[1]] * (S[i] < 0.0 ? -1.0 : 1.0);
            }
    } else {
        for (size_t i = 0; i < dim[1]; i++)
            for (int j = 0; j < *M; j++) {
                U[j + ldu * i] = U_[i + j * dim[0]] * (S[i] < 0.0 ? -1.0 : 1.0);
            }
    }
    if (dim[0] == static_cast<size_t>(*N)) { // Set V
        for (int i = 0; i < *N; i++)
            for (size_t j = 0; j < dim[1]; j++) {
                VT[j + ldv * i] = U_[j + i * dim[0]];
            }
    } else {
        for (int i = 0; i < *N; i++)
            for (size_t j = 0; j < dim[1]; j++) {
                VT[j + ldv * i] = V_[i + j * dim[1]];
            }
    }
    for (size_t i = 0; i < dim[1]; i++) {
        S[i] = S[i] * (S[i] < 0.0 ? -1.0 : 1.0);
    }

    //	mem::aligned_delete < T > (U_);
    //	mem::aligned_delete < T > (S_);
    //	mem::aligned_delete < T > (V_);

    if (0) { // Verify
        const size_t dim[2] = {static_cast<size_t>(std::max(*N, *M)), static_cast<size_t>(std::min(*N, *M))};
        const size_t lda = *LDA;
        const size_t ldu = *LDU;
        const size_t ldv = *LDVT;

        Eigen::MatrixXd A1(*M, *N);
        Eigen::MatrixXd S1(dim[1], dim[1]);
        Eigen::MatrixXd U1(*M, dim[1]);
        Eigen::MatrixXd V1(dim[1], *N);
        for (size_t i = 0; i < *N; i++)
            for (size_t j = 0; j < *M; j++) {
                //				A1[j][i] = A[j + i * lda];
                A1(j, i) = A[j + i * lda];
            }
        S1.setZero();
        for (size_t i = 0; i < dim[1]; i++) { // Set S
                                              //			S1[i][i] = S[i];
            S1(i, i) = S[i];
        }
        for (size_t i = 0; i < dim[1]; i++)
            for (size_t j = 0; j < *M; j++) {
                //				U1[j][i] = U[j + ldu * i];
                U1(j, i) = U[j + ldu * i];
            }
        for (size_t i = 0; i < *N; i++)
            for (size_t j = 0; j < dim[1]; j++) {
                //				V1[j][i] = VT[j + ldv * i];
                V1(j, i) = VT[j + ldv * i];
            }
        std::cout << U1 * S1 * V1 - A1 << '\n';
    }
}

template <class T>
void pinv_pvfmm(T *M, int n1, int n2, T eps, T *M_) {
    if (n1 * n2 == 0)
        return;
    int m = n2;
    int n = n1;
    int k = (m < n ? m : n);

    std::vector<T> Udata(m * k);
    std::vector<T> Sdata(k);
    std::vector<T> VTdata(k * n);

    //	T* tU = mem::aligned_new < T > (m * k);
    //	T* tS = mem::aligned_new < T > (k);
    //	T* tVT = mem::aligned_new < T > (k * n);

    T *tU = Udata.data();
    T *tS = Sdata.data();
    T *tVT = VTdata.data();

    // SVD
    int INFO = 0;
    char JOBU = 'S';
    char JOBVT = 'S';

    // int wssize = max(3*min(m,n)+max(m,n), 5*min(m,n));
    int wssize = 3 * (m < n ? m : n) + (m > n ? m : n);
    int wssize1 = 5 * (m < n ? m : n);
    wssize = (wssize > wssize1 ? wssize : wssize1);

    //	T* wsbuf = mem::aligned_new < T > (wssize);
    std::vector<T> wsbufdata(wssize);
    T *wsbuf = wsbufdata.data();

    svd(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k, wsbuf, &wssize, &INFO);
    if (INFO != 0)
        std::cout << INFO << '\n';
    assert(INFO == 0);
    //	mem::aligned_delete < T > (wsbuf);

    T eps_ = tS[0] * eps;
    for (int i = 0; i < k; i++)
        if (tS[i] < eps_) {
            tS[i] = 0;
        } else {
            tS[i] = 1.0 / tS[i];
            //			std::cout << tS[i] << std::endl;
        }
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            tU[i + j * m] *= tS[j];
        }
    }

    gemm<T>('T', 'T', n, m, k, 1.0, &tVT[0], k, &tU[0], m, 0.0, M_, n);
    //	mem::aligned_delete < T > (tU);
    //	mem::aligned_delete < T > (tS);
    //	mem::aligned_delete < T > (tVT);
}

inline void pinv(const Eigen::MatrixXd &Mat, Eigen::MatrixXd &MatPinv) {
    double eps = 1;
    while (eps + 1.0 > 1.0) {
        eps *= 0.5;
    }
    eps = sqrt(eps);

    std::vector<double> M(Mat.cols() * Mat.rows());
    std::vector<double> Mpinv(M.size());

    // row major
    for (int i = 0; i < Mat.rows(); i++) {
        for (int j = 0; j < Mat.cols(); j++) {
            M[i * Mat.cols() + j] = Mat(i, j);
        }
    }

    pinv_pvfmm<double>(M.data(), Mat.rows(), Mat.cols(), eps, Mpinv.data());

    //#define U(i,j) U_[(i)*dim[0]+(j)]
    // row major
    for (int i = 0; i < Mat.cols(); i++) {
        for (int j = 0; j < Mat.rows(); j++) {
            MatPinv(i, j) = Mpinv[i * Mat.rows() + j];
        }
    }
}

inline void pinv(const Eigen::MatrixXd &Mat, Eigen::MatrixXd &MatPinvU, Eigen::MatrixXd &MatPinvVT) {
    // this is the really backward stable SVD used by pvfmm

    double eps = 1;
    while (eps * (double)0.5 + (double)1.0 > 1.0) {
        eps *= 0.5;
    }

    std::vector<double> M(Mat.cols() * Mat.rows());
    std::vector<double> Mpinv(M.size());

    // row major
    for (int i = 0; i < Mat.rows(); i++) {
        for (int j = 0; j < Mat.cols(); j++) {
            M[i * Mat.cols() + j] = Mat(i, j);
        }
    }

    const int n1 = Mat.rows();
    const int n2 = Mat.cols();

    if (n1 * n2 == 0)
        return;
    int m = n2;
    int n = n1;
    int k = (m < n ? m : n);

    std::vector<double> Udata(m * k);
    std::vector<double> Sdata(k);
    std::vector<double> VTdata(k * n);

    //	T* tU = mem::aligned_new < T > (m * k);
    //	T* tS = mem::aligned_new < T > (k);
    //	T* tVT = mem::aligned_new < T > (k * n);

    double *tU = Udata.data();
    double *tS = Sdata.data();
    double *tVT = VTdata.data();

    // SVD
    int INFO = 0;
    char JOBU = 'S';
    char JOBVT = 'S';

    // int wssize = max(3*min(m,n)+max(m,n), 5*min(m,n));
    int wssize = 3 * (m < n ? m : n) + (m > n ? m : n);
    int wssize1 = 5 * (m < n ? m : n);
    wssize = (wssize > wssize1 ? wssize : wssize1);

    //	T* wsbuf = mem::aligned_new < T > (wssize);
    std::vector<double> wsbufdata(wssize);
    double *wsbuf = wsbufdata.data();

    svd(&JOBU, &JOBVT, &m, &n, &M[0], &m, &tS[0], &tU[0], &m, &tVT[0], &k, wsbuf, &wssize, &INFO);
    if (INFO != 0) {
        std::cout << INFO << '\n';
    }
    assert(INFO == 0);
    //	mem::aligned_delete < T > (wsbuf);

    double eps_ = tS[0] * eps;

    for (int i = 0; i < k; i++) {
        tS[i] = (tS[i] > eps_ * 4 ? 1.0 / tS[i] : 0.0);
    }

    //	for (int i = 0; i < k; i++)
    //		if (tS[i] < eps_) {
    //			tS[i] = 0;
    //		} else {
    //			tS[i] = 1.0 / tS[i];
    ////			std::cout << tS[i] << std::endl;
    //		}
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            tU[i + j * m] *= tS[j];
        }
    }

    MatPinvU.resize(m, k);
    MatPinvVT.resize(n, k);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < k; j++) {
            MatPinvU(i, j) = tU[i * k + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            MatPinvVT(i, j) = tVT[i * k + j];
        }
    }

    //#define U(i,j) U_[(i)*dim[0]+(j)]
    // row major
}

#endif /* M2LLAPLACE_1D3D_SVD_PVFMM_HPP_ */
