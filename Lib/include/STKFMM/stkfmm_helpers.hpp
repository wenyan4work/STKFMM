#ifndef STKFMM_HELPERS_HPP
#define STKFMM_HELPERS_HPP

#include <pvfmm.hpp>

// clang-format off
// do not format macro

namespace pvfmm {
constexpr int SRC_BLK = 500;
} // namespace pvfmm

namespace stkfmm {

/**
 * @brief delete the pointer ptr if not null
 *
 * @tparam T
 * @param ptr
 */
template <class T>
void safeDeletePtr(T *&ptr) {
    if (ptr != nullptr) {
        delete ptr;
        ptr = nullptr;
    }
}

/**
 * @brief set x to its fractional part
 *
 * @param x
 */
inline void fracwrap(double &x) { x = x - floor(x); }

/**
 * @brief generate equivalent point coordinate
 *
 * @tparam Real_t
 * @param p
 * @param c
 * @param alpha
 * @param depth
 * @return std::vector<Real_t>
 */
template <class Real_t>
std::vector<Real_t> surface(int p, Real_t *c, Real_t alpha, int depth) {
    int n_ = (6 * (p - 1) * (p - 1) + 2); // Total number of points.

    std::vector<Real_t> coord(n_ * 3);
    coord[0] = coord[1] = coord[2] = -1.0;
    int cnt = 1;
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
    for (int i = 0; i < (n_ / 2) * 3; i++)
        coord[cnt * 3 + i] = -coord[i];

    Real_t r = 0.5 * pow(0.5, depth);
    Real_t b = alpha * r;
    for (int i = 0; i < n_; i++) {
        coord[i * 3 + 0] = (coord[i * 3 + 0] + 1.0) * b + c[0];
        coord[i * 3 + 1] = (coord[i * 3 + 1] + 1.0) * b + c[1];
        coord[i * 3 + 2] = (coord[i * 3 + 2] + 1.0) * b + c[2];
    }
    return coord;
}

} // namespace stkfmm

#endif
