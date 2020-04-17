#include "SimpleKernel.hpp"
#include <cmath>
// auto generate with mathematica

constexpr double Pi = 3.141592653589793238462643383279;

inline double Abs(double x) { return std::abs(x); }

inline double Sqrt(double x) { return std::sqrt(x); }

inline double Power(double x, double y) { return std::pow(x, y); }

//                         3           3           4             4/16/9/7
void StokesSLPVel(double *s, double *t, double *f, double *pvel) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    if (sx == tx && sy == ty && sz == tz) {
        pvel[0] = pvel[1] = pvel[2] = pvel[3] = 0;
        return;
    }

    const double fx = f[0];
    const double fy = f[1];
    const double fz = f[2];
    const double TrD = f[3];

    double &p = pvel[0];
    double &vx = pvel[1];
    double &vy = pvel[2];
    double &vz = pvel[3];

    p = (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz)) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vx = ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) +
          fx * (2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) - 4 * sx * tx + 2 * Power(tx, 2) - 2 * sy * ty +
                Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vy = ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) +
          fy * (Power(sx, 2) + 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 4 * sy * ty +
                2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vz = ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) +
          fz * (Power(sx, 2) + Power(sy, 2) + 2 * Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 2 * sy * ty +
                Power(ty, 2) - 4 * sz * tz + 2 * Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
}

void StokesSLPVelGrad(double *s, double *t, double *f, double *pvelGrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double fx = f[0];
    const double fy = f[1];
    const double fz = f[2];
    const double TrD = f[3];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 16; i++) {
            pvelGrad[i] = 0;
        }
        return;
    }

    double &p = pvelGrad[0];
    double &vx = pvelGrad[1];
    double &vy = pvelGrad[2];
    double &vz = pvelGrad[3];
    double &pgx = pvelGrad[4];
    double &pgy = pvelGrad[5];
    double &pgz = pvelGrad[6];
    double &vxgx = pvelGrad[7];
    double &vxgy = pvelGrad[8];
    double &vxgz = pvelGrad[9];
    double &vygx = pvelGrad[10];
    double &vygy = pvelGrad[11];
    double &vygz = pvelGrad[12];
    double &vzgx = pvelGrad[13];
    double &vzgy = pvelGrad[14];
    double &vzgz = pvelGrad[15];

    p = (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz)) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vx = ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) +
          fx * (2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) - 4 * sx * tx + 2 * Power(tx, 2) - 2 * sy * ty +
                Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vy = ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) +
          fy * (Power(sx, 2) + 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 4 * sy * ty +
                2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vz = ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) +
          fz * (Power(sx, 2) + Power(sy, 2) + 2 * Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 2 * sy * ty +
                Power(ty, 2) - 4 * sz * tz + 2 * Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    pgx = fx / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
          (3 * (sx - tx) * (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz))) /
              (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    pgy = fy / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
          (3 * (sy - ty) * (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz))) /
              (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    pgz = fz / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
          (3 * (sz - tz) * (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz))) /
              (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    // grad vx
    vxgx = (-(fz * sz) - TrD + fx * (-4 * sx + 4 * tx) - fy * (sy - ty) + fz * tz) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sx - tx) *
            ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) +
             fx * (2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) - 4 * sx * tx + 2 * Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) - 2 * sz * tz + Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vxgy = (-(fy * (sx - tx)) + fx * (-2 * sy + 2 * ty)) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sy - ty) *
            ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) +
             fx * (2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) - 4 * sx * tx + 2 * Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) - 2 * sz * tz + Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vxgz = (-(fz * (sx - tx)) + fx * (-2 * sz + 2 * tz)) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sz - tz) *
            ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) +
             fx * (2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) - 4 * sx * tx + 2 * Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) - 2 * sz * tz + Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    // grad vy
    vygx = (fy * (-2 * sx + 2 * tx) - fx * (sy - ty)) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sx - tx) *
            ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) +
             fy * (Power(sx, 2) + 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 4 * sy * ty +
                   2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vygy = (-(fz * sz) - TrD - fx * (sx - tx) + fy * (-4 * sy + 4 * ty) + fz * tz) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sy - ty) *
            ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) +
             fy * (Power(sx, 2) + 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 4 * sy * ty +
                   2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vygz = (-(fz * (sy - ty)) + fy * (-2 * sz + 2 * tz)) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sz - tz) *
            ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) +
             fy * (Power(sx, 2) + 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 4 * sy * ty +
                   2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));

    // grad vz
    vzgx = (fz * (-2 * sx + 2 * tx) - fx * (sz - tz)) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sx - tx) *
            ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) +
             fz * (Power(sx, 2) + Power(sy, 2) + 2 * Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) - 4 * sz * tz + 2 * Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vzgy = (fz * (-2 * sy + 2 * ty) - fy * (sz - tz)) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sy - ty) *
            ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) +
             fz * (Power(sx, 2) + Power(sy, 2) + 2 * Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) - 4 * sz * tz + 2 * Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vzgz = (-(fy * sy) - TrD - fx * (sx - tx) + fy * ty + fz * (-4 * sz + 4 * tz)) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
           (3 * (sz - tz) *
            ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) +
             fz * (Power(sx, 2) + Power(sy, 2) + 2 * Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) - 4 * sz * tz + 2 * Power(tz, 2)))) /
               (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
}

void StokesSLTraction(double *s, double *t, double *f, double *traction) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double fx = f[0];
    const double fy = f[1];
    const double fz = f[2];
    const double TrD = f[3];

    double &txx = traction[0];
    double &txy = traction[1];
    double &txz = traction[2];
    double &tyx = traction[3];
    double &tyy = traction[4];
    double &tyz = traction[5];
    double &tzx = traction[6];
    double &tzy = traction[7];
    double &tzz = traction[8];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 9; i++) {
            traction[i] = 0;
        }
        return;
    }

    txx = (3 * fz * Power(sx, 2) * sz + 2 * Power(sx, 2) * TrD - Power(sy, 2) * TrD - Power(sz, 2) * TrD +
           3 * fx * Power(sx - tx, 3) - 6 * fz * sx * sz * tx - 4 * sx * TrD * tx + 3 * fz * sz * Power(tx, 2) +
           2 * TrD * Power(tx, 2) + 3 * fy * Power(sx - tx, 2) * (sy - ty) + 2 * sy * TrD * ty - TrD * Power(ty, 2) -
           3 * fz * Power(sx, 2) * tz + 2 * sz * TrD * tz + 6 * fz * sx * tx * tz - 3 * fz * Power(tx, 2) * tz -
           TrD * Power(tz, 2)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    txy = (3 * (sx - tx) * (sy - ty) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    txz = (3 * (sx - tx) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    tyx = (3 * (sx - tx) * (sy - ty) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    tyy = (3 * fz * Power(sy, 2) * sz - Power(sx, 2) * TrD + 2 * Power(sy, 2) * TrD - Power(sz, 2) * TrD +
           2 * sx * TrD * tx - TrD * Power(tx, 2) + 3 * fx * (sx - tx) * Power(sy - ty, 2) +
           3 * fy * Power(sy - ty, 3) - 6 * fz * sy * sz * ty - 4 * sy * TrD * ty + 3 * fz * sz * Power(ty, 2) +
           2 * TrD * Power(ty, 2) - 3 * fz * Power(sy, 2) * tz + 2 * sz * TrD * tz + 6 * fz * sy * ty * tz -
           3 * fz * Power(ty, 2) * tz - TrD * Power(tz, 2)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    tyz = (3 * (sy - ty) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    tzx = (3 * (sx - tx) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    tzy = (3 * (sy - ty) * (sz - tz) * (fz * sz + TrD + fx * (sx - tx) + fy * (sy - ty) - fz * tz)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    tzz = (3 * fz * Power(sz, 3) - (Power(sx, 2) + Power(sy, 2)) * TrD + 2 * Power(sz, 2) * TrD + 2 * sx * TrD * tx -
           TrD * Power(tx, 2) + 2 * sy * TrD * ty - TrD * Power(ty, 2) + 3 * fx * (sx - tx) * Power(sz - tz, 2) +
           3 * fy * (sy - ty) * Power(sz - tz, 2) - sz * (9 * fz * sz + 4 * TrD) * tz +
           (9 * fz * sz + 2 * TrD) * Power(tz, 2) - 3 * fz * Power(tz, 3)) /
          (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
}
void StokesSLPVelLaplacian(double *s, double *t, double *f, double *pvelLaplacian) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double fx = f[0];
    const double fy = f[1];
    const double fz = f[2];
    const double TrD = f[3];
    // Laplacian does not depend on TrD

    double &p = pvelLaplacian[0];
    double &vx = pvelLaplacian[1];
    double &vy = pvelLaplacian[2];
    double &vz = pvelLaplacian[3];
    double &vxlap = pvelLaplacian[4];
    double &vylap = pvelLaplacian[5];
    double &vzlap = pvelLaplacian[6];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 7; i++) {
            pvelLaplacian[i] = 0;
        }
        return;
    }

    p = (fx * (-sx + tx) + fy * (-sy + ty) + fz * (-sz + tz)) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vx = ((sx - tx) * (fz * sz + TrD + fy * (sy - ty) - fz * tz) +
          fx * (2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) - 4 * sx * tx + 2 * Power(tx, 2) - 2 * sy * ty +
                Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vy = ((sy - ty) * (fz * sz + TrD + fx * (sx - tx) - fz * tz) +
          fy * (Power(sx, 2) + 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 4 * sy * ty +
                2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    vz = ((fy * sy + TrD + fx * (sx - tx) - fy * ty) * (sz - tz) +
          fz * (Power(sx, 2) + Power(sy, 2) + 2 * Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 2 * sy * ty +
                Power(ty, 2) - 4 * sz * tz + 2 * Power(tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));

    vxlap = (-3 * (sx - tx) * (fy * (sy - ty) + fz * (sz - tz)) +
             fx * (-2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) + 4 * sx * tx - 2 * Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
            (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vylap = (-3 * (sy - ty) * (fx * (sx - tx) + fz * (sz - tz)) +
             fy * (Power(sx, 2) - 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) + 4 * sy * ty -
                   2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
            (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vzlap = (-3 * (fx * (sx - tx) + fy * (sy - ty)) * (sz - tz) +
             fz * (Power(sx, 2) + Power(sy, 2) - 2 * Power(sz, 2) - 2 * sx * tx + Power(tx, 2) - 2 * sy * ty +
                   Power(ty, 2) + 4 * sz * tz - 2 * Power(tz, 2))) /
            (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
}

//                         3           3           9             4/16/
void StokesDLPVel(double *s, double *t, double *db, double *pvel) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double dbxx = db[0];
    const double dbxy = db[1];
    const double dbxz = db[2];
    const double dbyx = db[3];
    const double dbyy = db[4];
    const double dbyz = db[5];
    const double dbzx = db[6];
    const double dbzy = db[7];
    const double dbzz = db[8];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 4; i++) {
            pvel[i] = 0;
        }
        return;
    }

    double &p = pvel[0];
    double &vx = pvel[1];
    double &vy = pvel[2];
    double &vz = pvel[3];

    p = (dbzz * Power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * Power(sy, 2) - 3 * dbxz * sx * sz -
         3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * Power(sz, 2) - 2 * dbzz * sx * tx +
         3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * Power(tx, 2) +
         3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty -
         3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * Power(ty, 2) + 3 * dbxz * sx * tz + 3 * dbzx * sx * tz +
         3 * dbyz * sy * tz + 3 * dbzy * sy * tz + 4 * dbzz * sz * tz - 3 * dbxz * tx * tz - 3 * dbzx * tx * tz -
         3 * dbyz * ty * tz - 3 * dbzy * ty * tz - 2 * dbzz * Power(tz, 2) +
         dbyy * (Power(sx, 2) - 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) + 4 * sy * ty -
                 2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2)) +
         dbxx * (-2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) + 4 * sx * tx - 2 * Power(tx, 2) - 2 * sy * ty +
                 Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));

    vx = ((-sx + tx) *
          (-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vy = ((-sy + ty) *
          (-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vz = ((-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2)) *
          (-sz + tz)) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
}
void StokesDLPVelGrad(double *s, double *t, double *db, double *pvelGrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double dbxx = db[0];
    const double dbxy = db[1];
    const double dbxz = db[2];
    const double dbyx = db[3];
    const double dbyy = db[4];
    const double dbyz = db[5];
    const double dbzx = db[6];
    const double dbzy = db[7];
    const double dbzz = db[8];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 16; i++) {
            pvelGrad[i] = 0;
        }
        return;
    }

    double &p = pvelGrad[0];
    double &vx = pvelGrad[1];
    double &vy = pvelGrad[2];
    double &vz = pvelGrad[3];
    double &pgx = pvelGrad[4];
    double &pgy = pvelGrad[5];
    double &pgz = pvelGrad[6];
    double &vxgx = pvelGrad[7];
    double &vxgy = pvelGrad[8];
    double &vxgz = pvelGrad[9];
    double &vygx = pvelGrad[10];
    double &vygy = pvelGrad[11];
    double &vygz = pvelGrad[12];
    double &vzgx = pvelGrad[13];
    double &vzgy = pvelGrad[14];
    double &vzgz = pvelGrad[15];
    p = (dbzz * Power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * Power(sy, 2) - 3 * dbxz * sx * sz -
         3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * Power(sz, 2) - 2 * dbzz * sx * tx +
         3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * Power(tx, 2) +
         3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty -
         3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * Power(ty, 2) + 3 * dbxz * sx * tz + 3 * dbzx * sx * tz +
         3 * dbyz * sy * tz + 3 * dbzy * sy * tz + 4 * dbzz * sz * tz - 3 * dbxz * tx * tz - 3 * dbzx * tx * tz -
         3 * dbyz * ty * tz - 3 * dbzy * ty * tz - 2 * dbzz * Power(tz, 2) +
         dbyy * (Power(sx, 2) - 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) + 4 * sy * ty -
                 2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2)) +
         dbxx * (-2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) + 4 * sx * tx - 2 * Power(tx, 2) - 2 * sy * ty +
                 Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));

    vx = ((-sx + tx) *
          (-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vy = ((-sy + ty) *
          (-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vz = ((-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2)) *
          (-sz + tz)) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));

    pgx =
        ((4 * dbxx * (sx - tx) - 2 * (dbyy + dbzz) * (sx - tx) + 3 * (dbxy + dbyx) * (sy - ty) +
          3 * (dbxz + dbzx) * (sz - tz)) *
             (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
         5 * (sx - tx) *
             (dbzz * Power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * Power(sy, 2) - 3 * dbxz * sx * sz -
              3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * Power(sz, 2) -
              2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx +
              dbzz * Power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty +
              3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * Power(ty, 2) +
              dbyy * (Power(sx - tx, 2) - 2 * Power(sy - ty, 2) + Power(sz - tz, 2)) +
              dbxx * (-2 * Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
              (4 * dbzz * sz + 3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty)) * tz -
              2 * dbzz * Power(tz, 2))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    pgy =
        ((3 * (dbxy + dbyx) * (sx - tx) - 2 * (dbxx - 2 * dbyy + dbzz) * (sy - ty) + 3 * (dbyz + dbzy) * (sz - tz)) *
             (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
         5 * (sy - ty) *
             (dbzz * Power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * Power(sy, 2) - 3 * dbxz * sx * sz -
              3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * Power(sz, 2) -
              2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx +
              dbzz * Power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty +
              3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * Power(ty, 2) +
              dbyy * (Power(sx - tx, 2) - 2 * Power(sy - ty, 2) + Power(sz - tz, 2)) +
              dbxx * (-2 * Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
              (4 * dbzz * sz + 3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty)) * tz -
              2 * dbzz * Power(tz, 2))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    pgz =
        ((3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty) - 2 * (dbxx + dbyy - 2 * dbzz) * (sz - tz)) *
             (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
         5 * (sz - tz) *
             (dbzz * Power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * Power(sy, 2) - 3 * dbxz * sx * sz -
              3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * Power(sz, 2) -
              2 * dbzz * sx * tx + 3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx +
              dbzz * Power(tx, 2) + 3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty +
              3 * dbzy * sz * ty - 3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * Power(ty, 2) +
              dbyy * (Power(sx - tx, 2) - 2 * Power(sy - ty, 2) + Power(sz - tz, 2)) +
              dbxx * (-2 * Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
              (4 * dbzz * sz + 3 * (dbxz + dbzx) * (sx - tx) + 3 * (dbyz + dbzy) * (sy - ty)) * tz -
              2 * dbzz * Power(tz, 2))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));

    // vx grad

    vxgx = (3 * (-sx + tx) * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) *
                (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
            5 * (sx - tx) * (-sx + tx) *
                (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                       (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                 3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                 3 * dbzz * Power(tz, 2)) +
            (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) *
                (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                       (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                 3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                 3 * dbzz * Power(tz, 2))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vxgy = ((-sx + tx) *
            (3 * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) *
                 (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
             5 * (sy - ty) *
                 (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                        (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                  3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                  3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                  3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                  3 * dbzz * Power(tz, 2)))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vxgz = ((-sx + tx) *
            (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) *
                 (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
             5 * (sz - tz) *
                 (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                        (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                  3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                  3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                  3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                  3 * dbzz * Power(tz, 2)))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));

    // vy grad
    vygx = ((-sy + ty) *
            (3 * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) *
                 (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
             5 * (sx - tx) *
                 (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                        (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                  3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                  3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                  3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                  3 * dbzz * Power(tz, 2)))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vygy = (3 * (-sy + ty) * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) *
                (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
            5 * (sy - ty) * (-sy + ty) *
                (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                       (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                 3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                 3 * dbzz * Power(tz, 2)) +
            (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) *
                (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                       (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                 3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                 3 * dbzz * Power(tz, 2))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vygz = ((-sy + ty) *
            (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) *
                 (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
             5 * (sz - tz) *
                 (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                        (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                  3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                  3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                  3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                  3 * dbzz * Power(tz, 2)))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));

    // vz grad
    vzgx = ((-sz + tz) *
            (3 * (2 * dbxx * (sx - tx) + (dbxy + dbyx) * (sy - ty) + (dbxz + dbzx) * (sz - tz)) *
                 (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
             5 * (sx - tx) *
                 (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                        (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                  3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                  3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                  3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                  3 * dbzz * Power(tz, 2)))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vzgy = ((-sz + tz) *
            (3 * ((dbxy + dbyx) * (sx - tx) + 2 * dbyy * (sy - ty) + (dbyz + dbzy) * (sz - tz)) *
                 (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) +
             5 * (sy - ty) *
                 (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                        (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                  3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                  3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                  3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                  3 * dbzz * Power(tz, 2)))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vzgz = (3 * ((dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty) + 2 * dbzz * (sz - tz)) *
                (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) * (-sz + tz) +
            (Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)) *
                (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                       (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                 3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                 3 * dbzz * Power(tz, 2)) +
            5 * (sz - tz) * (-sz + tz) *
                (-3 * ((dbxy + dbyx) * sx * sy + dbyy * Power(sy, 2) + (dbxz + dbzx) * sx * sz +
                       (dbyz + dbzy) * sy * sz + dbzz * Power(sz, 2)) -
                 3 * dbxx * Power(sx - tx, 2) + 3 * ((dbxy + dbyx) * sy + (dbxz + dbzx) * sz) * tx +
                 3 * (2 * dbyy * sy + (dbyz + dbzy) * sz + (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbyy * Power(ty, 2) +
                 3 * (2 * dbzz * sz + (dbxz + dbzx) * (sx - tx) + (dbyz + dbzy) * (sy - ty)) * tz -
                 3 * dbzz * Power(tz, 2))) /
           (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
}

void StokesDLTraction(double *s, double *t, double *db, double *traction) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double dbxx = db[0];
    const double dbxy = db[1];
    const double dbxz = db[2];
    const double dbyx = db[3];
    const double dbyy = db[4];
    const double dbyz = db[5];
    const double dbzx = db[6];
    const double dbzy = db[7];
    const double dbzz = db[8];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 9; i++) {
            traction[i] = 0;
        }
        return;
    }

    double &txx = traction[0];
    double &txy = traction[1];
    double &txz = traction[2];
    double &tyx = traction[3];
    double &tyy = traction[4];
    double &tyz = traction[5];
    double &tzx = traction[6];
    double &tzy = traction[7];
    double &tzz = traction[8];

    double pvelgrad[16];
    StokesDLPVelGrad(s, t, db, pvelgrad);
    double p = pvelgrad[0];
    // vel grad
    double vxx = pvelgrad[7];
    double vxy = pvelgrad[8];
    double vxz = pvelgrad[9];
    double vyx = pvelgrad[10];
    double vyy = pvelgrad[11];
    double vyz = pvelgrad[12];
    double vzx = pvelgrad[13];
    double vzy = pvelgrad[14];
    double vzz = pvelgrad[15];
    txx = vxx + vxx - p;
    txy = vxy + vyx;
    txz = vxz + vzx;
    tyx = vxy + vyx;
    tyy = vyy + vyy - p;
    tyz = vyz + vzy;
    tzx = vzx + vxz;
    tzy = vzy + vyz;
    tzz = vzz + vzz - p;
}
void StokesDLPVelLaplacian(double *s, double *t, double *db, double *pvelLaplacian) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double dbxx = db[0];
    const double dbxy = db[1];
    const double dbxz = db[2];
    const double dbyx = db[3];
    const double dbyy = db[4];
    const double dbyz = db[5];
    const double dbzx = db[6];
    const double dbzy = db[7];
    const double dbzz = db[8];

    double &p = pvelLaplacian[0];
    double &vx = pvelLaplacian[1];
    double &vy = pvelLaplacian[2];
    double &vz = pvelLaplacian[3];
    double &vxlap = pvelLaplacian[4];
    double &vylap = pvelLaplacian[5];
    double &vzlap = pvelLaplacian[6];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 7; i++) {
            pvelLaplacian[i] = 0;
        }
        return;
    }

    // p,vx,vy,vz,vxlap,vylap,vzlap
    p = (dbzz * Power(sx, 2) - 3 * dbxy * sx * sy - 3 * dbyx * sx * sy + dbzz * Power(sy, 2) - 3 * dbxz * sx * sz -
         3 * dbzx * sx * sz - 3 * dbyz * sy * sz - 3 * dbzy * sy * sz - 2 * dbzz * Power(sz, 2) - 2 * dbzz * sx * tx +
         3 * dbxy * sy * tx + 3 * dbyx * sy * tx + 3 * dbxz * sz * tx + 3 * dbzx * sz * tx + dbzz * Power(tx, 2) +
         3 * dbxy * sx * ty + 3 * dbyx * sx * ty - 2 * dbzz * sy * ty + 3 * dbyz * sz * ty + 3 * dbzy * sz * ty -
         3 * dbxy * tx * ty - 3 * dbyx * tx * ty + dbzz * Power(ty, 2) + 3 * dbxz * sx * tz + 3 * dbzx * sx * tz +
         3 * dbyz * sy * tz + 3 * dbzy * sy * tz + 4 * dbzz * sz * tz - 3 * dbxz * tx * tz - 3 * dbzx * tx * tz -
         3 * dbyz * ty * tz - 3 * dbzy * ty * tz - 2 * dbzz * Power(tz, 2) +
         dbyy * (Power(sx, 2) - 2 * Power(sy, 2) + Power(sz, 2) - 2 * sx * tx + Power(tx, 2) + 4 * sy * ty -
                 2 * Power(ty, 2) - 2 * sz * tz + Power(tz, 2)) +
         dbxx * (-2 * Power(sx, 2) + Power(sy, 2) + Power(sz, 2) + 4 * sx * tx - 2 * Power(tx, 2) - 2 * sy * ty +
                 Power(ty, 2) - 2 * sz * tz + Power(tz, 2))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));

    vx = ((-sx + tx) *
          (-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vy = ((-sy + ty) *
          (-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2))) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    vz = ((-3 * dbxx * Power(sx - tx, 2) - 3 * dbxy * (sx - tx) * (sy - ty) - 3 * dbyx * (sx - tx) * (sy - ty) -
           3 * dbyy * Power(sy - ty, 2) - 3 * dbxz * (sx - tx) * (sz - tz) - 3 * dbzx * (sx - tx) * (sz - tz) -
           3 * dbyz * (sy - ty) * (sz - tz) - 3 * dbzy * (sy - ty) * (sz - tz) - 3 * dbzz * Power(sz - tz, 2)) *
          (-sz + tz)) /
         (8. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));

    vxlap =
        (3 *
         (dbzz * Power(sx, 3) - 4 * dbxy * Power(sx, 2) * sy - 4 * dbyx * Power(sx, 2) * sy + dbzz * sx * Power(sy, 2) +
          dbxy * Power(sy, 3) + dbyx * Power(sy, 3) - 4 * dbxz * Power(sx, 2) * sz - 4 * dbzx * Power(sx, 2) * sz -
          5 * dbyz * sx * sy * sz - 5 * dbzy * sx * sy * sz + dbxz * Power(sy, 2) * sz + dbzx * Power(sy, 2) * sz -
          4 * dbzz * sx * Power(sz, 2) + dbxy * sy * Power(sz, 2) + dbyx * sy * Power(sz, 2) + dbxz * Power(sz, 3) +
          dbzx * Power(sz, 3) - 3 * dbzz * Power(sx, 2) * tx + 8 * dbxy * sx * sy * tx + 8 * dbyx * sx * sy * tx -
          dbzz * Power(sy, 2) * tx + 8 * dbxz * sx * sz * tx + 8 * dbzx * sx * sz * tx + 5 * dbyz * sy * sz * tx +
          5 * dbzy * sy * sz * tx + 4 * dbzz * Power(sz, 2) * tx + 3 * dbzz * sx * Power(tx, 2) -
          4 * dbxy * sy * Power(tx, 2) - 4 * dbyx * sy * Power(tx, 2) - 4 * dbxz * sz * Power(tx, 2) -
          4 * dbzx * sz * Power(tx, 2) - dbzz * Power(tx, 3) + 4 * dbxy * Power(sx, 2) * ty +
          4 * dbyx * Power(sx, 2) * ty - 2 * dbzz * sx * sy * ty - 3 * dbxy * Power(sy, 2) * ty -
          3 * dbyx * Power(sy, 2) * ty + 5 * dbyz * sx * sz * ty + 5 * dbzy * sx * sz * ty - 2 * dbxz * sy * sz * ty -
          2 * dbzx * sy * sz * ty - dbxy * Power(sz, 2) * ty - dbyx * Power(sz, 2) * ty - 8 * dbxy * sx * tx * ty -
          8 * dbyx * sx * tx * ty + 2 * dbzz * sy * tx * ty - 5 * dbyz * sz * tx * ty - 5 * dbzy * sz * tx * ty +
          4 * dbxy * Power(tx, 2) * ty + 4 * dbyx * Power(tx, 2) * ty + dbzz * sx * Power(ty, 2) +
          3 * dbxy * sy * Power(ty, 2) + 3 * dbyx * sy * Power(ty, 2) + dbxz * sz * Power(ty, 2) +
          dbzx * sz * Power(ty, 2) - dbzz * tx * Power(ty, 2) - dbxy * Power(ty, 3) - dbyx * Power(ty, 3) -
          dbxx * (sx - tx) * (2 * Power(sx - tx, 2) - 3 * (Power(sy - ty, 2) + Power(sz - tz, 2))) +
          dbyy * (sx - tx) * (Power(sx - tx, 2) - 4 * Power(sy - ty, 2) + Power(sz - tz, 2)) +
          (5 * dbyz * sx * sy + 5 * dbzy * sx * sy + 8 * dbzz * sx * sz - 2 * dbxy * sy * sz - 2 * dbyx * sy * sz -
           5 * dbyz * sy * tx - 5 * dbzy * sy * tx - 8 * dbzz * sz * tx +
           dbxz * (-3 * Power(sz, 2) + 4 * Power(sx - tx, 2) - Power(sy - ty, 2)) +
           dbzx * (-3 * Power(sz, 2) + 4 * Power(sx - tx, 2) - Power(sy - ty, 2)) - 5 * dbyz * sx * ty -
           5 * dbzy * sx * ty + 2 * dbxy * sz * ty + 2 * dbyx * sz * ty + 5 * dbyz * tx * ty + 5 * dbzy * tx * ty) *
              tz +
          (3 * (dbxz + dbzx) * sz + 4 * dbzz * (-sx + tx) + (dbxy + dbyx) * (sy - ty)) * Power(tz, 2) -
          (dbxz + dbzx) * Power(tz, 3))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vylap =
        (3 *
         (-4 * dbxx * Power(sx, 2) * sy + 3 * dbyy * Power(sx, 2) * sy + dbzz * Power(sx, 2) * sy +
          dbxx * Power(sy, 3) - 2 * dbyy * Power(sy, 3) + dbzz * Power(sy, 3) + dbyz * Power(sx, 2) * sz +
          dbzy * Power(sx, 2) * sz - 5 * dbxz * sx * sy * sz - 5 * dbzx * sx * sy * sz - 4 * dbyz * Power(sy, 2) * sz -
          4 * dbzy * Power(sy, 2) * sz + dbxx * sy * Power(sz, 2) + 3 * dbyy * sy * Power(sz, 2) -
          4 * dbzz * sy * Power(sz, 2) + dbyz * Power(sz, 3) + dbzy * Power(sz, 3) + 8 * dbxx * sx * sy * tx -
          6 * dbyy * sx * sy * tx - 2 * dbzz * sx * sy * tx - 2 * dbyz * sx * sz * tx - 2 * dbzy * sx * sz * tx +
          5 * dbxz * sy * sz * tx + 5 * dbzx * sy * sz * tx - 4 * dbxx * sy * Power(tx, 2) +
          3 * dbyy * sy * Power(tx, 2) + dbzz * sy * Power(tx, 2) + dbyz * sz * Power(tx, 2) +
          dbzy * sz * Power(tx, 2) + 4 * dbxx * Power(sx, 2) * ty - 3 * dbyy * Power(sx, 2) * ty -
          dbzz * Power(sx, 2) * ty - 3 * dbxx * Power(sy, 2) * ty + 6 * dbyy * Power(sy, 2) * ty -
          3 * dbzz * Power(sy, 2) * ty + 5 * dbxz * sx * sz * ty + 5 * dbzx * sx * sz * ty + 8 * dbyz * sy * sz * ty +
          8 * dbzy * sy * sz * ty - dbxx * Power(sz, 2) * ty - 3 * dbyy * Power(sz, 2) * ty +
          4 * dbzz * Power(sz, 2) * ty - 8 * dbxx * sx * tx * ty + 6 * dbyy * sx * tx * ty + 2 * dbzz * sx * tx * ty -
          5 * dbxz * sz * tx * ty - 5 * dbzx * sz * tx * ty + 4 * dbxx * Power(tx, 2) * ty -
          3 * dbyy * Power(tx, 2) * ty - dbzz * Power(tx, 2) * ty + 3 * dbxx * sy * Power(ty, 2) -
          6 * dbyy * sy * Power(ty, 2) + 3 * dbzz * sy * Power(ty, 2) - 4 * dbyz * sz * Power(ty, 2) -
          4 * dbzy * sz * Power(ty, 2) - dbxx * Power(ty, 3) + 2 * dbyy * Power(ty, 3) - dbzz * Power(ty, 3) +
          dbxy * (sx - tx) * (Power(sx - tx, 2) - 4 * Power(sy - ty, 2) + Power(sz - tz, 2)) +
          dbyx * (sx - tx) * (Power(sx - tx, 2) - 4 * Power(sy - ty, 2) + Power(sz - tz, 2)) -
          (dbyz * (3 * Power(sz, 2) + Power(sx - tx, 2) - 4 * Power(sy - ty, 2)) +
           dbzy * (3 * Power(sz, 2) + Power(sx - tx, 2) - 4 * Power(sy - ty, 2)) -
           (-2 * (dbxx + 3 * dbyy - 4 * dbzz) * sz + 5 * (dbxz + dbzx) * (sx - tx)) * (sy - ty)) *
              tz +
          (3 * (dbyz + dbzy) * sz + (dbxx + 3 * dbyy - 4 * dbzz) * (sy - ty)) * Power(tz, 2) -
          (dbyz + dbzy) * Power(tz, 3))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
    vzlap =
        (3 *
         (dbyz * Power(sx, 2) * sy + dbzy * Power(sx, 2) * sy + dbyz * Power(sy, 3) + dbzy * Power(sy, 3) -
          4 * dbxx * Power(sx, 2) * sz + dbyy * Power(sx, 2) * sz + 3 * dbzz * Power(sx, 2) * sz -
          5 * dbxy * sx * sy * sz - 5 * dbyx * sx * sy * sz + dbxx * Power(sy, 2) * sz - 4 * dbyy * Power(sy, 2) * sz +
          3 * dbzz * Power(sy, 2) * sz - 4 * dbyz * sy * Power(sz, 2) - 4 * dbzy * sy * Power(sz, 2) +
          dbxx * Power(sz, 3) + dbyy * Power(sz, 3) - 2 * dbzz * Power(sz, 3) - 2 * dbyz * sx * sy * tx -
          2 * dbzy * sx * sy * tx + 8 * dbxx * sx * sz * tx - 2 * dbyy * sx * sz * tx - 6 * dbzz * sx * sz * tx +
          5 * dbxy * sy * sz * tx + 5 * dbyx * sy * sz * tx + dbyz * sy * Power(tx, 2) + dbzy * sy * Power(tx, 2) -
          4 * dbxx * sz * Power(tx, 2) + dbyy * sz * Power(tx, 2) + 3 * dbzz * sz * Power(tx, 2) -
          dbyz * Power(sx, 2) * ty - dbzy * Power(sx, 2) * ty - 3 * dbyz * Power(sy, 2) * ty -
          3 * dbzy * Power(sy, 2) * ty + 5 * dbxy * sx * sz * ty + 5 * dbyx * sx * sz * ty - 2 * dbxx * sy * sz * ty +
          8 * dbyy * sy * sz * ty - 6 * dbzz * sy * sz * ty + 4 * dbyz * Power(sz, 2) * ty +
          4 * dbzy * Power(sz, 2) * ty + 2 * dbyz * sx * tx * ty + 2 * dbzy * sx * tx * ty - 5 * dbxy * sz * tx * ty -
          5 * dbyx * sz * tx * ty - dbyz * Power(tx, 2) * ty - dbzy * Power(tx, 2) * ty + 3 * dbyz * sy * Power(ty, 2) +
          3 * dbzy * sy * Power(ty, 2) + dbxx * sz * Power(ty, 2) - 4 * dbyy * sz * Power(ty, 2) +
          3 * dbzz * sz * Power(ty, 2) - dbyz * Power(ty, 3) - dbzy * Power(ty, 3) +
          dbxz * (sx - tx) * (Power(sx, 2) - 2 * sx * tx + Power(tx, 2) + Power(sy - ty, 2) - 4 * Power(sz - tz, 2)) +
          dbzx * (sx - tx) * (Power(sx, 2) - 2 * sx * tx + Power(tx, 2) + Power(sy - ty, 2) - 4 * Power(sz - tz, 2)) +
          (-3 * dbzz * Power(sx, 2) + 5 * dbxy * sx * sy + 5 * dbyx * sx * sy - 3 * dbzz * Power(sy, 2) +
           8 * dbyz * sy * sz + 8 * dbzy * sy * sz + 6 * dbzz * Power(sz, 2) + 6 * dbzz * sx * tx - 5 * dbxy * sy * tx -
           5 * dbyx * sy * tx - 3 * dbzz * Power(tx, 2) -
           dbyy * (3 * Power(sz, 2) + Power(sx - tx, 2) - 4 * Power(sy - ty, 2)) +
           dbxx * (-3 * Power(sz, 2) + 4 * Power(sx - tx, 2) - Power(sy - ty, 2)) +
           (6 * dbzz * sy - 8 * (dbyz + dbzy) * sz - 5 * (dbxy + dbyx) * (sx - tx)) * ty - 3 * dbzz * Power(ty, 2)) *
              tz +
          (3 * (dbxx + dbyy - 2 * dbzz) * sz - 4 * (dbyz + dbzy) * (sy - ty)) * Power(tz, 2) -
          (dbxx + dbyy - 2 * dbzz) * Power(tz, 3))) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 3.5));
}

void LaplaceSLPGrad(double *s, double *t, double *q, double *pgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double charge = q[0];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 4; i++) {
            pgrad[i] = 0;
        }
        return;
    }

    double &p = pgrad[0];
    double &gx = pgrad[1];
    double &gy = pgrad[2];
    double &gz = pgrad[3];
    p = charge / (4. * Pi * Sqrt(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)));
    gx = (charge * (sx - tx)) / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    gy = (charge * (sy - ty)) / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    gz = (charge * (sz - tz)) / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
}

void LaplaceDLPGrad(double *s, double *t, double *db, double *pgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double dx = db[0];
    const double dy = db[1];
    const double dz = db[2];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 4; i++) {
            pgrad[i] = 0;
        }
        return;
    }

    double &p = pgrad[0];
    double &gx = pgrad[1];
    double &gy = pgrad[2];
    double &gz = pgrad[3];

    p = (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz)) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    gx = dx / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
         (3 * (sx - tx) * (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz))) /
             (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    gy = dy / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
         (3 * (sy - ty) * (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz))) /
             (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    gz = dz / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
         (3 * (sz - tz) * (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz))) /
             (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
}

void LaplaceSLPGradGrad(double *s, double *t, double *q, double *pgradgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double charge = q[0];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 10; i++) {
            pgradgrad[i] = 0;
        }
        return;
    }

    double &p = pgradgrad[0];
    double &gx = pgradgrad[1];
    double &gy = pgradgrad[2];
    double &gz = pgradgrad[3];
    double &gxx = pgradgrad[4];
    double &gxy = pgradgrad[5];
    double &gxz = pgradgrad[6];
    double &gyy = pgradgrad[7];
    double &gyz = pgradgrad[8];
    double &gzz = pgradgrad[9];
    const double prefac = 1 / (4 * M_PI);

    p = charge / (4. * Pi * Sqrt(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2)));
    gx = (charge * (sx - tx)) / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    gy = (charge * (sy - ty)) / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    gz = (charge * (sz - tz)) / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    gxx = prefac * charge *
          ((3 * Power(-sx + tx, 2)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), -1.5));
    gxy = prefac * charge *
          ((3 * (-sx + tx) * (-sy + ty)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gxz = prefac * charge *
          ((3 * (-sx + tx) * (-sz + tz)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gyy = prefac * charge *
          ((3 * Power(-sy + ty, 2)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), -1.5));
    gyz = prefac * charge *
          ((3 * (-sy + ty) * (-sz + tz)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gzz = prefac * charge *
          ((3 * Power(-sz + tz, 2)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), -1.5));
}

void LaplaceDLPGradGrad(double *s, double *t, double *db, double *pgradgrad) {

    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double dx = db[0];
    const double dy = db[1];
    const double dz = db[2];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 10; i++) {
            pgradgrad[i] = 0;
        }
        return;
    }

    double &p = pgradgrad[0];
    double &gx = pgradgrad[1];
    double &gy = pgradgrad[2];
    double &gz = pgradgrad[3];
    double &gxx = pgradgrad[4];
    double &gxy = pgradgrad[5];
    double &gxz = pgradgrad[6];
    double &gyy = pgradgrad[7];
    double &gyz = pgradgrad[8];
    double &gzz = pgradgrad[9];
    const double prefac = 1 / (4 * M_PI);

    p = (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz)) /
        (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5));
    gx = dx / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
         (3 * (sx - tx) * (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz))) /
             (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    gy = dy / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
         (3 * (sy - ty) * (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz))) /
             (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    gz = dz / (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 1.5)) +
         (3 * (sz - tz) * (dx * (-sx + tx) + dy * (-sy + ty) + dz * (-sz + tz))) /
             (4. * Pi * Power(Power(sx - tx, 2) + Power(sy - ty, 2) + Power(sz - tz, 2), 2.5));
    const double s0 = db[0];
    const double s1 = db[1];
    const double s2 = db[2];
    gxx = prefac *
          ((15 * s0 * Power(-sx + tx, 3)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
           (15 * s1 * Power(-sx + tx, 2) * (-sy + ty)) /
               Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
           (15 * s2 * Power(-sx + tx, 2) * (-sz + tz)) /
               Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) -
           (9 * s0 * (-sx + tx)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           (3 * s1 * (-sy + ty)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           (3 * s2 * (-sz + tz)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gxy = prefac * ((15 * s0 * Power(-sx + tx, 2) * (-sy + ty)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
                    (15 * s1 * (-sx + tx) * Power(-sy + ty, 2)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
                    (15 * s2 * (-sx + tx) * (-sy + ty) * (-sz + tz)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) -
                    (3 * s1 * (-sx + tx)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
                    (3 * s0 * (-sy + ty)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gxz = prefac * ((15 * s0 * Power(-sx + tx, 2) * (-sz + tz)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
                    (15 * s1 * (-sx + tx) * (-sy + ty) * (-sz + tz)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
                    (15 * s2 * (-sx + tx) * Power(-sz + tz, 2)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) -
                    (3 * s2 * (-sx + tx)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
                    (3 * s0 * (-sz + tz)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gyy = prefac *
          ((15 * s0 * (-sx + tx) * Power(-sy + ty, 2)) /
               Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
           (15 * s1 * Power(-sy + ty, 3)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
           (15 * s2 * Power(-sy + ty, 2) * (-sz + tz)) /
               Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) -
           (3 * s0 * (-sx + tx)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           (9 * s1 * (-sy + ty)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           (3 * s2 * (-sz + tz)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gyz = prefac * ((15 * s0 * (-sx + tx) * (-sy + ty) * (-sz + tz)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
                    (15 * s1 * Power(-sy + ty, 2) * (-sz + tz)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
                    (15 * s2 * (-sy + ty) * Power(-sz + tz, 2)) /
                        Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) -
                    (3 * s2 * (-sy + ty)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
                    (3 * s1 * (-sz + tz)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
    gzz = prefac *
          ((15 * s0 * (-sx + tx) * Power(-sz + tz, 2)) /
               Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
           (15 * s1 * (-sy + ty) * Power(-sz + tz, 2)) /
               Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) +
           (15 * s2 * Power(-sz + tz, 3)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 3.5) -
           (3 * s0 * (-sx + tx)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           (3 * s1 * (-sy + ty)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5) -
           (9 * s2 * (-sz + tz)) / Power(Power(-sx + tx, 2) + Power(-sy + ty, 2) + Power(-sz + tz, 2), 2.5));
}

void StokesRegSLVel(double *s, double *t, double *f, double *vel) {
    double dx = t[0] - s[0];
    double dy = t[1] - s[1];
    double dz = t[2] - s[2];
    double fx = f[0];
    double fy = f[1];
    double fz = f[2];
    double eps = f[3];

    // length squared of r
    double r2 = dx * dx + dy * dy + dz * dz;
    // regularization parameter squared
    double eps2 = eps * eps;

    double denom = std::sqrt(eps2 + r2);
    double velocity_denom = denom * (eps2 + r2);
    double velocity_numer = r2 + 2 * eps2;
    double pressure_denom = velocity_denom * (eps2 + r2);
    double pressure_numer = 2 * r2 + 5 * eps2;
    double fdotr = dx * fx + dy * fy + dz * fz;

    vel[0] += (velocity_numer * fx + fdotr * dx) / (velocity_denom * 8 * M_PI);
    vel[1] += (velocity_numer * fy + fdotr * dy) / (velocity_denom * 8 * M_PI);
    vel[2] += (velocity_numer * fz + fdotr * dz) / (velocity_denom * 8 * M_PI);
}

void StokesRegSLVelOmega(double *s, double *t, double *f, double *velomega) {
    double dx = t[0] - s[0];
    double dy = t[1] - s[1];
    double dz = t[2] - s[2];
    double fx = f[0];
    double fy = f[1];
    double fz = f[2];
    double tx = f[3];
    double ty = f[4];
    double tz = f[5];
    double eps = f[6];

    // length squared of r
    double r2 = dx * dx + dy * dy + dz * dz;
    double r4 = r2 * r2;

    // regularization parameter squared
    double eps2 = eps * eps;
    double eps4 = eps2 * eps2;

    double denom_arg = eps2 + r2;
    double stokeslet_denom = M_PI * 8 * denom_arg * std::sqrt(denom_arg);
    double rotlet_denom = 2 * stokeslet_denom * denom_arg;
    double dipole_denom = 2 * rotlet_denom * denom_arg;
    double rotlet_coef = (2 * r2 + 5.0 * eps2) / rotlet_denom;
    double D1 = (10 * eps4 - 7 * eps2 * r2 - 2 * r4) / dipole_denom;
    double D2 = (21 * eps2 + 6 * r2) / dipole_denom;
    double H2 = 1.0 / stokeslet_denom;
    double H1 = (r2 + 2.0 * eps2) * H2;

    double fcurlrx = fy * dz - fz * dy;
    double fcurlry = fz * dx - fx * dz;
    double fcurlrz = fx * dy - fy * dx;

    double tcurlrx = ty * dz - tz * dy;
    double tcurlry = tz * dx - tx * dz;
    double tcurlrz = tx * dy - ty * dx;

    double fdotr = fx * dx + fy * dy + fz * dz;
    double tdotr = tx * dx + ty * dy + tz * dz;

    // x component of velocity from stokeslet
    velomega[0] += H1 * fx + H2 * fdotr * dx;

    // y component of velocity from stokeslet
    velomega[1] += H1 * fy + H2 * fdotr * dy;

    // z component of velocity from stokeslet
    velomega[2] += H1 * fz + H2 * fdotr * dz;

    // x component of velocity from rotlet
    velomega[0] += rotlet_coef * tcurlrx;

    // y component of velocity from rotlet
    velomega[1] += rotlet_coef * tcurlry;

    // z component of velocity from rotlet
    velomega[2] += rotlet_coef * tcurlrz;

    // x component of angular velocity from dipole
    velomega[3] += D1 * tx + D2 * tdotr * dx;

    // y component of angular velocity from dipole
    velomega[4] += D1 * ty + D2 * tdotr * dy;

    // z component of angular velocity from dipole
    velomega[5] += D1 * tz + D2 * tdotr * dz;

    // x component of angular velocity from rotlet
    velomega[3] += rotlet_coef * fcurlrx;

    // y component of angular velocity from rotlet
    velomega[4] += rotlet_coef * fcurlry;

    // z component of angular velocity from rotlet
    velomega[5] += rotlet_coef * fcurlrz;
}

void StokesSL(double *s, double *t, double *f, double *vlapv) {
    const double fx = f[0];
    const double fy = f[1];
    const double fz = f[2];
    const double a = 0;

    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];
    const double dx = t[0] - sx;
    const double dy = t[1] - sy;
    const double dz = t[2] - sz;

    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 == 0.0)
        return;

    double a2 = a * a;

    double invr = 1.0 / sqrt(r2);
    double invr3 = invr / r2;
    double invr5 = invr3 / r2;
    double fdotr = fx * dx + fy * dy + fz * dz;
    vlapv[0] += fx * invr + dx * fdotr * invr3;
    vlapv[1] += fy * invr + dy * fdotr * invr3;
    vlapv[2] += fz * invr + dz * fdotr * invr3;

    vlapv[0] += a2 * (2 * fx * invr3 - 6 * fdotr * dx * invr5) / 6.0;
    vlapv[1] += a2 * (2 * fy * invr3 - 6 * fdotr * dy * invr5) / 6.0;
    vlapv[2] += a2 * (2 * fz * invr3 - 6 * fdotr * dz * invr5) / 6.0;

    vlapv[3] += 2 * fx * invr3 - 6 * fdotr * dx * invr5;
    vlapv[4] += 2 * fy * invr3 - 6 * fdotr * dy * invr5;
    vlapv[5] += 2 * fz * invr3 - 6 * fdotr * dz * invr5;

    for (int i = 0; i < 6; ++i)
        vlapv[i] /= 8.0 * M_PI;
}

void StokesDL(double *s, double *t, double *f, double *vel){};

void StokesSLRPY(double *s, double *t, double *f, double *vlapv) {
    const double fx = f[0];
    const double fy = f[1];
    const double fz = f[2];
    const double a = f[3];

    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];
    const double dx = t[0] - sx;
    const double dy = t[1] - sy;
    const double dz = t[2] - sz;

    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 == 0.0)
        return;

    double a2 = a * a;

    double invr = 1.0 / sqrt(r2);
    double invr3 = invr / r2;
    double invr5 = invr3 / r2;
    double fdotr = fx * dx + fy * dy + fz * dz;
    vlapv[0] += fx * invr + dx * fdotr * invr3;
    vlapv[1] += fy * invr + dy * fdotr * invr3;
    vlapv[2] += fz * invr + dz * fdotr * invr3;

    vlapv[0] += a2 * (2 * fx * invr3 - 6 * fdotr * dx * invr5) / 6.0;
    vlapv[1] += a2 * (2 * fy * invr3 - 6 * fdotr * dy * invr5) / 6.0;
    vlapv[2] += a2 * (2 * fz * invr3 - 6 * fdotr * dz * invr5) / 6.0;

    vlapv[3] += 2 * fx * invr3 - 6 * fdotr * dx * invr5;
    vlapv[4] += 2 * fy * invr3 - 6 * fdotr * dy * invr5;
    vlapv[5] += 2 * fz * invr3 - 6 * fdotr * dz * invr5;

    for (int i = 0; i < 6; ++i)
        vlapv[i] /= 8.0 * M_PI;
}

void StokesDLRPY(double *s, double *t, double *f, double *vel){};

void StokesRegDLVel(double *s, double *t, double *f, double *vel){};
void StokesRegDLVelOmega(double *s, double *t, double *f, double *velomega){};

void LaplacePhiGradPhi(double *s, double *t, double *fin, double *phigradphi) {
    auto f = [fin](int i) { return fin[i]; };
    auto r = [s, t](int i) { return t[i] - s[i]; };

    phigradphi[0] =
        (2 * (f(0) * r(0) + f(1) * r(1)) * r(2) + f(2) * (6 * (Power(r(0), 2) + Power(r(1), 2)) + 4 * Power(r(2), 2))) /
        Power(Power(r(0), 2) + Power(r(1), 2) + Power(r(2), 2), 1.5);
    phigradphi[1] =
        (-6 * f(2) * r(0) * (Power(r(0), 2) + Power(r(1), 2)) +
         2 * r(2) * (-3 * f(1) * r(0) * r(1) + f(0) * (-2 * Power(r(0), 2) + Power(r(1), 2) + Power(r(2), 2)))) /
        Power(Power(r(0), 2) + Power(r(1), 2) + Power(r(2), 2), 2.5);
    phigradphi[2] =
        (-6 * f(2) * r(1) * (Power(r(0), 2) + Power(r(1), 2)) +
         2 * r(2) * (-3 * f(0) * r(0) * r(1) + f(1) * (Power(r(0), 2) - 2 * Power(r(1), 2) + Power(r(2), 2)))) /
        Power(Power(r(0), 2) + Power(r(1), 2) + Power(r(2), 2), 2.5);
    phigradphi[3] = (2 * (f(0) * r(0) + f(1) * r(1)) * (Power(r(0), 2) + Power(r(1), 2)) -
                     10 * f(2) * (Power(r(0), 2) + Power(r(1), 2)) * r(2) -
                     4 * (f(0) * r(0) + f(1) * r(1)) * Power(r(2), 2) - 4 * f(2) * Power(r(2), 3)) /
                    Power(Power(r(0), 2) + Power(r(1), 2) + Power(r(2), 2), 2.5);

    for (int i = 0; i < 4; ++i)
        phigradphi[i] *= f(3) * f(3) / (8 * M_PI);
}

void LaplaceQPGradGrad(double *s, double *t, double *q, double *pgradgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];
    const double dx = t[0] - sx;
    const double dy = t[1] - sy;
    const double dz = t[2] - sz;

    double r2 = dx * dx + dy * dy + dz * dz;
    if (r2 == 0.0)
        return;
    const double q0 = q[0];
    const double q1 = q[1];
    const double q2 = q[2];
    const double q3 = q[3];
    const double q4 = q[4];
    const double q5 = q[5];
    const double q6 = q[6];
    const double q7 = q[7];
    const double q8 = q[8];

    double &ql = pgradgrad[0];
    double &qlgx = pgradgrad[1];
    double &qlgy = pgradgrad[2];
    double &qlgz = pgradgrad[3];
    double &qlgxx = pgradgrad[4];
    double &qlgxy = pgradgrad[5];
    double &qlgxz = pgradgrad[6];
    double &qlgyy = pgradgrad[7];
    double &qlgyz = pgradgrad[8];
    double &qlgzz = pgradgrad[9];
    const double prefac = 1 / (4 * M_PI);

    ql = prefac * (((3 * Power(dx, 2)) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) -
                    Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), -1.5)) *
                       q0 +
                   (3 * dx * dy * q1) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) +
                   (3 * dx * dz * q2) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) +
                   (3 * dx * dy * q3) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) +
                   ((3 * Power(dy, 2)) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) -
                    Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), -1.5)) *
                       q4 +
                   (3 * dy * dz * q5) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) +
                   (3 * dx * dz * q6) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) +
                   (3 * dy * dz * q7) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) +
                   ((3 * Power(dz, 2)) / Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 2.5) -
                    Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), -1.5)) *
                       q8);
    qlgx =
        prefac *
        ((3 * (-4 * Power(dx, 2) * (dy * (q1 + q3) + dz * (q2 + q6)) +
               (Power(dy, 2) + Power(dz, 2)) * (dy * (q1 + q3) + dz * (q2 + q6)) + Power(dx, 3) * (-2 * q0 + q4 + q8) +
               dx * (-5 * dy * dz * (q5 + q7) + Power(dz, 2) * (3 * q0 + q4 - 4 * q8) +
                     Power(dy, 2) * (3 * q0 - 4 * q4 + q8)))) /
         Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 3.5));
    qlgy = prefac * ((3 * (Power(dx, 3) * (q1 + q3) +
                           dx * (-4 * Power(dy, 2) * (q1 + q3) + Power(dz, 2) * (q1 + q3) - 5 * dy * dz * (q2 + q6)) -
                           4 * Power(dy, 2) * dz * (q5 + q7) + Power(dz, 3) * (q5 + q7) +
                           dy * Power(dz, 2) * (q0 + 3 * q4 - 4 * q8) + Power(dy, 3) * (q0 - 2 * q4 + q8) +
                           Power(dx, 2) * (dz * (q5 + q7) + dy * (-4 * q0 + 3 * q4 + q8)))) /
                     Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 3.5));
    qlgz = prefac * ((3 * (Power(dx, 3) * (q2 + q6) +
                           dx * (-5 * dy * dz * (q1 + q3) + Power(dy, 2) * (q2 + q6) - 4 * Power(dz, 2) * (q2 + q6)) +
                           Power(dy, 3) * (q5 + q7) - 4 * dy * Power(dz, 2) * (q5 + q7) +
                           Power(dz, 3) * (q0 + q4 - 2 * q8) + Power(dy, 2) * dz * (q0 - 4 * q4 + 3 * q8) +
                           Power(dx, 2) * (dy * (q5 + q7) + dz * (-4 * q0 + q4 + 3 * q8)))) /
                     Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 3.5));

    qlgxx = prefac *
            ((3 * (20 * Power(dx, 3) * (dy * (q1 + q3) + dz * (q2 + q6)) -
                   15 * dx * (Power(dy, 2) + Power(dz, 2)) * (dy * (q1 + q3) + dz * (q2 + q6)) -
                   3 * Power(dx, 2) *
                       (-10 * dy * dz * (q5 + q7) + Power(dz, 2) * (8 * q0 + q4 - 9 * q8) +
                        Power(dy, 2) * (8 * q0 - 9 * q4 + q8)) +
                   (Power(dy, 2) + Power(dz, 2)) * (-5 * dy * dz * (q5 + q7) + Power(dz, 2) * (3 * q0 + q4 - 4 * q8) +
                                                    Power(dy, 2) * (3 * q0 - 4 * q4 + q8)) +
                   Power(dx, 4) * (8 * q0 - 4 * (q4 + q8)))) /
             Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 4.5));
    qlgxy = prefac * ((-3 * (4 * Power(dx, 4) * (q1 + q3) +
                             (Power(dy, 2) + Power(dz, 2)) *
                                 (4 * Power(dy, 2) * (q1 + q3) - Power(dz, 2) * (q1 + q3) + 5 * dy * dz * (q2 + q6)) -
                             3 * Power(dx, 2) *
                                 (9 * Power(dy, 2) * (q1 + q3) - Power(dz, 2) * (q1 + q3) + 10 * dy * dz * (q2 + q6)) +
                             5 * dx *
                                 (-6 * Power(dy, 2) * dz * (q5 + q7) + Power(dz, 3) * (q5 + q7) +
                                  3 * dy * Power(dz, 2) * (q0 + q4 - 2 * q8) + Power(dy, 3) * (3 * q0 - 4 * q4 + q8)) +
                             5 * Power(dx, 3) * (dz * (q5 + q7) + dy * (-4 * q0 + 3 * q4 + q8)))) /
                      Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 4.5));
    qlgxz = prefac * ((-3 * (4 * Power(dx, 4) * (q2 + q6) +
                             3 * Power(dx, 2) *
                                 (-10 * dy * dz * (q1 + q3) + Power(dy, 2) * (q2 + q6) - 9 * Power(dz, 2) * (q2 + q6)) -
                             (Power(dy, 2) + Power(dz, 2)) *
                                 (-5 * dy * dz * (q1 + q3) + Power(dy, 2) * (q2 + q6) - 4 * Power(dz, 2) * (q2 + q6)) +
                             5 * dx *
                                 (Power(dy, 3) * (q5 + q7) - 6 * dy * Power(dz, 2) * (q5 + q7) +
                                  Power(dz, 3) * (3 * q0 + q4 - 4 * q8) + 3 * Power(dy, 2) * dz * (q0 - 2 * q4 + q8)) +
                             5 * Power(dx, 3) * (dy * (q5 + q7) + dz * (-4 * q0 + q4 + 3 * q8)))) /
                      Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 4.5));
    qlgyy = prefac *
            ((-3 * (5 * Power(dx, 3) * (3 * dy * (q1 + q3) + dz * (q2 + q6)) -
                    5 * dx *
                        (4 * Power(dy, 3) * (q1 + q3) - 3 * dy * Power(dz, 2) * (q1 + q3) +
                         6 * Power(dy, 2) * dz * (q2 + q6) - Power(dz, 3) * (q2 + q6)) -
                    20 * Power(dy, 3) * dz * (q5 + q7) + 15 * dy * Power(dz, 3) * (q5 + q7) +
                    3 * Power(dy, 2) * Power(dz, 2) * (q0 + 8 * q4 - 9 * q8) - Power(dz, 4) * (q0 + 3 * q4 - 4 * q8) +
                    Power(dx, 4) * (4 * q0 - 3 * q4 - q8) + 4 * Power(dy, 4) * (q0 - 2 * q4 + q8) +
                    3 * Power(dx, 2) *
                        (5 * dy * dz * (q5 + q7) + Power(dz, 2) * (q0 - 2 * q4 + q8) +
                         Power(dy, 2) * (-9 * q0 + 8 * q4 + q8)))) /
             Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 4.5));
    qlgyz = prefac *
            ((3 * (-5 * Power(dx, 3) * (dz * (q1 + q3) + dy * (q2 + q6)) -
                   5 * dx *
                       (-6 * Power(dy, 2) * dz * (q1 + q3) + Power(dz, 3) * (q1 + q3) + Power(dy, 3) * (q2 + q6) -
                        6 * dy * Power(dz, 2) * (q2 + q6)) +
                   Power(dx, 4) * (q5 + q7) - 4 * Power(dy, 4) * (q5 + q7) +
                   27 * Power(dy, 2) * Power(dz, 2) * (q5 + q7) - 4 * Power(dz, 4) * (q5 + q7) -
                   5 * dy * Power(dz, 3) * (q0 + 3 * q4 - 4 * q8) - 5 * Power(dy, 3) * dz * (q0 - 4 * q4 + 3 * q8) -
                   3 * Power(dx, 2) *
                       (Power(dy, 2) * (q5 + q7) + Power(dz, 2) * (q5 + q7) + 5 * dy * dz * (-2 * q0 + q4 + q8)))) /
             Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 4.5));
    qlgzz = prefac *
            ((-3 * (5 * Power(dx, 3) * (dy * (q1 + q3) + 3 * dz * (q2 + q6)) +
                    5 * dx *
                        (Power(dy, 3) * (q1 + q3) - 6 * dy * Power(dz, 2) * (q1 + q3) +
                         3 * Power(dy, 2) * dz * (q2 + q6) - 4 * Power(dz, 3) * (q2 + q6)) +
                    15 * Power(dy, 3) * dz * (q5 + q7) - 20 * dy * Power(dz, 3) * (q5 + q7) +
                    Power(dx, 4) * (4 * q0 - q4 - 3 * q8) + 4 * Power(dz, 4) * (q0 + q4 - 2 * q8) -
                    Power(dy, 4) * (q0 - 4 * q4 + 3 * q8) + 3 * Power(dy, 2) * Power(dz, 2) * (q0 - 9 * q4 + 8 * q8) +
                    3 * Power(dx, 2) *
                        (5 * dy * dz * (q5 + q7) + Power(dy, 2) * (q0 + q4 - 2 * q8) +
                         Power(dz, 2) * (-9 * q0 + q4 + 8 * q8)))) /
             Power(Power(dx, 2) + Power(dy, 2) + Power(dz, 2), 4.5));
}