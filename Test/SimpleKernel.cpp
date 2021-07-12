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

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    p = (fx * rx + fy * ry + fz * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vx = (fx * (2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) + rx * (fy * ry + fz * rz - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vy = (fy * (Power(rx, 2) + 2 * Power(ry, 2) + Power(rz, 2)) + ry * (fx * rx + fz * rz - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vz = (fz * (Power(rx, 2) + Power(ry, 2) + 2 * Power(rz, 2)) + rz * (fx * rx + fy * ry - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
}

void StokesSLPVelGrad(double *s, double *t, double *f, double *pvelGrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    p = (fx * rx + fy * ry + fz * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vx = (fx * (2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) + rx * (fy * ry + fz * rz - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vy = (fy * (Power(rx, 2) + 2 * Power(ry, 2) + Power(rz, 2)) + ry * (fx * rx + fz * rz - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vz = (fz * (Power(rx, 2) + Power(ry, 2) + 2 * Power(rz, 2)) + rz * (fx * rx + fy * ry - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    // grad p
    pgx =
        (-3 * rx * (fx * rx + fy * ry + fz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        fx / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    pgy =
        (-3 * ry * (fx * rx + fy * ry + fz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        fy / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    pgz =
        (-3 * rz * (fx * rx + fy * ry + fz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        fz / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    // grad vx
    vxgx = (-3 * rx * (fx * (2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) + rx * (fy * ry + fz * rz - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
           (4 * fx * rx + fy * ry + fz * rz - TrD) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vxgy = (fy * rx + 2 * fx * ry) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5)) -
           (3 * ry * (fx * (2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) + rx * (fy * ry + fz * rz - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vxgz = (fz * rx + 2 * fx * rz) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5)) -
           (3 * rz * (fx * (2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) + rx * (fy * ry + fz * rz - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    // grad vy
    vygx = (2 * fy * rx + fx * ry) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5)) -
           (3 * rx * (fy * (Power(rx, 2) + 2 * Power(ry, 2) + Power(rz, 2)) + ry * (fx * rx + fz * rz - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vygy = (-3 * ry * (fy * (Power(rx, 2) + 2 * Power(ry, 2) + Power(rz, 2)) + ry * (fx * rx + fz * rz - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
           (fx * rx + 4 * fy * ry + fz * rz - TrD) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vygz = (fz * ry + 2 * fy * rz) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5)) -
           (3 * rz * (fy * (Power(rx, 2) + 2 * Power(ry, 2) + Power(rz, 2)) + ry * (fx * rx + fz * rz - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    // grad vz
    vzgx = (2 * fz * rx + fx * rz) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5)) -
           (3 * rx * (fz * (Power(rx, 2) + Power(ry, 2) + 2 * Power(rz, 2)) + rz * (fx * rx + fy * ry - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vzgy = (2 * fz * ry + fy * rz) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5)) -
           (3 * ry * (fz * (Power(rx, 2) + Power(ry, 2) + 2 * Power(rz, 2)) + rz * (fx * rx + fy * ry - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vzgz = (-3 * rz * (fz * (Power(rx, 2) + Power(ry, 2) + 2 * Power(rz, 2)) + rz * (fx * rx + fy * ry - TrD))) /
               (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
           (fx * rx + fy * ry + 4 * fz * rz - TrD) / (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
}

void StokesSLTraction(double *s, double *t, double *f, double *traction) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    txx = (-3 * Power(rx, 2) * (fx * rx + fy * ry + fz * rz) + (2 * Power(rx, 2) - Power(ry, 2) - Power(rz, 2)) * TrD) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    txy = (-3 * rx * ry * (fx * rx + fy * ry + fz * rz - TrD)) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    txz = (-3 * rx * rz * (fx * rx + fy * ry + fz * rz - TrD)) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    tyx = (-3 * rx * ry * (fx * rx + fy * ry + fz * rz - TrD)) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    tyy = (-3 * Power(ry, 2) * (fx * rx + fy * ry + fz * rz) - (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2)) * TrD) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    tyz = (-3 * ry * rz * (fx * rx + fy * ry + fz * rz - TrD)) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    tzx = (-3 * rx * rz * (fx * rx + fy * ry + fz * rz - TrD)) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    tzy = (-3 * ry * rz * (fx * rx + fy * ry + fz * rz - TrD)) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    tzz = (-3 * Power(rz, 2) * (fx * rx + fy * ry + fz * rz) - (Power(rx, 2) + Power(ry, 2) - 2 * Power(rz, 2)) * TrD) /
          (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
}

void StokesSLPVelLaplacian(double *s, double *t, double *f, double *pvelLaplacian) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    p = (fx * rx + fy * ry + fz * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vx = (fx * (2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) + rx * (fy * ry + fz * rz - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vy = (fy * (Power(rx, 2) + 2 * Power(ry, 2) + Power(rz, 2)) + ry * (fx * rx + fz * rz - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    vz = (fz * (Power(rx, 2) + Power(ry, 2) + 2 * Power(rz, 2)) + rz * (fx * rx + fy * ry - TrD)) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));

    vxlap = (-3 * rx * (fy * ry + fz * rz) + fx * (-2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) /
            (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vylap = (-3 * ry * (fx * rx + fz * rz) + fy * (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2))) /
            (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vzlap = (-3 * (fx * rx + fy * ry) * rz + fz * (Power(rx, 2) + Power(ry, 2) - 2 * Power(rz, 2))) /
            (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
}

//                         3           3           9             4/16/
void StokesDLPVel(double *s, double *t, double *db, double *pvel) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    p = (dbzz * Power(rx, 2) - 3 * dbxy * rx * ry - 3 * dbyx * rx * ry + dbzz * Power(ry, 2) - 3 * dbxz * rx * rz -
         3 * dbzx * rx * rz - 3 * dbyz * ry * rz - 3 * dbzy * ry * rz - 2 * dbzz * Power(rz, 2) +
         dbyy * (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2)) +
         dbxx * (-2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) /
        (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vx = (-3 * rx *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vy = (-3 * ry *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vz = (-3 * rz *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
}

void StokesDLPVelGrad(double *s, double *t, double *db, double *pvelGrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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
    p = (dbzz * Power(rx, 2) - 3 * dbxy * rx * ry - 3 * dbyx * rx * ry + dbzz * Power(ry, 2) - 3 * dbxz * rx * rz -
         3 * dbzx * rx * rz - 3 * dbyz * ry * rz - 3 * dbzy * ry * rz - 2 * dbzz * Power(rz, 2) +
         dbyy * (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2)) +
         dbxx * (-2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) /
        (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vx = (-3 * rx *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vy = (-3 * ry *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vz = (-3 * rz *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));

    // grad p
    pgx = (-4 * dbxx * rx + 2 * dbyy * rx + 2 * dbzz * rx - 3 * dbxy * ry - 3 * dbyx * ry - 3 * dbxz * rz -
           3 * dbzx * rz) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (5 * rx *
           (dbzz * Power(rx, 2) - 3 * dbxy * rx * ry - 3 * dbyx * rx * ry + dbzz * Power(ry, 2) - 3 * dbxz * rx * rz -
            3 * dbzx * rx * rz - 3 * dbyz * ry * rz - 3 * dbzy * ry * rz - 2 * dbzz * Power(rz, 2) +
            dbyy * (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2)) +
            dbxx * (-2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)))) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    pgy = (-3 * dbxy * rx - 3 * dbyx * rx + 2 * dbxx * ry - 4 * dbyy * ry + 2 * dbzz * ry - 3 * dbyz * rz -
           3 * dbzy * rz) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (5 * ry *
           (dbzz * Power(rx, 2) - 3 * dbxy * rx * ry - 3 * dbyx * rx * ry + dbzz * Power(ry, 2) - 3 * dbxz * rx * rz -
            3 * dbzx * rx * rz - 3 * dbyz * ry * rz - 3 * dbzy * ry * rz - 2 * dbzz * Power(rz, 2) +
            dbyy * (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2)) +
            dbxx * (-2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)))) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));
    pgz = (-3 * dbxz * rx - 3 * dbzx * rx - 3 * dbyz * ry - 3 * dbzy * ry + 2 * dbxx * rz + 2 * dbyy * rz -
           4 * dbzz * rz) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (5 * rz *
           (dbzz * Power(rx, 2) - 3 * dbxy * rx * ry - 3 * dbyx * rx * ry + dbzz * Power(ry, 2) - 3 * dbxz * rx * rz -
            3 * dbzx * rx * rz - 3 * dbyz * ry * rz - 3 * dbzy * ry * rz - 2 * dbzz * Power(rz, 2) +
            dbyy * (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2)) +
            dbxx * (-2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2)))) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    // vx grad
    vxgx = (3 * (-(rx * (2 * dbxx * rx + (dbxy + dbyx) * ry + (dbxz + dbzx) * rz) *
                   (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
                 5 * Power(rx, 2) *
                     (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
                      (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)) -
                 (Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) *
                     (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
                      (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
           (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    vxgy =
        (3 * rx *
         (-(((dbxy + dbyx) * rx + 2 * dbyy * ry + (dbyz + dbzy) * rz) * (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
          5 * ry *
              (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
               (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
        (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    vxgz =
        (3 * rx *
         (-(((dbxz + dbzx) * rx + (dbyz + dbzy) * ry + 2 * dbzz * rz) * (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
          5 * rz *
              (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
               (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
        (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    // vy grad
    vygx =
        (3 * ry *
         (-((2 * dbxx * rx + (dbxy + dbyx) * ry + (dbxz + dbzx) * rz) * (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
          5 * rx *
              (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
               (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
        (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    vygy = (3 * (-(ry * ((dbxy + dbyx) * rx + 2 * dbyy * ry + (dbyz + dbzy) * rz) *
                   (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
                 5 * Power(ry, 2) *
                     (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
                      (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)) -
                 (Power(rx, 2) + Power(ry, 2) + Power(rz, 2)) *
                     (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
                      (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
           (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    vygz =
        (3 * ry *
         (-(((dbxz + dbzx) * rx + (dbyz + dbzy) * ry + 2 * dbzz * rz) * (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
          5 * rz *
              (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
               (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
        (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    // vz grad
    vzgx =
        (3 * rz *
         (-((2 * dbxx * rx + (dbxy + dbyx) * ry + (dbxz + dbzx) * rz) * (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
          5 * rx *
              (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
               (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
        (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));

    vzgy =
        (3 * rz *
         (-(((dbxy + dbyx) * rx + 2 * dbyy * ry + (dbyz + dbzy) * rz) * (Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) +
          5 * ry *
              (dbxx * Power(rx, 2) + (dbxy + dbyx) * rx * ry + dbyy * Power(ry, 2) + (dbxz + dbzx) * rx * rz +
               (dbyz + dbzy) * ry * rz + dbzz * Power(rz, 2)))) /
        (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));
    vzgz = (-3 * (Power(rx, 2) + Power(ry, 2)) * (dbxx * Power(rx, 2) + ry * ((dbxy + dbyx) * rx + dbyy * ry)) -
            6 * ((dbxz + dbzx) * rx + (dbyz + dbzy) * ry) * (Power(rx, 2) + Power(ry, 2)) * rz +
            3 *
                (4 * dbxx * Power(rx, 2) + 4 * ry * ((dbxy + dbyx) * rx + dbyy * ry) -
                 3 * dbzz * (Power(rx, 2) + Power(ry, 2))) *
                Power(rz, 2) +
            9 * ((dbxz + dbzx) * rx + (dbyz + dbzy) * ry) * Power(rz, 3) + 6 * dbzz * Power(rz, 4)) /
           (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));
}

void StokesDLTraction(double *s, double *t, double *db, double *traction) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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
    p = (dbzz * Power(rx, 2) - 3 * dbxy * rx * ry - 3 * dbyx * rx * ry + dbzz * Power(ry, 2) - 3 * dbxz * rx * rz -
         3 * dbzx * rx * rz - 3 * dbyz * ry * rz - 3 * dbzy * ry * rz - 2 * dbzz * Power(rz, 2) +
         dbyy * (Power(rx, 2) - 2 * Power(ry, 2) + Power(rz, 2)) +
         dbxx * (-2 * Power(rx, 2) + Power(ry, 2) + Power(rz, 2))) /
        (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vx = (-3 * rx *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vy = (-3 * ry *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vz = (-3 * rz *
          (dbxx * Power(rx, 2) + dbxy * rx * ry + dbyx * rx * ry + dbyy * Power(ry, 2) + dbxz * rx * rz +
           dbzx * rx * rz + dbyz * ry * rz + dbzy * ry * rz + dbzz * Power(rz, 2))) /
         (8. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    vxlap =
        (3 * (2 * dbxx * Power(rx, 3) - dbyy * Power(rx, 3) - dbzz * Power(rx, 3) + 4 * dbxy * Power(rx, 2) * ry +
              4 * dbyx * Power(rx, 2) * ry - 3 * dbxx * rx * Power(ry, 2) + 4 * dbyy * rx * Power(ry, 2) -
              dbzz * rx * Power(ry, 2) - dbxy * Power(ry, 3) - dbyx * Power(ry, 3) +
              (4 * (dbxz + dbzx) * Power(rx, 2) + 5 * (dbyz + dbzy) * rx * ry - (dbxz + dbzx) * Power(ry, 2)) * rz -
              ((3 * dbxx + dbyy - 4 * dbzz) * rx + (dbxy + dbyx) * ry) * Power(rz, 2) - (dbxz + dbzx) * Power(rz, 3))) /
        (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));
    vylap =
        (3 * (-((dbxy + dbyx) * Power(rx, 3)) + (4 * dbxx - 3 * dbyy - dbzz) * Power(rx, 2) * ry +
              4 * (dbxy + dbyx) * rx * Power(ry, 2) - (dbxx - 2 * dbyy + dbzz) * Power(ry, 3) +
              (-((dbyz + dbzy) * Power(rx, 2)) + 5 * (dbxz + dbzx) * rx * ry + 4 * (dbyz + dbzy) * Power(ry, 2)) * rz -
              ((dbxy + dbyx) * rx + (dbxx + 3 * dbyy - 4 * dbzz) * ry) * Power(rz, 2) - (dbyz + dbzy) * Power(rz, 3))) /
        (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));
    vzlap =
        (-3 * ((dbxz + dbzx) * rx + (dbyz + dbzy) * ry) * (Power(rx, 2) + Power(ry, 2)) +
         3 *
             ((4 * dbxx - dbyy - 3 * dbzz) * Power(rx, 2) + 5 * (dbxy + dbyx) * rx * ry -
              (dbxx - 4 * dbyy + 3 * dbzz) * Power(ry, 2)) *
             rz +
         12 * ((dbxz + dbzx) * rx + (dbyz + dbzy) * ry) * Power(rz, 2) - 3 * (dbxx + dbyy - 2 * dbzz) * Power(rz, 3)) /
        (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5));
}

void LaplaceSLGrad(double *s, double *t, double *q, double *pgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

    const double charge = q[0];

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 3; i++) {
            pgrad[i] = 0;
        }
        return;
    }

    double &gx = pgrad[0];
    double &gy = pgrad[1];
    double &gz = pgrad[2];
    gx = -0.25 * (charge * rx) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gy = -0.25 * (charge * ry) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gz = -0.25 * (charge * rz) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
}

void LaplaceDLGrad(double *s, double *t, double *db, double *pgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double dx = db[0];
    const double dy = db[1];
    const double dz = db[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

    if (sx == tx && sy == ty && sz == tz) {
        for (int i = 0; i < 3; i++) {
            pgrad[i] = 0;
        }
        return;
    }

    double &gx = pgrad[0];
    double &gy = pgrad[1];
    double &gz = pgrad[2];

    gx =
        (-3 * rx * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dx / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gy =
        (-3 * ry * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dy / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gz =
        (-3 * rz * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dz / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
}

void LaplaceSLPGrad(double *s, double *t, double *q, double *pgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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
    p = charge / (4. * Pi * Sqrt(Power(rx, 2) + Power(ry, 2) + Power(rz, 2)));
    gx = -0.25 * (charge * rx) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gy = -0.25 * (charge * ry) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gz = -0.25 * (charge * rz) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
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

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    p = (dx * rx + dy * ry + dz * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gx =
        (-3 * rx * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dx / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gy =
        (-3 * ry * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dy / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gz =
        (-3 * rz * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dz / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
}

void LaplaceSLPGradGrad(double *s, double *t, double *q, double *pgradgrad) {
    const double sx = s[0];
    const double sy = s[1];
    const double sz = s[2];

    const double tx = t[0];
    const double ty = t[1];
    const double tz = t[2];

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    p = charge / (4. * Pi * Sqrt(Power(rx, 2) + Power(ry, 2) + Power(rz, 2)));
    gx = -0.25 * (charge * rx) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gy = -0.25 * (charge * ry) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gz = -0.25 * (charge * rz) / (Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gxx = (3 * charge * Power(rx, 2)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          charge / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gyy = (3 * charge * Power(ry, 2)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          charge / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gzz = (3 * charge * Power(rz, 2)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          charge / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gxy = (3 * charge * rx * ry) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    gxz = (3 * charge * rx * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    gyz = (3 * charge * ry * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
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

    const double rx = tx - sx;
    const double ry = ty - sy;
    const double rz = tz - sz;

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

    p = (dx * rx + dy * ry + dz * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gx =
        (-3 * rx * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dx / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gy =
        (-3 * ry * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dy / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));
    gz =
        (-3 * rz * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) +
        dz / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 1.5));

    gxx = (15 * Power(rx, 2) * (dx * rx + dy * ry + dz * rz)) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5)) -
          (3 * dx * rx) / (2. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (3 * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    gyy = (15 * Power(ry, 2) * (dx * rx + dy * ry + dz * rz)) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5)) -
          (3 * dy * ry) / (2. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (3 * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    gzz = (15 * Power(rz, 2) * (dx * rx + dy * ry + dz * rz)) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5)) -
          (3 * dz * rz) / (2. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (3 * (dx * rx + dy * ry + dz * rz)) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    gxy = (15 * rx * ry * (dx * rx + dy * ry + dz * rz)) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5)) -
          (3 * dy * rx) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (3 * dx * ry) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    gxz = (15 * rx * rz * (dx * rx + dy * ry + dz * rz)) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5)) -
          (3 * dz * rx) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (3 * dx * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
    gyz = (15 * ry * rz * (dx * rx + dy * ry + dz * rz)) /
              (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 3.5)) -
          (3 * dz * ry) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5)) -
          (3 * dy * rz) / (4. * Pi * Power(Power(rx, 2) + Power(ry, 2) + Power(rz, 2), 2.5));
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