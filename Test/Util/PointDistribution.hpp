#ifndef POINTDISTRIBUTION_HPP
#define POINTDISTRIBUTION_HPP

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

class PointDistribution {
    std::mt19937 gen_;

  public:
    PointDistribution(int seed) : gen_(seed){};

    // non-static methods depending on rng seed
    void randomPoints(int dim, int nPts, double box, double shift, std::vector<double> &ptsCoord);

    void randomUniformFill(std::vector<double> &vec, double low, double high);

    void randomLogNormalFill(std::vector<double> &vec, double a, double b);

    // static methods
    static void fixedPoints(int nPts, double box, double shift, std::vector<double> &srcCoord);

    static void shiftAndScalePoints(std::vector<double> &ptsCoord, double shift[3], double scale);

    static void meshPoints(int dim, int nPts, double box, double shift, std::vector<double> &ptsCoord,
                           bool cheb = false);

    static void dumpPoints(const std::string &filename, std::vector<double> &coord, std::vector<double> &value,
                           const int valueDimension);

    static void checkError(const std::vector<double> &value, const std::vector<double> &valueTrue, const int kdim = 0);

    static void distributePts(std::vector<double> &pts, int dimension);

    static void collectPts(std::vector<double> &pts);

    static void collectPtsAll(std::vector<double> &pts);
};

#endif
