#ifndef POINTDISTRIBUTION_HPP
#define POINTDISTRIBUTION_HPP

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

void fixedPoints(size_t nPts, double box, double shift, std::vector<double> &srcCoord);

void chebPoints(size_t nPts, double box, double shift, std::vector<double> &ptsCoord);

void randomPoints(size_t nPts, double box, double shift, std::vector<double> &ptsCoord);

void shiftAndScalePoints(std::vector<double> &ptsCoord, double shift[3], double scale);

void randomUniformFill(std::vector<double> &vec, double low, double high);

void randomLogNormalFill(std::vector<double> &vec, double a, double b);

void dumpPoints(const std::string &filename, std::vector<double> &coord, std::vector<double> &value,
                const int valueDimension);

void checkError(const std::vector<double> &value, const std::vector<double> &valueTrue);

void distributePts(std::vector<double> &pts, int dimension);

void collectPts(std::vector<double> &pts);

void collectPtsAll(std::vector<double> &pts);

#endif