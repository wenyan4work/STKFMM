#ifndef POINTDISTRIBUTION_HPP
#define POINTDISTRIBUTION_HPP

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>
#include <random>

class PointDistribution {
public:
  PointDistribution(int seed) : gen_(seed){};

  static void fixedPoints(int nPts, double box, double shift,
                   std::vector<double> &srcCoord);

  static void chebPoints(int nPts, double box, double shift,
                         std::vector<double> &ptsCoord);

  void randomPoints(int nPts, double box, double shift,
                           std::vector<double> &ptsCoord);

  static void shiftAndScalePoints(std::vector<double> &ptsCoord, double shift[3],
                           double scale);

  void randomUniformFill(std::vector<double> &vec, double low, double high);

  void randomLogNormalFill(std::vector<double> &vec, double a, double b);

  static void dumpPoints(const std::string &filename, std::vector<double> &coord,
                  std::vector<double> &value, const int valueDimension);

  static void checkError(const std::vector<double> &value,
                  const std::vector<double> &valueTrue);

  static void distributePts(std::vector<double> &pts, int dimension);

  static void collectPts(std::vector<double> &pts);

  static void collectPtsAll(std::vector<double> &pts);

private:
  std::mt19937 gen_;
};

#endif
