#ifndef TEST_HPP_
#define TEST_HPP_

#include "STKFMM/STKFMM.hpp"

#include "SimpleKernel.hpp"

#include "Util/PointDistribution.hpp"
#include "Util/Timer.hpp"

#include <unordered_map>
#include <vector>

constexpr int maxP = 16;
using KERNEL = stkfmm::KERNEL;
using Stk3DFMM = stkfmm::Stk3DFMM;
using StkWallFMM = stkfmm::StkWallFMM;

struct Config {
    int nSL = 1, nDL = 1, nTrg = 1;
    int rngseed = 0;
    double box = 1;
    double origin[3] = {0, 0, 0};
    int K = 1;
    int pbc = 0;
    int maxPoints = 50;
    double epsilon = 1e-3;
    bool random = true;
    bool direct = false;
    bool verify = true;
    bool convergence = true;
    bool wall = false;

    Config() = default;
    void parse(int argc, char **argv);
    void print() const;
};

struct Point {
    std::vector<double> srcLocalSL;
    std::vector<double> srcLocalDL;
    std::vector<double> trgLocal;
};

struct Source {
    std::vector<double> srcLocalSL;
    std::vector<double> srcLocalDL;
};

struct ComponentError {
    double drift = 0;       // mean drift
    double driftL2 = 0;     // drift * n / L2norm(vec)
    double errorL2 = 0;     // L2 error
    double errorRMS = 0;    // RMS error
    double errorMaxRel = 0; // max relative error

    ComponentError() = default;
    ComponentError(const std::vector<double> &A, const std::vector<double> &B);
};

struct Record {
    KERNEL kernel;
    int multOrder;
    double treeTime = 0;
    double runTime = 0;
    std::vector<ComponentError> errorConvergence; // error for each trgValue component
    std::vector<ComponentError> errorVerify;      // error for each trgValue component
    std::vector<ComponentError> errorTranslate;   // error for each trgValue component
};

using Input = std::unordered_map<KERNEL, Source>;
using Result = std::unordered_map<KERNEL, std::vector<double>>;
using Timing = std::unordered_map<KERNEL, std::pair<double, double>>;

void genPoint(const Config &config, Point &point, int dim);
void translatePoint(const Config &config, Point &point);

void genSrcValue(const Config &config, const Point &point, Input &input);

void runSimpleKernel(const Point &point, Input &input, Result &result);

void runFMM(const Config &config, const int p, const Point &point, Input &input, Result &result, Timing &timing);

void checkError(const int dim, const std::vector<double> &A, const std::vector<double> &B,
                std::vector<ComponentError> &error);

void appendHistory(std::vector<Record> &history, const int p, const Timing &timing, const Result &result,
                   const Result &verifyResult, const Result &convergeResult, const Result &translateResult);

void dumpValue(const std::string &tag, const Point &point, const Input &input, const Result &result);

void recordJson(const Config &config, const std::vector<Record> &record);

template <typename... Args>
inline void printf_rank0(Args... args) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (!rank) {
        printf(args...);
    }
}

#endif