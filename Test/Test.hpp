#ifndef TEST_HPP_
#define TEST_HPP_

#include "STKFMM/STKFMM.hpp"

#include "SimpleKernel.hpp"

#include "Util/PointDistribution.hpp"
#include "Util/Timer.hpp"
#include "Util/cmdparser.hpp"

#include <algorithm>
#include <cassert>
#include <iomanip>
#include <iostream>
#include <vector>

constexpr int maxP = 16;

struct FMMpoint {
    std::vector<double> srcLocalSL;
    std::vector<double> srcLocalDL;
    std::vector<double> trgLocal;
};

struct FMMsrcval {
    std::vector<double> srcLocalSL;
    std::vector<double> srcLocalDL;
};

using KERNEL = stkfmm::KERNEL;
using Stk3DFMM = stkfmm::Stk3DFMM;
using StkWallFMM = stkfmm::StkWallFMM;
using FMMinput = std::unordered_map<KERNEL, FMMsrcval>;
using FMMresult = std::unordered_map<KERNEL, std::vector<double>>;

typedef void (*kernel_func)(double *, double *, double *, double *);

extern std::unordered_map<KERNEL, std::pair<kernel_func, kernel_func>>
    SL_kernels;

void configure_parser(cli::Parser &parser);
void showOption(const cli::Parser &parser);

void genPoint(const cli::Parser &parser, FMMpoint &point, bool wall = false);
void genSrcValue(const cli::Parser &parser, const FMMpoint &point,
                 FMMinput &inputs);
void translatePoints(const cli::Parser &parser, FMMpoint &point);
void dumpValue(const std::string &tag, const FMMpoint &point,
               const FMMinput &inputs, const FMMresult &results);
void checkError(const FMMresult &A, const FMMresult &B, bool component = false);

void runSimpleKernel(const FMMpoint &point, FMMinput &inputs,
                     FMMresult &results);
void runFMM(const cli::Parser &parser, const int p, const FMMpoint &point,
            FMMinput &inputs, FMMresult &results, bool wall = false);

#endif